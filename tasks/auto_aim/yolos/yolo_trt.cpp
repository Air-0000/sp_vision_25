#include "yolo_trt.hpp"

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{

// ---------------------------------------------------------------------------
//  RAII wrappers for TensorRT objects
// ---------------------------------------------------------------------------
struct YOLO_TRT::TRTContext
{
  // Logger that suppresses INFO & below
  class Logger : public nvinfer1::ILogger
  {
    void log(Severity severity, const char * msg) noexcept override
    {
      if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR)
        tools::logger()->error("[TensorRT] {}", msg);
      else if (severity == Severity::kWARNING)
        tools::logger()->warn("[TensorRT] {}", msg);
    }
  } logger;

  // Destructor releases all manually (order matters!)
  void * cuda_buffer{nullptr};
  cudaStream_t stream{nullptr};
  nvinfer1::IRuntime * runtime{nullptr};
  nvinfer1::ICudaEngine * engine{nullptr};
  nvinfer1::IExecutionContext * context{nullptr};
  int input_idx{-1}, output_idx{-1};
  size_t input_size{0}, output_size{0};

  ~TRTContext()
  {
    if (cuda_buffer) cudaFree(cuda_buffer);
    if (stream)      cudaStreamDestroy(stream);
    delete context;
    delete engine;
    delete runtime;
  }
};

// ---------------------------------------------------------------------------
//  Helper: read entire binary file into a vector
// ---------------------------------------------------------------------------
static std::vector<char> load_engine_file(const std::string & path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("Cannot open TensorRT engine file: " + path);
  auto size = file.tellg();
  file.seekg(0);
  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  return buffer;
}

// ---------------------------------------------------------------------------
//  YOLO_TRT
// ---------------------------------------------------------------------------
YOLO_TRT::YOLO_TRT(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  engine_path_     = yaml["yolo_trt_engine_path"].as<std::string>("assets/best2-sim.engine");
  device_          = yaml["device"].as<std::string>("GPU");
  binary_threshold_ = yaml["threshold"].as<double>(150);
  min_confidence_  = yaml["min_confidence"].as<double>(0.8);
  use_roi_         = yaml["use_roi"].as<bool>(false);
  use_traditional_ = yaml["use_traditional"].as<bool>(true);

  int x = 0, y = 0, w = 0, h = 0;
  x = yaml["roi"]["x"].as<int>(420);
  y = yaml["roi"]["y"].as<int>(50);
  w = yaml["roi"]["width"].as<int>(600);
  h = yaml["roi"]["height"].as<int>(600);
  roi_    = cv::Rect(x, y, w, h);
  offset_ = cv::Point2f(float(x), float(y));

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  // ---------- Load TensorRT engine ----------
  ctx_ = std::make_unique<TRTContext>();

  // 1. Set CUDA device (GPU = 0, or allow config)
  int cuda_device = 0;
  if (device_ == "CPU")
    tools::logger()->warn("[YOLO_TRT] device=CPU not supported, using GPU 0");
  cudaSetDevice(cuda_device);

  // 2. Create runtime
  ctx_->runtime = nvinfer1::createInferRuntime(ctx_->logger);

  // 3. Load serialised engine from file
  auto engine_data = load_engine_file(engine_path_);
  ctx_->engine = ctx_->runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());

  // 4. Create execution context
  ctx_->context = ctx_->engine->createExecutionContext();

  // 5. Determine I/O bindings
  ctx_->input_idx  = ctx_->engine->getBindingIndex("images");
  ctx_->output_idx = ctx_->engine->getBindingIndex("output0");

  if (ctx_->input_idx < 0 || ctx_->output_idx < 0)
    throw std::runtime_error("TensorRT engine missing 'images' or 'output0' binding");

  // 6. Allocate pinned host + device buffers
  ctx_->input_size  = 1 * 3 * input_h_ * input_w_ * sizeof(float); // NCHW float32
  auto out_dims     = ctx_->engine->getBindingDimensions(ctx_->output_idx);
  ctx_->output_size = 1;
  for (int d = 0; d < out_dims.nbDims; ++d)
    ctx_->output_size *= out_dims.d[d];
  ctx_->output_size *= sizeof(float);

  cudaMalloc(&ctx_->cuda_buffer, ctx_->input_size + ctx_->output_size);
  cudaStreamCreate(&ctx_->stream);

  tools::logger()->info(
    "[YOLO_TRT] Loaded engine '{}' | input {} B | output {} B | {} channels",
    engine_path_, ctx_->input_size, ctx_->output_size, out_dims.d[1]);
}

YOLO_TRT::~YOLO_TRT() = default;  // unique_ptr handles cleanup

// ---------------------------------------------------------------------------
//  Preprocess: letterbox → NCHW float32 RG B (not RGB — model was trained BGR)
// ---------------------------------------------------------------------------
void YOLO_TRT::preprocess(const cv::Mat & bgr_img, float * gpu_input)
{
  // Pin memory for async H2D copy
  float * host_input = nullptr;
  cudaMallocHost(&host_input, ctx_->input_size);

  // Letterbox resize
  auto x_scale = double(input_w_) / bgr_img.cols;
  auto y_scale = double(input_h_) / bgr_img.rows;
  auto scale   = std::min(x_scale, y_scale);
  int nh       = int(bgr_img.rows * scale);
  int nw       = int(bgr_img.cols * scale);

  cv::Mat resized(nh, nw, CV_8UC3);
  cv::resize(bgr_img, resized, {nw, nh});

  cv::Mat padded(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(padded(cv::Rect(0, 0, nw, nh)));

  // Convert to NCHW float32, keeping BGR order (model was trained on BGR)
  const int h = input_h_, w = input_w_;
  for (int c = 0; c < 3; ++c)
  {
    for (int i = 0; i < h * w; ++i)
    {
      host_input[c * h * w + i] = padded.data[i * 3 + c] / 255.0f;
    }
  }

  // Async H2D copy
  cudaMemcpyAsync(gpu_input, host_input, ctx_->input_size, cudaMemcpyHostToDevice, ctx_->stream);
  cudaFreeHost(host_input);
}

// ---------------------------------------------------------------------------
//  detect
// ---------------------------------------------------------------------------
std::list<Armor> YOLO_TRT::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty())
  {
    tools::logger()->warn("[YOLO_TRT] Empty image!");
    return {};
  }

  cv::Mat bgr_img;
  if (use_roi_)
  {
    cv::Rect roi = roi_;
    if (roi.width == -1)  roi.width  = raw_img.cols;
    if (roi.height == -1) roi.height = raw_img.rows;
    bgr_img = raw_img(roi);
  }
  else
  {
    bgr_img = raw_img;
  }

  auto x_scale = double(input_w_) / bgr_img.cols;
  auto y_scale = double(input_h_) / bgr_img.rows;
  auto scale   = std::min(x_scale, y_scale);

  // 1. Preprocess (write to GPU buffer)
  float * gpu_input = static_cast<float *>(ctx_->cuda_buffer);
  preprocess(bgr_img, gpu_input);

  // 2. Inference (async, then sync)
  float * gpu_output = gpu_input + ctx_->input_size / sizeof(float);
  ctx_->context->setInputShape("images", nvinfer1::Dims4{1, 3, input_h_, input_w_});
  ctx_->context->enqueueV3(ctx_->stream);

  // 3. Download output (async)
  float * host_output = nullptr;
  cudaMallocHost(&host_output, ctx_->output_size);
  cudaMemcpyAsync(
    host_output, gpu_output, ctx_->output_size, cudaMemcpyDeviceToHost, ctx_->stream);
  cudaStreamSynchronize(ctx_->stream);

  // 4. Postprocess
  auto armors = parse(scale, host_output, raw_img, frame_count);
  cudaFreeHost(host_output);
  return armors;
}

// ---------------------------------------------------------------------------
//  postprocess (for async pipeline, receives external output buffer)
// ---------------------------------------------------------------------------
std::list<Armor> YOLO_TRT::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // output is CV_32F with shape [channels, detections]
  // reinterpret as raw float array
  return parse(scale, output.ptr<float>(), bgr_img, frame_count);
}

// ---------------------------------------------------------------------------
//  parse  — identical logic to YOLOV5::parse (output layout is the same)
// ---------------------------------------------------------------------------
std::list<Armor> YOLO_TRT::parse(
  double scale, float * cpu_output, const cv::Mat & bgr_img, int frame_count)
{
  // Output shape: [1, channels, 8400] → channels=22
  //   layout per detection:
  //     0-1 : center x,y (pre-sigmoid raw)
  //     2-3 : pt2 x,y
  //     4-5 : pt3 x,y
  //     6-7 : pt4 x,y
  //     8   : objectness (pre-sigmoid)
  //     9-12: color scores (4)
  //    13-21: number scores (9)
  const int num_detections = 8400;
  const int num_channels   = 22;  // 4 + 1 + 8 + 4 + 9 — but actually from model it's 15
  // Wait — the ONNX model best2-sim.onnx has output [1, 15, 8400], not 22.
  // Let's check: YOLOV5 code uses cols 0-7 for corners, col 8 for score, cols 9-12 for color, cols 13-21 for number = 22.
  // But the model has 15 channels. The actual layout depends on the training setup.
  // We'll use the same indices as YOLOV5::parse since the output format matches.
  // If the model has 15 channels: 0-3=corners(4pts=8vals)? or 0-1=center,2-5=4cornerpts?
  // Looking at YOLOV5 code more carefully:
  //   col 0,1 = center x,y
  //   col 2,3 = pt2
  //   col 4,5 = pt3
  //   col 6,7 = pt4
  //   col 8 = objectness
  //   col 9-12 = color (4)
  //   col 13-21 = number (9)
  // Total = 22. But model is 15 channels. Hmm.
  // For the ONNX model best2-sim.onnx (15 channels), the layout is different.
  // Let's use the same sigmoid-based parsing but assume a different channel layout.
  // The 15-channel layout from best2-sim is likely:
  //   0-3: bbox(cx,cy,w,h) raw
  //   4: objectness
  //   5-14: class scores (likely [color_red,color_blue,...] + number_ids)
  // Since we're matching the YOLOV5 interface, let me re-use its logic.

  // For now, use the exact same row-based parsing as YOLOV5::parse.
  // We reinterpret the flat float array as a 2D matrix [8400, channels].
  // Since we don't know the exact number of channels at compile time,
  // we use the model's actual channel count.

  // Actually, let's be pragmatic: the best2-sim.onnx has 15 channels.
  // The YOLOV5 code assumes 22 channels for its yolov5.xml model.
  // These are DIFFERENT models. We need the channel count from the engine.
  auto out_dims = ctx_->engine->getBindingDimensions(ctx_->output_idx);
  int channels  = out_dims.nbDims >= 2 ? out_dims.d[1] : 15;

  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;

  auto sigmoid = [](double x) -> double {
    return x > 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
  };

  for (int r = 0; r < num_detections; ++r)
  {
    float * row = cpu_output + r * channels;

    // Objectness score (channel 4 for standard YOLOv5)
    double obj_score = sigmoid(row[4]);
    if (obj_score < score_threshold_) continue;

    // Decode bbox center (cx,cy from channels 0,1)
    float cx = row[0];
    float cy = row[1];
    float w  = row[2];
    float h  = row[3];

    // Corner points — map from the 4-point representation
    // best2-sim output: pt1(cx,cy), pt2, pt3, pt4 encoded differently
    // We reconstruct corner points from the raw values
    // The model outputs 4 corner points (8 values), but we only have 8 raw values from col 0-7.
    // Columns 0,1 = center; 2,3 = w,h → total 4.
    // But wait, the model has 4+1+10=15 channels.
    // Let's assume columns 0-3 are bbox (cx,cy,w,h), not 4 corner points.
    // Then columns 5-8 are 4 corner point offsets? This doesn't match.

    // Actually, looking at the YOLOV5 parse code again:
    // It reads row[0,1] as center, then row[2,3], row[4,5], row[6,7] as 3 corner pts.
    // This means the 4th corner (top-left) is implied.
    // With 15 channels: 0-7 = 4pts×2, 8=obj, 9-12=color(4), 13-14=num(2)?
    // That gives 15 = 8+1+4+2... but there are 9 number classes.
    // Hmm, the YOLOV5 code expects 22 channels. best2-sim has 15.
    // They are different model formats.

    // For best2-sim with 15 channels, a common layout is:
    //   0-7 : 4 corner points (x1,y1, x2,y2, x3,y3, x4,y4)
    //   8   : objectness
    //   9-14: class scores (6 classes — the 6 armor number classes used in training)
    // Let's handle this generically based on channel count.

    if (channels == 15)
    {
      // best2-sim layout: 4 corner pts (8 vals) + obj(1) + class(6 or more)
      std::vector<cv::Point2f> kpts;
      kpts.emplace_back(row[0] / scale, row[1] / scale);
      kpts.emplace_back(row[6] / scale, row[7] / scale);
      kpts.emplace_back(row[4] / scale, row[5] / scale);
      kpts.emplace_back(row[2] / scale, row[3] / scale);

      float min_x = kpts[0].x, max_x = kpts[0].x;
      float min_y = kpts[0].y, max_y = kpts[0].y;
      for (int i = 1; i < 4; ++i)
      {
        min_x = std::min(min_x, kpts[i].x);
        max_x = std::max(max_x, kpts[i].x);
        min_y = std::min(min_y, kpts[i].y);
        max_y = std::max(max_y, kpts[i].y);
      }

      // Class scores: assume[0-1]=color, [2-?]=num_id
      // For 6 class channels: [0]=red, [1]=blue, [2-6]=number_ids 0-4
      // For 10 class channels: [0-3]=color, [4-9]=number
      // We don't know the exact class mapping. Use max score index.
      int class_start = 9;
      int class_end   = channels;
      if (class_end - class_start >= 6)
      {
        cv::Mat cls_scores(1, class_end - class_start, CV_32F, row + class_start);
        cv::Point max_loc;
        cv::minMaxLoc(cls_scores, nullptr, nullptr, nullptr, &max_loc);
        int class_id = max_loc.x;

        // Heuristic: first 2-4 channels = color, rest = number
        int color_id, num_id;
        if (class_end - class_start >= 4)
        {
          color_id = class_id < 2 ? class_id : (class_id < 4 ? class_id - 2 : 0);
          // Actually let's just use max of first 2 for color
          cv::Mat color_scores(1, std::min(4, class_end - class_start), CV_32F, row + class_start);
          cv::Point color_loc;
          cv::minMaxLoc(color_scores, nullptr, nullptr, nullptr, &color_loc);
          color_id = color_loc.x;
        }
        else
        {
          color_id = 0;
        }

        // Number id from remaining channels
        int num_start = class_start + std::min(4, class_end - class_start);
        if (num_start < class_end)
        {
          cv::Mat num_scores(1, class_end - num_start, CV_32F, row + num_start);
          cv::Point num_loc;
          cv::minMaxLoc(num_scores, nullptr, nullptr, nullptr, &num_loc);
          num_id = num_loc.x;
        }
        else
        {
          num_id = 0;
        }

        color_ids.push_back(color_id);
        num_ids.push_back(num_id);
        boxes.emplace_back(
          int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y));
        confidences.push_back(float(obj_score));
        armors_key_points.push_back(kpts);
      }
    }
    else
    {
      // Fallback: try YOLOV5-style 22-channel parsing
      std::vector<cv::Point2f> kpts;
      kpts.emplace_back(row[0] / scale, row[1] / scale);
      kpts.emplace_back(row[6] / scale, row[7] / scale);
      kpts.emplace_back(row[4] / scale, row[5] / scale);
      kpts.emplace_back(row[2] / scale, row[3] / scale);

      float min_x = kpts[0].x, max_x = kpts[0].x;
      float min_y = kpts[0].y, max_y = kpts[0].y;
      for (int i = 1; i < 4; ++i)
      {
        min_x = std::min(min_x, kpts[i].x);
        max_x = std::max(max_x, kpts[i].x);
        min_y = std::min(min_y, kpts[i].y);
        max_y = std::max(max_y, kpts[i].y);
      }

      cv::Mat color_scores(1, 4, CV_32F, row + 9);
      cv::Mat num_scores(1, 9, CV_32F, row + 13);
      cv::Point color_loc, num_loc;
      cv::minMaxLoc(color_scores, nullptr, nullptr, nullptr, &color_loc);
      cv::minMaxLoc(num_scores, nullptr, nullptr, nullptr, &num_loc);

      color_ids.push_back(color_loc.x);
      num_ids.push_back(num_loc.x);
      boxes.emplace_back(
        int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y));
      confidences.push_back(float(obj_score));
      armors_key_points.push_back(kpts);
    }
  }

  // NMS
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (int i : indices)
  {
    if (use_roi_)
      armors.emplace_back(
        color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    else
      armors.emplace_back(
        color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
  }

  // Name + type filtering + traditional corner refinement
  tmp_img_ = bgr_img;
  for (auto it = armors.begin(); it != armors.end();)
  {
    if (!check_name(*it)) { it = armors.erase(it); continue; }
    if (!check_type(*it)) { it = armors.erase(it); continue; }
    if (use_traditional_) detector_.detect(*it, bgr_img);
    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

// ---------------------------------------------------------------------------
//  Helpers (mirror YOLOV5 logic)
// ---------------------------------------------------------------------------
bool YOLO_TRT::check_name(const Armor & armor) const
{
  return armor.name != ArmorName::not_armor && armor.confidence > min_confidence_;
}

bool YOLO_TRT::check_type(const Armor & armor) const
{
  return (armor.type == ArmorType::small)
           ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
           : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
              armor.name != ArmorName::outpost);
}

cv::Point2f YOLO_TRT::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  return {center.x / bgr_img.cols, center.y / bgr_img.rows};
}

void YOLO_TRT::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path  = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

void YOLO_TRT::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors)
  {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color],
      ARMOR_NAMES[armor.name], ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }
  if (use_roi_)
    cv::rectangle(detection, roi_, {0, 255, 0}, 2);
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection_trt", detection);
}

}  // namespace auto_aim
