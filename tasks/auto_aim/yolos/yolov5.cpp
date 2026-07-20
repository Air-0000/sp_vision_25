/**
 * @file yolov5.cpp
 * @brief YOLOv5 OpenVINO 推理实现
 * @details 完整的 YOLOv5 检测管线实现：
 *          1. 预处理：ROI 裁剪（可选）→ letterbox resize 到 640×640 → NHWC uint8 张量
 *          2. 推理：OpenVINO 同步推理
 *          3. 后处理：sigmoid → 阈值过滤 → NMS → 颜色/编号 argmax → 名称/类型校验
 *             → 传统方法角点二次修正（可选）
 *
 *          OpenVINO 预处理配置：
 *          - 输入布局：NHWC（OpenVINO 输入格式要求）
 *          - 数据类型：uint8（减少 H2D 传输带宽）
 *          - 颜色格式：BGR（OpenVINO 内部转为 RGB 并归一化处理）
 *          - 后处理：OpenVINO 自动完成归一化（÷255.0）
 *
 * @note 预处理的 letterbox 操作保持宽高比，不足部分补 0（黑色边）。
 *       模型的输出有 8400 个候选框（3 个检测头的 Anchor 点总数）。
 */

#include "yolov5.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{

// ========== 步骤 1-1：YOLOv5 检测器初始化 ==========

YOLOV5::YOLOV5(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  /**
   * 读取配置：
   * - model_path：YOLOv5 的 OpenVINO IR 模型文件路径（.xml）
   * - device：推理设备（CPU / GPU / AUTO）
   * - threshold：传统方法二值化阈值（用于角点二次修正）
   * - min_confidence：分类器最低置信度
   * - roi：感兴趣区域（只检测该区域内的目标，减少计算量）
   * - use_roi：是否启用 ROI
   * - use_traditional：是否用传统方法二次修正角点
   */
  model_path_ = yaml["yolov5_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  // ---------- OpenVINO 模型加载 ----------
  auto model = core_.read_model(model_path_);

  /**
   * 配置预处理流程：
   * 1. 输入张量格式：NHWC uint8，尺寸 1×640×640×3，BGR 颜色顺序
   * 2. 模型内部格式：NCHW float32
   * 3. 预处理流水线：uint8 → float32 → BGR→RGB → ÷255.0 归一化
   *
   * 这样 CPU→GPU 传输的是 uint8（带宽仅为 float32 的 1/4），
   * 归一化由 OpenVINO 在 GPU 上完成，节省了 CPU 预处理时间。
   */
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 640, 640, 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  model = ppp.build();

  // 编译模型到指定设备，使用 LATENCY 模式（最小化单帧推理延迟）
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

// ========== 步骤 1-2：完整检测管线 ==========

std::list<Armor> YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
{
  // ---------- 输入检查 ----------
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  // ---------- 步骤 1-2a：ROI 裁剪（可选） ----------
  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) {  // -1 表示不裁切宽度
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {  // -1 表示不裁切高度
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  // ---------- 步骤 1-2b：Letterbox 预处理 ----------
  // 保持宽高比缩放到 640×640，不足部分补 0（黑色边框）
  // 这种处理方式比直接 resize 到 640×640 更好，因为不会改变目标的宽高比，
  // 模型看到的目标形状与训练时一致，检测精度更高。
  auto x_scale = static_cast<double>(640) / bgr_img.rows;
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);  // 取较小缩放比保证完整包含原图
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));  // 黑色画布
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input(roi), {w, h});  // 将缩放后的图像放到左上角

  // ---------- 步骤 1-2c：OpenVINO 推理 ----------
  // 创建 NHWC uint8 张量，指向 input 图像数据（零拷贝）
  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();  // 同步推理（阻塞直到返回结果）

  // ---------- 步骤 1-2d：获取输出 ----------
  // 输出形状：[1, 22, 8400] → 转为 [8400, 22] 的 CV_32F 矩阵
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

  // ---------- 步骤 1-2e：后处理（解析 + NMS + 校验） ----------
  return parse(scale, output, raw_img, frame_count);
}

// ---------- 对外暴露的 postprocess 接口（供多线程管线使用） ----------

std::list<Armor> YOLOV5::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

// ========== 步骤 1-3：输出解析（核心后处理逻辑） ==========

std::list<Armor> YOLOV5::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  /**
   * 解析 8400 个候选框，每个框有 22 个 float 值：
   *
   * 布局说明（以行为单位）：
   *   row[0..7] ：4 个角点的 x, y 像素坐标（模型输出的原始 raw 值，非归一化）
   *     [0,1] = 第 1 个角点（通常是最左上角）
   *     [2,3] = 第 2 个角点
   *     [4,5] = 第 3 个角点
   *     [6,7] = 第 4 个角点
   *   row[8]    ：objectness（目标置信度），pre-sigmoid raw 值
   *   row[9..12]：4 个颜色的 logits（顺序：red, blue, extinguish, purple）
   *   row[13..21]：9 个编号的 logits（顺序对应 ArmorName 枚举）
   *
   * 后处理流程：
   * 1. sigmoid(objectness) → 阈值过滤
   * 2. argmax 取颜色和编号
   * 3. 角点 → 最小外接矩形（用于 NMS）
   * 4. NMS 去除重叠框
   * 5. 名称/类型校验
   * 6. 传统方法二次修正角点（可选）
   */

  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;

  // 遍历 8400 个候选框（8400 = 80×80 + 40×40 + 20×20 三个检测头）
  for (int r = 0; r < output.rows; r++) {
    double score = output.at<float>(r, 8);
    score = sigmoid(score);

    if (score < score_threshold_) continue;

    std::vector<cv::Point2f> armor_key_points;

    // 颜色和类别独热向量
    cv::Mat color_scores = output.row(r).colRange(9, 13);     // 4 个颜色
    cv::Mat classes_scores = output.row(r).colRange(13, 22);  // 9 个编号
    cv::Point class_id, color_id;
    int _class_id, _color_id;
    double score_color, score_num;
    cv::minMaxLoc(classes_scores, NULL, &score_num, NULL, &class_id);
    cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);
    _class_id = class_id.x;
    _color_id = color_id.x;

    // 解析四个角点，坐标除以 scale 还原到原图尺寸
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 0) / scale, output.at<float>(r, 1) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 6) / scale, output.at<float>(r, 7) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 4) / scale, output.at<float>(r, 5) / scale));
    armor_key_points.push_back(
      cv::Point2f(output.at<float>(r, 2) / scale, output.at<float>(r, 3) / scale));

    // 从四个角点计算最小外接矩形（用于 NMS）
    float min_x = armor_key_points[0].x;
    float max_x = armor_key_points[0].x;
    float min_y = armor_key_points[0].y;
    float max_y = armor_key_points[0].y;

    for (int i = 1; i < armor_key_points.size(); i++) {
      if (armor_key_points[i].x < min_x) min_x = armor_key_points[i].x;
      if (armor_key_points[i].x > max_x) max_x = armor_key_points[i].x;
      if (armor_key_points[i].y < min_y) min_y = armor_key_points[i].y;
      if (armor_key_points[i].y > max_y) max_y = armor_key_points[i].y;
    }

    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    color_ids.emplace_back(_color_id);
    num_ids.emplace_back(_class_id);
    boxes.emplace_back(rect);
    confidences.emplace_back(score);
    armors_key_points.emplace_back(armor_key_points);
  }

  // ---------- NMS（非极大值抑制） ----------
  // 移除重叠度（IoU）超过阈值的重复检测框，保留置信度最高的
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  // 构造 Armor 对象
  std::list<Armor> armors;
  for (const auto & i : indices) {
    if (use_roi_) {
      // 如果使用 ROI，需要将 ROI 内的坐标加上 offset（ROI 左上角在原图中的位置）
      armors.emplace_back(
        color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  // ---------- 后处理过滤 ----------
  tmp_img_ = bgr_img;
  for (auto it = armors.begin(); it != armors.end();) {
    // 名称和置信度检查
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    // 类型匹配检查
    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }

    // 使用传统方法二次修正角点（YOLO 输出的角点可能不够精确）
    if (use_traditional_) detector_.detect(*it, bgr_img);

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

// ========== 步骤 1-4：后处理校验函数 ==========

bool YOLOV5::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;
  return name_ok && confidence_ok;
}

bool YOLOV5::check_type(const Armor & armor) const
{
  /**
   * 类型-名称匹配规则（同传统 Detector）：
   * 小装甲板 → 不能是 one 和 base
   * 大装甲板 → 不能是 two, sentry, outpost
   */
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);
  return name_ok;
}

cv::Point2f YOLOV5::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

// ========== 步骤 1-5：调试绘制 ==========

void YOLOV5::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
}

void YOLOV5::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

double YOLOV5::sigmoid(double x)
{
  /**
   * 数值稳定的 sigmoid 实现：
   * - x > 0：1 / (1 + exp(-x))
   * - x ≤ 0：exp(x) / (1 + exp(x))
   * 避免了当 x 为很大的负数时 exp(-x) 溢出。
   */
  if (x > 0)
    return 1.0 / (1.0 + exp(-x));
  else
    return exp(x) / (1.0 + exp(x));
}

}  // namespace auto_aim
