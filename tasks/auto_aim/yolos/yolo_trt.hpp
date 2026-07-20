#ifndef AUTO_AIM__YOLO_TRT_HPP
#define AUTO_AIM__YOLO_TRT_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim
{

/**
 * @brief TensorRT YOLO inference backend.
 *
 * Loads a TensorRT engine (.engine / .plan) built from best2-sim.onnx.
 * Supports FP16/INT8 quantization. Falls back to CPU if CUDA is unavailable.
 *
 * Usage:
 *   1. Convert ONNX to engine offline:
 *      trtexec --onnx=best2-sim.onnx --saveEngine=best2-sim.engine --fp16
 *   2. Set yolo_name: "yolov5_trt" in config
 *   3. Set yolo_trt_engine_path: "assets/best2-sim.engine"
 */
class YOLO_TRT : public YOLOBase
{
public:
  YOLO_TRT(const std::string & config_path, bool debug);
  ~YOLO_TRT() override;

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  // --- Config ---
  std::string engine_path_, device_;
  std::string save_path_;
  bool debug_, use_roi_, use_traditional_;
  double min_confidence_, binary_threshold_;

  const int input_h_ = 640;
  const int input_w_ = 640;
  const int class_num_ = 13;       // 4 color + 9 number (matches yolo11 output)
  const float nms_threshold_ = 0.3f;
  const float score_threshold_ = 0.7f;

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  Detector detector_;

  // --- TensorRT internals (opaque pointer to hide nvinfer1 headers) ---
  struct TRTContext;
  std::unique_ptr<TRTContext> ctx_;

  // --- Pre/Post processing ---
  void preprocess(const cv::Mat & bgr_img, float * gpu_input);
  std::list<Armor> parse(
    double scale, float * cpu_output, const cv::Mat & bgr_img, int frame_count);

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLO_TRT_HPP
