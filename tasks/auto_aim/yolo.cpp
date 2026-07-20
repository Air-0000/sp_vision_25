/**
 * @file yolo.cpp
 * @brief YOLO 工厂类实现
 * @details 根据配置文件中的 yolo_name 字段，选择并实例化对应的 YOLO 子类。
 *
 * 当前支持的 yolo_name 值：
 * - "yolov5"      → 使用 OpenVINO 加载 assets/yolov5.xml（YOLOv5s 架构）
 * - "yolov8"      → 使用 OpenVINO 加载 assets/yolov8.xml（YOLOv8s 架构）
 * - "yolo11"      → 使用 OpenVINO 加载 assets/yolo11.xml（YOLO11 架构）
 * - "yolov5_trt"  → 使用 TensorRT 加载 assets/best2-sim.engine
 * - "yolo11_trt"  → 同上（TensorRT 后端，引擎文件统一）
 */

#include "yolo.hpp"

#include <yaml-cpp/yaml.h>

#include "yolos/yolo11.hpp"
#include "yolos/yolov5.hpp"
#include "yolos/yolov8.hpp"
#include "yolos/yolo_trt.hpp"

namespace auto_aim
{
YOLO::YOLO(const std::string & config_path, bool debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();

  if (yolo_name == "yolov8") {
    yolo_ = std::make_unique<YOLOV8>(config_path, debug);
  }

  else if (yolo_name == "yolo11") {
    yolo_ = std::make_unique<YOLO11>(config_path, debug);
  }

  else if (yolo_name == "yolov5") {
    yolo_ = std::make_unique<YOLOV5>(config_path, debug);
  }

  else if (yolo_name == "yolov5_trt" || yolo_name == "yolo11_trt") {
    yolo_ = std::make_unique<YOLO_TRT>(config_path, debug);
  }

  else {
    throw std::runtime_error("Unknown yolo name: " + yolo_name + "!");
  }
}

std::list<Armor> YOLO::detect(const cv::Mat & img, int frame_count)
{
  return yolo_->detect(img, frame_count);
}

std::list<Armor> YOLO::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return yolo_->postprocess(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim
