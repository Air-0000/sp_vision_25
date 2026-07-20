/**
 * @file yolo.hpp
 * @brief YOLO 检测器抽象接口与工厂类
 * @details 本文件属于 auto_aim 模块的检测子模块，定义了统一的 YOLO 检测接口（YOLOBase）
 *          和工厂类（YOLO），支持多种 YOLO 版本（YOLOv5/YOLOv8/YOLO11）和推理后端（OpenVINO/TensorRT）。
 *
 *          工厂类 YOLO 根据配置文件的 yolo_name 字段自动创建对应的子类实例：
 *          - "yolov5"  → YOLOV5（OpenVINO 后端，使用 yolov5.xml）
 *          - "yolov8"  → YOLOV8（OpenVINO 后端，使用 yolov8.xml）
 *          - "yolo11"  → YOLO11（OpenVINO 后端，使用 yolo11.xml）
 *          - "yolov5_trt" / "yolo11_trt" → YOLO_TRT（TensorRT 后端）
 *
 * 管线位置：步骤 1（检测），YOLO::detect() 替代或补充传统 Detector
 * 输出：std::list<Armor>（同传统方法，包括角点、颜色、编号、置信度）
 *
 * @note 虽然 YOLO 已经能输出 4 个角点，但通常仍然会用传统 Detector::detect(Armor&, Mat)
 *       对角点做二次修正（PCA/minAreaRect 校正），以提升 PnP 精度。
 */

#ifndef AUTO_AIM__YOLO_HPP
#define AUTO_AIM__YOLO_HPP

#include <opencv2/opencv.hpp>

#include "armor.hpp"

namespace auto_aim
{

/**
 * @brief YOLO 检测器抽象基类
 * @details 定义所有 YOLO 版本必须实现的 detect() 和 postprocess() 接口。
 *          detect()：输入图像 → 输出装甲板列表（包含预处理 → 推理 → 后处理的完整流程）
 *          postprocess()：从原始输出张量解析装甲板（供多线程异步推理管线使用）
 *
 *          设计模式：策略模式（Strategy），通过 YOLO 工厂类选择具体策略。
 */
class YOLOBase
{
public:
  /**
   * @brief 完整检测管线（预处理 + 推理 + 后处理）
   * @param[in] img 输入 BGR 图像（可以是原图或 ROI 裁剪后的子图）
   * @param[in] frame_count 帧序号（用于调试窗口标题）
   * @return 检测到的装甲板列表（空列表表示未检测到任何目标）
   */
  virtual std::list<Armor> detect(const cv::Mat & img, int frame_count) = 0;

  /**
   * @brief 仅后处理（从推理输出张量解析装甲板）
   * @details 专为多线程异步推理管线设计：主线程做预处理 + 启动推理，
   *          此函数在推理完成后被调用，从原始 float 输出解析出装甲板列表。
   * @param[in] scale 预处理时的缩放比例（用于将输出坐标还原到原图坐标）
   * @param[in/out] output 推理输出的 float 矩阵（OpenCV Mat 封装）
   * @param[in] bgr_img 原始 BGR 图像（用于调试绘制和角点二次修正）
   * @param[in] frame_count 帧序号
   * @return 检测到的装甲板列表
   */
  virtual std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) = 0;
};

/**
 * @brief YOLO 工厂类（策略选择器）
 * @details 根据配置文件中的 yolo_name 字段，自动创建对应的 YOLO 子类实例。
 *          使用 unique_ptr 管理子类生命周期，对调用方透明。
 *
 * 用法：
 * @code
 *   YOLO detector(config_path);  // 自动选择子类
 *   auto armors = detector.detect(img);  // 多态调用
 * @endcode
 */
class YOLO
{
public:
  /**
   * @brief 构造函数：根据配置选择 YOLO 后端
   * @param[in] config_path YAML 配置文件路径
   * @param[in] debug 是否开启调试可视化
   * @throws std::runtime_error 如果 yolo_name 无法识别
   */
  YOLO(const std::string & config_path, bool debug = true);

  /**
   * @brief 委托给子类的 detect()
   */
  std::list<Armor> detect(const cv::Mat & img, int frame_count = -1);

  /**
   * @brief 委托给子类的 postprocess()
   */
  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

private:
  std::unique_ptr<YOLOBase> yolo_;  ///< 具体的 YOLO 子类实例（多态派发）
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLO_HPP
