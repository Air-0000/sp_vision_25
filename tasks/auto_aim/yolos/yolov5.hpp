/**
 * @file yolov5.hpp
 * @brief YOLOv5 OpenVINO 推理后端
 * @details 本文件实现基于 OpenVINO 的 YOLOv5 推理，使用 assets/yolov5.xml + yolov5.bin 模型。
 *          继承了 YOLOBase 抽象接口，提供了 detect() 和 postprocess() 的完整实现。
 *
 * 管线位置：步骤 1（检测），通过 YOLO 工厂创建
 * 模型来源：训练好的 YOLOv5s 四点检测模型，经 OpenVINO Model Optimizer 转为 IR 格式
 *          （.xml = 网络结构，.bin = 权重参数）
 *
 * 输出格式（每行 22 个 float）：
 *   [0-1]   center x, y（注意：不是 box 中心，而是第 1 个角点）
 *   [2-3]   第 2 个角点 x, y
 *   [4-5]   第 3 个角点 x, y
 *   [6-7]   第 4 个角点 x, y
 *   [8]     objectness 置信度（pre-sigmoid raw 值）
 *   [9-12]  颜色分类 logits（4 类：red/blue/extinguish/purple）
 *   [13-21] 数字编号分类 logits（9 类：one~five/sentry/outpost/base/not_armor）
 *
 * 总计 22 个通道 × 8400 个检测框。
 *
 * @note 与 YOLO11 的主要区别：输出通道数不同（22 vs 38），解析逻辑略有差异。
 */

#ifndef AUTO_AIM__YOLOV5_HPP
#define AUTO_AIM__YOLOV5_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim
{

/**
 * @brief YOLOv5 OpenVINO 推理类
 * @details 使用 OpenVINO Runtime 加载和执行 YOLOv5 模型。
 *          所有预处理（letterbox resize）和后处理（sigmoid + NMS + 类别解析）都在此类中完成。
 *
 * 生命周期：
 * 1. 构造函数：加载模型、配置预处理流程
 * 2. detect()：完整管线（预处理 → 推理 → 后处理）
 * 3. postprocess() / parse()：从原始输出解析装甲板
 *
 * @note 支持 ROI 区域检测（配置文件中的 use_roi 控制），
 *       也支持 use_traditional 开关，使用传统 Detector 做角点二次修正。
 */
class YOLOV5 : public YOLOBase
{
public:
  /**
   * @brief 构造函数
   * @param[in] config_path YAML 配置文件路径
   * @param[in] debug 是否开启调试可视化
   * @throws ov::Exception 如果模型加载或编译失败
   */
  YOLOV5(const std::string & config_path, bool debug);

  /**
   * @brief 完整检测管线
   * @param[in] bgr_img 输入 BGR 图像
   * @param[in] frame_count 帧序号（仅用于可视化）
   * @return 检测到的装甲板列表
   */
  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  /**
   * @brief 从原始推理输出解析结果
   * @param[in] scale 预处理缩放比例（坐标还原用）
   * @param[in/out] output OpenVINO 输出的 CV_32F 矩阵 [8400 × 22]
   * @param[in] bgr_img 原始图像（用于传统方法二次修正和调试绘制）
   * @param[in] frame_count 帧序号
   * @return 装甲板列表
   */
  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  // ---------- 配置参数 ----------
  std::string device_, model_path_;     ///< 推理设备（CPU/GPU）和模型路径
  std::string save_path_, debug_path_;  ///< 保存和调试路径
  bool debug_, use_roi_, use_traditional_;  ///< 调试开关、ROI、传统修正

  // ---------- 模型结构参数 ----------
  const int class_num_ = 13;           ///< 分类数（4 颜色 + 9 编号）
  const float nms_threshold_ = 0.3;    ///< NMS 的 IoU 阈值，值越小重复框越少
  const float score_threshold_ = 0.7;  ///< 置信度阈值，低于此值直接丢弃
  double min_confidence_,              ///< 分类器最小置信度（用于 check_name）
         binary_threshold_;            ///< 传统方法的二值化阈值

  // ---------- OpenVINO 对象 ----------
  ov::Core core_;                     ///< OpenVINO 核心对象
  ov::CompiledModel compiled_model_;  ///< 编译后的模型（包含硬件优化）

  // ---------- ROI 参数 ----------
  cv::Rect roi_;           ///< ROI 区域（在原图中的位置）
  cv::Point2f offset_;     ///< ROI 左上角偏移量（用于将 ROI 坐标转回原图坐标）
  cv::Mat tmp_img_;        ///< 临时图像缓存（用于检测绘制和保存）

  // ---------- 传统检测器（用于 YOLO 后处理中的角点二次修正） ----------
  Detector detector_;

  /// 允许 MultiThreadDetector 访问内部成员（用于异步推理管线）
  friend class MultiThreadDetector;

  // ---------- 内部方法 ----------
  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  /// 核心解析函数（被 postprocess 调用）
  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;

  /// Sigmoid 函数（将 raw logit 转为 [0,1] 概率）
  double sigmoid(double x);
};

}  // namespace auto_aim

#endif  //AUTO_AIM__YOLOV5_HPP
