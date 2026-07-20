/**
 * @file classifier.hpp
 * @brief 装甲板数字图案分类器接口
 * @details 本文件属于 auto_aim 模块的检测子模块。
 *          当 YOLO / 传统检测器检测到装甲板后，需要识别装甲板上的数字图案（1~5 / 哨兵 / 前哨站 / 基地 / 非装甲板），
 *          以确定敌方机器人身份和击打优先级。
 *
 *          本类封装了两种推理后端：
 *          - classify()    : OpenCV DNN (ONNX Runtime) 后端
 *          - ovclassify()  : OpenVINO 后端（通过 OpenVINO API 加载同一 ONNX 模型）
 *
 * 管线位置：步骤 1-2（检测后分类）：Detector 调用 classify() → 填充 Armor::name
 * 模型输入：tiny_resnet.onnx — 轻量 ResNet，输入 1×1×32×32 灰度图
 * 模型输出：9 分类 softmax（对应 ArmorName 枚举的 9 个值）
 * 依赖的外部文件：assets/tiny_resnet.onnx
 */

#ifndef AUTO_AIM__CLASSIFIER_HPP
#define AUTO_AIM__CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>

#include "armor.hpp"

namespace auto_aim
{

/**
 * @brief 装甲板数字图案分类器
 * @details 负责识别装甲板上的数字图案，判断属于哪一台机器人（或建筑）。
 *          输入：裁剪后的装甲板数字区域灰度图（已做透视校正的 pattern）。
 *          输出：填充 Armor::name（ArmorName 枚举）和 Armor::confidence。
 *
 *          包含两种实现路径：
 *          1. classify()  — 用 OpenCV DNN 模块加载 ONNX 模型推理，通用性强
 *          2. ovclassify() — 用 OpenVINO 推理，在 Intel 平台上可能更快
 *
 * @note 非线程安全。外部须保证同一时刻只有一个线程调用 classify 或 ovclassify。
 * @warning 必须先调用构造函数加载模型，否则运行时会抛异常。
 */
class Classifier
{
public:
  /**
   * @brief 构造函数：加载数字分类 ONNX 模型
   * @param[in] config_path YAML 配置文件路径，从中读取 classify_model 字段
   * @throws 如果模型文件不存在或加载失败，会抛出 cv::Exception 或 ov::Exception
   */
  explicit Classifier(const std::string & config_path);

  /**
   * @brief OpenCV DNN 后端推理（步骤 1-2a：传统分类推理）
   * @details 执行步骤：
   *          1. 将 armor.pattern 转为 32×32 灰度图（保持宽高比的 letterbox 缩放）
   *          2. 归一化到 [0,1] 后转为 NCHW blob
   *          3. 前向推理得到 9 维 logits
   *          4. Softmax 转为概率分布
   *          5. 取最大概率对应的类别写入 armor.name 和 armor.confidence
   * @param[in/out] armor 待分类的装甲板。函数执行前需填充 armor.pattern（裁剪的数字区域图像）；
   *                       函数返回后 armor.name 和 armor.confidence 被更新。
   * @note 如果 armor.pattern 为空，直接标记为 not_armor 并返回。
   */
  void classify(Armor & armor);

  /**
   * @brief OpenVINO 后端推理（步骤 1-2b：OpenVINO 加速分类推理）
   * @details 逻辑与 classify() 相同，但使用 OpenVINO API 加载和执行模型。
   *          在 Intel 平台上（NUC、笔记本等），OpenVINO 的推理延迟通常低于 OpenCV DNN。
   * @param[in/out] armor 待分类的装甲板
   */
  void ovclassify(Armor & armor);

private:
  cv::dnn::Net net_;               ///< OpenCV DNN 网络对象（用于 classify()）
  ov::Core core_;                   ///< OpenVINO 核心对象
  ov::CompiledModel compiled_model_; ///< OpenVINO 编译后的模型（用于 ovclassify()）
};

}  // namespace auto_aim

#endif  // AUTO_AIM__CLASSIFIER_HPP
