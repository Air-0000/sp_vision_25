/**
 * @file classifier.cpp
 * @brief 装甲板数字图案分类器实现
 * @details 实现两个推理后端（OpenCV DNN 和 OpenVINO），
 *          对裁剪出的装甲板数字区域灰度图进行分类，识别数字编号。
 *
 *          模型文件：assets/tiny_resnet.onnx
 *          - 输入：1×1×32×32 灰度图（像素值 [0, 255]，送入网络前归一化到 [0, 1]）
 *          - 输出：1×9 的 logits（对应 9 类：one~five, sentry, outpost, base, not_armor）
 *          - 后处理：Softmax 归一化后取 argmax
 *
 * @note 两种后端的预处理逻辑完全一致，仅推理引擎不同。
 */

#include "classifier.hpp"

#include <yaml-cpp/yaml.h>

namespace auto_aim
{

// ========== 步骤 1-2：初始化分类器 ==========

Classifier::Classifier(const std::string & config_path)
{
  /**
   * 从配置文件中读取模型路径，分别加载到 OpenCV DNN 和 OpenVINO 中。
   * OpenVINO 使用 AUTO 设备（自动选择 CPU/GPU/GPU 中可用的），
   * 性能模式设为 LATENCY（最小化单帧推理延迟，而非最大化吞吐量）。
   */
  auto yaml = YAML::LoadFile(config_path);
  auto model = yaml["classify_model"].as<std::string>();

  net_ = cv::dnn::readNetFromONNX(model);

  auto ovmodel = core_.read_model(model);
  compiled_model_ = core_.compile_model(
    ovmodel, "AUTO", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

// ========== 步骤 1-2a：OpenCV DNN 分类推理 ==========

void Classifier::classify(Armor & armor)
{
  /**
   * 预处理：
   * 1. 将 pattern 转为灰度图（因为颜色信息对数字识别无帮助，反而增加计算量）
   * 2. letterbox 缩放到 32×32（保持宽高比，不足部分补 0）
   * 3. 转为 blob：NCHW 格式 (1×1×32×32)，像素值归一化到 [0, 1]
   */
  if (armor.pattern.empty()) {
    armor.name = ArmorName::not_armor;
    return;
  }

  cv::Mat gray;
  cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

  auto input = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
  auto x_scale = static_cast<double>(32) / gray.cols;
  auto y_scale = static_cast<double>(32) / gray.rows;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(gray.rows * scale);
  auto w = static_cast<int>(gray.cols * scale);

  // 如果缩放后尺寸为 0（图案太小），标记为无效装甲板
  if (h == 0 || w == 0) {
    armor.name = ArmorName::not_armor;
    return;
  }
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(gray, input(roi), {w, h});

  auto blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar());

  // 前向推理
  net_.setInput(blob);
  cv::Mat outputs = net_.forward();

  // ---------- Softmax 归一化 ----------
  // 为数值稳定性，先减去最大值再取 exp
  float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
  cv::exp(outputs - max, outputs);
  float sum = cv::sum(outputs)[0];
  outputs /= sum;

  // 取最大概率类别
  double confidence;
  cv::Point label_point;
  cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
  int label_id = label_point.x;

  armor.confidence = confidence;
  armor.name = static_cast<ArmorName>(label_id);
}

// ========== 步骤 1-2b：OpenVINO 加速分类推理 ==========

void Classifier::ovclassify(Armor & armor)
{
  /**
   * 预处理逻辑与 classify() 完全相同，仅推理引擎改为 OpenVINO。
   * OpenVINO 的输入张量为 NCHW float32 [1×1×32×32]。
   */
  if (armor.pattern.empty()) {
    armor.name = ArmorName::not_armor;
    return;
  }

  cv::Mat gray;
  cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

  // Resize image to 32x32
  auto input = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
  auto x_scale = static_cast<double>(32) / gray.cols;
  auto y_scale = static_cast<double>(32) / gray.rows;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(gray.rows * scale);
  auto w = static_cast<int>(gray.cols * scale);

  if (h == 0 || w == 0) {
    armor.name = ArmorName::not_armor;
    return;
  }

  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(gray, input(roi), {w, h});
  // Normalize the input image to [0, 1] range
  input.convertTo(input, CV_32F, 1.0 / 255.0);

  ov::Tensor input_tensor(ov::element::f32, {1, 1, 32, 32}, input.data);

  ov::InferRequest infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat outputs(1, 9, CV_32F, output_tensor.data());

  // Softmax
  float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
  cv::exp(outputs - max, outputs);
  float sum = cv::sum(outputs)[0];
  outputs /= sum;

  double confidence;
  cv::Point label_point;
  cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
  int label_id = label_point.x;

  armor.confidence = confidence;
  armor.name = static_cast<ArmorName>(label_id);
}

}  // namespace auto_aim
