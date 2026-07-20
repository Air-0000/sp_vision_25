/**
 * @file detector.cpp
 * @brief 传统视觉方法装甲板检测器实现
 * @details 实现基于灯条（Lightbar）的传统装甲板检测算法。
 *          完整管线：颜色通道提取 → 二值化 → findContours → minAreaRect → 灯条几何筛选
 *          → 灯条配对（角度差/长度比/中心距）→ 装甲板构造 → 数字分类 → 重复灯条剔除。
 *
 *          除了完整检测管线外，还提供单装甲板修正功能（detect(Armor&, Mat)），
 *          用于 YOLO 检测后的角点二次矫正（PCA 修正），提高 PnP 解算精度。
 *
 * @note 传统方法参数高度依赖经验调参，threshold/min_lightbar_ratio 等参数需根据
 *       比赛场地光照条件调整。建议开启 debug 模式观察二值化效果。
 */

#include "detector.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{

// ========== 步骤 1-1：检测器初始化 ==========

Detector::Detector(const std::string & config_path, bool debug)
: classifier_(config_path), debug_(debug)
{
  /**
   * 从 YAML 配置文件中读取所有传统检测参数。
   * 角度参数从度（degree）转为弧度（rad），因为内部几何计算统一使用弧度。
   */
  auto yaml = YAML::LoadFile(config_path);

  threshold_ = yaml["threshold"].as<double>();
  max_angle_error_ = yaml["max_angle_error"].as<double>() / 57.3;  // degree → rad
  min_lightbar_ratio_ = yaml["min_lightbar_ratio"].as<double>();
  max_lightbar_ratio_ = yaml["max_lightbar_ratio"].as<double>();
  min_lightbar_length_ = yaml["min_lightbar_length"].as<double>();
  min_armor_ratio_ = yaml["min_armor_ratio"].as<double>();
  max_armor_ratio_ = yaml["max_armor_ratio"].as<double>();
  max_side_ratio_ = yaml["max_side_ratio"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  max_rectangular_error_ = yaml["max_rectangular_error"].as<double>() / 57.3;  // degree → rad

  // 创建图案保存目录（用于保存分类置信度低的样本，后续人工标注后扩充训练集）
  save_path_ = "patterns";
  std::filesystem::create_directory(save_path_);
}

// ========== 步骤 1-2：完整检测管线 ==========

std::list<Armor> Detector::detect(const cv::Mat & bgr_img, int frame_count)
{
  // ---------- 步骤 1-2a：灰度化 + 二值化 ----------
  // 先转灰度，再根据 threshold 阈值做全局二值化。
  // 灯条在图像中通常是非常亮（高灰度值）的垂直条带，二值化后灯条区域为白色。
  cv::Mat gray_img;
  cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);

  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, threshold_, 255, cv::THRESH_BINARY);
  cv::imshow("binary_img", binary_img);

  // ---------- 步骤 1-2b：轮廓提取 ----------
  // 从二值图像中找出所有白色连通域的轮廓。
  // RETR_EXTERNAL = 只提取最外层轮廓（忽略嵌套内轮廓）
  // CHAIN_APPROX_NONE = 保留轮廓所有点（不用近似）
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  // ---------- 步骤 1-2c：灯条提取与筛选 ----------
  // 对每个轮廓求最小外接旋转矩形，这个矩形就是候选灯条。
  // 筛选条件：灯条长宽比、长度、角度误差必须都在合理范围内。
  std::size_t lightbar_id = 0;
  std::list<Lightbar> lightbars;
  for (const auto & contour : contours) {
    auto rotated_rect = cv::minAreaRect(contour);
    auto lightbar = Lightbar(rotated_rect, lightbar_id);

    if (!check_geometry(lightbar)) continue;

    lightbar.color = get_color(bgr_img, contour);
    lightbars.emplace_back(lightbar);
    lightbar_id += 1;
  }

  // ---------- 步骤 1-2d：灯条排序 ----------
  // 按中心 x 坐标从左到右排序，确保后续配对时 left < right
  lightbars.sort([](const Lightbar & a, const Lightbar & b) { return a.center.x < b.center.x; });

  // ---------- 步骤 1-2e：灯条配对 → 装甲板构造 ----------
  // 对每一对 left < right 的灯条：
  // 1. 颜色必须相同（不能是一个红一个蓝）
  // 2. 构造装甲板后检查几何约束（宽高比、侧比、矩形度）
  // 3. 裁剪数字区域 → 分类器识别数字编号
  // 4. 检查名称和类型是否合法
  std::list<Armor> armors;
  for (auto left = lightbars.begin(); left != lightbars.end(); left++) {
    for (auto right = std::next(left); right != lightbars.end(); right++) {
      if (left->color != right->color) continue;

      auto armor = Armor(*left, *right);
      if (!check_geometry(armor)) continue;

      armor.pattern = get_pattern(bgr_img, armor);
      classifier_.classify(armor);
      if (!check_name(armor)) continue;

      armor.type = get_type(armor);
      if (!check_type(armor)) continue;

      armor.center_norm = get_center_norm(bgr_img, armor.center);
      armors.emplace_back(armor);
    }
  }

  // ---------- 步骤 1-2f：重复灯条剔除 ----------
  // 同一个灯条可能被多个装甲板共享（例如相邻的两个装甲板各使用了同一个灯条）。
  // 两种情况：
  // 1. 装甲板重叠（共享左或右灯条 ID）→ 保留 ROI 面积小的
  // 2. 装甲板相连（一个的右灯条 ID = 另一个的左灯条 ID）→ 保留置信度高的
  for (auto armor1 = armors.begin(); armor1 != armors.end(); armor1++) {
    for (auto armor2 = std::next(armor1); armor2 != armors.end(); armor2++) {
      if (
        armor1->left.id != armor2->left.id && armor1->left.id != armor2->right.id &&
        armor1->right.id != armor2->left.id && armor1->right.id != armor2->right.id) {
        continue;
      }

      // 装甲板重叠, 保留 roi 小的（即数字区域面积小的那个更可能是正确匹配）
      if (armor1->left.id == armor2->left.id || armor1->right.id == armor2->right.id) {
        auto area1 = armor1->pattern.cols * armor1->pattern.rows;
        auto area2 = armor2->pattern.cols * armor2->pattern.rows;
        if (area1 < area2)
          armor2->duplicated = true;
        else
          armor1->duplicated = true;
      }

      // 装甲板相连, 保留置信度大的
      if (armor1->left.id == armor2->right.id || armor1->right.id == armor2->left.id) {
        if (armor1->confidence < armor2->confidence)
          armor1->duplicated = true;
        else
          armor2->duplicated = true;
      }
    }
  }

  // 移除所有标记为 duplicated 的装甲板
  armors.remove_if([&](const Armor & a) { return a.duplicated; });

  if (debug_) show_result(binary_img, bgr_img, lightbars, armors, frame_count);

  return armors;
}

// ========== 步骤 1-3：单装甲板角点修正（用于 YOLO 后处理） ==========

bool Detector::detect(Armor & armor, const cv::Mat & bgr_img)
{
  /**
   * 对 YOLO 输出的单个装甲板，用传统灯条检测方法修正其四个角点。
   *
   * 流程：
   * 1. 根据 YOLO 给出的四个角点，计算左右灯条的候选区域（ROI）
   * 2. 在该 ROI 内做二值化 + 灯条检测
   * 3. 找到距左右灯条最近的轮廓，用其 top/bottom 更新角点
   *
   * 这样做的原因是 YOLO 输出的角点可能不够精确（像素级别的偏移），
   * 而传统方法的灯条角点基于轮廓的 minAreaRect，在某些情况下更准确。
   */

  // 取得四个角点
  auto tl = armor.points[0];  // 左上
  auto tr = armor.points[1];  // 右上
  auto br = armor.points[2];  // 右下
  auto bl = armor.points[3];  // 左下

  // 计算向量和调整后的点
  auto lt2b = bl - tl;        // 左灯条方向向量（从上到下）
  auto rt2b = br - tr;        // 右灯条方向向量（从上到下）

  // 将角点沿灯条方向延伸，得到更大范围的搜索区域
  auto tl1 = (tl + bl) / 2 - lt2b;
  auto bl1 = (tl + bl) / 2 + lt2b;
  auto br1 = (tr + br) / 2 + rt2b;
  auto tr1 = (tr + br) / 2 - rt2b;

  auto tl2tr = tr1 - tl1;     // 左右方向向量
  auto bl2br = br1 - bl1;     // 左右方向向量

  // 缩放到 75% 宽度（缩小搜索区域排除灯条两边的干扰）
  auto tl2 = (tl1 + tr) / 2 - 0.75 * tl2tr;
  auto tr2 = (tl1 + tr) / 2 + 0.75 * tl2tr;
  auto bl2 = (bl1 + br) / 2 - 0.75 * bl2br;
  auto br2 = (bl1 + br) / 2 + 0.75 * bl2br;

  // 构造新的四个角点
  std::vector<cv::Point> points = {tl2, tr2, br2, bl2};
  auto armor_rotaterect = cv::minAreaRect(points);
  cv::Rect boundingBox = armor_rotaterect.boundingRect();

  // 检查 boundingBox 是否超出图像边界（如果超出直接返回，无法在该 ROI 内做传统检测）
  if (
    boundingBox.x < 0 || boundingBox.y < 0 || boundingBox.x + boundingBox.width > bgr_img.cols ||
    boundingBox.y + boundingBox.height > bgr_img.rows) {
    return false;
  }

  // 在图像上裁剪出这个矩形区域（ROI）
  cv::Mat armor_roi = bgr_img(boundingBox);
  if (armor_roi.empty()) {
    return false;
  }

  // ---------- 在 ROI 内做传统灯条检测 ----------
  cv::Mat gray_img;
  cv::cvtColor(armor_roi, gray_img, cv::COLOR_BGR2GRAY);
  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, threshold_, 255, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  // 转换为 Lightbar 并筛选
  std::size_t lightbar_id = 0;
  std::list<Lightbar> lightbars;
  for (const auto & contour : contours) {
    auto rotated_rect = cv::minAreaRect(contour);
    auto lightbar = Lightbar(rotated_rect, lightbar_id);

    if (!check_geometry(lightbar)) continue;

    lightbar.color = get_color(bgr_img, contour);
    // PCA 角点修正当前已关闭（实测发现 OpenCV minAreaRect 已经够用）
    // lightbar_points_corrector(lightbar, gray_img);
    lightbars.emplace_back(lightbar);
    lightbar_id += 1;
  }

  if (lightbars.size() < 2) return false;

  lightbars.sort([](const Lightbar & a, const Lightbar & b) { return a.center.x < b.center.x; });

  // ---------- 灯条匹配 ----------
  // 计算与 tl/bl 距离最近 → 左灯条，与 br/tr 距离最近 → 右灯条
  // 因为 YOLO 给出的角点顺序：tl=左上, tr=右上, br=右下, bl=左下
  Lightbar * closest_left_lightbar = nullptr;
  Lightbar * closest_right_lightbar = nullptr;
  float min_distance_tl_bl = std::numeric_limits<float>::max();
  float min_distance_br_tr = std::numeric_limits<float>::max();

  for (auto & lightbar : lightbars) {
    // 左灯条距离 = 灯条 top 到 tl 的距离 + 灯条 bottom 到 bl 的距离
    float distance_tl_bl =
      cv::norm(tl - (lightbar.top + cv::Point2f(boundingBox.x, boundingBox.y))) +
      cv::norm(bl - (lightbar.bottom + cv::Point2f(boundingBox.x, boundingBox.y)));
    if (distance_tl_bl < min_distance_tl_bl) {
      min_distance_tl_bl = distance_tl_bl;
      closest_left_lightbar = &lightbar;
    }

    // 右灯条距离 = 灯条 bottom 到 br 的距离 + 灯条 top 到 tr 的距离
    float distance_br_tr =
      cv::norm(br - (lightbar.bottom + cv::Point2f(boundingBox.x, boundingBox.y))) +
      cv::norm(tr - (lightbar.top + cv::Point2f(boundingBox.x, boundingBox.y)));
    if (distance_br_tr < min_distance_br_tr) {
      min_distance_br_tr = distance_br_tr;
      closest_right_lightbar = &lightbar;
    }
  }

  // 如果总匹配距离 < 15 像素，则接受这组修正
  if (
    closest_left_lightbar && closest_right_lightbar &&
    min_distance_br_tr + min_distance_tl_bl < 15) {
    // 将四个点从 armor_roi 坐标系转换到原始图像坐标系
    armor.points[0] = closest_left_lightbar->top + cv::Point2f(boundingBox.x, boundingBox.y);
    armor.points[1] = closest_right_lightbar->top + cv::Point2f(boundingBox.x, boundingBox.y);
    armor.points[2] = closest_right_lightbar->bottom + cv::Point2f(boundingBox.x, boundingBox.y);
    armor.points[3] = closest_left_lightbar->bottom + cv::Point2f(boundingBox.x, boundingBox.y);
    return true;
  }

  return false;
}

// ========== 步骤 1-4：灯条几何校验 ==========

bool Detector::check_geometry(const Lightbar & lightbar) const
{
  /**
   * 灯条合法性检查：
   * - angle_ok：灯条接近垂直（角度误差小），因为 RoboMaster 装甲板灯条在车辆直立时近似竖直
   * - ratio_ok：长宽比合适（灯条是细长的条形，不会太方也不会太细）
   * - length_ok：像素长度足够（太小的可能是远处噪点）
   */
  auto angle_ok = lightbar.angle_error < max_angle_error_;
  auto ratio_ok = lightbar.ratio > min_lightbar_ratio_ && lightbar.ratio < max_lightbar_ratio_;
  auto length_ok = lightbar.length > min_lightbar_length_;
  return angle_ok && ratio_ok && length_ok;
}

// ========== 步骤 1-5：装甲板几何校验 ==========

bool Detector::check_geometry(const Armor & armor) const
{
  /**
   * 装甲板合法性检查（左右灯条配对约束）：
   * - ratio_ok：装甲板宽高比（左右灯条间距 / 灯条长度）。太窄或太宽都不可能是装甲板
   * - side_ratio_ok：左右灯条长度比接近 1（正常的装甲板两侧灯条等长）
   * - rectangular_error_ok：灯条法向与灯条中点连线夹角接近 90°
   *   即两个灯条近似平行（rugby 形状的装甲板两个灯条近似平行）
   */
  auto ratio_ok = armor.ratio > min_armor_ratio_ && armor.ratio < max_armor_ratio_;
  auto side_ratio_ok = armor.side_ratio < max_side_ratio_;
  auto rectangular_error_ok = armor.rectangular_error < max_rectangular_error_;
  return ratio_ok && side_ratio_ok && rectangular_error_ok;
}

// ========== 步骤 1-6：名称合法性检查 ==========

bool Detector::check_name(const Armor & armor) const
{
  /**
   * 检查分类器输出的名称是否有效：
   * - not_armor = 分类器认为这不是一个合法的装甲板（可能只是两个灯条恰好组合）
   * - confidence < min_confidence = 分类器信心不足
   */
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案（名称合法但置信度不够），用于后续人工标注后扩充分类器训练集
  if (name_ok && !confidence_ok) save(armor);

  // 出现 5 号则显示 debug 信息（5 号是英雄特有，值得关注）
  if (armor.name == ArmorName::five) tools::logger()->debug("See pattern 5");

  return name_ok && confidence_ok;
}

// ========== 步骤 1-7：类型匹配检查 ==========

bool Detector::check_type(const Armor & armor) const
{
  /**
   * 检查装甲板类型（big/small）与名称的匹配关系。
   * 根据 RoboMaster 25 赛季规则：
   * - 小装甲板：除 1 号和基地外所有兵种
   * - 大装甲板：仅 1 号（英雄）和基地
   * 如果出现不匹配（如小装甲板检测到 1 号），保存图案用于后续排查。
   */
  auto name_ok = armor.type == ArmorType::small
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name == ArmorName::one || armor.name == ArmorName::base);

  // 保存异常的图案，用于分类器的迭代
  if (!name_ok) {
    tools::logger()->debug(
      "see strange armor: {} {}", ARMOR_TYPES[armor.type], ARMOR_NAMES[armor.name]);
    save(armor);
  }

  return name_ok;
}

// ========== 步骤 1-8：灯条颜色判断 ==========

Color Detector::get_color(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour) const
{
  /**
   * 通过计算轮廓区域内所有像素的 B 通道和 R 通道总和来判断颜色。
   * 简单直观：蓝色灯条的 B 通道和 > R 通道和，红色灯条反之。
   * @note 这种方法对光照变化敏感，但胜在快速且无需额外模型。
   *       BGR 格式下：bgr_img[][0]=B, [][1]=G, [][2]=R
   */
  int red_sum = 0, blue_sum = 0;

  for (const auto & point : contour) {
    red_sum += bgr_img.at<cv::Vec3b>(point)[2];   // R 通道求和
    blue_sum += bgr_img.at<cv::Vec3b>(point)[0];  // B 通道求和
  }

  return blue_sum > red_sum ? Color::blue : Color::red;
}

// ========== 步骤 1-9：数字图案区域裁剪 ==========

cv::Mat Detector::get_pattern(const cv::Mat & bgr_img, const Armor & armor) const
{
  /**
   * 根据左右灯条的中心和方向向量，计算装甲板数字区域的 ROI。
   *
   * 灯条长度 ~56mm，装甲板高度 ~126mm（小装甲板尺寸 130×50mm），
   * 因此数字区域 ≈ 左右灯条中心连线 + 灯条长度方向延伸 1.125 倍（0.5×126/56）。
   *
   * 公式：装甲板上下边界 ≈ 灯条中心 ± 灯条方向向量 × 1.125
   */
  auto tl = armor.left.center - armor.left.top2bottom * 1.125;
  auto bl = armor.left.center + armor.left.top2bottom * 1.125;
  auto tr = armor.right.center - armor.right.top2bottom * 1.125;
  auto br = armor.right.center + armor.right.top2bottom * 1.125;

  // 边界约束（不能超过图像范围）
  auto roi_left = std::max<int>(std::min(tl.x, bl.x), 0);
  auto roi_top = std::max<int>(std::min(tl.y, tr.y), 0);
  auto roi_right = std::min<int>(std::max(tr.x, br.x), bgr_img.cols);
  auto roi_bottom = std::min<int>(std::max(bl.y, br.y), bgr_img.rows);
  auto roi_tl = cv::Point(roi_left, roi_top);
  auto roi_br = cv::Point(roi_right, roi_bottom);
  auto roi = cv::Rect(roi_tl, roi_br);

  return bgr_img(roi);
}

// ========== 步骤 1-10：装甲板类型判断 ==========

ArmorType Detector::get_type(const Armor & armor)
{
  /**
   * 根据装甲板宽高比（ratio）判断大小装甲板。
   * ratio = 左右灯条间距 / 灯条长度
   *
   * 小装甲板 130×50mm：灯条间距 ≈ 130-2×10=110mm，灯条长度 ≈ 50mm，ratio ≈ 2.2
   * 大装甲板 230×130mm：灯条间距 ≈ 230-2×20=190mm，灯条长度 ≈ 130mm，ratio ≈ 1.46
   * 但实际中由于透视变形，ratio 会变化，所以用 3.0 和 2.5 做阈值。
   *
   * @note 25 赛季规则下，也可以直接根据图案名称判断：
   *       - 1 号和 base → 大装甲板
   *       - 其他 → 小装甲板
   */
  if (armor.ratio > 3.0) {
    return ArmorType::big;
  }

  if (armor.ratio < 2.5) {
    return ArmorType::small;
  }

  // 比例在 2.5~3.0 之间的模糊区域，用名称辅助判断
  if (armor.name == ArmorName::one || armor.name == ArmorName::base) {
    return ArmorType::big;
  }

  return ArmorType::small;
}

// ========== 辅助函数 ==========

cv::Point2f Detector::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  /**
   * 计算归一化中心坐标（将像素坐标映射到 [0, 1]），
   * 用于后续的 ROI 跟踪或全向感知模块中的坐标对齐。
   */
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void Detector::save(const Armor & armor) const
{
  /**
   * 将分类置信度低的装甲板保存到磁盘，供后续人工标注和模型训练。
   * 文件名格式：{名称}_{时间戳}.jpg
   */
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, armor.pattern);
}

void Detector::show_result(
  const cv::Mat & binary_img, const cv::Mat & bgr_img, const std::list<Lightbar> & lightbars,
  const std::list<Armor> & armors, int frame_count) const
{
  /**
   * 调试可视化：在图像上绘制检测到的灯条（黄色）和装甲板（绿色），
   * 并标注每个目标的几何参数和分类结果。
   */
  auto detection = bgr_img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});

  for (const auto & lightbar : lightbars) {
    auto info = fmt::format(
      "{:.1f} {:.1f} {:.1f} {}", lightbar.angle_error * 57.3, lightbar.ratio, lightbar.length,
      COLORS[lightbar.color]);
    tools::draw_text(detection, info, lightbar.top, {0, 255, 255});
    tools::draw_points(detection, lightbar.points, {0, 255, 255}, 3);
  }

  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {:.2f} {:.1f} {:.2f} {} {}", armor.ratio, armor.side_ratio,
      armor.rectangular_error * 57.3, armor.confidence, ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.left.bottom, {0, 255, 0});
  }

  cv::Mat binary_img2;
  cv::resize(binary_img, binary_img2, {}, 0.5, 0.5);
  cv::resize(detection, detection, {}, 0.5, 0.5);

  cv::imshow("detection", detection);
}

// ========== 步骤 1-11：PCA 灯条角点修正 ==========

void Detector::lightbar_points_corrector(Lightbar & lightbar, const cv::Mat & gray_img) const
{
  /**
   * PCA（主成分分析）灯条角点修正器。
   *
   * 原理：灯条区域的像素在灰度图上的分布可以近似为一个椭圆形，
   * PCA 可以求出该椭圆的主轴方向（即灯条的长轴方向），
   * 然后沿着主轴方向搜索亮度跳变点来确定灯条的精确端点。
   *
   * 这样做比直接使用 minAreaRect 的角点更精确，因为 minAreaRect
   * 对轮廓噪声敏感，而 PCA 利用区域内的所有像素进行统计回归。
   *
   * @note 当前实现中 PCA 角点修正已关闭（在 detect(Armor&) 中被注释掉），
   *       因为实测发现 minAreaRect 在大多数场景下已经够用，
   *       且 PCA 计算量较大（需遍历 ROI 内所有像素），会降低帧率。
   */

  // 配置参数
  constexpr float MAX_BRIGHTNESS = 25;  // 归一化最大亮度值
  constexpr float ROI_SCALE = 0.07;     // ROI 扩展比例（灯条矩形向外扩展 7%）
  constexpr float SEARCH_START = 0.4;   // 搜索起始位置比例（相对灯条长度）
  constexpr float SEARCH_END = 0.6;     // 搜索结束位置比例

  // 扩展并裁剪 ROI
  cv::Rect roi_box = lightbar.rotated_rect.boundingRect();
  roi_box.x -= roi_box.width * ROI_SCALE;
  roi_box.y -= roi_box.height * ROI_SCALE;
  roi_box.width += 2 * roi_box.width * ROI_SCALE;
  roi_box.height += 2 * roi_box.height * ROI_SCALE;

  // 边界约束
  roi_box &= cv::Rect(0, 0, gray_img.cols, gray_img.rows);

  // 归一化 ROI
  cv::Mat roi = gray_img(roi_box);
  const float mean_val = cv::mean(roi)[0];
  roi.convertTo(roi, CV_32F);
  cv::normalize(roi, roi, 0, MAX_BRIGHTNESS, cv::NORM_MINMAX);

  // ---------- 计算质心 ----------
  // 用灰度值作为权重的加权质心，比几何中心更准确
  const cv::Moments moments = cv::moments(roi);
  const cv::Point2f centroid(
    moments.m10 / moments.m00 + roi_box.x, moments.m01 / moments.m00 + roi_box.y);

  // ---------- 生成稀疏点云 ----------
  // 将每个像素作为二维点，以其归一化亮度作为权重，送入 PCA
  std::vector<cv::Point2f> points;
  for (int i = 0; i < roi.rows; ++i) {
    for (int j = 0; j < roi.cols; ++j) {
      const float weight = roi.at<float>(i, j);
      if (weight > 1e-3) {          // 忽略极小值提升性能
        points.emplace_back(j, i);  // 坐标相对于 ROI 区域
      }
    }
  }

  // ---------- PCA 计算对称轴方向 ----------
  cv::PCA pca(cv::Mat(points).reshape(1), cv::Mat(), cv::PCA::DATA_AS_ROW);
  cv::Point2f axis(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
  axis /= cv::norm(axis);

  // 统一方向（让 y 分量为负，即向上）
  if (axis.y > 0) axis = -axis;

  // ---------- 沿主轴搜索端点 ----------
  const auto find_corner = [&](int direction) -> cv::Point2f {
    /**
     * 沿主轴方向搜索亮度跳变点。
     * 灯条中心的亮度最高，向两端延伸时亮度逐渐下降，
     * 在灯条端点处出现明显的亮度跳变（从亮变暗）。
     *
     * @param direction 1 = 向上搜索（top），-1 = 向下搜索（bottom）
     * @return 检测到的端点坐标，如果未找到则返回 (-1, -1)
     */
    const float dx = axis.x * direction;
    const float dy = axis.y * direction;
    const float search_length = lightbar.length * (SEARCH_END - SEARCH_START);

    std::vector<cv::Point2f> candidates;

    // 横向采样多个候选线（在灯条宽度范围内均匀采样，增加鲁棒性）
    const int half_width = (lightbar.width - 2) / 2;
    for (int i_offset = -half_width; i_offset <= half_width; ++i_offset) {
      cv::Point2f start_point(
        centroid.x + lightbar.length * SEARCH_START * dx + i_offset,
        centroid.y + lightbar.length * SEARCH_START * dy);

      // 沿轴搜索亮度跳变点
      cv::Point2f corner = start_point;
      float max_diff = 0;
      bool found = false;

      for (float step = 0; step < search_length; ++step) {
        const cv::Point2f cur_point(start_point.x + dx * step, start_point.y + dy * step);

        // 边界检查
        if (
          cur_point.x < 0 || cur_point.x >= gray_img.cols || cur_point.y < 0 ||
          cur_point.y >= gray_img.rows) {
          break;
        }

        // 计算亮度差（当前像素与前一像素的灰度差）
        const auto prev_val = gray_img.at<uchar>(cv::Point2i(cur_point - cv::Point2f(dx, dy)));
        const auto cur_val = gray_img.at<uchar>(cv::Point2i(cur_point));
        const float diff = prev_val - cur_val;

        // 记录亮度差最大的位置（即跳变最剧烈的位置）
        if (diff > max_diff && prev_val > mean_val) {
          max_diff = diff;
          corner = cur_point - cv::Point2f(dx, dy);  // 跳变发生在上一位置
          found = true;
        }
      }

      if (found) {
        candidates.push_back(corner);
      }
    }

    // 返回所有候选点的均值（多次采样的平均，提高稳定性）
    return candidates.empty()
             ? cv::Point2f(-1, -1)
             : std::accumulate(candidates.begin(), candidates.end(), cv::Point2f(0, 0)) /
                 static_cast<float>(candidates.size());
  };

  // 并行检测顶部和底部
  lightbar.top = find_corner(1);
  lightbar.bottom = find_corner(-1);
}

}  // namespace auto_aim
