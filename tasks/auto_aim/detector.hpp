/**
 * @file detector.hpp
 * @brief 传统视觉方法装甲板检测器接口
 * @details 本文件属于 auto_aim 模块的检测子模块，定义了基于传统图像处理的装甲板检测器。
 *          当 YOLO 检测失败或置信度不足时，使用传统方法作为补充或备选。
 *          传统方法流程：二值化 → 灯条轮廓提取 → 灯条几何筛选 → 灯条配对 → 装甲板构造。
 *
 * 管线位置：步骤 1（检测），可独立使用或作为 YOLO 检测的后处理补充
 *           YOLO 模式下 detect(Armor&, Mat) 用于角点二次矫正（PCA 修正）
 * 依赖：classifier.hpp（用于识别数字图案）
 *
 * @note 传统方法参数量大（阈值、角度容差、灯条长宽比等），调试依赖经验值。
 *       自 sp_vision_24 开始，团队重心已转向 YOLO 检测，传统方法作为降级方案保留。
 */

#ifndef AUTO_AIM__DETECTOR_HPP
#define AUTO_AIM__DETECTOR_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "armor.hpp"
#include "classifier.hpp"

namespace auto_aim
{

/**
 * @brief 传统视觉装甲板检测器
 * @details 实现基于灯条（Lightbar）的装甲板检测算法。
 *          主要步骤：
 *          1. 二值化（基于颜色通道分离 + 阈值）提取灯条区域
 *          2. findContours 寻找轮廓 → minAreaRect 获取最小外接矩形
 *          3. 对每个外接矩形进行灯条几何校验（长宽比、长度、角度）
 *          4. 灯条配对：角度差、长度比、中心距等约束
 *          5. 每对合法灯条构造一个 Armor
 *          6. 对每个 Armor 裁剪数字区域 → 送入 Classifier 识别数字编号
 *
 * @note 也提供单装甲板的 detect(Armor&, cv::Mat) 重载，用于 YOLO 输出后：
 *       - 用传统方法二次确认装甲板合法性
 *       - PCA 修正四个角点位置（提高 PnP 精度）
 */
class Detector
{
public:
  /**
   * @brief 构造函数：加载检测参数
   * @param[in] config_path YAML 配置文件路径
   * @param[in] debug 是否开启调试可视化（显示二值图、检测结果等）
   */
  Detector(const std::string & config_path, bool debug = true);

  /**
   * @brief 完整检测管线（步骤 1：从图像中检测所有装甲板）
   * @details 对输入 BGR 图像执行完整的传统方法检测流程：
   *          二值化 → 灯条提取 → 灯条配对 → 装甲板构造 → 数字分类
   * @param[in] bgr_img 输入的 BGR 三通道图像
   * @param[in] frame_count 帧序号（仅用于可视化窗口标题）
   * @return 检测到的合法装甲板列表（可能为空）
   */
  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count = -1);

  /**
   * @brief 单装甲板检测与角点修正（步骤 1-补充）
   * @details 对 YOLO 检测到的单个装甲板，使用传统灯条检测方法：
   *          1. 验证装甲板合法性（检查灯条几何）
   *          2. 使用 PCA 回归修正四个角点位置
   *          3. 填充装甲板的灯条信息
   * @param[in/out] armor 待修正的装甲板（函数返回后 points 可能被更新）
   * @param[in] bgr_img 原始 BGR 图像
   * @return true = 通过传统验证，false = 无效装甲板
   */
  bool detect(Armor & armor, const cv::Mat & bgr_img);

  /// 允许 YOLO 子类访问检测器内部（用于传统方法角点矫正）
  friend class YOLOV8;

private:
  // ---------- 子模块 ----------
  Classifier classifier_;  ///< 数字图案分类器（传统方法需要它来识别装甲板编号）

  // ---------- 阈值参数（从 YAML 读取） ----------
  double threshold_;              ///< 二值化阈值（0~255），越大越暗的区域会被二值化为背景
  double max_angle_error_;        ///< 灯条配对时允许的最大角度差，单位：度
  double min_lightbar_ratio_,     ///< 灯条最小长宽比（过小的比例说明不是灯条而是方形噪点）
         max_lightbar_ratio_;     ///< 灯条最大长宽比（过大的比例说明是细线状干扰）
  double min_lightbar_length_;    ///< 灯条最小像素长度（太短的灯条可能是噪声）
  double min_armor_ratio_,        ///< 装甲板最小宽高比（左右灯条间距 / 灯条长度）
         max_armor_ratio_;        ///< 装甲板最大宽高比
  double max_side_ratio_;         ///< 左右灯条长度比的最大值（接近 1 说明两个灯条等长，更可能是一对）
  double min_confidence_;         ///< 分类器最小置信度（低于此值的装甲板被丢弃）
  double max_rectangular_error_;  ///< 灯条与连线夹角的矩形误差上限，单位：度

  // ---------- 调试与保存 ----------
  bool debug_;                    ///< 是否显示调试窗口
  std::string save_path_;         ///< 异常装甲板图像的保存路径

  // ---------- 核心算法方法 ----------

  /**
   * @brief PCA 回归角点修正
   * @details 对灯条区域内的像素进行主成分分析，求出主方向。
   *          然后将灯条四个角点沿主方向对齐，消除因轮廓提取导致的角点偏移。
   * @param[in/out] lightbar 待修正的灯条
   * @param[in] gray_img 灰度图（用于 PCA 分析）
   */
  void lightbar_points_corrector(Lightbar & lightbar, const cv::Mat & gray_img) const;

  /**
   * @brief 检查单个灯条的几何合法性
   * @param[in] lightbar 待检查的灯条
   * @return true = 合法，false = 不合法的灯条（如长宽比异常）
   */
  bool check_geometry(const Lightbar & lightbar) const;

  /**
   * @brief 检查装甲板的几何合法性（左右灯条配对约束）
   * @param[in] armor 待检查的装甲板
   * @return true = 合法装甲板，false = 虚假目标
   */
  bool check_geometry(const Armor & armor) const;

  /**
   * @brief 检查装甲板名称是否在合法列表中
   * @param[in] armor 待检查的装甲板
   * @return true = 名称合法
   */
  bool check_name(const Armor & armor) const;

  /**
   * @brief 检查装甲板类型与名称的匹配关系
   * @details 某些装甲板名称不允许特定类型（如 one 号不允许 big，two 号不允许 big 等）
   * @param[in] armor 待检查的装甲板
   * @return true = 类型-名称匹配
   */
  bool check_type(const Armor & armor) const;

  /**
   * @brief 判断灯条颜色（根据 BGR 通道阈值）
   * @param[in] bgr_img 原始图像
   * @param[in] contour 灯条轮廓点集
   * @return 检测到的颜色枚举
   */
  Color get_color(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour) const;

  /**
   * @brief 获取装甲板数字区域的图像（裁剪 + 透视校正）
   * @param[in] bgr_img 原始 BGR 图像
   * @param[in] armor 装甲板（用其角点计算透视变换）
   * @return 校正后的数字区域灰度图（送入 Classifier）
   */
  cv::Mat get_pattern(const cv::Mat & bgr_img, const Armor & armor) const;

  /**
   * @brief 判断装甲板类型（big / small）
   * @param[in/out] armor 装甲板（函数返回后 armor.type 被赋值）
   * @return ArmorType 枚举
   */
  ArmorType get_type(const Armor & armor);

  /**
   * @brief 计算归一化中心坐标
   * @param[in] bgr_img 图像（取其尺寸进行归一化）
   * @param[in] center 装甲板中心像素坐标
   * @return 归一化坐标 (x/w, y/h)，范围 [0, 1]
   */
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  /**
   * @brief 保存装甲板图像到磁盘（用于后续数据集标注）
   * @param[in] armor 待保存的装甲板
   */
  void save(const Armor & armor) const;

  /**
   * @brief 调试可视化：绘制检测结果
   */
  void show_result(
    const cv::Mat & binary_img, const cv::Mat & bgr_img, const std::list<Lightbar> & lightbars,
    const std::list<Armor> & armors, int frame_count) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__DETECTOR_HPP
