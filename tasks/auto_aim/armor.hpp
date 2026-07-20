/**
 * @file armor.hpp
 * @brief 装甲板（Armor）及灯条（Lightbar）数据结构定义
 * @details 本文件属于 auto_aim 模块的基础数据层，定义了装甲板检测管线中使用的所有核心数据类型：
 *          枚举（颜色、类型、名称、优先级），以及 Lightbar 和 Armor 两个结构体。
 *          检测器（Detector/YOLO）输出 std::list<Armor>，后续的 PnP 解算、EKF 跟踪、瞄准决策
 *          都直接依赖此数据结构。
 *
 * 管线位置：步骤 1（检测输出）→ 步骤 2（PnP 解算）→ 步骤 3（EKF 跟踪）
 * 依赖的外部文件：无（纯数据类型定义）
 * 调用时序：Detector::detect() 返回 std::list<Armor> → Solver::solve() 填充 3D 位姿
 *           → Tracker 读取 armor 信息进行 EKF 更新
 */

#ifndef AUTO_AIM__ARMOR_HPP
#define AUTO_AIM__ARMOR_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace auto_aim
{

/**
 * @brief 装甲板颜色枚举
 * @details 对应 RoboMaster 比赛中装甲板的 LED 颜色。
 *          red = 红色（通常为我方/敌方红方），blue = 蓝色，extinguish = 熄灭状态，
 *          purple = 紫色（基地特有，基地的大/小装甲板灯条为紫色）。
 * @note 颜色由 YOLO 输出的 color_id(0~3) 或传统方法的灯条 RGB 阈值判断得到。
 */
enum Color
{
  red,         ///< 红色
  blue,        ///< 蓝色
  extinguish,  ///< 熄灭（未亮灯，近距离可见灯条轮廓但无颜色）
  purple       ///< 紫色（仅基地装甲板，含 base_big 和 base_small）
};

/// @brief 颜色枚举对应的字符串描述，用于日志输出和调试显示
const std::vector<std::string> COLORS = {"red", "blue", "extinguish", "purple"};

/**
 * @brief 装甲板类型枚举
 * @details RoboMaster 2025 赛季规则：装甲板分为大装甲板（big）和小装甲板（small）。
 *          大装甲板尺寸约 230×130mm（宽×高），小装甲板约 130×50mm。
 *          大小装甲板的 3D 模型不同，PnP 解算时须使用对应的 3D 顶点坐标。
 * @note 英雄/步兵/哨兵等兵种默认安装小装甲板；基地装甲板为大装甲板。
 */
enum ArmorType
{
  big,   ///< 大装甲板（230×130mm）
  small  ///< 小装甲板（130×50mm）
};

/// @brief 装甲板类型对应的字符串描述
const std::vector<std::string> ARMOR_TYPES = {"big", "small"};

/**
 * @brief 装甲板数字编号枚举
 * @details 对应 RoboMaster 装甲板上的数字图案（1 ~ 5），以及特殊兵种/建筑：
 *          one~five = 步兵/英雄的 1~5 号装甲板；sentry = 哨兵；outpost = 前哨站；base = 基地。
 *          not_armor = 检测到灯条组合但非有效装甲板（误检）。
 * @note 数字编号用于识别目标身份和决定击打优先级（例如基地优先级通常最高）。
 */
enum ArmorName
{
  one,       ///< 编号 1
  two,       ///< 编号 2
  three,     ///< 编号 3
  four,      ///< 编号 4
  five,      ///< 编号 5
  sentry,    ///< 哨兵（无数字，有特殊标记）
  outpost,   ///< 前哨站（基地旁的固定建筑）
  base,      ///< 基地（大建筑）
  not_armor  ///< 不是有效装甲板（过滤掉）
};

/// @brief 装甲板名称枚举对应的字符串描述
const std::vector<std::string> ARMOR_NAMES = {"one",    "two",     "three", "four",     "five",
                                              "sentry", "outpost", "base",  "not_armor"};

/**
 * @brief 装甲板击打优先级枚举
 * @details 当检测到多个装甲板时，选择优先级最高的作为瞄准目标。
 *          数值越小优先级越高。first(1) > second(2) > ... > fifth(5)。
 */
enum ArmorPriority
{
  first = 1,   ///< 最高优先级（例如最近的装甲板或最正面的）
  second,       ///< 次高优先级
  third,        ///< 第三优先级
  forth,        ///< 第四优先级
  fifth         ///< 第五优先级
};

// clang-format off

/**
 * @brief 装甲板属性表
 * @details 一个预定义的装甲板属性查找表，包含所有合法的 (Color, ArmorName, ArmorType) 三元组。
 *          用于验证 YOLO / 传统方法检测结果是否合理。例如：
 *          - blue + one + small = 合法的蓝色 1 号小装甲板
 *          - purple + base + big = 合法的紫色基地大装甲板
 *          - 不在此表中的组合（如 blue + base + small）将被过滤掉。
 * @note sentry（哨兵）只有小装甲板；基地（base）有大小两种但颜色为 purple。
 *       hero（英雄）3/4/5 号位有大装甲板配置（three_big / four_big / five_big）。
 */
const std::vector<std::tuple<Color, ArmorName, ArmorType>> armor_properties = {
  {blue, sentry, small},     {red, sentry, small},     {extinguish, sentry, small},
  {blue, one, small},        {red, one, small},        {extinguish, one, small},
  {blue, two, small},        {red, two, small},        {extinguish, two, small},
  {blue, three, small},      {red, three, small},      {extinguish, three, small},
  {blue, four, small},       {red, four, small},       {extinguish, four, small},
  {blue, five, small},       {red, five, small},       {extinguish, five, small},
  {blue, outpost, small},    {red, outpost, small},    {extinguish, outpost, small},
  {blue, base, big},         {red, base, big},         {extinguish, base, big},      {purple, base, big},
  {blue, base, small},       {red, base, small},       {extinguish, base, small},    {purple, base, small},
  {blue, three, big},        {red, three, big},        {extinguish, three, big},
  {blue, four, big},         {red, four, big},         {extinguish, four, big},
  {blue, five, big},         {red, five, big},         {extinguish, five, big}};

// clang-format on

/**
 * @brief 灯条结构体
 * @details 灯条（Lightbar）是装甲板两侧的 LED 发光条，是检测装甲板的基础特征。
 *          传统方法通过颜色阈值提取灯条轮廓，然后根据灯条的几何约束筛选可能的灯条对。
 *          每个有效的装甲板有左（left）右（right）两个灯条。
 *
 *          PCA 角点修正：检测到的灯条轮廓角点通常不够精确，通过 PCA（主成分分析）
 *          对灯条区域内的灰度图进行主方向回归，得到更精确的灯条矩形四角。
 *          参考自 CSU-FYT-Vision/FYT2024_vision。
 */
struct Lightbar
{
  std::size_t id;                    ///< 灯条唯一标识 ID，用于与装甲板配对
  Color color;                       ///< 灯条颜色，由 get_color() 判断得到

  /**
   * @name 灯条几何属性
   * @note 单位：像素（pixel），除非特别注明
   */
  //@{
  cv::Point2f center;                ///< 灯条矩形中心点坐标（像素）
  cv::Point2f top;                   ///< 灯条矩形上端点坐标（像素）
  cv::Point2f bottom;                ///< 灯条矩形下端点坐标（像素）
  cv::Point2f top2bottom;            ///< 从 top 指向 bottom 的向量（用于判断灯条方向）
  std::vector<cv::Point2f> points;   ///< 灯条四个角点（经 PCA 修正后），按顺时针顺序
  double angle;                      ///< 灯条长轴与图像水平轴的夹角，单位：弧度 [-π/2, π/2]
  double angle_error;                ///< 灯条角度误差（用于灯条配对时的几何约束）
  double length;                     ///< 灯条长度（沿长轴方向），单位：像素
  double width;                      ///< 灯条宽度（沿短轴方向），单位：像素
  double ratio;                      ///< 灯条长宽比 length/width，用于筛选合法灯条（通常 >1.5）
  cv::RotatedRect rotated_rect;      ///< OpenCV 最小外接旋转矩形（包含角度信息）
  //@}

  /**
   * @brief 构造函数：从 cv::RotatedRect 构造 Lightbar
   * @param[in] rotated_rect OpenCV 的最小外接矩形（由 findContours + minAreaRect 得到）
   * @param[in] id 灯条唯一编号
   */
  Lightbar(const cv::RotatedRect & rotated_rect, std::size_t id);
  Lightbar() {};  // 默认构造函数，用于 stl 容器
};

/**
 * @brief 装甲板结构体
 * @details 装甲板是 RoboMaster 自瞄系统的最核心数据实体。
 *          一个 Armor 代表图像中的一个被检测到的装甲板，包含：
 *          1. 2D 检测信息（灯条匹配/YOLO 输出）：颜色、类型、置信度、2D 框、四个角点
 *          2. 3D 解算信息（PnP 结果）：在云台坐标系/世界坐标系下的位置和姿态
 *
 *          生命周期：
 *          1. Detector::detect() 或 YOLO::detect() 创建 -> 2. Solver::solve() 填充 3D 姿态
 *          3. Tracker 消费（匹配已有目标或创建新目标）-> 4. Aimer 使用（弹道解算用 3D 位置）
 */
struct Armor
{
  // ---- 检测结果（由 Detector / YOLO 填充） ----
  Color color;                       ///< 装甲板颜色（red/blue/extinguish/purple）
  Lightbar left, right;              ///< 左右两个灯条（传统方法检测时才有效，YOLO 模式下为空）
  cv::Point2f center;                ///< 装甲板中心在图像上的 2D 投影（像素坐标）
                                     ///< @warning 不是对角线交点！不能作为装甲板实际中心！
  cv::Point2f center_norm;           ///< 归一化中心坐标 (x/w, y/h)，范围 [0,1]
  std::vector<cv::Point2f> points;   ///< 装甲板四个角点（像素坐标），顺序：
                                     ///< [0]=左上, [1]=右上, [2]=右下, [3]=左下
                                     ///< @note 用于 PnP 解算的 2D 输入点

  // ---- 几何特征（传统方法用，YOLO 模式下也用于二次校验） ----
  double ratio;                      ///< 两灯条中点连线距离 / 长灯条长度（用于装甲板几何校验）
  double side_ratio;                 ///< 长灯条长度 / 短灯条长度（区分正对和侧对情况）
  double rectangular_error;          ///< 灯条方向与灯条中点连线之间的夹角与 90° 的偏差弧度
                                     ///< 值越小说明灯条越平行，装甲板姿态越正对

  // ---- 分类结果（由 Classifier / YOLO 填充） ----
  ArmorType type;                    ///< 大装甲板 or 小装甲板
  ArmorName name;                    ///< 装甲板数字编号（one ~ five / sentry / outpost / base）
  ArmorPriority priority;            ///< 击打优先级（根据距离/角度自动评估）
  int class_id;                      ///< 分类结果的整数 ID（映射到 ArmorName）
  cv::Rect box;                      ///< 装甲板在图像中的最小轴对齐外接矩形（X, Y, W, H），单位：像素
  cv::Mat pattern;                   ///< 装甲板数字图案区域（裁剪后的小图，送入 Classifier 识别数字）
  double confidence;                 ///< 检测/分类置信度，范围 [0, 1]，低于 min_confidence 将被过滤
  bool duplicated;                   ///< 是否为重复检测（同一个装甲板被多个检测框覆盖），用于 NMS

  // ---- 三维解算结果（由 Solver::solve() 填充） ----
  Eigen::Vector3d xyz_in_gimbal;     ///< 装甲板中心在云台坐标系下的位置，单位：米
                                     ///< 坐标系：云台 yaw 轴为原点，x 前，y 左，z 上
  Eigen::Vector3d xyz_in_world;      ///< 装甲板中心在世界坐标系下的位置，单位：米
                                     ///< 世界坐标系 = 云台坐标系经 IMU 四元数旋转后
  Eigen::Vector3d ypr_in_gimbal;     ///< 装甲板在云台坐标系下的欧拉角（yaw/pitch/roll），单位：弧度
  Eigen::Vector3d ypr_in_world;      ///< 装甲板在世界坐标系下的欧拉角（yaw/pitch/roll），单位：弧度
  Eigen::Vector3d ypd_in_world;      ///< 装甲板中心在世界坐标系下的球坐标
                                     ///< ypd[0] = yaw 角（弧度），ypd[1] = pitch 角（弧度），ypd[2] = 距离（米）

  double yaw_raw;                    ///< PnP 解算出的直接 yaw 角（未经任何偏移修正），单位：弧度

  /**
   * @brief 构造函数 1：从灯条对构造装甲板（传统方法用）
   * @param[in] left 左灯条
   * @param[in] right 右灯条
   * @details 通过左右灯条的几何约束计算装甲板中心和角点。
   *          这是传统检测方法的入口：检测到灯条 → 灯条配对 → 构造 Armor。
   */
  Armor(const Lightbar & left, const Lightbar & right);

  /**
   * @brief 构造函数 2：从 YOLO 四点检测结果构造（无 offset，原图坐标）
   * @param[in] class_id 类别 ID（映射到 ArmorName）
   * @param[in] confidence 检测置信度
   * @param[in] box 2D 检测框
   * @param[in] armor_keypoints 四个角点像素坐标（YOLO 输出的四点）
   */
  Armor(
    int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints);

  /**
   * @brief 构造函数 3：从 YOLO 四点检测结果构造（带 offset，ROI 坐标修正用）
   * @param[in] class_id 类别 ID
   * @param[in] confidence 检测置信度
   * @param[in] box 2D 检测框（ROI 内的坐标）
   * @param[in] armor_keypoints 四个角点像素坐标（ROI 内的坐标）
   * @param[in] offset ROI 左上角在原图中的偏移量，用于将 ROI 坐标还原为原图坐标
   */
  Armor(
    int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints,
    cv::Point2f offset);

  /**
   * @brief 构造函数 4：从 YOLO 分类结果构造（color + num 独立输出，无 offset）
   * @param[in] color_id 颜色 ID（0=red, 1=blue, 2=extinguish, 3=purple）
   * @param[in] num_id 数字 ID（0~8 对应 ArmorName 枚举）
   * @param[in] confidence 检测置信度
   * @param[in] box 2D 检测框
   * @param[in] armor_keypoints 四个角点像素坐标
   */
  Armor(
    int color_id, int num_id, float confidence, const cv::Rect & box,
    std::vector<cv::Point2f> armor_keypoints);

  /**
   * @brief 构造函数 5：从 YOLO 分类结果构造（color + num 独立输出，带 offset）
   * @param[in] color_id 颜色 ID
   * @param[in] num_id 数字 ID
   * @param[in] confidence 检测置信度
   * @param[in] box 2D 检测框（ROI 内的坐标）
   * @param[in] armor_keypoints 四个角点像素坐标（ROI 内的坐标）
   * @param[in] offset ROI 左上角偏移量
   */
  Armor(
    int color_id, int num_id, float confidence, const cv::Rect & box,
    std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__ARMOR_HPP
