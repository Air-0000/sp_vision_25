/**
 * @file solver.cpp
 * @brief PnP 位姿解算器实现
 * @details 实现装甲板的 PnP 位姿解算、坐标系变换（相机→云台→世界）、yaw 角优化。
 *
 * 核心算法步骤：
 * 1. cv::solvePnP：输入 4 个 2D-3D 点对 → 输出旋转向量 rvec + 平移向量 tvec
 * 2. 坐标系变换：tvec（相机系）→ R_camera2gimbal * tvec + t_camera2gimbal（云台系）
 * 3. 姿态提取：旋转矩阵 → 欧拉角（yaw, pitch, roll）
 * 4. yaw 优化：在 ±70° 范围内搜索重投影误差最小的 yaw 值
 *
 * @note 装甲板 3D 模型尺寸常量定义在本文件顶部：
 *       LIGHTBAR_LENGTH = 56mm（灯条长度）
 *       BIG_ARMOR_WIDTH = 230mm（大装甲板宽度）
 *       SMALL_ARMOR_WIDTH = 135mm（小装甲板宽度）
 */

#include "solver.hpp"

#include <yaml-cpp/yaml.h>

#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{

/**
 * @name 装甲板 3D 模型常量
 * @note 单位：米（m）
 * @details 装甲板四顶点在物体坐标系（以装甲板中心为原点）下的坐标。
 *          物体坐标系定义：
 *          - 原点：装甲板几何中心
 *          - x 轴：垂直于装甲板平面指向前方（法线方向）
 *          - y 轴：沿装甲板宽度方向（水平）
 *          - z 轴：沿灯条方向（竖直向上）
 *
 *          坐标顺序：左上 → 右上 → 右下 → 左下（顺时针顺序）
 */
//@{
constexpr double LIGHTBAR_LENGTH = 56e-3;     // 灯条长度 = 0.056 米
constexpr double BIG_ARMOR_WIDTH = 230e-3;    // 大装甲板宽度 = 0.230 米
constexpr double SMALL_ARMOR_WIDTH = 135e-3;  // 小装甲板宽度 = 0.135 米

/// 大装甲板四顶点 3D 坐标（物体坐标系）
/// 顶点在 y 方向跨度 = 230mm，z 方向跨度 = 56mm
const std::vector<cv::Point3f> BIG_ARMOR_POINTS{
  {0, BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},    // 左上
  {0, -BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},   // 右上
  {0, -BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},  // 右下
  {0, BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};  // 左下

/// 小装甲板四顶点 3D 坐标（物体坐标系）
/// 顶点在 y 方向跨度 = 135mm，z 方向跨度 = 56mm
const std::vector<cv::Point3f> SMALL_ARMOR_POINTS{
  {0, SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},    // 左上
  {0, -SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},   // 右上
  {0, -SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},  // 右下
  {0, SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};  // 左下
//@}

// ========== 步骤 2-1：解算器初始化 ==========

Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  /**
   * 从 YAML 配置文件加载标定参数：
   *
   * R_gimbal2imubody (3×3)：
   *   云台坐标系 → IMU 坐标系的旋转矩阵。
   *   如果 IMU 的安装方向与云台坐标轴一致，应为单位矩阵。
   *   configs/standard3.yaml 中为 [1,0,0,0,1,0,0,0,1]
   *
   * R_camera2gimbal (3×3) 和 t_camera2gimbal (3×1)：
   *   手眼标定结果，描述相机坐标系相对于云台坐标系的位姿。
   *   通过张正友标定法 + 手眼标定（AX=XB）得到。
   *   参考 calibration/calibrate_handeye.cpp
   *
   * camera_matrix (3×3)：
   *   相机内参矩阵 [fx, 0, cx; 0, fy, cy; 0, 0, 1]
   *   fx, fy = 焦距（像素），cx, cy = 主点坐标（像素）
   *
   * distort_coeffs (5×1)：
   *   畸变系数 [k1, k2, p1, p2, k3]
   *   k1, k2, k3 = 径向畸变，p1, p2 = 切向畸变
   */
  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);
}

// ========== 步骤 1-5：外部接口 ==========

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

// ========== 步骤 1-6：IMU 姿态更新（每帧调用） ==========

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  /**
   * 根据 IMU 四元数更新云台→世界旋转矩阵。
   * 公式：R_gimbal2world = R_gimbal2imu^T * R_imu_abs * R_gimbal2imu
   *
   * 其中：
   * - q：IMU 输出的当前姿态四元数（相对于初始时刻的旋转）
   * - R_imubody2imuabs = q.toRotationMatrix()：IMU 本体系到世界系的旋转
   * - R_gimbal2imubody_：云台系到 IMU 系的固定旋转（安装角度偏移补偿）
   *
   * 为什么要左乘 R_gimbal2imubody^T 再右乘 R_gimbal2imubody？
   * 因为云台坐标系的定义可能和 IMU 坐标系不同，需要通过这个中间变换
   * 将 IMU 测到的旋转正确投影到云台坐标系上。
   */
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

// ========== 步骤 2-2：PnP 位姿解算 ==========

// solvePnP（获得姿态）
void Solver::solve(Armor & armor) const
{
  /**
   * ## PnP 解算完整步骤
   *
   * ### 输入
   * - object_points（3D 模型点，物体坐标系）
   * - armor.points（2D 图像角点，像素坐标）
   * - camera_matrix_（相机内参）
   * - distort_coeffs_（畸变系数）
   *
   * ### 输出
   * - rvec：旋转向量（从物体系到相机系）
   * - tvec：平移向量（物体原点在相机系下的位置）
   *
   * ### 坐标变换链
   * 物体系 → PnP → 相机系 → 手眼标定 → 云台系 → IMU → 世界系
   *
   * 具体公式：
   *   xyz_in_gimbal = R_camera2gimbal * xyz_in_camera + t_camera2gimbal
   *   xyz_in_world  = R_gimbal2world * xyz_in_gimbal
   *   R_armor2gimbal = R_camera2gimbal * R_armor2camera
   *   R_armor2world  = R_gimbal2world * R_armor2gimbal
   */

  // 根据装甲板类型选择对应的 3D 模型
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  cv::Vec3d rvec, tvec;

  // 使用 IPPE（Infinitesimal Plane-Based Pose Estimation）算法求解 PnP。
  // IPPE 专门针对共面点（装甲板四顶点在同一平面内）的情况做了优化，
  // 比传统的 PnP 算法（如 EPnP、P3P）更快速稳定。
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  // ---------- 平移向量变换 ----------
  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  // ---------- 旋转矩阵变换（提取欧拉角） ----------
  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);  // 旋转向量 → 旋转矩阵（3×3）
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;

  // 提取欧拉角，旋转顺序：先 Z（yaw）再 Y（pitch）再 X（roll）
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  // 球坐标转换（用于弹道解算中的距离计算）
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  // ---------- 平衡步兵不进行 yaw 优化 ----------
  // 平衡步兵（英雄的 3/4/5 号大装甲板处于倾斜安装状态）的 pitch 假设不成立，
  // yaw 优化会引入较大误差，因此跳过。
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);
  if (is_balance) return;

  optimize_yaw(armor);
}

// ========== 步骤 2-3：装甲板重投影 ==========

std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
{
  /**
   * 将装甲板 3D 模型按照给定的世界坐标和 yaw 角重投影到图像平面。
   * 流程：
   * 1. 根据 yaw 和 pitch 构造装甲板→世界旋转矩阵
   * 2. 变换到相机坐标系（R_armor2camera, t_armor2camera）
   * 3. 使用 cv::projectPoints 投影到图像
   *
   * @param[in] xyz_in_world 装甲板中心在世界坐标系下的位置（米）
   * @param[in] yaw 要测试的 yaw 角（弧度）
   * @param[in] type 大/小装甲板
   * @param[in] name 装甲板名称（前哨站用特殊 pitch 角）
   * @return 四个角点的像素坐标
   */
  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  // 前哨站的装甲板并非垂直安装，有一定的俯仰角度（约 -15°）
  auto pitch = (name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // clang-format off
  // 装甲板→世界旋转矩阵（假设无 roll，仅 yaw + pitch）
  const Eigen::Matrix3d R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  // 计算装甲板→相机旋转和平移
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  Eigen::Matrix3d R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_armor2world;
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

  // 转到 OpenCV 格式
  cv::Vec3d rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, rvec);
  cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  // 重投影
  std::vector<cv::Point2f> image_points;
  const auto & object_points = (type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}

// ========== 步骤 2-4：前哨站重投影误差 ==========

double Solver::oupost_reprojection_error(Armor armor, const double & pitch)
{
  /**
   * 前哨站特殊处理：先做 PnP 得到初始位姿，再用给定的 pitch 角做重投影误差计算。
   * 前哨站的装甲板安装角度比较特殊，使用通用的 yaw 优化可能不准确，
   * 因此提供了专门的误差计算方法。
   */
  // solve
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  auto yaw = armor.ypr_in_world[0];
  auto xyz_in_world = armor.xyz_in_world;

  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // clang-format off
  const Eigen::Matrix3d _R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  // get R_armor2camera t_armor2camera
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  Eigen::Matrix3d _R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * _R_armor2world;
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

  // get rvec tvec
  cv::Vec3d _rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(_R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, _rvec);
  cv::Vec3d _tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  // reproject
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, _rvec, _tvec, camera_matrix_, distort_coeffs_, image_points);

  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  return error;
}

// ========== 步骤 2-5：yaw 角优化 ==========

void Solver::optimize_yaw(Armor & armor) const
{
  /**
   * ## yaw 角优化原理
   *
   * ### 为什么要优化 yaw？
   * PnP 解算出的 yaw 角存在较大的不确定性，因为装甲板四个角点近似在一个平面上，
   * 平面物体绕法线旋转（yaw）时，图像投影变化很小，导致 PnP 对 yaw 不敏感。
   *
   * ### 优化方法
   * 在 ±70° 范围内，以 1° 为步长遍历所有可能的 yaw 值，
   * 对每个值计算重投影误差（投影角点与检测角点的像素距离），
   * 选择误差最小的 yaw 角作为最终结果。
   *
   * ### 为什么使用重投影误差作为评判标准？
   * 重投影误差直接反映了位姿解算结果与图像观测的一致性。
   * 误差越小，说明解算的位姿越符合实际观测。
   *
   * @note 搜索范围 ±70°（140° 总范围）可能偏大，但对于初次检测或装甲板切换时的
   *       大角度跳变是必要的。对已跟踪的目标，可用 EKF 的预测值缩小搜索范围。
   */

  // 获取当前云台在世界坐标系下的 yaw 角（用于确定搜索中心）
  Eigen::Vector3d gimbal_ypr = tools::eulers(R_gimbal2world_, 2, 1, 0);

  constexpr double SEARCH_RANGE = 140;  // 搜索范围：±70°，单位：度
  auto yaw0 = tools::limit_rad(gimbal_ypr[0] - SEARCH_RANGE / 2 * CV_PI / 180.0);

  auto min_error = 1e10;
  auto best_yaw = armor.ypr_in_world[0];

  for (int i = 0; i < SEARCH_RANGE; i++) {
    double yaw = tools::limit_rad(yaw0 + i * CV_PI / 180.0);
    auto error = armor_reprojection_error(armor, yaw, (i - SEARCH_RANGE / 2) * CV_PI / 180.0);

    if (error < min_error) {
      min_error = error;
      best_yaw = yaw;
    }
  }

  armor.yaw_raw = armor.ypr_in_world[0];     // 保存原始 yaw（未优化前）
  armor.ypr_in_world[0] = best_yaw;           // 更新为优化后的 yaw
}

// ========== 步骤 2-6：SJTU 代价函数 ==========

double Solver::SJTU_cost(
  const std::vector<cv::Point2f> & cv_refs, const std::vector<cv::Point2f> & cv_pts,
  const double & inclined) const
{
  /**
   * 参考上海交通大学开源自瞄实现的代价函数。
   *
   * 核心思想：重投影误差应该综合考虑：
   * 1. 像素位置偏移误差（平移误差）
   * 2. 边长比例误差（缩放误差）
   * 3. 角度误差（旋转误差）
   *
   * 根据装甲板的倾斜角度（inclined），动态调整这三者的权重：
   * - 正对（inclined ≈ 0）：角度误差权重更大
   * - 侧对（inclined 大）：像素偏移和边长误差权重更大
   *
   * @note 当前实现中 armor_reprojection_error 直接用了简单的四角点距离和，
   *       SJTU_cost 通过注释保留但未启用。
   */
  std::size_t size = cv_refs.size();
  std::vector<Eigen::Vector2d> refs;
  std::vector<Eigen::Vector2d> pts;
  for (std::size_t i = 0u; i < size; ++i) {
    refs.emplace_back(cv_refs[i].x, cv_refs[i].y);
    pts.emplace_back(cv_pts[i].x, cv_pts[i].y);
  }
  double cost = 0.;
  for (std::size_t i = 0u; i < size; ++i) {
    std::size_t p = (i + 1u) % size;
    // i - p 构成线段。
    Eigen::Vector2d ref_d = refs[p] - refs[i];
    Eigen::Vector2d pt_d = pts[p] - pts[i];

    // 长度差代价 + 起点差代价
    double pixel_dis =
      (0.5 * ((refs[i] - pts[i]).norm() + (refs[p] - pts[p]).norm()) +
       std::fabs(ref_d.norm() - pt_d.norm())) /
      ref_d.norm();
    double angular_dis = ref_d.norm() * tools::get_abs_angle(ref_d, pt_d) / ref_d.norm();

    // 根据装甲板倾斜角度动态调整权重
    double cost_i =
      tools::square(pixel_dis * std::sin(inclined)) +
      tools::square(angular_dis * std::cos(inclined)) * 2.0;

    cost += std::sqrt(cost_i);
  }
  return cost;
}

// ========== 步骤 2-7：重投影误差计算 ==========

double Solver::armor_reprojection_error(
  const Armor & armor, double yaw, const double & inclined) const
{
  /**
   * 计算给定 yaw 角下的重投影误差。
   * 误差 = 四个角点的检测位置与重投影位置之间的像素距离总和。
   *
   * @param[in] armor 装甲板（包含检测到的角点 armor.points）
   * @param[in] yaw 要测试的 yaw 角（弧度）
   * @param[in] inclined 倾斜角度（弧度，当前未使用，仅做接口兼容）
   * @return 总重投影误差（像素）
   */
  auto image_points = reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  return error;
}

// ========== 步骤 2-8：世界坐标 → 像素坐标 ==========

// 世界坐标到像素坐标的转换
std::vector<cv::Point2f> Solver::world2pixel(const std::vector<cv::Point3f> & worldPoints)
{
  /**
   * 将世界坐标系下的 3D 点投影到图像像素坐标。
   * 用于全向感知模块中的可视化，或调试时验证坐标系变换是否正确。
   *
   * 变换公式：
   *   P_camera = R_world2camera * P_world + t_world2camera
   *   其中 R_world2camera = R_camera2gimbal^T * R_gimbal2world^T
   *       t_world2camera = -R_camera2gimbal^T * t_camera2gimbal
   *
   * @note 只投影 z > 0（在相机前方）的点，z ≤ 0 的点说明在相机后方，不投影。
   */
  Eigen::Matrix3d R_world2camera = R_camera2gimbal_.transpose() * R_gimbal2world_.transpose();
  Eigen::Vector3d t_world2camera = -R_camera2gimbal_.transpose() * t_camera2gimbal_;

  cv::Mat rvec;
  cv::Mat tvec;
  cv::eigen2cv(R_world2camera, rvec);
  cv::eigen2cv(t_world2camera, tvec);

  std::vector<cv::Point3f> valid_world_points;
  for (const auto & world_point : worldPoints) {
    Eigen::Vector3d world_point_eigen(world_point.x, world_point.y, world_point.z);
    Eigen::Vector3d camera_point = R_world2camera * world_point_eigen + t_world2camera;

    // 只保留相机前方的点（z > 0）
    if (camera_point.z() > 0) {
      valid_world_points.push_back(world_point);
    }
  }

  if (valid_world_points.empty()) {
    return std::vector<cv::Point2f>();
  }
  std::vector<cv::Point2f> pixelPoints;
  cv::projectPoints(valid_world_points, rvec, tvec, camera_matrix_, distort_coeffs_, pixelPoints);
  return pixelPoints;
}
}  // namespace auto_aim
