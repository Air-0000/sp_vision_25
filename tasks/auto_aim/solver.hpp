/**
 * @file solver.hpp
 * @brief PnP 位姿解算器（装甲板 2D → 3D）
 * @details 本文件属于 auto_aim 模块的解算子模块，负责将检测到的装甲板四个角点像素坐标，
 *          通过 PnP（Perspective-n-Point）算法解算其在三维空间中的位置和姿态。
 *
 * 管线位置：步骤 2（PnP 解算）：YOLO/Detector 输出 std::list<Armor>（含 2D 角点）
 *          → Solver::solve() 填充 Armor 的 3D 位姿（xyz_in_gimbal, ypr_in_world 等）
 *          → Tracker 消费 3D 信息进行 EKF 更新
 *
 * 核心算法：
 * 1. 使用 cv::solvePnP 求解装甲板相对于相机的位姿
 * 2. 结合 IMU 四元数将相机坐标系结果转换到云台坐标系（gimbal frame）和世界坐标系（world frame）
 * 3. yaw 优化（optimize_yaw）：通过重投影误差最小化来微调 yaw 角，补偿 PnP 在 yaw 方向的不确定性
 *
 * 坐标系定义：
 * - 相机坐标系：原点在相机光心，z 轴沿光轴指向前方，x 轴向右，y 轴向下
 * - 云台坐标系：原点在云台 yaw 轴中心，x 轴指向前方，y 轴向左，z 轴向上
 * - 世界坐标系：云台坐标系经 IMU 四元数旋转后的结果（用于消除云台自身旋转的影响）
 *
 * @note PnP 解算的精度直接影响后续 EKF 跟踪和弹道解算的准确性。
 *       装甲板 3D 模型坐标（物体坐标系）定义在 configs/*.yaml 中。
 */

#ifndef AUTO_AIM__SOLVER_HPP
#define AUTO_AIM__SOLVER_HPP

#include <Eigen/Dense>  // 必须在 OpenCV eigen.hpp 之前包含
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include "armor.hpp"

namespace auto_aim
{

/**
 * @brief PnP 位姿解算器
 * @details 将装甲板的四个图像角点通过 PnP 解算为三维位姿。
 *          主要职责：
 *          1. 从配置文件加载相机内参、畸变系数、手眼标定结果
 *          2. 每次调用 solve() 时，先执行 PnP 解算，再进行 yaw 优化
 *          3. 根据 IMU 四元数更新 R_gimbal2world（将结果转到世界坐标系）
 *
 * 生命周期：
 * - 构造函数：加载标定参数（相机内参、手眼矩阵）
 * - set_R_gimbal2world()：每帧调用（传入当前 IMU 四元数）
 * - solve()：每帧对每个检测到的装甲板调用
 *
 * @note 非线程安全。当前帧内所有 solve() 调用应使用同一个 R_gimbal2world。
 */
class Solver
{
public:
  /**
   * @brief 构造函数：加载标定参数
   * @param[in] config_path YAML 配置文件路径，需包含：
   *            - camera_matrix（相机内参矩阵 3×3）
   *            - distort_coeffs（畸变系数 5×1）
   *            - R_camera2gimbal（手眼标定旋转矩阵 3×3）
   *            - t_camera2gimbal（手眼标定平移向量 3×1）
   *            - R_gimbal2imubody（IMU 安装角度 3×3）
   * @throws 如果配置文件缺少必要参数
   */
  explicit Solver(const std::string & config_path);

  /**
   * @brief 获取当前帧的云台→世界旋转变换矩阵
   * @return 3×3 旋转矩阵
   */
  Eigen::Matrix3d R_gimbal2world() const;

  /**
   * @brief 更新云台→世界旋转矩阵（每帧调用一次）
   * @param[in] q IMU 测得的云台姿态四元数（由 CBoard 提供）
   * @details R_gimbal2world = R_imu2world * R_gimbal2imu
   *          其中 R_imu2world 由 IMU 四元数 q 转换得到，
   *          R_gimbal2imu 是云台坐标系到 IMU 坐标系的固定旋转（出厂标定值）。
   */
  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  /**
   * @brief PnP 位姿解算（管线段步骤 2）
   * @param[in/out] armor 装甲板。输入时需包含 camera 坐标系下的 2D 角点（points），
   *                       输出时填充 xyz_in_gimbal, xyz_in_world, ypr_in_gimbal, ypr_in_world, yaw_raw
   * @details 执行步骤：
   *          1. 调用 cv::solvePnP(objectPoints, armor.points, cameraMatrix, distCoeffs, rvec, tvec)
   *             - objectPoints：装甲板四顶点在物体坐标系（装甲板中心为原点）的 3D 坐标
   *             取决于 type（big/small）和 name（决定哪一组 3D 尺寸）
   *          2. 将旋转向量 rvec 转为旋转矩阵
   *          3. 计算装甲板在相机坐标系下的位置和姿态
   *          4. 通过手眼标定矩阵转到云台坐标系（gimbal frame）
   *          5. 通过 IMU 四元数转到世界坐标系（world frame）
   *          6. 调用 optimize_yaw() 进行 yaw 角的微调优化
   *
   * @note 重投影误差隐含在 optimize_yaw 中作为优化目标。
   */
  void solve(Armor & armor) const;

  /**
   * @brief 重投影：将装甲板 3D 模型投影回图像平面
   * @param[in] xyz_in_world 装甲板中心在世界坐标系下的位置（米）
   * @param[in] yaw 装甲板的 yaw 角（弧度）
   * @param[in] type 装甲板类型（决定 3D 模型尺寸）
   * @param[in] name 装甲板名称（决定使用哪一组 3D 尺寸）
   * @return 投影到图像上的四个角点（像素坐标）
   * @details 用于前哨站的特殊重投影误差计算。
   */
  std::vector<cv::Point2f> reproject_armor(
    const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const;

  /**
   * @brief 前哨站重投影误差计算
   * @param[in] armor 检测到的装甲板
   * @param[in] pitch pitch 角
   * @return 重投影误差值
   */
  double oupost_reprojection_error(Armor armor, const double & picth);

  /**
   * @brief 世界坐标系 3D 点投影到像素坐标
   * @param[in] worldPoints 世界坐标系下的三维点列表
   * @return 对应的像素坐标（二维点列表）
   */
  std::vector<cv::Point2f> world2pixel(const std::vector<cv::Point3f> & worldPoints);

private:
  // ---------- 相机标定参数 ----------
  cv::Mat camera_matrix_;      ///< 相机内参矩阵 3×3（fx, fy, cx, cy）
  cv::Mat distort_coeffs_;     ///< 畸变系数 [k1, k2, p1, p2, k3]

  // ---------- 手眼标定参数 ----------
  Eigen::Matrix3d R_gimbal2imubody_;  ///< 云台坐标系到 IMU 坐标系的旋转矩阵（固定值）
  Eigen::Matrix3d R_camera2gimbal_;   ///< 相机坐标系到云台坐标系的旋转矩阵（手眼标定）
  Eigen::Vector3d t_camera2gimbal_;   ///< 相机坐标系到云台坐标系的平移向量（手眼标定），单位：米

  // ---------- 运行时状态 ----------
  Eigen::Matrix3d R_gimbal2world_;    ///< 云台坐标系到世界坐标系的旋转矩阵（每帧更新）

  /**
   * @brief yaw 角优化
   * @details PnP 解算出的 yaw 角通常不够精确，因为装甲板近似平面，yaw 方向信息弱。
   *          优化方法：在 yaw 角附近小范围内搜索，找到使重投影误差最小的 yaw 值。
   * @param[in/out] armor 待优化的装甲板（yaw_raw 被更新）
   */
  void optimize_yaw(Armor & armor) const;

  /**
   * @brief 计算装甲板在当前 yaw 和倾斜角度下的重投影误差
   * @param[in] armor 装甲板
   * @param[in] yaw yaw 角（弧度）
   * @param[in] inclined 倾斜角度（弧度），用于前哨站的倾角补偿
   * @return 重投影误差（像素）
   */
  double armor_reprojection_error(const Armor & armor, double yaw, const double & inclined) const;

  /**
   * @brief SJTU 风格的代价函数（参考上海交通大学的开源实现）
   * @param[in] cv_refs 参考点（检测到的角点）
   * @param[in] cv_pts 重投影点
   * @param[in] inclined 倾斜角度
   * @return 代价函数值
   */
  double SJTU_cost(
    const std::vector<cv::Point2f> & cv_refs, const std::vector<cv::Point2f> & cv_pts,
    const double & inclined) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__SOLVER_HPP
