/**
 * @file target.hpp
 * @brief EKF 目标状态估计器接口
 * @details 本文件属于 auto_aim 模块的跟踪子模块，定义了基于扩展卡尔曼滤波（EKF）的
 *          敌方机器人运动状态估计器 Target 类。
 *
 * 管线位置：步骤 3（EKF 跟踪）：Solver 输出 Armor（含 3D 位姿）→ Tracker 将装甲板与已有
 *          Target 匹配 → Target::update() 更新 EKF → Aimer 使用 Target 的预测状态进行瞄准决策
 *
 * 状态向量（11 维）：
 *   x = [x, vx, y, vy, z, vz, angle, omega, r, l, h]^T
 *     x, y, z       : 目标旋转中心在世界坐标系下的位置（米）
 *     vx, vy, vz    : 目标旋转中心的速度（米/秒）
 *     angle         : 目标当前旋转角度（弧度，即车头朝向）
 *     omega         : 目标旋转角速度（弧度/秒）
 *     r             : 装甲板到旋转中心的距离（米），即装甲板半径
 *     l             : 长轴装甲板的额外半径偏移（米），仅 4 装甲板车辆的长轴侧
 *     h             : 长轴装甲板的高度偏移（米），仅 4 装甲板车辆的长轴侧
 *
 * @note 估计器假设目标做匀速直线运动 + 恒定转速小陀螺（CTRV 模型的子集），
 *       使用 Piecewise White Noise Model 过程噪声。
 */

#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

/**
 * @brief EKF 目标状态估计器
 * @details 代表一个正在被跟踪的敌方机器人目标。每个 Target 对应一台敌方机器人，
 *          维护其旋转中心位置、速度、旋转角速度等 11 维状态。
 *
 * 生命周期：
 * 1. 创建：Tracker 首次检测到装甲板时构造 Target（传入首帧装甲板信息和初始协方差 P0）
 * 2. 预测：Tracker 每帧先调用 predict() 推进状态到当前时间
 * 3. 更新：Tracker 将当前帧检测到的装甲板与 Target 匹配 → update() 更新 EKF
 * 4. 丢失：超过 max_temp_lost_count 帧无匹配 → Tracker 销毁此 Target
 *
 * @note Target 内的 EKF 使用自定义加法（x_add）来处理角度（angle）的弧度绕环，
 *       确保 angle 始终在 (-π, π] 范围内。
 */
class Target
{
public:
  // ---- 公开状态（可直接读取） ----
  ArmorName name;              ///< 目标机器人名称（one~five / sentry / outpost / base）
  ArmorType armor_type;        ///< 装甲板类型（big / small）
  ArmorPriority priority;      ///< 击打优先级
  bool jumped;                 ///< 装甲板是否发生过跳变（旋转导致新的装甲板转到正面）
  int last_id;                 ///< 上一帧匹配到的装甲板编号（调试用）

  /**
   * @brief 构造函数 1：从首帧装甲板创建 Target
   * @param[in] armor 首帧检测到的装甲板
   * @param[in] t 当前时间戳
   * @param[in] radius 装甲板到旋转中心的初始距离（米），由机器人型号决定
   * @param[in] armor_num 装甲板数量（通常为 4 或 2）
   * @param[in] P0_dig 初始状态协方差对角线向量（11 维）
   */
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);

  /**
   * @brief 构造函数 2：从已知状态创建 Target（用于仿真/调试）
   * @param[in] x 旋转中心 x 坐标（米）
   * @param[in] vyaw 旋转角速度（弧度/秒）
   * @param[in] radius 装甲板半径（米）
   * @param[in] h 高度偏移（米）
   */
  Target(double x, double vyaw, double radius, double h);

  /**
   * @brief 按时间戳预测
   * @param[in] t 目标时间戳
   * @details 计算当前时间与记录时间的差 dt，然后调用 predict(dt)
   */
  void predict(std::chrono::steady_clock::time_point t);

  /**
   * @brief 按时间差预测（状态转移 + 过程噪声注入）
   * @param[in] dt 预测时间差（秒）
   * @details 执行 EKF 预测步骤：
   *          状态转移：x_{k|k-1} = F * x_{k-1}
   *          协方差传播：P_{k|k-1} = F * P_{k-1} * F^T + Q
   */
  void predict(double dt);

  /**
   * @brief EKF 更新步骤（匹配到装甲板后调用）
   * @param[in] armor 当前帧检测到的装甲板
   * @details 执行 EKF 更新：
   *          1. 计算装甲板 ID（匹配到 4 个虚拟装甲板中距离最近的一个）
   *          2. 计算观测残差 z - h(x)
   *          3. 计算卡尔曼增益 K
   *          4. 更新后验状态 x_{k|k} = x_{k|k-1} + K * (z - h(x))
   *          5. 更新后验协方差 P_{k|k}
   */
  void update(const Armor & armor);

  /**
   * @brief 获取 EKF 状态向量
   * @return 11 维状态向量
   */
  Eigen::VectorXd ekf_x() const;

  /**
   * @brief 获取 EKF 滤波器对象的常引用
   */
  const tools::ExtendedKalmanFilter & ekf() const;

  /**
   * @brief 计算所有虚拟装甲板在世界坐标系下的位置和朝向
   * @return vector<Eigen::Vector4d>，每个元素为 (x, y, z, angle)
   *         其中 (x,y,z) = 装甲板中心世界坐标（米），angle = 装甲板法线 yaw 角（弧度）
   */
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  /**
   * @brief 检查 EKF 是否发散
   * @details 根据装甲板半径 r 和偏移 l 是否在合理范围内判断。
   *          r 应在 [0.05, 0.5] 米内，r+l 也应在该范围内。
   * @return true = 发散，false = 未发散
   */
  bool diverged() const;

  /**
   * @brief 检查 EKF 是否已收敛
   * @details 满足条件时返回 true：
   *          - 非前哨站：update_count > 3 且未发散
   *          - 前哨站：update_count > 10 且未发散
   * @return true = 已收敛
   */
  bool convergened();

  bool isinit = false;

  bool checkinit();

private:
  // ---------- 机器人参数 ----------
  int armor_num_;      ///< 机器人装甲板数量（2 或 4）
  int switch_count_;   ///< 装甲板切换次数计数
  int update_count_;   ///< EKF 更新次数计数

  // ---------- 状态标记 ----------
  bool is_switch_,      ///< 当前帧是否发生了装甲板切换
       is_converged_;   ///< EKF 是否已收敛

  // ---------- EKF ----------
  tools::ExtendedKalmanFilter ekf_;  ///< 扩展卡尔曼滤波器实例
  std::chrono::steady_clock::time_point t_;  ///< 上次更新时间戳

  /**
   * @brief 更新 yaw/pitch/distance/angle 观测
   * @param[in] armor 当前帧匹配到的装甲板
   * @param[in] id 匹配到的虚拟装甲板 ID
   * @details 构建观测向量 z = [ypd_yaw, ypd_pitch, ypd_dist, armor_yaw]
   *          并调用 EKF 的 update() 方法
   */
  void update_ypda(const Armor & armor, int id);

  /**
   * @brief 计算第 id 号虚拟装甲板在世界坐标系下的 3D 位置
   * @param[in] x 当前状态向量
   * @param[in] id 虚拟装甲板 ID（0 ~ armor_num_-1）
   * @return 装甲板中心的世界坐标 (x, y, z)（米）
   */
  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;

  /**
   * @brief 计算观测函数 h 的雅可比矩阵
   * @param[in] x 当前状态向量
   * @param[in] id 虚拟装甲板 ID
   * @return 4×11 雅可比矩阵
   */
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
