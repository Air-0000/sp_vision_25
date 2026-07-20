/**
 * @file target.cpp
 * @brief EKF 目标状态估计器实现
 * @details 实现基于扩展卡尔曼滤波（EKF）的敌方机器人运动状态估计，
 *          包括状态预测、观测更新、装甲板匹配等核心逻辑。
 *
 * ## EKF 模型详解
 *
 * ### 状态向量（11 维）
 *   x = [x, vx, y, vy, z, vz, angle, omega, r, l, h]^T
 *
 * ### 状态转移模型（匀速 + 恒定转速）
 *   x_{k+1} = x_k + vx_k * dt
 *   y_{k+1} = y_k + vy_k * dt
 *   z_{k+1} = z_k + vz_k * dt
 *   angle_{k+1} = angle_k + omega_k * dt
 *   vx, vy, vz, omega, r, l, h 保持恒定（白噪声扰动）
 *
 * ### 观测模型
 *   对第 i 号虚拟装甲板：
 *   armor_xyz = [x - r*cos(angle_i), y - r*sin(angle_i), z + (id==1||id==3 ? h : 0)]
 *   观测 z = ypd(armor_xyz) + [0, 0, 0, armor_yaw]
 *   其中 ypd = yaw/pitch/distance 球坐标
 *
 * ### 过程噪声（Piecewise White Noise Model）
 *   Q 矩阵中加速度方差 v1=100，角加速度方差 v2=400（普通机器人），
 *   前哨站用较小的 v1=10, v2=0.1（前哨站为固定建筑，运动范围极小）。
 */

#include "target.hpp"

#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{

// ========== 步骤 3-1：Target 构造 ==========

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  /**
   * 从首帧装甲板初始化 EKF 状态。
   *
   * 初始状态 x0：
   *   - 旋转中心位置 = 装甲板位置 + 沿装甲板法线方向偏移半径 r
   *   - 速度为 0（首次观测无速度信息）
   *   - angle = 装甲板的 yaw 角
   *   - omega = 0（初始假设不旋转）
   *   - r = 传入的 radius（机器人型号决定）
   *   - l = 0, h = 0（初始假设为正装甲板）
   *
   * 公式：
   *   center_x = armor.x + r * cos(armor_yaw)
   *   center_y = armor.y + r * sin(armor_yaw)
   *   center_z = armor.z
   */
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // 状态向量：x vx y vy z vz a w r l h
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();  // 初始协方差矩阵（对角阵）

  // 自定义加法：角度 angle 需要做弧度归一化（limit_rad），避免角度跳变
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);  // angle 归一化到 (-π, π]
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

// 构造函数 2：仿真/调试用
Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

// ========== 步骤 3-2：EKF 预测 ==========

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  /**
   * ## EKF 预测步骤
   *
   * ### 状态转移矩阵 F（11×11）
   * 恒定速度 + 恒定转速模型：
   * - 位置 = 位置 + 速度 * dt（x, y, z 三个方向）
   * - 角度 = 角度 + 角速度 * dt
   * - 速度、角速度、半径、偏移：恒定（对角线为 1）
   *
   *    [1 dt 0  0  0  0  0  0  0  0  0]
   *    [0  1 0  0  0  0  0  0  0  0  0]
   *    [0  0 1 dt  0  0  0  0  0  0  0]
   *    [0  0 0  1  0  0  0  0  0  0  0]
   * F = [0  0 0  0  1 dt  0  0  0  0  0]
   *    [0  0 0  0  0  1  0  0  0  0  0]
   *    [0  0 0  0  0  0  1 dt  0  0  0]
   *    [0  0 0  0  0  0  0  1  0  0  0]
   *    [0  0 0  0  0  0  0  0  1  0  0]
   *    [0  0 0  0  0  0  0  0  0  1  0]
   *    [0  0 0  0  0  0  0  0  0  0  1]
   *
   * ### 过程噪声 Q（11×11）
   * 使用 Piecewise White Noise Model（分段白噪声模型）：
   * 加速度扰动通过 dt^4/4, dt^3/2, dt^2 等系数与方差 v1/v2 组合得到。
   *
   * 参考：https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
   */

  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  // Piecewise White Noise Model 的 Q 矩阵系数
  double v1, v2;
  if (name == ArmorName::outpost) {
    // 前哨站是固定建筑，运动范围极小，噪声方差小
    v1 = 10;    // 加速度方差（位置维度）
    v2 = 0.1;   // 角加速度方差（角度维度）
  } else {
    // 普通机器人（步兵/英雄/哨兵）运动范围大，噪声方差大
    v1 = 100;   // 加速度方差（位置维度）
    v2 = 400;   // 角加速度方差（角度维度，小陀螺可能高速旋转）
  }
  auto a = dt * dt * dt * dt / 4;  // dt^4/4
  auto b = dt * dt * dt / 2;       // dt^3/2
  auto c = dt * dt;                // dt^2

  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
  };
  // clang-format on

  // 非线性状态转移函数 f（用于 EKF 的预测步骤）
  // 主要是角度归一化处理
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  // 前哨站转速限制：如果前哨站的角速度超过 ±2 rad/s，强制限制到 ±2.51 rad/s
  // 因为前哨站的装甲板旋转速度已知（赛前可测量）
  if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2)
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;

  ekf_.predict(F, Q, f);
}

// ========== 步骤 3-3：EKF 更新（观测匹配 + 更新） ==========

void Target::update(const Armor & armor)
{
  /**
   * ## EKF 更新步骤
   *
   * ### 装甲板匹配
   * 当前观测到的装甲板需要与理论计算的 4 个（或 2 个）虚拟装甲板进行匹配。
   * 匹配标准：距离最近（球坐标 pitch 距离最小）。
   *
   * 取前 3 个距离最近的虚拟装甲板，再从中选择 yaw 角度差最小的作为匹配结果。
   * 这是因为简单的最近邻匹配可能选到角度差异大的错误装甲板。
   */
  int id;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  // 构建 (xyza, id) 对，并按球坐标距离排序
  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  // 按球坐标 pitch 角排序（pitch 角越小 ≈ 距离越近）
  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  // 取前 3 个距离最小的虚拟装甲板，选择角度差最小的
  for (int i = 0; i < 3; i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  // 如果匹配到的不是 0 号装甲板，说明发生了装甲板跳变（旋转到另一个装甲板到了正面）
  if (id != 0) jumped = true;

  // 检测是否发生装甲板切换（匹配的 ID 变了）
  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

  update_ypda(armor, id);
}

// ========== 步骤 3-4：观测更新（核心更新逻辑） ==========

void Target::update_ypda(const Armor & armor, int id)
{
  /**
   * ## 观测更新详细步骤
   *
   * ### 观测向量 z（4 维）
   *   z = [ypd_yaw, ypd_pitch, ypd_dist, armor_yaw]
   *     其中 ypd = 装甲板中心在球坐标系下的（yaw, pitch, distance）
   *     armor_yaw = 装甲板的 yaw 角（法线方向）
   *
   * ### 观测矩阵 H（4×11）
   *   通过 h_jacobian 计算雅可比矩阵。
   *
   * ### 观测噪声 R（4×4 对角阵）
   *   R 的对角线元素：
   *   - yaw 噪声：4e-3（弧度²），约 0.13°
   *   - pitch 噪声：4e-3（弧度²），约 0.13°
   *   - distance 噪声：与 delta_angle 有关
   *     log(|delta_angle| + 1) + 1：装甲板越偏转，距离观测噪声越大
   *   - armor_yaw 噪声：与距离有关
   *     log(|distance| + 1) / 200 + 9e-2：距离越远，yaw 观测噪声越大
   */

  // 观测雅可比矩阵
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);

  // 计算 delta_angle = 装甲板 yaw 与旋转中心 yaw 的差值
  // 用于自适应调整观测噪声
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

  // 自适应观测噪声：装甲板越偏转或越远，噪声越大
  Eigen::VectorXd R_dig{
    {4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};

  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 非线性观测函数 h：状态向量 → 观测量
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 自定义减法：角度需要做弧度归一化
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);  // yaw 差值归一化
    c[1] = tools::limit_rad(c[1]);  // pitch 差值归一化
    c[3] = tools::limit_rad(c[3]);  // angle 差值归一化
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

  ekf_.update(z, H, R, h, z_subtract);
}

// ========== 步骤 3-5：状态访问接口 ==========

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

// ========== 步骤 3-6：虚拟装甲板计算 ==========

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  /**
   * 根据当前 EKF 状态计算机器人的所有虚拟装甲板位置和朝向。
   *
   * 对于 armor_num_=4 的机器人，4 个装甲板等间隔分布（每 90° 一个），
   * 每个装甲板的位置 = 旋转中心 + 沿对应方向的半径偏移。
   *
   * @return vector<Eigen::Vector4d>，每项 = (x, y, z, angle)
   */
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

// ========== 步骤 3-7：EKF 状态检查 ==========

bool Target::diverged() const
{
  /**
   * EKF 发散检测：检查装甲板半径是否在合理范围内。
   * - r（短轴半径）应在 [0.05, 0.5] 米
   * - r + l（长轴半径）也应在 [0.05, 0.5] 米
   * 如果超出范围，说明 EKF 的估计值偏离真实值，标记为发散。
   */
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool Target::convergened()
{
  /**
   * 判断 EKF 是否收敛。
   * 非前哨站：3 次 update 后且未发散 → 认为收敛
   * 前哨站：需要 10 次 update（前哨站的观测更稀疏）
   */
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

// ========== 步骤 3-8：装甲板位置计算函数 h 及其雅可比 ==========

Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  /**
   * 计算第 id 号虚拟装甲板的世界坐标。
   *
   * 公式：
   *   angle_i = angle + i * 2π / N
   *   r_i = (id 是长轴方向) ? r + l : r
   *   z_i = (id 是长轴方向) ? z + h : z
   *   armor_x = center_x - r_i * cos(angle_i)
   *   armor_y = center_y - r_i * sin(angle_i)
   *   armor_z = center_z + (0 if short axis else h)
   *
   * @note 对于 4 装甲板的车辆，ID 1 和 3 是长轴方向（侧面装甲板），
   *       使用更大的半径 r + l 和更高的高度 z + h。
   */
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  /**
   * 计算观测函数 h 的雅可比矩阵 H（4×11）。
   *
   * H = H_ypda * H_xyza
   *
   * 其中 H_xyza 是装甲板位置对状态向量的偏导（4×11），
   * H_ypda 是球坐标对笛卡尔坐标的偏导（4×4）。
   *
   * 具体推导见工具函数 xyz2ypd_jacobian。
   */
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
