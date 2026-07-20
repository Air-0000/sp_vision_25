/**
 * @file extended_kalman_filter.cpp
 * @brief 通用扩展卡尔曼滤波器实现
 * @details 实现 EKF 的预测、更新步骤，以及卡方检验（NIS/NEES）的在线监控。
 *
 * 协方差更新使用 Joseph 形式（也称稳定形式）：
 *   P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
 * 这种形式在数值上比标准形式更稳定，保证 P 始终为对称正定矩阵。
 *
 * 参考：https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
 */

#include "extended_kalman_filter.hpp"

#include <numeric>

namespace tools
{

// ========== 构造函数 ==========

ExtendedKalmanFilter::ExtendedKalmanFilter(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(x_add)
{
  // 初始化诊断数据
  data["residual_yaw"] = 0.0;
  data["residual_pitch"] = 0.0;
  data["residual_distance"] = 0.0;
  data["residual_angle"] = 0.0;
  data["nis"] = 0.0;
  data["nees"] = 0.0;
  data["nis_fail"] = 0.0;
  data["nees_fail"] = 0.0;
  data["recent_nis_failures"] = 0.0;
}

// ========== EKF 预测 ==========

Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q)
{
  return predict(F, Q, [&](const Eigen::VectorXd & x) { return F * x; });
}

Eigen::VectorXd ExtendedKalmanFilter::predict(
  const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{
  /**
   * 预测步骤：
   *   协方差传播：P = F * P * F^T + Q
   *   状态转移：x = f(x)
   */
  P = F * P * F.transpose() + Q;
  x = f(x);
  return x;
}

// ========== EKF 更新 ==========

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  return update(z, H, R, [&](const Eigen::VectorXd & x) { return H * x; }, z_subtract);
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  /**
   * ## 更新步骤
   *
   * 1. 计算观测残差：y = z - h(x)
   * 2. 计算创新协方差：S = H * P * H^T + R
   * 3. 计算卡尔曼增益：K = P * H^T * S^{-1}
   * 4. 更新状态（使用 x_add 自定义加法）：x += K * y
   * 5. 更新协方差（Joseph 形式）
   * 6. 计算 NIS 和 NEES 进行卡方检验
   */
  Eigen::VectorXd x_prior = x;
  Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + R).inverse();

  // Joseph 形式的协方差更新（稳定）
  P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();

  // 自定义加法更新状态
  x = x_add(x, K * z_subtract(z, h(x)));

  // ---------- 卡方检验 ----------
  Eigen::VectorXd residual = z_subtract(z, h(x));
  Eigen::MatrixXd S = H * P * H.transpose() + R;
  double nis = residual.transpose() * S.inverse() * residual;
  double nees = (x - x_prior).transpose() * P.inverse() * (x - x_prior);

  // 卡方检验阈值（自由度 = 4，取置信水平 95%）
  // 4 自由度下，95% 置信水平的临界值约为 0.711（归一化后的阈值）
  constexpr double nis_threshold = 0.711;
  constexpr double nees_threshold = 0.711;

  if (nis > nis_threshold) nis_count_++, data["nis_fail"] = 1;
  if (nees > nees_threshold) nees_count_++, data["nees_fail"] = 1;
  total_count_++;
  last_nis = nis;

  // 滑动窗口记录 NIS 失败率
  recent_nis_failures.push_back(nis > nis_threshold ? 1 : 0);
  if (recent_nis_failures.size() > window_size) {
    recent_nis_failures.pop_front();
  }

  int recent_failures = std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);
  double recent_rate = static_cast<double>(recent_failures) / recent_nis_failures.size();

  // 填充诊断数据
  data["residual_yaw"] = residual[0];
  data["residual_pitch"] = residual[1];
  data["residual_distance"] = residual[2];
  data["residual_angle"] = residual[3];
  data["nis"] = nis;
  data["nees"] = nees;
  data["recent_nis_failures"] = recent_rate;

  return x;
}

}  // namespace tools
