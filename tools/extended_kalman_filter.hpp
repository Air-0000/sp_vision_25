/**
 * @file extended_kalman_filter.hpp
 * @brief 通用扩展卡尔曼滤波器（EKF）接口
 * @details 本文件属于 tools 工具模块，提供了一个通用的 EKF 实现，
 *          支持非线性状态转移和观测模型（通过函数对象注入）。
 *
 * 使用场景：
 * - Target 类使用此 EKF 进行敌方机器人的运动状态估计
 * - 状态向量可以是任意维度（模板化不够彻底，使用固定维度的运行时多态）
 *
 * ## EKF 算法步骤
 *
 * ### 预测步骤（Predict）
 *   x_{k|k-1} = f(x_{k-1})          // 非线性状态转移
 *   P_{k|k-1} = F * P_{k-1} * F^T + Q  // 协方差传播
 *
 * ### 更新步骤（Update）
 *   K = P_{k|k-1} * H^T * (H * P_{k|k-1} * H^T + R)^{-1}  // 卡尔曼增益
 *   x_{k|k} = x_{k|k-1} + K * (z - h(x_{k|k-1}))         // 状态更新
 *   P_{k|k} = (I - K*H) * P_{k|k-1} * (I - K*H)^T + K*R*K^T  // 协方差更新（稳定形式）
 *
 * ### 卡方检验（NIS/NEES）
 *   NIS = residual^T * S^{-1} * residual  // Normalized Innovation Squared
 *   NEES = (x - x_prior)^T * P^{-1} * (x - x_prior)  // Normalized Estimation Error Squared
 *   用于检测 EKF 是否发散或与观测不一致。
 *
 * @note 使用 Joseph 形式的协方差更新公式，保证了数值稳定性
 *       （始终为正定，即使卡尔曼增益计算存在数值误差）。
 */

#ifndef TOOLS__EXTENDED_KALMAN_FILTER_HPP
#define TOOLS__EXTENDED_KALMAN_FILTER_HPP

#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <map>

namespace tools
{

/**
 * @brief 通用扩展卡尔曼滤波器
 * @details 支持自定义状态转移函数 f(x) 和观测函数 h(x)，
 *          以及自定义状态加法和观测减法（用于处理角度等周期性变量）。
 *
 * 典型用法（11 维状态，4 维观测）：
 * @code
 *   // 定义状态转移函数 f(x) = F * x（线性）
 *   auto f = [&](const VectorXd & x) { return F * x; };
 *   ekf.predict(F, Q, f);
 *
 *   // 定义观测函数 h(x)
 *   auto h = [&](const VectorXd & x) -> Vector4d { ... };
 *   ekf.update(z, H, R, h);
 * @endcode
 */
class ExtendedKalmanFilter
{
public:
  // ---- 公开状态 ----
  Eigen::VectorXd x;   ///< 当前状态向量（后验估计 x_{k|k}）
  Eigen::MatrixXd P;   ///< 当前状态协方差矩阵（后验协方差 P_{k|k}）

  /**
   * @brief 默认构造函数（创建空滤波器，需要手动设置 x 和 P）
   */
  ExtendedKalmanFilter() = default;

  /**
   * @brief 带初始状态的构造函数
   * @param[in] x0 初始状态向量（n×1）
   * @param[in] P0 初始协方差矩阵（n×n）
   * @param[in] x_add 自定义状态加法函数，默认 a+b。
   *                   用于处理角度等特殊变量的加法（如角度需要 normalize）
   */
  ExtendedKalmanFilter(
    const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a + b; });

  /**
   * @brief EKF 预测步骤（线性状态转移）
   * @param[in] F 状态转移矩阵（n×n）
   * @param[in] Q 过程噪声协方差矩阵（n×n）
   * @return 预测后的先验状态 x_{k|k-1}
   */
  Eigen::VectorXd predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q);

  /**
   * @brief EKF 预测步骤（非线性状态转移）
   * @param[in] F 状态转移矩阵（n×n，雅可比）
   * @param[in] Q 过程噪声协方差矩阵（n×n）
   * @param[in] f 非线性状态转移函数 f(x)
   * @return 预测后的先验状态 x_{k|k-1}
   */
  Eigen::VectorXd predict(
    const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f);

  /**
   * @brief EKF 更新步骤（线性观测模型）
   * @param[in] z 观测向量（m×1）
   * @param[in] H 观测矩阵（m×n）
   * @param[in] R 观测噪声协方差矩阵（m×m）
   * @param[in] z_subtract 自定义观测减法函数，默认 a-b。
   *                        用于角度等需要 normalize 的差值计算
   * @return 更新后的后验状态 x_{k|k}
   */
  Eigen::VectorXd update(
    const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; });

  /**
   * @brief EKF 更新步骤（非线性观测模型）
   * @param[in] z 观测向量（m×1）
   * @param[in] H 观测雅可比矩阵（m×n）
   * @param[in] R 观测噪声协方差矩阵（m×m）
   * @param[in] h 非线性观测函数 h(x)
   * @param[in] z_subtract 自定义观测减法函数
   * @return 更新后的后验状态 x_{k|k}
   */
  Eigen::VectorXd update(
    const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; });

  // ---- 卡方检验数据 ----
  std::map<std::string, double> data;  ///< 诊断数据（残差、NIS、NEES 等）
  std::deque<int> recent_nis_failures{0};  ///< 最近 N 次更新的 NIS 是否失败的滑动窗口
  size_t window_size = 100;            ///< 滑动窗口大小
  double last_nis;                     ///< 最近一次的 NIS 值

private:
  Eigen::MatrixXd I;  ///< n×n 单位矩阵（在构造函数中根据 x0 维数初始化）
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add;

  int nees_count_ = 0;   ///< NEES 失败次数计数
  int nis_count_ = 0;    ///< NIS 失败次数计数
  int total_count_ = 0;  ///< 总更新次数计数
};

}  // namespace tools

#endif  // TOOLS__EXTENDED_KALMAN_FILTER_HPP
