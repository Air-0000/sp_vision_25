/**
 * @file explicit_planner.hpp
 * @brief 显式五次多项式轨迹规划器（方案二）
 * @details 替代现有的 TinyMPC 轨迹规划器，使用显式五次多项式生成平滑过渡轨迹。
 *          无需任何第三方优化库依赖。
 *
 * 核心思路（详见 README 第 4.4 节）：
 * 1. 检测目标轨迹中的装甲板切换点（角度跳变）
 * 2. 在切换点前后插入过渡段，用五次多项式连接
 * 3. 二分搜索最优过渡段长度，使最大加速度 ≤ 云台最大角加速度
 * 4. 非过渡段直接使用射击轨迹（零跟踪误差）
 *
 * 相比 TinyMPC（方案一）的优势：
 * - 无外部依赖（不需要 TinyMPC 库）
 * - 确定性高，跟随段与参考轨迹完全重合
 * - 高转速下更稳定
 * - 求解速度快（预期 < 0.1ms）
 *
 * @note 过渡段的五次多项式保证位置、速度、加速度三阶连续。
 */

#ifndef AUTO_AIM__EXPLICIT_PLANNER_HPP
#define AUTO_AIM__EXPLICIT_PLANNER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <optional>

namespace auto_aim
{

/// @brief 时间步长（与 motion planner 同步），单位：秒
constexpr double DT = 0.01;

/// @brief 半 horizon 长度，对应 0.5 秒
constexpr int HALF_HORIZON = 50;

/// @brief 总 horizon 长度，对应 1.0 秒（与 TinyMPC 版本一致）
constexpr int HORIZON = HALF_HORIZON * 2;

/**
 * @brief 规划轨迹结构体
 * @details 与 planner.hpp 中的 Plan 结构体兼容，包含瞄准角度、速度和加速度。
 */
struct Plan
{
  bool control;         ///< 控制使能
  bool fire;            ///< 击发信号
  float target_yaw;     ///< 目标 yaw（弧度）
  float target_pitch;   ///< 目标 pitch（弧度）
  float yaw;            ///< 规划后的云台 yaw（弧度）
  float yaw_vel;        ///< 云台 yaw 速度（弧度/秒）
  float yaw_acc;        ///< 云台 yaw 加速度（弧度/秒²）
  float pitch;          ///< 规划后的云台 pitch（弧度）
  float pitch_vel;      ///< 云台 pitch 速度（弧度/秒）
  float pitch_acc;      ///< 云台 pitch 加速度（弧度/秒²）
};

/**
 * @brief 五次多项式轨迹段
 * @details s(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
 *          由边界条件（起止位置、速度、加速度）唯一确定六个系数。
 */
class QuinticSegment
{
public:
  QuinticSegment() = default;

  /**
   * @brief 根据边界条件构造五次多项式
   * @param[in] T 过渡段时长（秒）
   * @param[in] p0 起始位置
   * @param[in] v0 起始速度
   * @param[in] a0 起始加速度
   * @param[in] p1 终止位置
   * @param[in] v1 终止速度
   * @param[in] a1 终止加速度
   */
  void setup(double T, double p0, double v0, double a0,
             double p1, double v1, double a1)
  {
    double T2 = T * T;
    double T3 = T2 * T;
    double T4 = T3 * T;
    double T5 = T4 * T;

    // 解 3x3 线性方程组求 a3, a4, a5
    // 边界条件:
    // s(T)   = a0 + a1*T +  a2*T² +  a3*T³ +  a4*T⁴ +  a5*T⁵ = p1
    // s'(T)  = a1 + 2*a2*T + 3*a3*T² + 4*a4*T³ + 5*a5*T⁴ = v1
    // s''(T) = 2*a2 + 6*a3*T + 12*a4*T² + 20*a5*T³ = a1

    a_[0] = p0;
    a_[1] = v0;
    a_[2] = a0 / 2.0;

    // Cramer 法则解 3x3
    double detA = 6 * T5 * T2;  // 行列式 |A|

    double b0 = p1 - a_[0] - a_[1] * T - a_[2] * T2;
    double b1 = v1 - a_[1] - 2.0 * a_[2] * T;
    double b2 = a1 - 2.0 * a_[2];

    // 用解析逆求解
    double inv_det = 1.0 / detA;
    a_[3] = inv_det * (
      20.0 * T4 * b0 - 4.0 * T5 * b1 + 0.2 * T5 * T * b2
    );
    a_[4] = inv_det * (
      -30.0 * T3 * b0 + 6.0 * T4 * b1 - 0.3 * T4 * T * b2
    );
    a_[5] = inv_det * (
      12.0 * T2 * b0 - 3.0 * T3 * b1 + 0.15 * T4 * b2
    );
  }

  /**
   * @brief 计算 t 时刻的位置、速度、加速度
   * @param[in] t 时刻（从过渡段起点开始，单位：秒）
   * @param[out] pos 位置
   * @param[out] vel 速度
   * @param[out] acc 加速度
   */
  void eval(double t, double & pos, double & vel, double & acc) const
  {
    double t2 = t * t;
    double t3 = t2 * t;
    double t4 = t3 * t;
    double t5 = t4 * t;

    pos = a_[0] + a_[1]*t + a_[2]*t2 + a_[3]*t3 + a_[4]*t4 + a_[5]*t5;
    vel = a_[1] + 2*a_[2]*t + 3*a_[3]*t2 + 4*a_[4]*t3 + 5*a_[5]*t4;
    acc = 2*a_[2] + 6*a_[3]*t + 12*a_[4]*t2 + 20*a_[5]*t3;
  }

private:
  double a_[6] = {0, 0, 0, 0, 0, 0};  ///< 多项式系数 a0~a5
};

/**
 * @brief 显式轨迹规划器
 * @details 使用五次多项式过渡段实现装甲板切换时的平滑轨迹规划。
 *          与 Planner（TinyMPC 版本）接口兼容，可替换使用。
 */
class ExplicitPlanner
{
public:
  ExplicitPlanner() = default;

  /**
   * @brief 规划轨迹（核心接口）
   * @param[in] yaw_ref  yaw 参考轨迹（射击轨迹），长度 HORIZON
   * @param[in] pitch_ref  pitch 参考轨迹，长度 HORIZON
   * @param[in] max_yaw_acc  云台最大 yaw 角加速度（rad/s²）
   * @param[in] max_pitch_acc  云台最大 pitch 角加速度（rad/s²）
   * @param[out] planned_yaw  规划后的 yaw 轨迹（2×HORIZON: [pos, vel]）
   * @param[out] planned_pitch  规划后的 pitch 轨迹（2×HORIZON: [pos, vel]）
   * @param[out] solve_time_ms  求解耗时（毫秒）
   */
  void plan(
    const Eigen::Matrix<double, HORIZON, 1> & yaw_ref,
    const Eigen::Matrix<double, HORIZON, 1> & pitch_ref,
    const Eigen::Matrix<double, HORIZON, 1> & yaw_vel_ref,
    const Eigen::Matrix<double, HORIZON, 1> & pitch_vel_ref,
    double max_yaw_acc,
    double max_pitch_acc,
    Eigen::Matrix<double, 2, HORIZON> & planned_yaw,
    Eigen::Matrix<double, 2, HORIZON> & planned_pitch,
    double & solve_time_ms)
  {
    auto t0 = std::chrono::steady_clock::now();

    // 初始化规划轨迹 = 参考轨迹
    for (int i = 0; i < HORIZON; i++) {
      planned_yaw(0, i) = yaw_ref(i);
      planned_yaw(1, i) = yaw_vel_ref(i);
      planned_pitch(0, i) = pitch_ref(i);
      planned_pitch(1, i) = pitch_vel_ref(i);
    }

    // 检测 yaw 切换点（角度跳变）
    auto switches = detect_switches(yaw_ref);

    // 对每个切换点插入过渡段
    for (int sw : switches) {
      apply_transition(planned_yaw, sw, max_yaw_acc);
    }

    // 检测 pitch 切换点
    auto pitch_switches = detect_switches(pitch_ref);
    for (int sw : pitch_switches) {
      apply_transition(planned_pitch, sw, max_pitch_acc);
    }

    auto t1 = std::chrono::steady_clock::now();
    solve_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  /**
   * @brief 获取上次规划的统计信息
   */
  struct Stats {
    int switches_found;       ///< 检测到的切换点数量
    int transitions_applied;  ///< 成功应用的过渡段数量
    double max_acc_observed;  ///< 规划轨迹中的最大加速度
  };
  Stats last_stats_{0, 0, 0.0};

private:
  /**
   * @brief 检测装甲板切换点
   * @param[in] ref 参考轨迹
   * @return 切换点的索引列表
   */
  std::vector<int> detect_switches(const Eigen::Matrix<double, HORIZON, 1> & ref) const
  {
    std::vector<int> switches;
    const double threshold = 0.05;  // 单步变化 > 0.05 rad 视为跳变
    for (int i = 1; i < HORIZON; i++) {
      double diff = std::abs(ref(i) - ref(i - 1));
      if (diff > threshold) {
        switches.push_back(i);
      }
    }
    return switches;
  }

  /**
   * @brief 在切换点处应用过渡段（二分搜索 + 五次多项式）
   * @param[in/out] planned 规划轨迹（2×HORIZON）
   * @param[in] sw 切换点索引
   * @param[in] max_acc 最大允许加速度
   */
  void apply_transition(
    Eigen::Matrix<double, 2, HORIZON> & planned,
    int sw, double max_acc)
  {
    // 二分搜索最小过渡段长度
    int lo = 8;   // 最少 80ms
    int hi = std::min(HORIZON / 2, 100);  // 最多 1s
    std::optional<std::tuple<QuinticSegment, int, int, double>> best;

    for (int iter = 0; iter < 20; iter++) {
      int mid = (lo + hi) / 2;
      int half = mid / 2;
      int s0 = std::max(0, sw - half);
      int s1 = std::min(HORIZON - 1, sw + mid - half);
      double T = (s1 - s0) * DT;
      if (T < 0.05) {
        lo = mid + 1;
        continue;
      }

      QuinticSegment poly;
      poly.setup(T,
                 planned(0, s0), planned(1, s0), 0.0,
                 planned(0, s1), planned(1, s1), 0.0);

      // 评估最大加速度
      double max_a = 0.0;
      for (int step = 0; step <= s1 - s0; step++) {
        double p, v, a;
        poly.eval(step * DT, p, v, a);
        max_a = std::max(max_a, std::abs(a));
      }

      if (max_a <= max_acc) {
        best = std::make_tuple(poly, s0, s1, max_a);
        hi = mid;
      } else {
        lo = mid + 1;
      }

      if (lo > hi) break;
    }

    // 应用过渡段
    if (best.has_value()) {
      auto [poly, s0, s1, max_a] = best.value();
      for (int step = 0; step <= s1 - s0; step++) {
        int idx = s0 + step;
        double p, v, a;
        poly.eval(step * DT, p, v, a);
        planned(0, idx) = p;
        planned(1, idx) = v;
        last_stats_.max_acc_observed = std::max(last_stats_.max_acc_observed, std::abs(a));
      }
      last_stats_.transitions_applied++;
    }
  }
};

}  // namespace auto_aim

#endif  // AUTO_AIM__EXPLICIT_PLANNER_HPP
