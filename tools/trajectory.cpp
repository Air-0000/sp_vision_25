/**
 * @file trajectory.cpp
 * @brief 弹道解算实现（纯重力解析解 + 空气阻力数值解）
 * @details 实现两种弹道模型：
 *          1. 纯重力模型：解析求解一元二次方程，O(1) 时间
 *          2. 重力+阻力模型：RK4 数值积分 + 二分法求根
 *
 * ## 空气阻力模型（RK4 数值积分）
 * 状态向量：[x, y, vx, vy]
 * 微分方程：
 *   dx/dt = vx
 *   dy/dt = vy
 *   dvx/dt = -k * v * vx          （阻力在 x 方向的分量）
 *   dvy/dt = -g - k * v * vy       （重力 + 阻力在 y 方向的分量）
 *   其中 v = sqrt(vx² + vy²)，k = 0.5*ρ*Cd*A/m
 *
 * ## 工程注意
 * 两个 Trajectory 构造函数参数数量不同（3 参数 vs 4 参数），
 * 编译器通过 bool use_drag 参数区分。当前 aimer.cpp 调用的是
 * 3 参数版本（纯重力模型）。启用阻力模型只需改为 4 参数调用：
 *   Trajectory(v0, d, h, true);
 *
 * 在 aimer.cpp 中，可根据目标距离动态选择模型：
 *   bool use_drag = (d > 5.0);
 *   Trajectory traj(bullet_speed, d, xyz.z(), use_drag);
 */

#include "trajectory.hpp"

#include <cmath>
#include <algorithm>

namespace tools
{

// ===================== 构造 1：纯重力模型（解析解） =====================

Trajectory::Trajectory(const double v0, const double d, const double h)
{
  /**
   * 一元二次方程系数（关于 tan(pitch)）：
   *   a * tan²(p) + b * tan(p) + c = 0
   * 其中 a = g*d² / (2 * v0²)
   *      b = -d
   *      c = a + h
   */
  auto a = G * d * d / (2 * v0 * v0);
  auto b = -d;
  auto c = a + h;
  auto delta = b * b - 4 * a * c;

  if (delta < 0) {
    unsolvable = true;
    fly_time = 0;
    pitch = 0;
    v_hit = 0;
    return;
  }

  unsolvable = false;

  // 两个解：高抛（tan 大，飞行时间长）和低抛（tan 小，飞行时间短）
  auto tan_pitch_1 = (-b + std::sqrt(delta)) / (2 * a);
  auto tan_pitch_2 = (-b - std::sqrt(delta)) / (2 * a);

  auto pitch_1 = std::atan(tan_pitch_1);
  auto pitch_2 = std::atan(tan_pitch_2);

  // 选择低抛弹道（飞行时间更短，命中率更高）
  auto t_1 = d / (v0 * std::cos(pitch_1));
  auto t_2 = d / (v0 * std::cos(pitch_2));

  if (t_1 < t_2) {
    pitch = pitch_1;
    fly_time = t_1;
  } else {
    pitch = pitch_2;
    fly_time = t_2;
  }

  // 命中速度（仅有重力作用，水平速度不变，竖直速度增加）
  double vx = v0 * std::cos(pitch);
  double vy = v0 * std::sin(pitch) - G * fly_time;
  v_hit = std::sqrt(vx * vx + vy * vy);
}

// ===================== 构造 2：重力+阻力模型（数值解） =====================

Trajectory::Trajectory(const double v0, const double d, const double h, bool use_drag)
{
  if (!use_drag) {
    // 如果 use_drag=false，退回到纯重力模型
    *this = Trajectory(v0, d, h);
    return;
  }

  /**
   * RK4 数值积分求解含空气阻力的弹道。
   *
   * 求解策略：
   * 1. 用纯重力模型的解作为 pitch 初始猜测
   * 2. 做一次完整 RK4 积分，得到落点 (x_hit, y_hit)
   * 3. 比较 y_hit 与目标高度 h 的偏差
   * 4. 用二分法修正 pitch（偏差 > 1cm 则继续迭代）
   */

  // 先用纯重力模型得到初始解
  Trajectory gravity_only(v0, d, h);
  if (gravity_only.unsolvable) {
    *this = gravity_only;
    return;
  }

  // 二分法搜索最优 pitch
  double lo = gravity_only.pitch - 0.1;  // 向下放宽 0.1 rad (~5.7°)
  double hi = gravity_only.pitch + 0.2;  // 向上放宽 0.2 rad (~11.4°)

  // 边界保护（pitch 不能太低也不能太高）
  lo = std::max(lo, -0.3);
  hi = std::min(hi, 0.6);

  double best_pitch = gravity_only.pitch;
  double best_y_err = 1e10;
  double best_t = gravity_only.fly_time;
  double best_vh = gravity_only.v_hit;

  for (int iter = 0; iter < MAX_ITER; iter++) {
    double pitch_mid = (lo + hi) / 2.0;

    // RK4 积分
    double vx = v0 * std::cos(pitch_mid);
    double vy = v0 * std::sin(pitch_mid);
    double x = 0, y = 0;
    double dt = 0.0005;  // 0.5ms 步长
    double t = 0;
    bool hit = false;

    // 最多积分 3 秒（15m 以内足够）
    constexpr double MAX_T = 3.0;
    constexpr int MAX_STEPS = int(MAX_T / dt);

    for (int step = 0; step < MAX_STEPS; step++) {
      // RK4 步进
      double k1x = vx, k1y = vy;
      double v = std::sqrt(vx * vx + vy * vy);
      double k1vx = -KD * v * vx;
      double k1vy = -G - KD * v * vy;

      double k2x = vx + 0.5 * dt * k1vx;
      double k2y = vy + 0.5 * dt * k1vy;
      double v2 = std::sqrt(k2x * k2x + k2y * k2y);
      double k2vx = -KD * v2 * k2x;
      double k2vy = -G - KD * v2 * k2y;

      double k3x = vx + 0.5 * dt * k2vx;
      double k3y = vy + 0.5 * dt * k2vy;
      double v3 = std::sqrt(k3x * k3x + k3y * k3y);
      double k3vx = -KD * v3 * k3x;
      double k3vy = -G - KD * v3 * k3y;

      double k4x = vx + dt * k3vx;
      double k4y = vy + dt * k3vy;
      double v4 = std::sqrt(k4x * k4x + k4y * k4y);
      double k4vx = -KD * v4 * k4x;
      double k4vy = -G - KD * v4 * k4y;

      double dx = (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x);
      double dy = (dt / 6.0) * (k1y + 2*k2y + 2*k3y + k4y);
      double dvx = (dt / 6.0) * (k1vx + 2*k2vx + 2*k3vx + k4vx);
      double dvy = (dt / 6.0) * (k1vy + 2*k2vy + 2*k3vy + k4vy);

      x += dx;
      y += dy;
      vx += dvx;
      vy += dvy;
      t += dt;

      // 检查是否越过目标水平距离
      if (x >= d && y <= h + 0.5) {  // 在目标附近穿过
        hit = true;
        break;
      }

      // 落地停止（y < 0 且正在下降）
      if (y < 0 && vy < 0) {
        break;
      }
    }

    // 如果未命中，偏差很大
    double y_err = hit ? std::abs(y - h) : 100.0;

    if (y_err < best_y_err) {
      best_y_err = y_err;
      best_pitch = pitch_mid;
      best_t = t;
      best_vh = std::sqrt(vx * vx + vy * vy);
    }

    // 收敛检查
    if (y_err < CONV_TOL) {
      break;
    }

    // 二分法更新边界
    // 目标在落点上方 → pitch 需要更大（抬高枪口）
    // 目标在落点下方 → pitch 需要更小（压低枪口）
    if (hit && y < h) {
      lo = pitch_mid;    // 当前 pitch 偏低，向上搜索
    } else {
      hi = pitch_mid;    // 当前 pitch 偏高，向下搜索
    }
  }

  unsolvable = (best_y_err > 0.5);  // 偏差 > 50cm 认为无解
  pitch = best_pitch;
  fly_time = best_t;
  v_hit = best_vh;
}

}  // namespace tools
