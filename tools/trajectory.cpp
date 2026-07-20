/**
 * @file trajectory.cpp
 * @brief 弹道解算实现
 * @details 实现重力弹道模型的计算。
 *
 * ## 数学推导
 *
 * 子弹运动方程（忽略空气阻力）：
 *   x(t) = v0·cos(pitch)·t  →  t = d / (v0·cos(pitch))
 *   y(t) = v0·sin(pitch)·t - ½·g·t²
 *
 * 代入 t 到 y(t) 方程：
 *   h = v0·sin(pitch)·[d / (v0·cos(pitch))] - ½·g·[d / (v0·cos(pitch))]²
 *   h = d·tan(pitch) - g·d² / (2·v0²·cos²(pitch))
 *
 * 利用 1/cos² = 1 + tan²：
 *   h = d·tan(pitch) - g·d²·(1 + tan²(pitch)) / (2·v0²)
 *
 * 整理为关于 tan(pitch) 的一元二次方程：
 *   [g·d² / (2·v0²)]·tan²(pitch) - d·tan(pitch) + [g·d² / (2·v0²) + h] = 0
 *
 * 即：a·tan²(pitch) + b·tan(pitch) + c = 0
 *    a = g·d² / (2·v0²)
 *    b = -d
 *    c = a + h
 *
 * 判别式：Δ = b² - 4ac = d² - 4a(a + h)
 *   若 Δ < 0，目标超出射程，无解
 *
 * 两个解对应两种弹道：
 *   解 1（高抛）：tan(pitch) 较大，pitch 角大，飞行时间长
 *   解 2（低抛）：tan(pitch) 较小，pitch 角小，飞行时间短
 *
 * 自瞄通常选择低抛弹道（飞行时间短，命中率更高）。
 *
 * @note 重力加速度 g = 9.7833 m/s²（上海地区）。
 *       精确的重力值对远距离（>5m）弹道有显著影响。
 */

#include "trajectory.hpp"

#include <cmath>

namespace tools
{

/// 重力加速度，单位：m/s²（上海地区）
constexpr double g = 9.7833;

Trajectory::Trajectory(const double v0, const double d, const double h)
{
  // 一元二次方程系数
  auto a = g * d * d / (2 * v0 * v0);
  auto b = -d;
  auto c = a + h;

  // 判别式
  auto delta = b * b - 4 * a * c;

  if (delta < 0) {
    unsolvable = true;  // 目标超出射程
    return;
  }

  unsolvable = false;

  // 两个解对应两种弹道
  auto tan_pitch_1 = (-b + std::sqrt(delta)) / (2 * a);  // 高抛（tan 大）
  auto tan_pitch_2 = (-b - std::sqrt(delta)) / (2 * a);  // 低抛（tan 小）

  auto pitch_1 = std::atan(tan_pitch_1);
  auto pitch_2 = std::atan(tan_pitch_2);

  auto t_1 = d / (v0 * std::cos(pitch_1));  // 高抛飞行时间（长）
  auto t_2 = d / (v0 * std::cos(pitch_2));  // 低抛飞行时间（短）

  // 选择飞行时间较短的低抛弹道
  pitch = (t_1 < t_2) ? pitch_1 : pitch_2;
  fly_time = (t_1 < t_2) ? t_1 : t_2;
}

}  // namespace tools
