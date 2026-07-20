/**
 * @file trajectory.hpp
 * @brief 弹道解算模型接口
 * @details 本文件属于 tools 工具模块，定义重力弹道解算器。
 *          在已知子弹初速度 v0、目标水平距离 d、目标高度差 h 的前提下，
 *          求解所需的射击仰角 pitch 和子弹飞行时间 fly_time。
 *
 * 管线位置：步骤 4-3（弹道解算）：Aimer 使用 Trajectory 计算每个瞄准点的弹道参数
 *
 * ## 弹道模型
 * 当前实现使用简化的重力弹道模型，忽略空气阻力。
 * 对于 RoboMaster 的比赛场景（通常 1~10m 距离），空气阻力的影响较小，
 * 简化模型已经能够满足命中率要求。
 *
 * ### 模型公式
 *   子弹运动方程为：
 *     x(t) = v0 * cos(pitch) * t          // 水平方向匀速
 *     y(t) = v0 * sin(pitch) * t - g*t²/2  // 竖直方向匀加速（重力）
 *
 *   代入目标位置 (d, h) 得：
 *     d = v0 * cos(pitch) * t
 *     h = v0 * sin(pitch) * t - g*t²/2
 *
 *   消去 t，得到 pitch 的一元二次方程（关于 tan(pitch)）：
 *     (g*d² / 2*v0²) * tan²(pitch) - d * tan(pitch) + (g*d² / 2*v0² + h) = 0
 *
 *   即：a * tan²(pitch) + b * tan(pitch) + c = 0
 *   其中 a = g*d²/(2*v0²), b = -d, c = a + h
 *
 *   方程有两个解（高抛和低抛弹道），选择飞行时间较短的低抛弹道。
 *
 * @note 上海地区的重力加速度 g = 9.7833 m/s²（纬度 ~31°，海拔 ~4m）。
 *       如果比赛在其他城市，应重新测量（可用手机传感器或网上查询）。
 */

#ifndef TOOLS__TRAJECTORY_HPP
#define TOOLS__TRAJECTORY_HPP

namespace tools
{

/**
 * @brief 弹道解算结果结构体
 * @details 对于给定的 (v0, d, h)，计算出一个弹道解。
 *          包含解的可行性、所需 pitch 角和飞行时间。
 */
struct Trajectory
{
  bool unsolvable;   ///< 是否无解（目标在射程之外），true = 无法命中
  double fly_time;   ///< 子弹飞行时间（秒），从击发到命中的时间
  double pitch;      ///< 所需射击仰角（弧度），抬头为正

  /**
   * @brief 弹道解算构造函数
   * @param[in] v0 子弹初速度（米/秒），通常由测速仪实测
   * @param[in] d 目标水平距离（米），目标在枪口水平面上的投影距离
   * @param[in] h 目标高度差（米），目标在枪口水平面之上的高度，正数=目标更高
   *
   * @note 如果 delta < 0（目标超出射程），unsolvable 被设为 true。
   *       此时 pitch 和 fly_time 的值无效。
   */
  Trajectory(const double v0, const double d, const double h);
};

}  // namespace tools

#endif  // TOOLS__TRAJECTORY_HPP
