/**
 * @file shooter.hpp
 * @brief 击发决策器接口
 * @details 本文件属于 auto_aim 模块的决策子模块，定义了 Shooter 类。
 *          Shooter 负责判断当前是否应该开枪射击，基于云台当前指令和上一帧指令的差异、
 *          云台实际位置与指令位置的偏差以及瞄准点的有效性来决策。
 *
 * 管线位置：步骤 4-5（击发决策）：Aimer 输出 Command → Shooter::shoot()
 *          判断是否应击发 → CBoard::send() 发送包含 shoot 标志的指令
 *
 * 击发条件：
 * 1. command.control == true（瞄准有效）
 * 2. targets 不为空（有目标）
 * 3. auto_fire == true（自瞄允许自动射击）
 * 4. 指令的 yaw 变化量 < tolerance * 2（防止指令突变时误射）
 * 5. 云台实际位置与指令位置的偏差 < tolerance（枪口已到位）
 * 6. 瞄准点有效（aimer.debug_aim_point.valid）
 *
 * @note 容差分为第一容差（近距离）和第二容差（远距离），
 *       远距离时需要更高的瞄准精度，因此容差更小。
 */

#ifndef AUTO_AIM__SHOOTER_HPP
#define AUTO_AIM__SHOOTER_HPP

#include <string>

#include "io/command.hpp"
#include "tasks/auto_aim/aimer.hpp"

namespace auto_aim
{

/**
 * @brief 击发决策器
 * @details 基于规则判断是否应击发子弹。核心逻辑是：
 *          - 避免指令突变时射击（如目标切换瞬间 yaw 跳变）
 *          - 等待云台实际位置跟上指令后再击发
 *          - 根据目标距离动态调整瞄准精度要求
 *
 * @note auto_fire 可以在配置中关闭，此时 Shooter 始终返回 false，
 *       由于动开关（遥控器）控制击发。
 */
class Shooter
{
public:
  /**
   * @brief 构造函数
   * @param[in] config_path YAML 配置文件路径
   */
  Shooter(const std::string & config_path);

  /**
   * @brief 击发决策（步骤 4-5）
   * @param[in] command 当前帧的瞄准指令（含 yaw/pitch 目标角度）
   * @param[in] aimer Aimer 实例（读取 debug_aim_point 判断瞄准点有效性）
   * @param[in] targets 当前跟踪的目标列表
   * @param[in] gimbal_pos 云台当前实际位置 (yaw, pitch, roll) in rad
   * @return true = 允许击发，false = 不允许
   */
  bool shoot(
    const io::Command & command, const auto_aim::Aimer & aimer,
    const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos);

private:
  io::Command last_command_;  ///< 上一帧的下发指令（用于检测指令突变）

  // ---------- 击发参数 ----------
  double judge_distance_;     ///< 距离判断阈值（米），用于区分近/远距离射击模式
  double first_tolerance_;    ///< 第一容差（近距离，度→弧度），近距离容差可更大
  double second_tolerance_;   ///< 第二容差（远距离，度→弧度），远距离要求更精确
  bool auto_fire_;            ///< 是否允许自瞄自动控制击发（false = 手动控制）
};

}  // namespace auto_aim

#endif  // AUTO_AIM__SHOOTER_HPP
