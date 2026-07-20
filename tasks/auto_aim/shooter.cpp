/**
 * @file shooter.cpp
 * @brief 击发决策器实现
 * @details 实现基于规则的击发决策逻辑。
 *
 * ## 击发决策规则
 *
 * ### 角度突变检测
 * 比较当前指令 yaw 与上一帧指令 yaw 的差值。
 * 如果差值 > tolerance * 2，说明发生了指令突变（如目标切换），
 * 此时不应射击，应等待云台重新稳定。
 *
 * ### 云台到位检测
 * 比较云台当前实际位置与指令位置的偏差。
 * 如果 |gimbal_actual_yaw - command_yaw| < tolerance，
 * 说明云台已到达目标位置，可以击发。
 *
 * ### 远/近距离自适应容差
 * 目标距离 < judge_distance（通常 2~3 米）时使用 first_tolerance（更大），
 * 目标距离 ≥ judge_distance 时使用 second_tolerance（更小）。
 */

#include "shooter.hpp"

#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{

// ========== 步骤 4-5：击发器初始化 ==========

Shooter::Shooter(const std::string & config_path) : last_command_{false, false, 0, 0}
{
  auto yaml = YAML::LoadFile(config_path);

  /**
   * 读取击发参数：
   *
   * first_tolerance：近距离（< judge_distance）射击容差
   *   此时目标在画面上较大，轻微瞄准偏差也能命中，容差可设大（如 3°）
   *
   * second_tolerance：远距离（≥ judge_distance）射击容差
   *   远距离目标在画面上小，需要更精确的瞄准，设小（如 2°）
   *
   * judge_distance：远近判断阈值（米）
   *
   * auto_fire：是否开启自动射击
   *   true = 自瞄满足条件时自动开枪
   *   false = 只输出瞄准角度，由操作手手动开枪
   */
  first_tolerance_ = yaml["first_tolerance"].as<double>() / 57.3;    // degree → rad
  second_tolerance_ = yaml["second_tolerance"].as<double>() / 57.3;  // degree → rad
  judge_distance_ = yaml["judge_distance"].as<double>();
  auto_fire_ = yaml["auto_fire"].as<bool>();
}

// ========== 步骤 4-5：击发决策 ==========

bool Shooter::shoot(
  const io::Command & command, const auto_aim::Aimer & aimer,
  const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos)
{
  /**
   * ## 击发条件判断
   *
   * 条件 1：command.control == true
   *   瞄准指令有效（Aimer 成功解算出瞄准角度）
   *
   * 条件 2：targets 不为空
   *   当前正在跟踪目标
   *
   * 条件 3：auto_fire == true
   *   自动射击模式已开启
   *
   * 条件 4：|last_command.yaw - command.yaw| < tolerance * 2
   *   指令未发生突变。如果目标刚切换，yaw 会跳变，此时不应射击
   *
   * 条件 5：|gimbal_actual_yaw - command.yaw| < tolerance
   *   云台实际位置已到达指令位置（枪口已对准目标）
   *
   * 条件 6：aimer.debug_aim_point.valid == true
   *   瞄准点有效
   */
  if (!command.control || targets.empty() || !auto_fire_) return false;

  // 根据目标距离选择射击容差
  auto target_x = targets.front().ekf_x()[0];
  auto target_y = targets.front().ekf_x()[2];
  auto tolerance = std::sqrt(tools::square(target_x) + tools::square(target_y)) > judge_distance_
                     ? second_tolerance_
                     : first_tolerance_;

  if (
    std::abs(last_command_.yaw - command.yaw) < tolerance * 2 &&  // 指令未突变
    std::abs(gimbal_pos[0] - last_command_.yaw) < tolerance &&    // 云台已到位
    aimer.debug_aim_point.valid) {                                // 瞄准点有效
    last_command_ = command;
    return true;
  }

  last_command_ = command;
  return false;
}

}  // namespace auto_aim
