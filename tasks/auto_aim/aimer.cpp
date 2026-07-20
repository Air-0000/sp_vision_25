/**
 * @file aimer.cpp
 * @brief 瞄准决策器实现
 * @details 实现弹道解算、目标预测、瞄准点选择和飞行时间迭代收敛。
 *
 * ## 核心算法
 *
 * ### 弹道模型
 * 使用简化的重力弹道模型（忽略空气阻力）：
 *   pitch = atan2(v² ± sqrt(v⁴ - g(g·d² + 2·v²·z)) , g·d)
 * 其中 v = 子弹速度（m/s），d = 水平距离（m），z = 高度差（m），g = 9.81 m/s²
 *
 * ### 飞行时间迭代
 * 1. 根据当前目标位置计算子弹飞行时间 t_fly
 * 2. 预测目标在 (当前时间 + t_fly) 时刻的位置
 * 3. 根据新位置重新计算 t_fly
 * 4. 重复直到收敛（相邻两次 t_fly 差 < 1ms）或达到最大迭代次数（10 次）
 *
 * ### 瞄准点选择策略
 * - 不转陀螺（|omega| < decision_speed）：选择朝向角度最接近的装甲板，锁定模式防抖
 * - 小陀螺（|omega| > decision_speed）：选择"正在进入视野"的装甲板（迎击面概率更高）
 */

#include "aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_aim
{

// ========== 步骤 4-1：瞄准器初始化 ==========

Aimer::Aimer(const std::string & config_path)
: left_yaw_offset_(std::nullopt), right_yaw_offset_(std::nullopt)
{
  auto yaml = YAML::LoadFile(config_path);

  /**
   * 读取瞄准参数（角度参数从度转为弧度）：
   *
   * yaw_offset：PnP 解算的系统性 yaw 偏移校准值。
   *   由于枪管与光心之间存在机械安装公差，PnP 解算出的 yaw 可能与实际不符。
   *   通过实弹校准得到该偏移值。正数 = yaw 向右偏。
   *
   * pitch_offset：同理的 pitch 偏移校准值。
   *   正数 = 枪口往上抬（因为准星比实际弹着点低）。
   *
   * comming_angle / leaving_angle：目标朝向判定阈值。
   *   用于小陀螺模式下判断装甲板是"正在进入视野"还是"正在离开视野"。
   *
   * decision_speed：高低速切换阈值。
   *   当目标旋转角速度 > decision_speed 时，认为目标处于高速小陀螺模式。
   *
   * high_speed_delay_time / low_speed_delay_time：
   *   补偿从检测到决策的时间延迟（包括图像传输、推理、通信等）。
   *   高速模式下使用更短的延迟（因为目标转动快，需要更实时）。
   */
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;        // degree → rad
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;    // degree → rad
  comming_angle_ = yaml["comming_angle"].as<double>() / 57.3;  // degree → rad
  leaving_angle_ = yaml["leaving_angle"].as<double>() / 57.3;  // degree → rad
  high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>();
  low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();
  decision_speed_ = yaml["decision_speed"].as<double>();

  // 可选参数：左右偏移射击模式
  if (yaml["left_yaw_offset"].IsDefined() && yaml["right_yaw_offset"].IsDefined()) {
    left_yaw_offset_ = yaml["left_yaw_offset"].as<double>() / 57.3;
    right_yaw_offset_ = yaml["right_yaw_offset"].as<double>() / 57.3;
    tools::logger()->info("[Aimer] successfully loading shootmode");
  }
}

// ========== 步骤 4-2：瞄准决策（普通模式） ==========

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  bool to_now)
{
  /**
   * ## 瞄准决策流程
   *
   * 1. 取第一个目标（当前只跟踪一个目标）
   * 2. 根据目标角速度决定决策延迟（高速/低速）
   * 3. 预测目标到 (当前时间 + 延迟) 时刻的状态
   * 4. 选择最佳瞄准点
   * 5. 弹道解算 + 飞行时间迭代求解
   * 6. 返回 yaw/pitch 指令
   */

  if (targets.empty()) return {false, false, 0, 0};
  auto target = targets.front();

  auto ekf = target.ekf();

  // 根据目标旋转角速度选择决策延迟时间
  double delay_time =
    target.ekf_x()[7] > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  // 子弹速度保护（防止异常值从 CAN 传进来）
  if (bullet_speed < 14) bullet_speed = 23;

  /**
   * ## 时间预测
   *
   * 考虑以下延迟环节：
   * - detector 的处理时间（YOLO 推理 + 后处理）
   * - tracker 的匹配时间
   * - 目标预测时间（aim 函数本身的执行时间可忽略不计）
   *
   * to_now = true：从 image timestamp 预测到 "现在"
   *                 dt = now - timestamp + delay_time
   * to_now = false：固定预测 dt = 5ms 检测耗时 + delay_time
   */
  auto future = timestamp;
  if (to_now) {
    double dt;
    dt = tools::delta_time(std::chrono::steady_clock::now(), timestamp) + delay_time;
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);
  }

  else {
    auto dt = 0.005 + delay_time;
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);
  }

  // ---------- 选择瞄准点 ----------
  auto aim_point0 = choose_aim_point(target);
  debug_aim_point = aim_point0;
  if (!aim_point0.valid) {
    return {false, false, 0, 0};
  }

  // ---------- 初值弹道解算 ----------
  Eigen::Vector3d xyz0 = aim_point0.xyza.head(3);
  auto d0 = std::sqrt(xyz0[0] * xyz0[0] + xyz0[1] * xyz0[1]);
  tools::Trajectory trajectory0(bullet_speed, d0, xyz0[2]);
  if (trajectory0.unsolvable) {
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d0, xyz0[2]);
    debug_aim_point.valid = false;
    return {false, false, 0, 0};
  }

  // ---------- 飞行时间迭代求解 ----------
  // 目标在子弹飞行时间内也会移动（尤其是高速小陀螺时），
  // 需要在目标预测位置和飞行时间之间迭代收敛。
  //
  // 迭代原理：
  //   1. 预测目标到 (future + t_fly) 的位置
  //   2. 根据新位置重新计算 t_fly
  //   3. 如果 |t_fly_new - t_fly_old| < 1ms，收敛
  //   4. 否则回到步骤 1（最多 10 次）
  bool converged = false;
  double prev_fly_time = trajectory0.fly_time;
  tools::Trajectory current_traj = trajectory0;
  std::vector<Target> iteration_target(10, target);  // 创建 10 个副本用于迭代预测

  for (int iter = 0; iter < 10; ++iter) {
    auto predict_time = future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
    iteration_target[iter].predict(predict_time);

    auto aim_point = choose_aim_point(iteration_target[iter]);
    debug_aim_point = aim_point;
    if (!aim_point.valid) {
      return {false, false, 0, 0};
    }

    Eigen::Vector3d xyz = aim_point.xyza.head(3);
    double d = std::sqrt(xyz.x() * xyz.x() + xyz.y() * xyz.y());
    current_traj = tools::Trajectory(bullet_speed, d, xyz.z());

    if (current_traj.unsolvable) {
      tools::logger()->debug(
        "[Aimer] Unsolvable trajectory in iter {}: speed={:.2f}, d={:.2f}, z={:.2f}", iter + 1,
        bullet_speed, d, xyz.z());
      debug_aim_point.valid = false;
      return {false, false, 0, 0};
    }

    if (std::abs(current_traj.fly_time - prev_fly_time) < 0.001) {
      converged = true;
      break;
    }
    prev_fly_time = current_traj.fly_time;
  }

  // ---------- 计算最终角度 ----------
  Eigen::Vector3d final_xyz = debug_aim_point.xyza.head(3);
  double yaw = std::atan2(final_xyz.y(), final_xyz.x()) + yaw_offset_;
  double pitch = -(current_traj.pitch + pitch_offset_);
  return {true, false, yaw, pitch};
}

// ========== 步骤 4-3：瞄准决策（带射击模式） ==========

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  io::ShootMode shoot_mode, bool to_now)
{
  /**
   * 带射击模式的瞄准决策。
   * 左/右射击模式：通过调整 yaw_offset 来实现子弹偏左/偏右的弹道。
   * 用于"打两发不同位置的弹道夹击同一目标"的战术。
   */
  double yaw_offset;
  if (shoot_mode == io::left_shoot && left_yaw_offset_.has_value()) {
    yaw_offset = left_yaw_offset_.value();
  } else if (shoot_mode == io::right_shoot && right_yaw_offset_.has_value()) {
    yaw_offset = right_yaw_offset_.value();
  } else {
    yaw_offset = yaw_offset_;
  }

  auto command = aim(targets, timestamp, bullet_speed, to_now);
  command.yaw = command.yaw - yaw_offset_ + yaw_offset;

  return command;
}

// ========== 步骤 4-4：瞄准点选择（核心逻辑） ==========

AimPoint Aimer::choose_aim_point(const Target & target)
{
  /**
   * ## 瞄准点选择策略
   *
   * 在 2 个或 4 个虚拟装甲板中选择最佳击打位置。
   *
   * ### 情况 1：未发生装甲板跳变
   * 直接使用当前正在击打的装甲板（用跳变标志判断）。
   *
   * ### 情况 2：发生跳变，但不小陀螺（|omega| < decision_speed）
   * 选择朝向角度最接近目标旋转中心的装甲板。
   * 锁定模式：如果两个装甲板的朝向角度都接近（都 < 60°），
   * 锁定其中一个并持续击打，防止在两个装甲板之间来回切换导致云台抖动。
   *
   * ### 情况 3：小陀螺模式（|omega| > decision_speed）
   * 选择"正在进入视野"的装甲板（迎击面）。
   * 判断依据：根据旋转方向（omega 的正负）和装甲板的 delta_angle：
   *   - omega > 0（逆时针旋转）：选择 delta_angle > leaving_angle 的装甲板
   *   - omega < 0（顺时针旋转）：选择 delta_angle < -leaving_angle 的装甲板
   * 这样选择的原因：正在进入视野的装甲板暴露时间更长，命中概率更高。
   */

  Eigen::VectorXd ekf_x = target.ekf_x();
  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  auto armor_num = armor_xyza_list.size();

  // 如果装甲板未发生过跳变，则只有当前装甲板的位置已知，直接使用
  if (!target.jumped) return {true, armor_xyza_list[0]};

  // 计算目标旋转中心的球坐标 yaw 角
  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  // 计算每个装甲板的朝向与目标旋转中心方向的夹角（delta_angle）
  // delta_angle 接近 0 → 该装甲板正对目标旋转中心
  // |delta_angle| 接近 π/2 → 装甲板在侧面
  std::vector<double> delta_angle_list;
  for (int i = 0; i < armor_num; i++) {
    auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    delta_angle_list.emplace_back(delta_angle);
  }

  // ---------- 情况：不转小陀螺 ----------
  if (std::abs(target.ekf_x()[8]) <= 2 && target.name != ArmorName::outpost) {
    // 筛选出在可射击范围内的装甲板（|delta_angle| < 60°）
    std::vector<int> id_list;
    for (int i = 0; i < armor_num; i++) {
      if (std::abs(delta_angle_list[i]) > 60 / 57.3) continue;
      id_list.push_back(i);
    }

    if (id_list.empty()) {
      tools::logger()->warn("Empty id list!");
      return {false, armor_xyza_list[0]};
    }

    // 锁定模式：防止在两个都呈 45 度的装甲板之间来回切换
    if (id_list.size() > 1) {
      int id0 = id_list[0], id1 = id_list[1];
      if (lock_id_ != id0 && lock_id_ != id1)
        lock_id_ = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;

      return {true, armor_xyza_list[lock_id_]};
    }

    lock_id_ = -1;
    return {true, armor_xyza_list[id_list[0]]};
  }

  // ---------- 情况：小陀螺模式 ----------
  double coming_angle, leaving_angle;
  if (target.name == ArmorName::outpost) {
    coming_angle = 70 / 57.3;
    leaving_angle = 30 / 57.3;
  } else {
    coming_angle = comming_angle_;
    leaving_angle = leaving_angle_;
  }

  // 在小陀螺时，一侧的装甲板不断出现，另一侧的装甲板不断消失，
  // "正在进入视野"的装甲板被打中的概率更高，因为停留时间更长。
  for (int i = 0; i < armor_num; i++) {
    if (std::abs(delta_angle_list[i]) > coming_angle) continue;
    if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle) return {true, armor_xyza_list[i]};
    if (ekf_x[7] < 0 && delta_angle_list[i] > -leaving_angle) return {true, armor_xyza_list[i]};
  }

  return {false, armor_xyza_list[0]};
}

}  // namespace auto_aim
