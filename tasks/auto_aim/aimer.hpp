/**
 * @file aimer.hpp
 * @brief 瞄准决策器接口
 * @details 本文件属于 auto_aim 模块的决策子模块，定义了 Aimer 类。
 *          Aimer 负责根据 EKF 跟踪到的目标状态（Target），计算云台的瞄准角度（yaw/pitch）
 *          以及最终的下发指令（Command）。
 *
 * 管线位置：步骤 4（瞄准决策）：Tracker 输出 Target → Aimer::aim() → 返回 io::Command
 *          → CBoard::send() 通过 CAN/串口发送到下位机
 *
 * 核心功能：
 * 1. 弹道解算：根据目标距离和子弹速度，计算子弹飞行时间和需要的射击仰角（pitch）
 * 2. 目标预测：考虑检测延迟 + 决策延迟 + 弹道飞行时间，预测目标未来位置
 * 3. 瞄准点选择：从多个虚拟装甲板中选择最佳击打位置（考虑旋转状态、朝向角）
 * 4. 弹道飞行时间迭代求解：在目标位置和飞行时间之间迭代，收敛到时域一致的解
 *
 * @note Aimer 有新旧两种模式：传统 Aimer（本文件）使用分段决策逻辑，
 *       Planner（tasks/auto_aim/planner/）使用基于 MPC 的轨迹规划器，
 *       两者通过不同的 main 函数选择。
 */

#ifndef AUTO_AIM__AIMER_HPP
#define AUTO_AIM__AIMER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>

#include "io/cboard.hpp"
#include "io/command.hpp"
#include "target.hpp"

namespace auto_aim
{

/**
 * @brief 瞄准点信息结构体
 * @details 记录 Aimer 选择的瞄准点的位置和有效性。
 *          xyza = (x, y, z, angle) 世界坐标系下的瞄准点位置和装甲板朝向
 */
struct AimPoint
{
  bool valid;              ///< 瞄准点是否有效（false = 没有可击打的位置）
  Eigen::Vector4d xyza;   ///< 瞄准点的世界坐标和朝向 (x, y, z, angle)
};

/**
 * @brief 传统瞄准决策器
 * @details 使用分段逻辑进行瞄准决策：
 *          1. 弹道解算（Trajectory 类）：计算子弹飞行时间和 pitch 角
 *          2. 飞行时间迭代：收敛求解一致的 (目标预测位置, 飞行时间)
 *          3. 瞄准点选择：根据目标是否小陀螺、装甲板朝向角等选择最佳装甲板
 *
 *          与传统 Aimer 相比，Planner（MPC 版本）用轨迹规划替代了分段逻辑。
 *
 * @note 瞄准点选择逻辑（choose_aim_point）是核心，包含锁定模式和低/高速模式，
 *       目的是防止在两个夹角相近的装甲板之间来回切换导致的云台抖动。
 */
class Aimer
{
public:
  AimPoint debug_aim_point;  ///< 调试用：上一帧选择的瞄准点

  /**
   * @brief 构造函数
   * @param[in] config_path YAML 配置文件路径
   */
  explicit Aimer(const std::string & config_path);

  /**
   * @brief 瞄准决策（步骤 4-1：计算瞄准角度）
   * @param[in] targets 当前跟踪到的目标列表（通常只有一个）
   * @param[in] timestamp 当前时间戳
   * @param[in] bullet_speed 子弹速度（米/秒），通常由测速仪实测
   * @param[in] to_now 是否考虑从检测到决策的延迟（true = 实时模式）
   * @return 包含 yaw/pitch 角度和射击信号的下发指令
   */
  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    bool to_now = true);

  /**
   * @brief 带射击模式参数的瞄准决策（支持左/右偏差射击）
   * @param[in] targets 目标列表
   * @param[in] timestamp 时间戳
   * @param[in] bullet_speed 子弹速度
   * @param[in] shoot_mode 射击模式（左射/右射/正常）
   * @param[in] to_now 实时模式标志
   * @return 下发指令
   */
  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    io::ShootMode shoot_mode, bool to_now = true);

private:
  // ---------- 标定参数 ----------
  double yaw_offset_;                      ///< yaw 方向校准偏移（度→弧度），补偿 PnP 系统偏差
  std::optional<double> left_yaw_offset_,   ///< 左射模式 yaw 偏移（补偿枪管安装偏差）
                     right_yaw_offset_;     ///< 右射模式 yaw 偏移
  double pitch_offset_;                    ///< pitch 方向校准偏移（度→弧度），补偿枪口与相机光心垂直偏差

  // ---------- 决策参数 ----------
  double comming_angle_;                   ///< 目标接近判定角度（弧度），超过此角度认为目标正在接近
  double leaving_angle_;                   ///< 目标远离判定角度（弧度），低于此角度认为目标正在远离
  double lock_id_ = -1;                    ///< 当前锁定的装甲板 ID（-1 = 未锁定）

  // ---------- 延迟参数 ----------
  double high_speed_delay_time_;           ///< 高速模式（目标角速度快）下的决策延迟（秒）
  double low_speed_delay_time_;            ///< 低速模式下的决策延迟（秒）
  double decision_speed_;                  ///< 速度阈值（弧度/秒），用于区分高/低速模式

  /**
   * @brief 从目标的多个虚拟装甲板中选择最佳瞄准点（步骤 4-2）
   * @param[in] target 当前目标
   * @return 选择的瞄准点（含位置和朝向）
   * @details 核心选择逻辑：
   *          - 未发生装甲板跳变 → 直接使用当前装甲板
   *          - 发生跳变但不小陀螺 → 选择朝向角度差最小的装甲板（锁定模式防抖）
   *          - 小陀螺模式 → 根据旋转方向选择"正在进入视野"的装甲板
   */
  AimPoint choose_aim_point(const Target & target);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__AIMER_HPP
