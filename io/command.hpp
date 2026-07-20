/**
 * @file command.hpp
 * @brief 下位机下发指令数据结构
 * @details 定义从视觉小电脑发送到 STM32 下位机的控制指令格式。
 *          包含云台目标角度和击发信号。
 *
 * 管线位置：步骤 5（CAN 下发）的最终输出格式：
 *          Aimer/Planner → Command → CBoard::send() → CAN 总线 → STM32
 *
 * CAN 通信协议（标准帧，11 位 ID）：
 *   帧 ID 0xFF（send_canid，可配置）
 *   数据段 8 字节：
 *     Byte 0-3：yaw 目标角度（float32，弧度），小端
 *     Byte 4-7：pitch 目标角度（float32，弧度），小端
 *     Bit 标志位（单独帧或嵌入在数据帧的特定字节）
 *
 * @note STM32 收到指令后，运行 PID 控制器驱动云台电机到达目标角度，
 *       同时根据 shoot 标志控制发射机构的摩擦轮和扳机。
 */

#ifndef IO__COMMAND_HPP
#define IO__COMMAND_HPP

namespace io
{

/**
 * @brief 下位机控制指令
 * @details 自瞄系统每帧计算出的最终指令，通过 CAN 总线发送给 STM32。
 *          包含云台角度目标值、击发信号等。
 */
struct Command
{
  bool control;      ///< 控制使能标志：true = 自瞄接管云台控制，false = 释放控制权给操作手
  bool shoot;        ///< 击发信号：true = 开火，false = 停火
  double yaw;        ///< 云台 yaw 目标角度（弧度），相对于云台初始位置
  double pitch;      ///< 云台 pitch 目标角度（弧度），抬头为正
  double horizon_distance = 0;  ///< 水平距离（米），仅无人机（UAV）兵种使用，用于高度补偿
};

}  // namespace io

#endif  // IO__COMMAND_HPP
