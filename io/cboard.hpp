/**
 * @file cboard.hpp
 * @brief 下位机（C 板 / STM32）通信接口
 * @details 本文件属于 io 模块的核心硬件通信接口，负责通过 SocketCAN 与 RoboMaster 开发板 C 型
 *          （STM32F407）进行双向通信：
 *          - 接收：云台姿态四元数（IMU 数据）、子弹速度、控制模式
 *          - 发送：瞄准指令（yaw/pitch 角度 + 击发信号）
 *
 * 通信协议（CAN 总线）：
 *   帧 ID 0x100（quaternion_canid）：接收 IMU 四元数
 *   帧 ID 0x101（bullet_speed_canid）：接收子弹速度
 *   帧 ID 0x0FF（send_canid）：发送瞄准指令
 *
 * 管线位置：步骤 0（初始化 + 接收）和步骤 5（下发）：
 *   主循环前：CBoard 构造函数打开 CAN 设备，启动接收线程
 *   每帧：cboard.imu_at(t) 获取与图像时间戳对齐的四元数
 *   决策后：cboard.send(command) 通过 CAN 发送指令
 *
 * 时间戳对齐：
 *   CAN 接收到的 IMU 数据带有接收到的时间戳（std::chrono::steady_clock::now()），
 *   图像数据同样带有采集完成的时间戳。
 *   imu_at(t) 通过线性插值（在两个最近的 IMU 数据之间插值）获得与图像时间对齐的四元数。
 *   这样可以消除 IMU 和图像之间的时间差，保证 PnP 解算时使用的姿态是准确的。
 *
 * @warning queue_ 必须在 can_ 之前声明，因为 can_ 的回调函数会访问 queue_，
 *          如果 can_ 先初始化，构造函数中 can_ 的回调可能会在 queue_ 初始化之前被调用。
 */

#ifndef IO__CBOARD_HPP
#define IO__CBOARD_HPP

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "io/command.hpp"
#include "io/socketcan.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{

/**
 * @brief 自瞄系统工作模式枚举
 * @details 通过 STM32 下位机的遥控器开关或上位机指令切换。
 *          idle = 空闲（自瞄不介入），auto_aim = 自瞄模式，
 *          small_buff = 小能量机关（打符），big_buff = 大能量机关，
 *          outpost = 前哨站模式
 */
enum Mode
{
  idle,        ///< 空闲模式，自瞄不输出任何指令
  auto_aim,    ///< 自瞄模式，正常跟踪敌方机器人
  small_buff,  ///< 小能量机关（打符模式）
  big_buff,    ///< 大能量机关（打符模式）
  outpost      ///< 前哨站模式（特殊瞄准策略）
};
const std::vector<std::string> MODES = {"idle", "auto_aim", "small_buff", "big_buff", "outpost"};

/**
 * @brief 射击模式枚举（哨兵专有）
 * @details 哨兵兵种支持左右双枪管交替射击。
 *          left_shoot = 左枪管射击，right_shoot = 右枪管射击，
 *          both_shoot = 双管齐射。
 */
enum ShootMode
{
  left_shoot,   ///< 左枪管
  right_shoot,  ///< 右枪管
  both_shoot    ///< 双管齐射
};
const std::vector<std::string> SHOOT_MODES = {"left_shoot", "right_shoot", "both_shoot"};

/**
 * @brief C 板通信管理器
 * @details 封装了与 STM32 开发板 C 型（RoboMaster 官方电控板）的所有通信逻辑。
 *
 * 功能：
 * 1. CAN 总线初始化（SocketCAN，设备名如 "can0"）
 * 2. 接收线程：持续监听 CAN 总线，解析 IMU 四元数、子弹速度、控制模式
 * 3. IMU 插值：根据时间戳线性插值四元数，与图像帧对齐
 * 4. 指令发送：将 Command 结构体打包为 CAN 帧发送
 *
 * @note 时间戳对齐是自瞄精度的关键。IMU 的采样率（通常 1kHz）远高于相机帧率（~60FPS），
 *       通过插值可以获得每个图像帧对应时刻的精确云台姿态。
 */
class CBoard
{
public:
  // ---- 公开状态（由 CAN 接收线程更新） ----
  double bullet_speed;   ///< 子弹初速度（米/秒），由弹丸测速仪测量后通过 CAN 发送
  Mode mode;             ///< 当前工作模式，由电控通过遥控器切换
  ShootMode shoot_mode;  ///< 当前射击模式（哨兵）
  double ft_angle;       ///< 摩擦轮角度（无人机专有参数）

  /**
   * @brief 构造函数：打开 CAN 设备并启动接收线程
   * @param[in] config_path YAML 配置文件路径
   */
  CBoard(const std::string & config_path);

  /**
   * @brief 获取与图像时间戳对齐的云台姿态四元数
   * @param[in] timestamp 图像采集完成的时间戳
   * @return 通过线性插值得到的云台姿态四元数
   * @details 在 IMU 数据流中找到 timestamp 前后的两个数据点，
   *          进行球面线性插值（Slerp）得到对齐后的四元数。
   */
  Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);

  /**
   * @brief 发送瞄准指令（通过 CAN 总线）
   * @param[in] command 包含 yaw/pitch 目标角度和击发信号的指令
   * @details 将 Command 结构体打包为 8 字节 CAN 帧发送。
   *          yaw 和 pitch 以 float32 格式发送，单位：弧度。
   */
  void send(Command command) const;

private:
  /**
   * @brief IMU 数据缓存结构体
   */
  struct IMUData
  {
    Eigen::Quaterniond q;              ///< 四元数（单位四元数，w+xi+yj+zk）
    std::chrono::steady_clock::time_point timestamp;  ///< 接收到的时间戳
  };

  tools::ThreadSafeQueue<IMUData> queue_;  ///< IMU 数据队列（生产者：CAN 回调，消费者：imu_at）
  SocketCAN can_;                          ///< SocketCAN 设备
  IMUData data_ahead_;                     ///< 时间戳在前的 IMU 数据（插值用）
  IMUData data_behind_;                    ///< 时间戳在后的 IMU 数据（插值用）

  // ---------- CAN 帧 ID（可配置） ----------
  int quaternion_canid_;      ///< 接收四元数的 CAN 帧 ID（默认 0x100）
  int bullet_speed_canid_;    ///< 接收子弹速度的 CAN 帧 ID（默认 0x101）
  int send_canid_;            ///< 发送指令的 CAN 帧 ID（默认 0x0FF）

  /**
   * @brief CAN 帧接收回调
   * @param[in] frame 接收到的 CAN 帧
   * @details 根据帧 ID 解析数据：
   *          0x100：四元数（4×float32）
   *          0x101：子弹速度（float32）
   *          其他帧：模式切换等信息
   */
  void callback(const can_frame & frame);

  /**
   * @brief 从 YAML 配置文件读取 CAN 相关参数
   * @param[in] config_path YAML 配置文件路径
   * @return CAN 设备名（如 "can0"）
   */
  std::string read_yaml(const std::string & config_path);
};

}  // namespace io

#endif  // IO__CBOARD_HPP
