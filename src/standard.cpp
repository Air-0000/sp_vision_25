/**
 * @file standard.cpp
 * @brief 步兵/英雄标准主程序（单线程版本）
 * @details 这是 RoboMaster 自瞄系统的标准入口程序，适用于步兵和英雄兵种。
 *          采用单线程同步处理模式：采集 → 检测 → 跟踪 → 瞄准 → 下发。
 *
 * ## 主循环流程
 *   每帧：
 *   1. camera.read(img, t)            → 采集图像和时间戳
 *   2. cboard.imu_at(t - 1ms)         → 获取对齐的云台姿态
 *   3. cboard.mode                    → 获取当前工作模式
 *   4. solver.set_R_gimbal2world(q)   → 更新旋转矩阵
 *   5. detector.detect(img)           → YOLO 检测
 *   6. tracker.track(armors, t)       → EKF 跟踪
 *   7. aimer.aim(targets, t, bs)      → 瞄准决策
 *   8. cboard.send(command)           → CAN 下发
 *
 * 对比：
 *   - standard.cpp：单线程，检测和决策串行执行，代码简单
 *   - mt_standard.cpp：多线程，检测和决策流水线并行，帧率更高
 *
 * @note 使用单线程版本时需要注意检测延迟对整体帧率的影响。
 *       如果 YOLO 推理时间较长（>15ms），建议使用 mt_standard.cpp。
 */

#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

using namespace std::chrono;

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | 位置参数，yaml 配置文件路径 }";

int main(int argc, char * argv[])
{
  // ---------- 解析命令行参数 ----------
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  // ---------- 初始化各模块 ----------
  tools::Exiter exiter;        // 退出信号监听（Ctrl+C）
  tools::Plotter plotter;      // 调试曲线绘制（PlotJuggler 兼容）
  tools::Recorder recorder;    // 视频录制器

  io::CBoard cboard(config_path);   // C 板通信（CAN 总线）
  io::Camera camera(config_path);   // 海康/大恒工业相机

  auto_aim::YOLO detector(config_path, false);  // YOLO 检测器
  auto_aim::Solver solver(config_path);         // PnP 解算器
  auto_aim::Tracker tracker(config_path, solver);  // EKF 跟踪器
  auto_aim::Aimer aimer(config_path);           // 瞄准决策器
  auto_aim::Shooter shooter(config_path);       // 击发决策器

  // ---------- 图像和状态变量 ----------
  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  auto mode = io::Mode::idle;
  auto last_mode = io::Mode::idle;

  // ========== 主循环 ==========
  while (!exiter.exit()) {
    // ---------- 步骤 1：图像采集 ----------
    camera.read(img, t);
    q = cboard.imu_at(t - 1ms);   // 获取 t-1ms 时刻的姿态（补偿 CAN 传输延迟）
    mode = cboard.mode;

    // ---------- 模式切换日志 ----------
    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", io::MODES[mode]);
      last_mode = mode;
    }

    // recorder.record(img, q, t);  // 录制（调试时开启）

    // ---------- 步骤 2：PnP 解算坐标系更新 ----------
    solver.set_R_gimbal2world(q);

    // ---------- 步骤 3：YOLO 检测 ----------
    auto armors = detector.detect(img);

    // ---------- 步骤 4：EKF 跟踪 ----------
    auto targets = tracker.track(armors, t);

    // ---------- 步骤 5：瞄准决策 + 击发决策 + 下发 ----------
    auto command = aimer.aim(targets, t, cboard.bullet_speed);
    cboard.send(command);
  }

  return 0;
}
