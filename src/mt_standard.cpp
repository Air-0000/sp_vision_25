/**
 * @file mt_standard.cpp
 * @brief 步兵标准主程序（多线程流水线版本）
 * @details 多线程流水线版本的入口程序，将检测和决策分离到不同线程：
 *          检测线程：相机采集 → 预处理 → OpenVINO 异步推理 → 推入队列
 *          主线程：从队列弹出结果 → PnP 解算 → EKF 跟踪 → 决策线程下发
 *
 * ## 线程架构
 * ```
 * 检测线程（独立 std::thread）：
 *   camera.read() → detector.push() → OpenVINO async infer → queue.push()
 *
 * 主线程（main thread）：
 *   detector.debug_pop() → solver → tracker → commandgener.push() → 决策线程 500Hz
 * ```
 *
 * 这种流水线架构的优势：
 * 1. 相机采集和 YOLO 推理可以在 GPU 上并行执行
 * 2. 决策线程以 500Hz 独立运行，不依赖检测帧率
 * 3. 检测线程的 queue 深度（16 帧）可以吸收推理延迟的波动
 *
 * @note 决策线程（CommandGener）以 ~500Hz 独立运行，
 *       每次收到新的检测结果就重新计算一次指令。
 *       即使检测线程因为某种原因卡住，决策线程仍然用最新可用的结果输出。
 */

#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/dm_imu/dm_imu.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml 配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  // ---------- 解析命令行参数 ----------
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // ---------- 初始化 ----------
  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::Camera camera(config_path);
  io::CBoard cboard(config_path);

  // 多线程检测器（内部维护推理线程和队列）
  auto_aim::multithread::MultiThreadDetector detector(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  // 打符模块（能量机关）
  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::SmallTarget buff_small_target;  // 小能量机关目标
  auto_buff::BigTarget buff_big_target;      // 大能量机关目标
  auto_buff::Aimer buff_aimer(config_path);

  // 决策线程（500Hz，独立线程）
  auto_aim::multithread::CommandGener commandgener(shooter, aimer, cboard, plotter);

  // ---------- 工作模式 ----------
  std::atomic<io::Mode> mode{io::Mode::idle};
  auto last_mode{io::Mode::idle};

  // ========== 启动检测线程 ==========
  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!exiter.exit()) {
      if (mode.load() == io::Mode::auto_aim) {
        camera.read(img, t);
        detector.push(img, t);  // 预处理 + 启动异步推理 → 入队
      } else
        continue;
    }
  });

  // ========== 主线程（决策 + 接口通信） ==========
  while (!exiter.exit()) {
    mode = cboard.mode;

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", io::MODES[mode]);
      last_mode = mode.load();
    }

    /// ========== 自瞄模式 ==========
    if (mode.load() == io::Mode::auto_aim) {
      // 从队列中取出最近一帧的检测结果
      auto [img, armors, t] = detector.debug_pop();
      Eigen::Quaterniond q = cboard.imu_at(t - 1ms);  // 获取对齐的姿态

      solver.set_R_gimbal2world(q);

      // EKF 跟踪（检测结果 → 目标状态估计）
      auto targets = tracker.track(armors, t);

      // 将结果推送给决策线程（500Hz 独立下发）
      commandgener.push(targets, t, cboard.bullet_speed, ypr);
    }

    /// ========== 打符模式（能量机关） ==========
    else if (mode.load() == io::Mode::small_buff || mode.load() == io::Mode::big_buff) {
      cv::Mat img;
      Eigen::Quaterniond q;
      std::chrono::steady_clock::time_point t;

      camera.read(img, t);
      q = cboard.imu_at(t - 1ms);

      buff_solver.set_R_gimbal2world(q);

      auto power_runes = buff_detector.detect(img);  // 检测能量机关
      buff_solver.solve(power_runes);               // PnP 解算

      io::Command buff_command;
      if (mode.load() == io::Mode::small_buff) {
        buff_small_target.get_target(power_runes, t);
        auto target_copy = buff_small_target;
        buff_command = buff_aimer.aim(target_copy, t, cboard.bullet_speed, true);
      } else if (mode.load() == io::Mode::big_buff) {
        buff_big_target.get_target(power_runes, t);
        auto target_copy = buff_big_target;
        buff_command = buff_aimer.aim(target_copy, t, cboard.bullet_speed, true);
      }
      cboard.send(buff_command);

    } else
      continue;
  }

  detect_thread.join();  // 等待检测线程退出

  return 0;
}
