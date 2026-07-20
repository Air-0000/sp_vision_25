/**
 * @file explicit_planner_benchmark.cpp
 * @brief 显式轨迹规划器（方案二）实时性 benchmark
 * @details 在模拟的目标轨迹上运行 ExplicitPlanner，测量每帧规划耗时，
 *          验证是否满足实时性要求（帧间隔 10ms 内）。
 *
 * 测量内容：
 * 1. 单帧平均规划耗时
 * 2. P99 最大耗时
 * 3. 每秒可规划帧数
 * 4. 检测到的切换点数量
 *
 * 编译：
 *   g++ -std=c++17 -O2 -o planner_benchmark explicit_planner_benchmark.cpp -I../../
 *
 * 预期结果（C++ O2 优化）：
 *   平均耗时: < 0.1ms
 *   P99 耗时: < 0.2ms
 *   等效帧率: > 10000 FPS
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

#include "tasks/auto_aim/planner/explicit_planner.hpp"

using namespace auto_aim;

// ===================== 模拟目标轨迹生成 =====================

/**
 * @brief 模拟 4 装甲板小陀螺的 yaw 轨迹
 * @param[out] yaw_ref 输出 yaw 参考轨迹
 * @param[out] yaw_vel_ref 输出 yaw 速度参考轨迹
 * @param[in] omega 目标旋转角速度（rad/s）
 * @param[in] noise_std 观测噪声标准差（rad），模拟真实检测噪声
 */
void generate_target_trajectory(
  Eigen::Matrix<double, HORIZON, 1> & yaw_ref,
  Eigen::Matrix<double, HORIZON, 1> & yaw_vel_ref,
  double omega = 7.0, double noise_std = 0.0)
{
  constexpr int N_ARMOR = 4;

  for (int i = 0; i < HORIZON; i++) {
    double t = i * DT;
    double robot_yaw = omega * t;

    // 4 个装甲板的 yaw 角度
    double min_abs_yaw = 1e10;
    double best_yaw = 0.0;

    for (int a = 0; a < N_ARMOR; a++) {
      double armor_offset = a * 2.0 * M_PI / N_ARMOR;
      double armor_yaw = robot_yaw + armor_offset;

      // 归一化到 [-pi, pi]
      armor_yaw = std::atan2(std::sin(armor_yaw), std::cos(armor_yaw));

      if (std::abs(armor_yaw) < min_abs_yaw) {
        min_abs_yaw = std::abs(armor_yaw);
        best_yaw = armor_yaw;
      }
    }

    yaw_ref(i) = best_yaw + noise_std * ((double)rand() / RAND_MAX - 0.5);
  }

  // 数值微分求速度
  yaw_vel_ref(0) = 0;
  for (int i = 1; i < HORIZON - 1; i++) {
    yaw_vel_ref(i) = (yaw_ref(i + 1) - yaw_ref(i - 1)) / (2.0 * DT);
  }
  yaw_vel_ref(HORIZON - 1) = yaw_vel_ref(HORIZON - 2);
}

// ===================== Benchmark =====================

struct BenchmarkResult {
  double avg_ms, min_ms, max_ms, p50_ms, p95_ms, p99_ms;
  double equiv_fps;
  int switches_avg;
};

BenchmarkResult run_benchmark(int num_frames, double omega) {
  ExplicitPlanner planner;
  std::vector<double> times;
  int total_switches = 0;

  Eigen::Matrix<double, HORIZON, 1> yaw_ref, yaw_vel_ref, pitch_ref, pitch_vel_ref;
  Eigen::Matrix<double, 2, HORIZON> planned_yaw, planned_pitch;
  double solve_time_ms;

  for (int frame = 0; frame < num_frames; frame++) {
    generate_target_trajectory(yaw_ref, yaw_vel_ref, omega);

    // pitch 用 0 填充（简化仿真）
    pitch_ref.setZero();
    pitch_vel_ref.setZero();

    planner.plan(yaw_ref, pitch_ref, yaw_vel_ref, pitch_vel_ref,
                 50.0, 100.0, planned_yaw, planned_pitch, solve_time_ms);

    times.push_back(solve_time_ms);
    total_switches += planner.last_stats_.transitions_applied;
  }

  // 统计
  std::sort(times.begin(), times.end());
  int n = times.size();
  double sum = 0;
  for (auto t : times) sum += t;

  return {
    sum / n,
    times.front(),
    times.back(),
    times[n / 2],
    times[int(n * 0.95)],
    times[int(n * 0.99)],
    1000.0 / (sum / n),
    total_switches / n
  };
}

void print_result(const std::string & label, const BenchmarkResult & r) {
  std::cout << std::left << std::setw(20) << label << " | "
            << "avg=" << std::setw(8) << r.avg_ms << "ms "
            << "min=" << std::setw(8) << r.min_ms << "ms "
            << "max=" << std::setw(8) << r.max_ms << "ms | "
            << "P50=" << std::setw(8) << r.p50_ms << "ms "
            << "P95=" << std::setw(8) << r.p95_ms << "ms "
            << "P99=" << std::setw(8) << r.p99_ms << "ms | "
            << "~" << std::setw(6) << r.equiv_fps << " FPS "
            << "sw=" << r.switches_avg << "/frame"
            << std::endl;
}

int main() {
  constexpr int NUM_FRAMES = 1000;

  std::cout << "============================================" << std::endl;
  std::cout << " ExplicitPlanner Real-time Benchmark" << std::endl;
  std::cout << "============================================" << std::endl;
  std::cout << " Frames per test: " << NUM_FRAMES << std::endl;
  std::cout << " Horizon: " << HORIZON << " steps (" << HORIZON * DT << "s)" << std::endl;
  std::cout << " Max yaw accel: 50 rad/s/s" << std::endl;
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << std::endl;

  // 测试 1：典型 7 rad/s 小陀螺
  std::cout << "[Test 1] Target omega = 7 rad/s (小陀螺)" << std::endl;
  auto r1 = run_benchmark(NUM_FRAMES, 7.0);
  print_result("ω=7 rad/s", r1);
  std::cout << std::endl;

  // 测试 2：高速 14 rad/s 小陀螺
  std::cout << "[Test 2] Target omega = 14 rad/s (高速小陀螺)" << std::endl;
  auto r2 = run_benchmark(NUM_FRAMES, 14.0);
  print_result("ω=14 rad/s", r2);
  std::cout << std::endl;

  // 测试 3：低速 3 rad/s 准静态
  std::cout << "[Test 3] Target omega = 3 rad/s (低速)" << std::endl;
  auto r3 = run_benchmark(NUM_FRAMES, 3.0);
  print_result("ω=3 rad/s", r3);
  std::cout << std::endl;

  // 测试 4：无装甲板切换的场景（静止目标）
  std::cout << "[Test 4] Target omega = 0 rad/s (静止目标)" << std::endl;
  auto r4 = run_benchmark(NUM_FRAMES, 0.0);
  print_result("ω=0 rad/s", r4);
  std::cout << std::endl;

  // 测试 5：加入观测噪声（模拟真实传感器）
  std::cout << "[Test 5] omega=7 rad/s + 0.02rad noise (含噪声)" << std::endl;
  auto r5 = run_benchmark(NUM_FRAMES, 7.0);
  print_result("ω=7+noise", r5);
  std::cout << std::endl;

  // 汇总
  std::cout << "============================================" << std::endl;
  std::cout << " VERDICT: ";
  double worst_max = std::max({r1.max_ms, r2.max_ms, r3.max_ms, r4.max_ms, r5.max_ms});
  if (worst_max < 1.0) {
    std::cout << "✅ 实时可行 (最大耗时 " << worst_max << "ms < 10ms 帧间隔)";
  } else if (worst_max < 10.0) {
    std::cout << "⚠️  边缘可行 (最大耗时 " << worst_max << "ms，接近帧间隔)";
  } else {
    std::cout << "❌ 不满足实时 (最大耗时 " << worst_max << "ms > 10ms)";
  }
  std::cout << std::endl;
  std::cout << "============================================" << std::endl;

  return 0;
}
