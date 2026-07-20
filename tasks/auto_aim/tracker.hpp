/**
 * @file tracker.hpp
 * @brief 装甲板跟踪器接口（状态机 + 目标匹配）
 * @details 本文件属于 auto_aim 模块的跟踪子模块，定义了 Tracker 类。
 *          Tracker 负责将每帧检测到的 Armor 列表与已有的 Target（EKF 目标）进行匹配，
 *          并管理目标的生存周期（创建、跟踪、丢失）。
 *
 * 管线位置：步骤 3（跟踪）：Detector/YOLO 输出 Armor 列表 → Solver 填充 3D 位姿 →
 *          Tracker::track() 更新 Target → Aimer 读取 Target 决策
 *
 * 状态机：
 * ├── "lost"（丢失）：无目标状态
 * │   └── detect_count < min_detect_count → "lost"
 * │   └── detect_count >= min_detect_count → "detected"
 * ├── "detected"（暂态检测到）：目标刚出现，需要连续确认
 * │   └── temp_lost_count > max_temp_lost_count → "lost"
 * │   └── 持续匹配到 → "tracking"
 * └── "tracking"（稳定跟踪）：正常跟踪状态
 *     └── temp_lost_count > max_temp_lost_count → "lost"
 *     └── 持续匹配到 → "tracking"
 *
 * @note 前哨站（outpost）有更宽松的丢失容忍（outpost_max_temp_lost_count），
 *       因为前哨站距离通常较远，偶尔漏检属于正常情况。
 */

#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <string>

#include "armor.hpp"
#include "solver.hpp"
#include "target.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{

/**
 * @brief 目标跟踪器
 * @details 管理单个敌方机器人的跟踪生命周期。核心职责：
 *          1. 每帧将检测到的 Armor 列表与当前 Target 匹配
 *          2. 匹配成功 → Target::update()
 *          3. 匹配失败 → 累加 temp_lost_count
 *          4. 新目标出现 → 创建新的 Target（构造 EKF）
 *          5. 状态机管理跟踪状态（lost/detected/tracking）
 *
 * @note 当前实现只维护单个 Target（跟踪最近的/最高优先级的目标）。
 *       多目标跟踪（并行跟踪多个敌方机器人）不在本版本支持范围内。
 */
class Tracker
{
public:
  /**
   * @brief 构造函数
   * @param[in] config_path YAML 配置文件路径
   * @param[in] solver Solver 实例（用于创建新 Target 时获取 3D 信息）
   */
  Tracker(const std::string & config_path, Solver & solver);

  /**
   * @brief 获取当前状态机状态
   * @return 状态字符串："lost" / "detected" / "tracking"
   */
  std::string state() const;

  /**
   * @brief 核心跟踪函数（步骤 3-1：目标匹配与更新）
   * @param[in/out] armors 当前帧检测到的装甲板列表
   * @param[in] t 当前时间戳
   * @param[in] use_enemy_color 是否使用敌方颜色过滤
   * @return 更新后的 Target 列表（最多一个目标）
   */
  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

  /**
   * @brief 带全向感知结果的重载（用于哨兵机器人）
   * @param[in] detection_queue 全向感知的检测结果队列
   * @param[in/out] armors 当前帧的装甲板列表
   * @param[in] t 当前时间戳
   * @param[in] use_enemy_color 是否使用敌方颜色过滤
   * @return (全向感知决策结果, Target 列表) 的元组
   */
  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
    std::chrono::steady_clock::time_point t, bool use_enemy_color = true);

private:
  Solver & solver_;             ///< Solver 引用（用于创建新 Target 时的位姿解算）
  Color enemy_color_;           ///< 敌方颜色（red/blue），从配置文件读取

  // ---------- 状态机参数 ----------
  int min_detect_count_;        ///< 连续检测到多少帧后才认为目标有效（暂态检测 → 稳定跟踪）
  int max_temp_lost_count_;     ///< 最多允许丢失多少帧（超过则标记为丢失）
  int detect_count_;            ///< 当前连续检测到的帧数计数
  int temp_lost_count_;         ///< 当前连续丢失的帧数计数
  int outpost_max_temp_lost_count_;  ///< 前哨站丢失容忍帧数（比普通目标大）
  int normal_temp_lost_count_;       ///< 普通目标丢失容忍帧数（等于 max_temp_lost_count_）

  // ---------- 运行时状态 ----------
  std::string state_, pre_state_;  ///< 当前状态和上一帧状态
  Target target_;                  ///< 当前正在跟踪的 Target 实例
  std::chrono::steady_clock::time_point last_timestamp_;  ///< 上次跟踪时间戳
  ArmorPriority omni_target_priority_;  ///< 全向感知给出的目标优先级

  /**
   * @brief 状态机转换
   * @param[in] found 当前帧是否匹配到了目标
   */
  void state_machine(bool found);

  /**
   * @brief 选择最佳目标装甲板（从多个检测到的装甲板中选一个）
   * @param[in/out] armors 候选装甲板列表
   * @param[in] t 当前时间戳
   * @return 是否成功设置了目标（true = 找到目标）
   */
  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  /**
   * @brief 用当前帧的装甲板更新已有 Target
   * @param[in/out] armors 候选装甲板列表
   * @param[in] t 当前时间戳
   * @return 是否匹配成功（true = 匹配并更新）
   */
  bool update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP
