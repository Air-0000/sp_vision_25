/**
 * @file tracker.cpp
 * @brief 目标跟踪器实现（状态机 + 目标匹配 + EKF 管理）
 * @details 实现装甲板跟踪状态机，负责将检测装甲板与已有 EKF 目标匹配，
 *          管理目标的创建、更新、丢失生命周期。
 *
 * @note 当前 tracker 有哨兵专用重载版本（带全向感知 DetectionResult 参数），
 *       用于哨兵兵种的主相机 + 全向感知相机的协同目标切换。
 */

#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <tuple>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{

// ========== 步骤 3-1：跟踪器初始化 ==========

Tracker::Tracker(const std::string & config_path, Solver & solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth}
{
  auto yaml = YAML::LoadFile(config_path);
  enemy_color_ = (yaml["enemy_color"].as<std::string>() == "red") ? Color::red : Color::blue;
  min_detect_count_ = yaml["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  outpost_max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  normal_temp_lost_count_ = max_temp_lost_count_;
}

// ========== 步骤 3-2：状态访问 ==========

std::string Tracker::state() const { return state_; }

// ========== 步骤 3-3：核心跟踪流程（普通版本） ==========

std::list<Target> Tracker::track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  /**
   * ## 跟踪主流程（每帧调用一次）
   *
   * 1. 时间检查：如果与上一帧间隔 > 0.1s，认为相机掉线，重置状态
   * 2. 颜色过滤：只保留敌方颜色的装甲板
   * 3. 装甲板排序：靠近图像中心的优先，优先级高的优先
   * 4. 状态机分支：
   *    - "lost"：调用 set_target() 尝试创建新目标
   *    - 其他状态：调用 update_target() 尝试匹配已有目标
   * 5. 状态机转换
   * 6. 发散检测和收敛检测
   * 7. 返回当前 Target（或空列表）
   */

  // ---------- 时间差检查 ----------
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 时间间隔过长，说明可能发生了相机离线或掉帧
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  // ---------- 颜色过滤 ----------
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // ---------- 前哨站顶部装甲板过滤（当前已注释，因为效果不稳定） ----------
  // 前哨站顶部装甲板的法线方向与常规装甲板不同（约 27.5° 俯角），
  // 通过比较两种 pitch 假设下的重投影误差来判断。
  // armors.remove_if([this](const auto_aim::Armor & a) {
  //   return a.name == ArmorName::outpost &&
  //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
  //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
  // });

  // ---------- 装甲板排序 ----------
  // 先按距离图像中心的距离排序（优先选择画面中央的目标）
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO: 应从配置读取
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 再按优先级排序（优先级高 = 数值小 = 排前面）
  armors.sort(
    [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });

  // ---------- 状态机分支 ----------
  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);    // 丢失状态 → 尝试创建新目标
  }

  else {
    found = update_target(armors, t);  // 跟踪中 → 匹配已有目标
  }

  state_machine(found);  // 执行状态转换

  // ---------- 发散检测 ----------
  // 如果 EKF 的半径估计超出合理范围，认为发散，重置到 lost
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {};
  }

  // ---------- NIS 收敛检测 ----------
  // NIS（Normalized Innovation Squared）卡方检验：
  // 如果 recent_nis_failures 中超过 40% 是失败的，说明 EKF 与观测不匹配
  if (
    std::accumulate(
      target_.ekf().recent_nis_failures.begin(), target_.ekf().recent_nis_failures.end(), 0) >=
    (0.4 * target_.ekf().window_size)) {
    tools::logger()->debug("[Target] Bad Converge Found!");
    state_ = "lost";
    return {};
  }

  if (state_ == "lost") return {};

  std::list<Target> targets = {target_};
  return targets;
}

// ========== 步骤 3-4：核心跟踪流程（哨兵全向感知版本） ==========

std::tuple<omniperception::DetectionResult, std::list<Target>> Tracker::track(
  const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
  std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  /**
   * ## 带全向感知的跟踪流程
   *
   * 哨兵机器人拥有多个相机（主相机 + 全向感知相机），
   * 主相机做正面自瞄检测，全向感知相机检测侧后方目标。
   * 当全向感知发现优先级更高的目标时，可以触发目标切换。
   *
   * 额外状态："switching" — 正在向全向感知指引的目标切换
   */

  omniperception::DetectionResult switch_target{std::list<Armor>(), t, 0, 0};
  omniperception::DetectionResult temp_target{std::list<Armor>(), t, 0, 0};
  if (!detection_queue.empty()) {
    temp_target = detection_queue.front();
  }

  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  // 排序
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  armors.sort([](const Armor & a, const Armor & b) { return a.priority < b.priority; });

  // ---------- 状态分支 ----------
  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }

  // 主相机检测到更高优先级 → 直接切换
  else if (state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
    found = set_target(armors, t);
    tools::logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
  }

  // 全向感知发现更高优先级目标 → 进入 switching 状态
  else if (
    state_ == "tracking" && !temp_target.armors.empty() &&
    temp_target.armors.front().priority < target_.priority && target_.convergened()) {
    state_ = "switching";
    switch_target = omniperception::DetectionResult{
      temp_target.armors, t, temp_target.delta_yaw, temp_target.delta_pitch};
    omni_target_priority_ = temp_target.armors.front().priority;
    found = false;
    tools::logger()->debug("omniperception find higher priority target");
  }

  // switching 中等待主相机回正
  else if (state_ == "switching") {
    found = !armors.empty() && armors.front().priority == omni_target_priority_;
  }

  // 切换完成后进入 detecting
  else if (state_ == "detecting" && pre_state_ == "switching") {
    found = set_target(armors, t);
  }

  else {
    found = update_target(armors, t);
  }

  pre_state_ = state_;
  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {switch_target, {}};
  }

  if (state_ == "lost") return {switch_target, {}};

  std::list<Target> targets = {target_};
  return {switch_target, targets};
}

// ========== 步骤 3-5：状态机 ==========

void Tracker::state_machine(bool found)
{
  /**
   * ## 跟踪状态机
   *
   * 状态转换图：
   * lost ──(并 detected)──→ detecting ──(连续 detected ≥ min_detect_count)──→ tracking
   * ↑                             │                                               │
   * │                    (检测丢失)│                                               │(检测丢失)
   * │                             ↓                                               ↓
   * └────(超过丢失容忍)──── temp_lost ←──(丢失不超过容忍)── temp_lost ←─── tracking
   *       ↑
   *       │(等待时间过长)
   *       └── switching ──(主相机检测到切换目标且回正)──→ detecting
   */

  if (state_ == "lost") {
    if (!found) return;

    state_ = "detecting";
    detect_count_ = 1;
  }

  else if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      if (detect_count_ >= min_detect_count_) state_ = "tracking";
    } else {
      detect_count_ = 0;
      state_ = "lost";
    }
  }

  else if (state_ == "tracking") {
    if (found) return;

    temp_lost_count_ = 1;
    state_ = "temp_lost";
  }

  else if (state_ == "switching") {
    if (found) {
      state_ = "detecting";
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > 200) state_ = "lost";
    }
  }

  else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
    } else {
      temp_lost_count_++;
      if (target_.name == ArmorName::outpost)
        max_temp_lost_count_ = outpost_max_temp_lost_count_;
      else
        max_temp_lost_count_ = normal_temp_lost_count_;

      if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";
    }
  }
}

// ========== 步骤 3-6：创建新目标 ==========

bool Tracker::set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  /**
   * 从当前帧检测到的装甲板中选择最佳的作为新目标。
   * 先做 PnP 解算，然后根据兵种类型设置不同的 EKF 初始化参数。
   *
   * 不同兵种有不同的装甲板参数：
   * - 普通步兵/英雄（非平衡）：4 装甲板，半径 0.2m，P0 中 omega 方差 100
   * - 平衡步兵（英雄 3/4/5 号大装甲板）：2 装甲板，半径 0.2m
   * - 前哨站（outpost）：3 装甲板，半径 0.2765m，r 的初始方差极小（1e-4）
   * - 基地（base）：3 装甲板，半径 0.3205m，r 的初始方差极小（1e-4）
   */
  if (armors.empty()) return false;

  auto & armor = armors.front();
  solver_.solve(armor);

  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);

  if (is_balance) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 2, P0_dig);
  }

  else if (armor.name == ArmorName::outpost) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.2765, 3, P0_dig);
  }

  else if (armor.name == ArmorName::base) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.3205, 3, P0_dig);
  }

  else {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 4, P0_dig);
  }

  return true;
}

// ========== 步骤 3-7：更新已有目标 ==========

bool Tracker::update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  /**
   * 用当前帧的装甲板更新已有 Target：
   * 1. EKF 预测（将状态推进到当前时间）
   * 2. 在装甲板列表中查找与 Target 匹配的（相同名称和类型）
   * 3. 对匹配到的第一个装甲板做 PnP 解算 → EKF 更新
   *
   * @note 当前实现中，如果有多块装甲板名称和类型相同（如两个 one 号），
   *       只更新第一个匹配到的。图中最左侧的装甲板优先（来自漏删的排序逻辑）。
   */
  target_.predict(t);

  int found_count = 0;
  double min_x = 1e10;
  for (const auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    found_count++;
    min_x = armor.center.x < min_x ? armor.center.x : min_x;
  }

  if (found_count == 0) return false;

  for (auto & armor : armors) {
    if (
      armor.name != target_.name || armor.type != target_.armor_type
      //  || armor.center.x != min_x
    )
      continue;

    solver_.solve(armor);    // PnP 解算
    target_.update(armor);   // EKF 更新

    // 只对第一个匹配到的更新（break 被注释掉了，但多个匹配只会被 update 一次）
  }

  return true;
}

}  // namespace auto_aim
