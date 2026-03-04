/**
 * 过程式轨迹优化示例 —— 不使用 NlpFormulation / Parameters
 *
 * 场景：单腿跳跃机器人（Monoped）在平地上向前跳 1 米。
 * 目的：逐步展示 NLP 问题的完整构建过程。
 *
 * 优化流程概览：
 *   1. 地形 & 机器人模型
 *   2. 步态规划（相位时长）
 *   3. 决策变量（基座位姿 + 足端位置/姿态/力/力矩）
 *   4. 初始猜测 & 边界条件
 *   5. 样条构建（连接变量与约束的桥梁）
 *   6. 物理约束（动力学、运动范围、地形、力锥、摆动平滑、加速度连续）
 *   7. 代价项（力正则化）
 *   8. 求解 & 输出
 */

#include <iostream>
#include <Eigen/Dense>

#include <ifopt/problem.h>
#include <ifopt/ipopt_solver.h>

// 变量
#include <towr/variables/variable_names.h>
#include <towr/variables/nodes_variables_all.h>
#include <towr/variables/nodes_variables_phase_based.h>
#include <towr/variables/phase_durations.h>
#include <towr/variables/spline_holder.h>
#include <towr/variables/cartesian_dimensions.h>

// 模型
#include <towr/models/examples/monoped_model.h>
#include <towr/terrain/examples/height_map_examples.h>

// 约束
#include <towr/constraints/dynamic_constraint.h>
#include <towr/constraints/range_of_motion_constraint.h>
#include <towr/constraints/terrain_constraint.h>
#include <towr/constraints/force_constraint.h>
#include <towr/constraints/swing_constraint.h>
#include <towr/constraints/spline_acc_constraint.h>
#include <towr/constraints/base_height_constraint.h>

// 代价
#include <towr/costs/node_cost.h>

// 工具
#include <towr/utils/save_data.h>

using namespace towr;
using Eigen::Vector3d;

int main()
{
  // ============================================================
  // 1. 地形 & 机器人模型
  //    地形提供高度函数 h(x,y) 和摩擦系数，约束需要它。
  //    机器人模型分运动学（工作空间范围）和动力学（质量、惯量）。
  // ============================================================
  auto terrain = std::make_shared<FlatGround>(0.0);

  // 运动学：名义站姿 (0,0,-0.58)m，工作空间 ±(0.3, 0.15, 0.3)m
  auto kin = std::make_shared<MonopedKinematicModel>();

  // 动力学：20kg 单刚体，惯量 (Ixx=1.2, Iyy=5.5, Izz=6.0, ...)，1个末端
  auto dyn = std::make_shared<MonopedDynamicModel>();

  // ============================================================
  // 2. 步态规划
  //    交替排列 stance / swing 相位。第一相为 stance（脚在地上）。
  //    相位时长之和 = 运动总时间。
  // ============================================================
  const int ee = 0; // 唯一的末端执行器
  const bool first_phase_is_stance = true;
  //          stance  swing  stance  swing  stance
  std::vector<double> phase_durations_vec = {0.3, 0.2, 0.3, 0.2, 0.3};
  const double T_total = 1.3; // 所有相位时长之和

  // ============================================================
  // 3. 决策变量
  //    优化器通过调整这些变量来寻找可行轨迹。
  //    (a) 基座线性位置 / 角度：固定多项式时长，每个节点独立优化
  //    (b) 足端运动 / 姿态 / 力 / 力矩：按相位参数化（摆动相运动可变，支撑相力可变）
  //    (c) 相位时长：此例固定不优化
  // ============================================================

  // --- 3a. 基座变量 ---
  // 基座用等时长三次多项式串联。多项式时长越短自由度越高，但变量更多。
  const double base_poly_dt = 0.1;
  std::vector<double> base_poly_durations;
  double t_left = T_total;
  while (t_left > 1e-10) {
    double dt = (t_left > base_poly_dt) ? base_poly_dt : t_left;
    base_poly_durations.push_back(dt);
    t_left -= base_poly_dt;
  }
  int n_base_nodes = base_poly_durations.size() + 1;

  auto base_lin = std::make_shared<NodesVariablesAll>(n_base_nodes, k3D, id::base_lin_nodes);
  auto base_ang = std::make_shared<NodesVariablesAll>(n_base_nodes, k3D, id::base_ang_nodes);

  // --- 3b. 末端执行器变量 ---
  const int n_phases = phase_durations_vec.size();
  const int polys_per_swing = 2;  // 每个摆动相用2段多项式 → 足端可抬起再落下
  const int polys_per_stance_force = 3; // 每个支撑相力用3段多项式 → 力曲线更灵活

  auto ee_motion_nodes = std::make_shared<NodesVariablesEEMotion>(
      n_phases, first_phase_is_stance, id::EEMotionNodes(ee), polys_per_swing);

  auto ee_ang_nodes = std::make_shared<NodesVariablesEEAng>(
      n_phases, first_phase_is_stance, id::EEAngNodes(ee), polys_per_swing);

  auto ee_force_nodes = std::make_shared<NodesVariablesEEForce>(
      n_phases, first_phase_is_stance, id::EEForceNodes(ee), polys_per_stance_force);

  auto ee_torque_nodes = std::make_shared<NodesVariablesEETorque>(
      n_phases, first_phase_is_stance, id::EETorqueNodes(ee), polys_per_stance_force);

  // --- 3c. 相位时长（此例不优化，仅作为 SplineHolder 的输入） ---
  auto phase_dur = std::make_shared<PhaseDurations>(
      ee, phase_durations_vec, first_phase_is_stance,
      0.2, 1.0); // min/max bounds（不优化时不起作用）

  // ============================================================
  // 4. 初始猜测 & 边界条件
  //    线性插值给一个粗略的初始猜测，帮助求解器收敛。
  //    边界条件固定起止状态。
  // ============================================================
  Vector3d init_base_pos(0.0, 0.0, 0.58);  // 名义站高 = |stance_z| = 0.58
  Vector3d goal_base_pos(1.0, 0.0, 0.58);  // 前进 1m
  Vector3d zero3 = Vector3d::Zero();

  // 基座线性
  base_lin->SetByLinearInterpolation(init_base_pos, goal_base_pos, T_total);
  base_lin->AddStartBound(kPos, {X, Y, Z}, init_base_pos);
  base_lin->AddStartBound(kVel, {X, Y, Z}, zero3);
  base_lin->AddFinalBound(kPos, {X, Y, Z}, goal_base_pos);
  base_lin->AddFinalBound(kVel, {X, Y, Z}, zero3);

  // 基座角度
  base_ang->SetByLinearInterpolation(zero3, zero3, T_total);
  base_ang->AddStartBound(kPos, {X, Y, Z}, zero3);
  base_ang->AddStartBound(kVel, {X, Y, Z}, zero3);
  base_ang->AddFinalBound(kPos, {X, Y, Z}, zero3);
  base_ang->AddFinalBound(kVel, {X, Y, Z}, zero3);

  // 足端运动
  Vector3d init_ee_pos(0.0, 0.0, 0.0); // 脚在地面
  Vector3d goal_ee_pos(1.0, 0.0, 0.0);
  ee_motion_nodes->SetByLinearInterpolation(init_ee_pos, goal_ee_pos, T_total);
  ee_motion_nodes->AddStartBound(kPos, {X, Y, Z}, init_ee_pos);
  ee_motion_nodes->AddFinalBound(kPos, {X, Y, Z}, goal_ee_pos);

  // 足端姿态
  ee_ang_nodes->SetByLinearInterpolation(zero3, zero3, T_total);
  ee_ang_nodes->AddStartBound(kPos, {X, Y, Z}, zero3);
  ee_ang_nodes->AddFinalBound(kPos, {X, Y, Z}, zero3);

  // 足端力：初始猜测为体重均匀分配（单腿承担全部 mg）
  double mg = dyn->m() * dyn->g();
  Vector3d f_stance(0.0, 0.0, mg);
  ee_force_nodes->SetByLinearInterpolation(f_stance, f_stance, T_total);

  // 足端力矩：初始为零
  ee_torque_nodes->SetByLinearInterpolation(zero3, zero3, T_total);

  // ============================================================
  // 5. 样条构建
  //    SplineHolder 将离散节点变量连接成连续的三次 Hermite 样条。
  //    约束和代价通过样条在任意时刻查询状态。
  //    观察者模式：节点值变化时样条自动更新系数。
  // ============================================================
  bool durations_change = false; // 不优化相位时长
  SplineHolder splines(base_lin, base_ang, base_poly_durations,
                       {ee_motion_nodes}, {ee_ang_nodes},
                       {ee_force_nodes}, {ee_torque_nodes},
                       {phase_dur}, durations_change);

  // ============================================================
  // 6. 物理约束
  //    每个约束强制轨迹满足一个物理条件。
  // ============================================================

  // 动力学：在离散时间点要求「样条加速度 == 牛顿-欧拉方程推导的加速度」
  auto dynamic_cstr = std::make_shared<DynamicConstraint>(
      dyn, T_total, /*dt=*/0.1, splines);

  // 运动范围：足端在基座坐标系下不超过工作空间盒
  auto rom_cstr = std::make_shared<RangeOfMotionConstraint>(
      kin, T_total, /*dt=*/0.08, ee, splines);

  // 地形：支撑相脚贴地，摆动相脚在地面上方
  auto terrain_cstr = std::make_shared<TerrainConstraint>(
      terrain, id::EEMotionNodes(ee), /*min_h=*/0.02, /*max_h=*/0.5);

  // 力锥：法向力单向（只能推不能拉）+ 切向力在摩擦锥内
  auto force_cstr = std::make_shared<ForceConstraint>(
      terrain, /*max_normal_force=*/1000.0, ee);

  // 摆动平滑：避免摆动相中段多项式速度过大导致穿透
  auto swing_cstr = std::make_shared<SwingConstraint>(id::EEMotionNodes(ee));

  // 加速度连续：多项式交界处加速度不跳变（否则需要瞬时无穷大力）
  auto base_lin_acc_cstr = std::make_shared<SplineAccConstraint>(
      splines.base_linear_, id::base_lin_nodes);
  auto base_ang_acc_cstr = std::make_shared<SplineAccConstraint>(
      splines.base_angular_, id::base_ang_nodes);

  // 基座高度：基座 z 始终高于地形 + 安全距离
  auto base_height_cstr = std::make_shared<BaseHeightConstraint>(
      terrain, /*safety_distance=*/0.3, id::base_lin_nodes);

  // ============================================================
  // 7. 代价项（可选）
  //    力正则化：惩罚力/力矩的幅值和变化率，让解更平滑。
  //    没有代价也能求解，但结果可能抖动。
  // ============================================================
  double w = 1e-5;

  // ============================================================
  // 8. 组装 NLP & 求解
  // ============================================================
  ifopt::Problem nlp;

  // 变量
  nlp.AddVariableSet(base_lin);
  nlp.AddVariableSet(base_ang);
  nlp.AddVariableSet(ee_motion_nodes);
  nlp.AddVariableSet(ee_ang_nodes);
  nlp.AddVariableSet(ee_force_nodes);
  nlp.AddVariableSet(ee_torque_nodes);

  // 约束
  nlp.AddConstraintSet(dynamic_cstr);
  nlp.AddConstraintSet(rom_cstr);
  nlp.AddConstraintSet(terrain_cstr);
  nlp.AddConstraintSet(force_cstr);
  nlp.AddConstraintSet(swing_cstr);
  nlp.AddConstraintSet(base_lin_acc_cstr);
  nlp.AddConstraintSet(base_ang_acc_cstr);
  nlp.AddConstraintSet(base_height_cstr);

  // 代价：力和力矩幅值 + 变化率
  for (int d : {X, Y, Z}) {
    nlp.AddCostSet(std::make_shared<NodeCost>(id::EEForceNodes(ee), kPos, d, w));
    nlp.AddCostSet(std::make_shared<NodeCost>(id::EEForceNodes(ee), kVel, d, 0.1 * w));
    nlp.AddCostSet(std::make_shared<NodeCost>(id::EETorqueNodes(ee), kPos, d, w));
  }

  // 求解
  auto solver = std::make_shared<ifopt::IpoptSolver>();
  solver->SetOption("jacobian_approximation", "exact");
  solver->SetOption("max_cpu_time", 20.0);
  solver->SetOption("max_iter", 1000);
  solver->Solve(nlp);

  // ============================================================
  // 9. 输出结果
  // ============================================================
  nlp.PrintCurrent();

  std::cout << "\n=== 轨迹采样 (每 0.1s) ===\n";
  std::cout << "  t   |  base x    y    z  |  foot x    y    z  |  fz\n";
  for (double t = 0.0; t <= T_total + 1e-6; t += 0.1) {
    double tc = std::min(t, T_total);
    auto base = splines.base_linear_->GetPoint(tc);
    auto foot = splines.ee_motion_.at(0)->GetPoint(tc);
    auto force = splines.ee_force_.at(0)->GetPoint(tc);
    printf("%.2f  | %5.2f %5.2f %5.2f | %5.2f %5.2f %5.2f | %6.1f\n",
           tc, base.p()(X), base.p()(Y), base.p()(Z),
           foot.p()(X), foot.p()(Y), foot.p()(Z), force.p()(Z));
  }

  SaveTrajectoryToCSV(splines, "procedural_trajectory.csv", 0.01);
  return 0;
}
