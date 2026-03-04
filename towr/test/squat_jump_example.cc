/**
 * 蹲跳示例 —— 演示如何用中间航点控制基座高度曲线
 *
 * 动作序列：站立 → 下蹲到设定高度 → 起跳 → 腾空 → 落地下蹲 → 恢复站立
 *
 * 核心技巧：
 *   base_lin->AddBounds(node_id, kPos, {Z}, target_z)
 *   在指定时刻的节点上施加位置约束，控制基座轨迹形状。
 *   node_id = round(t / base_poly_dt)，因为基座多项式等时长排列。
 */

#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include <ifopt/problem.h>
#include <ifopt/ipopt_solver.h>

#include <towr/variables/variable_names.h>
#include <towr/variables/nodes_variables_all.h>
#include <towr/variables/nodes_variables_phase_based.h>
#include <towr/variables/phase_durations.h>
#include <towr/variables/spline_holder.h>
#include <towr/variables/cartesian_dimensions.h>

#include <towr/models/examples/monoped_model.h>
#include <towr/terrain/examples/height_map_examples.h>

#include <towr/constraints/dynamic_constraint.h>
#include <towr/constraints/range_of_motion_constraint.h>
#include <towr/constraints/terrain_constraint.h>
#include <towr/constraints/force_constraint.h>
#include <towr/constraints/swing_constraint.h>
#include <towr/constraints/spline_acc_constraint.h>
#include <towr/constraints/base_height_constraint.h>

#include <towr/costs/node_cost.h>
#include <towr/utils/save_data.h>

using namespace towr;
using Eigen::Vector3d;

// 时刻 → 基座节点 ID（base_poly_dt 等间隔排列）
int TimeToNodeId(double t, double base_poly_dt) {
  return static_cast<int>(std::round(t / base_poly_dt));
}

int main()
{
  // ============================================================
  // 运动参数（可自由修改）
  // ============================================================
  const double h_stand = 0.58;  // 站立时基座高度（= 腿长）
  const double h_crouch = 0.35; // 下蹲时基座高度
  const double base_poly_dt = 0.1;

  // 动作时间线:
  //   [0, t_crouch]      stance: 下蹲
  //   [t_crouch, t_liftoff] stance: 蹬地起跳
  //   [t_liftoff, t_land]   swing:  腾空
  //   [t_land, t_recover]   stance: 落地缓冲（下蹲）
  //   [t_recover, T]        stance: 恢复站立
  const double t_crouch  = 0.3;  // 蹲到最低点
  const double t_liftoff = 0.6;  // 脚离地
  const double t_land    = 0.9;  // 脚着地
  const double t_recover = 1.2;  // 恢复到下蹲位（开始站起来）
  const double T_total   = 1.5;  // 完全恢复站立

  // 步态相位：stance(起跳) → swing(腾空) → stance(落地恢复)
  const double stance1_dur = t_liftoff;                // 0.6s
  const double swing_dur   = t_land - t_liftoff;       // 0.3s
  const double stance2_dur = T_total - t_land;          // 0.6s
  std::vector<double> phase_durations_vec = {stance1_dur, swing_dur, stance2_dur};

  // ============================================================
  // 模型
  // ============================================================
  auto terrain = std::make_shared<FlatGround>(0.0);
  auto kin = std::make_shared<MonopedKinematicModel>();
  auto dyn = std::make_shared<MonopedDynamicModel>();

  // ============================================================
  // 决策变量
  // ============================================================
  const int ee = 0;
  const bool first_stance = true;
  const int polys_per_swing = 2;
  const int polys_per_stance_force = 3;

  // 基座多项式时长
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

  int n_phases = phase_durations_vec.size();
  auto ee_motion = std::make_shared<NodesVariablesEEMotion>(
      n_phases, first_stance, id::EEMotionNodes(ee), polys_per_swing);
  auto ee_ang = std::make_shared<NodesVariablesEEAng>(
      n_phases, first_stance, id::EEAngNodes(ee), polys_per_swing);
  auto ee_force = std::make_shared<NodesVariablesEEForce>(
      n_phases, first_stance, id::EEForceNodes(ee), polys_per_stance_force);
  auto ee_torque = std::make_shared<NodesVariablesEETorque>(
      n_phases, first_stance, id::EETorqueNodes(ee), polys_per_stance_force);

  auto phase_dur = std::make_shared<PhaseDurations>(
      ee, phase_durations_vec, first_stance, 0.2, 1.5);

  // ============================================================
  // 初始猜测 & 边界条件
  // ============================================================
  Vector3d zero3 = Vector3d::Zero();
  Vector3d stand_pos(0.0, 0.0, h_stand);
  Vector3d foot_pos(0.0, 0.0, 0.0);
  double mg = dyn->m() * dyn->g();

  // 基座：起止都是站立高度，零速
  base_lin->SetByLinearInterpolation(stand_pos, stand_pos, T_total);
  base_lin->AddStartBound(kPos, {X, Y, Z}, stand_pos);
  base_lin->AddStartBound(kVel, {X, Y, Z}, zero3);
  base_lin->AddFinalBound(kPos, {X, Y, Z}, stand_pos);
  base_lin->AddFinalBound(kVel, {X, Y, Z}, zero3);

  // ============================================================
  // 关键：中间航点约束
  //   通过 AddBounds 在特定节点上钉住基座 z 高度。
  //   只约束 z，xy 自由（原地跳所以不需要约束 xy）。
  // ============================================================
  Vector3d crouch_pos(0.0, 0.0, h_crouch);

  // 下蹲最低点
  int node_crouch = TimeToNodeId(t_crouch, base_poly_dt);
  base_lin->AddBounds(node_crouch, kPos, {Z}, crouch_pos);

  // 落地缓冲最低点（对称）
  int node_recover = TimeToNodeId(t_recover, base_poly_dt);
  base_lin->AddBounds(node_recover, kPos, {Z}, crouch_pos);

  // 可选：约束下蹲时速度为零（达到最低点时不再下降）
  base_lin->AddBounds(node_crouch, kVel, {Z}, zero3);
  base_lin->AddBounds(node_recover, kVel, {Z}, zero3);

  // 基座角度：全程保持零
  base_ang->SetByLinearInterpolation(zero3, zero3, T_total);
  base_ang->AddStartBound(kPos, {X, Y, Z}, zero3);
  base_ang->AddStartBound(kVel, {X, Y, Z}, zero3);
  base_ang->AddFinalBound(kPos, {X, Y, Z}, zero3);
  base_ang->AddFinalBound(kVel, {X, Y, Z}, zero3);

  // 足端：原地跳，起止位置相同
  ee_motion->SetByLinearInterpolation(foot_pos, foot_pos, T_total);
  ee_motion->AddStartBound(kPos, {X, Y, Z}, foot_pos);
  ee_motion->AddFinalBound(kPos, {X, Y, Z}, foot_pos);

  ee_ang->SetByLinearInterpolation(zero3, zero3, T_total);
  ee_ang->AddStartBound(kPos, {X, Y, Z}, zero3);
  ee_ang->AddFinalBound(kPos, {X, Y, Z}, zero3);

  ee_force->SetByLinearInterpolation(Vector3d(0, 0, mg), Vector3d(0, 0, mg), T_total);
  ee_torque->SetByLinearInterpolation(zero3, zero3, T_total);

  // ============================================================
  // 样条 & 约束
  // ============================================================
  SplineHolder splines(base_lin, base_ang, base_poly_durations,
                       {ee_motion}, {ee_ang}, {ee_force}, {ee_torque},
                       {phase_dur}, false);

  // ============================================================
  // 组装 NLP
  // ============================================================
  ifopt::Problem nlp;

  nlp.AddVariableSet(base_lin);
  nlp.AddVariableSet(base_ang);
  nlp.AddVariableSet(ee_motion);
  nlp.AddVariableSet(ee_ang);
  nlp.AddVariableSet(ee_force);
  nlp.AddVariableSet(ee_torque);

  // 约束
  nlp.AddConstraintSet(std::make_shared<DynamicConstraint>(
      dyn, T_total, 0.1, splines));
  nlp.AddConstraintSet(std::make_shared<RangeOfMotionConstraint>(
      kin, T_total, 0.08, ee, splines));
  nlp.AddConstraintSet(std::make_shared<TerrainConstraint>(
      terrain, id::EEMotionNodes(ee)));
  nlp.AddConstraintSet(std::make_shared<ForceConstraint>(
      terrain, 1000.0, ee));
  nlp.AddConstraintSet(std::make_shared<SwingConstraint>(
      id::EEMotionNodes(ee)));
  nlp.AddConstraintSet(std::make_shared<SplineAccConstraint>(
      splines.base_linear_, id::base_lin_nodes));
  nlp.AddConstraintSet(std::make_shared<SplineAccConstraint>(
      splines.base_angular_, id::base_ang_nodes));
  nlp.AddConstraintSet(std::make_shared<BaseHeightConstraint>(
      terrain, 0.2, id::base_lin_nodes));

  // 代价：力正则化（减少抖动）
  double w = 1e-5;
  for (int d : {X, Y, Z}) {
    nlp.AddCostSet(std::make_shared<NodeCost>(id::EEForceNodes(ee), kPos, d, w));
    nlp.AddCostSet(std::make_shared<NodeCost>(id::EETorqueNodes(ee), kPos, d, w));
  }

  // ============================================================
  // 求解
  // ============================================================
  auto solver = std::make_shared<ifopt::IpoptSolver>();
  solver->SetOption("jacobian_approximation", "exact");
  solver->SetOption("max_cpu_time", 20.0);
  solver->SetOption("max_iter", 1000);
  solver->Solve(nlp);

  nlp.PrintCurrent();

  // ============================================================
  // 输出：观察基座 z 的蹲-跳-落-恢复曲线
  // ============================================================
  std::cout << "\n=== 蹲跳轨迹 ===\n";
  std::cout << "  t   | base_z | foot_z |   fz   | 阶段\n";
  std::cout << "------|--------|--------|--------|------\n";
  for (double t = 0.0; t <= T_total + 1e-6; t += 0.05) {
    double tc = std::min(t, T_total);
    double bz = splines.base_linear_->GetPoint(tc).p()(Z);
    double fz_pos = splines.ee_motion_.at(0)->GetPoint(tc).p()(Z);
    double fz_force = splines.ee_force_.at(0)->GetPoint(tc).p()(Z);

    const char* phase;
    if      (tc < t_crouch)  phase = "下蹲";
    else if (tc < t_liftoff) phase = "蹬地";
    else if (tc < t_land)    phase = "腾空";
    else if (tc < t_recover) phase = "缓冲";
    else                     phase = "恢复";

    printf("%.2f  | %5.3f  | %5.3f  | %6.1f | %s\n",
           tc, bz, fz_pos, fz_force, phase);
  }

  SaveTrajectoryToCSV(splines, "squat_jump_trajectory.csv", 0.01);
  return 0;
}
