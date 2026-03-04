/**
 * 原地旋转跳示例 —— 蹲跳 + 腾空中 yaw 旋转 180 度
 *
 * Yaw 是 ZYX 欧拉角的外轴，不触发万向锁（只要 pitch 不接近 ±90°）。
 * 中间航点引导旋转方向，防止优化器走捷径或反方向。
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

int TimeToNodeId(double t, double dt) {
  return static_cast<int>(std::round(t / dt));
}

int main()
{
  const double h_stand = 0.58;
  const double h_crouch = 0.32;
  const double base_poly_dt = 0.1;
  const double yaw_target = M_PI; // 180° 旋转，改为 2*M_PI 可实现 360°

  // 时间线：更长的飞行时间给旋转留空间
  const double t_crouch  = 0.3;
  const double t_liftoff = 0.6;
  const double t_land    = 1.1;  // 0.5s 飞行时间
  const double t_recover = 1.4;
  const double T_total   = 1.7;

  std::vector<double> phase_durations_vec = {
    t_liftoff,               // stance: 蹲+蹬
    t_land - t_liftoff,      // swing:  飞行+旋转
    T_total - t_land          // stance: 落地+恢复
  };

  auto terrain = std::make_shared<FlatGround>(0.0);
  auto kin = std::make_shared<MonopedKinematicModel>();
  auto dyn = std::make_shared<MonopedDynamicModel>();

  const int ee = 0;
  const bool first_stance = true;

  // 基座多项式
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
      n_phases, first_stance, id::EEMotionNodes(ee), 2);
  auto ee_ang = std::make_shared<NodesVariablesEEAng>(
      n_phases, first_stance, id::EEAngNodes(ee), 2);
  auto ee_force = std::make_shared<NodesVariablesEEForce>(
      n_phases, first_stance, id::EEForceNodes(ee), 3);
  auto ee_torque = std::make_shared<NodesVariablesEETorque>(
      n_phases, first_stance, id::EETorqueNodes(ee), 3);
  auto phase_dur = std::make_shared<PhaseDurations>(
      ee, phase_durations_vec, first_stance, 0.2, 1.5);

  // ============================================================
  // 初始猜测 & 边界条件
  // ============================================================
  Vector3d zero3 = Vector3d::Zero();
  Vector3d stand_pos(0.0, 0.0, h_stand);
  Vector3d foot_pos(0.0, 0.0, 0.0);
  double mg = dyn->m() * dyn->g();

  Vector3d init_ang(0, 0, 0);
  Vector3d final_ang(0, 0, yaw_target);

  // 基座位置：原地起跳
  base_lin->SetByLinearInterpolation(stand_pos, stand_pos, T_total);
  base_lin->AddStartBound(kPos, {X, Y, Z}, stand_pos);
  base_lin->AddStartBound(kVel, {X, Y, Z}, zero3);
  base_lin->AddFinalBound(kPos, {X, Y, Z}, stand_pos);
  base_lin->AddFinalBound(kVel, {X, Y, Z}, zero3);

  // 下蹲 & 落地航点
  Vector3d crouch_pos(0.0, 0.0, h_crouch);
  base_lin->AddBounds(TimeToNodeId(t_crouch, base_poly_dt), kPos, {Z}, crouch_pos);
  base_lin->AddBounds(TimeToNodeId(t_crouch, base_poly_dt), kVel, {Z}, zero3);
  base_lin->AddBounds(TimeToNodeId(t_recover, base_poly_dt), kPos, {Z}, crouch_pos);
  base_lin->AddBounds(TimeToNodeId(t_recover, base_poly_dt), kVel, {Z}, zero3);

  // 基座角度：从 0 → yaw_target，线性插值作为初始猜测
  base_ang->SetByLinearInterpolation(init_ang, final_ang, T_total);
  base_ang->AddStartBound(kPos, {X, Y, Z}, init_ang);
  base_ang->AddStartBound(kVel, {X, Y, Z}, zero3);
  base_ang->AddFinalBound(kPos, {X, Y, Z}, final_ang);
  base_ang->AddFinalBound(kVel, {X, Y, Z}, zero3);

  // 中间航点引导旋转方向：腾空中点处 yaw 约为目标一半
  double t_mid_flight = (t_liftoff + t_land) / 2.0;
  Vector3d mid_ang(0, 0, yaw_target / 2.0);
  base_ang->AddBounds(TimeToNodeId(t_mid_flight, base_poly_dt), kPos, {Z},  mid_ang);
  // 起飞和落地时保持 roll/pitch 为零
  base_ang->AddBounds(TimeToNodeId(t_liftoff, base_poly_dt), kPos, {X, Y}, zero3);
  base_ang->AddBounds(TimeToNodeId(t_land, base_poly_dt), kPos, {X, Y}, zero3);

  // 足端
  ee_motion->SetByLinearInterpolation(foot_pos, foot_pos, T_total);
  ee_motion->AddStartBound(kPos, {X, Y, Z}, foot_pos);
  ee_motion->AddFinalBound(kPos, {X, Y, Z}, foot_pos);

  ee_ang->SetByLinearInterpolation(zero3, zero3, T_total);
  ee_ang->AddStartBound(kPos, {X, Y, Z}, zero3);
  ee_ang->AddFinalBound(kPos, {X, Y, Z}, zero3);

  ee_force->SetByLinearInterpolation(Vector3d(0, 0, mg), Vector3d(0, 0, mg), T_total);
  ee_torque->SetByLinearInterpolation(zero3, zero3, T_total);

  // ============================================================
  // 样条 & NLP
  // ============================================================
  SplineHolder splines(base_lin, base_ang, base_poly_durations,
                       {ee_motion}, {ee_ang}, {ee_force}, {ee_torque},
                       {phase_dur}, false);

  ifopt::Problem nlp;

  nlp.AddVariableSet(base_lin);
  nlp.AddVariableSet(base_ang);
  nlp.AddVariableSet(ee_motion);
  nlp.AddVariableSet(ee_ang);
  nlp.AddVariableSet(ee_force);
  nlp.AddVariableSet(ee_torque);

  nlp.AddConstraintSet(std::make_shared<DynamicConstraint>(dyn, T_total, 0.1, splines));
  nlp.AddConstraintSet(std::make_shared<RangeOfMotionConstraint>(kin, T_total, 0.08, ee, splines));
  nlp.AddConstraintSet(std::make_shared<TerrainConstraint>(terrain, id::EEMotionNodes(ee)));
  nlp.AddConstraintSet(std::make_shared<ForceConstraint>(terrain, 1000.0, ee));
  nlp.AddConstraintSet(std::make_shared<SwingConstraint>(id::EEMotionNodes(ee)));
  nlp.AddConstraintSet(std::make_shared<SplineAccConstraint>(splines.base_linear_, id::base_lin_nodes));
  nlp.AddConstraintSet(std::make_shared<SplineAccConstraint>(splines.base_angular_, id::base_ang_nodes));
  nlp.AddConstraintSet(std::make_shared<BaseHeightConstraint>(terrain, 0.2, id::base_lin_nodes));

  double w = 1e-5;
  for (int d : {X, Y, Z}) {
    nlp.AddCostSet(std::make_shared<NodeCost>(id::EEForceNodes(ee), kPos, d, w));
    nlp.AddCostSet(std::make_shared<NodeCost>(id::EETorqueNodes(ee), kPos, d, w));
  }

  auto solver = std::make_shared<ifopt::IpoptSolver>();
  solver->SetOption("jacobian_approximation", "exact");
  solver->SetOption("max_cpu_time", 30.0);
  solver->SetOption("max_iter", 2000);
  solver->Solve(nlp);

  nlp.PrintCurrent();

  // ============================================================
  // 输出
  // ============================================================
  std::cout << "\n=== 旋转跳轨迹 ===\n";
  std::cout << "  t   | base_z |  yaw   | foot_z |   fz   | 阶段\n";
  std::cout << "------|--------|--------|--------|--------|------\n";
  for (double t = 0.0; t <= T_total + 1e-6; t += 0.05) {
    double tc = std::min(t, T_total);
    auto bp = splines.base_linear_->GetPoint(tc);
    auto ba = splines.base_angular_->GetPoint(tc);
    auto fp = splines.ee_motion_.at(0)->GetPoint(tc);
    auto ff = splines.ee_force_.at(0)->GetPoint(tc);

    const char* phase;
    if      (tc < t_crouch)  phase = "下蹲";
    else if (tc < t_liftoff) phase = "蹬地";
    else if (tc < t_land)    phase = "腾空";
    else if (tc < t_recover) phase = "缓冲";
    else                     phase = "恢复";

    printf("%.2f  | %5.3f  | %6.1f° | %5.3f  | %6.1f | %s\n",
           tc, bp.p()(Z), ba.p()(Z) * 180.0 / M_PI,
           fp.p()(Z), ff.p()(Z), phase);
  }

  SaveTrajectoryToCSV(splines, "yaw_spin_trajectory.csv", 0.01);
  return 0;
}
