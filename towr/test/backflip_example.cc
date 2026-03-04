/**
 * 后空翻示例 —— 验证旋转向量参数化支持大角度 pitch 旋转
 *
 * 使用 RotVecConverter 避免万向锁，RoM 飞行阶段放松全部维度。
 * 动作：蹲下 → 蹬地 → 腾空后空翻（pitch 旋转 ~2π）→ 落地。
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
#include <towr/variables/rotvec_converter.h>

#include <towr/models/examples/monoped_model.h>
#include <towr/terrain/examples/height_map_examples.h>

#include <towr/constraints/dynamic_constraint.h>
#include <towr/constraints/range_of_motion_constraint.h>
#include <towr/constraints/terrain_constraint.h>
#include <towr/constraints/force_constraint.h>
#include <towr/constraints/swing_constraint.h>
#include <towr/constraints/spline_acc_constraint.h>

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
  const double h_crouch = 0.30;
  const double base_poly_dt = 0.1;

  // Rotation vector for a backflip: pitch = -2π (rotate backward around Y)
  // In rotation vector: θ = (0, -2π, 0)
  const double pitch_target = -2.0 * M_PI;

  // Timeline
  const double t_crouch  = 0.3;
  const double t_liftoff = 0.5;
  const double t_land    = 1.3;  // 0.8s flight
  const double t_recover = 1.6;
  const double T_total   = 1.9;

  std::vector<double> phase_durations_vec = {
    t_liftoff,
    t_land - t_liftoff,
    T_total - t_land
  };

  auto terrain = std::make_shared<FlatGround>(0.0);
  auto kin = std::make_shared<MonopedKinematicModel>();
  auto dyn = std::make_shared<MonopedDynamicModel>();

  const int ee = 0;
  const bool first_stance = true;

  // Base polynomials
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
      n_phases, first_stance, id::EEMotionNodes(ee), 3);
  auto ee_ang_var = std::make_shared<NodesVariablesEEAng>(
      n_phases, first_stance, id::EEAngNodes(ee), 3);
  auto ee_force = std::make_shared<NodesVariablesEEForce>(
      n_phases, first_stance, id::EEForceNodes(ee), 4);
  auto ee_torque = std::make_shared<NodesVariablesEETorque>(
      n_phases, first_stance, id::EETorqueNodes(ee), 4);
  auto phase_dur = std::make_shared<PhaseDurations>(
      ee, phase_durations_vec, first_stance, 0.2, 2.0);

  // ============================================================
  // Initial guess & bounds
  // ============================================================
  Vector3d zero3 = Vector3d::Zero();
  Vector3d stand_pos(0.0, 0.0, h_stand);
  Vector3d foot_pos(0.0, 0.0, 0.0);
  double mg = dyn->m() * dyn->g();

  // Rotation vector initial/final: start at (0,0,0), end at (0, pitch_target, 0)
  Vector3d init_rv(0, 0, 0);
  Vector3d final_rv(0, pitch_target, 0);

  // Base position: start and end standing, peak height in flight
  base_lin->SetByLinearInterpolation(stand_pos, stand_pos, T_total);
  base_lin->AddStartBound(kPos, {X, Y, Z}, stand_pos);
  base_lin->AddStartBound(kVel, {X, Y, Z}, zero3);
  base_lin->AddFinalBound(kPos, {X, Y, Z}, stand_pos);
  base_lin->AddFinalBound(kVel, {X, Y, Z}, zero3);

  // Crouch waypoints
  Vector3d crouch_pos(0.0, 0.0, h_crouch);
  base_lin->AddBounds(TimeToNodeId(t_crouch, base_poly_dt), kPos, {Z}, crouch_pos);
  base_lin->AddBounds(TimeToNodeId(t_crouch, base_poly_dt), kVel, {Z}, zero3);
  base_lin->AddBounds(TimeToNodeId(t_recover, base_poly_dt), kPos, {Z}, crouch_pos);
  base_lin->AddBounds(TimeToNodeId(t_recover, base_poly_dt), kVel, {Z}, zero3);

  // Base angular: rotation vector parameterization
  base_ang->SetByLinearInterpolation(init_rv, final_rv, T_total);
  base_ang->AddStartBound(kPos, {X, Y, Z}, init_rv);
  base_ang->AddStartBound(kVel, {X, Y, Z}, zero3);
  base_ang->AddFinalBound(kPos, {X, Y, Z}, final_rv);
  base_ang->AddFinalBound(kVel, {X, Y, Z}, zero3);

  // Mid-flight waypoint: half rotation
  double t_mid = (t_liftoff + t_land) / 2.0;
  Vector3d mid_rv(0, pitch_target / 2.0, 0);
  base_ang->AddBounds(TimeToNodeId(t_mid, base_poly_dt), kPos, {Y}, mid_rv);
  // Keep roll & yaw near zero
  base_ang->AddBounds(TimeToNodeId(t_liftoff, base_poly_dt), kPos, {X, Z}, zero3);
  base_ang->AddBounds(TimeToNodeId(t_land, base_poly_dt), kPos, {X, Z}, zero3);

  // Foot
  ee_motion->SetByLinearInterpolation(foot_pos, foot_pos, T_total);
  ee_motion->AddStartBound(kPos, {X, Y, Z}, foot_pos);
  ee_motion->AddFinalBound(kPos, {X, Y, Z}, foot_pos);

  ee_ang_var->SetByLinearInterpolation(zero3, zero3, T_total);
  ee_ang_var->AddStartBound(kPos, {X, Y, Z}, zero3);
  ee_ang_var->AddFinalBound(kPos, {X, Y, Z}, zero3);

  ee_force->SetByLinearInterpolation(Vector3d(0, 0, mg), Vector3d(0, 0, mg), T_total);
  ee_torque->SetByLinearInterpolation(zero3, zero3, T_total);

  // ============================================================
  // Splines with RotVecConverter
  // ============================================================
  SplineHolder splines(base_lin, base_ang, base_poly_durations,
                       {ee_motion}, {ee_ang_var}, {ee_force}, {ee_torque},
                       {phase_dur}, false);
  splines.angular_converter_ = std::make_shared<RotVecConverter>(splines.base_angular_);

  // ============================================================
  // NLP
  // ============================================================
  ifopt::Problem nlp;

  nlp.AddVariableSet(base_lin);
  nlp.AddVariableSet(base_ang);
  nlp.AddVariableSet(ee_motion);
  nlp.AddVariableSet(ee_ang_var);
  nlp.AddVariableSet(ee_force);
  nlp.AddVariableSet(ee_torque);

  nlp.AddConstraintSet(std::make_shared<DynamicConstraint>(dyn, T_total, 0.1, splines));

  auto rom = std::make_shared<RangeOfMotionConstraint>(kin, T_total, 0.08, ee, splines);
  rom->SetSwingRelaxation(phase_dur.get(), {X, Y, Z});
  nlp.AddConstraintSet(rom);

  nlp.AddConstraintSet(std::make_shared<TerrainConstraint>(terrain, id::EEMotionNodes(ee)));
  nlp.AddConstraintSet(std::make_shared<ForceConstraint>(terrain, 2000.0, ee));
  nlp.AddConstraintSet(std::make_shared<SwingConstraint>(id::EEMotionNodes(ee)));
  nlp.AddConstraintSet(std::make_shared<SplineAccConstraint>(splines.base_linear_, id::base_lin_nodes));
  nlp.AddConstraintSet(std::make_shared<SplineAccConstraint>(splines.base_angular_, id::base_ang_nodes));

  // No BaseHeightConstraint during backflip (base goes inverted)

  double w = 1e-5;
  for (int d : {X, Y, Z}) {
    nlp.AddCostSet(std::make_shared<NodeCost>(id::EEForceNodes(ee), kPos, d, w));
    nlp.AddCostSet(std::make_shared<NodeCost>(id::EETorqueNodes(ee), kPos, d, w));
  }

  // ============================================================
  // Solve
  // ============================================================
  auto solver = std::make_shared<ifopt::IpoptSolver>();
  solver->SetOption("jacobian_approximation", "exact");
  solver->SetOption("max_cpu_time", 60.0);
  solver->SetOption("max_iter", 5000);
  solver->SetOption("tol", 1e-3);
  solver->Solve(nlp);

  nlp.PrintCurrent();

  // ============================================================
  // Output
  // ============================================================
  std::cout << "\n=== 后空翻轨迹（旋转向量参数化）===\n";
  std::cout << "  t   | base_z | pitch° | foot_z |   fz   | 阶段\n";
  std::cout << "------|--------|--------|--------|--------|------\n";
  for (double t = 0.0; t <= T_total + 1e-6; t += 0.05) {
    double tc = std::min(t, T_total);
    auto bp = splines.base_linear_->GetPoint(tc);
    auto ba = splines.base_angular_->GetPoint(tc);
    auto fp = splines.ee_motion_.at(0)->GetPoint(tc);
    auto ff = splines.ee_force_.at(0)->GetPoint(tc);

    // Rotation vector Y component → pitch angle (in degrees)
    double pitch_deg = ba.p()(Y) * 180.0 / M_PI;

    const char* phase;
    if      (tc < t_crouch)  phase = "下蹲";
    else if (tc < t_liftoff) phase = "蹬地";
    else if (tc < t_land)    phase = "腾空";
    else if (tc < t_recover) phase = "缓冲";
    else                     phase = "恢复";

    printf("%.2f  | %6.3f | %7.1f | %5.3f  | %7.1f | %s\n",
           tc, bp.p()(Z), pitch_deg, fp.p()(Z), ff.p()(Z), phase);
  }

  SaveTrajectoryToCSV(splines, "backflip_trajectory.csv", 0.01);
  return 0;
}
