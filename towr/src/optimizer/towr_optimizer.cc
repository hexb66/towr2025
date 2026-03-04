#include <towr/optimizer/towr_optimizer.h>

#include <cmath>
#include <iostream>
#include <cassert>

#include <ifopt/problem.h>
#include <ifopt/ipopt_solver.h>

#include <towr/variables/variable_names.h>
#include <towr/variables/nodes_variables_all.h>
#include <towr/variables/nodes_variables_phase_based.h>
#include <towr/variables/phase_durations.h>
#include <towr/variables/cartesian_dimensions.h>
#include <towr/variables/euler_converter.h>
#include <towr/variables/rotvec_converter.h>

#include <towr/constraints/dynamic_constraint.h>
#include <towr/constraints/range_of_motion_constraint.h>
#include <towr/constraints/terrain_constraint.h>
#include <towr/constraints/force_constraint.h>
#include <towr/constraints/swing_constraint.h>
#include <towr/constraints/spline_acc_constraint.h>
#include <towr/constraints/base_height_constraint.h>

#include <towr/costs/node_cost.h>
#include <towr/utils/save_data.h>

namespace towr {

using Eigen::Vector3d;

// ---------------------------------------------------------------------------
// Public setters
// ---------------------------------------------------------------------------

void TowrOptimizer::setModel(KinematicModel::Ptr kin, DynamicModel::Ptr dyn)
{
  kin_ = std::move(kin);
  dyn_ = std::move(dyn);
}

void TowrOptimizer::setTerrain(HeightMap::Ptr terrain)
{
  terrain_ = std::move(terrain);
}

void TowrOptimizer::setInitialEE(const std::vector<Vector3d>& ee_pos)
{
  initial_ee_ = ee_pos;
  ee_set_ = true;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::vector<Vector3d>
TowrOptimizer::computeInitialEE(const Vector3d& base_pos) const
{
  auto nominal = kin_->GetNominalStanceInBase();
  std::vector<Vector3d> ee(nominal.size());
  for (size_t i = 0; i < nominal.size(); ++i) {
    ee[i] = base_pos + nominal.at(i);
    ee[i].z() = terrain_->GetHeight(ee[i].x(), ee[i].y());
  }
  return ee;
}

std::vector<double> TowrOptimizer::makeBasePolyDurations(double T)
{
  std::vector<double> d;
  double left = T;
  while (left > 1e-10) {
    double dt = (left > kBasePolyDt) ? kBasePolyDt : left;
    d.push_back(dt);
    left -= kBasePolyDt;
  }
  return d;
}

int TowrOptimizer::timeToNodeId(double t)
{
  return static_cast<int>(std::round(t / kBasePolyDt));
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

bool TowrOptimizer::saveCSV(const std::string& filename, double dt) const
{
  return SaveTrajectoryToCSV(solution_, filename, dt);
}

void TowrOptimizer::printTrajectory(double dt) const
{
  if (total_time_ <= 0) return;

  int n_ee = static_cast<int>(solution_.ee_motion_.size());
  std::cout << "\n  t   | base_z  | pitch  |";
  for (int i = 0; i < n_ee; ++i) std::cout << " ee" << i << "_z |";
  std::cout << "\n";

  for (double t = 0.0; t <= total_time_ + 1e-6; t += dt) {
    double tc = std::min(t, total_time_);
    auto bp = solution_.base_linear_->GetPoint(tc);
    auto ba = solution_.base_angular_->GetPoint(tc);

    printf("%.2f  | %6.3f  | %6.1f |", tc, bp.p()(Z),
           ba.p()(Y) * 180.0 / M_PI);

    for (int i = 0; i < n_ee; ++i) {
      auto ep = solution_.ee_motion_.at(i)->GetPoint(tc);
      printf(" %5.3f |", ep.p()(Z));
    }
    printf("\n");
  }
}

// ---------------------------------------------------------------------------
// Core solver runner
// ---------------------------------------------------------------------------

static bool runNlp(ifopt::Problem& nlp, const SolverConfig& cfg)
{
  auto solver = std::make_shared<ifopt::IpoptSolver>();
  solver->SetOption("jacobian_approximation", cfg.jacobian_approx);
  solver->SetOption("max_cpu_time", cfg.max_cpu_time);
  solver->SetOption("max_iter", cfg.max_iter);
  solver->SetOption("tol", cfg.tol);
  solver->Solve(nlp);
  nlp.PrintCurrent();
  return true;
}

// ---------------------------------------------------------------------------
// solveJump
// ---------------------------------------------------------------------------

bool TowrOptimizer::solveJump(const JumpConfig& cfg)
{
  assert(kin_ && dyn_ && terrain_);

  const int n_ee = kin_->GetNumberOfEndeffectors();
  const int N = cfg.num_jumps;

  // --- Timeline per jump ---
  const double stance_pre  = cfg.crouch_duration + cfg.push_duration;
  const double swing_dur   = cfg.flight_duration;
  const double stance_post = cfg.absorb_duration + cfg.recover_duration;
  const double stance_mid  = cfg.absorb_duration + cfg.crouch_duration;

  // Build phase durations: stance-swing-[stance-swing-]...-stance
  std::vector<double> phases;
  for (int j = 0; j < N; ++j) {
    if (j == 0) {
      phases.push_back(stance_pre);
    } else {
      phases.push_back(stance_mid);
    }
    phases.push_back(swing_dur);
  }
  phases.push_back(stance_post);

  double T_total = 0;
  for (double d : phases) T_total += d;
  total_time_ = T_total;

  // --- Positions ---
  double start_terrain_h = terrain_->GetHeight(0, 0);
  Vector3d start_base(0, 0, cfg.standing_height + start_terrain_h);
  Vector3d end_base = start_base + cfg.displacement;
  double end_terrain_h = terrain_->GetHeight(end_base.x(), end_base.y());
  end_base.z() = cfg.standing_height + end_terrain_h;

  auto ee_pos = ee_set_ ? initial_ee_ : computeInitialEE(start_base);
  Vector3d end_ee_offset = cfg.displacement;
  end_ee_offset.z() = 0;

  double mg = dyn_->m() * dyn_->g();
  Vector3d zero3 = Vector3d::Zero();

  // --- Base poly durations ---
  auto base_poly_dur = makeBasePolyDurations(T_total);
  int n_base_nodes = static_cast<int>(base_poly_dur.size()) + 1;

  // --- Variables ---
  auto base_lin = std::make_shared<NodesVariablesAll>(
      n_base_nodes, k3D, id::base_lin_nodes);
  auto base_ang = std::make_shared<NodesVariablesAll>(
      n_base_nodes, k3D, id::base_ang_nodes);

  base_lin->SetByLinearInterpolation(start_base, end_base, T_total);
  base_lin->AddStartBound(kPos, {X, Y, Z}, start_base);
  base_lin->AddStartBound(kVel, {X, Y, Z}, zero3);
  base_lin->AddFinalBound(kPos, {X, Y, Z}, end_base);
  base_lin->AddFinalBound(kVel, {X, Y, Z}, zero3);

  // Crouch waypoints for each jump
  double t_accum = 0;
  for (int j = 0; j < N; ++j) {
    double t_phase_start = t_accum;
    double t_stance = (j == 0) ? stance_pre : stance_mid;
    double t_crouch_abs = t_phase_start + cfg.crouch_duration;
    double t_liftoff_abs = t_phase_start + t_stance;
    double t_land_abs = t_liftoff_abs + swing_dur;
    double t_recover_abs = t_land_abs + cfg.absorb_duration;

    double frac = static_cast<double>(j + 0.5) / N;
    double interp_terrain = terrain_->GetHeight(
        cfg.displacement.x() * frac, cfg.displacement.y() * frac);
    double crouch_h = cfg.standing_height * cfg.crouch_ratio + interp_terrain;
    Vector3d crouch_pos = start_base * (1 - frac) + end_base * frac;
    crouch_pos.z() = crouch_h;

    base_lin->AddBounds(timeToNodeId(t_crouch_abs), kPos, {Z}, crouch_pos);
    base_lin->AddBounds(timeToNodeId(t_crouch_abs), kVel, {Z}, zero3);

    if (j < N - 1) {
      // mid-stance recover point
      base_lin->AddBounds(timeToNodeId(t_recover_abs), kPos, {Z}, crouch_pos);
      base_lin->AddBounds(timeToNodeId(t_recover_abs), kVel, {Z}, zero3);
    } else {
      double recover_h = cfg.standing_height * cfg.crouch_ratio + end_terrain_h;
      Vector3d recover_pos = end_base;
      recover_pos.z() = recover_h;
      base_lin->AddBounds(timeToNodeId(t_recover_abs), kPos, {Z}, recover_pos);
      base_lin->AddBounds(timeToNodeId(t_recover_abs), kVel, {Z}, zero3);
    }

    t_accum = t_land_abs;
    if (j < N - 1) t_accum = t_land_abs; // next iteration starts at land
  }

  // Angular: yaw rotation
  Vector3d init_ang = zero3;
  Vector3d final_ang(0, 0, cfg.yaw_rotation);
  base_ang->SetByLinearInterpolation(init_ang, final_ang, T_total);
  base_ang->AddStartBound(kPos, {X, Y, Z}, init_ang);
  base_ang->AddStartBound(kVel, {X, Y, Z}, zero3);
  base_ang->AddFinalBound(kPos, {X, Y, Z}, final_ang);
  base_ang->AddFinalBound(kVel, {X, Y, Z}, zero3);

  if (std::abs(cfg.yaw_rotation) > 1e-6) {
    // Set yaw waypoints at mid-flight of each jump
    double t_acc2 = 0;
    for (int j = 0; j < N; ++j) {
      double t_stance = (j == 0) ? stance_pre : stance_mid;
      double t_liftoff = t_acc2 + t_stance;
      double t_mid_flight = t_liftoff + swing_dur / 2.0;
      double yaw_frac = static_cast<double>(2 * j + 1) / (2 * N);
      Vector3d mid_ang(0, 0, cfg.yaw_rotation * yaw_frac);
      base_ang->AddBounds(timeToNodeId(t_mid_flight), kPos, {Z}, mid_ang);
      t_acc2 = t_liftoff + swing_dur;
    }
  }

  // EE variables
  int n_phases = static_cast<int>(phases.size());
  std::vector<std::shared_ptr<NodesVariablesEEMotion>> ee_motion_vars(n_ee);
  std::vector<std::shared_ptr<NodesVariablesEEAng>>    ee_ang_vars(n_ee);
  std::vector<std::shared_ptr<NodesVariablesEEForce>>  ee_force_vars(n_ee);
  std::vector<std::shared_ptr<NodesVariablesEETorque>> ee_torque_vars(n_ee);
  std::vector<std::shared_ptr<PhaseDurations>>         phase_dur_vars(n_ee);

  for (int i = 0; i < n_ee; ++i) {
    int swing_polys = (cfg.swing_polys > 0) ? cfg.swing_polys : kPolysPerSwing;
    ee_motion_vars[i] = std::make_shared<NodesVariablesEEMotion>(
        n_phases, true, id::EEMotionNodes(i), swing_polys);
    ee_ang_vars[i] = std::make_shared<NodesVariablesEEAng>(
        n_phases, true, id::EEAngNodes(i), swing_polys);
    ee_force_vars[i] = std::make_shared<NodesVariablesEEForce>(
        n_phases, true, id::EEForceNodes(i), kPolysPerStanceForce);
    ee_torque_vars[i] = std::make_shared<NodesVariablesEETorque>(
        n_phases, true, id::EETorqueNodes(i), kPolysPerStanceForce);
    phase_dur_vars[i] = std::make_shared<PhaseDurations>(
        i, phases, true, 0.1, 3.0);

    Vector3d foot_start = ee_pos.at(i);
    Vector3d foot_end = foot_start + end_ee_offset;
    foot_end.z() = terrain_->GetHeight(foot_end.x(), foot_end.y());

    ee_motion_vars[i]->SetByLinearInterpolation(foot_start, foot_end, T_total);
    ee_motion_vars[i]->AddStartBound(kPos, {X, Y, Z}, foot_start);
    ee_motion_vars[i]->AddFinalBound(kPos, {X, Y, Z}, foot_end);

    ee_ang_vars[i]->SetByLinearInterpolation(zero3, zero3, T_total);
    ee_ang_vars[i]->AddStartBound(kPos, {X, Y, Z}, zero3);
    ee_ang_vars[i]->AddFinalBound(kPos, {X, Y, Z}, zero3);

    ee_force_vars[i]->SetByLinearInterpolation(
        Vector3d(0, 0, mg / n_ee), Vector3d(0, 0, mg / n_ee), T_total);
    ee_torque_vars[i]->SetByLinearInterpolation(zero3, zero3, T_total);
  }

  // --- Build SplineHolder ---
  std::vector<NodesVariablesPhaseBased::Ptr> ee_m, ee_a, ee_f, ee_t;
  std::vector<PhaseDurations::Ptr> pd;
  for (int i = 0; i < n_ee; ++i) {
    ee_m.push_back(ee_motion_vars[i]);
    ee_a.push_back(ee_ang_vars[i]);
    ee_f.push_back(ee_force_vars[i]);
    ee_t.push_back(ee_torque_vars[i]);
    pd.push_back(phase_dur_vars[i]);
  }

  solution_ = SplineHolder(base_lin, base_ang, base_poly_dur,
                            ee_m, ee_a, ee_f, ee_t, pd, false);

  // --- NLP ---
  ifopt::Problem nlp;

  nlp.AddVariableSet(base_lin);
  nlp.AddVariableSet(base_ang);
  for (int i = 0; i < n_ee; ++i) {
    nlp.AddVariableSet(ee_motion_vars[i]);
    nlp.AddVariableSet(ee_ang_vars[i]);
    nlp.AddVariableSet(ee_force_vars[i]);
    nlp.AddVariableSet(ee_torque_vars[i]);
  }

  // Constraints
  nlp.AddConstraintSet(std::make_shared<DynamicConstraint>(
      dyn_, T_total, 0.1, solution_));

  for (int i = 0; i < n_ee; ++i) {
    nlp.AddConstraintSet(std::make_shared<RangeOfMotionConstraint>(
        kin_, T_total, 0.08, i, solution_));
    nlp.AddConstraintSet(std::make_shared<TerrainConstraint>(
        terrain_, id::EEMotionNodes(i)));
    nlp.AddConstraintSet(std::make_shared<ForceConstraint>(
        terrain_, cfg.force_limit, i));
    nlp.AddConstraintSet(std::make_shared<SwingConstraint>(
        id::EEMotionNodes(i)));
  }

  nlp.AddConstraintSet(std::make_shared<SplineAccConstraint>(
      solution_.base_linear_, id::base_lin_nodes));
  nlp.AddConstraintSet(std::make_shared<SplineAccConstraint>(
      solution_.base_angular_, id::base_ang_nodes));
  nlp.AddConstraintSet(std::make_shared<BaseHeightConstraint>(
      terrain_, 0.2, id::base_lin_nodes));

  // Costs: force/torque regularization
  for (int i = 0; i < n_ee; ++i) {
    for (int d : {X, Y, Z}) {
      nlp.AddCostSet(std::make_shared<NodeCost>(
          id::EEForceNodes(i), kPos, d, 1e-5));
      nlp.AddCostSet(std::make_shared<NodeCost>(
          id::EETorqueNodes(i), kPos, d, 1e-5));
    }
  }

  // Base motion regularization
  for (int d : {X, Y, Z})
    nlp.AddCostSet(std::make_shared<NodeCost>(
        id::base_lin_nodes, kVel, d, 1e-3));
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_ang_nodes, kPos, X, 1e-3));
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_ang_nodes, kPos, Y, 1e-2));
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_ang_nodes, kPos, Z, 1e-3));
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_ang_nodes, kVel, X, 1e-3));
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_ang_nodes, kVel, Y, 1e-1));
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_ang_nodes, kVel, Z, 1e-3));

  return runNlp(nlp, cfg.solver);
}

// ---------------------------------------------------------------------------
// solveFlip — single pass + post-process foot arc
//
// During flight there are NO contact forces, so the foot position has zero
// effect on dynamics.  We solve the NLP with fully relaxed swing RoM to get
// the correct base trajectory, then rewrite swing nodes so the foot traces
// a kinematically correct arc: foot_W = base_W + R(θ) * nominal_stance.
// ---------------------------------------------------------------------------

bool TowrOptimizer::solveFlip(const FlipConfig& cfg)
{
  assert(kin_ && dyn_ && terrain_);

  const int n_ee = kin_->GetNumberOfEndeffectors();

  // --- Timeline ---
  const double stance_pre  = cfg.crouch_duration + cfg.push_duration;
  const double swing_dur   = cfg.flight_duration;
  const double stance_post = cfg.absorb_duration + cfg.recover_duration;
  double T_total = stance_pre + swing_dur + stance_post;
  total_time_ = T_total;

  std::vector<double> phases = {stance_pre, swing_dur, stance_post};

  // --- Rotation ---
  int rot_axis;
  double rot_sign;
  switch (cfg.type) {
    case FlipConfig::BackFlip:     rot_axis = Y; rot_sign = -1; break;
    case FlipConfig::FrontFlip:    rot_axis = Y; rot_sign = +1; break;
    case FlipConfig::SideFlipLeft: rot_axis = X; rot_sign = +1; break;
    case FlipConfig::SideFlipRight:rot_axis = X; rot_sign = -1; break;
  }
  double rotation = rot_sign * cfg.rotation_amount;

  std::vector<int> other_axes;
  for (int d : {X, Y, Z})
    if (d != rot_axis) other_axes.push_back(d);

  // --- Positions ---
  double terrain_h = terrain_->GetHeight(0, 0);
  Vector3d start_base(0, 0, cfg.standing_height + terrain_h);
  auto ee_pos = ee_set_ ? initial_ee_ : computeInitialEE(start_base);
  double mg = dyn_->m() * dyn_->g();
  Vector3d zero3 = Vector3d::Zero();

  // --- Base variables ---
  auto base_poly_dur = makeBasePolyDurations(T_total);
  int n_base_nodes = static_cast<int>(base_poly_dur.size()) + 1;

  auto base_lin = std::make_shared<NodesVariablesAll>(
      n_base_nodes, k3D, id::base_lin_nodes);
  auto base_ang = std::make_shared<NodesVariablesAll>(
      n_base_nodes, k3D, id::base_ang_nodes);

  base_lin->SetByLinearInterpolation(start_base, start_base, T_total);
  base_lin->AddStartBound(kPos, {X, Y, Z}, start_base);
  base_lin->AddStartBound(kVel, {X, Y, Z}, zero3);
  base_lin->AddFinalBound(kPos, {X, Y, Z}, start_base);
  base_lin->AddFinalBound(kVel, {X, Y, Z}, zero3);

  double crouch_h = cfg.standing_height * cfg.crouch_ratio + terrain_h;
  Vector3d crouch_pos(0, 0, crouch_h);
  double t_crouch = cfg.crouch_duration;
  double t_recover = stance_pre + swing_dur + cfg.absorb_duration;
  base_lin->AddBounds(timeToNodeId(t_crouch), kPos, {Z}, crouch_pos);
  base_lin->AddBounds(timeToNodeId(t_crouch), kVel, {Z}, zero3);
  base_lin->AddBounds(timeToNodeId(t_recover), kPos, {Z}, crouch_pos);
  base_lin->AddBounds(timeToNodeId(t_recover), kVel, {Z}, zero3);

  Vector3d init_rv = zero3, final_rv = zero3;
  final_rv(rot_axis) = rotation;

  double t_liftoff = stance_pre;
  double t_land    = stance_pre + swing_dur;
  double t_mid     = (t_liftoff + t_land) / 2.0;

  // Piecewise initial guess: stance=const, flight=linear rotation
  {
    Eigen::VectorXd avals = base_ang->GetValues();
    for (int n = 0; n < n_base_nodes; ++n) {
      double t_n = n * kBasePolyDt;
      Vector3d rv = zero3;
      if (t_n <= t_liftoff) {
        rv = init_rv;
      } else if (t_n >= t_land) {
        rv = final_rv;
      } else {
        double f = (t_n - t_liftoff) / swing_dur;
        rv = init_rv + f * (final_rv - init_rv);
      }
      for (int d : {X, Y, Z}) {
        int idx = base_ang->GetOptIndex(
            NodesVariables::NodeValueInfo(n, kPos, d));
        if (idx >= 0) avals(idx) = rv(d);
        int idxv = base_ang->GetOptIndex(
            NodesVariables::NodeValueInfo(n, kVel, d));
        if (idxv >= 0) {
          if (t_n > t_liftoff && t_n < t_land)
            avals(idxv) = (final_rv(d) - init_rv(d)) / swing_dur;
          else
            avals(idxv) = 0.0;
        }
      }
    }
    base_ang->SetVariables(avals);
  }

  base_ang->AddStartBound(kPos, {X, Y, Z}, init_rv);
  base_ang->AddStartBound(kVel, {X, Y, Z}, zero3);
  base_ang->AddFinalBound(kPos, {X, Y, Z}, final_rv);
  base_ang->AddFinalBound(kVel, {X, Y, Z}, zero3);

  // Pin angle at every stance node: pre-stance=0, post-stance=final_rv
  for (int n = 0; n < n_base_nodes; ++n) {
    double t_n = n * kBasePolyDt;
    if (t_n <= t_liftoff + 1e-6) {
      base_ang->AddBounds(n, kPos, {X, Y, Z}, init_rv);
    } else if (t_n >= t_land - 1e-6) {
      base_ang->AddBounds(n, kPos, {X, Y, Z}, final_rv);
    }
  }

  // Flight waypoints
  Vector3d mid_rv = zero3;
  mid_rv(rot_axis) = rotation / 2.0;
  base_ang->AddBounds(timeToNodeId(t_mid), kPos, {X, Y, Z}, mid_rv);

  // --- EE variables ---
  int n_phases = static_cast<int>(phases.size());
  std::vector<std::shared_ptr<NodesVariablesEEMotion>> ee_motion_vars(n_ee);
  std::vector<std::shared_ptr<NodesVariablesEEAng>>    ee_ang_vars(n_ee);
  std::vector<std::shared_ptr<NodesVariablesEEForce>>  ee_force_vars(n_ee);
  std::vector<std::shared_ptr<NodesVariablesEETorque>> ee_torque_vars(n_ee);
  std::vector<std::shared_ptr<PhaseDurations>>         phase_dur_vars(n_ee);

  for (int i = 0; i < n_ee; ++i) {
    // Default to 1 poly per swing for flips; allow user override via cfg.
    int flip_polys_per_swing = (cfg.swing_polys > 0) ? cfg.swing_polys : 1;
    ee_motion_vars[i] = std::make_shared<NodesVariablesEEMotion>(
        n_phases, true, id::EEMotionNodes(i), flip_polys_per_swing);
    ee_ang_vars[i] = std::make_shared<NodesVariablesEEAng>(
        n_phases, true, id::EEAngNodes(i), flip_polys_per_swing);
    ee_force_vars[i] = std::make_shared<NodesVariablesEEForce>(
        n_phases, true, id::EEForceNodes(i), kPolysPerStanceForce);
    ee_torque_vars[i] = std::make_shared<NodesVariablesEETorque>(
        n_phases, true, id::EETorqueNodes(i), kPolysPerStanceForce);
    phase_dur_vars[i] = std::make_shared<PhaseDurations>(
        i, phases, true, 0.1, 3.0);

    Vector3d foot = ee_pos.at(i);
    ee_motion_vars[i]->SetByLinearInterpolation(foot, foot, T_total);
    ee_motion_vars[i]->AddStartBound(kPos, {X, Y, Z}, foot);
    ee_motion_vars[i]->AddFinalBound(kPos, {X, Y, Z}, foot);

    ee_ang_vars[i]->SetByLinearInterpolation(zero3, zero3, T_total);
    ee_ang_vars[i]->AddStartBound(kPos, {X, Y, Z}, zero3);
    ee_ang_vars[i]->AddFinalBound(kPos, {X, Y, Z}, zero3);

    ee_force_vars[i]->SetByLinearInterpolation(
        Vector3d(0, 0, mg / n_ee), Vector3d(0, 0, mg / n_ee), T_total);
    ee_torque_vars[i]->SetByLinearInterpolation(zero3, zero3, T_total);
  }

  // --- SplineHolder ---
  std::vector<NodesVariablesPhaseBased::Ptr> ee_m, ee_a, ee_f, ee_t;
  std::vector<PhaseDurations::Ptr> pd;
  for (int i = 0; i < n_ee; ++i) {
    ee_m.push_back(ee_motion_vars[i]);
    ee_a.push_back(ee_ang_vars[i]);
    ee_f.push_back(ee_force_vars[i]);
    ee_t.push_back(ee_torque_vars[i]);
    pd.push_back(phase_dur_vars[i]);
  }
  solution_ = SplineHolder(base_lin, base_ang, base_poly_dur,
                            ee_m, ee_a, ee_f, ee_t, pd, false);
  solution_.angular_converter_ =
      std::make_shared<RotVecConverter>(solution_.base_angular_);

  // --- NLP: solve with full swing RoM relaxation ---
  ifopt::Problem nlp;
  nlp.AddVariableSet(base_lin);
  nlp.AddVariableSet(base_ang);
  for (int i = 0; i < n_ee; ++i) {
    nlp.AddVariableSet(ee_motion_vars[i]);
    nlp.AddVariableSet(ee_ang_vars[i]);
    nlp.AddVariableSet(ee_force_vars[i]);
    nlp.AddVariableSet(ee_torque_vars[i]);
  }

  nlp.AddConstraintSet(std::make_shared<DynamicConstraint>(
      dyn_, T_total, 0.1, solution_));

  double flip_max_swing_h = 2.0 * cfg.standing_height + 0.5;
  for (int i = 0; i < n_ee; ++i) {
    auto rom = std::make_shared<RangeOfMotionConstraint>(
        kin_, T_total, 0.08, i, solution_);
    rom->SetSwingRelaxation(phase_dur_vars[i].get(), {X, Y, Z});
    nlp.AddConstraintSet(rom);
    nlp.AddConstraintSet(std::make_shared<TerrainConstraint>(
        terrain_, id::EEMotionNodes(i), 0.02, flip_max_swing_h));
    nlp.AddConstraintSet(std::make_shared<ForceConstraint>(
        terrain_, cfg.force_limit, i));
    nlp.AddConstraintSet(std::make_shared<SwingConstraint>(
        id::EEMotionNodes(i)));
  }

  nlp.AddConstraintSet(std::make_shared<SplineAccConstraint>(
      solution_.base_linear_, id::base_lin_nodes));
  nlp.AddConstraintSet(std::make_shared<SplineAccConstraint>(
      solution_.base_angular_, id::base_ang_nodes));
  nlp.AddConstraintSet(std::make_shared<BaseHeightConstraint>(
      terrain_, 0.15, id::base_lin_nodes));

  for (int i = 0; i < n_ee; ++i) {
    for (int d : {X, Y, Z}) {
      nlp.AddCostSet(std::make_shared<NodeCost>(
          id::EEForceNodes(i), kPos, d, 1e-5));
      nlp.AddCostSet(std::make_shared<NodeCost>(
          id::EETorqueNodes(i), kPos, d, 1e-5));
    }
  }
  for (int d : {X, Y, Z})
    nlp.AddCostSet(std::make_shared<NodeCost>(
        id::base_lin_nodes, kVel, d, 1e-3));
  // Penalize all angular axes (not just non-rotation) to keep stance clean
  for (int d : {X, Y, Z}) {
    nlp.AddCostSet(std::make_shared<NodeCost>(
        id::base_ang_nodes, kPos, d, 5e-2));
    nlp.AddCostSet(std::make_shared<NodeCost>(
        id::base_ang_nodes, kVel, d, 5e-2));
  }

  runNlp(nlp, cfg.solver);

  // --- Post-process: rewrite swing nodes with tuck profile (position only) ---
  // Human-like backflip: extend legs at takeoff/landing, tuck in mid-flight.
  // tuck_scale(frac) = 1 - (1-tuck_ratio) * sin(π·frac)
  //   frac=0,1 → scale=1 (full leg length)
  //   frac=0.5 → scale=tuck_ratio (shortest, ~45% of leg)
  //
  // We只修改位置 kPos，让样条在节点间自动插平滑速度/加速度，避免手工写速度
  // 造成的分段感。
  for (int i = 0; i < n_ee; ++i) {
    auto nodes = ee_motion_vars[i]->GetNodes();
    Eigen::VectorXd vals = ee_motion_vars[i]->GetValues();

    int n_swing = 0;
    for (size_t n = 0; n < nodes.size(); ++n)
      if (!ee_motion_vars[i]->IsConstantNode(n)) n_swing++;

    Vector3d nominal_B = kin_->GetNominalStanceInBase().at(i);
    double tuck_depth = 1.0 - cfg.tuck_ratio;

    int si = 0;
    for (size_t n = 0; n < nodes.size(); ++n) {
      if (!ee_motion_vars[i]->IsConstantNode(n)) {
        double frac = static_cast<double>(si + 1) / (n_swing + 1);
        double t_node = t_liftoff + swing_dur * frac;

        // Tuck profile
        double tuck_scale = 1.0 - tuck_depth * std::sin(M_PI * frac);

        auto base_pt = solution_.base_linear_->GetPoint(t_node);
        auto ang_pt  = solution_.base_angular_->GetPoint(t_node);
        Vector3d base_W   = base_pt.p();
        Vector3d rv       = ang_pt.p();

        double angle = rv.norm();
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        if (angle > 1e-10) {
          Eigen::AngleAxisd aa(angle, rv / angle);
          R = aa.toRotationMatrix();
        }

        Vector3d tucked_B = nominal_B * tuck_scale;
        Vector3d r_W = R * tucked_B;
        Vector3d foot_W = base_W + r_W;
        foot_W.z() = std::max(foot_W.z(), terrain_h);

        for (int d : {X, Y, Z}) {
          int idx_p = ee_motion_vars[i]->GetOptIndex(
              NodesVariables::NodeValueInfo(n, kPos, d));
          if (idx_p >= 0) vals(idx_p) = foot_W(d);
        }
        si++;
      }
    }
    ee_motion_vars[i]->SetVariables(vals);
  }

  return true;
}

} // namespace towr
