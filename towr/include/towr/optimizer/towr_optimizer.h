#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>

#include <towr/models/kinematic_model.h>
#include <towr/models/dynamic_model.h>
#include <towr/terrain/height_map.h>
#include <towr/variables/spline_holder.h>

namespace towr {

struct SolverConfig {
  int    max_iter     = 3000;
  double max_cpu_time = 30.0;
  double tol          = 1e-3;
  std::string jacobian_approx = "exact";
};

struct JumpConfig {
  double standing_height;                          // required
  Eigen::Vector3d displacement = {0, 0, 0};        // base displacement start->end
  double yaw_rotation          = 0.0;              // yaw change [rad]
  double crouch_ratio          = 0.6;              // crouch_h = standing_h * ratio
  int    num_jumps             = 1;

  double crouch_duration  = 0.3;
  double push_duration    = 0.2;
  double flight_duration  = 0.3;
  double absorb_duration  = 0.2;
  double recover_duration = 0.3;

  // Number of polynomials per swing phase for EE motion/orientation.
  // 0  → use TowrOptimizer default (kPolysPerSwing).
  // >0 → override, e.g. 1 for a single smooth swing spline.
  int swing_polys = 0;

  double force_limit = 1000.0;
  SolverConfig solver;
};

struct FlipConfig {
  enum Type { BackFlip, FrontFlip, SideFlipLeft, SideFlipRight };

  Type   type             = BackFlip;
  double standing_height;                          // required
  double rotation_amount  = 2.0 * M_PI;
  double crouch_ratio     = 0.5;
  double tuck_ratio       = 0.45;  // mid-flight leg length = standing_h * tuck_ratio

  double crouch_duration  = 0.3;
  double push_duration    = 0.2;
  double flight_duration  = 0.8;
  double absorb_duration  = 0.2;
  double recover_duration = 0.3;

  // Same semantics as JumpConfig::swing_polys.
  int swing_polys = 0;

  double force_limit = 2000.0;
  SolverConfig solver = {5000, 60.0, 1e-3, "exact"};
};

class TowrOptimizer {
public:
  TowrOptimizer() = default;

  void setModel(KinematicModel::Ptr kin, DynamicModel::Ptr dyn);
  void setTerrain(HeightMap::Ptr terrain);
  void setInitialEE(const std::vector<Eigen::Vector3d>& ee_pos);

  bool solveJump(const JumpConfig& cfg);
  bool solveFlip(const FlipConfig& cfg);

  const SplineHolder& solution() const { return solution_; }
  double totalTime() const { return total_time_; }

  bool saveCSV(const std::string& filename, double dt = 0.01) const;
  void printTrajectory(double dt = 0.05) const;

private:
  KinematicModel::Ptr kin_;
  DynamicModel::Ptr   dyn_;
  HeightMap::Ptr      terrain_;
  std::vector<Eigen::Vector3d> initial_ee_;
  bool ee_set_ = false;

  SplineHolder solution_;
  double total_time_ = 0.0;

  static constexpr double kBasePolyDt = 0.1;
  static constexpr int    kPolysPerSwing = 3;
  static constexpr int    kPolysPerStanceForce = 4;

  std::vector<Eigen::Vector3d> computeInitialEE(
      const Eigen::Vector3d& base_pos) const;

  static std::vector<double> makeBasePolyDurations(double T);
  static int timeToNodeId(double t);
};

} // namespace towr
