#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <towr/terrain/examples/height_map_examples.h>
#include <towr/nlp_formulation.h>
#include <ifopt/ipopt_solver.h>
#include <towr/costs/node_cost.h>
#include <towr/costs/base_height_cost.h>
#include <towr/variables/variable_names.h>

using namespace towr;

class FiveStepStairs : public HeightMap {
public:
  FiveStepStairs() {
    step_depth_ = 0.3;  // 每阶纵深0.3m
    step_height_ = 0.15; // 每阶高度0.15m
    num_steps_ = 5;      // 总共5阶
    stairs_start_ = 0.5; // 楼梯从x=0.5m开始
  }

  double GetHeight(double x, double y) const override {
    // 楼梯前的平地
    if (x < stairs_start_) {
      return 0.0;
    }
    
    // 计算在第几阶楼梯上
    double relative_x = x - stairs_start_;
    int step_number = static_cast<int>(relative_x / step_depth_);
    
    // 超过楼梯范围，保持最后一阶高度
    if (step_number >= num_steps_) {
      return num_steps_ * step_height_;
    }
    
    // 返回对应楼梯的高度
    return (step_number + 1) * step_height_;
  }

private:
  double step_depth_;   // 每阶纵深
  double step_height_;  // 每阶高度
  int num_steps_;       // 楼梯总数
  double stairs_start_; // 楼梯开始位置
};

int main()
{
  NlpFormulation formulation;

  // terrain - 使用自定义的5阶楼梯地形
  formulation.terrain_ = std::make_shared<FiveStepStairs>();

  // Kinematic limits and dynamic parameters of the hopper
  formulation.model_ = RobotModel(RobotModel::Biped);
  formulation.params_.force_polynomials_per_stance_phase_=4;
  formulation.params_.torque_polynomials_per_stance_phase_=4;
  formulation.params_.ee_polynomials_per_swing_phase_=2;

  // formulation.params_.duration_base_polynomial_      = 0.050;
  // formulation.params_.dt_constraint_range_of_motion_ = 0.005;
  // formulation.params_.dt_constraint_dynamic_         = 0.050;
  // formulation.params_.dt_constraint_base_motion_     = 0.050;

  // Parameters that define the motion. See c'tor for default values or
  // other values that can however be modified.
  double z0 = 0.65;
  double x0 = 0.0;
  double step_t = 0.60;
  double stand_t = 0.5;
  double ds_rate = 0.2;
  double st = step_t*(1+ds_rate);
  double ft = step_t*(1-ds_rate);
  double phase_diff_rate = 1.0;
  double xend = 2.0+x0;
  double step_len = 0.5*step_t;
  
  formulation.params_.ee_phase_durations_.push_back({stand_t, ft});
  formulation.params_.ee_phase_durations_.push_back({stand_t+step_t*phase_diff_rate, ft});
  formulation.params_.ee_stance_position_.push_back({{x0, 0.1}});
  formulation.params_.ee_stance_position_.push_back({{x0,-0.1}});
  auto &PhaseL = formulation.params_.ee_phase_durations_[0];
  auto &PhaseR = formulation.params_.ee_phase_durations_[1];
  auto &StanceL = formulation.params_.ee_stance_position_[0];
  auto &StanceR = formulation.params_.ee_stance_position_[1];

  for(double stance_x = x0; stance_x < xend; stance_x += 2*step_len)
  {
    PhaseL.push_back(st); PhaseL.push_back(ft);
    PhaseR.push_back(st); PhaseR.push_back(ft);

    StanceL.push_back({std::max(stance_x-step_len, x0),  0.1});
    StanceR.push_back({stance_x, -0.1});
    std::cout<<"Left stance: "<<stance_x-step_len<<", Right stance: "<<stance_x<<std::endl;
  }
  PhaseL.push_back(stand_t+step_t*phase_diff_rate);
  PhaseR.push_back(stand_t);
  StanceL.push_back({xend,  0.1});
  StanceR.push_back({xend, -0.1});

  // formulation.params_.ee_phase_durations_.push_back({
  //   stand_t, ft,
  //   st, ft,  
  //   st, ft,  
  //   st, ft,  
  //   st, ft,  
  //   st, ft,  
  //   st, ft,  
  //   st, ft,  
  //   stand_t+step_t*phase_diff_rate
  // });
  // formulation.params_.ee_stance_position_.push_back({
  //   {0.0+x0, 0.1}, 
  //   {0.0+x0, 0.1}, 
  //   {0.0+x0, 0.1}, 
  //   {0.3+x0, 0.1},
  //   {0.9+x0, 0.1},
  //   {1.5+x0, 0.1},
  //   {2.0+x0, 0.1},
  //   {2.0+x0, 0.1},
  //   {2.0+x0, 0.1}
  // });
  
  // formulation.params_.ee_phase_durations_.push_back({
  //   stand_t+step_t*phase_diff_rate, ft,
  //   st, ft, 
  //   st, ft, 
  //   st, ft, 
  //   st, ft, 
  //   st, ft, 
  //   st, ft, 
  //   st, ft, 
  //   stand_t
  // });
  // formulation.params_.ee_stance_position_.push_back({
  //   {0.0+x0, -0.1}, 
  //   {0.0+x0, -0.1}, 
  //   {0.0+x0, -0.1}, 
  //   {0.6+x0, -0.1}, 
  //   {1.2+x0, -0.1},
  //   {1.8+x0, -0.1},
  //   {2.0+x0, -0.1},
  //   {2.0+x0, -0.1},
  //   {2.0+x0, -0.1}
  //   // {0.0+x0, -0.1}, 
  //   // {0.3+x0, -0.1},
  //   // {0.9+x0, -0.1},
  //   // {1.5+x0, -0.1},
  //   // {2.0+x0, -0.1}
  // });

  // set the initial position of the hopper - 在楼梯下方开始
  formulation.initial_base_.lin.at(kPos) << x0, 0.0, z0+FiveStepStairs().GetHeight(x0, 0.0);  // 在楼梯前的平地上
  formulation.initial_base_.lin.at(kVel) << 0.0, 0.0, 0.0;
  formulation.initial_base_.ang.at(kPos) << 0.0, 0.0, 0.0;
  formulation.initial_base_.ang.at(kVel) << 0.0, 0.0, 0.0;
  formulation.initial_ee_W_.push_back(Eigen::Vector3d(x0, 0.1, FiveStepStairs().GetHeight(x0, 0.0))); // 左足端在地面
  formulation.initial_ee_W_.push_back(Eigen::Vector3d(x0,-0.1, FiveStepStairs().GetHeight(x0, 0.0))); // 右足端在地面

  // define the desired goal state of the hopper - 在楼梯顶部结束
  formulation.final_base_.lin.at(towr::kPos) << xend, 0.0, z0+FiveStepStairs().GetHeight(xend, 0.0); // 楼梯后，高度0.75m + 0.5m身高
  formulation.final_base_.lin.at(towr::kVel) << 0.0, 0.0, 0.0;
  formulation.final_base_.ang.at(towr::kPos) << 0.0, 0.0, 0.0;
  formulation.final_base_.ang.at(towr::kVel) << 0.0, 0.0, 0.0;

  formulation.params_.ee_in_contact_at_start_.push_back(true);
  formulation.params_.ee_in_contact_at_start_.push_back(true);
  
  // 手动添加力矩约束（默认参数中没有包含）
  formulation.params_.constraints_.push_back(towr::Parameters::Torque);
  formulation.params_.constraints_.push_back(towr::Parameters::TerrainHard);

  formulation.params_.costs_.push_back(std::make_pair(towr::Parameters::ForcesCostID, 5*1e-9));
  formulation.params_.costs_.push_back(std::make_pair(towr::Parameters::EEMotionCostID, 1e-4));

  // formulation.params_.OptimizePhaseDurations();

  // Initialize the nonlinear-programming problem with the variables,
  // constraints and costs.
  ifopt::Problem nlp;
  SplineHolder solution;
  for (auto c : formulation.GetVariableSets(solution))
    nlp.AddVariableSet(c);
  for (auto c : formulation.GetConstraints(solution))
    nlp.AddConstraintSet(c);
  for (auto c : formulation.GetCosts())
    nlp.AddCostSet(c);

  // You can add your own elements to the nlp as well, simply by calling:
  // nlp.AddVariablesSet(your_custom_variables);
  // nlp.AddConstraintSet(your_custom_constraints);
  
  // 添加基座高度成本函数 - 保持基座到支撑点的高度在指定值附近
  double target_base_height = z0; // 目标基座到支撑点的高度 [m]
  double base_height_weight = 1e-1; // 基座高度成本权重 - 增大权重以更强烈地保持高度
  nlp.AddCostSet(std::make_shared<BaseHeightCost>(solution, 
                                                  formulation.terrain_, 
                                                  formulation.model_, 
                                                  target_base_height, 
                                                  base_height_weight));
  
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_lin_nodes, kVel, X, 1e-4));
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_lin_nodes, kVel, Y, 1e-2));
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_lin_nodes, kVel, Z, 1e-3));
  
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_ang_nodes, kPos, X, 1e-3));
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_ang_nodes, kPos, Y, 1e-3));
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_ang_nodes, kPos, Z, 1e-3));
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_ang_nodes, kVel, X, 1e-4));
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_ang_nodes, kVel, Y, 1e-4));
  nlp.AddCostSet(std::make_shared<NodeCost>(id::base_ang_nodes, kVel, Z, 1e-4));

  // Choose ifopt solver (IPOPT or SNOPT), set some parameters and solve.
  // solver->SetOption("derivative_test", "first-order");
  auto solver = std::make_shared<ifopt::IpoptSolver>();
  solver->SetOption("jacobian_approximation", "exact"); // "finite difference-values"
  solver->SetOption("max_cpu_time", 120.0);  // 爬楼梯优化需要更多时间
  solver->SetOption("max_iter", 3000);      // 增加最大迭代次数
  solver->SetOption("tol", 1e-4);           // 适当放宽容差
  solver->Solve(nlp);

  // Can directly view the optimization variables through:
  // Eigen::VectorXd x = nlp.GetVariableValues()
  // However, it's more convenient to access the splines constructed from these
  // variables and query their values at specific times:
  using namespace std;
  cout.precision(2);
  nlp.PrintCurrent(); // view variable-set, constraint violations, indices,...
  cout << fixed;
  cout << "\n====================\nMonoped trajectory:\n====================\n";

  // ====================
  // CSV Trajectory Export
  // ====================
  double T_sample = 0.001; // 1ms sampling period
  double T_total = solution.base_linear_->GetTotalTime();
  
  // Create CSV file
  ofstream csv_file("biped_trajectory.csv");
  csv_file << fixed << setprecision(6);
  
  // Write CSV header
  csv_file << "time,"
           << "base_pos_x,base_pos_y,base_pos_z,"
           << "base_vel_x,base_vel_y,base_vel_z,"
           << "base_acc_x,base_acc_y,base_acc_z,"
           << "base_euler_roll,base_euler_pitch,base_euler_yaw,"
           << "base_omega_x,base_omega_y,base_omega_z,"
           << "base_omegadot_x,base_omegadot_y,base_omegadot_z,"
           << "ee_pos_x_L,ee_pos_y_L,ee_pos_z_L,"
           << "ee_vel_x_L,ee_vel_y_L,ee_vel_z_L,"
           << "ee_acc_x_L,ee_acc_y_L,ee_acc_z_L,"
           << "contact_force_x_L,contact_force_y_L,contact_force_z_L,"
           << "contact_torque_x_L,contact_torque_y_L,contact_torque_z_L,"
           << "ee_pos_x_R,ee_pos_y_R,ee_pos_z_R,"
           << "ee_vel_x_R,ee_vel_y_R,ee_vel_z_R,"
           << "ee_acc_x_R,ee_acc_y_R,ee_acc_z_R,"
           << "contact_force_x_R,contact_force_y_R,contact_force_z_R,"
           << "contact_torque_x_R,contact_torque_y_R,contact_torque_z_R,"
           << "is_contact_phase_L,is_contact_phase_R"
           << endl;

  cout << "Exporting trajectory to CSV file..." << endl;
  cout << "Total time: " << T_total << " seconds" << endl;
  cout << "Sampling period: " << T_sample << " seconds" << endl;
  
  int sample_count = 0;
  double t = 0.0;
  while (t <= T_total + 1e-9) {
    // Base linear motion
    auto base_linear_state = solution.base_linear_->GetPoint(t);
    Eigen::Vector3d base_pos = base_linear_state.p();
    Eigen::Vector3d base_vel = base_linear_state.v();
    Eigen::Vector3d base_acc = base_linear_state.a();
    
    // Base angular motion
    auto base_angular_state = solution.base_angular_->GetPoint(t);
    Eigen::Vector3d base_euler = base_angular_state.p();
    Eigen::Vector3d base_omega = base_angular_state.v();
    Eigen::Vector3d base_omegadot = base_angular_state.a();
    
    // End-effector motion
    auto ee_state_L = solution.ee_motion_.at(0)->GetPoint(t);
    Eigen::Vector3d ee_pos_L = ee_state_L.p();
    Eigen::Vector3d ee_vel_L = ee_state_L.v();
    Eigen::Vector3d ee_acc_L = ee_state_L.a();
    
    // Contact forces and torques
    Eigen::Vector3d contact_force_L = solution.ee_force_.at(0)->GetPoint(t).p();
    Eigen::Vector3d contact_torque_L = solution.ee_torque_.at(0)->GetPoint(t).p();

    auto ee_state_R = solution.ee_motion_.at(1)->GetPoint(t);
    Eigen::Vector3d ee_pos_R = ee_state_R.p();
    Eigen::Vector3d ee_vel_R = ee_state_R.v();
    Eigen::Vector3d ee_acc_R = ee_state_R.a();
    
    Eigen::Vector3d contact_force_R = solution.ee_force_.at(1)->GetPoint(t).p();
    Eigen::Vector3d contact_torque_R = solution.ee_torque_.at(1)->GetPoint(t).p();
    
    // Contact phase
    bool is_contact_L = solution.phase_durations_.at(0)->IsContactPhase(t);
    bool is_contact_R = solution.phase_durations_.at(1)->IsContactPhase(t);
    
    // Write to CSV
    csv_file << t << ","
             << base_pos.x() << "," << base_pos.y() << "," << base_pos.z() << ","
             << base_vel.x() << "," << base_vel.y() << "," << base_vel.z() << ","
             << base_acc.x() << "," << base_acc.y() << "," << base_acc.z() << ","
             << base_euler.x() << "," << base_euler.y() << "," << base_euler.z() << ","
             << base_omega.x() << "," << base_omega.y() << "," << base_omega.z() << ","
             << base_omegadot.x() << "," << base_omegadot.y() << "," << base_omegadot.z() << ","
             << ee_pos_L.x() << "," << ee_pos_L.y() << "," << ee_pos_L.z() << ","
             << ee_vel_L.x() << "," << ee_vel_L.y() << "," << ee_vel_L.z() << ","
             << ee_acc_L.x() << "," << ee_acc_L.y() << "," << ee_acc_L.z() << ","
             << contact_force_L.x() << "," << contact_force_L.y() << "," << contact_force_L.z() << ","
             << contact_torque_L.x() << "," << contact_torque_L.y() << "," << contact_torque_L.z() << ","
             << ee_pos_R.x() << "," << ee_pos_R.y() << "," << ee_pos_R.z() << ","
             << ee_vel_R.x() << "," << ee_vel_R.y() << "," << ee_vel_R.z() << ","
             << ee_acc_R.x() << "," << ee_acc_R.y() << "," << ee_acc_R.z() << ","
             << contact_force_R.x() << "," << contact_force_R.y() << "," << contact_force_R.z() << ","
             << contact_torque_R.x() << "," << contact_torque_R.y() << "," << contact_torque_R.z() << ","
             << (is_contact_L ? 1 : 0) << "," << (is_contact_R ? 1 : 0)
             << endl;
    
    t += T_sample;
    sample_count++;
  }
  
  csv_file.close();
  cout << "Trajectory exported successfully!" << endl;
  cout << "Total samples: " << sample_count << endl;
  cout << "File: biped_trajectory.csv" << endl;

  // Summary statistics
  cout << "\n====================\nTrajectory Summary:\n====================\n";
  cout << "Total trajectory time: " << T_total << " seconds" << endl;
  cout << "CSV sampling rate: " << (1.0/T_sample) << " Hz" << endl;
  cout << "Total CSV samples: " << sample_count << endl;
  cout << "CSV file size: ~" << (sample_count * 25 * 8 / 1024) << " KB" << endl;
}
