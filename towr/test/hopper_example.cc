/******************************************************************************
Copyright (c) 2018, Alexander W. Winkler. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <towr/terrain/examples/height_map_examples.h>
#include <towr/nlp_formulation.h>
#include <ifopt/ipopt_solver.h>

#include <towr/costs/node_cost.h>
#include <towr/variables/variable_names.h>

#include <towr/utils/save_data.h>


using namespace towr;

/**
 * @brief Custom 5-step stairs terrain.
 * 
 * Creates stairs with 5 steps, each 0.3m deep and 0.15m high.
 * Total horizontal distance: 1.5m, total height: 0.75m
 */
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

// A minimal example how to build a trajectory optimization problem using TOWR.
//
// The more advanced example that includes ROS integration, GUI, rviz
// visualization and plotting can be found here:
// towr_ros/src/towr_ros_app.cc
int main()
{
  NlpFormulation formulation;

  // terrain - 使用自定义的5阶楼梯地形
  formulation.terrain_ = std::make_shared<FiveStepStairs>();

  // Kinematic limits and dynamic parameters of the hopper
  formulation.model_ = RobotModel(RobotModel::Monoped);

  // Parameters that define the motion. See c'tor for default values or
  // other values that can however be modified.
  formulation.params_.ee_phase_durations_.push_back({
    0.5, 0.3,  
    0.4, 0.3,  
    0.4, 0.3,  
    0.4, 0.3,  
    0.4, 0.3,  
    0.4, 0.3,  
    0.4    // 最后稳定在第五阶上
  });
  formulation.params_.ee_stance_position_.push_back({
    {0.0, 0.0},
    {0.4, 0.0},
    {0.7,-0.0},
    {0.7, 0.0},
    {1.3,-0.0},
    {1.3, 0.0},
    {0.0, 0.0}
  });

  // set the initial position of the hopper - 在楼梯下方开始
  formulation.initial_base_.lin.at(kPos) << 0.0, 0.0, 0.6;  // 在楼梯前的平地上
  formulation.initial_base_.lin.at(kVel) << 0.0, 0.0, 0.0;
  formulation.initial_base_.ang.at(kPos) << 0.0, 0.0, 0.0;
  formulation.initial_base_.ang.at(kVel) << 0.0, 0.0, 0.0;
  formulation.initial_ee_W_.push_back(Eigen::Vector3d(0.0, 0.0, 0.0)); // 足端在地面

  // define the desired goal state of the hopper - 在楼梯顶部结束
  formulation.final_base_.lin.at(towr::kPos) << 0.0, 0.0, 0.6+FiveStepStairs().GetHeight(0.0, 0.0); // 楼梯后，高度0.75m + 0.5m身高
  // formulation.final_base_.lin.at(towr::kPos) << 0.6, 0.0, 0.6+0.15;
  formulation.final_base_.lin.at(towr::kVel) << 0.0, 0.0, 0.0;
  formulation.final_base_.ang.at(towr::kPos) << 0.0, 0.0, 0.0;
  formulation.final_base_.ang.at(towr::kVel) << 0.0, 0.0, 0.0;

  formulation.params_.ee_in_contact_at_start_.push_back(true);

  // formulation.params_.ee_polynomials_per_swing_phase_ = 2;
  // formulation.params_.force_polynomials_per_stance_phase_ = 2;
  // formulation.params_.torque_polynomials_per_stance_phase_ = 2;
  
  // 手动添加力矩约束（默认参数中没有包含）
  formulation.params_.constraints_.push_back(towr::Parameters::Torque);

  formulation.params_.costs_.push_back(std::make_pair(towr::Parameters::ForcesCostID, 1e-9));
  formulation.params_.costs_.push_back(std::make_pair(towr::Parameters::EEMotionCostID, 1e-4));

  formulation.params_.OptimizePhaseDurations();

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
  for (auto dim:{X,Y,Z}){
    nlp.AddCostSet(std::make_shared<NodeCost>(id::base_lin_nodes, kPos, dim, 1e-4));
    nlp.AddCostSet(std::make_shared<NodeCost>(id::base_lin_nodes, kVel, dim, 1e-4));
    nlp.AddCostSet(std::make_shared<NodeCost>(id::base_ang_nodes, kPos, dim, 1e-4));
    nlp.AddCostSet(std::make_shared<NodeCost>(id::base_ang_nodes, kVel, dim, 1e-4));
  }

  // Choose ifopt solver (IPOPT or SNOPT), set some parameters and solve.
  // solver->SetOption("derivative_test", "first-order");
  auto solver = std::make_shared<ifopt::IpoptSolver>();
  solver->SetOption("jacobian_approximation", "exact"); // "finite difference-values"
  solver->SetOption("max_cpu_time", 60.0);  // 爬楼梯优化需要更多时间
  solver->SetOption("max_iter", 3000);      // 增加最大迭代次数
  solver->SetOption("tol", 1e-4);           // 适当放宽容差
  solver->Solve(nlp);

  nlp.PrintCurrent();

  // save data
  SaveTrajectoryToCSV(solution, "hopper_trajectory.csv", 0.001);
}
