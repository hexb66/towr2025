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

#include <towr/costs/base_height_cost.h>
#include <towr/variables/variable_names.h>
#include <cmath>

namespace towr {

BaseHeightCost::BaseHeightCost(const SplineHolder& spline_holder,
                               const HeightMap::Ptr& terrain,
                               const RobotModel& robot_model,
                               double target_height,
                               double weight,
                               double dt)
    : ifopt::CostTerm("base_height_cost"),
      base_linear_(spline_holder.base_linear_),
      ee_motion_(spline_holder.ee_motion_),
      phase_durations_(spline_holder.phase_durations_),
      terrain_(terrain),
      robot_model_(robot_model),
      target_height_(target_height),
      weight_(weight),
      dt_(dt),
      total_time_(spline_holder.base_linear_->GetTotalTime()),
      ee_count_(spline_holder.ee_motion_.size())
{
  // 成本函数不需要额外的变量初始化
}

void
BaseHeightCost::InitVariableDependedQuantities(const VariablesPtr& x)
{
  // 这个成本函数不需要额外的变量初始化
}

double
BaseHeightCost::GetCost() const
{
  double total_cost = 0.0;
  
  // 在整个轨迹上积分高度偏差的平方
  double t = 0.0;
  while (t <= total_time_ + 1e-9) {
    double deviation = GetHeightDeviation(t);
    total_cost += weight_ * deviation * deviation * dt_;
    t += dt_;
  }
  
  return total_cost;
}

void
BaseHeightCost::FillJacobianBlock(std::string var_set, Jacobian& jac) const
{
  if (var_set == id::base_lin_nodes) {
    double t = 0.0;
    
    while (t <= total_time_ + 1e-9) {
      double deviation = GetHeightDeviation(t);
      
      // 获取基座位置的雅可比矩阵
      auto jac_pos = base_linear_->GetJacobianWrtNodes(t, kPos);
      
      // 只对Z方向（高度）的雅可比进行加权
      // 成本函数对基座Z位置的偏导数为：2 * weight * deviation * dt
      jac.middleRows(0, 1) += 2.0 * weight_ * deviation * dt_ * jac_pos.row(Z);
      
      t += dt_;
    }
  }
}

double
BaseHeightCost::GetHeightDeviation(double t) const
{
  // 获取基座当前位置
  Vector3d base_pos = base_linear_->GetPoint(t).p();
  
  // 计算目标高度：支撑点平均高度 + 指定的基座到支撑点高度
  double target_z = GetSupportPointAverageHeight(t) + target_height_;
  
  // 返回高度偏差
  return base_pos.z() - target_z;
}

double
BaseHeightCost::GetSupportPointAverageHeight(double t) const
{
  double total_height = 0.0;
  int contact_count = 0;
  
  // 计算所有接触足端的平均高度
  for (int ee = 0; ee < ee_count_; ++ee) {
    if (IsEndeffectorInContact(ee, t)) {
      Vector3d ee_pos = ee_motion_.at(ee)->GetPoint(t).p();
      total_height += ee_pos.z();
      contact_count++;
    }
  }
  
  // 如果没有接触的足端，使用地形高度作为备选
  if (contact_count == 0) {
    Vector3d base_pos = base_linear_->GetPoint(t).p();
    return terrain_->GetHeight(base_pos.x(), base_pos.y());
  }
  
  return total_height / contact_count;
}

bool
BaseHeightCost::IsEndeffectorInContact(int ee, double t) const
{
  return phase_durations_.at(ee)->IsContactPhase(t);
}

} /* namespace towr */ 