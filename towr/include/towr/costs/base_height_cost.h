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

#ifndef TOWR_COSTS_BASE_HEIGHT_COST_H_
#define TOWR_COSTS_BASE_HEIGHT_COST_H_

#include <ifopt/cost_term.h>
#include <towr/variables/spline_holder.h>
#include <towr/terrain/height_map.h>
#include <towr/models/robot_model.h>

namespace towr {

/**
 * @brief 成本函数：保持基座到支撑点的高度在指定值附近
 *
 * 这个成本函数计算基座高度与目标高度之间的偏差，并对其进行惩罚。
 * 目标高度 = 支撑点平均高度 + 指定的基座到支撑点高度
 *
 * @ingroup Costs
 */
class BaseHeightCost : public ifopt::CostTerm {
public:
  using Vector3d = Eigen::Vector3d;

  /**
   * @brief 构造函数
   * @param spline_holder 包含基座和足端运动轨迹的spline holder
   * @param terrain 地形高度图
   * @param robot_model 机器人模型
   * @param target_height 目标基座到支撑点的高度 [m]
   * @param weight 成本权重
   * @param dt 时间离散化间隔 [s]
   */
  BaseHeightCost(const SplineHolder& spline_holder,
                 const HeightMap::Ptr& terrain,
                 const RobotModel& robot_model,
                 double target_height,
                 double weight,
                 double dt = 0.01);

  virtual ~BaseHeightCost() = default;

  void InitVariableDependedQuantities(const VariablesPtr& x) override;

  double GetCost() const override;

private:
  NodeSpline::Ptr base_linear_;
  std::vector<NodeSpline::Ptr> ee_motion_;
  std::vector<PhaseDurations::Ptr> phase_durations_;
  HeightMap::Ptr terrain_;
  RobotModel robot_model_;
  double target_height_;
  double weight_;
  double dt_;
  double total_time_;
  int ee_count_;

  void FillJacobianBlock(std::string var_set, Jacobian&) const override;

  /**
   * @brief 计算在给定时间点的基座高度偏差
   * @param t 时间 [s]
   * @return 高度偏差 [m]
   */
  double GetHeightDeviation(double t) const;

  /**
   * @brief 计算在给定时间点的支撑点平均高度
   * @param t 时间 [s]
   * @return 支撑点平均高度 [m]
   */
  double GetSupportPointAverageHeight(double t) const;

  /**
   * @brief 检查足端是否在接触状态
   * @param ee 足端索引
   * @param t 时间 [s]
   * @return 是否在接触状态
   */
  bool IsEndeffectorInContact(int ee, double t) const;
};

} /* namespace towr */

#endif /* TOWR_COSTS_BASE_HEIGHT_COST_H_ */ 