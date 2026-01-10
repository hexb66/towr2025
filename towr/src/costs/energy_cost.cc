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

#include <towr/costs/energy_cost.h>

#include <towr/variables/variable_names.h>
#include <towr/variables/cartesian_dimensions.h>

namespace towr {

EnergyCost::EnergyCost(const SplineHolder& splines,
                       double weight,
                       double torque_weight,
                       double dt)
    : CostTerm("energy_cost")
{
  splines_ = splines;
  weight_ = weight;
  torque_weight_ = torque_weight;
  dt_ = dt;
}

std::vector<double>
EnergyCost::GetSampleTimes() const
{
  std::vector<double> ts;
  double T = splines_.base_linear_->GetTotalTime();
  if (dt_ <= 0.0) {
    ts.push_back(0.0);
    ts.push_back(T);
    return ts;
  }
  for (double t=0.0; t<=T+1e-9; t+=dt_) {
    ts.push_back(t);
  }
  return ts;
}

double
EnergyCost::GetCost() const
{
  if (weight_ <= 0.0) {
    return 0.0;
  }

  double cost = 0.0;
  const int n_ee = splines_.ee_force_.size();

  for (double t : GetSampleTimes()) {
    double inst = 0.0;
    for (int ee=0; ee<n_ee; ++ee) {
      Vector3d f = splines_.ee_force_.at(ee)->GetPoint(t).p();
      Vector3d tau = splines_.ee_torque_.at(ee)->GetPoint(t).p();
      inst += f.squaredNorm() + torque_weight_ * tau.squaredNorm();
    }
    cost += weight_ * inst * (dt_ > 0.0 ? dt_ : 1.0);
  }

  return cost;
}

void
EnergyCost::FillJacobianBlock(std::string var_set, Jacobian& jac) const
{
  if (weight_ <= 0.0) {
    return;
  }

  const double wdt = weight_ * (dt_ > 0.0 ? dt_ : 1.0);
  const int n_ee = splines_.ee_force_.size();

  for (double t : GetSampleTimes()) {
    for (int ee=0; ee<n_ee; ++ee) {
      const Vector3d f = splines_.ee_force_.at(ee)->GetPoint(t).p();
      const Vector3d tau = splines_.ee_torque_.at(ee)->GetPoint(t).p();

      if (var_set == id::EEForceNodes(ee)) {
        auto J = splines_.ee_force_.at(ee)->GetJacobianWrtNodes(t, kPos); // 3 x n
        Eigen::RowVector3d m = (2.0*wdt) * f.transpose(); // 1x3
        for (int r=0; r<k3D; ++r) {
          for (NodeSpline::Jacobian::InnerIterator it(J, r); it; ++it) {
            jac.coeffRef(0, it.col()) += m(r) * it.value();
          }
        }
      }

      if (var_set == id::EETorqueNodes(ee) && torque_weight_ != 0.0) {
        auto J = splines_.ee_torque_.at(ee)->GetJacobianWrtNodes(t, kPos); // 3 x n
        Eigen::RowVector3d m = (2.0*wdt*torque_weight_) * tau.transpose(); // 1x3
        for (int r=0; r<k3D; ++r) {
          for (NodeSpline::Jacobian::InnerIterator it(J, r); it; ++it) {
            jac.coeffRef(0, it.col()) += m(r) * it.value();
          }
        }
      }

      // timing sensitivity (only when durations are optimized and splines are PhaseSpline)
      if (var_set == id::EESchedule(ee)) {
        // force
        {
          auto JdT = splines_.ee_force_.at(ee)->GetJacobianOfPosWrtDurations(t); // 3 x nT
          Eigen::RowVector3d m = (2.0*wdt) * f.transpose();
          for (int r=0; r<k3D; ++r) {
            for (NodeSpline::Jacobian::InnerIterator it(JdT, r); it; ++it) {
              jac.coeffRef(0, it.col()) += m(r) * it.value();
            }
          }
        }
        // torque
        if (torque_weight_ != 0.0) {
          auto JdT = splines_.ee_torque_.at(ee)->GetJacobianOfPosWrtDurations(t); // 3 x nT
          Eigen::RowVector3d m = (2.0*wdt*torque_weight_) * tau.transpose();
          for (int r=0; r<k3D; ++r) {
            for (NodeSpline::Jacobian::InnerIterator it(JdT, r); it; ++it) {
              jac.coeffRef(0, it.col()) += m(r) * it.value();
            }
          }
        }
      }
    }
  }
}

} // namespace towr


