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
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include <towr/costs/ee_base_pos_cost.h>

#include <towr/variables/variable_names.h>
#include <towr/variables/cartesian_dimensions.h>

namespace towr {

EEBasePosCost::EEBasePosCost(const SplineHolder& splines,
                             EE ee,
                             const Vector3d& p_ref_B,
                             double weight,
                             double dt)
    : CostTerm("eeBasePos-" + std::to_string(ee))
{
  splines_ = splines;
  ee_ = ee;
  p_ref_B_ = p_ref_B;
  weight_ = weight;
  dt_ = dt;
}

std::vector<double>
EEBasePosCost::GetSampleTimes() const
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
EEBasePosCost::GetCost() const
{
  if (weight_ <= 0.0) {
    return 0.0;
  }

  double cost = 0.0;
  EulerConverter base_ang(splines_.base_angular_);

  for (double t : GetSampleTimes()) {
    // only during swing
    if (splines_.phase_durations_.at(ee_)->IsContactPhase(t)) {
      continue;
    }

    Vector3d p_base = splines_.base_linear_->GetPoint(t).p();
    Vector3d p_ee = splines_.ee_motion_.at(ee_)->GetPoint(t).p();
    Vector3d r_W = p_ee - p_base;
    Eigen::Matrix3d b_R_w = base_ang.GetRotationMatrixBaseToWorld(t).transpose();
    Vector3d p_B = b_R_w * r_W;

    Vector3d e = p_B - p_ref_B_;
    cost += weight_ * e.squaredNorm();
  }

  return cost;
}

void
EEBasePosCost::FillJacobianBlock(std::string var_set, Jacobian& jac) const
{
  if (weight_ <= 0.0) {
    return;
  }

  EulerConverter base_ang(splines_.base_angular_);

  for (double t : GetSampleTimes()) {
    // only during swing
    if (splines_.phase_durations_.at(ee_)->IsContactPhase(t)) {
      continue;
    }

    Vector3d p_base = splines_.base_linear_->GetPoint(t).p();
    Vector3d p_ee = splines_.ee_motion_.at(ee_)->GetPoint(t).p();
    Vector3d r_W = p_ee - p_base;
    Eigen::Matrix3d b_R_w = base_ang.GetRotationMatrixBaseToWorld(t).transpose();
    Vector3d p_B = b_R_w * r_W;
    Vector3d e = p_B - p_ref_B_;

    // common multiplier (d/dx weight*||e||^2 = 2*weight*e^T * de/dx)
    Eigen::RowVector3d m = (2.0*weight_) * e.transpose();

    if (var_set == id::EEMotionNodes(ee_)) {
      auto J = splines_.ee_motion_.at(ee_)->GetJacobianWrtNodes(t, kPos); // 3 x n
      Eigen::RowVector3d mW = m * b_R_w; // 1x3
      for (int r=0; r<k3D; ++r) {
        for (NodeSpline::Jacobian::InnerIterator it(J, r); it; ++it) {
          jac.coeffRef(0, it.col()) += mW(r) * it.value();
        }
      }
    }

    if (var_set == id::base_lin_nodes) {
      auto J = splines_.base_linear_->GetJacobianWrtNodes(t, kPos); // 3 x n
      Eigen::RowVector3d mW = -m * b_R_w; // 1x3
      for (int r=0; r<k3D; ++r) {
        for (NodeSpline::Jacobian::InnerIterator it(J, r); it; ++it) {
          jac.coeffRef(0, it.col()) += mW(r) * it.value();
        }
      }
    }

    if (var_set == id::base_ang_nodes) {
      // derivative of (b_R_w * r_W) w.r.t base euler nodes
      auto Jrot = base_ang.DerivOfRotVecMult(t, r_W, /*inverse=*/true); // 3 x n
      for (int r=0; r<k3D; ++r) {
        for (EulerConverter::Jacobian::InnerIterator it(Jrot, r); it; ++it) {
          jac.coeffRef(0, it.col()) += m(r) * it.value();
        }
      }
    }

    if (var_set == id::EESchedule(ee_)) {
      // timing sensitivity: p_B changes with time shift; approximate via -e^T * dp_B/dt * dt/dT
      // We intentionally omit this block to keep it simple (still works well when timings are fixed).
    }
  }
}

} // namespace towr


