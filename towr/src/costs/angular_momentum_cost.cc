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

#include <towr/costs/angular_momentum_cost.h>

#include <towr/models/single_rigid_body_dynamics.h>
#include <towr/variables/euler_converter.h>
#include <towr/variables/variable_names.h>

namespace towr {

AngularMomentumCost::AngularMomentumCost(const SplineHolder& splines,
                                         const DynamicModel::Ptr& model,
                                         double weight,
                                         double dt)
    : CostTerm("angular_momentum_cost")
{
  splines_ = splines;
  model_ = model;
  weight_ = weight;
  dt_ = dt;
}

std::vector<double>
AngularMomentumCost::GetSampleTimes() const
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
AngularMomentumCost::GetCost() const
{
  if (weight_ <= 0.0) {
    return 0.0;
  }

  EulerConverter base_ang(splines_.base_angular_);
  const double wdt = weight_ * (dt_ > 0.0 ? dt_ : 1.0);

  // Prefer true angular momentum if SRBD model is used.
  const auto* srbd = dynamic_cast<const SingleRigidBodyDynamics*>(model_.get());
  Eigen::Matrix3d I_b = Eigen::Matrix3d::Identity();
  bool use_true_L = false;
  if (srbd) {
    I_b = srbd->GetInertiaB();
    use_true_L = true;
  }

  double cost = 0.0;
  for (double t : GetSampleTimes()) {
    Eigen::Matrix3d R = base_ang.GetRotationMatrixBaseToWorld(t).toDense();
    Vector3d omega = base_ang.GetAngularVelocityInWorld(t);

    Vector3d v = omega;
    if (use_true_L) {
      Eigen::Matrix3d I_w = R * I_b * R.transpose();
      v = I_w * omega; // angular momentum
    }

    cost += wdt * v.squaredNorm();
  }

  return cost;
}

void
AngularMomentumCost::FillJacobianBlock(std::string var_set, Jacobian& jac) const
{
  if (weight_ <= 0.0) {
    return;
  }

  if (var_set != id::base_ang_nodes) {
    return;
  }

  EulerConverter base_ang(splines_.base_angular_);
  const double wdt = weight_ * (dt_ > 0.0 ? dt_ : 1.0);

  const auto* srbd = dynamic_cast<const SingleRigidBodyDynamics*>(model_.get());
  Eigen::Matrix3d I_b = Eigen::Matrix3d::Identity();
  bool use_true_L = false;
  if (srbd) {
    I_b = srbd->GetInertiaB();
    use_true_L = true;
  }

  const int n = splines_.base_angular_->GetNodeVariablesCount();

  for (double t : GetSampleTimes()) {
    Eigen::Matrix3d R = base_ang.GetRotationMatrixBaseToWorld(t).toDense();
    Eigen::Matrix3d R_T = R.transpose();
    Vector3d omega = base_ang.GetAngularVelocityInWorld(t);

    // Compute vector to penalize: v = L or omega
    Vector3d v = omega;
    Eigen::Matrix3d I_w = Eigen::Matrix3d::Identity();
    if (use_true_L) {
      I_w = R * I_b * R.transpose();
      v = I_w * omega;
    }

    // d/dx (||v||^2) = 2 v^T dv/dx
    Eigen::RowVector3d m = (2.0 * wdt) * v.transpose(); // 1x3

    // dv/dx
    // If fallback: v=omega, dv/dx is simply d omega / dx
    if (!use_true_L) {
      auto Jw = base_ang.GetDerivOfAngVelWrtEulerNodes(t); // 3 x n (sparse)
      for (int r=0; r<k3D; ++r) {
        for (EulerConverter::Jacobian::InnerIterator it(Jw, r); it; ++it) {
          jac.coeffRef(0, it.col()) += m(r) * it.value();
        }
      }
      continue;
    }

    // True angular momentum: L = R * I_b * R^T * omega
    // Let u = R^T*omega, v2 = I_b*u, L = R*v2
    Vector3d u = R_T * omega;
    Vector3d v2 = I_b * u;

    auto J_R_v2 = base_ang.DerivOfRotVecMult(t, v2, /*inverse=*/false); // 3 x n (sparse)
    auto J_Rt_omega = base_ang.DerivOfRotVecMult(t, omega, /*inverse=*/true); // 3 x n (sparse)
    auto J_omega = base_ang.GetDerivOfAngVelWrtEulerNodes(t); // 3 x n (sparse)

    // Build dense 3 x n for the second term: R * I_b * ( d(R^T*omega)/dx + R^T * d(omega)/dx )
    Eigen::MatrixXd J2 = Eigen::MatrixXd::Zero(k3D, n);

    // term_a = d(R^T*omega)/dx  (3 x n)
    Eigen::MatrixXd term_a = Eigen::MatrixXd::Zero(k3D, n);
    for (int r=0; r<k3D; ++r) {
      for (EulerConverter::Jacobian::InnerIterator it(J_Rt_omega, r); it; ++it) {
        term_a(r, it.col()) += it.value();
      }
    }

    // term_b = R^T * d(omega)/dx (3 x n)
    Eigen::MatrixXd term_b = Eigen::MatrixXd::Zero(k3D, n);
    for (int r=0; r<k3D; ++r) {
      for (EulerConverter::Jacobian::InnerIterator it(J_omega, r); it; ++it) {
        term_b(r, it.col()) += it.value();
      }
    }
    term_b = R_T * term_b;

    Eigen::MatrixXd term_u = term_a + term_b; // 3 x n
    J2 = R * (I_b * term_u);

    // Now accumulate gradient: m * (J_R_v2 + J2)
    // J_R_v2 is sparse 3xn
    for (int r=0; r<k3D; ++r) {
      for (EulerConverter::Jacobian::InnerIterator it(J_R_v2, r); it; ++it) {
        jac.coeffRef(0, it.col()) += m(r) * it.value();
      }
    }
    // J2 dense
    Eigen::RowVectorXd grad2 = m * J2; // 1 x n
    for (int col=0; col<grad2.cols(); ++col) {
      double vcol = grad2(col);
      if (vcol != 0.0) {
        jac.coeffRef(0, col) += vcol;
      }
    }
  }
}

} // namespace towr


