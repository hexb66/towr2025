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

#include <towr/constraints/torque_constraint_discretized.h>

#include <towr/variables/variable_names.h>

namespace towr {

namespace {

inline void AccumulateLinearFormJacobian(const ifopt::ConstraintSet::Jacobian& J3xN,
                                         const Eigen::Vector3d& b,
                                         int dst_row,
                                         ifopt::ConstraintSet::Jacobian& out)
{
  for (int outer = 0; outer < J3xN.outerSize(); ++outer) {
    for (ifopt::ConstraintSet::Jacobian::InnerIterator it(J3xN, outer); it; ++it) {
      const int r = it.row(); // 0..2
      out.coeffRef(dst_row, it.col()) += b(r) * it.value();
    }
  }
}

inline void AccumulateScaledRowJacobian(const ifopt::ConstraintSet::Jacobian& J3xN,
                                        int src_row,
                                        double scale,
                                        int dst_row,
                                        ifopt::ConstraintSet::Jacobian& out)
{
  if (scale == 0.0) return;
  for (int outer = 0; outer < J3xN.outerSize(); ++outer) {
    for (ifopt::ConstraintSet::Jacobian::InnerIterator it(J3xN, outer); it; ++it) {
      if (it.row() == src_row) {
        out.coeffRef(dst_row, it.col()) += scale * it.value();
      }
    }
  }
}

} // namespace

TorqueConstraintDiscretized::TorqueConstraintDiscretized(
    const HeightMap::Ptr& terrain,
    double T, double dt,
    double tx_min, double tx_max,
    double ty_min, double ty_max,
    double k_friction,
    EE endeffector_id,
    const SplineHolder& spline_holder)
    : TimeDiscretizationConstraint(T, dt, "torque-disc-" + id::EETorqueNodes(endeffector_id))
{
  terrain_     = terrain;
  tx_min_      = tx_min;
  tx_max_      = tx_max;
  ty_min_      = ty_min;
  ty_max_      = ty_max;
  k_friction_  = k_friction;
  mu_          = terrain->GetFrictionCoeff();
  ee_          = endeffector_id;

  ee_torque_ = spline_holder.ee_torque_.at(ee_);
  ee_force_  = spline_holder.ee_force_.at(ee_);
  ee_motion_ = spline_holder.ee_motion_.at(ee_);

  // tx, ty, and two inequalities for normal torque friction: (tau_n - k*mu*f_n) <= 0 and (-tau_n - k*mu*f_n) <= 0
  n_constraints_per_instance_ = 4;
  SetRows(GetNumberOfNodes() * n_constraints_per_instance_);
}

int
TorqueConstraintDiscretized::GetRow(int k, int local_row) const
{
  return k*n_constraints_per_instance_ + local_row;
}

void
TorqueConstraintDiscretized::UpdateConstraintAtInstance(double t, int k, VectorXd& g) const
{
  const Vector3d p   = ee_motion_->GetPoint(t).p();
  const Vector3d f   = ee_force_->GetPoint(t).p();
  const Vector3d tau = ee_torque_->GetPoint(t).p();

  const Vector3d n  = terrain_->GetNormalizedBasis(HeightMap::Normal,   p.x(), p.y());
  const Vector3d t1 = terrain_->GetNormalizedBasis(HeightMap::Tangent1, p.x(), p.y());
  const Vector3d t2 = terrain_->GetNormalizedBasis(HeightMap::Tangent2, p.x(), p.y());

  const double tau_t1 = tau.dot(t1);
  const double tau_t2 = tau.dot(t2);
  const double tau_n  = tau.dot(n);
  const double f_n    = f.dot(n);

  const double tz_lim = k_friction_ * mu_ * f_n;

  int row = GetRow(k, 0);
  g(row++) = tau_t1;
  g(row++) = tau_t2;
  g(row++) = tau_n - tz_lim;
  g(row++) = -tau_n - tz_lim;
}

void
TorqueConstraintDiscretized::UpdateBoundsAtInstance(double /*t*/, int k, VecBound& bounds) const
{
  int row = GetRow(k, 0);
  bounds.at(row++) = ifopt::Bounds(tx_min_, tx_max_); // tau_t1 bounds
  bounds.at(row++) = ifopt::Bounds(ty_min_, ty_max_); // tau_t2 bounds
  bounds.at(row++) = ifopt::BoundSmallerZero;         // tau_n - tz_lim <= 0
  bounds.at(row++) = ifopt::BoundSmallerZero;         // -tau_n - tz_lim <= 0
}

void
TorqueConstraintDiscretized::UpdateJacobianAtInstance(double t, int k,
                                                      std::string var_set,
                                                      Jacobian& jac) const
{
  const Vector3d p   = ee_motion_->GetPoint(t).p();
  const Vector3d f   = ee_force_->GetPoint(t).p();
  const Vector3d tau = ee_torque_->GetPoint(t).p();

  const Vector3d n  = terrain_->GetNormalizedBasis(HeightMap::Normal,   p.x(), p.y());
  const Vector3d t1 = terrain_->GetNormalizedBasis(HeightMap::Tangent1, p.x(), p.y());
  const Vector3d t2 = terrain_->GetNormalizedBasis(HeightMap::Tangent2, p.x(), p.y());

  const int r0 = GetRow(k, 0); // tau_t1
  const int r1 = GetRow(k, 1); // tau_t2
  const int r2 = GetRow(k, 2); // tau_n - tz_lim
  const int r3 = GetRow(k, 3); // -tau_n - tz_lim

  // w.r.t. torque nodes
  if (var_set == id::EETorqueNodes(ee_)) {
    Jacobian Jtau = ee_torque_->GetJacobianWrtNodes(t, kPos); // 3 x nvars
    AccumulateLinearFormJacobian(Jtau, t1, r0, jac);
    AccumulateLinearFormJacobian(Jtau, t2, r1, jac);
    AccumulateLinearFormJacobian(Jtau, n,  r2, jac);

    Eigen::Vector3d minus_n = -n;
    AccumulateLinearFormJacobian(Jtau, minus_n, r3, jac);
  }

  // w.r.t. force nodes (only affects friction normal torque constraints)
  if (var_set == id::EEForceNodes(ee_)) {
    Jacobian Jf = ee_force_->GetJacobianWrtNodes(t, kPos); // 3 x nvars
    const Eigen::Vector3d b = -k_friction_ * mu_ * n;
    AccumulateLinearFormJacobian(Jf, b, r2, jac);
    AccumulateLinearFormJacobian(Jf, b, r3, jac);
  }

  // w.r.t. motion nodes (through terrain basis change with x,y)
  if (var_set == id::EEMotionNodes(ee_)) {
    Jacobian Jp = ee_motion_->GetJacobianWrtNodes(t, kPos); // 3 x nvars

    for (auto dim : {X_, Y_}) {
      const Vector3d dn  = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Normal,   dim, p.x(), p.y());
      const Vector3d dt1 = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Tangent1, dim, p.x(), p.y());
      const Vector3d dt2 = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Tangent2, dim, p.x(), p.y());

      // derivatives w.r.t. p_dim:
      const double s_tx = tau.dot(dt1);
      const double s_ty = tau.dot(dt2);

      const double s_tau_n = tau.dot(dn);
      const double s_f_n   = f.dot(dn);
      const double s_lim   = k_friction_ * mu_ * s_f_n;

      AccumulateScaledRowJacobian(Jp, dim, s_tx, r0, jac);
      AccumulateScaledRowJacobian(Jp, dim, s_ty, r1, jac);
      AccumulateScaledRowJacobian(Jp, dim, (s_tau_n - s_lim), r2, jac);
      AccumulateScaledRowJacobian(Jp, dim, (-s_tau_n - s_lim), r3, jac);
    }
  }

  // w.r.t. timing variables (phase durations) if enabled
  if (var_set == id::EESchedule(ee_)) {
    Jacobian Jtau_dT = ee_torque_->GetJacobianOfPosWrtDurations(t); // 3 x nT
    Jacobian Jf_dT   = ee_force_->GetJacobianOfPosWrtDurations(t);  // 3 x nT
    Jacobian Jp_dT   = ee_motion_->GetJacobianOfPosWrtDurations(t); // 3 x nT

    // direct dependence through tau(t)
    AccumulateLinearFormJacobian(Jtau_dT, t1, r0, jac);
    AccumulateLinearFormJacobian(Jtau_dT, t2, r1, jac);
    AccumulateLinearFormJacobian(Jtau_dT, n,  r2, jac);
    AccumulateLinearFormJacobian(Jtau_dT, -n, r3, jac);

    // direct dependence through f(t) in tz_lim
    const Eigen::Vector3d b = -k_friction_ * mu_ * n;
    AccumulateLinearFormJacobian(Jf_dT, b, r2, jac);
    AccumulateLinearFormJacobian(Jf_dT, b, r3, jac);

    // indirect dependence through p(t) -> basis(p)
    for (auto dim : {X_, Y_}) {
      const Vector3d dn  = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Normal,   dim, p.x(), p.y());
      const Vector3d dt1 = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Tangent1, dim, p.x(), p.y());
      const Vector3d dt2 = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Tangent2, dim, p.x(), p.y());

      const double s_tx = tau.dot(dt1);
      const double s_ty = tau.dot(dt2);

      const double s_tau_n = tau.dot(dn);
      const double s_f_n   = f.dot(dn);
      const double s_lim   = k_friction_ * mu_ * s_f_n;

      AccumulateScaledRowJacobian(Jp_dT, dim, s_tx, r0, jac);
      AccumulateScaledRowJacobian(Jp_dT, dim, s_ty, r1, jac);
      AccumulateScaledRowJacobian(Jp_dT, dim, (s_tau_n - s_lim), r2, jac);
      AccumulateScaledRowJacobian(Jp_dT, dim, (-s_tau_n - s_lim), r3, jac);
    }
  }
}

} /* namespace towr */


