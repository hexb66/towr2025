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

#include <towr/constraints/force_constraint_discretized.h>

#include <towr/variables/variable_names.h>

namespace towr {

namespace {

inline void AccumulateLinearFormJacobian(const ifopt::ConstraintSet::Jacobian& J3xN,
                                         const Eigen::Vector3d& b,
                                         int dst_row,
                                         ifopt::ConstraintSet::Jacobian& out)
{
  // NOTE: Jacobian is typically column-major sparse. Iterate over all non-zeros.
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

ForceConstraintDiscretized::ForceConstraintDiscretized(
    const HeightMap::Ptr& terrain,
    double T, double dt,
    double force_limit_in_normal_direction,
    EE endeffector_id,
    const SplineHolder& spline_holder)
    : TimeDiscretizationConstraint(T, dt, "force-disc-" + id::EEForceNodes(endeffector_id))
{
  terrain_ = terrain;
  fn_max_  = force_limit_in_normal_direction;
  mu_      = terrain->GetFrictionCoeff();
  ee_      = endeffector_id;

  ee_force_  = spline_holder.ee_force_.at(ee_);
  ee_motion_ = spline_holder.ee_motion_.at(ee_);

  n_constraints_per_instance_ = 1 + 2*k2D; // positive normal force + 4 friction pyramid constraints
  SetRows(GetNumberOfNodes() * n_constraints_per_instance_);
}

int
ForceConstraintDiscretized::GetRow(int k, int local_row) const
{
  return k*n_constraints_per_instance_ + local_row;
}

void
ForceConstraintDiscretized::UpdateConstraintAtInstance(double t, int k, VectorXd& g) const
{
  Vector3d p = ee_motion_->GetPoint(t).p();
  Vector3d f = ee_force_->GetPoint(t).p();

  Vector3d n  = terrain_->GetNormalizedBasis(HeightMap::Normal,   p.x(), p.y());
  Vector3d t1 = terrain_->GetNormalizedBasis(HeightMap::Tangent1, p.x(), p.y());
  Vector3d t2 = terrain_->GetNormalizedBasis(HeightMap::Tangent2, p.x(), p.y());

  int row = GetRow(k, 0);

  // unilateral force
  g(row++) = f.dot(n);

  // friction pyramid
  g(row++) = f.dot(t1 - mu_*n);
  g(row++) = f.dot(t1 + mu_*n);
  g(row++) = f.dot(t2 - mu_*n);
  g(row++) = f.dot(t2 + mu_*n);
}

void
ForceConstraintDiscretized::UpdateBoundsAtInstance(double /*t*/, int k, VecBound& bounds) const
{
  int row = GetRow(k, 0);

  bounds.at(row++) = ifopt::Bounds(0.0, fn_max_); // unilateral forces
  bounds.at(row++) = ifopt::BoundSmallerZero;     // f_t1 <  mu*n
  bounds.at(row++) = ifopt::BoundGreaterZero;     // f_t1 > -mu*n
  bounds.at(row++) = ifopt::BoundSmallerZero;     // f_t2 <  mu*n
  bounds.at(row++) = ifopt::BoundGreaterZero;     // f_t2 > -mu*n
}

void
ForceConstraintDiscretized::UpdateJacobianAtInstance(double t, int k,
                                                     std::string var_set,
                                                     Jacobian& jac) const
{
  const Vector3d p = ee_motion_->GetPoint(t).p();
  const Vector3d f = ee_force_->GetPoint(t).p();

  const Vector3d n  = terrain_->GetNormalizedBasis(HeightMap::Normal,   p.x(), p.y());
  const Vector3d t1 = terrain_->GetNormalizedBasis(HeightMap::Tangent1, p.x(), p.y());
  const Vector3d t2 = terrain_->GetNormalizedBasis(HeightMap::Tangent2, p.x(), p.y());

  const Vector3d b0 = n;
  const Vector3d b1 = t1 - mu_*n;
  const Vector3d b2 = t1 + mu_*n;
  const Vector3d b3 = t2 - mu_*n;
  const Vector3d b4 = t2 + mu_*n;

  const int r0 = GetRow(k, 0);
  const int r1 = GetRow(k, 1);
  const int r2 = GetRow(k, 2);
  const int r3 = GetRow(k, 3);
  const int r4 = GetRow(k, 4);

  // w.r.t. force nodes
  if (var_set == id::EEForceNodes(ee_)) {
    Jacobian Jf = ee_force_->GetJacobianWrtNodes(t, kPos); // 3 x nvars

    // Each constraint is linear in f: g = b^T f
    AccumulateLinearFormJacobian(Jf, b0, r0, jac);
    AccumulateLinearFormJacobian(Jf, b1, r1, jac);
    AccumulateLinearFormJacobian(Jf, b2, r2, jac);
    AccumulateLinearFormJacobian(Jf, b3, r3, jac);
    AccumulateLinearFormJacobian(Jf, b4, r4, jac);
  }

  // w.r.t. motion nodes (through terrain basis change with x,y)
  if (var_set == id::EEMotionNodes(ee_)) {
    Jacobian Jp = ee_motion_->GetJacobianWrtNodes(t, kPos); // 3 x nvars

    for (auto dim : {X_, Y_}) {
      Vector3d dn  = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Normal,   dim, p.x(), p.y());
      Vector3d dt1 = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Tangent1, dim, p.x(), p.y());
      Vector3d dt2 = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Tangent2, dim, p.x(), p.y());

      const double s0 = f.dot(dn);
      const double s1 = f.dot(dt1 - mu_*dn);
      const double s2 = f.dot(dt1 + mu_*dn);
      const double s3 = f.dot(dt2 - mu_*dn);
      const double s4 = f.dot(dt2 + mu_*dn);

      AccumulateScaledRowJacobian(Jp, dim, s0, r0, jac);
      AccumulateScaledRowJacobian(Jp, dim, s1, r1, jac);
      AccumulateScaledRowJacobian(Jp, dim, s2, r2, jac);
      AccumulateScaledRowJacobian(Jp, dim, s3, r3, jac);
      AccumulateScaledRowJacobian(Jp, dim, s4, r4, jac);
    }
  }

  // w.r.t. timing variables (phase durations) if enabled
  if (var_set == id::EESchedule(ee_)) {
    Jacobian Jf_dT = ee_force_->GetJacobianOfPosWrtDurations(t);  // 3 x nT
    Jacobian Jp_dT = ee_motion_->GetJacobianOfPosWrtDurations(t); // 3 x nT

    // direct dependence through f(t)
    AccumulateLinearFormJacobian(Jf_dT, b0, r0, jac);
    AccumulateLinearFormJacobian(Jf_dT, b1, r1, jac);
    AccumulateLinearFormJacobian(Jf_dT, b2, r2, jac);
    AccumulateLinearFormJacobian(Jf_dT, b3, r3, jac);
    AccumulateLinearFormJacobian(Jf_dT, b4, r4, jac);

    // indirect dependence through p(t) -> basis(p)
    for (auto dim : {X_, Y_}) {
      Vector3d dn  = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Normal,   dim, p.x(), p.y());
      Vector3d dt1 = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Tangent1, dim, p.x(), p.y());
      Vector3d dt2 = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Tangent2, dim, p.x(), p.y());

      const double s0 = f.dot(dn);
      const double s1 = f.dot(dt1 - mu_*dn);
      const double s2 = f.dot(dt1 + mu_*dn);
      const double s3 = f.dot(dt2 - mu_*dn);
      const double s4 = f.dot(dt2 + mu_*dn);

      AccumulateScaledRowJacobian(Jp_dT, dim, s0, r0, jac);
      AccumulateScaledRowJacobian(Jp_dT, dim, s1, r1, jac);
      AccumulateScaledRowJacobian(Jp_dT, dim, s2, r2, jac);
      AccumulateScaledRowJacobian(Jp_dT, dim, s3, r3, jac);
      AccumulateScaledRowJacobian(Jp_dT, dim, s4, r4, jac);
    }
  }
}

} /* namespace towr */


