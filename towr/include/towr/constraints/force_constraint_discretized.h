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

#ifndef TOWR_CONSTRAINTS_FORCE_CONSTRAINT_DISCRETIZED_H_
#define TOWR_CONSTRAINTS_FORCE_CONSTRAINT_DISCRETIZED_H_

#include <towr/constraints/time_discretization_constraint.h>

#include <towr/terrain/height_map.h>
#include <towr/variables/cartesian_dimensions.h>
#include <towr/variables/spline_holder.h>
#include <towr/variables/spline.h>

namespace towr {

/**
 * @brief Force constraint enforced at discretized times along the trajectory.
 *
 * Compared to @ref ForceConstraint (node-only), this enforces the unilateral
 * normal force and friction pyramid at multiple time instances. This reduces
 * violations that can occur when evaluating the force spline between nodes
 * (e.g. when exporting a dense, time-interpolated trajectory).
 *
 * Note: This constraint still enforces "push into terrain" w.r.t. the local
 * terrain normal n(p): f(t)·n(p(t)) >= 0. On flat ground this is equivalent to
 * world-frame Fz >= 0.
 *
 * @ingroup Constraints
 */
class ForceConstraintDiscretized : public TimeDiscretizationConstraint {
public:
  using Vector3d = Eigen::Vector3d;
  using EE = uint;

  ForceConstraintDiscretized(const HeightMap::Ptr& terrain,
                             double T, double dt,
                             double force_limit_in_normal_direction,
                             EE endeffector_id,
                             const SplineHolder& spline_holder);
  virtual ~ForceConstraintDiscretized() = default;

private:
  HeightMap::Ptr terrain_;
  double fn_max_;
  double mu_;
  EE ee_;

  NodeSpline::Ptr ee_force_;  ///< endeffector force spline (world frame)
  NodeSpline::Ptr ee_motion_; ///< endeffector position spline (world frame)

  int n_constraints_per_instance_;

  int GetRow(int k, int local_row) const;

  void UpdateConstraintAtInstance(double t, int k, VectorXd& g) const override;
  void UpdateBoundsAtInstance(double t, int k, VecBound& bounds) const override;
  void UpdateJacobianAtInstance(double t, int k, std::string var_set,
                                Jacobian& jac) const override;
};

} /* namespace towr */

#endif /* TOWR_CONSTRAINTS_FORCE_CONSTRAINT_DISCRETIZED_H_ */


