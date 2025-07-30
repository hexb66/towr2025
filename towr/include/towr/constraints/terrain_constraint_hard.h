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

#ifndef TOWR_CONSTRAINTS_TERRAIN_CONSTRAINT_HARD_H_
#define TOWR_CONSTRAINTS_TERRAIN_CONSTRAINT_HARD_H_

#include <towr/constraints/time_discretization_constraint.h>
#include <towr/variables/node_spline.h>
#include <towr/variables/spline_holder.h>
#include <towr/terrain/height_map.h>

namespace towr {

using EE = uint;

/**
 * @brief Ensures the endeffectors maintain appropriate distance from terrain based on velocity.
 *
 * This constraint enforces terrain constraints at all sampled time points
 * along the trajectory, preventing feet from going through terrain and sliding
 * along the surface. The constraint requires: delta_z >= k * v_tangent_magnitude,
 * where delta_z is the distance above terrain, v_tangent_magnitude is the
 * tangential velocity magnitude, and k is a fixed coefficient.
 *
 * @ingroup Constraints
 */
class TerrainConstraintHard : public TimeDiscretizationConstraint {
public:
  using Vector3d = Eigen::Vector3d;

  /**
   * @brief Constructs a terrain constraint.
   * @param terrain  The terrain height value and slope for each position x,y.
   * @param T The total time duration.
   * @param dt The time discretization step.
   * @param ee The end-effector index.
   * @param spline_holder The spline holder containing all splines.
   */
  TerrainConstraintHard (const HeightMap::Ptr& terrain,
                         double T, double dt,
                         const EE& ee,
                         const SplineHolder& spline_holder);

  virtual ~TerrainConstraintHard () = default;

protected:
  void UpdateConstraintAtInstance(double t, int k, VectorXd& g) const override;
  void UpdateBoundsAtInstance(double t, int k, VecBound& bounds) const override;
  void UpdateJacobianAtInstance(double t, int k, std::string var_set, Jacobian& jac) const override;

private:
  NodeSpline::Ptr ee_motion_; ///< the position of the endeffector.
  HeightMap::Ptr terrain_;    ///< the height map of the current terrain.
  EE ee_;                     ///< the end-effector index.
  double k_coeff_;            ///< the coefficient for the terrain constraint.
};

} /* namespace towr */

#endif /* TOWR_CONSTRAINTS_TERRAIN_CONSTRAINT_HARD_H_ */
