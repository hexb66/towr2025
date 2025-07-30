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

#ifndef TOWR_CONSTRAINTS_TORQUE_CONSTRAINT_H_
#define TOWR_CONSTRAINTS_TORQUE_CONSTRAINT_H_

#include <ifopt/constraint_set.h>

#include <towr/variables/nodes_variables_phase_based.h>
#include <towr/terrain/height_map.h> // for terrain basis

namespace towr {

/**
 * @brief Ensures endeffector torques are within reasonable bounds.
 *
 * This class is responsible for constraining the endeffector xyz-torques to
 * stay within physically reasonable limits. The constraint includes:
 * - Tangential torque limits: tx_min < tx < tx_max, ty_min < ty < ty_max
 * - Normal torque limits: -k*mu*fz < tz < k*mu*fz (friction-based)
 *
 * Attention: Constraint is enforced only at the spline nodes. In between
 * violations of this constraint can occur.
 *
 * @ingroup Constraints
 */
class TorqueConstraint : public ifopt::ConstraintSet {
public:
  using Vector3d = Eigen::Vector3d;
  using EE = uint;

  /**
   * @brief Constructs a torque constraint.
   * @param terrain  The gradient information of the terrain for friction cone.
   * @param tx_min  Minimum tangential torque in x direction [Nm].
   * @param tx_max  Maximum tangential torque in x direction [Nm].
   * @param ty_min  Minimum tangential torque in y direction [Nm].
   * @param ty_max  Maximum tangential torque in y direction [Nm].
   * @param k_friction  Friction moment approximation coefficient.
   * @param endeffector_id Which endeffector torque should be constrained.
   */
  TorqueConstraint (const HeightMap::Ptr& terrain,
                    double tx_min, double tx_max,
                    double ty_min, double ty_max,
                    double k_friction,
                    EE endeffector_id);
  virtual ~TorqueConstraint () = default;

  void InitVariableDependedQuantities(const VariablesPtr& x) override;

  VectorXd GetValues() const override;
  VecBound GetBounds() const override;
  void FillJacobianBlock (std::string var_set, Jacobian&) const override;

private:
  NodesVariablesPhaseBased::Ptr ee_torque_;  ///< the current xyz foot torques.
  NodesVariablesPhaseBased::Ptr ee_motion_;  ///< the current xyz foot positions.
  NodesVariablesPhaseBased::Ptr ee_force_;   ///< the current xyz foot forces.

  HeightMap::Ptr terrain_; ///< gradient information at every position (x,y).
  double tx_min_;          ///< minimum tangential torque in x direction [Nm].
  double tx_max_;          ///< maximum tangential torque in x direction [Nm].
  double ty_min_;          ///< minimum tangential torque in y direction [Nm].
  double ty_max_;          ///< maximum tangential torque in y direction [Nm].
  double k_friction_;      ///< friction moment approximation coefficient.
  double mu_;              ///< friction coeff between robot feet and terrain.
  int n_constraints_per_node_; ///< number of constraint for each node.
  EE ee_;                  ///< The endeffector torque to be constrained.

  /**
   * The are those Hermite-nodes that shape the polynomial during the
   * stance phases, while all the others are already set to zero torque (swing)
   **/
  std::vector<int> pure_stance_torque_node_ids_;
};

} /* namespace towr */

#endif /* TOWR_CONSTRAINTS_TORQUE_CONSTRAINT_H_ */ 