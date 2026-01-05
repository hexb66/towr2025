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

#include <towr/constraints/torque_constraint.h>

#include <towr/variables/variable_names.h>
#include <towr/variables/cartesian_dimensions.h>

namespace towr {

TorqueConstraint::TorqueConstraint (const HeightMap::Ptr& terrain,
                                    double tx_min, double tx_max,
                                    double ty_min, double ty_max,
                                    double k_friction,
                                    EE ee)
    :ifopt::ConstraintSet(kSpecifyLater, "torque-" + id::EETorqueNodes(ee))
{
  terrain_ = terrain;
  tx_min_ = tx_min;
  tx_max_ = tx_max;
  ty_min_ = ty_min;
  ty_max_ = ty_max;
  k_friction_ = k_friction;
  mu_ = terrain->GetFrictionCoeff();
  ee_ = ee;

  n_constraints_per_node_ = 3; // tx, ty, tz_friction
}

void
TorqueConstraint::InitVariableDependedQuantities (const VariablesPtr& x)
{
  ee_torque_ = x->GetComponent<NodesVariablesPhaseBased>(id::EETorqueNodes(ee_));
  ee_motion_ = x->GetComponent<NodesVariablesPhaseBased>(id::EEMotionNodes(ee_));
  ee_force_ = x->GetComponent<NodesVariablesPhaseBased>(id::EEForceNodes(ee_));

  pure_stance_torque_node_ids_ = ee_torque_->GetIndicesOfNonConstantNodes();

  int constraint_count = pure_stance_torque_node_ids_.size()*n_constraints_per_node_;
  SetRows(constraint_count);
}

Eigen::VectorXd
TorqueConstraint::GetValues () const
{
  VectorXd g(GetRows());

  int row=0;
  auto torque_nodes = ee_torque_->GetNodes();
  for (int t_node_id : pure_stance_torque_node_ids_) {
    int phase = ee_torque_->GetPhase(t_node_id);
    Vector3d p = ee_motion_->GetValueAtStartOfPhase(phase).head<3>(); // doesn't change during stance phase
    Vector3d f = ee_force_->GetValueAtStartOfPhase(phase).head<3>();  // doesn't change during stance phase
    Vector3d tau = torque_nodes.at(t_node_id).p();

    // Get terrain basis vectors
    Vector3d n = terrain_->GetNormalizedBasis(HeightMap::Normal, p.x(), p.y());
    Vector3d t1 = terrain_->GetNormalizedBasis(HeightMap::Tangent1, p.x(), p.y());
    Vector3d t2 = terrain_->GetNormalizedBasis(HeightMap::Tangent2, p.x(), p.y());

    // Transform torque to terrain basis
    double tau_t1 = tau.dot(t1);  // tangential torque in t1 direction
    double tau_t2 = tau.dot(t2);  // tangential torque in t2 direction
    double tau_n = tau.dot(n);    // normal torque

    // Tangential torque constraints
    g(row++) = tau_t1;  // tx_min < tx < tx_max
    g(row++) = tau_t2;  // ty_min < ty < ty_max

    // Normal torque constraint based on friction
    g(row++) = tau_n;   // -k*mu*fz < tz < k*mu*fz
  }

  return g;
}

TorqueConstraint::VecBound
TorqueConstraint::GetBounds () const
{
  VecBound bounds;

  for (int t_node_id : pure_stance_torque_node_ids_) {
    int phase = ee_torque_->GetPhase(t_node_id);
    Vector3d f = ee_force_->GetValueAtStartOfPhase(phase).head<3>();
    Vector3d p = ee_motion_->GetValueAtStartOfPhase(phase).head<3>();
    
    // Get terrain normal for normal force calculation
    Vector3d n = terrain_->GetNormalizedBasis(HeightMap::Normal, p.x(), p.y());
    double f_n = f.dot(n);
    
    // Tangential torque constraints
    bounds.push_back(ifopt::Bounds(tx_min_, tx_max_));  // tx_min < tx < tx_max
    bounds.push_back(ifopt::Bounds(ty_min_, ty_max_));  // ty_min < ty < ty_max

    // Normal torque constraint based on friction: -k*mu*fz < tz < k*mu*fz
    double tz_limit = k_friction_ * mu_ * f_n;
    bounds.push_back(ifopt::Bounds(-tz_limit, tz_limit));
  }

  return bounds;
}

void
TorqueConstraint::FillJacobianBlock (std::string var_set,
                                     Jacobian& jac) const
{
  if (var_set == ee_torque_->GetName()) {
    int row = 0;
    for (int t_node_id : pure_stance_torque_node_ids_) {
      int phase = ee_torque_->GetPhase(t_node_id);
      Vector3d p = ee_motion_->GetValueAtStartOfPhase(phase).head<3>();
      
      // Get terrain basis vectors
      Vector3d n = terrain_->GetNormalizedBasis(HeightMap::Normal, p.x(), p.y());
      Vector3d t1 = terrain_->GetNormalizedBasis(HeightMap::Tangent1, p.x(), p.y());
      Vector3d t2 = terrain_->GetNormalizedBasis(HeightMap::Tangent2, p.x(), p.y());

      for (auto dim : {X,Y,Z}) {
        int idx = ee_torque_->GetOptIndex(NodesVariables::NodeValueInfo(t_node_id, kPos, dim));
        int row_reset = row;

        // Tangential torque constraints
        jac.coeffRef(row_reset++, idx) = t1(dim);  // d(tau_t1)/d(tau)
        jac.coeffRef(row_reset++, idx) = t2(dim);  // d(tau_t2)/d(tau)

        // Normal torque constraint
        jac.coeffRef(row_reset++, idx) = n(dim);   // d(tau_n)/d(tau)
      }

      row += n_constraints_per_node_;
    }
  }

  if (var_set == ee_motion_->GetName()) {
    int row = 0;
    for (int t_node_id : pure_stance_torque_node_ids_) {
      int phase = ee_torque_->GetPhase(t_node_id);
      int ee_node_id = ee_motion_->GetNodeIDAtStartOfPhase(phase);
      Vector3d p = ee_motion_->GetValueAtStartOfPhase(phase).head<3>();
      Vector3d tau = ee_torque_->GetValueAtStartOfPhase(phase).head<3>();

      for (auto dim : {X_,Y_}) {
        Vector3d dt1 = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Tangent1, dim, p.x(), p.y());
        Vector3d dt2 = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Tangent2, dim, p.x(), p.y());
        Vector3d dn = terrain_->GetDerivativeOfNormalizedBasisWrt(HeightMap::Normal, dim, p.x(), p.y());

        int idx = ee_motion_->GetOptIndex(NodesVariables::NodeValueInfo(ee_node_id, kPos, dim));
        int row_reset = row;

        // Tangential torque constraints
        jac.coeffRef(row_reset++, idx) = tau.dot(dt1);  // d(tau_t1)/d(p)
        jac.coeffRef(row_reset++, idx) = tau.dot(dt2);  // d(tau_t2)/d(p)

        // Normal torque constraint
        jac.coeffRef(row_reset++, idx) = tau.dot(dn);   // d(tau_n)/d(p)
      }

      row += n_constraints_per_node_;
    }
  }

  if (var_set == ee_force_->GetName()) {
    // The normal torque constraint depends on the normal force
    // This is handled in the bounds calculation
    // No direct Jacobian contribution needed here
  }
}

} /* namespace towr */ 