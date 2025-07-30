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

#include <towr/constraints/base_height_constraint.h>
#include <towr/variables/cartesian_dimensions.h>

namespace towr {

BaseHeightConstraint::BaseHeightConstraint (const HeightMap::Ptr& terrain,
                                           double safety_distance,
                                           std::string base_lin_id)
    :ConstraintSet(kSpecifyLater, "base-height-" + base_lin_id)
{
  base_lin_id_ = base_lin_id;
  terrain_ = terrain;
  safety_distance_ = safety_distance;
}

void
BaseHeightConstraint::InitVariableDependedQuantities (const VariablesPtr& x)
{
  base_linear_ = x->GetComponent<NodesVariablesAll>(base_lin_id_);

  // Constrain all nodes except the first one (initial position)
  for (int id=1; id<base_linear_->GetNodes().size(); ++id)
    node_ids_.push_back(id);

  int constraint_count = node_ids_.size();
  SetRows(constraint_count);
}

Eigen::VectorXd
BaseHeightConstraint::GetValues () const
{
  VectorXd g(GetRows());

  auto nodes = base_linear_->GetNodes();
  int row = 0;
  for (int id : node_ids_) {
    Vector3d p = nodes.at(id).p();
    g(row++) = p.z() - terrain_->GetHeight(p.x(), p.y()) - safety_distance_;
  }

  return g;
}

BaseHeightConstraint::VecBound
BaseHeightConstraint::GetBounds () const
{
  VecBound bounds(GetRows());
  
  // Constraint: base_z >= terrain_height + safety_distance
  // This means: base_z - terrain_height - safety_distance >= 0
  double min_distance = 0.0;  // minimum allowed distance above terrain
  double max_distance = 1e20; // maximum allowed distance (practically unlimited)

  for (int i=0; i<GetRows(); ++i) {
    bounds.at(i) = ifopt::Bounds(min_distance, max_distance);
  }

  return bounds;
}

void
BaseHeightConstraint::FillJacobianBlock (std::string var_set, Jacobian& jac) const
{
  if (var_set == base_lin_id_) {
    auto nodes = base_linear_->GetNodes();
    int row = 0;
    for (int id : node_ids_) {
      // Derivative with respect to base z position
      int idx_z = base_linear_->GetOptIndex(NodesVariables::NodeValueInfo(id, kPos, Z));
      jac.coeffRef(row, idx_z) = 1.0;

      // Derivatives with respect to base x and y positions (due to terrain height variation)
      Vector3d p = nodes.at(id).p();
      for (auto dim : {X,Y}) {
        int idx = base_linear_->GetOptIndex(NodesVariables::NodeValueInfo(id, kPos, dim));
        jac.coeffRef(row, idx) = -terrain_->GetDerivativeOfHeightWrt(To2D(dim), p.x(), p.y());
      }
      row++;
    }
  }
}

} /* namespace towr */ 