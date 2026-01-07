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

#include <towr/variables/nodes_variables.h>
#include <towr/variables/cartesian_dimensions.h>
#include <towr/variables/euler_converter.h>

#include <stdexcept>

namespace towr {


NodesVariables::NodesVariables (const std::string& name)
    : VariableSet(kSpecifyLater, name)
{
}

int
NodesVariables::GetOptIndex(const NodeValueInfo& nvi_des) const
{
  // could also cache this as map for more efficiency, but adding complexity
  for (int idx=0; idx<GetRows(); ++idx)
    for ( NodeValueInfo nvi : GetNodeValuesInfo(idx))
      if ( nvi == nvi_des )
        return idx;

  return NodeValueNotOptimized; // index representing these quantities doesn't exist
}

Eigen::VectorXd
NodesVariables::GetValues () const
{
  VectorXd x(GetRows());

  for (int idx=0; idx<x.rows(); ++idx)
    for (auto nvi : GetNodeValuesInfo(idx))
      x(idx) = nodes_.at(nvi.id_).at(nvi.deriv_)(nvi.dim_);

  return x;
}

void
NodesVariables::SetVariables (const VectorXd& x)
{
  for (int idx=0; idx<x.rows(); ++idx)
    for (auto nvi : GetNodeValuesInfo(idx))
      nodes_.at(nvi.id_).at(nvi.deriv_)(nvi.dim_) = x(idx);

  UpdateObservers();
}

void
NodesVariables::UpdateObservers() const
{
  for (auto& o : observers_)
    o->UpdateNodes();
}

void
NodesVariables::AddObserver(ObserverPtr const o)
{
   observers_.push_back(o);
}

int
NodesVariables::GetNodeId (int poly_id, Side side)
{
  return poly_id + side;
}

const std::vector<Node>
NodesVariables::GetBoundaryNodes(int poly_id) const
{
  std::vector<Node> nodes;
  nodes.push_back(nodes_.at(GetNodeId(poly_id, Side::Start)));
  nodes.push_back(nodes_.at(GetNodeId(poly_id, Side::End)));
  return nodes;
}

int
NodesVariables::GetDim() const
{
  return n_dim_;
}

int
NodesVariables::GetPolynomialCount() const
{
  return nodes_.size() - 1;
}

NodesVariables::VecBound
NodesVariables::GetBounds () const
{
  return bounds_;
}

const std::vector<Node>
NodesVariables::GetNodes() const
{
  return nodes_;
}

void
NodesVariables::SetByLinearInterpolation(const VectorXd& initial_val,
                                         const VectorXd& final_val,
                                         double t_total)
{
  // only set those that are part of optimization variables,
  // do not overwrite phase-based parameterization
  VectorXd dp = final_val-initial_val;
  VectorXd average_velocity = dp / t_total;
  int num_nodes = nodes_.size();

  for (int idx=0; idx<GetRows(); ++idx) {
    for (auto nvi : GetNodeValuesInfo(idx)) {

      if (nvi.deriv_ == kPos) {
        VectorXd pos = initial_val + nvi.id_/static_cast<double>(num_nodes-1)*dp;
        nodes_.at(nvi.id_).at(kPos)(nvi.dim_) = pos(nvi.dim_);
      }

      if (nvi.deriv_ == kVel) {
        nodes_.at(nvi.id_).at(kVel)(nvi.dim_) = average_velocity(nvi.dim_);
      }
    }
  }
}

void
NodesVariables::SetByLinearInterpolationRelativeToBase(const Eigen::Vector3d& ee_initial_W,
                                                       const Eigen::Vector3d& ee_final_W,
                                                       const Eigen::Vector3d& base_pos_initial_W,
                                                       const Eigen::Vector3d& base_pos_final_W,
                                                       const Eigen::Vector3d& base_rpy_initial_W,
                                                       const Eigen::Vector3d& base_rpy_final_W,
                                                       double t_total)
{
  if (GetDim() != k3D) {
    throw std::runtime_error("SetByLinearInterpolationRelativeToBase requires 3D variables.");
  }
  if (t_total <= 0.0) {
    throw std::runtime_error("SetByLinearInterpolationRelativeToBase requires t_total > 0.");
  }

  const int num_nodes = nodes_.size();
  if (num_nodes < 2) {
    return;
  }

  // Compute relative positions in base frame at start/end.
  const Eigen::Matrix3d w_R_b0 = EulerConverter::GetRotationMatrixBaseToWorld(base_rpy_initial_W);
  const Eigen::Matrix3d w_R_bT = EulerConverter::GetRotationMatrixBaseToWorld(base_rpy_final_W);

  const Eigen::Vector3d r0_B = w_R_b0.transpose() * (ee_initial_W - base_pos_initial_W);
  const Eigen::Vector3d rT_B = w_R_bT.transpose() * (ee_final_W - base_pos_final_W);

  const Eigen::Vector3d dp_B = rT_B - r0_B;
  const Eigen::Vector3d avg_vel_B = dp_B / t_total;
  const Eigen::Vector3d base_avg_vel_W = (base_pos_final_W - base_pos_initial_W) / t_total;

  // Same pattern as SetByLinearInterpolation():
  // use node id to compute interpolation fraction, write pos/vel for optimized node values.
  for (int idx=0; idx<GetRows(); ++idx) {
    for (auto nvi : GetNodeValuesInfo(idx)) {
      const double alpha = nvi.id_/static_cast<double>(num_nodes-1);

      const Eigen::Vector3d base_pos_W = (1.0-alpha)*base_pos_initial_W + alpha*base_pos_final_W;
      const Eigen::Vector3d base_rpy_W = (1.0-alpha)*base_rpy_initial_W + alpha*base_rpy_final_W;
      const Eigen::Matrix3d w_R_b = EulerConverter::GetRotationMatrixBaseToWorld(base_rpy_W);

      const Eigen::Vector3d r_B = r0_B + alpha*dp_B;

      if (nvi.deriv_ == kPos) {
        const Eigen::Vector3d ee_pos_W = base_pos_W + w_R_b*r_B;
        nodes_.at(nvi.id_).at(kPos)(nvi.dim_) = ee_pos_W(nvi.dim_);
      }

      if (nvi.deriv_ == kVel) {
        // Simple initial guess (same style as SetByLinearInterpolation):
        // base translation velocity + rotated average relative velocity in base frame.
        const Eigen::Vector3d ee_vel_W = base_avg_vel_W + w_R_b*avg_vel_B;
        nodes_.at(nvi.id_).at(kVel)(nvi.dim_) = ee_vel_W(nvi.dim_);
      }
    }
  }

  // Sync phase-based mappings: if one optimization variable maps to multiple node values
  // (e.g. stance start/end), make them consistent by applying the current x to all mapped nodes.
  SetVariables(GetValues());
}

void
NodesVariables::AddBounds(int node_id, Dx deriv,
                 const std::vector<int>& dimensions,
                 const VectorXd& val)
{
  for (auto dim : dimensions)
    AddBound(NodeValueInfo(node_id, deriv, dim), val(dim));
}

void
NodesVariables::AddBound (const NodeValueInfo& nvi_des, double val)
{
  for (int idx=0; idx<GetRows(); ++idx)
    for (auto nvi : GetNodeValuesInfo(idx))
      if (nvi == nvi_des)
        bounds_.at(idx) = ifopt::Bounds(val, val);
}

void
NodesVariables::AddStartBound (Dx d, const std::vector<int>& dimensions, const VectorXd& val)
{
  AddBounds(0, d, dimensions, val);
}

void
NodesVariables::AddFinalBound (Dx deriv, const std::vector<int>& dimensions,
                      const VectorXd& val)
{
  AddBounds(nodes_.size()-1, deriv, dimensions, val);
}

NodesVariables::NodeValueInfo::NodeValueInfo(int node_id, Dx deriv, int node_dim)
{
  id_    = node_id;
  deriv_ = deriv;
  dim_   = node_dim;
}

int
NodesVariables::NodeValueInfo::operator==(const NodeValueInfo& right) const
{
  return (id_    == right.id_)
      && (deriv_ == right.deriv_)
      && (dim_   == right.dim_);
};

} /* namespace towr */
