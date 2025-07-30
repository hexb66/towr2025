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

#include <towr/constraints/terrain_constraint_hard.h>
#include <towr/variables/variable_names.h>

namespace towr {

TerrainConstraintHard::TerrainConstraintHard (const HeightMap::Ptr& terrain,
                                              double T, double dt,
                                              const EE& ee,
                                              const SplineHolder& spline_holder)
    :TimeDiscretizationConstraint(T, dt, "terrainhard-" + std::to_string(ee))
{
  ee_motion_ = spline_holder.ee_motion_.at(ee);
  terrain_ = terrain;
  ee_ = ee;

  k_coeff_ = 0.02;

  SetRows(GetNumberOfNodes());
}

void
TerrainConstraintHard::UpdateConstraintAtInstance (double t, int k, VectorXd& g) const
{
  Vector3d p = ee_motion_->GetPoint(t).p();
  Vector3d v = ee_motion_->GetPoint(t).v();  // 获取速度
  
  // 计算地形法向量和切向量
  Vector3d n = terrain_->GetNormalizedBasis(HeightMap::Normal, p.x(), p.y());
  Vector3d t1 = terrain_->GetNormalizedBasis(HeightMap::Tangent1, p.x(), p.y());
  Vector3d t2 = terrain_->GetNormalizedBasis(HeightMap::Tangent2, p.x(), p.y());
  
  // 计算切向速度大小
  double v_tangent1 = v.dot(t1);
  double v_tangent2 = v.dot(t2);
  double v_tangent_magnitude = sqrt(v_tangent1*v_tangent1 + v_tangent2*v_tangent2);
  
  // 计算脚离地面的法向距离
  double delta_z = p.z() - terrain_->GetHeight(p.x(), p.y());
  
  // 约束：h > min(k*abs(v), 0.05)
  // 即：delta_z >= min(k_coeff * v_tangent_magnitude, 0.05)
  double min_height = std::min(k_coeff_ * v_tangent_magnitude, k_coeff_);
  g(k) = delta_z - min_height;
}

void
TerrainConstraintHard::UpdateBoundsAtInstance (double t, int k, VecBound& bounds) const
{
  double max_distance_above_terrain = 1e20; // [m]
  
  // 约束形式：delta_z >= min(k*abs(v), 0.05)
  // 约束值应该大于等于0
  bounds.at(k) = ifopt::Bounds(0.0, max_distance_above_terrain);
}

void
TerrainConstraintHard::UpdateJacobianAtInstance (double t, int k,
                                                 std::string var_set,
                                                 Jacobian& jac) const
{
  if (var_set == id::EEMotionNodes(ee_)) {
    Vector3d p = ee_motion_->GetPoint(t).p();
    Vector3d v = ee_motion_->GetPoint(t).v();
    
    // 计算地形法向量和切向量
    Vector3d n = terrain_->GetNormalizedBasis(HeightMap::Normal, p.x(), p.y());
    Vector3d t1 = terrain_->GetNormalizedBasis(HeightMap::Tangent1, p.x(), p.y());
    Vector3d t2 = terrain_->GetNormalizedBasis(HeightMap::Tangent2, p.x(), p.y());
    
    // 计算切向速度大小
    double v_tangent1 = v.dot(t1);
    double v_tangent2 = v.dot(t2);
    double v_tangent_magnitude = sqrt(v_tangent1*v_tangent1 + v_tangent2*v_tangent2);
    
    // Get the Jacobian matrices for position and velocity
    auto jac_pos = ee_motion_->GetJacobianWrtNodes(t, kPos);
    auto jac_vel = ee_motion_->GetJacobianWrtNodes(t, kVel);
    
    // Position derivatives (Z component)
    jac.middleRows(k, 1) = jac_pos.row(Z);

    // Position derivatives (X and Y components for terrain slope)
    for (auto dim : {X,Y}) {
      double terrain_deriv = terrain_->GetDerivativeOfHeightWrt(To2D(dim), p.x(), p.y());
      jac.middleRows(k, 1) -= terrain_deriv * jac_pos.row(dim);
    }
    
    // Velocity derivatives (for min(k*abs(v), 0.05) term)
    if (v_tangent_magnitude > 1e-6) {  // Avoid division by zero
      // 只有当 k*abs(v) < 0.05 时，才需要对速度求导
      if (k_coeff_ * v_tangent_magnitude < 0.05 - 1e-6) { // 添加小的容差避免边界问题
        // ∂v_tangent_magnitude/∂v = (v_tangent1 * t1 + v_tangent2 * t2) / v_tangent_magnitude
        Vector3d tangent_deriv = (v_tangent1 * t1 + v_tangent2 * t2) / v_tangent_magnitude;
        
        // ∂g/∂v = -k_coeff * ∂v_tangent_magnitude/∂v
        for (auto dim : {X,Y,Z}) {
          jac.middleRows(k, 1) -= k_coeff_ * tangent_deriv(dim) * jac_vel.row(dim);
        }
      }
      // 如果 k*abs(v) >= 0.05，则 min(k*abs(v), 0.05) = 0.05，对速度的导数为0
    }
  }
}

} /* namespace towr */
