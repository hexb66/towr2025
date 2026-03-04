#ifndef TOWR_VARIABLES_ROTVEC_CONVERTER_H_
#define TOWR_VARIABLES_ROTVEC_CONVERTER_H_

#include "angular_converter.h"
#include "node_spline.h"

namespace towr {

/**
 * @brief Converts rotation-vector (axis-angle / exponential map) orientation
 *        to angular quantities and their Jacobians.
 *
 * The rotation vector θ ∈ R³ encodes a rotation of angle ‖θ‖ around the axis
 * θ/‖θ‖.  This parameterization has no gimbal lock and can represent any
 * single rotation up to (but not including) 2π.  It is therefore suitable for
 * backflips and other large-angle maneuvers.
 *
 * Key formulas (Rodrigues):
 *   R(θ)  = I + sinc(θ)·[θ]× + h(θ)·[θ]×²
 *   ω     = J_L(θ)·θ̇
 *   α     = J̇_L·θ̇ + J_L·θ̈
 *
 * where sinc(θ) = sin(θ)/θ,  h(θ) = (1−cos(θ))/θ²,  and J_L is the left
 * Jacobian of SO(3).
 */
class RotVecConverter : public AngularConverter {
public:
  RotVecConverter() = default;
  explicit RotVecConverter(const NodeSpline::Ptr& rotvec_spline);
  ~RotVecConverter() override = default;

  Eigen::Quaterniond GetQuaternionBaseToWorld(double t) const override;
  MatrixSXd GetRotationMatrixBaseToWorld(double t) const override;
  Vector3d GetAngularVelocityInWorld(double t) const override;
  Vector3d GetAngularAccelerationInWorld(double t) const override;
  Jacobian GetDerivOfAngVelWrtNodes(double t) const override;
  Jacobian GetDerivOfAngAccWrtNodes(double t) const override;
  Jacobian DerivOfRotVecMult(double t, const Vector3d& v, bool inverse) const override;

  static Eigen::Matrix3d Rodrigues(const Vector3d& rv);
  static Eigen::Matrix3d LeftJacobian(const Vector3d& rv);

private:
  NodeSpline::Ptr spline_;
  Jacobian jac_structure_;

  static Eigen::Matrix3d Skew(const Vector3d& v);

  // scalar helpers (θ = angle magnitude)
  struct ScalarCoeffs {
    double alpha;  // sin(θ)/θ
    double beta;   // (1 - alpha)/θ²  = (θ-sinθ)/θ³
    double gamma;  // (1-cosθ)/θ²
    double dalpha; // dα/dθ
    double dbeta;  // dβ/dθ
    double dgamma; // dγ/dθ
  };
  static ScalarCoeffs ComputeCoeffs(double theta);

  // J_L time derivative: J̇_L(θ, θ̇)
  static Eigen::Matrix3d LeftJacobianDot(const Vector3d& rv, const Vector3d& rv_dot);

  // dJ_L(row_dim, :)/d(nodes)  — 3×n Jacobian
  Jacobian GetDerivJLwrtNodes(double t, int row_dim) const;

  // dJ̇_L(row_dim, :)/d(nodes)  — 3×n Jacobian
  Jacobian GetDerivJLdotwrtNodes(double t, int row_dim) const;

  JacobianRow GetJac(double t, Dx deriv, int dim) const;

  // IPOPT requires a fixed sparsity pattern across all iterations.
  // RotVecConverter's Jacobians have variable sparsity because J_L(θ)
  // couples components differently at θ≈0 vs θ≠0. This post-processing
  // step inserts explicit zeros so every active column has entries in all 3 rows.
  void EnsureFullPattern(Jacobian& J, double t) const;
};

} // namespace towr

#endif // TOWR_VARIABLES_ROTVEC_CONVERTER_H_
