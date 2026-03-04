#ifndef TOWR_VARIABLES_ANGULAR_CONVERTER_H_
#define TOWR_VARIABLES_ANGULAR_CONVERTER_H_

#include <memory>
#include <array>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "cartesian_dimensions.h"

namespace towr {

/**
 * @brief Abstract interface for converting an orientation spline to angular
 *        quantities (rotation matrix, angular velocity/acceleration) and
 *        their Jacobians w.r.t. the spline node values.
 *
 * Concrete implementations include EulerConverter (ZYX Euler angles) and
 * RotVecConverter (rotation-vector / axis-angle / exponential map).
 */
class AngularConverter {
public:
  using Ptr         = std::shared_ptr<AngularConverter>;
  using Vector3d    = Eigen::Vector3d;
  using MatrixSXd   = Eigen::SparseMatrix<double, Eigen::RowMajor>;
  using Jacobian    = MatrixSXd;
  using JacobianRow = Eigen::SparseVector<double, Eigen::RowMajor>;

  virtual ~AngularConverter() = default;

  virtual Eigen::Quaterniond GetQuaternionBaseToWorld(double t) const = 0;
  virtual MatrixSXd GetRotationMatrixBaseToWorld(double t) const = 0;
  virtual Vector3d GetAngularVelocityInWorld(double t) const = 0;
  virtual Vector3d GetAngularAccelerationInWorld(double t) const = 0;

  virtual Jacobian GetDerivOfAngVelWrtNodes(double t) const = 0;
  virtual Jacobian GetDerivOfAngAccWrtNodes(double t) const = 0;

  /**
   * @brief Derivative of R*v (or R^T*v) w.r.t. the orientation node values.
   * @param t        time at which the orientation is evaluated.
   * @param v        vector (independent of orientation nodes).
   * @param inverse  if true, computes derivative of R^T*v.
   * @returns        3 x n Jacobian.
   */
  virtual Jacobian DerivOfRotVecMult(double t, const Vector3d& v,
                                     bool inverse) const = 0;
};

} // namespace towr

#endif // TOWR_VARIABLES_ANGULAR_CONVERTER_H_
