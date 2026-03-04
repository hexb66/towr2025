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

#ifndef TOWR_VARIABLES_ANGULAR_STATE_CONVERTER_H_
#define TOWR_VARIABLES_ANGULAR_STATE_CONVERTER_H_

#include <array>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "angular_converter.h"
#include "cartesian_dimensions.h"
#include "node_spline.h"

namespace towr {

/**
 * @brief Converts ZYX Euler angles and derivatives to angular quantities.
 *
 * Implements AngularConverter using Euler-angle parameterization
 * (yaw-pitch-roll / Z-Y'-X'').
 *
 * Formulas from kindr:
 * http://docs.leggedrobotics.com/kindr/cheatsheet_latest.pdf
 */
class EulerConverter : public AngularConverter {
public:
  using EulerAngles = Vector3d; ///< roll, pitch, yaw.
  using EulerRates  = Vector3d;
  using JacRowMatrix = std::array<std::array<JacobianRow, k3D>, k3D>;

  EulerConverter () = default;

  /**
   * @brief Constructs and links this object to the Euler angle values
   * @param euler_angles_spline  The Euler angle spline defined by node values.
   *
   * The 3-dimensional Euler angles store values in order (roll, pitch, yaw).
   * Rotation order is Z-Y'-X'' (yaw first, then pitch, then roll).
   */
  EulerConverter (const NodeSpline::Ptr& euler_angles);
  ~EulerConverter () override = default;

  Eigen::Quaterniond GetQuaternionBaseToWorld (double t) const override;
  MatrixSXd GetRotationMatrixBaseToWorld(double t) const override;
  Vector3d GetAngularVelocityInWorld(double t) const override;
  Vector3d GetAngularAccelerationInWorld(double t) const override;
  Jacobian GetDerivOfAngVelWrtNodes(double t) const override;
  Jacobian GetDerivOfAngAccWrtNodes(double t) const override;
  Jacobian DerivOfRotVecMult(double t, const Vector3d& v, bool inverse) const override;

  static MatrixSXd GetRotationMatrixBaseToWorld(const EulerAngles& xyz);
  static Eigen::Quaterniond GetQuaternionBaseToWorld(const EulerAngles& pos);

private:
  NodeSpline::Ptr euler_;

  // Internal calculations for the conversion from euler rates to angular
  // velocities and accelerations. These are done using the matrix M defined
  // here: http://docs.leggedrobotics.com/kindr/cheatsheet_latest.pdf
  /**
   * @brief Matrix that maps euler rates to angular velocities in world.
   *
   * Make sure euler rates are ordered roll-pitch-yaw. They are however applied
   * in the order yaw-pitch-role to determine the angular velocities.
   */
  static MatrixSXd GetM(const EulerAngles& xyz);

  /**
   *  @brief time derivative of GetM()
   */
  static MatrixSXd GetMdot(const EulerAngles& xyz, const EulerRates& xyz_d);

  /**
   *  @brief Derivative of the @a dim row of matrix M with respect to
   *         the node values.
   *
   *  @param dim  Which dimension of the angular acceleration is desired.
   *  @returns    the Jacobian w.r.t the coefficients for each of the 3 rows
   *              of the matrix stacked on top of each other.
   */
  Jacobian GetDerivMwrtNodes(double t, Dim3D dim) const;

  /** @brief Derivative of the @a dim row of the time derivative of M with
   *         respect to the node values.
   *
   *  @param dim Which dimension of the angular acceleration is desired.
   */
  Jacobian GetDerivMdotwrtNodes(double t, Dim3D dim) const;

  /** @brief matrix of derivatives of each cell w.r.t node values.
   *
   * This 2d-array has the same dimensions as the rotation matrix M_IB, but
   * each cell if filled with a row vector.
   */
  JacRowMatrix GetDerivativeOfRotationMatrixWrtNodes(double t) const;

  /** @see GetAngularAccelerationInWorld(t)  */
  static Vector3d GetAngularAccelerationInWorld(State euler);

  /** @see GetAngularVelocityInWorld(t)  */
  static Vector3d GetAngularVelocityInWorld(const EulerAngles& pos,
                                            const EulerRates& vel);

  JacobianRow GetJac(double t, Dx deriv, Dim3D dim) const;
  Jacobian jac_wrt_nodes_structure_;
};

} /* namespace towr */

#endif /* TOWR_VARIABLES_ANGULAR_STATE_CONVERTER_H_ */
