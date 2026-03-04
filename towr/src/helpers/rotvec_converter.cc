#include <towr/variables/rotvec_converter.h>

#include <cassert>
#include <cmath>
#include <set>
#include <map>

namespace towr {

static constexpr double kEps = 1e-10;

RotVecConverter::RotVecConverter(const NodeSpline::Ptr& spline)
{
  spline_ = spline;
  jac_structure_ = Jacobian(k3D, spline->GetNodeVariablesCount());
}

// ===== Static helpers =====

Eigen::Matrix3d
RotVecConverter::Skew(const Vector3d& v)
{
  Eigen::Matrix3d S;
  S <<     0, -v(2),  v(1),
        v(2),     0, -v(0),
       -v(1),  v(0),     0;
  return S;
}

RotVecConverter::ScalarCoeffs
RotVecConverter::ComputeCoeffs(double theta)
{
  ScalarCoeffs c;
  double t2 = theta * theta;

  if (theta < kEps) {
    // Taylor expansions around θ=0
    c.alpha  = 1.0 - t2/6.0;
    c.beta   = 1.0/6.0 - t2/120.0;
    c.gamma  = 0.5 - t2/24.0;
    c.dalpha = -theta/3.0;
    c.dbeta  = -theta/60.0;
    c.dgamma = -theta/12.0;
  } else {
    double st = std::sin(theta);
    double ct = std::cos(theta);
    double t3 = t2 * theta;
    double t4 = t3 * theta;

    c.alpha  = st / theta;
    c.beta   = (theta - st) / t3;
    c.gamma  = (1.0 - ct) / t2;

    c.dalpha = (theta*ct - st) / t2;
    c.dbeta  = (-2.0*theta - theta*ct + 3.0*st) / t4;
    c.dgamma = (theta*st - 2.0 + 2.0*ct) / t3;
  }
  return c;
}

Eigen::Matrix3d
RotVecConverter::Rodrigues(const Vector3d& rv)
{
  double theta = rv.norm();
  if (theta < kEps) {
    return Eigen::Matrix3d::Identity() + Skew(rv);
  }
  Eigen::Matrix3d K = Skew(rv);
  double s = std::sin(theta) / theta;
  double h = (1.0 - std::cos(theta)) / (theta * theta);
  return Eigen::Matrix3d::Identity() + s * K + h * K * K;
}

Eigen::Matrix3d
RotVecConverter::LeftJacobian(const Vector3d& rv)
{
  double theta = rv.norm();
  if (theta < kEps) {
    return Eigen::Matrix3d::Identity() + 0.5 * Skew(rv);
  }
  auto c = ComputeCoeffs(theta);
  return c.alpha * Eigen::Matrix3d::Identity()
       + c.beta  * rv * rv.transpose()
       + c.gamma * Skew(rv);
}

Eigen::Matrix3d
RotVecConverter::LeftJacobianDot(const Vector3d& rv, const Vector3d& rv_dot)
{
  double theta = rv.norm();
  if (theta < kEps) {
    return 0.5 * Skew(rv_dot);
  }

  auto c = ComputeCoeffs(theta);
  double theta_dot = rv.dot(rv_dot) / theta;

  double alpha_dot = c.dalpha * theta_dot;
  double beta_dot  = c.dbeta  * theta_dot;
  double gamma_dot = c.dgamma * theta_dot;

  return alpha_dot * Eigen::Matrix3d::Identity()
       + beta_dot  * rv * rv.transpose()
       + c.beta    * (rv_dot * rv.transpose() + rv * rv_dot.transpose())
       + gamma_dot * Skew(rv)
       + c.gamma   * Skew(rv_dot);
}

// ===== AngularConverter interface =====

Eigen::Quaterniond
RotVecConverter::GetQuaternionBaseToWorld(double t) const
{
  Eigen::Matrix3d R = Rodrigues(spline_->GetPoint(t).p());
  return Eigen::Quaterniond(R);
}

RotVecConverter::MatrixSXd
RotVecConverter::GetRotationMatrixBaseToWorld(double t) const
{
  Eigen::Matrix3d R = Rodrigues(spline_->GetPoint(t).p());
  return R.sparseView(1.0, -1.0);
}

RotVecConverter::Vector3d
RotVecConverter::GetAngularVelocityInWorld(double t) const
{
  State s = spline_->GetPoint(t);
  return LeftJacobian(s.p()) * s.v();
}

RotVecConverter::Vector3d
RotVecConverter::GetAngularAccelerationInWorld(double t) const
{
  State s = spline_->GetPoint(t);
  return LeftJacobianDot(s.p(), s.v()) * s.v()
       + LeftJacobian(s.p()) * s.a();
}

// ===== Jacobians =====

RotVecConverter::JacobianRow
RotVecConverter::GetJac(double t, Dx deriv, int dim) const
{
  return spline_->GetJacobianWrtNodes(t, deriv).row(dim);
}

// Dense 3×3 × sparse 3×n → sparse 3×n.
// Always produces entries in all 3 rows for every active column of B,
// so the sparsity pattern is independent of A's values.
static void DenseTimesSparse(const Eigen::Matrix3d& A,
                             const RotVecConverter::Jacobian& B,
                             RotVecConverter::Jacobian& out)
{
  int n = B.cols();

  std::set<int> active;
  for (int r = 0; r < k3D; ++r)
    for (RotVecConverter::Jacobian::InnerIterator it(B, r); it; ++it)
      active.insert(it.col());

  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(active.size() * k3D);
  for (int col : active) {
    Eigen::Vector3d bc;
    for (int r = 0; r < k3D; ++r) bc(r) = B.coeff(r, col);
    Eigen::Vector3d rc = A * bc;
    for (int r = 0; r < k3D; ++r)
      trips.emplace_back(r, col, rc(r));
  }

  out.resize(k3D, n);
  out.setFromTriplets(trips.begin(), trips.end());
}

void
RotVecConverter::EnsureFullPattern(Jacobian& J, double t) const
{
  Jacobian jp = spline_->GetJacobianWrtNodes(t, kPos);
  Jacobian jv = spline_->GetJacobianWrtNodes(t, kVel);

  int cols = J.cols();
  std::vector<bool> active(cols, false);
  for (int r = 0; r < k3D; ++r) {
    for (Jacobian::InnerIterator it(jp, r); it; ++it) active[it.col()] = true;
    for (Jacobian::InnerIterator it(jv, r); it; ++it) active[it.col()] = true;
  }
  for (int r = 0; r < J.outerSize(); ++r)
    for (Jacobian::InnerIterator it(J, r); it; ++it)
      active[it.col()] = true;

  std::map<std::pair<int,int>, double> existing;
  for (int r = 0; r < J.outerSize(); ++r)
    for (Jacobian::InnerIterator it(J, r); it; ++it)
      existing[{it.row(), it.col()}] = it.value();

  std::vector<Eigen::Triplet<double>> trips;
  for (int c = 0; c < cols; ++c) {
    if (!active[c]) continue;
    for (int r = 0; r < k3D; ++r) {
      auto it = existing.find({r, c});
      trips.emplace_back(r, c, it != existing.end() ? it->second : 0.0);
    }
  }

  J.resize(k3D, cols);
  J.setFromTriplets(trips.begin(), trips.end());
}

RotVecConverter::Jacobian
RotVecConverter::DerivOfRotVecMult(double t, const Vector3d& v, bool inverse) const
{
  // d(R·v)/dθ = -[Rv]_× · J_L,   d(R·v)/d(nodes) = -[Rv]_× · J_L · dθ/d(nodes)
  // d(R^T·v)/dθ = R^T · [v]_× · J_L
  State s = spline_->GetPoint(t);
  Vector3d rv = s.p();

  Eigen::Matrix3d R  = Rodrigues(rv);
  Eigen::Matrix3d JL = LeftJacobian(rv);
  Jacobian jac_pos   = spline_->GetJacobianWrtNodes(t, kPos);

  Eigen::Matrix3d A;
  if (inverse) {
    A = R.transpose() * Skew(v) * JL;
  } else {
    A = -Skew(R * v) * JL;
  }

  Jacobian result;
  DenseTimesSparse(A, jac_pos, result);
  EnsureFullPattern(result, t);
  return result;
}

// Derivative of row `dim` of J_L w.r.t. spline node values (3×n)
// jac.row(j) = d(J_L_{dim,j})/d(nodes)
RotVecConverter::Jacobian
RotVecConverter::GetDerivJLwrtNodes(double t, int dim) const
{
  State s = spline_->GetPoint(t);
  Vector3d rv = s.p();
  double theta = rv.norm();
  auto c = ComputeCoeffs(theta);

  Jacobian jac_pos = spline_->GetJacobianWrtNodes(t, kPos);
  Jacobian result  = jac_structure_;

  // J_L_{i,j} = α·δ_{ij} + β·θ_i·θ_j + γ·[θ]×_{ij}
  // d(J_L_{dim,j})/d(nodes) = dα/d(nodes)·δ_{dim,j}
  //                          + dβ/d(nodes)·θ_dim·θ_j
  //                          + β·(jac_dim · θ_j + θ_dim · jac_j)
  //                          + dγ/d(nodes)·[θ]×_{dim,j}
  //                          + γ · d([θ]×_{dim,j})/d(nodes)

  // dα/d(nodes) = α'/θ · θ^T · jac_pos  (1×n)
  // dβ/d(nodes) = β'/θ · θ^T · jac_pos
  // dγ/d(nodes) = γ'/θ · θ^T · jac_pos
  JacobianRow da_du, db_du, dg_du;
  if (theta < kEps) {
    // derivatives are effectively zero at θ=0
    da_du = JacobianRow(jac_pos.cols());
    db_du = JacobianRow(jac_pos.cols());
    dg_du = JacobianRow(jac_pos.cols());
  } else {
    double inv_theta = 1.0 / theta;
    // θ^T · jac_pos = Σ_l θ_l/θ · jac_pos.row(l)
    JacobianRow thetaT_jac(jac_pos.cols());
    for (int l = 0; l < k3D; ++l) {
      if (std::abs(rv(l)) > 1e-15)
        thetaT_jac += (rv(l) * inv_theta) * jac_pos.row(l);
    }
    da_du = c.dalpha * thetaT_jac;
    db_du = c.dbeta  * thetaT_jac;
    dg_du = c.dgamma * thetaT_jac;
  }

  Eigen::Matrix3d Sk = Skew(rv);

  for (int j = 0; j < k3D; ++j) {
    // term 1: dα/d(nodes) · δ_{dim,j}
    if (dim == j) {
      result.row(j) += da_du;
    }

    // term 2: dβ/d(nodes) · θ_dim · θ_j
    double rv_dim_j = rv(dim) * rv(j);
    if (std::abs(rv_dim_j) > 1e-15) {
      result.row(j) += rv_dim_j * db_du;
    }

    // term 3: β · (jac_dim · θ_j + θ_dim · jac_j)
    if (std::abs(c.beta) > 1e-15) {
      if (std::abs(rv(j)) > 1e-15)
        result.row(j) += (c.beta * rv(j)) * jac_pos.row(dim);
      if (std::abs(rv(dim)) > 1e-15)
        result.row(j) += (c.beta * rv(dim)) * jac_pos.row(j);
    }

    // term 4: dγ/d(nodes) · [θ]×_{dim,j}
    double sk_dj = Sk(dim, j);
    if (std::abs(sk_dj) > 1e-15) {
      result.row(j) += sk_dj * dg_du;
    }

    // term 5: γ · d([θ]×_{dim,j})/d(nodes)
    // [θ]×_{dim,j} depends on one component of θ:
    //   (0,1)→-θ_z, (0,2)→θ_y, (1,0)→θ_z, (1,2)→-θ_x, (2,0)→-θ_y, (2,1)→θ_x
    if (std::abs(c.gamma) > 1e-15) {
      // The (dim,j) element of skew depends on which e_k and sign
      int k_idx = 3 - dim - j; // the "other" index (only valid for off-diag)
      if (dim != j) {
        // [θ]×_{dim,j} = ε_{dim,j,k} · θ_k where ε is Levi-Civita
        // sign: +1 if (dim,j,k) is even permutation, -1 if odd
        // encoded: (0,1)→k=2,sign=-1, (0,2)→k=1,sign=+1
        //          (1,0)→k=2,sign=+1, (1,2)→k=0,sign=-1
        //          (2,0)→k=1,sign=-1, (2,1)→k=0,sign=+1
        double sign = ((j - dim + 3) % 3 == 1) ? -1.0 : 1.0;
        result.row(j) += (c.gamma * sign) * jac_pos.row(k_idx);
      }
      // diagonal: [θ]×_{dim,dim} = 0, derivative = 0
    }
  }

  return result;
}

// Derivative of row `dim` of J̇_L w.r.t. spline node values (3×n)
RotVecConverter::Jacobian
RotVecConverter::GetDerivJLdotwrtNodes(double t, int dim) const
{
  State s = spline_->GetPoint(t);
  Vector3d rv = s.p();
  Vector3d rvd = s.v();
  double theta = rv.norm();
  auto c = ComputeCoeffs(theta);

  Jacobian jac_pos = spline_->GetJacobianWrtNodes(t, kPos);
  Jacobian jac_vel = spline_->GetJacobianWrtNodes(t, kVel);
  Jacobian result  = jac_structure_;

  // J̇_L = α̇·I + β̇·θθ^T + β·(θ̇θ^T + θθ̇^T) + γ̇·[θ]× + γ·[θ̇]×
  //
  // d(J̇_L_{dim,j})/d(nodes) has many terms from differentiating each
  // component w.r.t. nodes. Follow the same structure as GetDerivJLwrtNodes
  // but for the time derivative.

  double theta_dot = 0.0;
  if (theta > kEps) {
    theta_dot = rv.dot(rvd) / theta;
  }
  double alpha_dot = c.dalpha * theta_dot;
  double beta_dot  = c.dbeta  * theta_dot;
  double gamma_dot = c.dgamma * theta_dot;

  // We need:
  // d(α̇)/d(nodes), d(β̇)/d(nodes), d(γ̇)/d(nodes)
  // d(θ̇_i)/d(nodes) = jac_vel.row(i)
  // d(θ_i)/d(nodes) = jac_pos.row(i)

  // α̇ = α' · θ̇_angle  where θ̇_angle = θ^T·θ̇/θ
  // d(α̇)/d(nodes) is complex; we use product rule and chain rule.
  // For practical robustness, compute numerically-friendly forms.

  JacobianRow da_dot_du(jac_pos.cols());
  JacobianRow db_dot_du(jac_pos.cols());
  JacobianRow dg_dot_du(jac_pos.cols());

  if (theta > kEps) {
    double inv_theta = 1.0 / theta;
    double t2 = theta * theta;

    // d(θ̇_angle)/d(nodes) = d(θ^T·θ̇/θ)/d(nodes)
    // = (1/θ)(θ̇^T·dθ/d(nodes) + θ^T·dθ̇/d(nodes)) - θ̇_angle/θ · (θ^T/θ · dθ/d(nodes))
    JacobianRow dtheta_dot_du(jac_pos.cols());
    for (int l = 0; l < k3D; ++l) {
      dtheta_dot_du += (rvd(l) * inv_theta) * jac_pos.row(l);
      dtheta_dot_du += (rv(l) * inv_theta) * jac_vel.row(l);
    }
    // subtract the θ̇_angle/θ · n̂^T · jac_pos term
    JacobianRow nhat_jac(jac_pos.cols());
    for (int l = 0; l < k3D; ++l) {
      nhat_jac += (rv(l) * inv_theta) * jac_pos.row(l);
    }
    dtheta_dot_du -= theta_dot * inv_theta * nhat_jac;

    // d(α̇)/d(nodes) = d(α'·θ̇_angle)/d(nodes)
    //   = α'' · θ̇_angle · dθ/d(nodes)·n̂ + α' · dθ̇_angle/d(nodes)
    // where α'' = d²α/dθ² = d(α')/dθ
    // α' = (θcosθ - sinθ)/θ²
    // α'' = (-θsinθ + cosθ - cosθ + 2sinθ/θ) / θ² ... complex
    // Simpler: d(α')/d(nodes) = α'' · (n̂^T · jac_pos)
    double st = std::sin(theta);
    double ct = std::cos(theta);
    double alpha_pp = (-theta*st - 2.0*(theta*ct - st)/theta) / t2;
    double beta_pp  = 0.0; // will compute below
    double gamma_pp = 0.0;

    // β' = (-2θ - θcosθ + 3sinθ)/θ⁴
    // β'' = d(β')/dθ ... very complex, compute inline
    {
      double num  = -2.0*theta - theta*ct + 3.0*st;
      double dnum = -2.0 - ct + theta*st + 3.0*ct;
      double t4   = t2*t2;
      beta_pp = (dnum*t4 - 4.0*t2*theta*num) / (t4*t4/theta);
      // simplified: (dnum - 4*num/θ)/θ⁴ ... but let's be more careful
      beta_pp = (dnum - 4.0*num/theta) / t4;
    }
    {
      double num  = theta*st - 2.0 + 2.0*ct;
      double dnum = st + theta*ct - 2.0*st;
      double t3   = t2*theta;
      gamma_pp = (dnum - 3.0*num/theta) / t3;
    }

    // da_dot/d(nodes) = (α'' · θ̇_angle) · nhat_jac + α' · dtheta_dot_du
    da_dot_du = (alpha_pp * theta_dot) * nhat_jac + c.dalpha * dtheta_dot_du;
    db_dot_du = (beta_pp  * theta_dot) * nhat_jac + c.dbeta  * dtheta_dot_du;
    dg_dot_du = (gamma_pp * theta_dot) * nhat_jac + c.dgamma * dtheta_dot_du;
  } else {
    // at θ≈0, α̇≈0, β̇≈0, γ̇≈0 and their derivatives w.r.t. nodes ≈ 0
    // only J̇_L ≈ γ·[θ̇]×, so d(J̇_L)/d(nodes) ≈ γ·d([θ̇]×)/d(nodes) = γ·[d(θ̇)/d(nodes)]×...
    // This is handled via the γ·[θ̇]× term below.
  }

  Eigen::Matrix3d Sk  = Skew(rv);
  Eigen::Matrix3d Skd = Skew(rvd);

  for (int j = 0; j < k3D; ++j) {
    // α̇·δ_{dim,j}
    if (dim == j) {
      result.row(j) += da_dot_du;
    }

    // β̇·θ_dim·θ_j
    double rv_dj = rv(dim) * rv(j);
    if (std::abs(rv_dj) > 1e-15) {
      result.row(j) += rv_dj * db_dot_du;
    }
    // d(β̇)/d(nodes) contributes above; also need d(θ_dim·θ_j)/d(nodes) · β̇
    if (std::abs(beta_dot) > 1e-15) {
      if (std::abs(rv(j)) > 1e-15)
        result.row(j) += (beta_dot * rv(j)) * jac_pos.row(dim);
      if (std::abs(rv(dim)) > 1e-15)
        result.row(j) += (beta_dot * rv(dim)) * jac_pos.row(j);
    }

    // β·(θ̇_dim·θ_j + θ_dim·θ̇_j)
    double td_dj = rvd(dim) * rv(j) + rv(dim) * rvd(j);
    if (std::abs(td_dj) > 1e-15 && std::abs(c.beta) > 1e-15) {
      // d(β)/d(nodes) · (rvd_dim*rv_j + rv_dim*rvd_j)
      if (theta > kEps) {
        double inv_theta = 1.0 / theta;
        JacobianRow nhat_jac(jac_pos.cols());
        for (int l = 0; l < k3D; ++l)
          nhat_jac += (rv(l) * inv_theta) * jac_pos.row(l);
        result.row(j) += (c.dbeta * td_dj) * nhat_jac;
      }
    }
    // β · d(θ̇_dim·θ_j + θ_dim·θ̇_j)/d(nodes)
    if (std::abs(c.beta) > 1e-15) {
      // d(θ̇_dim·θ_j)/d(nodes) = θ_j·jac_vel.row(dim) + θ̇_dim·jac_pos.row(j)
      if (std::abs(rv(j)) > 1e-15)
        result.row(j) += (c.beta * rv(j)) * jac_vel.row(dim);
      if (std::abs(rvd(dim)) > 1e-15)
        result.row(j) += (c.beta * rvd(dim)) * jac_pos.row(j);
      // d(θ_dim·θ̇_j)/d(nodes)
      if (std::abs(rvd(j)) > 1e-15)
        result.row(j) += (c.beta * rvd(j)) * jac_pos.row(dim);
      if (std::abs(rv(dim)) > 1e-15)
        result.row(j) += (c.beta * rv(dim)) * jac_vel.row(j);
    }

    // γ̇·[θ]×_{dim,j}
    double sk_dj = Sk(dim, j);
    if (std::abs(sk_dj) > 1e-15) {
      result.row(j) += sk_dj * dg_dot_du;
    }
    // γ · d([θ]×_{dim,j})/d(nodes) ... for γ̇ term (already handled via d(γ̇)/d(nodes) above)
    // Actually we need: d(γ̇ · [θ]×_{dim,j})/d(nodes) = dγ̇/d(nodes)·sk_dj + γ̇·d(sk_dj)/d(nodes)
    if (std::abs(gamma_dot) > 1e-15 && dim != j) {
      int k_idx = 3 - dim - j;
      double sign = ((j - dim + 3) % 3 == 1) ? -1.0 : 1.0;
      result.row(j) += (gamma_dot * sign) * jac_pos.row(k_idx);
    }

    // γ·[θ̇]×_{dim,j}
    double skd_dj = Skd(dim, j);
    if (std::abs(skd_dj) > 1e-15 && std::abs(c.gamma) > 1e-15) {
      // d(γ·skd_dj)/d(nodes) = dγ/d(nodes)·skd_dj + γ·d(skd_dj)/d(nodes)
      if (theta > kEps) {
        double inv_theta = 1.0 / theta;
        JacobianRow nhat_jac(jac_pos.cols());
        for (int l = 0; l < k3D; ++l)
          nhat_jac += (rv(l) * inv_theta) * jac_pos.row(l);
        result.row(j) += (c.dgamma * skd_dj) * nhat_jac;
      }
    }
    if (std::abs(c.gamma) > 1e-15 && dim != j) {
      int k_idx = 3 - dim - j;
      double sign = ((j - dim + 3) % 3 == 1) ? -1.0 : 1.0;
      result.row(j) += (c.gamma * sign) * jac_vel.row(k_idx);
    }
  }

  return result;
}

RotVecConverter::Jacobian
RotVecConverter::GetDerivOfAngVelWrtNodes(double t) const
{
  Jacobian jac = jac_structure_;

  State s = spline_->GetPoint(t);
  JacobianRow vel = s.v().transpose().sparseView(1.0, -1.0);
  Jacobian dVel_du = spline_->GetJacobianWrtNodes(t, kVel);
  Eigen::Matrix3d JL = LeftJacobian(s.p());

  Jacobian JL_dVel;
  DenseTimesSparse(JL, dVel_du, JL_dVel);

  for (int dim : {X, Y, Z}) {
    Jacobian dJL_du = GetDerivJLwrtNodes(t, dim);
    jac.row(dim) = vel * dJL_du + JL_dVel.row(dim);
  }

  EnsureFullPattern(jac, t);
  return jac;
}

RotVecConverter::Jacobian
RotVecConverter::GetDerivOfAngAccWrtNodes(double t) const
{
  Jacobian jac = jac_structure_;

  State s = spline_->GetPoint(t);
  JacobianRow vel = s.v().transpose().sparseView(1.0, -1.0);
  JacobianRow acc = s.a().transpose().sparseView(1.0, -1.0);

  Jacobian dVel_du = spline_->GetJacobianWrtNodes(t, kVel);
  Jacobian dAcc_du = spline_->GetJacobianWrtNodes(t, kAcc);

  Eigen::Matrix3d JL    = LeftJacobian(s.p());
  Eigen::Matrix3d JLdot = LeftJacobianDot(s.p(), s.v());

  Jacobian JLdot_dVel, JL_dAcc;
  DenseTimesSparse(JLdot, dVel_du, JLdot_dVel);
  DenseTimesSparse(JL, dAcc_du, JL_dAcc);

  for (int dim : {X, Y, Z}) {
    Jacobian dJLdot_du = GetDerivJLdotwrtNodes(t, dim);
    Jacobian dJL_du    = GetDerivJLwrtNodes(t, dim);

    jac.row(dim) = vel * dJLdot_du
                 + JLdot_dVel.row(dim)
                 + acc * dJL_du
                 + JL_dAcc.row(dim);
  }

  EnsureFullPattern(jac, t);
  return jac;
}

} // namespace towr
