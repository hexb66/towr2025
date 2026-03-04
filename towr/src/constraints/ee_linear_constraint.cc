#include <towr/constraints/ee_linear_constraint.h>

namespace towr {

EELinearConstraint::EELinearConstraint(
    const std::vector<Term>& terms,
    const std::vector<NodeSpline::Ptr>& splines,
    const std::vector<std::string>& var_names,
    Dx deriv,
    double tolerance,
    double T, double dt)
    : TimeDiscretizationConstraint(T, dt, "ee-linear"),
      terms_(terms), splines_(splines), var_names_(var_names),
      deriv_(deriv), tolerance_(tolerance)
{
  SetRows(GetNumberOfNodes());
}

void
EELinearConstraint::UpdateConstraintAtInstance(double t, int k, VectorXd& g) const
{
  double val = 0.0;
  for (const auto& term : terms_) {
    auto pt = splines_.at(term.ee)->GetPoint(t);
    double v = (deriv_ == kPos) ? pt.p()(term.dim) : pt.v()(term.dim);
    val += term.coeff * v;
  }
  g(k) = val;
}

void
EELinearConstraint::UpdateBoundsAtInstance(double t, int k, VecBound& bounds) const
{
  bounds.at(k) = ifopt::Bounds(-tolerance_, tolerance_);
}

void
EELinearConstraint::UpdateJacobianAtInstance(double t, int k,
                                             std::string var_set,
                                             Jacobian& jac) const
{
  for (const auto& term : terms_) {
    if (var_set == var_names_.at(term.ee)) {
      auto jac_nodes = splines_.at(term.ee)->GetJacobianWrtNodes(t, deriv_);
      jac.row(k) += term.coeff * jac_nodes.row(term.dim);
    }
  }
}

} /* namespace towr */
