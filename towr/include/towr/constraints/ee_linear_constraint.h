#ifndef TOWR_CONSTRAINTS_EE_LINEAR_CONSTRAINT_H_
#define TOWR_CONSTRAINTS_EE_LINEAR_CONSTRAINT_H_

#include <towr/constraints/time_discretization_constraint.h>
#include <towr/variables/node_spline.h>
#include <towr/variables/cartesian_dimensions.h>

namespace towr {

/**
 * @brief Linear combination constraint across endeffector splines.
 *
 * Enforces |sum_i(coeff_i * spline_i(t)[dim_i])| <= tolerance
 * at discretized time points along the trajectory.
 *
 * Example: keep two feet at similar x position:
 *   terms = {{ee=0, dim=X, coeff=1.0}, {ee=1, dim=X, coeff=-1.0}}
 *   tolerance = 0.05
 *   → |ee0_x(t) - ee1_x(t)| <= 0.05
 *
 * @ingroup Constraints
 */
class EELinearConstraint : public TimeDiscretizationConstraint {
public:
  struct Term {
    int ee;
    int dim;
    double coeff;
  };

  /**
   * @param terms       Linear combination terms.
   * @param splines     The ee splines to evaluate (indexed by ee).
   * @param var_names   Variable set names (indexed by ee), for Jacobian routing.
   * @param deriv       Which derivative to constrain (kPos or kVel).
   * @param tolerance   Allowed deviation: g ∈ [-tolerance, +tolerance].
   * @param T           Total trajectory duration.
   * @param dt          Discretization interval.
   */
  EELinearConstraint(const std::vector<Term>& terms,
                     const std::vector<NodeSpline::Ptr>& splines,
                     const std::vector<std::string>& var_names,
                     Dx deriv,
                     double tolerance,
                     double T, double dt);

  virtual ~EELinearConstraint() = default;

private:
  std::vector<Term> terms_;
  std::vector<NodeSpline::Ptr> splines_;
  std::vector<std::string> var_names_;
  Dx deriv_;
  double tolerance_;

  void UpdateConstraintAtInstance(double t, int k, VectorXd& g) const override;
  void UpdateBoundsAtInstance(double t, int k, VecBound& bounds) const override;
  void UpdateJacobianAtInstance(double t, int k, std::string var_set,
                                Jacobian& jac) const override;
};

} /* namespace towr */

#endif /* TOWR_CONSTRAINTS_EE_LINEAR_CONSTRAINT_H_ */
