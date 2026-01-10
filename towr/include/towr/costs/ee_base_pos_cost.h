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

#ifndef TOWR_COSTS_EE_BASE_POS_COST_H_
#define TOWR_COSTS_EE_BASE_POS_COST_H_

#include <ifopt/cost_term.h>

#include <towr/variables/spline_holder.h>
#include <towr/variables/euler_converter.h>

namespace towr {

/**
 * @brief Softly keeps the endeffector position fixed in base frame during swing.
 *
 * Penalizes || p_ee^B(t) - p_ref^B ||^2 for all samples in swing phases, where:
 * p_ee^B(t) = R_BW(t) * (p_ee^W(t) - p_base^W(t)).
 *
 * This encourages feet to stay at the robot sides during turning jumps.
 *
 * @ingroup Costs
 */
class EEBasePosCost : public ifopt::CostTerm {
public:
  using EE = uint;
  using Vector3d = Eigen::Vector3d;
  using Jacobian = ifopt::Component::Jacobian;

  EEBasePosCost(const SplineHolder& splines,
                EE ee,
                const Vector3d& p_ref_B,
                double weight,
                double dt);

  double GetCost() const override;
  void FillJacobianBlock(std::string var_set, Jacobian& jac) const override;

private:
  std::vector<double> GetSampleTimes() const;

  SplineHolder splines_;
  EE ee_;
  Vector3d p_ref_B_;
  double weight_;
  double dt_;
};

} // namespace towr

#endif /* TOWR_COSTS_EE_BASE_POS_COST_H_ */


