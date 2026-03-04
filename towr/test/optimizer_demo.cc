#include <cmath>
#include <iostream>

#include <towr/optimizer/towr_optimizer.h>
#include <towr/models/examples/monoped_model.h>
#include <towr/models/examples/biped_model.h>
#include <towr/terrain/examples/height_map_examples.h>

using namespace towr;

int main()
{
  // ====================================================================
  //  Part A: Monoped 单足示例
  // ====================================================================
  TowrOptimizer mono;
  mono.setModel(std::make_shared<MonopedKinematicModel>(),
                std::make_shared<MonopedDynamicModel>());
  mono.setTerrain(std::make_shared<FlatGround>(0.0));

  std::cout << "\n========== [Mono] 原地垂直跳 ==========\n";
  {
    JumpConfig jc;
    jc.standing_height = 0.58;
    mono.solveJump(jc);
    mono.printTrajectory();
    mono.saveCSV("mono_vertical_jump.csv");
  }

  std::cout << "\n========== [Mono] 后空翻 ==========\n";
  {
    FlipConfig fc;
    fc.type = FlipConfig::BackFlip;
    fc.standing_height = 0.58;
    mono.solveFlip(fc);
    mono.printTrajectory();
    mono.saveCSV("mono_backflip.csv");
  }

  // ====================================================================
  //  Part B: Biped 双足示例
  // ====================================================================
  TowrOptimizer biped;
  biped.setModel(std::make_shared<BipedKinematicModel>(),
                 std::make_shared<BipedDynamicModel>());
  biped.setTerrain(std::make_shared<FlatGround>(0.0));

  // --- 双足原地跳 ---
  std::cout << "\n========== [Biped] 原地跳 ==========\n";
  {
    JumpConfig jc;
    jc.standing_height = 0.65;        // BipedModel 的 nominal Z
    jc.flight_duration = 0.3;
    biped.solveJump(jc);
    biped.printTrajectory();
    biped.saveCSV("biped_jump.csv");
  }

  // --- 双足前跳 1m ---
  std::cout << "\n========== [Biped] 前跳 1m ==========\n";
  {
    JumpConfig jc;
    jc.standing_height = 0.65;
    jc.displacement = {1.0, 0.0, 0.0};
    jc.flight_duration = 0.4;
    biped.solveJump(jc);
    biped.printTrajectory();
    biped.saveCSV("biped_forward_jump.csv");
  }

  // --- 双足后空翻 ---
  std::cout << "\n========== [Biped] 后空翻 ==========\n";
  {
    FlipConfig fc;
    fc.type             = FlipConfig::BackFlip;
    fc.standing_height  = 0.65;       // 和运动学模型一致
    fc.rotation_amount  = 2.0 * M_PI; // 一整圈
    fc.crouch_ratio     = 0.5;        // 蹲到站立高度 50%
    fc.crouch_duration  = 0.3;
    fc.push_duration    = 0.2;
    fc.flight_duration  = 0.8;        // 腾空 0.8s 完成 360°
    fc.absorb_duration  = 0.2;
    fc.recover_duration = 0.3;
    fc.force_limit      = 2000.0;
    fc.solver.max_iter     = 5000;
    fc.solver.max_cpu_time = 60.0;
    fc.solver.tol          = 1e-3;
    fc.solver.jacobian_approx = "exact";

    biped.solveFlip(fc);
    biped.printTrajectory();
    biped.saveCSV("biped_backflip.csv");
  }

  // --- 双足前空翻 ---
  std::cout << "\n========== [Biped] 前空翻 ==========\n";
  {
    FlipConfig fc;
    fc.type            = FlipConfig::FrontFlip;
    fc.standing_height = 0.65;
    fc.flight_duration = 0.8;
    fc.solver.max_iter = 5000;

    biped.solveFlip(fc);
    biped.printTrajectory();
    biped.saveCSV("biped_frontflip.csv");
  }

  return 0;
}
