#pragma once

#include <string>
#include <towr/variables/spline_holder.h>

namespace towr {

/**
 * @brief 保存轨迹数据到CSV文件
 * @param solution 轨迹优化解
 * @param filename 输出文件名
 * @param T_sample 采样时间间隔（秒）
 * @return 是否保存成功
 */
bool SaveTrajectoryToCSV(const towr::SplineHolder& solution, 
                        const std::string& filename, 
                        double T_sample = 0.001);

} // namespace tra_opt