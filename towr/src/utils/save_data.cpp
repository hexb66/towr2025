#include <towr/utils/save_data.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>

namespace towr {

bool SaveTrajectoryToCSV(const towr::SplineHolder& solution, 
                        const std::string& filename, 
                        double T_sample) {
    
    // 获取总时间和接触点个数
    double T_total = solution.base_linear_->GetTotalTime();
    int ee_count = solution.ee_motion_.size();
    
    // 创建CSV文件
    std::ofstream csv_file(filename);
    if (!csv_file.is_open()) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return false;
    }
    
    csv_file << std::fixed << std::setprecision(6);
    
    // 写入CSV头部
    csv_file << "time,"
             << "base_pos_x,base_pos_y,base_pos_z,"
             << "base_vel_x,base_vel_y,base_vel_z,"
             << "base_acc_x,base_acc_y,base_acc_z,"
             << "base_euler_roll,base_euler_pitch,base_euler_yaw,"
             << "base_omega_x,base_omega_y,base_omega_z,"
             << "base_omegadot_x,base_omegadot_y,base_omegadot_z";
    
    // 为每个接触点添加列
    for (int i = 0; i < ee_count; ++i) {
        csv_file << ",ee_pos_x_" << i << ",ee_pos_y_" << i << ",ee_pos_z_" << i
                 << ",ee_vel_x_" << i << ",ee_vel_y_" << i << ",ee_vel_z_" << i
                 << ",ee_acc_x_" << i << ",ee_acc_y_" << i << ",ee_acc_z_" << i
                 << ",ee_yaw_" << i << ",ee_yaw_rate_" << i << ",ee_yaw_acc_" << i
                 << ",contact_force_x_" << i << ",contact_force_y_" << i << ",contact_force_z_" << i
                 << ",contact_torque_x_" << i << ",contact_torque_y_" << i << ",contact_torque_z_" << i
                 << ",is_contact_phase_" << i;
    }
    csv_file << std::endl;
    
    std::cout << "正在导出轨迹到CSV文件..." << std::endl;
    std::cout << "总时间: " << T_total << " 秒" << std::endl;
    std::cout << "采样周期: " << T_sample << " 秒" << std::endl;
    
    int sample_count = 0;
    double t = 0.0;
    
    while (t <= T_total + 1e-9) {
        // 基座线性运动
        auto base_linear_state = solution.base_linear_->GetPoint(t);
        Eigen::Vector3d base_pos = base_linear_state.p();
        Eigen::Vector3d base_vel = base_linear_state.v();
        Eigen::Vector3d base_acc = base_linear_state.a();
        
        // 基座角运动
        auto base_angular_state = solution.base_angular_->GetPoint(t);
        Eigen::Vector3d base_euler = base_angular_state.p();
        Eigen::Vector3d base_omega = base_angular_state.v();
        Eigen::Vector3d base_omegadot = base_angular_state.a();
        
        // 写入基座数据
        csv_file << t << ","
                 << base_pos.x() << "," << base_pos.y() << "," << base_pos.z() << ","
                 << base_vel.x() << "," << base_vel.y() << "," << base_vel.z() << ","
                 << base_acc.x() << "," << base_acc.y() << "," << base_acc.z() << ","
                 << base_euler.x() << "," << base_euler.y() << "," << base_euler.z() << ","
                 << base_omega.x() << "," << base_omega.y() << "," << base_omega.z() << ","
                 << base_omegadot.x() << "," << base_omegadot.y() << "," << base_omegadot.z();
        
        // 为每个接触点写入数据
        for (int i = 0; i < ee_count; i++) {
            // 末端执行器运动
            auto ee_state = solution.ee_motion_.at(i)->GetPoint(t);
            Eigen::Vector3d ee_pos = ee_state.p();
            Eigen::Vector3d ee_vel = ee_state.v();
            Eigen::Vector3d ee_acc = ee_state.a();

            // 末端执行器 yaw
            auto ee_yaw_state = solution.ee_yaw_.at(i)->GetPoint(t);
            double ee_yaw = ee_yaw_state.p()(0);
            double ee_yaw_rate = ee_yaw_state.v()(0);
            double ee_yaw_acc = ee_yaw_state.a()(0);
            
            // 接触力和力矩
            Eigen::Vector3d contact_force = solution.ee_force_.at(i)->GetPoint(t).p();
            Eigen::Vector3d contact_torque = solution.ee_torque_.at(i)->GetPoint(t).p();
            
            // 接触相位
            bool is_contact = solution.phase_durations_.at(i)->IsContactPhase(t);
            
            // 写入接触点数据
            csv_file << "," << ee_pos.x() << "," << ee_pos.y() << "," << ee_pos.z()
                     << "," << ee_vel.x() << "," << ee_vel.y() << "," << ee_vel.z()
                     << "," << ee_acc.x() << "," << ee_acc.y() << "," << ee_acc.z()
                     << "," << ee_yaw << "," << ee_yaw_rate << "," << ee_yaw_acc
                     << "," << contact_force.x() << "," << contact_force.y() << "," << contact_force.z()
                     << "," << contact_torque.x() << "," << contact_torque.y() << "," << contact_torque.z()
                     << "," << (is_contact ? 1 : 0);
        }
        
        csv_file << std::endl;
        
        t += T_sample;
        sample_count++;
    }
    
    csv_file.close();
    
    std::cout << "轨迹导出成功！" << std::endl;
    std::cout << "总样本数: " << sample_count << std::endl;
    std::cout << "文件: " << filename << std::endl;
    
    // 统计信息
    std::cout << "\n====================\n轨迹摘要:\n====================\n";
    std::cout << "总轨迹时间: " << T_total << " 秒" << std::endl;
    std::cout << "CSV采样率: " << (1.0/T_sample) << " Hz" << std::endl;
    std::cout << "总CSV样本数: " << sample_count << std::endl;
    std::cout << "接触点数量: " << ee_count << std::endl;
    std::cout << "CSV文件大小: ~" << (sample_count * (19 + ee_count * 16) * 8 / 1024) << " KB" << std::endl;
    
    return true;
}

} // namespace tra_opt
