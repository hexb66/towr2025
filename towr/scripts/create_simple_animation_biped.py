#!/usr/bin/env python3
"""
Simple TOWR Robot Animation Script

This script creates a simplified animated visualization focusing on robot motion.

Usage: python3 create_simple_animation_biped.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import matplotlib.gridspec as gridspec

def load_trajectory_data(csv_file="./build/biped_trajectory.csv"):
    """Load trajectory data from CSV file"""
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found!")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"Loaded trajectory data: {len(df)} samples")
    print(f"Time range: {df['time'].min():.3f} - {df['time'].max():.3f} seconds")
    return df

def create_simple_robot_animation(df, skip_frames=20):
    """Create simplified robot motion animation"""
    print(f"\nCreating simple robot animation (skip_frames={skip_frames})...")
    
    # Downsample data for smoother animation
    df_anim = df.iloc[::skip_frames].copy()
    print(f"Animation will have {len(df_anim)} frames")
    
    # Extract trajectory data
    time = df_anim['time'].values
    base_x = df_anim['base_pos_x'].values
    base_y = df_anim['base_pos_y'].values
    base_z = df_anim['base_pos_z'].values
    
    foot_x_L = df_anim['ee_pos_x_L'].values
    foot_y_L = df_anim['ee_pos_y_L'].values
    foot_z_L = df_anim['ee_pos_z_L'].values
    
    foot_x_R = df_anim['ee_pos_x_R'].values
    foot_y_R = df_anim['ee_pos_y_R'].values
    foot_z_R = df_anim['ee_pos_z_R'].values
    
    force_x_L = df_anim['contact_force_x_L'].values
    force_y_L = df_anim['contact_force_y_L'].values
    force_z_L = df_anim['contact_force_z_L'].values
    torque_y_L = df_anim['contact_torque_y_L'].values
    
    force_x_R = df_anim['contact_force_x_R'].values
    force_y_R = df_anim['contact_force_y_R'].values
    force_z_R = df_anim['contact_force_z_R'].values
    torque_y_R = df_anim['contact_torque_y_R'].values
    
    contact_phase_L = df_anim['is_contact_phase_L'].values
    contact_phase_R = df_anim['is_contact_phase_R'].values
    
    base_roll = df_anim['base_euler_roll'].values
    base_pitch = df_anim['base_euler_pitch'].values
    base_yaw = df_anim['base_euler_yaw'].values
    
    # Set up the figure - 增加宽度
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.7, 0.3])
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax2 = fig.add_subplot(gs[1])
    fig.suptitle('TOWR Biped Simple Animation (3D)', fontsize=14)
    
    # 3D plot range - 计算真实尺寸范围
    x_range = [min(np.min(base_x), np.min(foot_x_L), np.min(foot_x_R)) - 0.3, max(np.max(base_x), np.max(foot_x_L), np.max(foot_x_R)) + 0.3]
    y_range = [min(np.min(base_y), np.min(foot_y_L), np.min(foot_y_R)) - 0.3, max(np.max(base_y), np.max(foot_y_L), np.max(foot_y_R)) + 0.3]
    z_range = [-0.1, max(np.max(base_z), np.max(foot_z_L), np.max(foot_z_R)) + 0.2]
    
    # 设置坐标轴范围
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.set_zlim(z_range)
    
    # 使用固定的比例而不是数据范围，这样可以更好地控制3D视图的大小
    # 根据数据范围计算合适的比例
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0] 
    z_span = z_range[1] - z_range[0]
    
    # 归一化比例，使最大的轴为1.0，其他轴按比例缩放
    max_span = max(x_span, y_span, z_span)
    aspect_ratio = [x_span/max_span, y_span/max_span, z_span/max_span]
    
    ax1.set_box_aspect(aspect_ratio)
    
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.set_title('Robot Motion (3D View)')
    
    # ----------- 楼梯地形3D立方体绘制 -----------
    step_depth = 0.3
    step_height = 0.15
    num_steps = 5
    stairs_start = 0.5
    stair_width = 1.0  # y方向宽度
    stair_boxes = []
    for i in range(num_steps):
        x0 = stairs_start + i*step_depth
        x1 = stairs_start + (i+1)*step_depth
        y0 = -stair_width/2
        y1 = stair_width/2
        z0 = 0 if i == 0 else step_height*i
        z1 = step_height*(i+1)
        # 每一级台阶的上表面四点
        top = np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
        # 前侧面
        front = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]])
        # 后侧面
        back = np.array([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]])
        # 左侧面
        left = np.array([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]])
        # 右侧面
        right = np.array([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]])
        # 底面
        bottom = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]])
        # stair_boxes.append([top, front, back, left, right, bottom])
        stair_boxes.append([top, left])
    # 后平台
    x0 = stairs_start + num_steps*step_depth
    x1 = x0 + 0.5
    y0 = -stair_width/2
    y1 = stair_width/2
    z0 = step_height*num_steps
    top = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]])
    stair_boxes.append([top])
    # 绘制所有台阶
    for faces in stair_boxes:
        pc = Poly3DCollection(faces, facecolor='saddlebrown', edgecolor='saddlebrown', alpha=0.5)
        ax1.add_collection3d(pc)
    
    # Robot parameters
    body_width = 0.3
    body_height = 0.5
    body_depth = 0.2
    
    # Initialize plot elements
    body_point, = ax1.plot([], [], [], 'bo', markersize=10, label='Base', zorder=10)
    foot_point_L, = ax1.plot([], [], [], 'ro', markersize=8, label='Foot', zorder=10)
    foot_point_R, = ax1.plot([], [], [], 'ro', markersize=8, label='Foot', zorder=10)
    leg_line_L, = ax1.plot([], [], [], 'k-', linewidth=3, label='Leg_L', zorder=10)
    leg_line_R, = ax1.plot([], [], [], 'k-', linewidth=3, label='Leg_R', zorder=10)
    trajectory, = ax1.plot([], [], [], 'b--', alpha=0.6, linewidth=1, label='Base Trajectory', zorder=10)
    info_text = ax1.text2D(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10, zorder=10,
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # 设置3D视图为透视模式
    ax1.set_proj_type('ortho')
    ax1.view_init(elev=0, azim=-90) 
    ax1.legend()
    
    # Force plot
    ax2.set_xlim(time[0], time[-1])
    force_max_L = max(np.sqrt(force_x_L**2 + force_y_L**2 + force_z_L**2)) if len(force_x_L) > 0 else 1
    force_max_R = max(np.sqrt(force_x_R**2 + force_y_R**2 + force_z_R**2)) if len(force_x_R) > 0 else 1
    force_max = max(force_max_L, force_max_R)
    ax2.set_ylim(-force_max * 1.1, force_max * 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Force [N]')
    ax2.set_title('Contact Forces')
    
    force_line_L, = ax2.plot([], [], 'g-', linewidth=2, label='Force Z_L')
    torque_line_L, = ax2.plot([], [], 'purple', linewidth=2, label='Torque Y_L')
    force_line_R, = ax2.plot([], [], 'g-', linewidth=2, label='Force Z_R')
    torque_line_R, = ax2.plot([], [], 'purple', linewidth=2, label='Torque Y_R')
    ax2.legend()
    
    # 初始化body立方体
    def get_body_vertices(center, roll, pitch, yaw):
        w, d, h = body_width, body_depth, body_height
        x = np.array([-w/2, w/2])
        y = np.array([-d/2, d/2])
        z = np.array([0, h])
        corners = np.array([[xi, yi, zi] for xi in x for yi in y for zi in z])
        # 欧拉角ZYX顺序：先yaw(z),再pitch(y),再roll(x)
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw),  np.cos(yaw), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll),  np.cos(roll)]])
        R = Rz @ Ry @ Rx
        rotated = corners @ R.T
        return rotated + center
    # 初始化body多面体
    body_faces = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
    
    def animate(frame):
        """Simple animation function with error checking"""
        try:
            if frame >= len(time):
                return
            
            current_time = time[frame]
            current_contact_L = contact_phase_L[frame] > 0.5
            current_contact_R = contact_phase_R[frame] > 0.5
            
            # 机体
            body_point.set_data([base_x[frame]], [base_y[frame]])
            body_point.set_3d_properties([base_z[frame]])
            # 足端
            foot_point_L.set_data([foot_x_L[frame]], [foot_y_L[frame]])
            foot_point_L.set_3d_properties([foot_z_L[frame]])
            foot_point_R.set_data([foot_x_R[frame]], [foot_y_R[frame]])
            foot_point_R.set_3d_properties([foot_z_R[frame]])
            # 腿
            leg_line_L.set_data([base_x[frame], foot_x_L[frame]], [base_y[frame], foot_y_L[frame]])
            leg_line_L.set_3d_properties([base_z[frame], foot_z_L[frame]])
            leg_line_R.set_data([base_x[frame], foot_x_R[frame]], [base_y[frame], foot_y_R[frame]])
            leg_line_R.set_3d_properties([base_z[frame], foot_z_R[frame]])
            # 轨迹
            current_idx = min(frame + 1, len(time))
            if current_idx > 0:
                trajectory.set_data(base_x[:current_idx], base_y[:current_idx])
                trajectory.set_3d_properties(base_z[:current_idx])
            
            # 信息
            force_mag_L = np.sqrt(force_x_L[frame]**2 + force_y_L[frame]**2 + force_z_L[frame]**2)
            force_mag_R = np.sqrt(force_x_R[frame]**2 + force_y_R[frame]**2 + force_z_R[frame]**2)
            force_mag = max(force_mag_L, force_mag_R)
            info_text.set_text(f'Time: {current_time:.3f}s\n'
                               f'Contact_L: {"YES" if current_contact_L else "NO"}\n'
                               f'Contact_R: {"YES" if current_contact_R else "NO"}\n'
                               f'Force: {force_mag:.1f}N\n'
                               f'Torque: {torque_y_L[frame]:.1f}Nm')
            
            # 力曲线
            if current_idx > 0:
                force_line_L.set_data(time[:current_idx], force_z_L[:current_idx])
                torque_line_L.set_data(time[:current_idx], torque_y_L[:current_idx])
                force_line_R.set_data(time[:current_idx], force_z_R[:current_idx])
                torque_line_R.set_data(time[:current_idx], torque_y_R[:current_idx])
                
            # 机体姿态（roll, pitch, yaw）
            roll = base_roll[frame]
            pitch = base_pitch[frame]
            yaw = base_yaw[frame]
            center = np.array([base_x[frame], base_y[frame], base_z[frame]])
            verts = get_body_vertices(center, roll, pitch, yaw)
            face_verts = [verts[face] for face in body_faces]
            if hasattr(animate, 'body_box'):
                animate.body_box.set_verts(face_verts)
            else:
                animate.body_box = Poly3DCollection(face_verts, facecolor='lightblue', edgecolor='blue', alpha=0.8, zorder=10)
                ax1.add_collection3d(animate.body_box)
                
        except Exception as e:
            print(f"Animation error at frame {frame}: {e}")
    
    # Create animation
    # 计算动画参数
    original_time_step = time[1] - time[0]  # 原始数据时间间隔
    total_time = time[-1] - time[0]         # 总时长
    total_frames = len(time)                # 动画总帧数
    
    # 计算GIF的FPS，让GIF在真实时间内播放完
    gif_fps = (total_frames-1) / total_time
    
    print(f"Original time step: {original_time_step:.6f}s")
    print(f"Skip frames: {skip_frames}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Total frames: {total_frames}")
    print(f"GIF FPS: {gif_fps:.1f}")
    
    # 设置动画播放间隔（毫秒）
    animation_interval = 1
    
    anim = FuncAnimation(fig, animate, frames=total_frames, interval=animation_interval, 
                        blit=False, repeat=True)
    
    plt.tight_layout()

    anim.save("res/biped_simple_animation_3d.mp4", fps=gif_fps, writer="ffmpeg")
    
    # # Save animation
    # print("Saving simple animation as GIF...")
    # try:
    #     writer = PillowWriter(fps=gif_fps)
    #     anim.save('biped_simple_animation_3d.gif', writer=writer)
    #     print(f"Simple animation saved as: biped_simple_animation_3d.gif (FPS: {gif_fps:.1f})")
        
    #     # Check file
    #     import os
    #     if os.path.exists('biped_simple_animation_3d.gif'):
    #         file_size = os.path.getsize('biped_simple_animation_3d.gif') / (1024*1024)
    #         print(f"GIF file size: {file_size:.1f} MB")
        
    # except Exception as e:
    #     print(f"Error saving simple animation: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    # Show animation
    print("Showing simple animation...")
    plt.show()
    
    return fig, anim

def main():
    """Main function"""
    print("TOWR Simple Robot Animation (3D)")
    print("=" * 40)
    
    # Load data
    df = load_trajectory_data()
    if df is None:
        print("Cannot load trajectory data. Please run biped example first.")
        return
    
    # Create animation with smaller skip_frames for smoother real-time animation
    fig, anim = create_simple_robot_animation(df, skip_frames=50)
    
    print("Simple animation complete!")

if __name__ == "__main__":
    main() 