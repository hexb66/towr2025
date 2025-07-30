#!/usr/bin/env python3
"""
Simple TOWR Robot Animation Script

This script creates a simplified animated visualization focusing on robot motion.

Usage: python3 create_simple_animation.py
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

def load_trajectory_data(csv_file="./build/hopper_trajectory.csv"):
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
    
    foot_x = df_anim['ee_pos_x'].values
    foot_y = df_anim['ee_pos_y'].values
    foot_z = df_anim['ee_pos_z'].values
    
    force_x = df_anim['contact_force_x'].values
    force_y = df_anim['contact_force_y'].values
    force_z = df_anim['contact_force_z'].values
    torque_y = df_anim['contact_torque_y'].values
    
    contact_phase = df_anim['is_contact_phase'].values
    base_roll = df_anim['base_euler_roll'].values
    base_pitch = df_anim['base_euler_pitch'].values
    base_yaw = df_anim['base_euler_yaw'].values
    
    # Set up the figure
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.7, 0.3])
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax2 = fig.add_subplot(gs[1])
    fig.suptitle('TOWR Hopper Simple Animation (3D)', fontsize=14)
    
    # 3D plot range
    x_range = [min(np.min(base_x), np.min(foot_x)) - 0.3, max(np.max(base_x), np.max(foot_x)) + 0.3]
    y_range = [min(np.min(base_y), np.min(foot_y)) - 0.3, max(np.max(base_y), np.max(foot_y)) + 0.3]
    z_range = [-0.1, max(np.max(base_z), np.max(foot_z)) + 0.2]
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.set_zlim(z_range)
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
    body_width = 0.12
    body_height = 0.06
    body_depth = 0.08
    
    # Initialize plot elements
    body_point, = ax1.plot([], [], [], 'bo', markersize=10, label='Base', zorder=10)
    foot_point, = ax1.plot([], [], [], 'ro', markersize=8, label='Foot', zorder=10)
    leg_line, = ax1.plot([], [], [], 'k-', linewidth=3, label='Leg', zorder=10)
    trajectory, = ax1.plot([], [], [], 'b--', alpha=0.6, linewidth=1, label='Base Trajectory', zorder=10)
    info_text = ax1.text2D(0.02, 0.98, '', transform=ax1.transAxes, fontsize=10, zorder=10,
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    # 设置3D视图为透视模式
    ax1.set_proj_type('persp')
    ax1.legend()
    
    # Force plot
    ax2.set_xlim(time[0], time[-1])
    force_max = max(np.sqrt(force_x**2 + force_y**2 + force_z**2)) if len(force_x) > 0 else 1
    ax2.set_ylim(-force_max * 1.1, force_max * 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Force [N]')
    ax2.set_title('Contact Forces')
    
    force_line, = ax2.plot([], [], 'g-', linewidth=2, label='Force Z')
    torque_line, = ax2.plot([], [], 'purple', linewidth=2, label='Torque Y')
    ax2.legend()
    
    # 初始化body立方体
    def get_body_vertices(center, roll, pitch, yaw):
        w, d, h = body_width, body_depth, body_height
        x = np.array([-w/2, w/2])
        y = np.array([-d/2, d/2])
        z = np.array([-h/2, h/2])
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
            current_contact = contact_phase[frame] > 0.5
            
            # 机体
            body_point.set_data([base_x[frame]], [base_y[frame]])
            body_point.set_3d_properties([base_z[frame]])
            # 足端
            foot_point.set_data([foot_x[frame]], [foot_y[frame]])
            foot_point.set_3d_properties([foot_z[frame]])
            # 腿
            leg_line.set_data([base_x[frame], foot_x[frame]], [base_y[frame], foot_y[frame]])
            leg_line.set_3d_properties([base_z[frame], foot_z[frame]])
            # 轨迹
            current_idx = min(frame + 1, len(time))
            if current_idx > 0:
                trajectory.set_data(base_x[:current_idx], base_y[:current_idx])
                trajectory.set_3d_properties(base_z[:current_idx])
            
            # 信息
            force_mag = np.sqrt(force_x[frame]**2 + force_y[frame]**2 + force_z[frame]**2)
            info_text.set_text(f'Time: {current_time:.3f}s\n'
                               f'Contact: {"YES" if current_contact else "NO"}\n'
                               f'Force: {force_mag:.1f}N\n'
                               f'Torque: {torque_y[frame]:.1f}Nm')
            
            # 力曲线
            if current_idx > 0:
                force_line.set_data(time[:current_idx], force_z[:current_idx])
                torque_line.set_data(time[:current_idx], torque_y[:current_idx])
                
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
    interval = max(100, int(1000 * (time[1] - time[0]) * skip_frames))
    anim = FuncAnimation(fig, animate, frames=len(time), interval=interval, 
                        blit=False, repeat=True)
    
    plt.tight_layout()
    
    # Save animation
    print("Saving simple animation as GIF...")
    try:
        writer = PillowWriter(fps=15)
        anim.save('res/hopper_simple_animation_3d.gif', writer=writer)
        print("Simple animation saved as: hopper_simple_animation_3d.gif")
        
        # Check file
        import os
        if os.path.exists('res/hopper_simple_animation_3d.gif'):
            file_size = os.path.getsize('res/hopper_simple_animation_3d.gif') / (1024*1024)
            print(f"GIF file size: {file_size:.1f} MB")
        
    except Exception as e:
        print(f"Error saving simple animation: {e}")
        import traceback
        traceback.print_exc()
    
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
        print("Cannot load trajectory data. Please run hopper example first.")
        return
    
    # Create animation
    fig, anim = create_simple_robot_animation(df, skip_frames=30)
    
    print("Simple animation complete!")

if __name__ == "__main__":
    main() 