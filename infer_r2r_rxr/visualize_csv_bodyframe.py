"""
Trajectory Visualization with Standard Body Frame Convention
============================================================

네트워크 출력 좌표계:
  - dx: lateral (+ = 오른쪽)
  - dy: forward (+ = 앞, 진행방향)

표준 Robot Body Frame (ROS 컨벤션):
  - x_body: forward (+ = 앞)
  - y_body: left (+ = 왼쪽)

변환:
  x_body = dy_net
  y_body = -dx_net

theta 계산:
  theta = atan2(-dx_net, dy_net)  # 이미 body frame 기준

Global Frame 적분:
  Gx += x_body * cos(Gtheta) - y_body * sin(Gtheta)
  Gy += x_body * sin(Gtheta) + y_body * cos(Gtheta)
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import math
import csv

# ============================================================
# Configuration
# ============================================================
input_file = './results/waypoint_data_online_20260201_060434.csv'
output_file = './results/new_test_trajectory3.png'
PREDICT_SCALE = 0.3  # Network output scaling factor

# ============================================================
# Read CSV Data
# ============================================================
data = []
with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            'frame_idx': int(row['frame_idx']),
            'subframe_idx': int(row['subframe_idx']),
            'dx': float(row['dx']),  # Network: lateral (+ = right)
            'dy': float(row['dy']),  # Network: forward (+ = ahead)
            'dtheta': float(row['dtheta']),  # degrees
            'arrive': int(row['arrive'])
        })

# Use only first waypoint of each frame (subframe_idx = 0)
subframe_to_use = 0
filtered_data = [d for d in data if d['subframe_idx'] == subframe_to_use]

print(f"Total records: {len(data)}, Filtered (subframe={subframe_to_use}): {len(filtered_data)}")

# ============================================================
# Coordinate Transformation and Trajectory Integration
# ============================================================
# Global pose: starts at origin, facing +X direction (theta=0)
Gx, Gy, Gtheta = 0.0, 0.0, 0.0
trajectory = [(Gx, Gy, Gtheta)]

for i, record in enumerate(filtered_data):
    # Raw network output (before PREDICT_SCALE)
    dx_net = record['dx']  # lateral: + = right
    dy_net = record['dy']  # forward: + = ahead
    dtheta_deg = record['dtheta']  # degrees
    
    # Apply scaling (network output → real distance in meters)
    dx_net_scaled = dx_net * PREDICT_SCALE
    dy_net_scaled = dy_net * PREDICT_SCALE
    
    # Convert to standard body frame (+x forward, +y left)
    x_body = dy_net_scaled   # forward distance
    y_body = -dx_net_scaled  # left distance (since dx+ = right = -y_body)
    
    # Transform local displacement to global frame
    # Using standard 2D rotation: R(theta) * [x_body, y_body]^T
    delta_Gx = x_body * math.cos(Gtheta) - y_body * math.sin(Gtheta)
    delta_Gy = x_body * math.sin(Gtheta) + y_body * math.cos(Gtheta)
    
    # Update global position
    Gx += delta_Gx
    Gy += delta_Gy
    
    # Update heading (dtheta is in degrees, convert to radians)
    # Positive dtheta = turn left (CCW) in our convention
    Gtheta += math.radians(dtheta_deg)
    
    trajectory.append((Gx, Gy, Gtheta))
    
    # Debug first few steps
    if i < 5:
        print(f"Frame {record['frame_idx']}: dx_net={dx_net:.4f}, dy_net={dy_net:.4f}, "
              f"x_body={x_body:.4f}, y_body={y_body:.4f}, dtheta={dtheta_deg:.2f}°")
        print(f"  → Global: ({Gx:.4f}, {Gy:.4f}), heading={math.degrees(Gtheta):.2f}°")

# ============================================================
# Plotting
# ============================================================
fig, ax = plt.subplots(figsize=(14, 12))

x_coords = [p[0] for p in trajectory]
y_coords = [p[1] for p in trajectory]
n_points = len(trajectory)

# Color gradient: blue (start) → red (end)
cmap = cm.jet
norm = mcolors.Normalize(vmin=0, vmax=n_points - 1)

# Draw trajectory line segments with gradient color
for i in range(n_points - 1):
    color = cmap(norm(i))
    ax.plot(x_coords[i:i+2], y_coords[i:i+2], color=color, linewidth=2)

# Draw heading arrows
arrow_freq = 2  # Every N frames
arrow_scale = 0.15  # Arrow length (meters)
for i in range(0, len(trajectory), arrow_freq):
    x, y, theta = trajectory[i]
    # Arrow in heading direction (+x_body direction in global frame)
    dx_arrow = arrow_scale * math.cos(theta)
    dy_arrow = arrow_scale * math.sin(theta)
    color = cmap(norm(i))
    ax.arrow(x, y, dx_arrow, dy_arrow, head_width=0.05, head_length=0.05, 
             fc=color, ec=color, alpha=0.8)

# Mark start and end points
ax.scatter(0, 0, color='green', s=200, marker='*', label='Start (0, 0)', zorder=5)
ax.scatter(x_coords[-1], y_coords[-1], color='red', s=200, marker='o', 
           label=f'End ({x_coords[-1]:.2f}, {y_coords[-1]:.2f})', zorder=5)

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Time Step (Frame Index)')

# Labels and formatting
ax.set_title(f'Odometry Trajectory (Standard Body Frame: +X forward, +Y left)\n'
             f'File: {input_file} | Subframe {subframe_to_use} | Scale: {PREDICT_SCALE}')
ax.set_xlabel('Global X (m) - Initial Forward Direction')
ax.set_ylabel('Global Y (m) - Initial Left Direction')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Add coordinate system indicator at origin
ax.annotate('', xy=(0.3, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.annotate('', xy=(0, 0.3), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(0.35, 0, '+X (fwd)', fontsize=10, ha='left', va='center')
ax.text(0, 0.35, '+Y (left)', fontsize=10, ha='center', va='bottom')

plt.tight_layout()
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Trajectory saved to: {output_file}")

# Print summary statistics
total_distance = sum(math.sqrt((x_coords[i+1]-x_coords[i])**2 + (y_coords[i+1]-y_coords[i])**2) 
                     for i in range(len(x_coords)-1))
final_heading = math.degrees(trajectory[-1][2])
print(f"\n📊 Summary:")
print(f"  - Total frames: {len(filtered_data)}")
print(f"  - Total distance traveled: {total_distance:.2f} m")
print(f"  - Final position: ({x_coords[-1]:.2f}, {y_coords[-1]:.2f}) m")
print(f"  - Final heading: {final_heading:.2f}°")