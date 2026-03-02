"""
Base Trajectory Visualization - Direct Base Frame Control
==========================================================

제어 입력: Base 프레임 기준 (v_x, omega_z)
- v_x: 전진 선속도 (x_base 방향)
- omega_z: 각속도 (z_base 축 기준 yaw rate)

Unicycle/Differential Drive Model:
- Nonholonomic constraint: lateral velocity (v_y) = 0
- Motion은 body frame에서 forward direction으로만 발생

좌표계 변환 (Camera → Base):
  Camera (OpenCV): X-right, Y-down, Z-forward, Yaw around Y
  Base (ROS):      X-forward, Y-left, Z-up, Yaw around Z
  
  x_base = z_cam (forward)
  y_base = -x_cam (left)
  theta_base = -theta_cam (ROS convention: CCW positive)
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import re
import os
import glob
import argparse
import csv

# ============================================================
# SE(3) Utilities (simplified for planar motion)
# ============================================================
def se3_from_xytheta(x: float, y: float, theta: float) -> np.ndarray:
    """Construct SE(3) matrix from 2D pose (x, y, theta)"""
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[0, 3] = x
    T[1, 3] = y
    return T

def se3_to_xytheta(T: np.ndarray) -> tuple:
    """Extract 2D pose (x, y, theta) from SE(3) matrix"""
    x = T[0, 3]
    y = T[1, 3]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return x, y, theta

# ============================================================
# Camera to Base Coordinate Transform
# ============================================================
def cam_waypoint_to_base(dx_cam: float, dz_cam: float, dtheta_y_cam_deg: float) -> tuple:
    """
    카메라 프레임 waypoint를 base 프레임으로 변환
    
    Camera frame:
      - dx_cam: lateral (right +)
      - dz_cam: forward (+)
      - dtheta_y_cam: yaw around Y (CW positive when viewed from above)
    
    Base frame (ROS):
      - dx_base: forward (+)
      - dy_base: left (+)
      - dtheta_z_base: yaw around Z (CCW positive when viewed from above)
    
    Returns:
        (dx_base, dy_base, dtheta_z_base_rad)
    """
    dx_base = dz_cam       # forward = z_cam
    dy_base = -dx_cam      # left = -x_cam
    dtheta_z_base = -np.radians(dtheta_y_cam_deg)  # ROS CCW convention
    
    return dx_base, dy_base, dtheta_z_base

# ============================================================
# Configuration
# ============================================================
parser = argparse.ArgumentParser(description='Visualize trajectory with base frame control input')
parser.add_argument('--scale', type=float, default=0.3, 
                    help='PREDICT_SCALE factor (default: 0.3)')
parser.add_argument('--waypoint-index', type=int, default=0, 
                    help='Waypoint index to use from log (0-4, default: 0)')
parser.add_argument('--log-file', type=str, default=None,
                    help='Path to log file (if not specified, auto-detect when not using --csv-file)')
parser.add_argument('--csv-file', type=str, default=None,
                    help='Path to parsed CSV (e.g. ./results/waypoint_data_online_20260201_060434.csv). If set, use CSV instead of log.')
parser.add_argument('--dt-executed', type=float, default=None,
                    help='Override dt_executed (if not in log)')
parser.add_argument('--dt-multiplier', type=float, default=1.0,
                    help='Multiplier for dt_to_waypoint values (default: 1.0)')
args = parser.parse_args()

PREDICT_SCALE = args.scale
waypoint_index = args.waypoint_index
dt_multiplier = args.dt_multiplier

# dt_to_waypoint lookup table
DT_TO_WAYPOINT_BASE = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0}
DT_TO_WAYPOINT = {k: v * dt_multiplier for k, v in DT_TO_WAYPOINT_BASE.items()}

if waypoint_index < 0 or waypoint_index > 4:
    print(f"❌ waypoint_index는 0-4 사이여야 합니다. (현재: {waypoint_index})")
    exit(1)

# Input: CSV or log file
use_csv = args.csv_file is not None and args.csv_file.strip() != ''
input_basename = None

if use_csv:
    csv_file = args.csv_file.strip()
    if not os.path.exists(csv_file):
        print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_file}")
        exit(1)
    input_basename = os.path.basename(csv_file).replace('.csv', '')
    print(f"✅ CSV 입력: {os.path.basename(csv_file)}")
else:
    # Find log file
    if args.log_file:
        if os.path.exists(args.log_file):
            log_file = args.log_file
            print(f"✅ 지정된 로그 파일: {os.path.basename(log_file)}")
        else:
            print(f"❌ 지정된 로그 파일을 찾을 수 없습니다: {args.log_file}")
            exit(1)
    else:
        current_dir = os.getcwd()
        log_files = glob.glob(os.path.join(current_dir, '*.log'))
        if len(log_files) == 0:
            print("❌ 현재 폴더에 .log 파일이 없습니다.")
            exit(1)
        elif len(log_files) == 1:
            log_file = log_files[0]
            print(f"✅ 자동으로 발견된 로그 파일: {os.path.basename(log_file)}")
        else:
            print(f"📁 {len(log_files)}개의 .log 파일을 발견했습니다:")
            for i, f in enumerate(log_files, 1):
                print(f"  {i}. {os.path.basename(f)}")
            while True:
                try:
                    choice = input(f"\n사용할 파일 번호를 선택하세요 (1-{len(log_files)}): ")
                    idx = int(choice) - 1
                    if 0 <= idx < len(log_files):
                        log_file = log_files[idx]
                        print(f"✅ 선택된 파일: {os.path.basename(log_file)}")
                        break
                except (ValueError, KeyboardInterrupt):
                    print("❌ 취소되었습니다.")
                    exit(1)
    input_basename = os.path.basename(log_file).replace('.log', '')
    # If user passed a .csv path to --log-file, treat it as CSV
    if log_file.lower().endswith('.csv'):
        use_csv = True
        csv_file = log_file
        input_basename = os.path.basename(csv_file).replace('.csv', '')
        print(f"✅ .csv 파일이 감지되어 CSV 입력으로 사용합니다: {os.path.basename(csv_file)}")

# ============================================================
# Parse Log File
# ============================================================
def parse_log_file(log_file):
    """Parse log file to extract waypoint predictions"""
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    waypoints = []
    current_wp_pred = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if '[DEBUG] wp_pred raw:' in line:
            wp_coords = []
            match = re.search(r'\[\[\[\s*([-]?\d+\.?\d*(?:e[-+]?\d+)?)\s+([-]?\d+\.?\d*(?:e[-+]?\d+)?)\s*\]', line)
            if match:
                wp_coords.append((float(match.group(1)), float(match.group(2))))
            
            for j in range(1, 4):
                if i + j < len(lines):
                    mid_line = lines[i + j].strip()
                    match = re.search(r'\[\s*([-]?\d+\.?\d*(?:e[-+]?\d+)?)\s+([-]?\d+\.?\d*(?:e[-+]?\d+)?)\s*\]', mid_line)
                    if match:
                        wp_coords.append((float(match.group(1)), float(match.group(2))))
            
            if i + 4 < len(lines):
                last_line = lines[i + 4].strip()
                match = re.search(r'\[\s*([-]?\d+\.?\d*(?:e[-+]?\d+)?)\s+([-]?\d+\.?\d*(?:e[-+]?\d+)?)\s*\]\]\]', last_line)
                if match:
                    wp_coords.append((float(match.group(1)), float(match.group(2))))
            
            if len(wp_coords) == 5:
                current_wp_pred = wp_coords
        
        if line.startswith('[Frame '):
            frame_match = re.search(r'\[Frame\s+(\d+)\]', line)
            
            if frame_match and current_wp_pred:
                frame_idx = int(frame_match.group(1))
                
                dt_executed = None
                for search_offset in range(3):
                    if i + search_offset < len(lines):
                        search_line = lines[i + search_offset]
                        dt_match = re.search(r'dt_executed[=:]\s*([\d.]+)', search_line)
                        if dt_match:
                            dt_executed = float(dt_match.group(1))
                            break
                        dt_match = re.search(r'\bdt[=:]\s*([\d.]+)', search_line)
                        if dt_match:
                            dt_executed = float(dt_match.group(1))
                            break
                
                if dt_executed is None:
                    dt_executed = args.dt_executed if args.dt_executed else 0.1
                
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    dthetas_match = re.search(r'dthetas=\[([-\d.,\s]+)\]', next_line)
                    
                    if dthetas_match:
                        dthetas = [float(dt.strip()) for dt in dthetas_match.group(1).split(',')]
                        
                        if len(dthetas) == 5 and len(current_wp_pred) == 5:
                            arrive_match = re.search(r'arrive=(\d+)', next_line)
                            arrive = int(arrive_match.group(1)) if arrive_match else 0
                            
                            dx_cam, dz_cam = current_wp_pred[waypoint_index]
                            dtheta_y_cam = dthetas[waypoint_index]
                            
                            waypoints.append({
                                'frame_idx': frame_idx,
                                'dx_cam': dx_cam,
                                'dz_cam': dz_cam,
                                'dtheta_y_cam': dtheta_y_cam,
                                'arrive': arrive,
                                'dt_executed': dt_executed,
                                'dt_to_waypoint': DT_TO_WAYPOINT[waypoint_index]
                            })
                
                current_wp_pred = None
        i += 1
    
    return waypoints

# ============================================================
# Load waypoints from parsed CSV (same format as run_infer_online_panorama output)
# ============================================================
def load_waypoints_from_csv(csv_path, waypoint_index, dt_executed_default=0.2, dt_to_waypoint_map=None):
    """
    Load waypoints from CSV with columns: frame_idx, subframe_idx, dx, dy, dtheta, arrive, infer_time_s.
    Returns list of dicts in same shape as parse_log_file() for downstream processing.
    CSV dx, dy are already in meters (scaled); set already_scaled=True so compute_base_control uses scale=1.
    """
    if dt_to_waypoint_map is None:
        dt_to_waypoint_map = DT_TO_WAYPOINT
    waypoints = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subframe_idx = int(row.get('subframe_idx', 0))
            if subframe_idx != waypoint_index:
                continue
            frame_idx = int(row['frame_idx'])
            dx = float(row['dx'])   # lateral (camera: right +)
            dy = float(row['dy'])   # forward (camera: ahead +)
            dtheta_deg = float(row['dtheta'])
            arrive = int(row.get('arrive', 0))
            try:
                val = row.get('infer_time_s', '') or ''
                dt_exec = float(val) if (val and str(val).strip()) else dt_executed_default
            except (ValueError, TypeError):
                dt_exec = dt_executed_default
            waypoints.append({
                'frame_idx': frame_idx,
                'dx_cam': dx,
                'dz_cam': dy,
                'dtheta_y_cam': dtheta_deg,
                'arrive': arrive,
                'dt_executed': dt_exec,
                'dt_to_waypoint': dt_to_waypoint_map[waypoint_index],
                'already_scaled': True,
            })
    return waypoints

# ============================================================
# Base Frame Control Input Calculation
# ============================================================
def compute_base_control(wp, scale):
    """
    Waypoint에서 base 프레임 제어 입력 (v_x, omega_z) 계산
    
    Unicycle model:
      - v_x: forward linear velocity (m/s)
      - omega_z: angular velocity around z-axis (rad/s)
      - v_y = 0 (nonholonomic constraint)
    
    Returns:
        v_x: forward velocity (m/s)
        omega_z: angular velocity (rad/s)
        control_info: dict with detailed info
    """
    # CSV input is already in meters (already_scaled); log input is raw, multiply by scale
    s = 1.0 if wp.get('already_scaled') else scale
    # Camera → Base 변환
    dx_base, dy_base, dtheta_base = cam_waypoint_to_base(
        wp['dx_cam'] * s,
        wp['dz_cam'] * s,
        wp['dtheta_y_cam']
    )
    
    dt_to_wp = wp['dt_to_waypoint']
    dt_exec = wp['dt_executed']
    
    # Waypoint까지의 forward distance (nonholonomic이므로 lateral은 무시하거나 arc로 처리)
    # 방법 1: 직선 거리 사용 (간단)
    # 방법 2: Forward component만 사용 (더 정확한 unicycle model)
    
    # 여기서는 방법 2 사용: forward distance = dx_base
    # lateral distance (dy_base)는 회전으로 보정된다고 가정
    forward_distance = dx_base  # base frame에서 전진 거리
    
    # Required velocities
    if dt_to_wp > 1e-6:
        v_x = forward_distance / dt_to_wp  # m/s
        omega_z = dtheta_base / dt_to_wp   # rad/s
    else:
        v_x = 0.0
        omega_z = 0.0
    
    control_info = {
        'v_x': v_x,
        'omega_z': omega_z,
        'omega_z_deg': np.degrees(omega_z),
        'dx_base_target': dx_base,
        'dy_base_target': dy_base,
        'dtheta_base_target': dtheta_base,
        'dt_to_waypoint': dt_to_wp,
        'dt_executed': dt_exec,
    }
    
    return v_x, omega_z, control_info

def integrate_unicycle(x, y, theta, v_x, omega_z, dt):
    """
    Unicycle model integration (Euler method)
    
    State: (x, y, theta) in world frame
    Control: (v_x, omega_z) in body frame
    
    dx/dt = v_x * cos(theta)
    dy/dt = v_x * sin(theta)
    dtheta/dt = omega_z
    """
    # 더 정확한 적분: 회전 중심점 기준
    if abs(omega_z) > 1e-6:
        # Arc motion
        r = v_x / omega_z  # turning radius
        dtheta = omega_z * dt
        dx = r * (np.sin(theta + dtheta) - np.sin(theta))
        dy = r * (-np.cos(theta + dtheta) + np.cos(theta))
        new_theta = theta + dtheta
    else:
        # Straight line motion
        dx = v_x * np.cos(theta) * dt
        dy = v_x * np.sin(theta) * dt
        new_theta = theta
    
    return x + dx, y + dy, new_theta

# ============================================================
# Main Processing
# ============================================================
if use_csv:
    dt_exec_default = args.dt_executed if args.dt_executed is not None else 0.2
    print(f"Loading CSV: {csv_file}")
    waypoints = load_waypoints_from_csv(csv_file, waypoint_index, dt_executed_default=dt_exec_default)
    print(f"Loaded {len(waypoints)} waypoints (subframe_idx={waypoint_index})")
else:
    print(f"Parsing log file: {log_file}")
    waypoints = parse_log_file(log_file)
    print(f"Extracted {len(waypoints)} waypoints")

if len(waypoints) == 0:
    print("❌ waypoint가 0개입니다. 로그 형식이 맞는지 확인하거나, CSV라면 --csv-file 또는 --log-file 에 .csv 경로를 주세요.")
    exit(1)

planning_freq_hz = 1.0 / dt_multiplier if dt_multiplier > 1e-6 else 0.0

if waypoints:
    dt_executed_used = waypoints[0]['dt_executed']
    dt_executed_avg = np.mean([wp['dt_executed'] for wp in waypoints])
else:
    dt_executed_used = args.dt_executed if args.dt_executed else 0.1
    dt_executed_avg = dt_executed_used

print(f"Using waypoint index {waypoint_index} (dt_to_waypoint = {DT_TO_WAYPOINT[waypoint_index]:.3f}s)")
print(f"Planning frequency: {planning_freq_hz:.2f} Hz")
print(f"Execution dt: {dt_executed_used:.3f}s (avg: {dt_executed_avg:.3f}s)")

planning_freq_str = f'_{planning_freq_hz:.2f}Hz' if dt_multiplier != 1.0 else ''
dt_exec_str = f'_exec{dt_executed_used:.3f}s' if args.dt_executed or dt_executed_used != 0.1 else ''
output_file = f'{input_basename}_trajectory_basecontrol_scale{PREDICT_SCALE}_wp{waypoint_index}{planning_freq_str}{dt_exec_str}.png'

# Global pose (starts at origin)
x, y, theta = 0.0, 0.0, 0.0
trajectory = [(x, y, theta)]
arrived_points = []
control_history = []

for i, wp in enumerate(waypoints):
    # Base 프레임에서 직접 제어 입력 계산
    v_x, omega_z, ctrl_info = compute_base_control(wp, PREDICT_SCALE)
    control_history.append(ctrl_info)
    
    dt_exec = wp['dt_executed']
    
    # Unicycle model로 적분
    x, y, theta = integrate_unicycle(x, y, theta, v_x, omega_z, dt_exec)
    trajectory.append((x, y, theta))
    
    if wp.get('arrive', 0) == 1:
        arrived_points.append(len(trajectory) - 1)
    
    # Debug first few steps
    if i < 5:
        print(f"\nFrame {wp['frame_idx']}:")
        print(f"  Camera waypoint: dx={wp['dx_cam']:.4f}, dz={wp['dz_cam']:.4f}, "
              f"dtheta={wp['dtheta_y_cam']:.2f}°")
        print(f"  Base target: dx={ctrl_info['dx_base_target']:.4f}, "
              f"dy={ctrl_info['dy_base_target']:.4f}, "
              f"dtheta={np.degrees(ctrl_info['dtheta_base_target']):.2f}°")
        print(f"  Control input: v_x={v_x:.3f} m/s, omega_z={ctrl_info['omega_z_deg']:.2f} °/s")
        print(f"  dt_executed={dt_exec:.3f}s")
        print(f"  → New pose: ({x:.4f}, {y:.4f}), heading={np.degrees(theta):.2f}°")

# ============================================================
# Plotting
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

# --- Left plot: Trajectory ---
ax = axes[0]

# Display: plot_x = y_base (left), plot_y = x_base (forward)
plot_x = [p[1] for p in trajectory]  # y_base
plot_y = [p[0] for p in trajectory]  # x_base
n_points = len(trajectory)

x_range = max(plot_x) - min(plot_x) if plot_x else 1
y_range = max(plot_y) - min(plot_y) if plot_y else 1
traj_size = max(0.1, np.sqrt(x_range**2 + y_range**2))
arrow_scale = max(0.1, traj_size * 0.03)

cmap = cm.jet
norm = mcolors.Normalize(vmin=0, vmax=max(1, n_points - 1))

for i in range(n_points - 1):
    ax.plot(plot_x[i:i+2], plot_y[i:i+2], color=cmap(norm(i)), linewidth=2)

for i in range(len(trajectory)):
    theta_i = trajectory[i][2]
    dx_arrow = arrow_scale * np.sin(theta_i)
    dy_arrow = arrow_scale * np.cos(theta_i)
    
    ax.arrow(plot_x[i], plot_y[i], dx_arrow, dy_arrow,
             head_width=arrow_scale * 0.3, head_length=arrow_scale * 0.2,
             fc=cmap(norm(i)), ec=cmap(norm(i)), alpha=0.8)

ax.scatter(0, 0, color='green', s=200, marker='*', label='Start', zorder=5, edgecolors='black')

if arrived_points:
    ax.scatter([plot_x[i] for i in arrived_points], [plot_y[i] for i in arrived_points],
               color='black', s=150, marker='o', label=f'Arrived ({len(arrived_points)})', zorder=6)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Time Step')

input_label = os.path.basename(csv_file) if use_csv else os.path.basename(log_file)
ax.set_title(f'Trajectory with Base Frame Control (v_x, ω_z)\n'
             f'Input: {input_label} | Scale: {PREDICT_SCALE} | Waypoint: {waypoint_index}\n'
             f'Planning: {planning_freq_hz:.2f}Hz | Exec dt: {dt_executed_used:.3f}s')
ax.set_xlabel('Y_base (left +)')
ax.set_ylabel('X_base (forward +)')
ax.invert_xaxis()
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

ind_scale = max(0.2, traj_size * 0.05)
ax.annotate('', xy=(ind_scale, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.annotate('', xy=(0, ind_scale), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(ind_scale * 1.2, 0, '+Y', fontsize=10, ha='left', va='center', color='blue')
ax.text(0, ind_scale * 1.2, '+X', fontsize=10, ha='center', va='bottom', color='red')

# --- Right plot: Control input profile ---
ax2 = axes[1]

frames = [wp['frame_idx'] for wp in waypoints]
v_x_history = [c['v_x'] for c in control_history]
omega_z_history = [c['omega_z_deg'] for c in control_history]

ax2_twin = ax2.twinx()

l1, = ax2.plot(frames, v_x_history, 'b-', linewidth=1.5, label='v_x (m/s)')
l2, = ax2_twin.plot(frames, omega_z_history, 'r-', linewidth=1.5, label='ω_z (°/s)')

ax2.set_xlabel('Frame Index')
ax2.set_ylabel('Linear Velocity v_x (m/s)', color='blue')
ax2_twin.set_ylabel('Angular Velocity ω_z (°/s)', color='red')
ax2.set_title('Base Frame Control Inputs Over Time')
ax2.grid(True, alpha=0.3)

lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='upper right')

ax2.tick_params(axis='y', labelcolor='blue')
ax2_twin.tick_params(axis='y', labelcolor='red')

plt.tight_layout()
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: {output_file}")

# ============================================================
# Summary Statistics
# ============================================================
total_dist = sum(np.sqrt((plot_x[i+1]-plot_x[i])**2 + (plot_y[i+1]-plot_y[i])**2) 
                 for i in range(len(plot_x)-1))
v_x_history = [c['v_x'] for c in control_history]
omega_z_history = [c['omega_z_deg'] for c in control_history]
avg_v_x = np.mean(v_x_history) if v_x_history else 0.0
avg_omega_z = np.mean(omega_z_history) if omega_z_history else 0.0

print(f"\n📊 Summary:")
print(f"  - Frames: {len(waypoints)}")
print(f"  - Total distance traveled: {total_dist:.2f} m")
print(f"  - Final position: ({trajectory[-1][0]:.2f}, {trajectory[-1][1]:.2f}) m")
print(f"  - Final heading: {np.degrees(trajectory[-1][2]):.1f}°")
print(f"\n🎮 Control Input Statistics:")
print(f"  - Avg v_x: {avg_v_x:.3f} m/s")
print(f"  - Avg ω_z: {avg_omega_z:.2f} °/s")
if v_x_history and omega_z_history:
    print(f"  - Max v_x: {max(v_x_history):.3f} m/s")
    print(f"  - Max |ω_z|: {max(abs(w) for w in omega_z_history):.2f} °/s")