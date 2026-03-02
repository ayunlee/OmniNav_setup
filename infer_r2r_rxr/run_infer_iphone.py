#!/usr/bin/env python3
"""
OmniNav inference using iPhone data
Based on evaluate_agent from infer_r2r_rxr,
but replaces simulator image input with iPhone data reading
"""
# HPC-X/UCC library conflict prevention (NGC container issue)
# Error: libucc.so.1: undefined symbol: ucs_config_doc_nop
# LD_PRELOAD must be set before process starts, so auto re-exec
import os
import sys

_LD_PRELOAD_LIBS = "/opt/hpcx/ucx/lib/libucs.so.0:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucm.so.0"
_REEXEC_VAR = "_OMNINAV_REEXEC"

if os.environ.get(_REEXEC_VAR) != "1" and os.path.exists("/opt/hpcx/ucx/lib/libucs.so.0"):
    # Re-execute self with correct LD_PRELOAD
    os.environ["LD_PRELOAD"] = _LD_PRELOAD_LIBS
    os.environ[_REEXEC_VAR] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import argparse
import torch
import os
import csv
import json
import sys
import time
from pathlib import Path
from tqdm import trange
import cv2
from PIL import Image
from datetime import datetime

from agent.waypoint_agent import Waypoint_Agent

# Map size and settings
MAP_SIZE = 1024  # Map resolution
MAP_METERS_PER_PIXEL = 0.05  # 1 pixel = 0.05m
MAP_CENTER = (MAP_SIZE // 2, MAP_SIZE // 2)  # Map center


def load_instruction(instruction_path):
    """Read instruction.txt"""
    with open(instruction_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_odometry(odom_path):
    """Read odometry.csv and convert pose
    
    Camera coordinate system (iPhone): X=right, Y=down, Z=forward
    Habitat/Robot coordinate system: X=forward, Y=left, Z=up
    
    Conversion: Camera Z-axis is forward, so use position as-is
    """
    poses = []
    with open(odom_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # timestamp, frame, x, y, z, qx, qy, qz, qw
        
        for row in reader:
            try:
                frame = int(row[1])
                x_cam = float(row[2])
                y_cam = float(row[3])
                z_cam = float(row[4])
                qx = float(row[5])
                qy = float(row[6])
                qz = float(row[7])
                qw = float(row[8])
                
                # Convert camera coordinate system to Habitat coordinate system
                # Camera: (x, y, z) = (right, down, forward)
                # Habitat: (x, y, z) = (forward, left, up)
                # Camera Z is forward -> Habitat X
                # Camera X is right -> Habitat -Y
                # Camera Y is down -> Habitat Z
                
                # Use position as-is for now (modify later if coordinate conversion needed)
                position = [x_cam, y_cam, z_cam]
                rotation = [qw, qx, qy, qz]  # Habitat uses (w, x, y, z) order
                
                poses.append({
                    'frame': frame,
                    'position': position,
                    'rotation': rotation,
                    'raw': (x_cam, y_cam, z_cam, qx, qy, qz, qw)
                })
            except (ValueError, IndexError) as e:
                print(f"[WARN] Skipping invalid row: {row}, error: {e}")
                continue
    
    return poses


def get_rgb_images(rgb_dir):
    """Get RGB image list (sorted)"""
    rgb_path = Path(rgb_dir)
    image_files = sorted(rgb_path.glob("*.png"))
    return [str(p) for p in image_files]


def load_rgb_image(image_path):
    """Load RGB image (BGR -> RGB conversion)"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def create_fake_observations(frame_idx, image_path, instruction, pose):
    """Create dictionary mimicking simulator observations
    
    iPhone data only has front image, so left/right are copied from front
    """
    rgb = load_rgb_image(image_path)
    
    # iPhone data only has front, left/right are copied from front
    # (Actually model only uses left/right in 'special_token' mode)
    observations = {
        'front': rgb,
        'left': rgb.copy(),  # Copy from front for now
        'right': rgb.copy(),  # Copy from front for now
        'rgb': rgb,
        'instruction': {'text': instruction},
        'pose': {
            'position': pose['position'],
            'rotation': pose['rotation']
        }
    }
    
    return observations


def world_to_map_coords(world_pos, map_center, meters_per_pixel):
    """Convert world coordinates to map pixel coordinates
    
    Args:
        world_pos: (x, y) world coordinates (robot frame: x=left-right, y=forward)
        map_center: (cx, cy) map center pixel coordinates
        meters_per_pixel: meters per pixel
    """
    x_world, y_world = world_pos
    # Robot frame: x=left-right(left=-, right=+), y=forward(+)
    # Map frame: x=column(left=0), y=row(top=0)
    # World x increase -> map x increase, world y increase -> map y decrease (top-down view)
    map_x = map_center[0] + int(x_world / meters_per_pixel)
    map_y = map_center[1] - int(y_world / meters_per_pixel)  # Flip y-axis
    
    return (map_x, map_y)


def create_topdown_map_gt_only(odom_poses, current_idx, map_center, meters_per_pixel):
    """Create map image with GT path only on white background
    
    Args:
        odom_poses: Odometry poses list
        current_idx: Current frame index
        map_center: Map center pixel coordinates (cx, cy)
        meters_per_pixel: Meters per pixel
    """
    map_img = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    
    # Draw GT path (path from odometry up to current frame)
    if current_idx >= 0 and len(odom_poses) > 0:
        gt_path_map_coords = []
        for i in range(min(current_idx + 1, len(odom_poses))):
            pose = odom_poses[i]
            world_x = pose['position'][0]
            world_y = pose['position'][2]
            
            if i == 0:
                origin = (world_x, world_y)
            
            rel_x = world_x - origin[0]
            rel_y = world_y - origin[1]
            
            map_coord = world_to_map_coords((rel_x, rel_y), map_center, meters_per_pixel)
            gt_path_map_coords.append(map_coord)
        
        # Draw GT path line (black)
        for i, coord in enumerate(gt_path_map_coords):
            x, y = coord
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                map_img[y, x] = 1  # MAP_VALID_POINT
                cv2.circle(map_img, (x, y), 3, 10, -1)  # 10 = MAP_REFERENCE_POINT (black)
                
                if i > 0:
                    prev_coord = gt_path_map_coords[i-1]
                    cv2.line(map_img, prev_coord, coord, 10, 2)  # black
    
    return map_img


def create_topdown_map_pred_only(pred_path, map_center, meters_per_pixel):
    """Create map image with prediction path only on white background
    
    Args:
        pred_path: Prediction path coordinate list [(x, y), ...]
        map_center: Map center pixel coordinates (cx, cy)
        meters_per_pixel: Meters per pixel
    """
    map_img = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    
    # Draw prediction path
    if len(pred_path) > 0:
        pred_path_map_coords = []
        for pos in pred_path:
            map_coord = world_to_map_coords(pos, map_center, meters_per_pixel)
            pred_path_map_coords.append(map_coord)
        
        # Draw prediction path line (yellow)
        for i, coord in enumerate(pred_path_map_coords):
            x, y = coord
            if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
                map_img[y, x] = 1  # MAP_VALID_POINT
                cv2.circle(map_img, (x, y), 3, 12, -1)  # 12 = MAP_WAYPOINT_PREDICTION (yellow)
                
                if i > 0:
                    prev_coord = pred_path_map_coords[i-1]
                    cv2.line(map_img, prev_coord, coord, 12, 2)  # yellow
    
    return map_img


def create_topdown_map_with_path(odom_poses, current_idx, pred_path=[]):
    """Create map image with GT and prediction paths on white background (legacy, no longer used)
    
    New method: Draw GT and prediction separately and stack vertically
    """
    # This function is no longer used but kept for compatibility
    # Use create_fake_info instead
    info = create_fake_info(odom_poses, current_idx, pred_path)
    return info['gt_map']


def create_fake_info(odom_poses, current_idx, pred_path=[]):
    """Create dictionary mimicking simulator info
    
    iPhone data has no topdown map, so create map with paths on white background
    Generate GT and prediction paths separately with matched scale
    """
    # 1. Collect all coordinates from GT and prediction paths
    all_coords_x = []
    all_coords_y = []
    
    # Collect GT path coordinates
    if current_idx >= 0 and len(odom_poses) > 0:
        origin = None
        for i in range(min(current_idx + 1, len(odom_poses))):
            pose = odom_poses[i]
            world_x = pose['position'][0]
            world_y = pose['position'][2]
            
            if i == 0:
                origin = (world_x, world_y)
            
            rel_x = world_x - origin[0]
            rel_y = world_y - origin[1]
            all_coords_x.append(rel_x)
            all_coords_y.append(rel_y)
    
    # Collect prediction path coordinates
    if len(pred_path) > 0:
        for pos in pred_path:
            all_coords_x.append(pos[0])
            all_coords_y.append(pos[1])
    
    # 2. Calculate coordinate range (with margin)
    if len(all_coords_x) > 0 and len(all_coords_y) > 0:
        min_x, max_x = min(all_coords_x), max(all_coords_x)
        min_y, max_y = min(all_coords_y), max(all_coords_y)
        
        # Add margin (so path doesn't touch map edge)
        margin = 2.0  # in meters
        range_x = max(max_x - min_x, 0.1) + 2 * margin
        range_y = max(max_y - min_y, 0.1) + 2 * margin
        
        # Calculate scale to fit map range
        map_range_meters = min(range_x, range_y)
        if map_range_meters > 0:
            # Adjust scale to use 80% of map size
            available_pixels = MAP_SIZE * 0.8
            meters_per_pixel = map_range_meters / available_pixels
        else:
            meters_per_pixel = MAP_METERS_PER_PIXEL
        
        # Calculate map center (GT path start point at map center)
        if current_idx >= 0 and len(odom_poses) > 0:
            center_world_x = 0.0  # Based on GT origin
            center_world_y = 0.0
        else:
            # Only prediction path exists
            center_world_x = (min_x + max_x) / 2
            center_world_y = (min_y + max_y) / 2
        
        map_center = (MAP_SIZE // 2, MAP_SIZE // 2)
    else:
        # No path, use default values
        map_center = MAP_CENTER
        meters_per_pixel = MAP_METERS_PER_PIXEL
    
    # 3. Generate GT and prediction maps with same scale
    gt_map = create_topdown_map_gt_only(odom_poses, current_idx, map_center, meters_per_pixel)
    pred_map = create_topdown_map_pred_only(pred_path, map_center, meters_per_pixel)
    
    # Stack two maps vertically (GT on top, prediction on bottom)
    if gt_map.shape == pred_map.shape:
        combined_map = np.concatenate([gt_map, pred_map], axis=0)
    else:
        # Resize if shapes differ
        target_w = max(gt_map.shape[1], pred_map.shape[1])
        gt_map_resized = cv2.resize(gt_map, (target_w, gt_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred_map_resized = cv2.resize(pred_map, (target_w, pred_map.shape[0]), interpolation=cv2.INTER_NEAREST)
        combined_map = np.concatenate([gt_map_resized, pred_map_resized], axis=0)
    
    return {
        'top_down_map_vlnce': combined_map,
        'gt_map': gt_map,  # GT map stored separately
        'pred_map': pred_map,  # Prediction map stored separately
    }


def evaluate_iphone_data(data_dir, model_path, result_path, max_frames=0):
    """Run inference with iPhone data
    
    Args:
        data_dir: iPhone data directory (contains instruction.txt, odometry.csv, rgb/)
        model_path: OmniNav model path
        result_path: Result save path
        max_frames: Maximum number of frames (0=all)
    """
    data_dir = Path(data_dir)
    
    # 1. Load instruction
    instruction_path = data_dir / 'instruction.txt'
    instruction = load_instruction(instruction_path)
    # print(f"[INFO] Instruction: {instruction}")
    
    # 2. Load odometry
    odom_path = data_dir / 'odometry.csv'
    odom_poses = load_odometry(odom_path)
    print(f"[INFO] Loaded {len(odom_poses)} odometry poses")
    
    # 3. Get RGB image list
    rgb_dir = data_dir / 'rgb'
    image_files = get_rgb_images(rgb_dir)
    print(f"[INFO] Found {len(image_files)} RGB images")
    
    if max_frames > 0:
        image_files = image_files[:max_frames]
        odom_poses = odom_poses[:max_frames]
        print(f"[INFO] Limited to {len(image_files)} frames")
    
    # 4. Initialize agent
    model_name = '/'.join(model_path.split('/')[-3:])
    result_path_full = os.path.join(result_path, model_name)
    
    # Log file setup (in result folder)
    log_dir = os.path.join(result_path_full, "log")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"inference_{timestamp}.log")
    
    # Setup stdout/stderr to output to both log file and console
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_f = open(log_file, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)
    
    print(f"[INFO] All output will be saved to log file")
    
    # Set require_map=True for map_vis generation
    agent = Waypoint_Agent(model_path, result_path_full, require_map=True)
    
    # 5. Episode ID (currently treating as single episode)
    episode_id = data_dir.name  # e.g., 'a1af0cece0'
    
    agent.reset()
    agent.episode_id = episode_id
    
    # 6. Run inference per frame
    num_frames = len(image_files)
    
    csv_records = []  # CSV output records (frame_idx, subframe_idx, dx, dy, dtheta, arrive, infer_time)
    pred_path = []  # Accumulated prediction path
    curr_pos = np.array([0.0, 0.0])  # Current position (robot frame)
    total_infer_time = 0.0  # Total inference time
    
    for i in trange(num_frames, desc=f"Processing {episode_id}"):
        image_path = image_files[i]
        frame_idx = int(Path(image_path).stem)
        
        # Find pose for this frame from odometry
        pose = None
        for p in odom_poses:
            if p['frame'] == frame_idx:
                pose = p
                break
        
        if pose is None:
            print(f"[WARN] No pose found for frame {frame_idx}, using default")
            pose = {
                'position': [0.0, 0.0, 0.0],
                'rotation': [1.0, 0.0, 0.0, 0.0]
            }
        
        # Create observations
        obs = create_fake_observations(frame_idx, image_path, instruction, pose)
        
        # Create info (draw path on topdown map, use prediction path up to previous frame)
        # Current frame's prediction is not known yet, so only show path up to previous frame
        info = create_fake_info(odom_poses, i, pred_path)
        
        # Agent act (inference) - pass info for map visualization
        start_time = time.time()
        with torch.no_grad():
            action = agent.act(obs, info, episode_id)
        infer_time = time.time() - start_time
        total_infer_time += infer_time
        
        # Print inference time per frame
        print(f"[Frame {frame_idx:06d}] Inference time: {infer_time:.3f}s")
        
        # Accumulate prediction path (extract waypoint from action for next frame)
        if 'action' in action and len(action['action']) > 0:
            waypoint = action['action'][0]  # First waypoint
            # Waypoint is in local coordinates (x, y)
            curr_pos = curr_pos + np.array([waypoint[0], waypoint[1]]) * 0.3  # PREDICT_SCALE = 0.3
            pred_path.append(curr_pos.copy())
        
        # Save CSV records (all 5 waypoints)
        # Note: action['action'] already has PREDICT_SCALE=0.3 applied
        # Restore to original scale for saving (same format as log_to_csv.py)
        PREDICT_SCALE = 0.3
        
        if 'arrive_pred' in action and 'action' in action and 'recover_angle' in action:
            arrive = int(action['arrive_pred'])
            waypoints = action['action']  # shape (5, 2), scale already applied
            recover_angles = action['recover_angle']  # shape (5,)
            
            # Flatten if needed
            if isinstance(waypoints, np.ndarray) and waypoints.ndim > 1:
                waypoints = waypoints.reshape(-1, 2)
            if isinstance(recover_angles, np.ndarray) and recover_angles.ndim > 1:
                recover_angles = recover_angles.flatten()
            
            # Save each of 5 waypoints as CSV record
            for subframe_idx in range(min(5, len(waypoints))):
                # Restore to original scale (/ PREDICT_SCALE)
                dx = waypoints[subframe_idx][0] / PREDICT_SCALE
                dy = waypoints[subframe_idx][1] / PREDICT_SCALE
                dtheta = np.degrees(recover_angles[subframe_idx]) if subframe_idx < len(recover_angles) else 0.0
                
                csv_records.append({
                    'frame_idx': frame_idx,
                    'subframe_idx': subframe_idx,
                    'dx': float(dx),
                    'dy': float(dy),
                    'dtheta': float(dtheta),
                    'arrive': arrive,
                    'infer_time_s': float(infer_time) if subframe_idx == 0 else 0.0  # Record once per frame
                })
            
            # Simple log output (first waypoint only)
            wp = waypoints[0]
            dtheta0 = np.degrees(recover_angles[0]) if len(recover_angles) > 0 else 0.0
            print(f"  -> arrive={arrive}, dtheta={dtheta0:.2f}, wp[0]=({wp[0]:.4f}, {wp[1]:.4f})")
    
    # 7. Agent reset (save GIF)
    agent.reset()
    
    # 8. Print inference time statistics
    avg_infer_time = total_infer_time / num_frames if num_frames > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"[TIMING] Total inference time: {total_infer_time:.3f}s")
    print(f"[TIMING] Average inference time per frame: {avg_infer_time:.3f}s")
    print(f"[TIMING] FPS: {1.0/avg_infer_time:.2f}" if avg_infer_time > 0 else "[TIMING] FPS: N/A")
    print(f"{'='*60}\n")
    
    # 9. Save CSV results (waypoint data)
    csv_file = os.path.join(log_dir, f"waypoint_data_{episode_id}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame_idx', 'subframe_idx', 'dx', 'dy', 'dtheta', 'arrive', 'infer_time_s'])
        writer.writeheader()
        writer.writerows(csv_records)
    
    print(f"[INFO] CSV saved: {csv_file} ({len(csv_records)} records, {len(csv_records)//5} frames)")
    
    # 10. Save statistics (JSON)
    stats_file = os.path.join(log_dir, f"stats_{episode_id}.json")
    stats = {
        'id': episode_id,
        'num_frames': num_frames,
        'instruction': instruction,
        'csv_file': csv_file
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"[INFO] Stats saved: {stats_file}")
    print(f"[INFO] Visualization saved to: {result_path_full}")
    print(f"[INFO] Log file: {log_file}")
    
    # Restore stdout/stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_f.close()
    
    return csv_records


def main():
    parser = argparse.ArgumentParser(description='OmniNav inference with iPhone data')
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="iPhone data directory (contains instruction.txt, odometry.csv, rgb/)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="OmniNav model path"
    )
    
    parser.add_argument(
        "--result-path",
        type=str,
        required=False,
        default="./data/result_iphone",
        help="Result save path"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        required=False,
        default=0,
        help="Maximum number of frames (0=all)"
    )
    
    args = parser.parse_args()
    
    evaluate_iphone_data(
        args.data_dir,
        args.model_path,
        args.result_path,
        args.max_frames
    )


if __name__ == "__main__":
    main()
