#!/usr/bin/env python3
"""
OmniNav Real-time Inference with Front/Left/Right Cameras
Subscribes to /cam_front/color/image_raw, /cam_left/color/image_raw, /cam_right/color/image_raw,
runs inference, publishes waypoints to /action.
"""
# HPC-X/UCC library conflict prevention (NGC container issue)
import os
import sys

_LD_PRELOAD_LIBS = "/opt/hpcx/ucx/lib/libucs.so.0:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucm.so.0"
_REEXEC_VAR = "_OMNINAV_REEXEC"

if os.environ.get(_REEXEC_VAR) != "1" and os.path.exists("/opt/hpcx/ucx/lib/libucs.so.0"):
    os.environ["LD_PRELOAD"] = _LD_PRELOAD_LIBS
    os.environ[_REEXEC_VAR] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import argparse
from typing import Optional
import torch
import json
import csv
import time
import threading
import cv2
from datetime import datetime
import math

import matplotlib.cm as mpl_cm

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from agent.waypoint_agent import Waypoint_Agent


def load_rgb_image_from_array(img_bgr: np.ndarray) -> np.ndarray:
    """Load RGB image from BGR array (identical to run_infer_iphone.py's load_rgb_image)
    
    Args:
        img_bgr: BGR image array from cv2.imdecode()
        
    Returns:
        RGB image array
    """
    if img_bgr is None:
        raise ValueError("Failed to decode image: img_bgr is None")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def draw_waypoint_arrows_fpv(
    img: np.ndarray,
    waypoints: list,
    arrow_thickness: int = 2,
    tipLength: float = 0.45,
    stop_color: tuple = (0, 0, 255),   # BGR red for STOP
    stop_radius: int = 8,
    arrow_scale: float = 0.15,   # arrow length in body-frame meters
    vis_scale: float = 120.0,    # meters -> pixel
    arrow_gap: int = 1.2,        # pixel gap between stacked arrows (0 = tight)
) -> np.ndarray:
    """
    Draw each waypoint as an arrow stacked vertically (UniNaVid-style layout).
    - Explicit Coordinate Transform applied: Network -> Robot Body Frame
    - 1st waypoint: center bottom (base)
    - Stacked upwards sequentially
    """
    out = img.copy()
    h, w = out.shape[:2]
    base_x, base_y = w // 2, int(h * 0.95)  # origin at bottom center

    # Slot height 계산
    slot_height = max(1, int(vis_scale * arrow_scale) + arrow_gap)

    # Color map 설정
    try:
        cmap = mpl_cm.get_cmap('turbo')
    except Exception:
        cmap = mpl_cm.get_cmap('viridis')

    n_wp = len(waypoints)
    for i, wp in enumerate(waypoints):
        # 0. Raw Network Output 가져오기
        dx_net = wp.get('dx', 0.0)
        dy_net = wp.get('dy', 0.0)
        arrive = wp.get('arrive', 0)

        # 화살표 시작점 계산 (위로 쌓아올림)
        start_y = int(base_y - i * slot_height)
        start_pixel = (base_x, start_y)

        # 도착(Arrive) 시 처리
        if arrive > 0:
            arrive_color = (0, 200, 0)  # Green
            cv2.circle(out, start_pixel, stop_radius, arrive_color, -1)
            cv2.circle(out, start_pixel, stop_radius, (255, 255, 255), 1)
            cv2.putText(out, "OK", (start_pixel[0] - 12, start_pixel[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            continue

        # -------------------------------------------------------------
        # 1. 좌표계 변환 (Network -> Robot Body ISO 8855)
        # -------------------------------------------------------------
        # Network: dx(Right+), dy(Forward+)
        # Robot:   x(Forward+), y(Left+)
        
        target_x = dy_net   # Net Forward -> Robot X (Forward)
        target_y = -dx_net  # Net Right   -> Robot -Y (Left)
        
        # -------------------------------------------------------------
        # 2. 로봇 바디 프레임 기준 각도 및 성분 계산
        # -------------------------------------------------------------
        # theta: 로봇 정면(X) 기준, 왼쪽(Y)으로 회전한 각도 (CCW +)
        theta = math.atan2(target_y, target_x) if (target_x != 0 or target_y != 0) else 0.0
        
        # 화살표 끝점 계산 (로봇 바디 프레임에서의 벡터 성분)
        # body_dx: 로봇 기준 전진 성분
        # body_dy: 로봇 기준 좌측 성분
        body_dx = arrow_scale * math.cos(theta)
        body_dy = arrow_scale * math.sin(theta)

        # -------------------------------------------------------------
        # 3. 이미지 픽셀 매핑 (Robot Body -> Image Pixel)
        # -------------------------------------------------------------
        # Image Up:    -y pixel (Robot Forward 매핑)
        # Image Left:  -x pixel (Robot Left 매핑)
        # Image Right: +x pixel (Robot Right 매핑)
        
        # start_x에서 왼쪽 성분(body_dy)만큼 빼면 왼쪽으로, 더하면 오른쪽으로 감 (dy가 음수면 오른쪽)
        head_x = int(start_pixel[0] - body_dy * vis_scale)
        
        # start_y에서 전진 성분(body_dx)만큼 빼면 위쪽으로 올라감
        head_y = int(start_pixel[1] - body_dx * vis_scale)
        
        head_pixel = (np.clip(head_x, 0, w - 1), np.clip(head_y, 0, h - 1))

        # 그리기 (스무스 그라데이션)
        t = (i + 0.5) / n_wp if n_wp > 0 else 0.5
        rgba = cmap(t)[:3]
        color = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))  # BGR

        if np.linalg.norm(np.array(head_pixel) - np.array(start_pixel)) > 2:
            cv2.arrowedLine(out, start_pixel, head_pixel, color, arrow_thickness,
                            tipLength=tipLength, line_type=cv2.LINE_AA)

    return out


def add_instruction_bar(img_rgb: np.ndarray, instruction: str, bar_height: int = 80) -> np.ndarray:
    """Append instruction text bar below image. Returns RGB image."""
    h, w = img_rgb.shape[:2]
    new_img = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
    new_img.fill(255)  # white background for bar
    new_img[:h, :w] = img_rgb

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    prefix = "Instruction: "
    full_text = prefix + (instruction or "")
    (_, line_h), _ = cv2.getTextSize("Ay", font, font_scale, thickness)
    margin_x, margin_y = 10, 10
    max_line_w = w - 2 * margin_x

    # Word wrap
    words = full_text.split()
    lines = []
    line = ""
    for word in words:
        test = (line + " " + word) if line else word
        (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if tw <= max_line_w:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)

    y = h + margin_y + line_h
    for i, ln in enumerate(lines[:4]):  # max 4 lines to fit in bar
        dy = y + i * (line_h + 4)
        if dy > h + bar_height - 5:
            break
        cv2.putText(new_img, ln, (margin_x, dy),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return new_img


class OmniNavOnlineInference:
    """Real-time OmniNav inference with Front/Left/Right cameras (ROS topics)."""

    def __init__(self, model_path: str, instruction: str, result_path: str = "./data/result_online",
                 output_csv: Optional[str] = None, output_video: Optional[str] = None):
        """
        Args:
            model_path: Path to OmniNav model
            instruction: Navigation instruction text
            result_path: Path to save results (used when output_csv/output_video are relative)
            output_csv: Optional CSV filename or full path. If None, use waypoint_data_online_<timestamp>.csv under result_path.
            output_video: Optional MP4 filename or full path. If None, use omninav_online_<timestamp>.mp4 under result_path.
        """
        self.result_path = result_path
        self.output_csv = output_csv
        self.output_video = output_video
        self.instruction = instruction

        # Image buffers for front/left/right from ROS
        self.latest_front = None
        self.latest_left = None
        self.latest_right = None
        self.image_lock = threading.Lock()
        self.image_timestamp = None
        
        # Frame counter
        self.frame_count = 0
        
        # Visualization storage
        self.vis_frame_list = []
        self.save_video = True
        
        # CSV records for waypoint data
        self.csv_records = []
        self.total_infer_time = 0.0
        
        # Initialize ROS2
        rclpy.init()
        self.ros_node = rclpy.create_node('omninav_inference')
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # /action: must match omninav_control subscriber exactly (all policies)
        qos_action = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Image subscribers (front, left, right)
        self.ros_node.create_subscription(
            Image,
            '/cam_front/color/image_raw',
            self._image_front_callback,
            qos_profile
        )
        self.ros_node.create_subscription(
            Image,
            '/cam_left/color/image_raw',
            self._image_left_callback,
            qos_profile
        )
        self.ros_node.create_subscription(
            Image,
            '/cam_right/color/image_raw',
            self._image_right_callback,
            qos_profile
        )

        # Action publisher (waypoints as JSON string)
        self.action_pub = self.ros_node.create_publisher(String, '/action', qos_action)
        
        print("=" * 60)
        print("[OmniNav Online] ROS2 Node Ready")
        print("[OmniNav Online] Subscribing to: /cam_front/color/image_raw, /cam_left/color/image_raw, /cam_right/color/image_raw")
        print("[OmniNav Online] Publishing to: /action")
        print(f"[OmniNav Online] Instruction: {self.instruction[:80]}...")
        print("=" * 60)
        
        # Initialize OmniNav agent
        # Use system temp directory for agent (it needs result_path but we don't use its outputs)
        # We only save video to result_path, not agent's internal folders
        import tempfile
        temp_agent_path = tempfile.mkdtemp(prefix="omninav_agent_")
        
        print("[OmniNav Online] Loading model...")
        self.agent = Waypoint_Agent(model_path, temp_agent_path, require_map=False)
        self.agent.reset()
        self.agent.episode_id = "online_session"
        print("[OmniNav Online] Model loaded successfully")
        
        # Start ROS2 spin thread
        self.spin_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.spin_thread.start()
    
    def _image_msg_to_rgb(self, msg: Image) -> np.ndarray:
        """Decode sensor_msgs/Image to RGB numpy array. Supports rgb8, bgr8, rgba8, bgra8."""
        try:
            # ROS2 may expose msg.data as list or bytes; use np.array for compatibility
            data = np.array(msg.data, dtype=np.uint8)
            if msg.encoding in ('bgr8', 'bgra8'):
                channels = 4 if msg.encoding == 'bgra8' else 3
                shape = (msg.height, msg.width, channels)
                img = data.reshape(shape)
                if channels == 4:
                    img = img[:, :, :3]
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif msg.encoding in ('rgb8', 'rgba8'):
                channels = 4 if msg.encoding == 'rgba8' else 3
                shape = (msg.height, msg.width, channels)
                img = data.reshape(shape)
                if channels == 4:
                    img_rgb = img[:, :, :3].copy()
                else:
                    img_rgb = img
            else:
                # mono8 or unknown: treat as grayscale and duplicate to RGB
                img = data.reshape((msg.height, msg.width))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return img_rgb
        except Exception as e:
            print(f"[OmniNav Online] _image_msg_to_rgb error (encoding={getattr(msg, 'encoding', '?')}): {e}")
            return None

    def _image_front_callback(self, msg: Image):
        """Store latest front camera image."""
        img_rgb = self._image_msg_to_rgb(msg)
        if img_rgb is not None:
            with self.image_lock:
                self.latest_front = img_rgb
                self.image_timestamp = time.time()

    def _image_left_callback(self, msg: Image):
        """Store latest left camera image."""
        img_rgb = self._image_msg_to_rgb(msg)
        if img_rgb is not None:
            with self.image_lock:
                self.latest_left = img_rgb
                self.image_timestamp = time.time()

    def _image_right_callback(self, msg: Image):
        """Store latest right camera image."""
        img_rgb = self._image_msg_to_rgb(msg)
        if img_rgb is not None:
            with self.image_lock:
                self.latest_right = img_rgb
                self.image_timestamp = time.time()
    
    def _spin_ros(self):
        """ROS2 spin in background thread"""
        rclpy.spin(self.ros_node)
    
    def get_latest_images(self) -> tuple:
        """Get the latest front, left, right images (thread-safe).
        
        Returns:
            (front_rgb, left_rgb, right_rgb, timestamp) or (None, None, None, None) if any is missing
        """
        with self.image_lock:
            if self.latest_front is not None and self.latest_left is not None and self.latest_right is not None:
                return (
                    self.latest_front.copy(),
                    self.latest_left.copy(),
                    self.latest_right.copy(),
                    self.image_timestamp,
                )
            return None, None, None, None
    
    def publish_action(self, action: dict):
        """Publish all 5 waypoints to /action topic (for sequential execution), return full waypoint list for visualization"""
        if 'arrive_pred' not in action or 'action' not in action or 'recover_angle' not in action:
            print("[OmniNav Online] Invalid action format, skipping publish")
            return None
        
        arrive = int(action['arrive_pred'])
        waypoints = action['action']  # shape (5, 2)
        recover_angles = action['recover_angle']  # shape (5,)
        
        # Flatten if needed
        if isinstance(waypoints, np.ndarray) and waypoints.ndim > 1:
            waypoints = waypoints.reshape(-1, 2)
        if isinstance(recover_angles, np.ndarray) and recover_angles.ndim > 1:
            recover_angles = recover_angles.flatten()
        
        # Create full waypoint list (for visualization)
        full_waypoint_list = []
        for i in range(min(5, len(waypoints))):
            dx = float(waypoints[i][0])
            dy = float(waypoints[i][1])
            dtheta = float(np.degrees(recover_angles[i])) if i < len(recover_angles) else 0.0
            
            full_waypoint_list.append({
                'dx': dx,
                'dy': dy,
                'dtheta': dtheta,
                'arrive': arrive
            })
        
        # Control: publish all 5 waypoints for sequential execution
        if len(full_waypoint_list) == 0:
            print(f"[Frame {self.frame_count:04d}] ERROR: full_waypoint_list is empty!")
            return None
        control_waypoint_list = full_waypoint_list  # all 5 waypoints

        # Create JSON message
        msg_data = {
            'waypoints': control_waypoint_list,
            'arrive_pred': arrive,
            'timestamp': time.time(),
            'frame_count': self.frame_count
        }
        
        msg = String()
        msg.data = json.dumps(msg_data)
        
        # Check subscription count before publishing
        subscription_count = self.action_pub.get_subscription_count()
        if subscription_count == 0:
            print(f"[OmniNav Online] WARNING: No subscribers for /action topic! (frame {self.frame_count})")
        
        self.action_pub.publish(msg)
        
        # Return full list for visualization
        return full_waypoint_list
        
    def run_loop(self, inference_interval: float = 1.0):
        """Main inference loop - Step-by-Step Mode. Uses camera images from ROS topics."""
        move_duration = inference_interval
        PREDICT_SCALE = 0.3
        info = {'top_down_map_vlnce': None, 'gt_map': None, 'pred_map': None}

        self._run_loop_ros(move_duration, PREDICT_SCALE, info)

    def _run_loop_ros(self, move_duration, PREDICT_SCALE, info):
        """Use subscribed front/left/right images from /cam_front/color/image_raw, /cam_left/color/image_raw, /cam_right/color/image_raw."""
        print(f"\n[OmniNav Online] Starting Step-by-Step Loop (Move Duration: {move_duration}s)")
        print("[OmniNav Online] Waiting for first images (front, left, right)...")
        while True:
            front, left, right, _ = self.get_latest_images()
            if front is not None and left is not None and right is not None:
                print(f"[OmniNav Online] First images received! front={front.shape}, left={left.shape}, right={right.shape}")
                break
            time.sleep(0.1)
        print("[OmniNav Online] Running inference loop. Press Ctrl+C to stop.")
        print("=" * 60)
        try:
            while True:
                front, left, right, _ = self.get_latest_images()
                if front is None or left is None or right is None:
                    print("[OmniNav Online] Waiting for front/left/right images...")
                    time.sleep(0.1)
                    continue
                default_pose = {'position': [0.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0]}
                obs = {
                    'front': front,
                    'left': left,
                    'right': right,
                    'rgb': front,
                    'instruction': {'text': self.instruction},
                    'pose': default_pose
                }
                start_time = time.time()
                with torch.no_grad():
                    action = self.agent.act(obs, info, "online_session")
                infer_time = time.time() - start_time
                self.total_infer_time += infer_time
                self.frame_count += 1
                print(f"[Frame {self.frame_count:04d}] Inference time: {infer_time:.3f}s")
                
                if 'arrive_pred' in action and 'action' in action and 'recover_angle' in action:
                    arrive = int(action['arrive_pred'])
                    waypoints = action['action']
                    recover_angles = action['recover_angle']
                    if isinstance(waypoints, np.ndarray) and waypoints.ndim > 1:
                        waypoints = waypoints.reshape(-1, 2)
                    if isinstance(recover_angles, np.ndarray) and recover_angles.ndim > 1:
                        recover_angles = recover_angles.flatten()
                    for subframe_idx in range(min(5, len(waypoints))):
                        dx = waypoints[subframe_idx][0] / PREDICT_SCALE
                        dy = waypoints[subframe_idx][1] / PREDICT_SCALE
                        dtheta = np.degrees(recover_angles[subframe_idx]) if subframe_idx < len(recover_angles) else 0.0
                        self.csv_records.append({
                            'frame_idx': self.frame_count,
                            'subframe_idx': subframe_idx,
                            'dx': float(dx), 'dy': float(dy), 'dtheta': float(dtheta),
                            'arrive': arrive,
                            'infer_time_s': float(infer_time) if subframe_idx == 0 else 0.0
                        })
                    wp, dtheta0 = waypoints[0], np.degrees(recover_angles[0]) if len(recover_angles) > 0 else 0.0
                    print(f"  -> arrive={arrive}, dtheta={dtheta0:.2f}, wp[0]=({wp[0]:.4f}, {wp[1]:.4f})")
                
                waypoint_list = self.publish_action(action)
                if waypoint_list:
                    arrive_per_wp = [wp.get('arrive', -1) for wp in waypoint_list]
                    print(f"  -> waypoint별 arrive_pred: {arrive_per_wp}")
                # Store left, front (with waypoint arrows), right for video (1-row 3-col layout)
                if waypoint_list and len(waypoint_list) > 0:
                    front_vis = draw_waypoint_arrows_fpv(front, waypoint_list)
                else:
                    front_vis = front.copy()
                self.vis_frame_list.append((left.copy(), front_vis, right.copy()))
                
                if action.get('arrive_pred', 0) == 1:
                    print("\n" + "=" * 60)
                    print("[OmniNav Online] 목적지에 도달하여 루프를 종료합니다 🎉")
                    print("=" * 60)
                    break
                print(f"  >> Moving robot for {move_duration}s...")
                time.sleep(move_duration)
        except KeyboardInterrupt:
            print("\n[OmniNav Online] Stopping...")
        finally:
            self.shutdown()


    def shutdown(self):
        """Clean up resources and save video/CSV"""
        print("[OmniNav Online] Shutting down...")
        
        # Save CSV results (same format as run_infer_iphone.py)
        if len(self.csv_records) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.output_csv:
                csv_file = self.output_csv if os.path.isabs(self.output_csv) else os.path.join(self.result_path, self.output_csv)
            else:
                csv_file = os.path.join(self.result_path, f"waypoint_data_online_{timestamp}.csv")
            os.makedirs(os.path.dirname(os.path.abspath(csv_file)), exist_ok=True)
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['frame_idx', 'subframe_idx', 'dx', 'dy', 'dtheta', 'arrive', 'infer_time_s'])
                writer.writeheader()
                writer.writerows(self.csv_records)
            
            print(f"[OmniNav Online] CSV saved: {csv_file} ({len(self.csv_records)} records, {len(self.csv_records)//5} frames)")
        
        # Print inference time statistics
        if self.frame_count > 0:
            avg_infer_time = self.total_infer_time / self.frame_count
            print(f"\n{'='*60}")
            print(f"[TIMING] Total inference time: {self.total_infer_time:.3f}s")
            print(f"[TIMING] Average inference time per frame: {avg_infer_time:.3f}s")
            print(f"[TIMING] FPS: {1.0/avg_infer_time:.2f}" if avg_infer_time > 0 else "[TIMING] FPS: N/A")
            print(f"{'='*60}\n")
        
        # Save video if frames were collected
        result_path_abs = os.path.abspath(self.result_path)
        print(f"[OmniNav Online] Video save: save_video={self.save_video}, vis_frame_list={len(self.vis_frame_list)} frames, result_path={result_path_abs}")
        if self.save_video and len(self.vis_frame_list) > 0:
            self._save_video()
        elif self.save_video and len(self.vis_frame_list) == 0:
            print("[OmniNav Online] Video NOT saved: no frames collected (did you stop before first inference?)")
        elif not self.save_video:
            print("[OmniNav Online] Video NOT saved: disabled (use default or omit --no-save-video to enable)")

        self.agent.reset()
        self.ros_node.destroy_node()
        rclpy.shutdown()
        print("[OmniNav Online] Shutdown complete")
    
    def _save_video(self):
        """Save visualized frames as MP4: 1-row 3-col layout [LEFT | FRONT(with arrows) | RIGHT]"""
        if len(self.vis_frame_list) == 0:
            print("[OmniNav Online] _save_video: No frames to save, skipping")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.output_video:
            video_path = self.output_video if os.path.isabs(self.output_video) else os.path.join(self.result_path, self.output_video)
        else:
            os.makedirs(self.result_path, exist_ok=True)
            video_path = os.path.join(self.result_path, f"omninav_online_{timestamp}.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
        video_path_abs = os.path.abspath(video_path)

        left_f, front_f, right_f = self.vis_frame_list[0]
        h_ref, w_ref = front_f.shape[:2]
        # Ensure all views same size; resize if needed
        target_h, target_w = h_ref, w_ref

        print(f"\n[OmniNav Online] Saving {len(self.vis_frame_list)} frames to MP4 (LEFT|FRONT|RIGHT)...")

        combined_w = target_w * 3
        combined_h = target_h
        # Add instruction bar below (bar_height from add_instruction_bar)
        out_h = combined_h + 80
        out_w = combined_w
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 1.0
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (out_w, out_h))

        if not video_writer.isOpened():
            print("[OmniNav Online] Warning: mp4v failed, trying avc1...")
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (out_w, out_h))

        if not video_writer.isOpened():
            print("[OmniNav Online] Error: Failed to create video file (check write permission for result_path)")
            return

        for left_f, front_f, right_f in self.vis_frame_list:
            # Resize to match if dimensions differ
            if left_f.shape[:2] != (target_h, target_w):
                left_f = cv2.resize(left_f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if front_f.shape[:2] != (target_h, target_w):
                front_f = cv2.resize(front_f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if right_f.shape[:2] != (target_h, target_w):
                right_f = cv2.resize(right_f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            combined = np.concatenate([left_f, front_f, right_f], axis=1)
            # Add instruction bar below
            frame_with_instruction = add_instruction_bar(combined, self.instruction)
            frame_bgr = cv2.cvtColor(frame_with_instruction, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        print("=" * 60)
        print(f"[OmniNav Online] VIDEO SAVED: {video_path_abs}")
        print(f"[OmniNav Online] Video info: {len(self.vis_frame_list)} frames, {out_w}x{out_h} (LEFT|FRONT|RIGHT), {fps} fps")
        print("=" * 60)


# ========== 저장 경로/파일명 직접 설정 (필요 시 여기만 수정) ==========
# 현재 파일(run_infer_online_panorama.py) 위치 기준 상대경로
SAVE_PATH = "./field_test_results"        # 예: "results", "output", "../data/out"
# CSV/MP4 파일명. None 이면 자동: waypoint_data_online_<timestamp>.csv / omninav_online_<timestamp>.mp4
# 문자열이면 해당 이름 사용 (상대경로는 SAVE_PATH 아래, 절대경로는 그대로)
OUTPUT_CSV_FILENAME = "traj6_prompt1.csv"    # 예: "my_run.csv" 또는 "/path/to/out.csv"
OUTPUT_VIDEO_FILENAME = "traj6_prompt1.mp4"  # 예: "my_run.mp4" 또는 "/path/to/out.mp4"
# =================================================================


def main():
    parser = argparse.ArgumentParser(description='OmniNav Real-time Inference')
    
    parser.add_argument("--model-path", type=str, default="../OmniNav", help="OmniNav model path (default: ../OmniNav)")
    parser.add_argument("--instruction", type=str, required=True, help="Navigation instruction")
    parser.add_argument("--result-path", type=str, default=None, help="Override save path (default: use SAVE_PATH relative to this file)")
    parser.add_argument("--inference-interval", type=float, default=1.0, help="Time between inferences in seconds (default: 1.0)")
    parser.add_argument("--no-save-video", action="store_true", help="Disable video saving")
    args = parser.parse_args()

    save_video = not args.no_save_video

    # 저장 경로: --result-path 미지정 시 현재 파일 위치 + SAVE_PATH
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    result_path = args.result_path if args.result_path is not None else os.path.normpath(os.path.join(_script_dir, SAVE_PATH))

    inference = OmniNavOnlineInference(
        model_path=args.model_path,
        instruction=args.instruction,
        result_path=result_path,
        output_csv=OUTPUT_CSV_FILENAME,
        output_video=OUTPUT_VIDEO_FILENAME,
    )
    inference.save_video = save_video
    result_abs = os.path.abspath(result_path)
    if save_video:
        print(f"[OmniNav Online] Video saving enabled (MP4) -> will save to: {result_abs}")
    else:
        print("[OmniNav Online] Video saving disabled")
    
    inference.run_loop(inference_interval=args.inference_interval)


if __name__ == "__main__":
    main()
