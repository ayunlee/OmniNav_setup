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
import torch
import json
import csv
import time
import threading
import cv2
from datetime import datetime
import math
from collections import deque

import matplotlib.cm as mpl_cm

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from message_filters import Subscriber, ApproximateTimeSynchronizer

from agent.waypoint_agent import Waypoint_Agent


class _TeeTextIO:
    """Write text to multiple streams (used to mirror stdout/stderr into a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        # Keep terminal behavior where possible.
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


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
    stop_color: tuple = (0, 0, 255),   # BGR red for STOP/OK marker
    stop_radius: int = 8,
    arrow_scale: float = 0.15,   # arrow length in body-frame meters
    vis_scale: float = 120.0,    # meters -> pixel
    arrow_gap: int = 1.2,          # pixel gap between stacked arrows (0 = tight)
) -> np.ndarray:
    """
    Draw each waypoint as an arrow stacked vertically (UniNaVid-style layout).
    - 1st waypoint: center bottom (base)
    - 2nd waypoint: directly above 1st (no gap)
    - 3rd above 2nd, ... 5th at top
    - Color gradient: turbo (smooth blue->cyan->green->yellow->red)
    - Body frame: theta = atan2(-dx, dy), forward -> up in image
    """
    out = img.copy()
    h, w = out.shape[:2]
    base_x, base_y = w // 2, int(h * 0.95)  # origin at bottom center

    # Slot height = max vertical extent of one arrow (when pointing straight up) + gap
    slot_height = max(1, int(vis_scale * arrow_scale) + arrow_gap)

    # Smooth gradient: turbo is perceptually uniform and continuous
    try:
        cmap = mpl_cm.get_cmap('turbo')
    except Exception:
        cmap = mpl_cm.get_cmap('viridis')

    n_wp = len(waypoints)
    for i, wp in enumerate(waypoints):
        dx_net = wp.get('dx', 0.0)
        dy_net = wp.get('dy', 0.0)
        arrive = wp.get('arrive', 0)

        # Stack upward: each arrow starts right below the previous arrow's top
        start_y = int(base_y - i * slot_height)
        start_pixel = (base_x, start_y)

        if arrive > 0:
            cv2.circle(out, start_pixel, stop_radius, stop_color, -1)
            cv2.putText(out, "OK", (start_pixel[0] - 12, start_pixel[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            break

        # Direction: theta = atan2(-dx, dy) (same as visualize_csv_bodyframe)
        theta = math.atan2(-dx_net, dy_net) if (dx_net != 0 or dy_net != 0) else 0.0
        dx_arrow = arrow_scale * math.cos(theta)
        dy_arrow = arrow_scale * math.sin(theta)

        # Map body frame to image: forward -> up (-y), left -> left (-x)
        head_x = int(start_pixel[0] - dy_arrow * vis_scale)
        head_y = int(start_pixel[1] - dx_arrow * vis_scale)
        head_pixel = (np.clip(head_x, 0, w - 1), np.clip(head_y, 0, h - 1))

        # Smooth gradient: use fine interpolation for continuity
        t = (i + 0.5) / n_wp if n_wp > 0 else 0.5  # center of each band for smoother transition
        rgba = cmap(t)[:3]
        color = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))  # RGB -> BGR

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

    def __init__(
        self,
        model_path: str,
        instruction: str = "",
        result_path: str = "./data/result_online",
        pose_topic: str = "/omni/pose2d",
        subgoal_topic: str = "/omni/subgoal",
        instruction_topic: str = "/omni/instruction",
        use_subgoal_hint: bool = True,
        use_coordinate_tokens: bool = False,
        max_vis_frames: int = 100,
        max_camera_skew_ms: float = 200.0,
        max_pose_dt_ms: float = 150.0,
        max_subgoal_age_ms: float = 5000.0,
    ):
        """
        Args:
            model_path: Path to OmniNav model
            instruction: Optional CLI instruction text (if empty, read from instruction_topic)
            result_path: Path to save results
        """
        self.result_path = result_path
        self.instruction = (instruction or "").strip()
        self.pose_topic = pose_topic
        self.subgoal_topic = subgoal_topic
        self.instruction_topic = instruction_topic
        self.use_subgoal_hint = use_subgoal_hint
        self.use_coordinate_tokens = use_coordinate_tokens
        self.max_camera_skew_s = max(0.0, float(max_camera_skew_ms) / 1000.0)
        self.max_pose_dt_s = max(0.0, float(max_pose_dt_ms) / 1000.0)
        self.max_subgoal_age_s = max(0.0, float(max_subgoal_age_ms) / 1000.0)

        # Image buffers for front/left/right from ROS
        self.latest_front = None
        self.latest_left = None
        self.latest_right = None
        self.latest_front_stamp = None
        self.latest_left_stamp = None
        self.latest_right_stamp = None
        self.latest_frame_stamp = None
        self.latest_cam_skew = None
        self.image_lock = threading.Lock()
        self.image_timestamp = None

        # Navigation state buffers from ROS
        self.latest_pose2d = None
        self.pose_buffer = deque(maxlen=512)
        self.latest_subgoal = None
        self.latest_instruction = None
        self.state_lock = threading.Lock()
        self.last_runtime_instruction = self.instruction
        self._last_sync_warn_ts = 0.0
        
        # Frame counter
        self.frame_count = 0
        
        # Visualization storage
        self.vis_frame_list = deque(maxlen=max(10, int(max_vis_frames)))
        self.save_video = True
        
        # CSV records for waypoint data
        self.csv_records = []
        self.total_infer_time = 0.0
        
        # Initialize ROS2
        rclpy.init()
        self.ros_node = rclpy.create_node('omninav_inference')
        
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        # /action: must match omninav_control subscriber exactly (all policies)
        qos_action = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Image subscribers (front, left, right) with timestamp synchronization.
        # Using message_filters avoids mixing stale "latest" frames from different cameras.
        self.front_sub = Subscriber(self.ros_node, Image, '/cam_front/color/image_raw', qos_profile=qos_profile_sensor_data)
        self.left_sub = Subscriber(self.ros_node, Image, '/cam_left/color/image_raw', qos_profile=qos_profile_sensor_data)
        self.right_sub = Subscriber(self.ros_node, Image, '/cam_right/color/image_raw', qos_profile=qos_profile_sensor_data)
        self.image_sync = ApproximateTimeSynchronizer(
            [self.front_sub, self.left_sub, self.right_sub],
            queue_size=100,
            slop=float(max(0.02, self.max_camera_skew_s)),
        )
        self.image_sync.registerCallback(self._image_synced_callback)
        self.ros_node.create_subscription(
            PoseStamped,
            self.pose_topic,
            self._pose2d_callback,
            qos_reliable
        )
        self.ros_node.create_subscription(
            Odometry,
            '/Odometry',
            self._odometry_fallback_callback,
            qos_reliable
        )
        self.ros_node.create_subscription(
            PoseStamped,
            self.subgoal_topic,
            self._subgoal_callback,
            qos_reliable
        )
        self.ros_node.create_subscription(
            String,
            self.instruction_topic,
            self._instruction_callback,
            qos_reliable
        )

        # Action publisher (waypoints as JSON string)
        self.action_pub = self.ros_node.create_publisher(String, '/action', qos_action)
        
        print("=" * 60)
        print("[OmniNav Online] ROS2 Node Ready")
        print(
            "[OmniNav Online] Subscribing to: "
            "/cam_front/color/image_raw, /cam_left/color/image_raw, /cam_right/color/image_raw, "
            f"{self.pose_topic}, {self.subgoal_topic}, {self.instruction_topic}"
        )
        print("[OmniNav Online] Publishing to: /action")
        print(
            f"[OmniNav Online] Sync gates: camera_skew<={self.max_camera_skew_s*1000:.0f}ms, "
            f"pose_dt<={self.max_pose_dt_s*1000:.0f}ms, subgoal_age<={self.max_subgoal_age_s*1000:.0f}ms"
        )
        if self.instruction:
            print(f"[OmniNav Online] Instruction source: CLI ({self.instruction[:80]}...)")
        else:
            print("[OmniNav Online] Instruction source: /omni/instruction topic (waiting first message)")
        print("=" * 60)
        
        # Initialize OmniNav agent
        # Use system temp directory for agent (it needs result_path but we don't use its outputs)
        # We only save video to result_path, not agent's internal folders
        import tempfile
        temp_agent_path = tempfile.mkdtemp(prefix="omninav_agent_")
        
        print("[OmniNav Online] Loading model...")
        self.agent = Waypoint_Agent(
            model_path,
            temp_agent_path,
            require_map=False,
            use_coordinate_tokens=self.use_coordinate_tokens,
        )
        self.agent.reset()
        self.agent.episode_id = "online_session"
        print("[OmniNav Online] Model loaded successfully")
        print(f"[OmniNav Online] Coordinate tokens: {self.use_coordinate_tokens}")
        
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

    def _image_synced_callback(self, front_msg: Image, left_msg: Image, right_msg: Image):
        """Store synchronized front/left/right image triplet."""
        front_rgb = self._image_msg_to_rgb(front_msg)
        left_rgb = self._image_msg_to_rgb(left_msg)
        right_rgb = self._image_msg_to_rgb(right_msg)
        if front_rgb is None or left_rgb is None or right_rgb is None:
            return

        front_stamp = self._stamp_to_sec(front_msg.header.stamp)
        left_stamp = self._stamp_to_sec(left_msg.header.stamp)
        right_stamp = self._stamp_to_sec(right_msg.header.stamp)
        stamps = [front_stamp, left_stamp, right_stamp]
        frame_stamp = float(sum(stamps) / 3.0)
        cam_skew = float(max(stamps) - min(stamps))

        with self.image_lock:
            self.latest_front = front_rgb
            self.latest_left = left_rgb
            self.latest_right = right_rgb
            self.latest_front_stamp = front_stamp
            self.latest_left_stamp = left_stamp
            self.latest_right_stamp = right_stamp
            self.latest_frame_stamp = frame_stamp
            self.latest_cam_skew = cam_skew
            self.image_timestamp = time.time()

    @staticmethod
    def _yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _pose2d_callback(self, msg: PoseStamped):
        q = msg.pose.orientation
        yaw = self._yaw_from_quat_xyzw(q.x, q.y, q.z, q.w)
        pose_msg = {
            'x': float(msg.pose.position.x),
            'y': float(msg.pose.position.y),
            'yaw': float(yaw),
            'stamp': self._stamp_to_sec(msg.header.stamp),
            'source': 'pose2d',
        }
        with self.state_lock:
            self.latest_pose2d = pose_msg
            self.pose_buffer.append(pose_msg)

    def _odometry_fallback_callback(self, msg: Odometry):
        # Fallback only: /omni/pose2d has priority when available.
        with self.state_lock:
            if self.latest_pose2d is not None and self.latest_pose2d.get('source') == 'pose2d':
                return
        q = msg.pose.pose.orientation
        yaw = self._yaw_from_quat_xyzw(q.x, q.y, q.z, q.w)
        pose_msg = {
            'x': float(msg.pose.pose.position.x),
            'y': float(msg.pose.pose.position.y),
            'yaw': float(yaw),
            'stamp': self._stamp_to_sec(msg.header.stamp),
            'source': 'odometry',
        }
        with self.state_lock:
            self.latest_pose2d = pose_msg
            self.pose_buffer.append(pose_msg)

    def _subgoal_callback(self, msg: PoseStamped):
        with self.state_lock:
            self.latest_subgoal = {
                'x': float(msg.pose.position.x),
                'y': float(msg.pose.position.y),
                'stamp': self._stamp_to_sec(msg.header.stamp),
            }

    def _instruction_callback(self, msg: String):
        text = (msg.data or "").strip()
        if not text:
            return
        with self.state_lock:
            self.latest_instruction = text
    
    def _spin_ros(self):
        """ROS2 spin in background thread"""
        rclpy.spin(self.ros_node)
    
    def get_latest_images(self) -> tuple:
        """Get the latest front, left, right images (thread-safe).
        
        Returns:
            (front_rgb, left_rgb, right_rgb, frame_stamp, cam_skew)
            or (None, None, None, None, None) if any is missing
        """
        with self.image_lock:
            if (
                self.latest_front is not None and self.latest_left is not None and self.latest_right is not None
                and self.latest_front_stamp is not None
                and self.latest_left_stamp is not None
                and self.latest_right_stamp is not None
                and self.latest_frame_stamp is not None
                and self.latest_cam_skew is not None
            ):
                return (
                    self.latest_front.copy(),
                    self.latest_left.copy(),
                    self.latest_right.copy(),
                    float(self.latest_frame_stamp),
                    float(self.latest_cam_skew),
                )
            return None, None, None, None, None

    def get_latest_nav_state(self):
        with self.state_lock:
            pose = dict(self.latest_pose2d) if self.latest_pose2d is not None else None
            subgoal = dict(self.latest_subgoal) if self.latest_subgoal is not None else None
            instr = self.latest_instruction
        return pose, subgoal, instr

    def _get_pose_near_stamp(self, target_stamp: float):
        with self.state_lock:
            if len(self.pose_buffer) == 0:
                return None, None
            pose = min(self.pose_buffer, key=lambda p: abs(float(p['stamp']) - target_stamp))
        dt = abs(float(pose['stamp']) - target_stamp)
        if dt > self.max_pose_dt_s:
            return None, dt
        return dict(pose), dt

    def _get_fresh_subgoal(self, target_stamp: float):
        with self.state_lock:
            subgoal = dict(self.latest_subgoal) if self.latest_subgoal is not None else None
        if subgoal is None:
            return None, None
        age = abs(float(subgoal['stamp']) - target_stamp)
        if age > self.max_subgoal_age_s:
            return None, age
        return subgoal, age

    def _sync_warn(self, message: str):
        now = time.time()
        if now - self._last_sync_warn_ts >= 1.0:
            print(f"[OmniNav Online][SYNC] {message}")
            self._last_sync_warn_ts = now

    def _get_active_instruction(self) -> str:
        if self.instruction:
            return self.instruction
        with self.state_lock:
            if self.latest_instruction:
                return self.latest_instruction
        return ""

    def _pose2d_to_model_pose(self, pose2d: dict) -> dict:
        # Convert ROS planar map pose to Habitat-like format expected by Waypoint_Agent.
        # Position uses [x, y_up, z] => [map_x, 0, map_y]
        # Rotation uses quaternion order [w, x, y, z].
        yaw = float(pose2d['yaw'])
        half = 0.5 * yaw
        return {
            'position': [float(pose2d['x']), 0.0, float(pose2d['y'])],
            'rotation': [float(math.cos(half)), 0.0, float(math.sin(half)), 0.0],
        }

    @staticmethod
    def _wrap_pi(angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi

    def _subgoal_to_target_waypoint(self, model_pose: dict, subgoal: dict):
        """
        Convert subgoal map/world coordinate to expected network waypoint frame.
        Returns:
            (dx_target, dy_target, local_xz)
            - local_xz follows coordinate-token convention: [x_local, z_local]
            - network waypoint convention: dx=right(+), dy=forward(+)
        """
        subgoal_world_h = np.array(
            [float(subgoal["x"]), 0.0, float(subgoal["y"]), 1.0],
            dtype=np.float32,
        )
        current_T = self.agent.pose_to_matrix(model_pose)
        subgoal_local_h = np.linalg.inv(current_T) @ subgoal_world_h
        local_x = float(subgoal_local_h[0])
        local_z = float(subgoal_local_h[2])
        # local [x, z] -> waypoint [dx(right), dy(forward)]
        dx_target = local_z
        dy_target = local_x
        return dx_target, dy_target, (local_x, local_z)

    @staticmethod
    def _angle_error_deg(dx_target: float, dy_target: float, dx_wp: float, dy_wp: float):
        nt = math.hypot(dx_target, dy_target)
        nw = math.hypot(dx_wp, dy_wp)
        if nt < 1e-6 or nw < 1e-6:
            return None
        dot = dx_target * dx_wp + dy_target * dy_wp
        cosv = dot / (nt * nw)
        cosv = max(-1.0, min(1.0, cosv))
        return math.degrees(math.acos(cosv))

    def _build_runtime_instruction(self, pose2d: dict, subgoal: dict) -> str:
        base_instruction = self._get_active_instruction()
        if not base_instruction:
            base_instruction = "Find the target object."
        if not self.use_subgoal_hint or pose2d is None or subgoal is None:
            return base_instruction
        dx = float(subgoal['x']) - float(pose2d['x'])
        dy = float(subgoal['y']) - float(pose2d['y'])
        dist = math.hypot(dx, dy)
        goal_heading = math.atan2(dy, dx)
        rel_heading = self._wrap_pi(goal_heading - float(pose2d['yaw']))
        hint = (
            f" Intermediate subgoal map=({subgoal['x']:.2f}, {subgoal['y']:.2f}), "
            f"distance={dist:.2f}m, relative_heading={math.degrees(rel_heading):.1f}deg."
        )
        return base_instruction + hint
    
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

        # Wait for /action subscriber (omninav_control) so messages are not lost
        wait_start = time.time()
        while self.action_pub.get_subscription_count() == 0:
            if time.time() - wait_start > 30.0:
                print("[OmniNav Online] WARNING: No /action subscriber after 30s. Start omninav_control first? Proceeding anyway.")
                break
            print("[OmniNav Online] Waiting for /action subscriber (run omninav_control in another terminal)...")
            time.sleep(0.5)
        if self.action_pub.get_subscription_count() > 0:
            print("[OmniNav Online] /action subscriber connected.")

        self._run_loop_ros(move_duration, PREDICT_SCALE, info)

    def _run_loop_ros(self, move_duration, PREDICT_SCALE, info):
        """Use subscribed front/left/right images from /cam_front/color/image_raw, /cam_left/color/image_raw, /cam_right/color/image_raw."""
        print(f"\n[OmniNav Online] Starting Step-by-Step Loop (Move Duration: {move_duration}s)")
        print("[OmniNav Online] Waiting for first images (front, left, right), pose, and instruction...")
        while True:
            front, left, right, frame_stamp, cam_skew = self.get_latest_images()
            pose2d, pose_dt = (None, None)
            if frame_stamp is not None and cam_skew is not None and cam_skew <= self.max_camera_skew_s:
                pose2d, pose_dt = self._get_pose_near_stamp(frame_stamp)
            _, _, instruction_from_topic = self.get_latest_nav_state()
            active_instruction = self.instruction if self.instruction else (instruction_from_topic or "")
            if (
                front is not None
                and left is not None
                and right is not None
                and pose2d is not None
                and bool(active_instruction.strip())
            ):
                print(
                    "[OmniNav Online] First data received! "
                    f"front={front.shape}, left={left.shape}, right={right.shape}, "
                    f"cam_skew_ms={cam_skew*1000.0:.1f}, pose_dt_ms={pose_dt*1000.0:.1f}, "
                    f"pose_source={pose2d.get('source', 'unknown')}, "
                    f"instruction_source={'cli' if self.instruction else 'topic'}"
                )
                break
            if not active_instruction.strip():
                print(f"[OmniNav Online] Waiting for instruction topic: {self.instruction_topic}")
            elif cam_skew is not None and cam_skew > self.max_camera_skew_s:
                self._sync_warn(
                    f"waiting synced tri-view: cam_skew={cam_skew*1000.0:.1f}ms > {self.max_camera_skew_s*1000.0:.1f}ms"
                )
            elif frame_stamp is not None and pose2d is None and pose_dt is not None:
                self._sync_warn(
                    f"waiting synced pose: nearest_dt={pose_dt*1000.0:.1f}ms > {self.max_pose_dt_s*1000.0:.1f}ms"
                )
            time.sleep(0.1)
        print("[OmniNav Online] Running inference loop. Press Ctrl+C to stop.")
        print("=" * 60)
        try:
            while True:
                front, left, right, frame_stamp, cam_skew = self.get_latest_images()
                if front is None or left is None or right is None:
                    print("[OmniNav Online] Waiting for front/left/right images...")
                    time.sleep(0.1)
                    continue
                if cam_skew is None or cam_skew > self.max_camera_skew_s:
                    if cam_skew is not None:
                        self._sync_warn(
                            f"drop frame: cam_skew={cam_skew*1000.0:.1f}ms > {self.max_camera_skew_s*1000.0:.1f}ms"
                        )
                    time.sleep(0.02)
                    continue
                pose2d, pose_dt = self._get_pose_near_stamp(frame_stamp)
                if pose2d is None:
                    if pose_dt is None:
                        self._sync_warn("waiting pose topic...")
                    else:
                        self._sync_warn(
                            f"drop frame: nearest pose dt={pose_dt*1000.0:.1f}ms > {self.max_pose_dt_s*1000.0:.1f}ms"
                        )
                    time.sleep(0.02)
                    continue
                subgoal, subgoal_age = self._get_fresh_subgoal(frame_stamp)
                if self.use_coordinate_tokens and subgoal is None:
                    if subgoal_age is None:
                        self._sync_warn("waiting subgoal topic for coordinate tokens...")
                    else:
                        self._sync_warn(
                            f"stale subgoal: age={subgoal_age*1000.0:.1f}ms > {self.max_subgoal_age_s*1000.0:.1f}ms"
                        )
                    time.sleep(0.02)
                    continue
                runtime_instruction = self._build_runtime_instruction(pose2d, subgoal)
                self.last_runtime_instruction = runtime_instruction
                model_pose = self._pose2d_to_model_pose(pose2d)
                align_dx_target = None
                align_dy_target = None
                align_err_deg = None
                align_local_x = None
                align_local_z = None
                if subgoal is not None:
                    try:
                        align_dx_target, align_dy_target, local_xz = self._subgoal_to_target_waypoint(model_pose, subgoal)
                        align_local_x, align_local_z = local_xz
                    except Exception as exc:
                        self._sync_warn(f"failed target projection: {exc}")
                obs = {
                    'front': front,
                    'left': left,
                    'right': right,
                    'rgb': front,
                    'instruction': {'text': runtime_instruction},
                    'pose': model_pose,
                    # Optional conditioning path for coordinate-token fusion in Waypoint_Agent.
                    'subgoal': subgoal,
                }
                start_time = time.time()
                with torch.no_grad():
                    action = self.agent.act(obs, info, "online_session")
                infer_time = time.time() - start_time
                self.total_infer_time += infer_time
                self.frame_count += 1
                print(
                    f"[Frame {self.frame_count:04d}] Inference time: {infer_time:.3f}s "
                    f"(cam_skew={cam_skew*1000.0:.1f}ms, pose_dt={pose_dt*1000.0:.1f}ms, "
                    f"subgoal_age_ms={'none' if subgoal_age is None else f'{subgoal_age*1000.0:.1f}'})"
                )
                if action.get("coord_token_fallback", False):
                    print(
                        f"[FALLBACK][Frame {self.frame_count:04d}] "
                        f"reasons={action.get('fallback_reasons', [])} "
                        f"total={action.get('fallback_total_count', -1)}"
                    )
                
                if 'arrive_pred' in action and 'action' in action and 'recover_angle' in action:
                    arrive = int(action['arrive_pred'])
                    waypoints = action['action']
                    recover_angles = action['recover_angle']
                    if isinstance(waypoints, np.ndarray) and waypoints.ndim > 1:
                        waypoints = waypoints.reshape(-1, 2)
                    if isinstance(recover_angles, np.ndarray) and recover_angles.ndim > 1:
                        recover_angles = recover_angles.flatten()
                    if len(waypoints) > 0 and align_dx_target is not None and align_dy_target is not None:
                        wp0_dx = float(waypoints[0][0])
                        wp0_dy = float(waypoints[0][1])
                        align_err_deg = self._angle_error_deg(
                            align_dx_target, align_dy_target, wp0_dx, wp0_dy
                        )
                        if align_err_deg is not None:
                            print(
                                "[ALIGN] "
                                f"subgoal_local[x,z]=({align_local_x:.4f},{align_local_z:.4f}) "
                                f"target(dx,dy)=({align_dx_target:.4f},{align_dy_target:.4f}) "
                                f"wp0(dx,dy)=({wp0_dx:.4f},{wp0_dy:.4f}) "
                                f"angle_err={align_err_deg:.1f}deg"
                            )
                    for subframe_idx in range(min(5, len(waypoints))):
                        dx = waypoints[subframe_idx][0] / PREDICT_SCALE
                        dy = waypoints[subframe_idx][1] / PREDICT_SCALE
                        dtheta = np.degrees(recover_angles[subframe_idx]) if subframe_idx < len(recover_angles) else 0.0
                        self.csv_records.append({
                            'frame_idx': self.frame_count,
                            'subframe_idx': subframe_idx,
                            'dx': float(dx), 'dy': float(dy), 'dtheta': float(dtheta),
                            'arrive': arrive,
                            'infer_time_s': float(infer_time) if subframe_idx == 0 else 0.0,
                            'target_dx': float(align_dx_target) if (subframe_idx == 0 and align_dx_target is not None) else "",
                            'target_dy': float(align_dy_target) if (subframe_idx == 0 and align_dy_target is not None) else "",
                            'angle_error_deg': float(align_err_deg) if (subframe_idx == 0 and align_err_deg is not None) else "",
                        })
                    wp, dtheta0 = waypoints[0], np.degrees(recover_angles[0]) if len(recover_angles) > 0 else 0.0
                    print(f"  -> arrive={arrive}, dtheta={dtheta0:.2f}, wp[0]=({wp[0]:.4f}, {wp[1]:.4f})")
                
                waypoint_list = self.publish_action(action)
                # Store left, front (with waypoint arrows), right for video (1-row 3-col layout)
                if waypoint_list and len(waypoint_list) > 0:
                    front_vis = draw_waypoint_arrows_fpv(front, waypoint_list)
                else:
                    front_vis = front.copy()
                if int(action.get('arrive_pred', 0)) != 0:
                    cv2.putText(
                        front_vis,
                        "OK",
                        (max(10, int(front_vis.shape[1] * 0.45)), max(30, int(front_vis.shape[0] * 0.12))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                self.vis_frame_list.append((left.copy(), front_vis, right.copy()))
                
                if action.get('arrive_pred', 0) != 0:
                    print("\n" + "=" * 60)
                    print("[OmniNav Online] ARRIVED! Navigation complete.")
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
            os.makedirs(self.result_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = os.path.join(self.result_path, f"waypoint_data_online_{timestamp}.csv")
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'frame_idx', 'subframe_idx', 'dx', 'dy', 'dtheta', 'arrive', 'infer_time_s',
                        'target_dx', 'target_dy', 'angle_error_deg',
                    ],
                )
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

        os.makedirs(self.result_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.result_path, f"omninav_online_{timestamp}.mp4")
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
            frame_with_instruction = add_instruction_bar(combined, self.last_runtime_instruction)
            frame_bgr = cv2.cvtColor(frame_with_instruction, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        print("=" * 60)
        print(f"[OmniNav Online] VIDEO SAVED: {video_path_abs}")
        print(f"[OmniNav Online] Video info: {len(self.vis_frame_list)} frames, {out_w}x{out_h} (LEFT|FRONT|RIGHT), {fps} fps")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='OmniNav Real-time Inference')
    
    parser.add_argument("--model-path", type=str, default="/workspace/OmniNav/OmniNav_Slowfast", help="OmniNav model path")
    parser.add_argument("--instruction", type=str, default="", help="Navigation instruction (optional; if empty, subscribe from slow)")
    parser.add_argument("--result-path", type=str, default="./results", help="Result save path")
    parser.add_argument("--inference-interval", type=float, default=0.5, help="Time between inferences in seconds")
    parser.add_argument("--no-save-video", action="store_true", help="Disable video saving")
    parser.add_argument("--pose-topic", type=str, default="/omni/pose2d", help="Pose topic (PoseStamped)")
    parser.add_argument("--subgoal-topic", type=str, default="/omni/subgoal", help="Subgoal topic (PoseStamped)")
    parser.add_argument("--instruction-topic", type=str, default="/omni/instruction", help="Instruction topic (String)")
    parser.set_defaults(use_subgoal_hint=False, use_coordinate_tokens=True)
    parser.add_argument("--no-subgoal-hint", dest="use_subgoal_hint", action="store_false", help="Disable subgoal text hint injection")
    parser.add_argument("--use-subgoal-hint", dest="use_subgoal_hint", action="store_true", help="Enable subgoal text hint injection")
    parser.add_argument("--use-coordinate-tokens", dest="use_coordinate_tokens", action="store_true", help="Enable coordinate-token conditioning (input_waypoints)")
    parser.add_argument("--no-coordinate-tokens", dest="use_coordinate_tokens", action="store_false", help="Disable coordinate-token conditioning (input_waypoints)")
    parser.add_argument("--max-camera-skew-ms", type=float, default=200.0, help="Max timestamp skew among front/left/right images")
    parser.add_argument("--max-pose-dt-ms", type=float, default=150.0, help="Max |pose_stamp-image_stamp| for synced inference")
    parser.add_argument("--max-subgoal-age-ms", type=float, default=5000.0, help="Max |subgoal_stamp-image_stamp| when using coordinate tokens")
    parser.add_argument("--max-vis-frames", type=int, default=100, help="Keep only the latest N frames for MP4 saving")
    args = parser.parse_args()

    # Auto-save console logs into result_path without requiring shell tee.
    os.makedirs(args.result_path, exist_ok=True)
    log_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = os.path.join(args.result_path, f"run_online_{log_ts}.log")
    run_log_abs = os.path.abspath(run_log_path)
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    log_fp = open(run_log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeTextIO(_orig_stdout, log_fp)
    sys.stderr = _TeeTextIO(_orig_stderr, log_fp)
    print(f"[OmniNav Online] Runtime log file: {run_log_abs}")

    save_video = not args.no_save_video

    try:
        inference = OmniNavOnlineInference(
            model_path=args.model_path,
            instruction=args.instruction,
            result_path=args.result_path,
            pose_topic=args.pose_topic,
            subgoal_topic=args.subgoal_topic,
            instruction_topic=args.instruction_topic,
            use_subgoal_hint=args.use_subgoal_hint,
            use_coordinate_tokens=args.use_coordinate_tokens,
            max_vis_frames=args.max_vis_frames,
            max_camera_skew_ms=args.max_camera_skew_ms,
            max_pose_dt_ms=args.max_pose_dt_ms,
            max_subgoal_age_ms=args.max_subgoal_age_ms,
        )
        inference.save_video = save_video
        result_abs = os.path.abspath(args.result_path)
        if save_video:
            print(f"[OmniNav Online] Video saving enabled (MP4) -> will save to: {result_abs}")
        else:
            print("[OmniNav Online] Video saving disabled")

        inference.run_loop(inference_interval=args.inference_interval)
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        try:
            log_fp.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
