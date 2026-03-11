#!/usr/bin/env python3
"""WP-C v1: Habitat adapter node for real-world OmniNav slow pipeline.

v1 policy:
- Keep OVON slow data-flow shape (frontiers -> subgoal), but replace Habitat I/O by ROS2 topics.
- Internal memory bank is embedded in this node (WP-B folded into WP-C).
- Default decision mode is `rule` for robust bring-up.
- Optional `model` mode exists as extension point and falls back to rule if runtime deps fail.
"""

from __future__ import annotations

import os
import sys

# Keep runtime identical to existing OmniNav inference scripts in this container.
_LD_PRELOAD_LIBS = "/opt/hpcx/ucx/lib/libucs.so.0:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucm.so.0"
_REEXEC_VAR = "_OMNINAV_REEXEC"
if os.environ.get(_REEXEC_VAR) != "1" and os.path.exists("/opt/hpcx/ucx/lib/libucs.so.0"):
    cur_preload = os.environ.get("LD_PRELOAD", "")
    if _LD_PRELOAD_LIBS not in cur_preload:
        os.environ["LD_PRELOAD"] = f"{_LD_PRELOAD_LIBS}:{cur_preload}" if cur_preload else _LD_PRELOAD_LIBS
    os.environ[_REEXEC_VAR] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import json
import math
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from PIL import Image as PILImage


OBJ_GOAL_TEMPLATE = [
    "Find a {} in your immediate surroundings and stop when you see one.",
    "Explore the area until you locate a {}. Stop when you've reached its location.",
    "Move through the environment to discover a {}. Your task is complete when you're directly facing it.",
    "Navigate to any visible {}. Stop immediately upon successful discovery.",
    "Search for an instance of {} within this space. Terminate navigation once you've positioned yourself within arm's reach of it.",
    "Survey the surroundings until you identify a {}. Stop navigating as soon as you are positioned directly in front of it",
    "Roam through the space until a {} is spotted. Terminate navigation the moment you’re certain you’re facing it.",
    "Go to the {}, then stop at the front of it.",
    "Move to the nearst {}, then stop",
    "Navigate to a nearst {}, then stop over there.",
    "Get close to the {}, then stop",
    "Could you help me find a {}? Show me the way",
]

_COORD_PATTERN = re.compile(r"coordinate (\[.*?\])")


def parse_ovon_response(response_text: str) -> Tuple[Optional[List[float]], bool]:
    match = _COORD_PATTERN.search(response_text)
    if match is None:
        return None, False
    try:
        coord_list = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None, False
    if not isinstance(coord_list, list):
        return None, False
    return coord_list, ("found" in response_text)


def process_ovon_vision(color_list: Sequence[np.ndarray]) -> List[PILImage.Image]:
    images: List[PILImage.Image] = []
    for image in color_list:
        pil = PILImage.fromarray(image)
        images.append(pil.resize((486, 420)))
    return images


def quat_xyzw_to_rot_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def transform_to_local_frame(
    world_point: np.ndarray,
    agent_world_coord: np.ndarray,
    agent_world_quat: dict,
) -> List[float]:
    relative_point = np.asarray(world_point, dtype=np.float32) - np.asarray(agent_world_coord, dtype=np.float32)
    rot = quat_xyzw_to_rot_matrix(
        float(agent_world_quat["x"]),
        float(agent_world_quat["y"]),
        float(agent_world_quat["z"]),
        float(agent_world_quat["w"]),
    )
    local_point = rot.T @ relative_point
    local_point = np.round(local_point, 2)
    return [float(local_point[0]), float(-local_point[1]), float(-local_point[2])]


def transform_from_local_frame(
    local_point: np.ndarray,
    agent_world_coord: np.ndarray,
    agent_world_quat: dict,
) -> np.ndarray:
    local = np.asarray(local_point, dtype=np.float32)
    local_fix = np.array([local[0], -local[1], -local[2]], dtype=np.float32)
    rot = quat_xyzw_to_rot_matrix(
        float(agent_world_quat["x"]),
        float(agent_world_quat["y"]),
        float(agent_world_quat["z"]),
        float(agent_world_quat["w"]),
    )
    return (rot @ local_fix) + np.asarray(agent_world_coord, dtype=np.float32)


@dataclass
class OvonRotation:
    x: float
    y: float
    z: float
    w: float


@dataclass
class OvonAgentState:
    position: np.ndarray
    rotation: OvonRotation


@dataclass
class Pose2DState:
    x: float
    y: float
    yaw: float
    qx: float
    qy: float
    qz: float
    qw: float
    stamp_sec: float


@dataclass
class MemoryEntry:
    stamp_sec: float
    pose: Pose2DState
    front_rgb: np.ndarray
    left_rgb: np.ndarray
    right_rgb: np.ndarray


@dataclass
class GridMeta:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    frame_id: str


def yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def shortest_angular_distance(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


def msg_stamp_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class InternalMemoryBank:
    def __init__(self, maxlen: int):
        self._buf: Deque[MemoryEntry] = deque(maxlen=max(10, int(maxlen)))

    def add(self, entry: MemoryEntry) -> None:
        self._buf.append(entry)

    def __len__(self) -> int:
        return len(self._buf)

    def latest(self) -> Optional[MemoryEntry]:
        if not self._buf:
            return None
        return self._buf[-1]

    def nearest_by_xy(self, x: float, y: float) -> Optional[MemoryEntry]:
        if not self._buf:
            return None
        best = None
        best_d2 = float("inf")
        for e in self._buf:
            d2 = (e.pose.x - x) ** 2 + (e.pose.y - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = e
        return best

    def spin_images_for_pose(self, pose: Pose2DState) -> Optional[List[np.ndarray]]:
        """Build 5-image list used by OVON-style slow prompt:
        [yaw+90, yaw+180, yaw-90, yaw(0), reference]
        We approximate panoramic images from historical front views.
        """
        if not self._buf:
            return None

        targets = [
            pose.yaw + math.pi / 2.0,
            pose.yaw + math.pi,
            pose.yaw - math.pi / 2.0,
            pose.yaw,
        ]
        entries = list(self._buf)

        out: List[np.ndarray] = []
        for tyaw in targets:
            best = min(entries, key=lambda e: shortest_angular_distance(e.pose.yaw, tyaw))
            out.append(best.front_rgb)

        # reference image: current front if available, else nearest yaw/front
        latest = self.latest()
        out.append(latest.front_rgb if latest is not None else out[-1])
        return out


class OvonSpinBankAdapter:
    """Adapter to keep qwen_utils.getresult-style bank interface."""

    def __init__(self, memory: InternalMemoryBank, pose: Pose2DState, state: OvonAgentState):
        self._memory = memory
        self._pose = pose
        self._state = state

    def get_spin_data(self):
        spin_images = self._memory.spin_images_for_pose(self._pose)
        if not spin_images:
            return [], []
        spin_states = [self._state for _ in range(len(spin_images))]
        return spin_images, spin_states


def ovon_getresult(
    qwen,
    processor,
    bank: OvonSpinBankAdapter,
    current_frontiers: Sequence[np.ndarray],
    decision_agent_state: OvonAgentState,
    instruction_text: str,
    object_category: str,
    max_new_tokens: int,
) -> Tuple[Optional[np.ndarray], bool, List[str]]:
    """OVON getresult-compatible model decision path (Habitat-free)."""
    ref_coord = decision_agent_state.position
    ref_quat_dict = {
        "x": decision_agent_state.rotation.x,
        "y": decision_agent_state.rotation.y,
        "z": decision_agent_state.rotation.z,
        "w": decision_agent_state.rotation.w,
    }

    all_prompt_images: List[np.ndarray] = []
    spin_images, _ = bank.get_spin_data()
    if len(spin_images) < 5:
        return None, False, []
    all_prompt_images.extend(spin_images[:5])

    spin_images_content = (
        "1: These four images show a 360-degree panoramic view around Observer's perspective,"
        "position is all [0.00,0.00], taken at 90-degree intervals: <image><image><image><image>"
    )
    main_images_content = "2: This is the reference image from Observer's perspective for all coordinates: <image>"

    local_to_global_map = {}
    frontier_parts = ["3: The coordinates of the explorable frontiers are: "]
    for frontier_coord in current_frontiers:
        local_coord = transform_to_local_frame(frontier_coord, ref_coord, ref_quat_dict)
        x, z = local_coord[0], local_coord[2]
        local_to_global_map[(x, z)] = frontier_coord
        frontier_parts.append(f"[{x:.2f}, {z:.2f}]")
    frontier_content = "".join(frontier_parts)

    text = instruction_text.strip() if instruction_text else ""
    if not text:
        text = random.choice(OBJ_GOAL_TEMPLATE).format(object_category)
    task_sentence = "instruction: " + text
    user_content = "\n".join([spin_images_content, main_images_content, frontier_content, task_sentence])

    current_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": all_prompt_images},
                {"type": "text", "text": user_content},
            ],
        }
    ]
    text = processor.apply_chat_template(current_messages, tokenize=False, add_generation_prompt=True)
    text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
    text = text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")

    image_inputs = process_ovon_vision(all_prompt_images)
    inputs = processor(
        text=text,
        images=image_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    inputs = inputs.to(qwen.device)

    # Deferred torch import to keep startup lightweight in rule mode.
    import torch  # pylint: disable=import-outside-toplevel

    with torch.no_grad():
        generated_ids = qwen.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        parsed_local_target, is_final_decision = parse_ovon_response(output_texts[0] if output_texts else "")

    if parsed_local_target is None:
        if current_frontiers:
            return np.asarray(current_frontiers[0], dtype=np.float32), False, output_texts
        return None, False, output_texts

    model_choice_coord = np.asarray(parsed_local_target, dtype=np.float32)
    min_dist = float("inf")
    best_match_global_coord = None

    if is_final_decision:
        model_choice_coord = np.insert(model_choice_coord, 1, 0.0)
        best_match_global_coord = transform_from_local_frame(model_choice_coord, ref_coord, ref_quat_dict)
    else:
        for local_key, global_coord_val in local_to_global_map.items():
            dist = np.linalg.norm(model_choice_coord - np.asarray(local_key, dtype=np.float32))
            if dist < min_dist:
                min_dist = dist
                best_match_global_coord = global_coord_val

    if best_match_global_coord is None:
        return None, False, output_texts
    return np.asarray(best_match_global_coord, dtype=np.float32), is_final_decision, output_texts


class HabitatAdapterNode(Node):
    def __init__(self, cfg: dict):
        super().__init__("habitat_adapter_node")
        self.cfg = cfg

        topics = cfg["topics"]
        params = cfg["params"]

        self.map_topic = str(topics["occupancy"])
        self.pose_topic = str(topics["pose2d"])
        self.front_topic = str(topics["cam_front"])
        self.left_topic = str(topics["cam_left"])
        self.right_topic = str(topics["cam_right"])
        self.frontiers_topic = str(topics["frontiers"])
        self.subgoal_topic = str(topics["subgoal"])
        self.instruction_topic = str(topics.get("instruction", "/omni/instruction"))

        self.decision_hz = float(params["decision_hz"])
        self.min_frontier_cells = int(params["min_frontier_cells"])
        self.max_frontiers = int(params["max_frontiers"])
        self.occ_threshold = int(params["occupied_threshold"])
        self.subgoal_min_dist_m = float(params["subgoal_min_dist_m"])
        self.goal_reach_radius_m = float(params["goal_reach_radius_m"])
        self.visited_quantization_m = float(params["visited_quantization_m"])
        self.subgoal_stale_sec = float(params["subgoal_stale_sec"])
        self.memory_maxlen = int(params["memory_maxlen"])
        self.memory_image_width = max(16, int(params.get("memory_image_width", 480)))
        self.memory_image_height = max(16, int(params.get("memory_image_height", 426)))

        self.decision_mode = str(params.get("decision_mode", "rule")).strip().lower()
        if self.decision_mode not in {"rule", "model"}:
            self.decision_mode = "rule"
        self.object_category = str(params.get("object_category", "target object")).strip()
        instruction_override = str(params.get("instruction_text", "")).strip()
        if instruction_override:
            self.runtime_instruction = instruction_override
        else:
            self.runtime_instruction = random.choice(OBJ_GOAL_TEMPLATE).format(self.object_category)
        self.model_path = str(params.get("model_path", "/workspace/OmniNav/OmniNav_Slowfast")).strip()
        self.model_min_pixels = int(params.get("model_min_pixels", 56 * 56))
        self.model_max_pixels = int(params.get("model_max_pixels", 4480 * 4480))
        self.model_dtype = str(params.get("model_dtype", "bfloat16")).strip()
        self.model_device_map = params.get("model_device_map", "auto")
        self.model_attn_implementation = str(params.get("model_attn_implementation", "flash_attention_2")).strip()
        self.model_max_new_tokens = int(params.get("model_max_new_tokens", 512))
        self.model_min_memory_images = int(params.get("model_min_memory_images", 5))

        self.map_data: Optional[np.ndarray] = None
        self.map_meta: Optional[GridMeta] = None
        self.latest_pose: Optional[Pose2DState] = None
        self.memory = InternalMemoryBank(self.memory_maxlen)

        self.visited_frontiers: set[Tuple[float, float]] = set()
        self.current_subgoal: Optional[Tuple[float, float]] = None
        self.current_subgoal_t: float = 0.0
        self.decision_count = 0

        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.frontiers_pub = self.create_publisher(PoseArray, self.frontiers_topic, pub_qos)
        self.subgoal_pub = self.create_publisher(PoseStamped, self.subgoal_topic, pub_qos)
        instruction_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.instruction_pub = self.create_publisher(String, self.instruction_topic, instruction_qos)

        self.create_subscription(OccupancyGrid, self.map_topic, self._map_cb, 10)
        self.create_subscription(PoseStamped, self.pose_topic, self._pose_cb, 20)

        # Internal bank sync: tri-view + pose2d
        self.front_sub = Subscriber(self, Image, self.front_topic, qos_profile=qos_profile_sensor_data)
        self.left_sub = Subscriber(self, Image, self.left_topic, qos_profile=qos_profile_sensor_data)
        self.right_sub = Subscriber(self, Image, self.right_topic, qos_profile=qos_profile_sensor_data)
        self.pose_sub = Subscriber(self, PoseStamped, self.pose_topic, qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer(
            [self.front_sub, self.left_sub, self.right_sub, self.pose_sub],
            queue_size=int(params["sync_queue_size"]),
            slop=float(params["sync_slop_sec"]),
        )
        self.sync.registerCallback(self._rgb_pose_cb)

        self.timer = self.create_timer(1.0 / max(0.2, self.decision_hz), self._decision_tick)

        self.model_backend_ready = False
        self.model_initialized = False
        self.model = None
        self.processor = None
        self.model_warned = False
        self.get_logger().info(
            "habitat_adapter_node started: "
            f"mode={self.decision_mode}, map={self.map_topic}, pose={self.pose_topic}, "
            f"cams=[{self.front_topic}, {self.left_topic}, {self.right_topic}], "
            f"model_path={self.model_path}, object={self.object_category}, "
            f"instruction_topic={self.instruction_topic}"
        )
        self.get_logger().info(f"instruction: {self.runtime_instruction}")
        self._publish_instruction()

    def _image_msg_to_rgb(self, msg: Image) -> Optional[np.ndarray]:
        try:
            data = np.array(msg.data, dtype=np.uint8)
            if msg.encoding in ("bgr8", "bgra8"):
                ch = 4 if msg.encoding == "bgra8" else 3
                img = data.reshape((msg.height, msg.width, ch))
                if ch == 4:
                    img = img[:, :, :3]
                return img[:, :, ::-1].copy()
            if msg.encoding in ("rgb8", "rgba8"):
                ch = 4 if msg.encoding == "rgba8" else 3
                img = data.reshape((msg.height, msg.width, ch))
                return img[:, :, :3].copy()
            # grayscale fallback
            img = data.reshape((msg.height, msg.width))
            return np.stack([img, img, img], axis=-1)
        except Exception:  # noqa: BLE001
            return None

    def _resize_memory_image(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return img
        if img.shape[1] == self.memory_image_width and img.shape[0] == self.memory_image_height:
            return img
        pil = PILImage.fromarray(img)
        pil = pil.resize((self.memory_image_width, self.memory_image_height), resample=PILImage.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)

    def _map_cb(self, msg: OccupancyGrid) -> None:
        arr = np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        self.map_data = arr
        self.map_meta = GridMeta(
            width=int(msg.info.width),
            height=int(msg.info.height),
            resolution=float(msg.info.resolution),
            origin_x=float(msg.info.origin.position.x),
            origin_y=float(msg.info.origin.position.y),
            frame_id=str(msg.header.frame_id),
        )

    def _pose_cb(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        yaw = yaw_from_quat_xyzw(q.x, q.y, q.z, q.w)
        self.latest_pose = Pose2DState(
            x=float(msg.pose.position.x),
            y=float(msg.pose.position.y),
            yaw=float(yaw),
            qx=float(q.x),
            qy=float(q.y),
            qz=float(q.z),
            qw=float(q.w),
            stamp_sec=msg_stamp_sec(msg.header.stamp),
        )

    def _rgb_pose_cb(self, front: Image, left: Image, right: Image, pose_msg: PoseStamped) -> None:
        f = self._image_msg_to_rgb(front)
        l = self._image_msg_to_rgb(left)
        r = self._image_msg_to_rgb(right)
        if f is None or l is None or r is None:
            return
        f = self._resize_memory_image(f)
        l = self._resize_memory_image(l)
        r = self._resize_memory_image(r)

        q = pose_msg.pose.orientation
        yaw = yaw_from_quat_xyzw(q.x, q.y, q.z, q.w)
        pose = Pose2DState(
            x=float(pose_msg.pose.position.x),
            y=float(pose_msg.pose.position.y),
            yaw=float(yaw),
            qx=float(q.x),
            qy=float(q.y),
            qz=float(q.z),
            qw=float(q.w),
            stamp_sec=msg_stamp_sec(pose_msg.header.stamp),
        )
        self.memory.add(
            MemoryEntry(
                stamp_sec=pose.stamp_sec,
                pose=pose,
                front_rgb=f,
                left_rgb=l,
                right_rgb=r,
            )
        )

    def _neighbor_unknown(self, unknown: np.ndarray) -> np.ndarray:
        out = np.zeros_like(unknown, dtype=bool)
        out[1:, :] |= unknown[:-1, :]
        out[:-1, :] |= unknown[1:, :]
        out[:, 1:] |= unknown[:, :-1]
        out[:, :-1] |= unknown[:, 1:]
        return out

    def _frontier_components_cells(self, frontier: np.ndarray) -> List[np.ndarray]:
        h, w = frontier.shape
        visited = np.zeros_like(frontier, dtype=bool)
        comps: List[np.ndarray] = []

        for r0 in range(h):
            for c0 in range(w):
                if not frontier[r0, c0] or visited[r0, c0]:
                    continue
                stack = [(r0, c0)]
                visited[r0, c0] = True
                cells: List[Tuple[int, int]] = []

                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            rr = r + dr
                            cc = c + dc
                            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                                continue
                            if visited[rr, cc] or not frontier[rr, cc]:
                                continue
                            visited[rr, cc] = True
                            stack.append((rr, cc))

                if len(cells) >= self.min_frontier_cells:
                    comps.append(np.array(cells, dtype=np.int32))
        return comps

    def _grid_to_world(self, row: float, col: float, meta: GridMeta) -> Tuple[float, float]:
        x = meta.origin_x + (col + 0.5) * meta.resolution
        y = meta.origin_y + (row + 0.5) * meta.resolution
        return x, y

    def _extract_frontiers_world(self) -> List[Tuple[float, float]]:
        if self.map_data is None or self.map_meta is None:
            return []

        occ = self.map_data
        unknown = occ < 0
        free = (occ >= 0) & (occ < self.occ_threshold)
        neigh_unk = self._neighbor_unknown(unknown)
        frontier_mask = free & neigh_unk

        comps = self._frontier_components_cells(frontier_mask)
        if not comps:
            return []

        points: List[Tuple[float, float]] = []
        robot_rc: Optional[Tuple[int, int]] = None
        if self.latest_pose is not None:
            rr = int((self.latest_pose.y - self.map_meta.origin_y) / self.map_meta.resolution)
            rc = int((self.latest_pose.x - self.map_meta.origin_x) / self.map_meta.resolution)
            robot_rc = (rr, rc)

        for comp in comps:
            if robot_rc is None:
                r = float(np.mean(comp[:, 0]))
                c = float(np.mean(comp[:, 1]))
            else:
                dr = comp[:, 0].astype(np.float32) - float(robot_rc[0])
                dc = comp[:, 1].astype(np.float32) - float(robot_rc[1])
                idx = int(np.argmin(dr * dr + dc * dc))
                r = float(comp[idx, 0])
                c = float(comp[idx, 1])
            points.append(self._grid_to_world(r, c, self.map_meta))

        if self.latest_pose is not None:
            px, py = self.latest_pose.x, self.latest_pose.y
            points.sort(key=lambda p: (p[0] - px) ** 2 + (p[1] - py) ** 2)

        return points[: self.max_frontiers]

    def _quantize(self, x: float, y: float) -> Tuple[float, float]:
        q = max(1e-3, self.visited_quantization_m)
        return (round(x / q) * q, round(y / q) * q)

    def _publish_frontiers(self, pts: Sequence[Tuple[float, float]]) -> None:
        if self.map_meta is None:
            return
        msg = PoseArray()
        msg.header.frame_id = self.map_meta.frame_id or "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        poses: List[Pose] = []
        for x, y in pts:
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = 0.0
            p.orientation.w = 1.0
            poses.append(p)
        msg.poses = poses
        self.frontiers_pub.publish(msg)

    def _publish_subgoal(self, x: float, y: float) -> None:
        if self.map_meta is None:
            return
        msg = PoseStamped()
        msg.header.frame_id = self.map_meta.frame_id or "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.subgoal_pub.publish(msg)

    def _publish_instruction(self) -> None:
        if not self.runtime_instruction:
            return
        msg = String()
        msg.data = self.runtime_instruction
        self.instruction_pub.publish(msg)

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _select_subgoal_rule(self, frontiers: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not frontiers or self.latest_pose is None:
            return None
        px, py = self.latest_pose.x, self.latest_pose.y

        candidates = []
        for fx, fy in frontiers:
            d = math.hypot(fx - px, fy - py)
            if d < self.subgoal_min_dist_m:
                continue
            if self._quantize(fx, fy) in self.visited_frontiers:
                continue
            candidates.append((d, fx, fy))

        if not candidates:
            self.visited_frontiers.clear()
            for fx, fy in frontiers:
                d = math.hypot(fx - px, fy - py)
                if d >= self.subgoal_min_dist_m:
                    candidates.append((d, fx, fy))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        return (candidates[0][1], candidates[0][2])

    def _to_ovon_agent_state(self, pose: Pose2DState) -> OvonAgentState:
        # OVON slow logic expects Habitat-style [x, y, z] with y as up-axis.
        # ROS map pose is [x, y] on planar world, so map-y is mapped to Habitat-z.
        half = 0.5 * pose.yaw
        habitat_like_rot = OvonRotation(
            x=0.0,
            y=math.sin(half),
            z=0.0,
            w=math.cos(half),
        )
        return OvonAgentState(
            position=np.asarray([pose.x, 0.0, pose.y], dtype=np.float32),
            rotation=habitat_like_rot,
        )

    def _ensure_model_backend(self) -> bool:
        if self.model_initialized:
            return self.model_backend_ready
        self.model_initialized = True

        try:
            import torch  # pylint: disable=import-outside-toplevel
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # pylint: disable=import-outside-toplevel

            dtype = getattr(torch, self.model_dtype, torch.bfloat16)
            model_kwargs = {
                "torch_dtype": dtype,
                "device_map": self.model_device_map,
            }
            if self.model_attn_implementation:
                model_kwargs["attn_implementation"] = self.model_attn_implementation

            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    **model_kwargs,
                )
            except Exception:
                model_kwargs.pop("attn_implementation", None)
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    **model_kwargs,
                )

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=self.model_min_pixels,
                max_pixels=self.model_max_pixels,
            )
            self.model_backend_ready = True
            self.get_logger().info(
                "model backend initialized: "
                f"path={self.model_path}, dtype={self.model_dtype}, device_map={self.model_device_map}"
            )
        except Exception as exc:  # noqa: BLE001
            self.model_backend_ready = False
            self.get_logger().error(f"model backend init failed: {exc!r}")
        return self.model_backend_ready

    def _select_subgoal_model(self, frontiers: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if self.latest_pose is None:
            return None
        if len(self.memory) < self.model_min_memory_images:
            return None
        if not self._ensure_model_backend():
            return None
        if self.model is None or self.processor is None:
            return None

        state = self._to_ovon_agent_state(self.latest_pose)
        frontiers_ovon = [np.asarray([fx, 0.0, fy], dtype=np.float32) for fx, fy in frontiers]
        bank = OvonSpinBankAdapter(self.memory, self.latest_pose, state)

        target_position, is_final_decision, output_texts = ovon_getresult(
            qwen=self.model,
            processor=self.processor,
            bank=bank,
            current_frontiers=frontiers_ovon,
            decision_agent_state=state,
            instruction_text=self.runtime_instruction,
            object_category=self.object_category,
            max_new_tokens=self.model_max_new_tokens,
        )
        if target_position is None:
            return None

        tx = float(target_position[0])
        ty = float(target_position[2]) if target_position.shape[0] >= 3 else float(target_position[1])
        if not np.isfinite(tx) or not np.isfinite(ty):
            return None

        if not is_final_decision and self._quantize(tx, ty) in self.visited_frontiers:
            return None

        if output_texts and (self.decision_count % 5 == 0):
            preview = output_texts[0].replace("\n", " ").strip()
            if len(preview) > 180:
                preview = preview[:180] + "..."
            self.get_logger().info(f"model output preview: {preview}")

        return (tx, ty)

    def _select_subgoal(self, frontiers: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if self.decision_mode != "model":
            return self._select_subgoal_rule(frontiers)

        subgoal = self._select_subgoal_model(frontiers)
        if subgoal is not None:
            return subgoal
        if not self.model_warned:
            self.get_logger().warn("model mode fallback -> rule mode (backend/data not ready)")
            self.model_warned = True
        return self._select_subgoal_rule(frontiers)

    def _decision_tick(self) -> None:
        # Transient-local publisher usually latches this, but periodic publish keeps consumers robust.
        self._publish_instruction()
        if self.map_data is None or self.map_meta is None or self.latest_pose is None:
            return

        frontiers = self._extract_frontiers_world()
        self._publish_frontiers(frontiers)
        if not frontiers:
            return

        now = time.time()
        robot_xy = (self.latest_pose.x, self.latest_pose.y)
        if self.current_subgoal is not None:
            if self._distance(robot_xy, self.current_subgoal) <= self.goal_reach_radius_m:
                self.visited_frontiers.add(self._quantize(*self.current_subgoal))
                self.current_subgoal = None
            elif now - self.current_subgoal_t < self.subgoal_stale_sec:
                self._publish_subgoal(*self.current_subgoal)
                return

        subgoal = self._select_subgoal(frontiers)
        if subgoal is None:
            return

        self.current_subgoal = subgoal
        self.current_subgoal_t = now
        self.decision_count += 1
        self._publish_subgoal(*subgoal)
        if self.decision_count % 5 == 0:
            self.get_logger().info(
                f"decision#{self.decision_count}: subgoal=({subgoal[0]:.2f}, {subgoal[1]:.2f}), "
                f"frontiers={len(frontiers)}, memory={len(self.memory)}"
            )


def parse_args() -> argparse.Namespace:
    default_cfg = Path(__file__).resolve().parents[1] / "config" / "habitat_adapter.yaml"
    parser = argparse.ArgumentParser(description="Phase3 Habitat Adapter Node")
    parser.add_argument("--config", default=str(default_cfg), help="Path to habitat_adapter.yaml")
    parser.add_argument(
        "--object-category",
        default=None,
        help="Override params.object_category from config",
    )
    parser.add_argument(
        "--instruction-text",
        default=None,
        help="Override params.instruction_text from config",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if not isinstance(cfg, dict):
        cfg = {}
    params = cfg.get("params")
    if not isinstance(params, dict):
        params = {}
        cfg["params"] = params
    if args.object_category is not None:
        params["object_category"] = args.object_category
    if args.instruction_text is not None:
        params["instruction_text"] = args.instruction_text

    rclpy.init()
    node = HabitatAdapterNode(cfg)
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
