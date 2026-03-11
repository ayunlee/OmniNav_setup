#!/usr/bin/env python3
"""Build 2D occupancy grid from FAST-LIO /cloud_registered and /Odometry.

WP-A (Phase 3) implementation notes:
- Input: /cloud_registered (PointCloud2), /Odometry
- Sync: message_filters ApproximateTime
- Output: /omni/occupancy (OccupancyGrid), /omni/pose2d (PoseStamped)
- Core logic follows scripts/gridmapper.cpp:
  - dynamic map growth
  - ray-casting (Bresenham) for free cells
  - log-odds occupancy update
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

try:
    import cv2  # optional, for inflation dilation
except Exception:  # noqa: BLE001
    cv2 = None


@dataclass
class GridParams:
    resolution: float
    initial_rows: int
    initial_cols: int
    origin_x: float
    origin_y: float
    grow_padding_cells: int
    free_log_odds: float
    occupied_log_odds: float
    clamp_min: float
    clamp_max: float
    publish_hz: float
    occupied_threshold: float
    inflation_radius_m: float


class DynamicLogOddsGrid:
    def __init__(self, p: GridParams) -> None:
        self.p = p
        self.rows = int(p.initial_rows)
        self.cols = int(p.initial_cols)
        self.origin_x = float(p.origin_x)
        self.origin_y = float(p.origin_y)
        self.log_odds = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.observed = np.zeros((self.rows, self.cols), dtype=bool)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        col = int(math.floor((x - self.origin_x) / self.p.resolution))
        row = int(math.floor((y - self.origin_y) / self.p.resolution))
        return row, col

    def ensure_bounds(self, min_row: int, min_col: int, max_row: int, max_col: int) -> None:
        pad = int(self.p.grow_padding_cells)
        pad_top = (-min_row + pad) if min_row < 0 else 0
        pad_left = (-min_col + pad) if min_col < 0 else 0
        pad_bottom = (max_row - self.rows + 1 + pad) if max_row >= self.rows else 0
        pad_right = (max_col - self.cols + 1 + pad) if max_col >= self.cols else 0

        if pad_top == 0 and pad_left == 0 and pad_bottom == 0 and pad_right == 0:
            return

        self.log_odds = np.pad(
            self.log_odds,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0.0,
        )
        self.observed = np.pad(
            self.observed,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=False,
        )
        self.rows, self.cols = self.log_odds.shape
        self.origin_x -= pad_left * self.p.resolution
        self.origin_y -= pad_top * self.p.resolution

    def update(self, row: int, col: int, delta: float) -> None:
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return
        self.observed[row, col] = True
        self.log_odds[row, col] = np.clip(
            self.log_odds[row, col] + float(delta),
            self.p.clamp_min,
            self.p.clamp_max,
        )

    @staticmethod
    def bresenham(x0: int, y0: int, x1: int, y1: int) -> Iterable[Tuple[int, int]]:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            yield x, y
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _inflate(self, grid_data: np.ndarray) -> np.ndarray:
        radius_cells = int(math.ceil(self.p.inflation_radius_m / self.p.resolution))
        if radius_cells <= 0 or cv2 is None:
            return grid_data

        occ_thr = int(round(self.p.occupied_threshold * 100.0))
        occ_mask = (grid_data >= occ_thr).astype(np.uint8)
        if np.count_nonzero(occ_mask) == 0:
            return grid_data

        kernel_size = 2 * radius_cells + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        occ_dilated = cv2.dilate(occ_mask, kernel, iterations=1)
        out = grid_data.copy()
        out[occ_dilated > 0] = 100
        return out

    def to_occupancy_data(self) -> np.ndarray:
        # unknown by default
        data = np.full((self.rows, self.cols), -1, dtype=np.int16)
        if np.any(self.observed):
            probs = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
            obs_vals = np.rint(np.clip(probs * 100.0, 0.0, 100.0)).astype(np.int16)
            data[self.observed] = obs_vals[self.observed]

        data = self._inflate(data)
        return data.astype(np.int8)


class LioGridNode(Node):
    def __init__(self, cfg: dict) -> None:
        super().__init__("lio_grid_node")
        self.cfg = cfg

        topics = cfg["topics"]
        frames = cfg["frames"]
        sync = cfg["sync"]
        filt = cfg["filter"]
        grid_cfg = cfg["grid"]

        self.map_frame = str(frames["map"])
        self.cloud_topic = str(topics["cloud"])
        self.odom_topic = str(topics["odometry"])
        self.occupancy_topic = str(topics["occupancy"])
        self.pose2d_topic = str(topics["pose2d"])

        self.use_relative_height = bool(filt["use_relative_height"])
        self.slice_z_min = float(filt["slice_z_min"])
        self.slice_z_max = float(filt["slice_z_max"])
        self.point_stride = max(1, int(filt["point_stride"]))
        self.max_points_per_scan = int(filt["max_points_per_scan"])

        p = GridParams(
            resolution=float(grid_cfg["resolution"]),
            initial_rows=int(grid_cfg["initial_rows"]),
            initial_cols=int(grid_cfg["initial_cols"]),
            origin_x=float(grid_cfg["origin_x"]),
            origin_y=float(grid_cfg["origin_y"]),
            grow_padding_cells=int(grid_cfg["grow_padding_cells"]),
            free_log_odds=float(grid_cfg["free_log_odds"]),
            occupied_log_odds=float(grid_cfg["occupied_log_odds"]),
            clamp_min=float(grid_cfg["clamp_min"]),
            clamp_max=float(grid_cfg["clamp_max"]),
            publish_hz=float(grid_cfg["publish_hz"]),
            occupied_threshold=float(grid_cfg["occupied_threshold"]),
            inflation_radius_m=float(grid_cfg["inflation_radius_m"]),
        )
        self.grid = DynamicLogOddsGrid(p)
        self.publish_period_ns = int(1e9 / max(0.1, p.publish_hz))
        self.last_pub_ns = 0

        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.occ_pub = self.create_publisher(OccupancyGrid, self.occupancy_topic, pub_qos)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose2d_topic, pub_qos)

        self.cloud_sub = Subscriber(
            self,
            PointCloud2,
            self.cloud_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.odom_sub = Subscriber(
            self,
            Odometry,
            self.odom_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.cloud_sub, self.odom_sub],
            queue_size=int(sync["queue_size"]),
            slop=float(sync["slop_sec"]),
        )
        self.sync.registerCallback(self._synced_cb)

        self.processed_scans = 0
        self.get_logger().info(
            "lio_grid_node started: "
            f"cloud={self.cloud_topic}, odom={self.odom_topic} -> "
            f"occupancy={self.occupancy_topic}, pose2d={self.pose2d_topic}"
        )

    def _iter_xyz(self, cloud_msg: PointCloud2) -> Iterable[Tuple[float, float, float]]:
        for point in point_cloud2.read_points(
            cloud_msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        ):
            yield float(point[0]), float(point[1]), float(point[2])

    def _publish_pose2d(self, odom_msg: Odometry) -> None:
        pose2d = PoseStamped()
        pose2d.header.stamp = odom_msg.header.stamp
        pose2d.header.frame_id = self.map_frame
        pose2d.pose.position.x = float(odom_msg.pose.pose.position.x)
        pose2d.pose.position.y = float(odom_msg.pose.pose.position.y)
        pose2d.pose.position.z = 0.0
        pose2d.pose.orientation = odom_msg.pose.pose.orientation
        self.pose_pub.publish(pose2d)

    def _should_publish_map(self) -> bool:
        now_ns = self.get_clock().now().nanoseconds
        if self.last_pub_ns == 0 or now_ns - self.last_pub_ns >= self.publish_period_ns:
            self.last_pub_ns = now_ns
            return True
        return False

    def _make_occ_msg(self, stamp) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.info.resolution = float(self.grid.p.resolution)
        msg.info.width = int(self.grid.cols)
        msg.info.height = int(self.grid.rows)
        msg.info.origin = Pose()
        msg.info.origin.position.x = float(self.grid.origin_x)
        msg.info.origin.position.y = float(self.grid.origin_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0

        data = self.grid.to_occupancy_data()
        msg.data = data.reshape(-1).tolist()
        return msg

    def _synced_cb(self, cloud_msg: PointCloud2, odom_msg: Odometry) -> None:
        self._publish_pose2d(odom_msg)

        robot_x = float(odom_msg.pose.pose.position.x)
        robot_y = float(odom_msg.pose.pose.position.y)
        robot_z = float(odom_msg.pose.pose.position.z)

        filtered_xy: List[Tuple[float, float]] = []
        stride_count = 0
        for x, y, z in self._iter_xyz(cloud_msg):
            stride_count += 1
            if (stride_count % self.point_stride) != 0:
                continue
            z_eval = (z - robot_z) if self.use_relative_height else z
            if z_eval < self.slice_z_min or z_eval > self.slice_z_max:
                continue
            filtered_xy.append((x, y))
            if self.max_points_per_scan > 0 and len(filtered_xy) >= self.max_points_per_scan:
                break

        if not filtered_xy:
            if self._should_publish_map():
                self.occ_pub.publish(self._make_occ_msg(cloud_msg.header.stamp))
            return

        min_row, min_col = self.grid.world_to_grid(robot_x, robot_y)
        max_row, max_col = min_row, min_col
        for x, y in filtered_xy:
            row, col = self.grid.world_to_grid(x, y)
            min_row = min(min_row, row)
            min_col = min(min_col, col)
            max_row = max(max_row, row)
            max_col = max(max_col, col)

        self.grid.ensure_bounds(min_row, min_col, max_row, max_col)
        robot_row, robot_col = self.grid.world_to_grid(robot_x, robot_y)

        for x, y in filtered_xy:
            row, col = self.grid.world_to_grid(x, y)
            for rr, cc in self.grid.bresenham(robot_row, robot_col, row, col):
                self.grid.update(rr, cc, self.grid.p.free_log_odds)
            self.grid.update(row, col, self.grid.p.occupied_log_odds)

        self.processed_scans += 1
        if self._should_publish_map():
            occ_msg = self._make_occ_msg(cloud_msg.header.stamp)
            self.occ_pub.publish(occ_msg)
            if self.processed_scans % 10 == 0:
                self.get_logger().info(
                    "published map "
                    f"{self.grid.rows}x{self.grid.cols}, "
                    f"filtered_points={len(filtered_xy)}"
                )


def parse_args() -> argparse.Namespace:
    default_cfg = Path(__file__).resolve().parents[1] / "config" / "lio_grid.yaml"
    parser = argparse.ArgumentParser(description="FAST-LIO -> OccupancyGrid node")
    parser.add_argument("--config", default=str(default_cfg), help="Path to lio_grid.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rclpy.init()
    node = LioGridNode(cfg)
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
