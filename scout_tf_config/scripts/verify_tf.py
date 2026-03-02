#!/usr/bin/env python3
"""Phase 2 verification script for TF chain and timestamp consistency."""

import argparse
from typing import Dict, Optional, Tuple

import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener


def stamp_to_sec(stamp: TimeMsg) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class Phase2Verifier(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("phase2_tf_verifier")
        self.args = args
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last_odom_stamp: Optional[float] = None
        self.last_cam_stamps: Dict[str, Optional[float]] = {
            "front": None,
            "left": None,
            "right": None,
        }

        self.create_subscription(Odometry, args.odom_topic, self._odom_cb, 20)
        self.create_subscription(Image, args.cam_front_topic, self._front_cb, 10)
        self.create_subscription(Image, args.cam_left_topic, self._left_cb, 10)
        self.create_subscription(Image, args.cam_right_topic, self._right_cb, 10)

        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(0.5, self._run_checks)

    def _odom_cb(self, msg: Odometry) -> None:
        self.last_odom_stamp = stamp_to_sec(msg.header.stamp)

    def _front_cb(self, msg: Image) -> None:
        self.last_cam_stamps["front"] = stamp_to_sec(msg.header.stamp)

    def _left_cb(self, msg: Image) -> None:
        self.last_cam_stamps["left"] = stamp_to_sec(msg.header.stamp)

    def _right_cb(self, msg: Image) -> None:
        self.last_cam_stamps["right"] = stamp_to_sec(msg.header.stamp)

    def _check_tf_pair(self, parent: str, child: str) -> Tuple[bool, str]:
        timeout = Duration(seconds=self.args.tf_timeout_sec)
        ok = self.tf_buffer.can_transform(parent, child, rclpy.time.Time(), timeout)
        if not ok:
            return False, f"TF lookup failed: {parent} -> {child}"
        try:
            self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time(), timeout)
            return True, f"TF OK: {parent} -> {child}"
        except Exception as exc:  # noqa: BLE001
            return False, f"TF exception {parent}->{child}: {exc}"

    def _run_checks(self) -> None:
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed < self.args.warmup_sec:
            return

        checks = [
            (self.args.map_frame, self.args.base_frame),
            (self.args.map_frame, self.args.cloud_frame),
            (self.args.base_frame, self.args.lidar_frame),
            (self.args.lidar_frame, self.args.cam_front_frame),
            (self.args.lidar_frame, self.args.cam_left_frame),
            (self.args.lidar_frame, self.args.cam_right_frame),
            (self.args.map_frame, self.args.cam_front_frame),
        ]

        tf_ok = True
        for parent, child in checks:
            ok, message = self._check_tf_pair(parent, child)
            tf_ok = tf_ok and ok
            self.get_logger().info(message)

        ts_ok = True
        if self.last_odom_stamp is None:
            ts_ok = False
            self.get_logger().error("No /Odometry received")
        else:
            for key, cam_stamp in self.last_cam_stamps.items():
                if cam_stamp is None:
                    ts_ok = False
                    self.get_logger().error(f"No camera stamp received: {key}")
                    continue
                dt = abs(self.last_odom_stamp - cam_stamp)
                self.get_logger().info(f"stamp delta odom-{key}: {dt:.3f}s")
                if dt > self.args.max_stamp_delta_sec:
                    ts_ok = False

        if tf_ok and ts_ok:
            self.get_logger().info("Phase 2 verification PASSED")
            raise SystemExit(0)

        self.get_logger().error("Phase 2 verification FAILED")
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify TF chain and timestamp offsets")
    parser.add_argument("--warmup-sec", type=float, default=5.0)
    parser.add_argument("--tf-timeout-sec", type=float, default=1.0)
    parser.add_argument("--max-stamp-delta-sec", type=float, default=0.10)

    parser.add_argument("--odom-topic", default="/Odometry")
    parser.add_argument("--cam-front-topic", default="/cam_front/color/image_raw")
    parser.add_argument("--cam-left-topic", default="/cam_left/color/image_raw")
    parser.add_argument("--cam-right-topic", default="/cam_right/color/image_raw")

    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--cloud-frame", default="camera_init")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--lidar-frame", default="lidar_frame")
    parser.add_argument("--cam-front-frame", default="cam_front_color_optical_frame")
    parser.add_argument("--cam-left-frame", default="cam_left_color_optical_frame")
    parser.add_argument("--cam-right-frame", default="cam_right_color_optical_frame")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = Phase2Verifier(args)
    exit_code = 1
    try:
        rclpy.spin(node)
    except SystemExit as exc:
        exit_code = int(exc.code)
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
