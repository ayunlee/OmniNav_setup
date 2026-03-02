#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import select
import tty
import termios
import threading

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge

bridge = CvBridge()


def msg_to_cv2(msg, encoding="passthrough"):
    return bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)


class CaptureNode(Node):
    def __init__(self):
        super().__init__('capture_node')

        # --- camera selection from arg ---
        self.declare_parameter('camera', 'front')
        self.camera = self.get_parameter('camera').get_parameter_value().string_value
        cam_topic = f'/cam_{self.camera}/color/image_raw'

        # --- directories ---
        self.save_dir = "./capture"
        self.color_dir = os.path.join(self.save_dir, f"color_{self.camera}")
        self.livox_dir = os.path.join(self.save_dir, "livox")
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.livox_dir, exist_ok=True)

        # --- resume from last saved number ---
        self.cnt = self._get_next_count()

        self.last_color = None
        self.last_livox = None

        # --- subscribers ---
        self.create_subscription(
            PointCloud2, '/livox/lidar', self.livox_points_cb, 10)

        self.create_subscription(
            Image, cam_topic, self.color_cb, 10)

        # --- keyboard listener ---
        self.keyboard_thread = threading.Thread(target=self.listen_keyboard, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info(f"Camera: {cam_topic} | Starting from #{self.cnt} | Press 'c' to capture")

    def _get_next_count(self):
        """Find the highest existing number in livox/ and color_*/ dirs and return next."""
        max_num = 0
        for d in [self.color_dir, self.livox_dir]:
            if not os.path.exists(d):
                continue
            for f in os.listdir(d):
                name, _ = os.path.splitext(f)
                if name.isdigit():
                    max_num = max(max_num, int(name))
        return max_num + 1

        # --- keyboard listener ---
        self.keyboard_thread = threading.Thread(target=self.listen_keyboard, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info("Capture node ready — press 'c' to capture")

    # ---- callbacks ----
    def livox_points_cb(self, msg: PointCloud2):
        gen = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        pts_structured = np.array(list(gen))
        points = np.column_stack([
            pts_structured['x'].astype(np.float32),
            pts_structured['y'].astype(np.float32),
            pts_structured['z'].astype(np.float32),
        ])
        self.last_livox = points

    def color_cb(self, msg: Image):
        self.last_color = msg_to_cv2(msg)

    # ---- keyboard ----
    def listen_keyboard(self):
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        try:
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 'c':
                        self.save_data()
                    elif key.lower() == 'q':
                        self.get_logger().info("Quit requested")
                        rclpy.shutdown()
                        break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    # ---- PCD writer (no open3d needed) ----
    @staticmethod
    def write_pcd(path: str, points: np.ndarray):
        """Write xyz points to binary PCD file."""
        pts = points.astype(np.float32)
        n = len(pts)
        header = (
            f"# .PCD v0.7 - Point Cloud Data file format\n"
            f"VERSION 0.7\n"
            f"FIELDS x y z\n"
            f"SIZE 4 4 4\n"
            f"TYPE F F F\n"
            f"COUNT 1 1 1\n"
            f"WIDTH {n}\n"
            f"HEIGHT 1\n"
            f"VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {n}\n"
            f"DATA binary\n"
        )
        with open(path, 'wb') as f:
            f.write(header.encode('ascii'))
            f.write(pts.tobytes())

    # ---- save ----
    def save_data(self):
        self.get_logger().info(f"[Capture] Saving data #{self.cnt}")

        # color
        if self.last_color is not None:
            img_path = os.path.join(self.color_dir, f"{self.cnt}.png")
            rgb_image = cv2.cvtColor(self.last_color, cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_path, rgb_image)
            self.get_logger().info(f"  Saved image  : {img_path}")
        else:
            self.get_logger().warn("  Image is None")

        # livox
        if self.last_livox is not None:
            pcd_path = os.path.join(self.livox_dir, f"{self.cnt}.pcd")
            self.write_pcd(pcd_path, self.last_livox)
            self.get_logger().info(f"  Saved livox  : {pcd_path}")
        else:
            self.get_logger().warn("  Livox points is None")

        self.cnt += 1


def main(args=None):
    rclpy.init(args=args)
    node = CaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
