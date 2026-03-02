#!/usr/bin/env python3
"""Launch static TF publishers + /Odometry TF bridge for Phase 2.

Usage:
  python3 scout_tf_config/launch/sensor_tf_launch.py
  python3 scout_tf_config/launch/sensor_tf_launch.py --config scout_tf_config/config/sensor_tf.yaml
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List

import yaml
from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> List[float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qx, qy, qz, qw]


def static_tf_action(parent: str, child: str, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float) -> ExecuteProcess:
    return ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "tf2_ros",
            "static_transform_publisher",
            str(x),
            str(y),
            str(z),
            str(qx),
            str(qy),
            str(qz),
            str(qw),
            parent,
            child,
        ],
        output="screen",
    )


def build_launch_description(cfg: Dict, run_verify: bool) -> LaunchDescription:
    frames = cfg["frames"]
    base_to_lidar = cfg["base_to_lidar"]
    l2c = cfg["lidar_to_camera"]
    bridge = cfg["bridge"]
    topics = cfg["topics"]
    map_to_cloud = cfg.get("map_to_cloud_static_tf", {})

    qx, qy, qz, qw = rpy_to_quat(
        float(base_to_lidar["roll"]),
        float(base_to_lidar["pitch"]),
        float(base_to_lidar["yaw"]),
    )

    actions = [
        static_tf_action(
            frames["base_link"],
            frames["lidar"],
            float(base_to_lidar["x"]),
            float(base_to_lidar["y"]),
            float(base_to_lidar["z"]),
            qx,
            qy,
            qz,
            qw,
        ),
        static_tf_action(
            frames["lidar"],
            frames["cam_front"],
            float(l2c["front"]["x"]),
            float(l2c["front"]["y"]),
            float(l2c["front"]["z"]),
            float(l2c["front"]["qx"]),
            float(l2c["front"]["qy"]),
            float(l2c["front"]["qz"]),
            float(l2c["front"]["qw"]),
        ),
        static_tf_action(
            frames["lidar"],
            frames["cam_left"],
            float(l2c["left"]["x"]),
            float(l2c["left"]["y"]),
            float(l2c["left"]["z"]),
            float(l2c["left"]["qx"]),
            float(l2c["left"]["qy"]),
            float(l2c["left"]["qz"]),
            float(l2c["left"]["qw"]),
        ),
        static_tf_action(
            frames["lidar"],
            frames["cam_right"],
            float(l2c["right"]["x"]),
            float(l2c["right"]["y"]),
            float(l2c["right"]["z"]),
            float(l2c["right"]["qx"]),
            float(l2c["right"]["qy"]),
            float(l2c["right"]["qz"]),
            float(l2c["right"]["qw"]),
        ),
    ]

    if bool(map_to_cloud.get("enable", False)):
        actions.append(
            static_tf_action(
                frames["map"],
                frames["cloud"],
                float(map_to_cloud.get("x", 0.0)),
                float(map_to_cloud.get("y", 0.0)),
                float(map_to_cloud.get("z", 0.0)),
                float(map_to_cloud.get("qx", 0.0)),
                float(map_to_cloud.get("qy", 0.0)),
                float(map_to_cloud.get("qz", 0.0)),
                float(map_to_cloud.get("qw", 1.0)),
            )
        )

    if bool(bridge.get("use_odom_tf_bridge", True)):
        bridge_script = Path(__file__).resolve().parents[1] / "scripts" / "odom_tf_bridge.py"
        actions.append(
            ExecuteProcess(
                cmd=[
                    "python3",
                    str(bridge_script),
                    "--odom-topic",
                    str(bridge["source_topic"]),
                    "--parent-frame",
                    str(bridge["parent_frame"]),
                    "--child-frame",
                    str(bridge["child_frame"]),
                ],
                output="screen",
            )
        )

    if run_verify:
        verify_script = Path(__file__).resolve().parents[1] / "scripts" / "verify_tf.py"
        actions.append(
            ExecuteProcess(
                cmd=[
                    "python3",
                    str(verify_script),
                    "--odom-topic",
                    str(topics["odometry"]),
                    "--cam-front-topic",
                    str(topics["cam_front"]),
                    "--cam-left-topic",
                    str(topics["cam_left"]),
                    "--cam-right-topic",
                    str(topics["cam_right"]),
                    "--map-frame",
                    str(frames["map"]),
                    "--base-frame",
                    str(frames["base_link"]),
                    "--lidar-frame",
                    str(frames["lidar"]),
                    "--cam-front-frame",
                    str(frames["cam_front"]),
                    "--cam-left-frame",
                    str(frames["cam_left"]),
                    "--cam-right-frame",
                    str(frames["cam_right"]),
                    "--cloud-frame",
                    str(frames["cloud"]),
                ],
                output="screen",
            )
        )

    return LaunchDescription(actions)


def parse_args() -> argparse.Namespace:
    default_cfg = Path(__file__).resolve().parents[1] / "config" / "sensor_tf.yaml"
    parser = argparse.ArgumentParser(description="Phase 2 TF launch")
    parser.add_argument("--config", default=str(default_cfg), help="Path to sensor_tf.yaml")
    parser.add_argument("--run-verify", action="store_true", help="Run verify_tf.py together")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ld = build_launch_description(cfg, args.run_verify)
    ls = LaunchService(argv=[])
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == "__main__":
    raise SystemExit(main())
