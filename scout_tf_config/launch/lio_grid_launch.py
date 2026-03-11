#!/usr/bin/env python3
"""Launch lio_grid_node.py with config."""

import argparse
from pathlib import Path

from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess


def build_launch_description(config_path: str) -> LaunchDescription:
    script = Path(__file__).resolve().parents[1] / "scripts" / "lio_grid_node.py"
    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=["python3", str(script), "--config", str(config_path)],
                output="screen",
            )
        ]
    )


def parse_args() -> argparse.Namespace:
    default_cfg = Path(__file__).resolve().parents[1] / "config" / "lio_grid.yaml"
    parser = argparse.ArgumentParser(description="Launch lio_grid_node")
    parser.add_argument("--config", default=str(default_cfg), help="Path to lio_grid.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    ld = build_launch_description(str(cfg_path))
    ls = LaunchService(argv=[])
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == "__main__":
    raise SystemExit(main())
