#!/usr/bin/env python3
"""Launch habitat_adapter_node.py with config."""

import argparse
from pathlib import Path

from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess


def build_launch_description(
    config_path: str,
    object_category: str | None = None,
    instruction_text: str | None = None,
) -> LaunchDescription:
    script = Path(__file__).resolve().parents[1] / "scripts" / "habitat_adapter_node.py"
    cmd = ["python3", str(script), "--config", str(config_path)]
    if object_category is not None:
        cmd.extend(["--object-category", object_category])
    if instruction_text is not None:
        cmd.extend(["--instruction-text", instruction_text])
    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=cmd,
                output="screen",
            )
        ]
    )


def parse_args() -> argparse.Namespace:
    default_cfg = Path(__file__).resolve().parents[1] / "config" / "habitat_adapter.yaml"
    parser = argparse.ArgumentParser(description="Launch habitat_adapter_node")
    parser.add_argument("--config", default=str(default_cfg), help="Path to habitat_adapter.yaml")
    parser.add_argument(
        "--object-category",
        default=None,
        help="Override params.object_category for habitat_adapter_node",
    )
    parser.add_argument(
        "--instruction-text",
        default=None,
        help="Override params.instruction_text for habitat_adapter_node",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Path(args.config)
    if not cfg.exists():
        raise FileNotFoundError(f"Config not found: {cfg}")
    ls = LaunchService(argv=[])
    ls.include_launch_description(
        build_launch_description(
            str(cfg),
            object_category=args.object_category,
            instruction_text=args.instruction_text,
        )
    )
    return ls.run()


if __name__ == "__main__":
    raise SystemExit(main())
