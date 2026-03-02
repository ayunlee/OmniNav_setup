#!/usr/bin/env python3
"""Bridge FAST-LIO /Odometry to map->base_link TF using non-zero message timestamps."""

import argparse

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class OdomTfBridge(Node):
    def __init__(self, odom_topic: str, parent_frame: str, child_frame: str) -> None:
        super().__init__("odom_tf_bridge")
        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self.tf_broadcaster = TransformBroadcaster(self)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self.sub = self.create_subscription(Odometry, odom_topic, self._odom_cb, qos)
        self.last_log_time = self.get_clock().now()
        self.msg_count = 0
        self.get_logger().info(
            f"Bridge started: {odom_topic} -> TF {self.parent_frame} -> {self.child_frame}"
        )

    def _odom_cb(self, msg: Odometry) -> None:
        tf_msg = TransformStamped()

        # Use source timestamp when available; fallback only if zero stamp arrives.
        if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
            tf_msg.header.stamp = self.get_clock().now().to_msg()
        else:
            tf_msg.header.stamp = msg.header.stamp

        tf_msg.header.frame_id = self.parent_frame
        tf_msg.child_frame_id = self.child_frame
        tf_msg.transform.translation.x = msg.pose.pose.position.x
        tf_msg.transform.translation.y = msg.pose.pose.position.y
        tf_msg.transform.translation.z = msg.pose.pose.position.z
        tf_msg.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(tf_msg)

        self.msg_count += 1
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds > int(5e9):
            self.get_logger().info(f"Forwarded {self.msg_count} odometry messages")
            self.last_log_time = now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="/Odometry -> map->base_link TF bridge")
    parser.add_argument("--odom-topic", default="/Odometry")
    parser.add_argument("--parent-frame", default="map")
    parser.add_argument("--child-frame", default="base_link")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = OdomTfBridge(args.odom_topic, args.parent_frame, args.child_frame)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
