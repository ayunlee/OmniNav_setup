#!/usr/bin/env python3
"""
OmniNav Robot Controller (Curvature-based Pure Pursuit Version)
Subscribes to /action topic and controls Scout Mini via cmd_vel.

Algorithm:
- Transforms Network Coordinates (dx: Right, dy: Forward) to Robot Body Frame (x: Forward, y: Left).
- Calculates Curvature (kappa) to reach the target point with a smooth arc.
- Controls linear velocity based on distance/time and angular velocity based on curvature.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped
import math
import json
import threading
import sys

class OmniNavController(Node):
    """
    OmniNav Robot Controller for Scout Mini
    """
    
    # Robot parameters (Scout Mini)
    MAX_LINEAR_VEL = 0.6      # m/s (약간 상향 조정 가능, 안전을 위해 0.8 설정)
    MAX_ANGULAR_VEL = 0.6   # rad/s (로봇 최대 각속도, ~30 deg/s)
    WAYPOINT_DURATION = 0.15    # seconds per waypoint (5 waypoints = 1.0s total)
    CONTROL_RATE = 10          # Hz (제어 주기, 10Hz로 높여 더 부드럽게 반응)
    
    # Control tuning
    MIN_DISTANCE_THRESHOLD = 0.05  # 5cm 미만은 이동 안 함 (노이즈 필터링 강화)
    
    def __init__(self):
        super().__init__('omninav_controller')
        
        self.callback_group = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self._is_active = True  # 종료 시그널 플래그
        
        # QoS profile (cmd_vel + /action: BEST_EFFORT so publisher/subscriber match)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # /action: must match run_infer_online_panorama publisher exactly (all policies)
        qos_action = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Subscriber
        self.action_sub = self.create_subscription(
            String,
            '/action',
            self._action_callback,
            qos_action,
            callback_group=self.callback_group
        )
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(
            TwistStamped,
            '/scout_mini_base_controller/cmd_vel',
            qos_profile
        )
        
        # Timer
        self.timer = self.create_timer(
            1.0 / self.CONTROL_RATE,
            self._control_loop,
            callback_group=self.callback_group
        )
        
        # State variables (5-waypoint sequential execution)
        self.current_waypoint_list = []   # list of waypoints to execute
        self.waypoint_batch_start_time = 0.0
        self.is_executing = False
        self.arrive_flag = False
        
        # Debug info
        self.get_logger().info("=" * 60)
        self.get_logger().info("OmniNav Controller (Curvature Pursuit) Ready")
        self.get_logger().info(f"  Max V: {self.MAX_LINEAR_VEL} m/s, Max W: {self.MAX_ANGULAR_VEL} rad/s")
        self.get_logger().info(f"  Waypoint duration: {self.WAYPOINT_DURATION}s per waypoint (5 wp = {5 * self.WAYPOINT_DURATION}s)")
        self.get_logger().info("=" * 60)

    def _action_callback(self, msg: String):
        """Callback for /action topic"""
        if not self._is_active: return

        try:
            data = json.loads(msg.data)
            waypoints = data.get('waypoints', [])
            arrive_pred = data.get('arrive_pred', 0)
            
            if not waypoints:
                return
            
            self.get_logger().info(f"Received /action: {len(waypoints)} waypoints, arrive={arrive_pred}")
            
            with self.lock:
                # 5개 waypoint 전체를 순차 실행할 리스트로 저장
                self.current_waypoint_list = list(waypoints)
                self.arrive_flag = (arrive_pred > 0)
                
                # 새 명령 수신 시 배치 시작 시각 리셋
                self.waypoint_batch_start_time = self.get_clock().now().nanoseconds / 1e9
                self.is_executing = True

        except Exception as e:
            self.get_logger().error(f"Action callback error: {e}")

    def _waypoint_to_cmd_vel(self, waypoint: dict) -> tuple:
        """
        [핵심 알고리즘 변경] Curvature-based Pure Pursuit
        Network Output: dx (Right+), dy (Forward+)
        Robot Frame:    x (Forward+), y (Left+)
        """
        dx_net = waypoint.get('dx', 0.0)
        dy_net = waypoint.get('dy', 0.0)
        
        # 1. 좌표계 변환 (Network -> Robot Body ISO 8855)
        # Net Forward (dy) -> Robot X
        # Net Right (dx)   -> Robot -Y (Left)
        target_x = dy_net
        target_y = -dx_net
        
        # 2. 거리 제곱 계산 (L^2)
        dist_sq = target_x**2 + target_y**2
        distance = math.sqrt(dist_sq)
        
        # 노이즈 필터 (너무 가까우면 정지)
        if distance < self.MIN_DISTANCE_THRESHOLD:
            return 0.0, 0.0

        # -----------------------------------------------------------
        # [NEW] Curvature Calculation
        # kappa = 2 * y / L^2
        # 로봇이 (0,0)에서 (x,y)로 접선 방향을 유지하며 도달하는 원의 곡률
        # -----------------------------------------------------------
        curvature = (2.0 * target_y) / dist_sq
        
        # 3. 선형 속도 (v) 설정
        # 기본 전략: 남은 거리를 WAYPOINT_DURATION(0.2s) 내에 가도록 설정하되, Max로 제한
        linear_vel = distance / self.WAYPOINT_DURATION
        
        # 최대 속도 클리핑
        linear_vel = self._clip(linear_vel, -self.MAX_LINEAR_VEL, self.MAX_LINEAR_VEL)
        
        # 4. 각속도 (omega) 계산
        # v = r * omega, kappa = 1/r  =>  omega = v * kappa
        angular_vel = linear_vel * curvature
        
        # 각속도 클리핑 (물리적 한계 보호)
        angular_vel = self._clip(angular_vel, -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL)
        
        return linear_vel, angular_vel

    def _clip(self, value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))
      
    def _control_loop(self):
        """주기적 제어 루프 (5 waypoint 순차 실행, 각 0.2초)"""
        if not self._is_active: return

        current_time = self.get_clock().now().nanoseconds / 1e9
        linear_vel = 0.0
        angular_vel = 0.0
        
        with self.lock:
            if self.is_executing and len(self.current_waypoint_list) > 0:
                elapsed = current_time - self.waypoint_batch_start_time
                total_batch_duration = len(self.current_waypoint_list) * self.WAYPOINT_DURATION
                
                if self.arrive_flag:
                    # 도착 예측 시 즉시 정지
                    linear_vel, angular_vel = 0.0, 0.0
                    self.is_executing = False
                elif elapsed < total_batch_duration:
                    # 현재 구간에 해당하는 waypoint 인덱스 (0.2초마다 다음 waypoint)
                    waypoint_index = min(int(elapsed / self.WAYPOINT_DURATION), len(self.current_waypoint_list) - 1)
                    current_waypoint = self.current_waypoint_list[waypoint_index]
                    linear_vel, angular_vel = self._waypoint_to_cmd_vel(current_waypoint)
                else:
                    # 배치 종료 (5 * 0.2s 경과)
                    self.is_executing = False
                    linear_vel = 0.0
                    angular_vel = 0.0
            else:
                # Idle state
                linear_vel = 0.0
                angular_vel = 0.0

        self._publish_cmd_vel(linear_vel, angular_vel)
    
    def _publish_cmd_vel(self, linear_vel: float, angular_vel: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(linear_vel)
        msg.twist.angular.z = float(angular_vel)
        self.cmd_vel_pub.publish(msg)
    
    def stop_robot(self):
        """Emergency stop & Cleanup"""
        self._is_active = False
        self.get_logger().warn("Stopping robot and cleaning up...")
        
        # 즉시 정지 명령 3회 전송 (패킷 손실 대비)
        for _ in range(3):
            self._publish_cmd_vel(0.0, 0.0)
            
        # 내부 상태 초기화
        with self.lock:
            self.is_executing = False
            self.current_waypoint_list = []


import signal
import sys

def main(args=None):
    rclpy.init(args=args)
    
    node = OmniNavController()
    
    # 멀티스레드 실행기 (스레드 수 자동 할당)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    # [좀비 방지 핵심 1] 종료 시그널 핸들러 정의
    # Ctrl+C (SIGINT)나 kill 명령 (SIGTERM)을 받으면 실행되는 함수
    def signal_handler(sig, frame):
        node.get_logger().warn(f"Received signal {sig}. Initiating forceful shutdown...")
        
        # 1. 로봇 즉시 정지 (가장 중요)
        node.stop_robot()
        
        # 2. 실행기 종료 (스레드 정리)
        try:
            executor.shutdown()
        except Exception as e:
            pass # 이미 종료 중이면 무시
            
        # 3. 노드 파괴
        try:
            node.destroy_node()
        except:
            pass
            
        # 4. ROS 종료
        if rclpy.ok():
            rclpy.shutdown()
            
        # 5. 프로세스 강제 종료 (가장 확실한 방법)
        node.get_logger().info("Forcefully exiting process.")
        sys.exit(0)

    # [좀비 방지 핵심 2] 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 실행
        executor.spin()
    except KeyboardInterrupt:
        pass # signal_handler가 이미 처리함
    except ExternalShutdownException:
        pass # ROS2 시스템 종료 시그널
    finally:
        # 혹시 spin이 그냥 끝났을 경우를 대비한 안전망
        if rclpy.ok():
            node.stop_robot()
            executor.shutdown()
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
