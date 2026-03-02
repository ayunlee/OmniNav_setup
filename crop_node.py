#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # 마침표(.) 제거됨
from cv_bridge import CvBridge
import cv2
import sys

class ImageResizer(Node):
    def __init__(self):
        super().__init__('image_resizer_node')
        
        # 파라미터로 타겟 해상도 설정 (기본값: 480x426)
        self.declare_parameter('target_w', 480)
        self.declare_parameter('target_h', 426)
        
        self.TARGET_W = self.get_parameter('target_w').value
        self.TARGET_H = self.get_parameter('target_h').value

        # QoS 설정 (카메라 데이터 유실 방지)
        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 구독 및 발행
        self.sub = self.create_subscription(Image, 'color/image_raw', self.listener_callback, qos_policy)
        self.pub = self.create_publisher(Image, 'color/image_target', 10)
        
        self.bridge = CvBridge()
        self.get_logger().info(f'Node Started: Input(Any) -> Target({self.TARGET_W}x{self.TARGET_H})')

    def listener_callback(self, msg):
        try:
            # 1. ROS Image -> OpenCV Image 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 원본 이미지 크기 (예: 640x400)
            h, w, _ = cv_image.shape
            
            # =========================================================
            # 로직: 높이(Height)를 강제로 맞추고, 너비(Width)를 중앙 크롭
            # =========================================================
            
            # 2. 스케일링 (높이 기준 업스케일링)
            # 400 -> 426이 되려면 약 1.065배 확대 필요
            scale = self.TARGET_H / h
            new_w = int(w * scale)      # 640 * 1.065 = 681
            new_h = self.TARGET_H       # 426 (고정)
            
            # 이미지 리사이즈 (비율 유지 확대)
            # 보간법: 확대 시 INTER_LINEAR나 INTER_CUBIC이 좋음 (기본값은 LINEAR)
            resized_img = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 3. Center Crop (가로 중앙만 남기기)
            # 현재 resized_img는 681x426 상태 -> 480x426으로 잘라야 함
            if new_w >= self.TARGET_W:
                center_x = new_w // 2
                start_x = center_x - (self.TARGET_W // 2)
                end_x = start_x + self.TARGET_W
                
                # 배열 슬라이싱 [높이전체, 가로시작:가로끝]
                final_img = resized_img[:, start_x:end_x]
            else:
                # 만약 (희박하지만) 리사이즈 해도 폭이 목표보다 작다면
                # 그냥 그대로 내보내거나 블랙 패딩을 해야 함. 여기선 그대로 내보냄.
                self.get_logger().warn(f"Image too narrow: {new_w} < {self.TARGET_W}")
                final_img = resized_img

            # 4. 결과 퍼블리시
            out_msg = self.bridge.cv2_to_imgmsg(final_img, "bgr8")
            out_msg.header = msg.header # 시간 동기화 유지
            self.pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageResizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
