#!/bin/bash
# Safe RealSense Camera Launch Script (Compressed Plugin Version)
# 컨테이너 내부에서 실행하세요: ./safe_camera_launch_inside.sh

set -e

echo "=========================================="
echo "Safe RealSense Camera Launcher (Native)"
echo "=========================================="

# 1. 기존 프로세스 정리
echo "[1/4] Cleaning up existing RealSense processes..."
pkill -9 -f "realsense2_camera" 2>/dev/null || true
pkill -9 -f "rs-" 2>/dev/null || true

# 2. 카메라 연결 확인
echo "[2/4] Verifying camera connection..."
if ! rs-enumerate-devices --compact 2>&1 | grep -q 'Intel RealSense'; then
    echo "ERROR: RealSense camera not detected!"
    exit 1
fi
echo "  Camera detected successfully"

# 3. 종료 트랩 설정 (republish 관련 프로세스 kill 제거됨)
cleanup() {
    echo ""
    echo "Stopping camera gracefully..."
    pkill -2 -f "realsense2_camera" 2>/dev/null || true
    sleep 2
    pkill -9 -f "realsense2_camera" 2>/dev/null || true
    echo "Camera stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ROS2 환경 설정
source /opt/ros/jazzy/setup.bash
source /workspace/OmniNav/install/setup.bash 2>/dev/null || true
export RS2_AC_DISABLE_CONDITIONS_CHECK=1

# 4. 카메라 시작
# image_transport 플러그인이 설치되어 있으면 자동으로 compressed 토픽이 생성됩니다.
echo "[3/4] Starting RealSense camera node..."
ros2 run realsense2_camera realsense2_camera_node --ros-args \
    -p enable_color:=true \
    -p color_width:=640 \
    -p color_height:=480 \
    -p color_fps:=5 \
    -p depth_module.enable_depth:=false \
    -p enable_infra1:=false \
    -p enable_infra2:=false \
    -p enable_gyro:=false \
    -p enable_accel:=false \
    -p initial_reset:=true \
    -p reconnect_timeout:=5.0 \
    -p wait_for_device_timeout:=10.0 &

CAM_PID=$!

echo ""
echo "Camera started!"
echo "If 'ros-jazzy-image-transport-plugins' is installed, you will see:"
echo "  - /camera/camera/color/image_raw (raw)"
echo "  - /camera/camera/color/image_raw/compressed (compressed)"
echo ""
echo "Waiting for processes..."

wait $CAM_PID
