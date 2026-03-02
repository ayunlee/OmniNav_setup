#!/bin/bash

# ==========================================
# ROS 2 Camera Safe Launcher (Final Stable)
# ==========================================

# 1. 환경 설정
source ./gemini_driver/install/setup.bash

# 2. 청소 함수
cleanup() {
    echo ""
    echo "🛑 [Cleanup] 종료 신호 감지! 프로세스 정리 중..."
    pkill -2 -f "ros2 launch"
    pkill -9 -f "component_container"
    pkill -9 -f "crop_node.py"
    echo "✅ [Cleanup] 완료."
    exit 0
}

# 3. 트랩 설정
trap cleanup SIGINT SIGTERM EXIT

# 4. [핵심] 사전 청소
echo "🧹 [Pre-flight] 좀비 프로세스 사살 중..."
pkill -9 -f "component_container" 2>/dev/null
pkill -9 -f "crop_node.py" 2>/dev/null
sleep 2

# 5. [옵션] USB 장치 연결 확인 (디버깅용)
echo "🔍 [Check] 연결된 Orbbec 장치 수 확인..."
count=$(lsusb | grep -c "Orbbec")
echo "   -> 현재 감지된 카메라: $count 대"

if [ "$count" -lt 3 ]; then
    echo "⚠️  [Warning] 카메라가 3대보다 적습니다! (USB 컨트롤러가 죽었을 수 있음)"
    echo "    -> 해결법: 컴퓨터 재부팅 또는 USB 선을 뽑았다 다시 꽂으세요."
    # 강제로 진행은 함
fi

# 6. 런치 실행
echo "🚀 [Launch] 카메라 시스템 시작 (5 FPS 모드)..."
ros2 launch ./triple_camera_launch.py

# 대기
wait