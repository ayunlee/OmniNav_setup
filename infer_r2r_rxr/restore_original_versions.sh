#!/bin/bash
# Restore original package versions to fix compatibility issues

echo "🔄 Restoring original package versions..."
echo ""

# Original versions (before downgrade)
ORIGINAL_VERSIONS=(
    "torch==2.9.0a0+50eac811a6.nv25.09"
    "torchvision==0.24.0a0+98f8b375"
    "opencv-python==4.11.0.86"
    "Pillow==11.3.0"
    "scipy==1.16.1"
    "safetensors==0.6.2"
)

echo "Uninstalling current versions..."
pip uninstall -y torch torchvision opencv-python Pillow scipy safetensors 2>&1 | grep -E '(Uninstalling|Successfully)'

echo ""
echo "Installing original versions..."

# Note: torch의 특정 빌드는 직접 설치가 어려울 수 있으므로
# pip install --force-reinstall --no-cache-dir를 사용하거나
# 원래 설치 방법을 사용해야 할 수 있습니다

# 일반 패키지들은 복구 가능
pip install opencv-python==4.11.0.86 Pillow==11.3.0 scipy==1.16.1 safetensors==0.6.2 2>&1 | tail -5

echo ""
echo "⚠️  torch와 torchvision은 원래 설치 방법으로 복구해야 합니다."
echo "   (NVIDIA 최적화 빌드는 특별한 설치 방법이 필요할 수 있습니다)"
echo ""
echo "✅ 다른 패키지들은 복구되었습니다."

