#!/bin/bash
# Fix torch/torchvision compatibility issue after downgrade

echo "🔧 Fixing torch/torchvision compatibility issue..."
echo ""

# Check current versions
echo "Current versions:"
pip show torch torchvision | grep -E '(Name|Version)'
echo ""

# Uninstall current versions
echo "Uninstalling current torch and torchvision..."
pip uninstall -y torch torchvision
echo ""

# Install compatible versions
# For torch 2.6.0, we need torchvision 0.21.0 or compatible
# But since we're on NVIDIA GB10, we should use CUDA-enabled versions

echo "Installing torch 2.6.0 with CUDA support..."
# Try to install with CUDA support
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -10

# If that fails, try with cu118
if [ $? -ne 0 ]; then
    echo "Trying with CUDA 11.8..."
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -10
fi

# Verify installation
echo ""
echo "Verifying installation:"
python3 -c "import torch; print(f'torch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); import torchvision; print(f'torchvision: {torchvision.__version__}')" 2>&1 | grep -v Warning

echo ""
echo "✅ Done!"

