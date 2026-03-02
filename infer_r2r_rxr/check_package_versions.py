#!/usr/bin/env python3
"""
Check versions of all packages required for run_infer_iphone_panorama.py

This script checks the versions of all external packages used in the inference pipeline.
Run this script inside the Docker container to verify package versions.
"""
import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*pynvml.*')

# HPC-X/UCC library conflict prevention (same as run_infer_iphone_panorama.py)
_LD_PRELOAD_LIBS = "/opt/hpcx/ucx/lib/libucs.so.0:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucm.so.0"
_REEXEC_VAR = "_OMNINAV_REEXEC"

if os.environ.get(_REEXEC_VAR) != "1" and os.path.exists("/opt/hpcx/ucx/lib/libucs.so.0"):
    # Re-execute self with correct LD_PRELOAD
    os.environ["LD_PRELOAD"] = _LD_PRELOAD_LIBS
    os.environ[_REEXEC_VAR] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

def get_package_version_from_pip(package_name):
    """Get package version from pip list"""
    try:
        import subprocess
        result = subprocess.run(
            ["pip", "show", package_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
    except:
        pass
    return None

def get_package_version(package_name, import_name=None):
    """Get version of a package, return None if not installed"""
    if import_name is None:
        import_name = package_name
    
    # Try to get version from module first
    version = None
    try:
        module = __import__(import_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = str(module.version)
        elif hasattr(module, 'VERSION'):
            version = str(module.VERSION)
    except ImportError:
        return None
    except Exception:
        pass
    
    # If version not found from module, try pip
    if version is None or version == "installed (version unknown)":
        pip_version = get_package_version_from_pip(package_name)
        if pip_version:
            return pip_version
        elif version:
            return version
    
    return version if version else None

def main():
    # Header
    print("\n" + "=" * 80)
    print("📦 Package Version Check for run_infer_iphone_panorama.py".center(80))
    print("=" * 80 + "\n")
    
    # Python version
    py_version = sys.version.split()[0]
    print(f"🐍 Python: {py_version}\n")
    
    # Core packages used in run_infer_iphone_panorama.py
    # (package_name, import_name)
    packages = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("tqdm", "tqdm"),
        ("opencv-python", "cv2"),
        ("Pillow", "PIL"),
        ("transformers", "transformers"),
        ("qwen-vl-utils", "qwen_vl_utils"),
        ("scipy", "scipy"),
        ("safetensors", "safetensors"),
    ]
    
    # Get package versions and status
    package_info = []
    for package_name, import_name in packages:
        version = get_package_version(package_name, import_name)
        if version is not None:
            status = "✅ INSTALLED"
            version_str = str(version)
        else:
            status = "❌ NOT INSTALLED"
            version_str = "-"
        package_info.append((package_name, status, version_str))
    
    # Print table
    print(f"{'Package':<25} {'Status':<20} {'Version':<35}")
    print("-" * 80)
    for pkg_name, status, version in package_info:
        print(f"{pkg_name:<25} {status:<20} {version:<35}")
    
    # PyTorch CUDA info (간단히)
    try:
        import torch
        print(f"\n🔥 PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   CUDA: Not Available")
    except ImportError:
        pass
    
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()

