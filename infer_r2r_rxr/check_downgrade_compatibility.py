#!/usr/bin/env python3
"""
Check if downgrading packages is safe by checking dependencies
"""
import os
import sys
import warnings
import subprocess

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*pynvml.*')

# HPC-X/UCC library conflict prevention
_LD_PRELOAD_LIBS = "/opt/hpcx/ucx/lib/libucs.so.0:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucm.so.0"
_REEXEC_VAR = "_OMNINAV_REEXEC"

if os.environ.get(_REEXEC_VAR) != "1" and os.path.exists("/opt/hpcx/ucx/lib/libucs.so.0"):
    os.environ["LD_PRELOAD"] = _LD_PRELOAD_LIBS
    os.environ[_REEXEC_VAR] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

def get_package_requirements(package_name):
    """Get requirements for a package"""
    try:
        result = subprocess.run(
            ["pip", "show", package_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            requires = []
            in_requires = False
            for line in result.stdout.split('\n'):
                if line.startswith('Requires:'):
                    in_requires = True
                    req_line = line.split(':', 1)[1].strip()
                    if req_line:
                        requires.extend([r.strip() for r in req_line.split(',')])
                elif in_requires and line and not line.startswith(' '):
                    break
                elif in_requires and line.startswith(' '):
                    requires.extend([r.strip() for r in line.strip().split(',')])
            return requires
    except:
        pass
    return []

def check_version_compatibility(package, target_version, required_by_packages):
    """Check if target version is compatible with requirements"""
    issues = []
    
    for req_pkg in required_by_packages:
        reqs = get_package_requirements(req_pkg)
        for req in reqs:
            # Parse requirement like "safetensors>=0.4.3"
            if '>=' in req:
                req_name, req_version = req.split('>=')
                req_name = req_name.strip()
                req_version = req_version.strip()
                
                if req_name.lower() == package.lower():
                    # Compare versions (simple string comparison for now)
                    if target_version < req_version:
                        issues.append(f"{req_pkg} requires {req_name}>={req_version}, but target is {target_version}")
    
    return issues

def main():
    print("\n" + "=" * 80)
    print("🔍 Package Downgrade Compatibility Check".center(80))
    print("=" * 80 + "\n")
    
    # Current versions
    current_versions = {
        'torch': '2.9.0a0+50eac811a6.nv25.09',
        'opencv-python': '4.11.0',
        'Pillow': '11.3.0',
        'scipy': '1.16.1',
        'safetensors': '0.6.2',
    }
    
    # Target versions
    target_versions = {
        'torch': '2.6.0',
        'opencv-python': '4.10.0',
        'Pillow': '11.1.0',
        'scipy': '1.14.1',
        'safetensors': '0.5.2',
    }
    
    # Packages that depend on these
    dependent_packages = ['transformers', 'qwen-vl-utils', 'trl', 'peft']
    
    print("📋 Downgrade Plan:")
    print("-" * 80)
    print(f"{'Package':<25} {'Current':<30} {'Target':<30}")
    print("-" * 80)
    for pkg in target_versions.keys():
        print(f"{pkg:<25} {current_versions.get(pkg, 'N/A'):<30} {target_versions[pkg]:<30}")
    
    print("\n🔍 Checking Dependencies:")
    print("-" * 80)
    
    all_issues = []
    
    # Check each package
    for pkg, target_ver in target_versions.items():
        issues = check_version_compatibility(pkg, target_ver, dependent_packages)
        if issues:
            print(f"\n⚠️  {pkg} ({target_ver}):")
            for issue in issues:
                print(f"   - {issue}")
                all_issues.append(issue)
        else:
            print(f"✅ {pkg} ({target_ver}): No conflicts detected")
    
    # Check transformers specific requirements
    print("\n📦 Transformers Requirements:")
    print("-" * 80)
    try:
        result = subprocess.run(
            ["pip", "show", "transformers"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Requires:'):
                    print(f"   {line}")
                    # Check safetensors requirement
                    if 'safetensors>=' in line:
                        import re
                        match = re.search(r'safetensors>=([0-9.]+)', line)
                        if match:
                            min_ver = match.group(1)
                            if target_versions['safetensors'] < min_ver:
                                print(f"   ⚠️  WARNING: transformers requires safetensors>={min_ver}, but target is {target_versions['safetensors']}")
                            else:
                                print(f"   ✅ safetensors {target_versions['safetensors']} >= {min_ver} (OK)")
    except:
        pass
    
    # Summary
    print("\n" + "=" * 80)
    if all_issues:
        print("❌ Compatibility Issues Found:")
        for issue in all_issues:
            print(f"   - {issue}")
        print("\n⚠️  Downgrade may cause dependency conflicts!")
    else:
        print("✅ No major compatibility issues detected")
        print("   (Still recommended to test after downgrade)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

