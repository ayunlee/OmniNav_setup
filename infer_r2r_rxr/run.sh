#!/bin/bash
# HPC-X/UCC 라이브러리 충돌 방지
export LD_PRELOAD="/opt/hpcx/ucx/lib/libucs.so.0:/opt/hpcx/ucx/lib/libucp.so.0:/opt/hpcx/ucx/lib/libucm.so.0"
exec python /workspace/OmniNav/infer_r2r_rxr/run_infer_iphone.py "$@"
