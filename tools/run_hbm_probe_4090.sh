#!/usr/bin/env bash
# run_hbm_probe_4090.sh — RTX 4090 sustained memory-bandwidth probe (one-shot).
#
# Purpose: measure the 4090's sustained DRAM read bandwidth to backfill
# `sustained_read_gbs` in bench/ops/gpu_specs.h (currently 0.0 placeholder).
#
# HOW TO RUN (from the 4090 host, Windows or bash):
#   1. Stop the resident inference service first (the probe needs >= 8 GiB
#      free VRAM and an uncontended memory bus, or the numbers are useless).
#   2. If the dev container is not running:  docker start ninfer-4090-kaso-devel
#   3. Run:
#        docker exec ninfer-4090-kaso-devel bash /ninfer-4090-kaso/tools/run_hbm_probe_4090.sh
#   4. Result is printed to stdout and saved to /ninfer-4090-kaso/hbm_probe_4090.log
#      (host path: C:\Data\ninfer\ninfer-4090-kaso\hbm_probe_4090.log).
#
# Notes:
#   - --peak-gbps 1008 is the 4090 GDDR6X advertised bus bandwidth
#     (the probe's built-in default of 1792 is the RTX 5090 value — do not use it).
#   - The "kernel uint4 read" row (best/median bus GB/s) is the value that
#     belongs in gpu_specs.h sustained_read_gbs.
#   - The probe uses ~2 x min(4 GiB, free/5) of VRAM, auto-scaled down, and
#     refuses to run unless two buffers fit in 2/3 of the FREE memory.

set -euo pipefail

REPO=/ninfer-4090-kaso
OUT="$REPO/hbm_probe_4090.log"
BIN="$REPO/hbm_bandwidth_probe"

exec > >(tee "$OUT") 2>&1

echo "== $(date '+%F %T') container=$(hostname) =="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv

FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)"
if [ "${FREE_MIB:-0}" -lt 8192 ]; then
  echo "ABORT: only ${FREE_MIB} MiB VRAM free (< 8192). The resident inference" >&2
  echo "service is probably still running — stop it, then re-run this script." >&2
  exit 1
fi

if [ ! -x "$BIN" ] || [ "$REPO/tools/hbm_bandwidth_probe.cu" -nt "$BIN" ]; then
  echo "-- compiling hbm_bandwidth_probe (-arch=sm_89) --"
  nvcc -O3 -std=c++17 -arch=sm_89 "$REPO/tools/hbm_bandwidth_probe.cu" -o "$BIN"
fi

echo "-- running probe (peak reference 1008 GB/s) --"
"$BIN" --peak-gbps 1008

echo
echo "-- post-run VRAM state --"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
echo "DONE. Log: $OUT"
