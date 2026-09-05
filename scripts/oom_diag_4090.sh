#!/usr/bin/env bash
# One-shot OOM diagnostic battery for the 24GB RTX 4090.
# Runs INSIDE the ninfer-4090-kaso-dev container (repo bind-mounted at
# /ninfer-4090-kaso, model volume at /models). The Windows host wrapper
# (run_oom_diag.ps1) starts the container, invokes this script, collects
# the output from /ninfer-4090-kaso/oom_diag_out, and stops the container.
#
# Purpose: pin down which allocation OOMs with context-cache (prefix reuse)
# enabled, versus --no-prefix-reuse, on the REAL production baseline:
#   --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 600000
#   --spec mtp --draft-tokens 3 --lm-head-draft
#   --chat-template-file /models/chat_template_sharp_v22_4_1.jinja
#   --prefill-chunk 1024   (plus per-phase --kv-dtype/--max-context/reuse)
#
# Phases (all include the production baseline flags above):
#   P0  control: ctx 200000, rk8v4, --no-prefix-reuse  (== production, must work)
#   R1  ctx 100000, rk4v4-e8, reuse ON (all context-cache defaults)
#   R2  ctx 100000, rk4v4-e8, --no-prefix-reuse
#   R3  ctx 100000, rk4v4-e8, reuse ON + slim slots
#       (--device-state-slots 0 --host-state-slots 2
#        --max-private-continuations 1 --max-shared-prefixes 0
#        --max-long-anchors-per-continuation 0)
#   R4  (only if R1 failed) reuse ON, rk4v4-e8, ctx/kv descent 90k..60k
#   R5  ctx 100000, rk4v4-e8, reuse ON + --device-state-slots 0 --no-cuda-graph
#
# Each phase: start serve -> wait for "listening" (or detect crash/OOM) ->
# on success: fetch /metrics, nvidia-smi, then fire one ~100k-token prompt
# to reproduce runtime OOM; always: nvidia-smi + log tail captured.

set -u

REPO=/ninfer-4090-kaso
OUT=$REPO/oom_diag_out
BUILD=$REPO/build
SERVE=$BUILD/apps/ninfer-serve
PORT=1234
CTX=100000

mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/diag.log"; }
snap() { nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$OUT/$1.nvidia-smi" 2>&1; }

# ---------- preflight ----------
log "=== PREFLIGHT ==="
{
  echo "## git state"
  git -C "$REPO" log --oneline -1
  git -C "$REPO" status --short | head -20
  echo "## build tree"
  ls -la "$SERVE" 2>&1
  grep -m1 CMAKE_HOME_DIRECTORY "$BUILD/CMakeCache.txt" 2>&1
  grep -m1 "CMAKE_CUDA_ARCHITECTURES" "$BUILD/CMakeCache.txt" 2>&1
  echo "## model volume"
  ls -la /models/ 2>&1
  echo "## python3"
  python3 --version 2>&1
  echo "## nvidia-smi"
  nvidia-smi 2>&1
} > "$OUT/preflight.txt" 2>&1
cat "$OUT/preflight.txt"

# Exact production artifact; fall back to the largest .ninfer under /models.
MODEL=/models/qwen3_8_27b.ninfer
[ -f "$MODEL" ] || MODEL=$(ls -S /models/*.ninfer 2>/dev/null | head -1)
if [ -z "$MODEL" ]; then
  log "FATAL: no .ninfer artifact under /models"; exit 2
fi
log "model: $MODEL"

# Production external Jinja template; if missing, drop the flag and note it.
TEMPLATE_FLAG=()
if [ -f /models/chat_template_sharp_v22_4_1.jinja ]; then
  TEMPLATE_FLAG=(--chat-template-file /models/chat_template_sharp_v22_4_1.jinja)
else
  log "WARN: /models/chat_template_sharp_v22_4_1.jinja not found; phases run WITHOUT --chat-template-file"
fi

# Production baseline flags, identical in every phase.
BASE_FLAGS=(--host 0.0.0.0 --port "$PORT" --max-concurrency 1
  --max-pending-requests 16 --pending-timeout-ms 600000
  --spec mtp --draft-tokens 3 --lm-head-draft
  --prefill-chunk 1024 "${TEMPLATE_FLAG[@]}")

# Incremental rebuild so the tree matches the checked-out sources.
log "=== INCREMENTAL BUILD ==="
cmake --build "$BUILD" --parallel > "$OUT/build.log" 2>&1
rc=$?
log "build rc=$rc (tail):"
tail -5 "$OUT/build.log" | tee -a "$OUT/diag.log"
if [ $rc -ne 0 ]; then log "FATAL: incremental build failed"; exit 2; fi

# ~950k chars of repetitive English -> well over 100k tokens (server
# truncates to --max-context).
{ for i in $(seq 1 10000); do \
    echo "the quick brown fox jumps over the lazy dog near the quiet riverbank and the tall oak trees"; \
  done; } > "$OUT/long_prompt.txt"

# /metrics fetcher (python3 is guaranteed by the devel image).
cat > "$OUT/fetch_metrics.py" <<'PY'
import sys, urllib.request
try:
    data = urllib.request.urlopen("http://127.0.0.1:%s/metrics" % sys.argv[1], timeout=10).read()
    sys.stdout.write(data.decode(errors="replace"))
except Exception as e:
    print("METRICS_ERROR", type(e).__name__, e)
PY

# Long-prompt client. Prints HTTP status + body head, or CLIENT_ERROR.
cat > "$OUT/long_request.py" <<'PY'
import json, sys, urllib.request, urllib.error
port = sys.argv[1]
prompt = open(sys.argv[2], encoding="utf-8").read()
body = json.dumps({
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 8,
    "stream": False,
}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:%s/v1/chat/completions" % port,
    data=body,
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        print("HTTP", r.status)
        print(r.read().decode(errors="replace")[:3000])
except urllib.error.HTTPError as e:
    print("HTTP", e.code)
    print(e.read().decode(errors="replace")[:3000])
except Exception as e:
    print("CLIENT_ERROR", type(e).__name__, e)
PY

cleanup_serve() {
  pkill -f "ninfer-serve" 2>/dev/null
  sleep 8
  pkill -9 -f "ninfer-serve" 2>/dev/null
  sleep 2
}

# run_phase <name> <ctx> <kv> <kvdtype> [extra flags...]
# Result line: PHASE_RESULT <name> <outcome>
# outcomes: OOM_STARTUP | CRASH_STARTUP | LISTEN_OK | STARTUP_TIMEOUT
#           | OOM_REQUEST | CRASH_REQUEST
run_phase() {
  local name="$1" ctx="$2" kv="$3" kvt="$4"; shift 4
  local logf="$OUT/$name.serve.log"
  local result="STARTUP_TIMEOUT"
  log "PHASE $name START ctx=$ctx kv=$kv kvdtype=$kvt flags:$*"
  snap "before_$name"

  "$SERVE" "$MODEL" "${BASE_FLAGS[@]}" \
    --max-context "$ctx" --kv-capacity "$kv" --kv-dtype "$kvt" "$@" \
    > "$logf" 2>&1 &
  local pid=$!

  local i
  for i in $(seq 1 140); do          # up to ~700 s for startup (MTP graphs)
    sleep 5
    if ! kill -0 "$pid" 2>/dev/null; then
      if grep -qi "out of memory\|cudaMalloc\|CUDA_ERROR" "$logf"; then
        result=OOM_STARTUP
      else
        result=CRASH_STARTUP
      fi
      break
    fi
    if grep -q "listening" "$logf" 2>/dev/null; then
      result=LISTEN_OK
      break
    fi
  done
  [ $result = "STARTUP_TIMEOUT" ] && cleanup_serve
  snap "startup_$name"

  if [ $result = "LISTEN_OK" ]; then
    python3 "$OUT/fetch_metrics.py" "$PORT" > "$OUT/$name.metrics.txt" 2>&1
    log "PHASE $name fired long prompt ..."
    python3 "$OUT/long_request.py" "$PORT" "$OUT/long_prompt.txt" > "$OUT/$name.request.txt" 2>&1
    sleep 5
    if kill -0 "$pid" 2>/dev/null; then
      result=LISTEN_OK
    else
      if grep -qi "out of memory\|cudaMalloc\|CUDA_ERROR" "$logf"; then
        result=OOM_REQUEST
      else
        result=CRASH_REQUEST
      fi
    fi
  fi
  snap "after_$name"
  cleanup_serve
  tail -15 "$logf" > "$OUT/$name.logtail.txt" 2>&1
  echo "PHASE_RESULT $name $result" >> "$OUT/summary.txt"
  log "PHASE $name RESULT: $result"
}

log "=== PHASES ==="
rm -f "$OUT/summary.txt"

# P0: the exact production configuration on the NEW binary (control).
run_phase P0_production_baseline 200000 200000 rk8v4 --no-prefix-reuse
# R1: reproduce today's OOM (100k, rk4v4-e8, context cache defaults).
run_phase R1_default_reuse_on    "$CTX" "$CTX" rk4v4-e8
# R2: same but prefix reuse disabled.
run_phase R2_no_prefix_reuse     "$CTX" "$CTX" rk4v4-e8 --no-prefix-reuse
# R3: reuse ON with all device-resident cache structures slimmed to zero.
run_phase R3_slim_slots          "$CTX" "$CTX" rk4v4-e8 \
  --device-state-slots 0 --host-state-slots 2 \
  --max-private-continuations 1 --max-shared-prefixes 0 \
  --max-long-anchors-per-continuation 0

# R4: only meaningful if R1 failed at startup (arena/initial cudaMalloc) or
# during the long request (KV pool pressure).
if grep -Eq "PHASE_RESULT R1_default_reuse_on (OOM_STARTUP|OOM_REQUEST|CRASH_STARTUP|CRASH_REQUEST|STARTUP_TIMEOUT)" "$OUT/summary.txt"; then
  log "R1 failed -> running R4 capacity descent"
  for pair in 90000 80000 70000 60000; do
    run_phase R4_desc_$pair "$pair" "$pair" rk4v4-e8
    grep -q "PHASE_RESULT R4_desc_$pair LISTEN_OK" "$OUT/summary.txt" && break
  done
fi

# R5: reuse ON, no device slots, no CUDA graphs (isolates graph allowance).
run_phase R5_nograph_no_devslots "$CTX" "$CTX" rk4v4-e8 --device-state-slots 0 --no-cuda-graph

log "=== DONE ==="
cat "$OUT/summary.txt" | tee -a "$OUT/diag.log"
snap "final"
exit 0