#!/usr/bin/env bash
# Round-2 OOM diagnostic for the 24GB RTX 4090 (runs inside ninfer-4090-kaso-dev).
# Prerequisite: repo at the cuda-oom fix commit (context-cache host tiers now
# default to 0), incremental build up to date, inference container stopped.
#
# Purpose: confirm reuse ON with the fixed defaults boots and serves at the
# production context lengths, and probe how much pinned host RAM the box can
# give an explicit host tier.
#
# All phases include the production baseline flags (see round-1 script) plus
# --max-context/--kv-capacity/--kv-dtype:
#   S1  ctx 200000, rk8v4,    reuse ON, host tiers default (0)  == target state
#   S2  ctx 100000, rk4v4-e8, reuse ON, host tiers default (0)
#   S3  ctx 200000, rk8v4,    reuse ON, --host-kv-mib 512 (pinned probe)

set -u

REPO=/ninfer-4090-kaso
OUT=$REPO/oom_diag_out_r2
BUILD=$REPO/build
SERVE=$BUILD/apps/ninfer-serve
PORT=1234

mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/diag.log"; }
snap() { nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$OUT/$1.nvidia-smi" 2>&1; }

log "=== PREFLIGHT ==="
{
  echo "## git state"
  git -C "$REPO" log --oneline -1
  git -C "$REPO" status --short | head -10
  echo "## build tree"
  ls -la "$SERVE" 2>&1
  echo "## model volume"
  ls -la /models/ 2>&1
  echo "## nvidia-smi"
  nvidia-smi 2>&1
} > "$OUT/preflight.txt" 2>&1
cat "$OUT/preflight.txt"

# Incremental rebuild so the tree matches the fix commit.
log "=== INCREMENTAL BUILD ==="
cmake --build "$BUILD" --parallel > "$OUT/build.log" 2>&1
rc=$?
log "build rc=$rc (tail):"
tail -5 "$OUT/build.log" | tee -a "$OUT/diag.log"
if [ $rc -ne 0 ]; then log "FATAL: incremental build failed"; exit 2; fi

{ for i in $(seq 1 10000); do \
    echo "the quick brown fox jumps over the lazy dog near the quiet riverbank and the tall oak trees"; \
  done; } > "$OUT/long_prompt.txt"

cat > "$OUT/fetch_metrics.py" <<'PY'
import sys, urllib.request
try:
    data = urllib.request.urlopen("http://127.0.0.1:%s/metrics" % sys.argv[1], timeout=10).read()
    sys.stdout.write(data.decode(errors="replace"))
except Exception as e:
    print("METRICS_ERROR", type(e).__name__, e)
PY

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
run_phase() {
  local name="$1" ctx="$2" kv="$3" kvt="$4"; shift 4
  local logf="$OUT/$name.serve.log"
  local result="STARTUP_TIMEOUT"
  log "PHASE $name START ctx=$ctx kv=$kv kvdtype=$kvt flags:$*"
  snap "before_$name"

  "$SERVE" /models/qwen3_8_27b.ninfer --host 0.0.0.0 --port "$PORT" \
    --max-concurrency 1 --max-pending-requests 16 --pending-timeout-ms 600000 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --chat-template-file /models/chat_template_sharp_v22_4_1.jinja \
    --prefill-chunk 1024 \
    --max-context "$ctx" --kv-capacity "$kv" --kv-dtype "$kvt" "$@" \
    > "$logf" 2>&1 &
  local pid=$!

  local i
  for i in $(seq 1 140); do
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
  tail -25 "$logf" > "$OUT/$name.logtail.txt" 2>&1
  echo "PHASE_RESULT $name $result" >> "$OUT/summary.txt"
  log "PHASE $name RESULT: $result"
}

log "=== PHASES (round 2) ==="
rm -f "$OUT/summary.txt"
run_phase S1_prod_ctx_reuse_on 200000 200000 rk8v4
run_phase S2_100k_reuse_on     100000 100000 rk4v4-e8
run_phase S3_hostkv_512        200000 200000 rk8v4 --host-kv-mib 512

log "=== DONE ==="
cat "$OUT/summary.txt" | tee -a "$OUT/diag.log"
snap "final"
exit 0