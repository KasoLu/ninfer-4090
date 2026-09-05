"""Greedy-output quality battery for the rk6v4-e8 KV mode.

Fork of run_rk8v4_quality.py: the prompt, CLI flags, and greedy/MTP3 sampling
are byte-identical so results stay comparable with the original battery. The
fork additionally:

- compares three modes side by side: bf16 (reference), rk8v4 (previous
  capacity tier) and rk6v4-e8 (new 6-bit E8 lattice key tier);
- takes the engine binary, model artifact, and output directory from
  environment variables (NINFER_EXE, NINFER_MODEL, NINFER_QUALITY_OUT), so
  the same script runs on any host, e.g. the RTX 4090 build container;
- records pairwise greedy-output consistency (identical characters within the
  shortest answer) in the manifest as a proxy for how much the new mode
  degrades relative to bf16 and rk8v4.

Usage:
    NINFER_MODEL=/models/qwen3_8_27b.ninfer \
        python3 tools/bench/run_rk6v4e8_quality.py
    NINFER_EXE defaults to "ninfer" (resolved via PATH); NINFER_QUALITY_OUT
    defaults to benchmark_results/rk6v4e8_quality_<UTC timestamp>.
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

MAX_CONTEXT = 4096
MAX_NEW = 1024
KV_MODES = ("bf16", "rk8v4", "rk6v4-e8")

EXE = os.environ.get("NINFER_EXE", "ninfer")
MODEL = Path(os.environ["NINFER_MODEL"]) if os.environ.get("NINFER_MODEL") else None
if MODEL is None:
    raise SystemExit("NINFER_MODEL is required (path to the .ninfer artifact)")

_STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(
    os.environ.get("NINFER_QUALITY_OUT", f"benchmark_results/rk6v4e8_quality_{_STAMP}")
)

PROMPT = r"""You are reviewing a production algorithm. Design and implement a complete C++20 solution for offline dynamic connectivity in an undirected graph.

The input is a time-ordered list of operations:
- add(id, u, v): activate an edge with a unique edge instance id;
- remove(id): deactivate that exact edge instance;
- connected(u, v): report whether u and v are connected at that moment.

The graph permits parallel edges and self-loops. An id is added at most once and removed at most once, but an edge may remain active through the end. Vertex labels are 0..n-1. Use a segment tree over time and a rollback DSU. Do not use path compression.

Deliver all of the following:
1. compilable C++20 code with explicit operation and answer types;
2. a precise explanation of how each edge's active half-open interval is constructed;
3. the rollback invariant, including what is recorded for a union that changes nothing;
4. a proof that every query sees exactly the edges active at its time;
5. time and memory complexity in terms of operations q, vertices n, and edge lifetimes;
6. handling of empty input, self-loops, parallel edges, invalid remove ids, and still-active edges.

Treat invalid remove ids as input errors and show where the implementation rejects them. Keep the answer rigorous and avoid replacing code with pseudocode."""


def run_mode(mode: str) -> dict:
    command = [
        EXE,
        str(MODEL),
        "--prompt",
        PROMPT,
        "--max-context",
        str(MAX_CONTEXT),
        "--kv-capacity",
        str(MAX_CONTEXT),
        "--prefill-chunk",
        "1024",
        "--max-new",
        str(MAX_NEW),
        "--kv-dtype",
        mode,
        "--spec",
        "mtp",
        "--draft-tokens",
        "3",
        "--lm-head-draft",
        "--greedy",
        "--no-thinking",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=600, check=False)
    (OUTPUT_DIR / f"{mode}.answer.md").write_text(completed.stdout, encoding="utf-8")
    (OUTPUT_DIR / f"{mode}.metrics.txt").write_text(completed.stderr, encoding="utf-8")
    return {
        "mode": mode,
        "returncode": completed.returncode,
        "answer_characters": len(completed.stdout),
    }


def pairwise_consistency(answers: dict) -> list:
    """Identical-character fraction within the shortest answer of each pair."""
    pairs = []
    modes = list(answers)
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            a, b = answers[modes[i]], answers[modes[j]]
            if not a or not b:
                continue
            n = min(len(a), len(b))
            same = sum(1 for x, y in zip(a[:n], b[:n]) if x == y)
            pairs.append(
                {
                    "pair": f"{modes[i]} vs {modes[j]}",
                    "identical_chars": same,
                    "min_chars": n,
                    "consistency": round(same / n, 4),
                    "exact_match": a == b,
                }
            )
    return pairs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "prompt.md").write_text(PROMPT + "\n", encoding="utf-8")
    results = [run_mode(mode) for mode in KV_MODES]
    answers = {}
    for mode in KV_MODES:
        answers[mode] = (OUTPUT_DIR / f"{mode}.answer.md").read_text(encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "executable": EXE,
        "model": str(MODEL),
        "max_context": MAX_CONTEXT,
        "max_new_tokens": MAX_NEW,
        "sampling": "greedy",
        "speculation": "MTP3 with lm-head-draft",
        "results": results,
        "pairwise_consistency": pairwise_consistency(answers),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
