#!/usr/bin/env python3
"""Objective KV-quantization precision audit against a bf16 reference.

Why this replaces the naive "character identity on one long generation"
comparison (run_rk6v4e8_quality.py):

  * character-level identity on long code generation measures chaotic
    sensitivity to a single flipped argmax, not quality. Greedy decoding
    through a quantised KV cache diverges after the first differing token
    and then the remaining characters are uncorrelated noise -- a 10-17 %
    character match between two *correct* generators is normal, not a bug;
  * speculative decoding (MTP3) was a confounder: this suite runs
    plain greedy decoding only, so the only variable between modes is
    the KV-cache quantisation;
  * there was no ground-truth set and no determinism self-check.

What this suite measures, per mode, against the bf16 reference:

  T0  determinism: identical prompt run twice must be byte-identical.
                   (Any difference is itself a finding.)
  T1  short answers: 12 numeric questions with unambiguous ground truth.
       accuracy = fraction correct (objective), plus agreement rate with
       the bf16 answer on the same questions.
  T2  long generation: token-level LCS similarity vs the bf16 reference
       (robust to where the sequences diverge), unigram
       precision/Jaccard, longest-common-prefix ratio, and length.

Composite score per mode (0..1): 0.5 * short-accuracy + 0.5 * long-LCS.
A mode's "degradation vs bf16" is the gap between its composite and the
bf16 composite.

Environment variables:
    NINFER_MODEL        (required) path to the .ninfer artifact
    NINFER_EXE          (default: ninfer, resolved via PATH)
    NINFER_QUALITY_OUT  (default: benchmark_results/precision_suite)
    NINFER_KV_MODES     (default: bf16,rk8v4,rk6v4-e8; comma separated)

Example:
    NINFER_MODEL=/models/qwen3_8_27b.ninfer \
        NINFER_EXE=/ninfer-4090-kaso/build/apps/ninfer \
        python3 tools/bench/run_kv_precision_suite.py

Note: each metric run spawns a fresh `ninfer` process, so the model is
re-loaded per run; a full 3-mode suite takes roughly 30-60 minutes on a
4090. Narrow it with NINFER_KV_MODES if needed.
"""

import collections
import difflib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

EXE = os.environ.get("NINFER_EXE", "ninfer")
MODEL = os.environ.get("NINFER_MODEL")
OUT = Path(os.environ.get("NINFER_QUALITY_OUT", "benchmark_results/precision_suite"))
MODES = [m.strip() for m in os.environ.get("NINFER_KV_MODES", "bf16,rk8v4,rk6v4-e8").split(",") if m.strip()]
VALID_KV_DTYPES = {
    "bf16", "int8", "fp8", "rk8v4", "rk4v4", "rk4v4-e8", "rk2v4-e8", "rk6v4-e8",
}  # must match apps/cli/options.cpp parse_kv_cache() exactly

MAX_CONTEXT = 4096
LONG_MAX_NEW = 1024
MID_MAX_NEW = 300
SHORT_MAX_NEW = 48

TIMEOUT_SHORT = 180
TIMEOUT_MID = 900
TIMEOUT_LONG = 1800

# Bytes per (token, kv_head) per mode; head_dim = 256. Static reference info
# so the score table can be read against the capacity cost.
BYTES_PER_TOKEN_KVHEAD = {
    "bf16": 1024,
    "rk8v4": 400,
    "rk4v4": 272,
    "rk4v4-e8": 272,
    "rk6v4-e8": 336,
}

# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

LONG_PROMPT = """You are reviewing a production algorithm. Design and implement a complete C++20 solution for offline dynamic connectivity in an undirected graph.

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

MID_PROMPT = "用 C++20 写一个迭代版快速排序（含 main 函数，完整可编译，不要伪代码）。"

# (prompt, acceptable answers) -- every question has an unambiguous numeric
# ground truth so T1 is fully objective and needs no LLM judge.
SHORT_QUESTIONS = [
    ("请只输出一个数字，不要任何解释。17 乘以 23 等于多少？", (391,)),
    ("请只输出一个数字，不要任何解释。凸六边形的内角和是多少度？", (720,)),
    ("请只输出一个数字，不要任何解释。97 之后的下一个质数是多少？", (101,)),
    ("请只输出一个数字，不要任何解释。840 的最大质因数是多少？", (7,)),
    ("请只输出一个数字，不要任何解释。2 的 10 次方等于多少？", (1024,)),
    ("请只输出一个数字，不要任何解释。1024 的算术平方根是多少？", (32,)),
    ("请只输出一个数字，不要任何解释。17 除以 5 的余数是多少？", (2,)),
    ("请只输出一个数字，不要任何解释。能被 4 和 6 同时整除的最小正整数是多少？", (12,)),
    ("请只输出一个数字，不要任何解释。1 到 100 所有整数之和是多少？", (5050,)),
    ("请只输出一个数字，不要任何解释。一个闰年有多少天？", (366,)),
    ("请只输出一个数字，不要任何解释。若 x + 5 = 17，x 是多少？", (12,)),
    ("请只输出一个数字，不要任何解释。45 乘以 12 等于多少？", (540,)),
]


# ---------------------------------------------------------------------------
# engine driver
# ---------------------------------------------------------------------------

def run_once(mode: str, prompt: str, max_new: int, timeout: int) -> dict:
    """One greedy, non-speculative inference run of the ninfer binary."""
    command = [
        EXE,
        MODEL,
        "--prompt", prompt,
        "--max-context", str(MAX_CONTEXT),
        "--kv-capacity", str(MAX_CONTEXT),
        "--kv-dtype", mode,
        "--max-new", str(max_new),
        "--greedy",
        "--no-thinking",
    ]
    t0 = time.time()
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        rc, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
        note = ""
    except subprocess.TimeoutExpired as exc:
        rc = -999
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "")
        note = "timeout"
    if rc != 0 and not note:
        note = "exit_code=%d" % rc + ((": " + stderr[-500:]) if stderr else "")
    return {
        "returncode": rc,
        "stdout": stdout,
        "stderr_tail": stderr[-4000:],
        "wall_seconds": round(time.time() - t0, 1),
        "note": note,
    }


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------

def extract_number(text: str):
    """Last number in the answer (the position a 'answer only' prompt puts it)."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not nums:
        return None
    try:
        v = float(nums[-1])
    except ValueError:
        return None
    return int(v) if v == int(v) else v


def word_tokens(text: str):
    return re.findall(r"[a-z0-9_]+", text.lower())


def unigram_prj(a_tokens, b_tokens):
    ca, cb = collections.Counter(a_tokens), collections.Counter(b_tokens)
    inter = sum((ca & cb).values())
    prec = inter / sum(ca.values()) if ca else 0.0
    jac = inter / sum((ca | cb).values()) if (ca or cb) else 0.0
    return prec, jac


def lcs_ratio(a, b):
    """Token-level LCS similarity: 2*M / (|A|+|B|)."""
    if not a or not b:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(blk.size for blk in sm.get_matching_blocks())
    return 2.0 * matched / (len(a) + len(b))


def common_prefix_ratio(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / n if n else 0.0


# ---------------------------------------------------------------------------
# per-mode collection
# ---------------------------------------------------------------------------

def collect_mode(mode: str) -> dict:
    print(f"== mode: {mode} ==", flush=True)
    rec = {
        "mode": mode,
        "bytes_per_token_kvhead": BYTES_PER_TOKEN_KVHEAD.get(mode),
    }

    # T1: objective short-answer battery.
    short_rows = []
    for i, (q, expected) in enumerate(SHORT_QUESTIONS, 1):
        r = run_once(mode, q, SHORT_MAX_NEW, TIMEOUT_SHORT)
        num = extract_number(r["stdout"])
        correct = num is not None and any(abs(num - e) < 1e-9 for e in expected)
        short_rows.append({
            "q": q,
            "raw_answer": r["stdout"].strip()[:200],
            "number": num,
            "expected": list(expected),
            "correct": correct,
            "rc": r["returncode"],
            "wall": r["wall_seconds"],
        })
        print(f"  T1 {i:2d}/{len(SHORT_QUESTIONS)} ok={correct!s:<5} "
              f"(num={num}, expected={list(expected)}, rc={r['returncode']}, {r['wall_seconds']}s)", flush=True)
    rec["short"] = short_rows

    # T0: determinism self-check on a mid-size generation.
    m1 = run_once(mode, MID_PROMPT, MID_MAX_NEW, TIMEOUT_MID)
    m2 = run_once(mode, MID_PROMPT, MID_MAX_NEW, TIMEOUT_MID)
    rec["t0_determinism"] = {
        "identical": m1["stdout"] == m2["stdout"],
        "chars_run1": len(m1["stdout"]),
        "chars_run2": len(m2["stdout"]),
        "wall_1": m1["wall_seconds"],
        "wall_2": m2["wall_seconds"],
    }
    print(f"  T0 determinism: identical={rec['t0_determinism']['identical']}", flush=True)

    # T2: long generation (quality + length signal).
    lg = run_once(mode, LONG_PROMPT, LONG_MAX_NEW, TIMEOUT_LONG)
    rec["long"] = {
        "chars": len(lg["stdout"]),
        "wall": lg["wall_seconds"],
        "rc": lg["returncode"],
        "note": lg["note"],
        "text": lg["stdout"],
    }
    t2_note = "" if lg["returncode"] == 0 else f" (exit_code={lg['returncode']}!)"
    print(f"  T2 long generation: {len(lg['stdout'])} chars in {lg['wall_seconds']}s{t2_note}", flush=True)
    return rec


# ---------------------------------------------------------------------------
# cross-mode analysis
# ---------------------------------------------------------------------------

def build_summary(records: dict) -> dict:
    modes = list(records)
    ref_mode = "bf16" if "bf16" in records else modes[0]
    ref = records[ref_mode]
    ref_long_tokens = word_tokens(ref["long"]["text"])

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "exe": EXE,
        "model": MODEL,
        "max_context": MAX_CONTEXT,
        "reference_mode": ref_mode,
        "modes": [],
    }

    for mode in modes:
        rec = records[mode]
        rows = rec["short"]
        n_ok = sum(1 for r in rows if r["correct"])
        short_acc = n_ok / len(rows) if rows else None

        # agreement of the extracted number with the reference mode
        agree = None
        if mode != ref_mode and ref.get("short"):
            pairs = [(a, b) for a, b in zip(rows, ref["short"])
                     if a["number"] is not None and b["number"] is not None]
            if pairs:
                agree = sum(1 for a, b in pairs if a["number"] == b["number"]) / len(pairs)

        entry = {
            "mode": mode,
            "bytes_per_token_kvhead": rec["bytes_per_token_kvhead"],
            "t0_determinism_identical": rec["t0_determinism"]["identical"],
            "short_correct": n_ok,
            "short_total": len(rows),
            "short_accuracy": short_acc,
            "short_agreement_vs_reference": agree,
            "short_run_errors": sum(1 for r in rows if r.get("rc") not in (0, None)),
        }

        long = rec["long"]
        if long["chars"] > 0:
            toks = word_tokens(long["text"])
            if mode == ref_mode:
                lcs = 1.0
                prec = jac = lcp = 1.0
            else:
                lcs = lcs_ratio(toks, ref_long_tokens)
                prec, jac = unigram_prj(toks, ref_long_tokens)
                lcp = common_prefix_ratio(long["text"], ref["long"]["text"])
            composite = 0.5 * (short_acc or 0.0) + 0.5 * lcs
            entry.update({
                "long_chars": long["chars"],
                "long_tokens": len(toks),
                "lcs_ratio_vs_ref": lcs,
                "unigram_precision": prec,
                "jaccard": jac,
                "long_prefix_ratio": lcp,
                "long_wall": long["wall"],
                "composite": composite,
            })
        else:
            entry.update({
                "long_chars": 0,
                "composite": None,
                "error": long.get("note") or ("no long output" if long["rc"] == 0 else f"exit_code={long['rc']}"),
            })
        out["modes"].append(entry)

    # per-question cross-mode table
    table = {}
    for i, (q, _) in enumerate(SHORT_QUESTIONS):
        row = {"question": q, "expected": list(SHORT_QUESTIONS[i][1])}
        for mode in modes:
            row[mode] = records[mode]["short"][i]
        table[str(i)] = row
    out["per_question"] = table
    return out


def _fmt(v, fmt):
    return fmt % v if isinstance(v, (int, float)) else "n/a"


def print_summary(summary: dict) -> None:
    print("=" * 100)
    print("KV precision suite summary   (reference = %s, greedy, no speculative decoding)"
          % summary["reference_mode"])
    print("=" * 100)
    for m in summary["modes"]:
        lcs = m.get("lcs_ratio_vs_ref")
        prec = m.get("unigram_precision")
        jac = m.get("jaccard")
        agree = m["short_agreement_vs_reference"]
        comp = m.get("composite")
        long_wall = m.get("long_wall")
        suffix = "  [!! %d failed run(s)]" % m["short_run_errors"] if m.get("short_run_errors") else ""
        print("%-12s %6s  T0=%s  short %d/%d  agree=%-6s longLCS=%-8s "
              "uniP=%-6s jac=%-6s long=%-8s comp=%s%s" % (
                  m["mode"],
                  str(m["bytes_per_token_kvhead"]),
                  "Y" if m["t0_determinism_identical"] else "N",
                  m["short_correct"], m["short_total"],
                  _fmt(agree, "%.3f"),
                  _fmt(lcs, "%.4f"),
                  _fmt(prec, "%.3f"),
                  _fmt(jac, "%.3f"),
                  _fmt(long_wall, "%.0fs") if long_wall is not None else "n/a",
                  _fmt(comp, "%.3f"),
                  suffix,
              ))
    print("\nper-question short answers (Y=correct):")
    for i, row in summary["per_question"].items():
        marks = " ".join(("Y" if row[mm["mode"]]["correct"] else "N") for mm in summary["modes"])
        print("q%-3s %-44s %-8s exp=%s" % (i, row["question"][:36], marks, row["expected"]))


def _check_modes() -> None:
    """Fail fast on a mistyped --kv-dtype (e.g. 'rk6v4e8' vs 'rk6v4-e8').

    The ninfer CLI throws 'invalid kv-dtype: <value>' and exits before any
    generation, which previously showed up as a row of 0/12 empty answers.
    """
    bad = [m for m in MODES if m not in VALID_KV_DTYPES]
    if bad:
        raise SystemExit(
            "refusing to run: unknown --kv-dtype value(s) %r\n"
            "ninfer accepts exactly: %s\n"
            "note: the K6 mode is spelled 'rk6v4-e8' (with a hyphen)."
            % (bad, ", ".join(sorted(VALID_KV_DTYPES)))
        )


def main() -> None:
    if not MODEL:
        raise SystemExit("NINFER_MODEL is required")
    _check_modes()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "long_prompt.md").write_text(LONG_PROMPT, encoding="utf-8")
    (OUT / "mid_prompt.md").write_text(MID_PROMPT, encoding="utf-8")

    records = {}
    for mode in MODES:
        records[mode] = collect_mode(mode)

    summary = build_summary(records)
    print_summary(summary)

    manifest = {
        "suite": "kv_precision_suite_v1",
        "summary": summary,
        "short_questions": [{"q": q, "expected": list(a)} for q, a in SHORT_QUESTIONS],
        "per_mode_records": {
            m: {
                "mode": r["mode"],
                "t0": r["t0_determinism"],
                "short": r["short"],
                "long_chars": r["long"]["chars"],
                "long_wall": r["long"]["wall"],
                "long_note": r["long"].get("note", ""),
                "long_text": r["long"]["text"],
            }
            for m, r in records.items()
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("\nfull manifest: %s" % (OUT / "manifest.json"))


if __name__ == "__main__":
    main()
