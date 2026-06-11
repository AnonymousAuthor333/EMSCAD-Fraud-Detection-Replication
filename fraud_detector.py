#!/usr/bin/env python3
"""
fraud_detector.py
Main pipeline for the EMSCAD multi-agent fraud detection system.

For each row in EMSCAD_fraud_detection.csv the pipeline runs:
  Phase 1 — three sub-agents in parallel:
      Legitimacy Agent  → flags 1, 2, 3, 7
      Context Agent     → flag  4
      Consistency Agent → flags 5, 6, 8

  Phase 2 — four orchestrator calls in parallel:
      Full (all 3 agents)          → risk_score_all
      Legitimacy + Context         → risk_score_legit_context
      Context + Consistency        → risk_score_context_consist
      Legitimacy + Consistency     → risk_score_legit_consist

Output columns added to the CSV (existing columns untouched):
  flag1_sensitive_info     flag2_unrealistic_comp   flag3_pressure_tactics
  flag4_anon_employer      flag5_req_role_mismatch  flag6_info_conflict
  flag7_vague_info         flag8_promo_template
  risk_score_all           risk_score_legit_context
  risk_score_context_consist  risk_score_legit_consist

Key properties:
  - Model: gpt-4o, temperature 0.1 (high replicability)
  - Stateless: every API call is a fresh context — no memory between rows
  - fraudulent_agent column is NEVER passed to any agent
  - Parallel rows (ROW_WORKERS) + parallel API calls (API_WORKERS)
  - Periodic auto-save; progress display with ETA
  - User specifies a 0-based row range; existing scores are overwritten

Usage:
  python fraud_detector.py
"""

import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openai
import pandas as pd

# ── Local imports (same folder only) ──────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from agents import (
    AGENT_FLAGS, ABLATION_COMBOS, FLAG_COL,
    format_posting, run_sub_agent, run_orchestrator,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH      = _HERE / "EMSCAD_fraud_detection.csv"
FLAG_DEFS_PATH = _HERE / "flag_definitions_agent.json"

# ── Parallelism ────────────────────────────────────────────────────────────────
# ROW_WORKERS rows are active simultaneously.
# Each active row submits up to 7 API calls to the shared API_POOL.
# API_SEMAPHORE caps the number of live (in-flight) API calls at any moment.
#
# Safe sizing rule:  API_WORKERS >= ROW_WORKERS * 7   (avoids API pool saturation)
#                    API_SEMAPHORE <= API_WORKERS      (caps real concurrency)
ROW_WORKERS   = 3     # rows processed in parallel
API_WORKERS   = 24    # thread-pool size for individual API calls
API_SEMAPHORE = 12    # max simultaneous live API calls (rate-limit guard)
SAVE_EVERY    = 20    # write CSV to disk every N completed rows
CALL_TIMEOUT  = 90    # seconds before a single API call is abandoned

# ── Output columns ─────────────────────────────────────────────────────────────
FLAG_COLS = [
    "flag1_sensitive_info",
    "flag2_unrealistic_comp",
    "flag3_pressure_tactics",
    "flag4_anon_employer",
    "flag5_req_role_mismatch",
    "flag6_info_conflict",
    "flag7_vague_info",
    "flag8_promo_template",
]
SCORE_COLS = [
    "risk_score_all",
    "risk_score_legit_context",
    "risk_score_context_consist",
    "risk_score_legit_consist",
]
ALL_OUTPUT_COLS = FLAG_COLS + SCORE_COLS


# ══════════════════════════════════════════════════════════════════════════════
# Row processor
# ══════════════════════════════════════════════════════════════════════════════

def _process_row(
    row_idx: int,
    row: dict,
    flag_defs: dict,
    client: openai.OpenAI,
    api_pool: concurrent.futures.ThreadPoolExecutor,
    sem: threading.Semaphore,
) -> Dict:
    """
    Run all 7 API calls for one posting row.

    Phase 1: three sub-agents submitted to api_pool simultaneously.
    Phase 2: four orchestrators submitted to api_pool simultaneously,
             after all Phase 1 futures have resolved.

    Returns a flat dict of {output_column: value}.
    """
    posting_text = format_posting(row)

    # ── Semaphore-guarded wrappers ─────────────────────────────────────────────
    def _sub(agent_name: str) -> Dict[str, bool]:
        with sem:
            return run_sub_agent(client, agent_name, posting_text, flag_defs)

    def _orch(combo_flags: Dict[str, bool], agents: List[str]) -> int:
        with sem:
            return run_orchestrator(client, combo_flags, flag_defs, agents)

    # ── Phase 1: sub-agents ────────────────────────────────────────────────────
    agent_names = ["legitimacy", "context", "consistency"]
    sub_futs = {name: api_pool.submit(_sub, name) for name in agent_names}

    agent_results: Dict[str, Dict[str, bool]] = {}
    for name, fut in sub_futs.items():
        try:
            agent_results[name] = fut.result(timeout=CALL_TIMEOUT)
        except Exception as exc:
            print(f"  [row {row_idx}] sub-agent '{name}' failed: {exc}")
            # Default all flags for this agent to False
            agent_results[name] = {FLAG_COL[fid]: False for fid in AGENT_FLAGS[name]}

    # Flat merge of all flag results
    all_flags: Dict[str, bool] = {}
    for r in agent_results.values():
        all_flags.update(r)

    # ── Phase 2: orchestrators (full + 3 ablation combos) ─────────────────────
    orch_configs = {
        "all":             ["legitimacy", "context", "consistency"],
        "legit_context":   ["legitimacy", "context"],
        "context_consist": ["context",    "consistency"],
        "legit_consist":   ["legitimacy", "consistency"],
    }

    orch_futs = {}
    for combo_key, agents in orch_configs.items():
        # Pass only the flags that belong to the included agents
        combo_flags = {
            FLAG_COL[fid]: all_flags.get(FLAG_COL[fid], False)
            for agent in agents
            for fid in AGENT_FLAGS[agent]
        }
        orch_futs[combo_key] = api_pool.submit(_orch, combo_flags, agents)

    scores: Dict[str, Optional[int]] = {}
    for combo_key, fut in orch_futs.items():
        try:
            scores[combo_key] = fut.result(timeout=CALL_TIMEOUT)
        except Exception as exc:
            print(f"  [row {row_idx}] orchestrator '{combo_key}' failed: {exc}")
            scores[combo_key] = None

    # ── Build clean output record ──────────────────────────────────────────────
    record: Dict = {}
    for col in FLAG_COLS:
        record[col] = 1 if all_flags.get(col, False) else 0
    record["risk_score_all"]             = scores.get("all")
    record["risk_score_legit_context"]   = scores.get("legit_context")
    record["risk_score_context_consist"] = scores.get("context_consist")
    record["risk_score_legit_consist"]   = scores.get("legit_consist")
    return record


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_flag_defs() -> dict:
    with open(FLAG_DEFS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add output columns with None if they don't already exist."""
    for col in ALL_OUTPUT_COLS:
        if col not in df.columns:
            df[col] = None
    return df


def _save(df: pd.DataFrame) -> None:
    """Write the full DataFrame back to CSV (utf-8, no BOM)."""
    df.to_csv(DATA_PATH, index=False, encoding="utf-8")


def _fmt_eta(seconds: float) -> str:
    if seconds == float("inf") or seconds < 0:
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 62)
    print("  EMSCAD Multi-Agent Fraud Detector")
    print("=" * 62)

    # ── File checks ───────────────────────────────────────────────────────────
    for p in [DATA_PATH, FLAG_DEFS_PATH]:
        if not p.exists():
            sys.exit(f"\nRequired file not found:\n  {p}\n")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"\nLoading {DATA_PATH.name} …")
    df = pd.read_csv(DATA_PATH, encoding="latin-1", low_memory=False)
    total = len(df)
    print(f"  {total:,} rows  |  {len(df.columns)} columns")
    df = _ensure_output_columns(df)

    # ── Load flag definitions ─────────────────────────────────────────────────
    flag_defs = _load_flag_defs()
    print(f"  {len(flag_defs['flags'])} flag definitions loaded from "
          f"{FLAG_DEFS_PATH.name}")

    # ── Status summary ────────────────────────────────────────────────────────
    done_mask   = df["risk_score_all"].notna()
    n_done      = int(done_mask.sum())
    n_todo      = total - n_done
    unproc_idx  = df[~done_mask].index
    first_empty = int(unproc_idx[0]) if len(unproc_idx) > 0 else None
    last_empty  = int(unproc_idx[-1]) if len(unproc_idx) > 0 else None

    print(f"\n  Completed : {n_done:,}  |  "
          f"Remaining : {n_todo:,}")
    if first_empty is not None:
        print(f"  First unprocessed row index : {first_empty}")
        print(f"  Last  unprocessed row index : {last_empty}")

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        api_key = input("\nOpenAI API key: ").strip()
    if not api_key:
        sys.exit("No API key provided.")
    client = openai.OpenAI(api_key=api_key)

    # ── Range selection ───────────────────────────────────────────────────────
    print(f"\nRow indices are 0-based (0 = first data row, {total-1} = last).")
    print("Existing scores in the range will be overwritten.\n")
    try:
        raw_start = input(f"  Start index [default 0]: ").strip()
        start = int(raw_start) if raw_start else 0

        raw_end = input(f"  End index   [default {total-1}, or -1 for last]: ").strip()
        if not raw_end:
            end = total - 1
        else:
            end = int(raw_end)
            if end < 0:
                end = total - 1
    except ValueError:
        sys.exit("Invalid index input.")

    if not (0 <= start <= end < total):
        sys.exit(f"Invalid range {start}–{end} for dataset of {total} rows.")

    target = list(range(start, end + 1))
    n_target = len(target)

    print(f"\nTarget range : rows {start} – {end}  ({n_target:,} rows)")
    print(f"Parallelism  : {ROW_WORKERS} rows × up to {API_SEMAPHORE} live API calls")
    print(f"Auto-save    : every {SAVE_EVERY} completed rows")
    print()

    # ── Parallel processing ───────────────────────────────────────────────────
    sem           = threading.Semaphore(API_SEMAPHORE)
    save_lock     = threading.Lock()
    completed     = [0]       # mutable list so nested fn can increment
    last_saved_at = [0]
    start_time    = time.time()

    def _do_row(df_idx: int,
                api_pool: concurrent.futures.ThreadPoolExecutor
                ) -> Tuple[int, Dict]:
        """Worker function: process one row and return (index, result_dict)."""
        row = df.loc[df_idx].to_dict()
        result = _process_row(df_idx, row, flag_defs, client, api_pool, sem)
        return df_idx, result

    # Two nested pools:
    #   api_pool  – one pool shared by all active rows for individual API calls
    #   row_pool  – drives ROW_WORKERS rows at a time
    # No deadlock risk: row_pool threads block waiting for api_pool futures,
    # but they never hold api_pool resources themselves.
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=API_WORKERS, thread_name_prefix="api") as api_pool:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=ROW_WORKERS, thread_name_prefix="row") as row_pool:

            row_futures = {
                row_pool.submit(_do_row, idx, api_pool): idx
                for idx in target
            }

            for future in concurrent.futures.as_completed(row_futures):
                df_idx = row_futures[future]
                try:
                    df_idx, record = future.result()
                    with save_lock:
                        for col, val in record.items():
                            df.at[df_idx, col] = val
                        completed[0] += 1
                        done = completed[0]
                except Exception as exc:
                    print(f"  ERROR row {df_idx}: {exc}")
                    with save_lock:
                        completed[0] += 1
                        done = completed[0]
                    continue

                # ── Progress report (every 10 rows or at the end) ──────────
                if done % 10 == 0 or done == n_target:
                    elapsed = time.time() - start_time
                    rate    = done / elapsed if elapsed > 0 else 0
                    eta     = (n_target - done) / rate if rate > 0 else float("inf")
                    pct     = done / n_target * 100
                    print(f"  {done:>{len(str(n_target))}}/{n_target} "
                          f"({pct:5.1f}%)  |  "
                          f"{elapsed/60:5.1f} min elapsed  |  "
                          f"ETA {_fmt_eta(eta)}")

                # ── Periodic save ──────────────────────────────────────────
                with save_lock:
                    since_save = completed[0] - last_saved_at[0]
                    if since_save >= SAVE_EVERY or done == n_target:
                        _save(df)
                        last_saved_at[0] = completed[0]
                        print(f"  [auto-saved — {completed[0]} rows complete]")

    # ── Final save (guarantees latest state on disk) ──────────────────────────
    _save(df)
    elapsed = time.time() - start_time

    print(f"\n{'=' * 62}")
    print(f"  Finished {completed[0]}/{n_target} rows in {elapsed/60:.1f} min")
    print(f"  Results written to {DATA_PATH.name}")
    print("=" * 62)


if __name__ == "__main__":
    main()
