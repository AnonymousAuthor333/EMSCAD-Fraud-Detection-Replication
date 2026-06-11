#!/usr/bin/env python3
"""
single_pass_detector.py

Adds a  single_pass_risk_score  column to EMSCAD_fraud_detection.csv.

Each posting is evaluated by a SINGLE GPT-4o API call that sees only the raw
posting text — no flag framework, no multi-agent system, no flag results.
This provides a controlled baseline for comparing against the multi-agent
fraud_detector.py results.

Design principles:
  - Model and temperature IDENTICAL to fraud_detector.py / agents.py
    (gpt-4o, temperature=0.1) so any performance difference is due to the
    architecture, not the model.
  - The fraudulent_agent column is NEVER passed to the model.
  - The model receives NO flag definitions, NO orchestrator calibration,
    and NO hint that a multi-agent system exists.
  - Each call is stateless (no message history between rows).
  - Posting format reused from agents.py for consistency.

The exact prompt (system message + user message template) is saved to
single_pass_query.txt so the evaluation protocol is fully reproducible.

Usage:
  python single_pass_detector.py
  → Enter API key, start row, end row.
    Existing scores in the range are overwritten.
"""

import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import openai
import pandas as pd

# ── Local imports (same folder only) ──────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from agents import format_posting   # reuses the same posting formatter as the
                                    # multi-agent pipeline; excludes fraudulent_agent

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH         = _HERE / "EMSCAD_fraud_detection.csv"
QUERY_RECORD_PATH = _HERE / "single_pass_query.txt"

# ── Model — must stay consistent with agents.py / fraud_detector.py ────────────
MODEL       = "gpt-4o"
TEMPERATURE = 0.1   # low → high replicability

# ── Parallelism ────────────────────────────────────────────────────────────────
MAX_WORKERS  = 10   # concurrent API calls (single-pass is 1 call/row → more headroom)
SAVE_EVERY   = 25   # write CSV to disk every N completed rows
CALL_TIMEOUT = 60   # seconds before abandoning one API call

# ── Output column ──────────────────────────────────────────────────────────────
OUTPUT_COL = "single_pass_risk_score"


# ══════════════════════════════════════════════════════════════════════════════
# Prompt definition
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_MSG = (
    "You are reviewing job postings that were submitted by employers and recruiters "
    "on a public online employment platform around 2013. "
    "The dataset contains a mix of legitimate job opportunities and fraudulent postings. "
    "Your task is to read each posting as it was originally submitted and judge how "
    "likely it is to be fraudulent.\n\n"
    "Each posting is evaluated independently — retain no memory of previous calls. "
    "Return valid JSON only — no preamble, no explanation, no markdown."
)

# {posting_text} is the only substitution made at runtime
USER_MSG_TEMPLATE = """\
The following is a job posting submitted by an employer or recruiter on an online \
employment website. Some postings on this platform are legitimate; others are \
fraudulent. Using only the information provided in this posting, assign a fraud \
risk score from 0 to 10.

Score scale:
  0  : definitely legitimate
  1-2: very likely legitimate
  3-4: more likely legitimate than not
  5  : uncertain
  6-7: more likely fraudulent than not
  8-9: very likely fraudulent
  10 : definitely fraudulent

=== JOB POSTING ===
{posting_text}

Return ONLY: {{"risk_score": <integer 0-10>}}"""


def _save_query_record() -> None:
    """
    Write the exact prompt template used to single_pass_query.txt.
    This file is the reproducibility record for the single-pass protocol.
    """
    record_lines = [
        "=" * 70,
        "SINGLE-PASS LLM RISK SCORE — QUERY RECORD",
        "=" * 70,
        "",
        f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model     : {MODEL}",
        f"Temperature: {TEMPERATURE}",
        f"Output col: {OUTPUT_COL}",
        "",
        "PURPOSE",
        "-------",
        "Each row of EMSCAD_fraud_detection.csv is evaluated with ONE API call.",
        "The model sees only the raw posting text and general fraud knowledge.",
        "No flag definitions, no orchestrator calibration, no multi-agent context.",
        "This serves as a controlled baseline for comparing against the multi-agent",
        "fraud_detector.py approach.",
        "",
        "=" * 70,
        "SYSTEM MESSAGE (sent as role='system')",
        "=" * 70,
        "",
        SYSTEM_MSG,
        "",
        "=" * 70,
        "USER MESSAGE TEMPLATE (sent as role='user')",
        "  {posting_text} is replaced at runtime by the formatted posting.",
        "  Posting format is identical to that used in the multi-agent pipeline",
        "  (agents.py::format_posting) to ensure a fair comparison.",
        "=" * 70,
        "",
        USER_MSG_TEMPLATE,
        "",
        "=" * 70,
        "POSTING FIELDS INCLUDED (in order, from agents.py::format_posting)",
        "=" * 70,
        "",
        "  Title, Location, Department, Salary Range, Employment Type,",
        "  Required Experience, Required Education, Industry, Function,",
        "  Telecommuting, Has Company Logo, Has Questions, Company Profile,",
        "  Description, Requirements, Benefits,",
        "  QCEW Annual Avg Employment, QCEW Avg Annual Pay,",
        "  QCEW OTY Employment Change, QCEW OTY Employment Pct Chg",
        "",
        "  NOTE: 'fraudulent_agent' is NEVER included (excluded in format_posting).",
        "",
    ]
    QUERY_RECORD_PATH.write_text("\n".join(record_lines), encoding="utf-8")
    print(f"  Query record saved → {QUERY_RECORD_PATH.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Single-pass API call
# ══════════════════════════════════════════════════════════════════════════════

def _score_row(
    row: dict,
    client: openai.OpenAI,
    sem: threading.Semaphore,
) -> Optional[int]:
    """
    Make one GPT-4o call for the given posting row.
    Returns an integer 0-10, or None on unrecoverable failure.
    """
    posting_text = format_posting(row)
    user_msg = USER_MSG_TEMPLATE.format(posting_text=posting_text)

    with sem:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user",   "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )

    raw = json.loads(resp.choices[0].message.content)
    score = raw.get("risk_score", None)
    if score is None:
        return None
    return max(0, min(10, int(round(float(score)))))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(df: pd.DataFrame) -> None:
    df.to_csv(DATA_PATH, index=False, encoding="utf-8")


def _fmt_eta(seconds: float) -> str:
    if seconds in (float("inf"), float("nan")) or seconds < 0:
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 62)
    print("  EMSCAD Single-Pass Fraud Risk Scorer")
    print("=" * 62)

    # ── File checks ───────────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        sys.exit(f"\nRequired file not found:\n  {DATA_PATH}\n")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"\nLoading {DATA_PATH.name} …")
    df = pd.read_csv(DATA_PATH, encoding="latin-1", low_memory=False)
    total = len(df)
    print(f"  {total:,} rows  |  {len(df.columns)} columns")

    if OUTPUT_COL not in df.columns:
        df[OUTPUT_COL] = None
        print(f"  Added column '{OUTPUT_COL}' (all empty)")
    else:
        n_done = int(df[OUTPUT_COL].notna().sum())
        print(f"  Column '{OUTPUT_COL}' found — {n_done:,} rows already scored")

    # ── Status ────────────────────────────────────────────────────────────────
    n_done    = int(df[OUTPUT_COL].notna().sum())
    n_todo    = total - n_done
    unproc    = df[df[OUTPUT_COL].isna()].index
    first_emp = int(unproc[0])  if len(unproc) > 0 else None
    last_emp  = int(unproc[-1]) if len(unproc) > 0 else None

    print(f"\n  Completed : {n_done:,}  |  Remaining : {n_todo:,}")
    if first_emp is not None:
        print(f"  First unscored index : {first_emp}")
        print(f"  Last  unscored index : {last_emp}")

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        api_key = input("\nOpenAI API key: ").strip()
    if not api_key:
        sys.exit("No API key provided.")
    client = openai.OpenAI(api_key=api_key)

    # ── Save query record (before any processing begins) ─────────────────────
    print()
    _save_query_record()

    # ── Range selection ───────────────────────────────────────────────────────
    print(f"\nRow indices are 0-based (0 = first data row, {total-1} = last).")
    print("Existing scores in the range will be overwritten.\n")
    try:
        raw_s = input(f"  Start index [default 0]: ").strip()
        start = int(raw_s) if raw_s else 0

        raw_e = input(f"  End index   [default {total-1}, or -1 for last]: ").strip()
        if not raw_e:
            end = total - 1
        else:
            end = int(raw_e)
            if end < 0:
                end = total - 1
    except ValueError:
        sys.exit("Invalid index input.")

    if not (0 <= start <= end < total):
        sys.exit(f"Invalid range {start}–{end} for dataset of {total} rows.")

    target = list(range(start, end + 1))
    n_target = len(target)

    print(f"\nTarget range : rows {start} – {end}  ({n_target:,} rows)")
    print(f"Model        : {MODEL}  |  Temperature : {TEMPERATURE}")
    print(f"Workers      : {MAX_WORKERS}  |  Auto-save every {SAVE_EVERY} rows")
    print()

    # ── Parallel processing ───────────────────────────────────────────────────
    sem       = threading.Semaphore(MAX_WORKERS)
    save_lock = threading.Lock()
    completed = [0]
    saved_at  = [0]
    start_t   = time.time()

    def _do(df_idx: int) -> Tuple[int, Optional[int]]:
        row = df.loc[df_idx].to_dict()
        try:
            score = _score_row(row, client, sem)
        except Exception as exc:
            print(f"  [row {df_idx}] error: {exc}")
            score = None
        return df_idx, score

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_WORKERS, thread_name_prefix="sp") as pool:

        futures = {pool.submit(_do, idx): idx for idx in target}

        for future in concurrent.futures.as_completed(futures):
            df_idx = futures[future]
            try:
                df_idx, score = future.result(timeout=CALL_TIMEOUT)
            except Exception as exc:
                print(f"  ERROR row {df_idx}: {exc}")
                score = None

            with save_lock:
                df.at[df_idx, OUTPUT_COL] = score
                completed[0] += 1
                done = completed[0]

            # Progress report
            if done % 10 == 0 or done == n_target:
                elapsed = time.time() - start_t
                rate    = done / elapsed if elapsed > 0 else 0
                eta     = (n_target - done) / rate if rate > 0 else float("inf")
                pct     = done / n_target * 100
                print(f"  {done:>{len(str(n_target))}}/{n_target} "
                      f"({pct:5.1f}%)  |  "
                      f"{elapsed/60:5.1f} min elapsed  |  "
                      f"ETA {_fmt_eta(eta)}")

            # Periodic save
            with save_lock:
                if completed[0] - saved_at[0] >= SAVE_EVERY or done == n_target:
                    _save(df)
                    saved_at[0] = completed[0]
                    print(f"  [auto-saved — {completed[0]} rows complete]")

    # ── Final save ────────────────────────────────────────────────────────────
    _save(df)
    elapsed = time.time() - start_t

    print(f"\n{'=' * 62}")
    print(f"  Finished {completed[0]}/{n_target} rows in {elapsed/60:.1f} min")
    print(f"  Column '{OUTPUT_COL}' updated in {DATA_PATH.name}")
    print(f"  Query record saved to {QUERY_RECORD_PATH.name}")
    print("=" * 62)


if __name__ == "__main__":
    main()
