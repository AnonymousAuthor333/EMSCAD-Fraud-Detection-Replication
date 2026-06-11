#!/usr/bin/env python3
"""
generate_manual_eval_sample.py

Draws 50 stratified postings from EMSCAD_fraud_detection.csv for manual
flag evaluation, then saves them to a new  flags_manual_eval/  subfolder.

Sampling strategy
─────────────────────────────────────────────────────────────────────
  Goal: at least TARGET_PER_FLAG (default 6) AI-flagged positives for
  each of the 8 flags across the 50 postings.

  Algorithm (greedy set-cover, rarest flag first):
    1. Sort flags ascending by number of positives in the full dataset
       so the rarest flags get priority.
    2. For each flag that still needs more coverage, candidate postings
       are those where this flag = 1 and not yet selected.
       Among candidates, prefer postings with the highest total flag
       count (they satisfy multiple needs at once).
    3. Add candidates until this flag reaches TARGET_PER_FLAG OR the
       50-posting budget is exhausted.
    4. Fill any remaining budget slots with a stratified random draw
       (equal legit / fraud share where possible) from unselected rows.

  Flags with very few positives (< TARGET_PER_FLAG in the whole
  dataset) are automatically handled — the program includes all
  available positives and reports the actual coverage achieved.

Output files (all in  flags_manual_eval/ )
─────────────────────────────────────────────────────────────────────
  sample_for_review.csv
      50 postings with posting text and empty manual-flag columns.
      NO AI flags and NO fraud label so your review is unbiased.
      Fill in the manual_flag* columns, then compare with the
      reference file below.

  ai_flags_reference.csv
      AI-generated flag columns + ground-truth fraud label for the
      same 50 postings.
      Open this AFTER completing your manual review to compare.

  sampling_summary.txt
      Coverage table and any warnings (e.g. flags that fell short).

Usage
─────────────────────────────────────────────────────────────────────
  python generate_manual_eval_sample.py
  Optional: set RANDOM_SEED at the top to reproduce the same sample.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
RANDOM_SEED     = 42     # change for a different draw
N_TOTAL         = 50     # total postings in the sample
TARGET_PER_FLAG = 6      # minimum AI-positive instances per flag

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
DATA_PATH  = _HERE / "EMSCAD_fraud_detection.csv"
OUT_DIR    = _HERE / "flags_manual_eval"

# ── Column definitions ─────────────────────────────────────────────────────────
FLAG_COLS = [
    'flag1_sensitive_info',
    'flag2_unrealistic_comp',
    'flag3_pressure_tactics',
    'flag4_anon_employer',
    'flag5_req_role_mismatch',
    'flag6_info_conflict',
    'flag7_vague_info',
    'flag8_promo_template',
]

FLAG_LABELS = {
    'flag1_sensitive_info':   'Sensitive Information Request',
    'flag2_unrealistic_comp': 'Unrealistic Compensation',
    'flag3_pressure_tactics': 'Pressure Tactics',
    'flag4_anon_employer':    'Unidentifiable Employer',
    'flag5_req_role_mismatch':'Requirements Role Mismatch',
    'flag6_info_conflict':    'Internal Information Conflict',
    'flag7_vague_info':       'Vague Information',
    'flag8_promo_template':   'Promotional Template Substitution',
}

# Columns shown to the reviewer (text + structure, NO labels or AI flags)
REVIEW_COLS = [
    'sample_id', 'job_id',
    'title', 'location', 'department',
    'salary_range', 'employment_type',
    'required_experience', 'required_education',
    'industry', 'function',
    'telecommuting', 'has_company_logo', 'has_questions',
    'company_profile', 'description', 'requirements', 'benefits',
    'qcew_avg_annual_pay',
]

# Manual flag columns (blank — reviewer fills these in)
MANUAL_FLAG_COLS = {
    'manual_flag1_sensitive_info':   'Sensitive Information Request  [0=No  1=Yes]',
    'manual_flag2_unrealistic_comp': 'Unrealistic Compensation       [0=No  1=Yes]',
    'manual_flag3_pressure_tactics': 'Pressure Tactics               [0=No  1=Yes]',
    'manual_flag4_anon_employer':    'Unidentifiable Employer           [0=No  1=Yes]',
    'manual_flag5_req_role_mismatch':'Requirements Role Mismatch     [0=No  1=Yes]',
    'manual_flag6_info_conflict':    'Internal Information Conflict  [0=No  1=Yes]',
    'manual_flag7_vague_info':       'Vague Information              [0=No  1=Yes]',
    'manual_flag8_promo_template':   'Promotional Template Subst.    [0=No  1=Yes]',
}


# ══════════════════════════════════════════════════════════════════════════════
# Greedy stratified sampler
# ══════════════════════════════════════════════════════════════════════════════

def greedy_sample(df: pd.DataFrame,
                  flag_cols: list,
                  n_total: int = 50,
                  target_per_flag: int = 6,
                  rng: np.random.Generator = None) -> list:
    """
    Greedy set-cover sampling.

    Returns a list of DataFrame index values (length ≤ n_total).
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # Availability: only flags present in df
    avail_flags = [c for c in flag_cols if c in df.columns]

    # Sort flags by rarity (ascending positives) so rarest get priority
    flag_counts = {c: int(df[c].sum()) for c in avail_flags}
    sorted_flags = sorted(avail_flags, key=lambda c: flag_counts[c])

    selected  = set()
    coverage  = {c: 0 for c in avail_flags}

    # ── Pass 1: greedy coverage ───────────────────────────────────────────
    for flag in sorted_flags:
        needed = target_per_flag - coverage[flag]
        if needed <= 0:
            continue

        # Candidates: flag = 1, not yet selected
        mask       = (df[flag] == 1) & (~df.index.isin(selected))
        candidates = df[mask].copy()
        if candidates.empty:
            continue

        # Prefer candidates that satisfy the most *still-needed* flags
        still_needed = [c for c in avail_flags
                        if coverage[c] < target_per_flag]
        candidates['_score'] = candidates[still_needed].sum(axis=1)
        candidates = candidates.sort_values('_score', ascending=False)

        to_add = min(needed, len(candidates), n_total - len(selected))
        for idx in candidates.index[:to_add]:
            selected.add(idx)
            for fc in avail_flags:
                if df.at[idx, fc] == 1:
                    coverage[fc] += 1
            if len(selected) >= n_total:
                break

        if len(selected) >= n_total:
            break

    # ── Pass 2: fill remaining budget with stratified random draw ─────────
    remaining = n_total - len(selected)
    if remaining > 0:
        not_selected = df.index[~df.index.isin(selected)]
        pool = df.loc[not_selected]

        # Try to maintain ~50/50 fraud/legit among the fill rows
        if 'fraudulent_agent' in df.columns:
            legit_pool = pool[pool['fraudulent_agent'] == 0].index.tolist()
            fraud_pool = pool[pool['fraudulent_agent'] == 1].index.tolist()
            rng.shuffle(legit_pool); rng.shuffle(fraud_pool)
            n_fraud  = min(remaining // 2, len(fraud_pool))
            n_legit  = min(remaining - n_fraud, len(legit_pool))
            fill_idx = legit_pool[:n_legit] + fraud_pool[:n_fraud]
            # top up if still short
            still_short = remaining - len(fill_idx)
            if still_short > 0:
                extra_pool = [i for i in not_selected if i not in selected
                              and i not in fill_idx]
                rng.shuffle(extra_pool)
                fill_idx += extra_pool[:still_short]
        else:
            fill_list = not_selected.tolist()
            rng.shuffle(fill_list)
            fill_idx = fill_list[:remaining]

        selected.update(fill_idx)

    return list(selected)


# ══════════════════════════════════════════════════════════════════════════════
# Output writers
# ══════════════════════════════════════════════════════════════════════════════

def _keep(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return only columns that exist in df."""
    return df[[c for c in cols if c in df.columns]].copy()


def write_review_file(sample: pd.DataFrame, path: Path) -> None:
    """
    sample_for_review.csv — posting text + empty manual-flag columns.
    No AI flags, no fraud label.
    """
    df_out = _keep(sample, REVIEW_COLS)

    # Add empty manual-flag columns (reviewer fills these in)
    for col, header in MANUAL_FLAG_COLS.items():
        df_out[col] = ''          # blank for the reviewer

    df_out.to_csv(path, index=False, encoding='utf-8')


def write_reference_file(sample: pd.DataFrame, path: Path) -> None:
    """
    ai_flags_reference.csv — AI flags + ground-truth label.
    Open AFTER completing manual review.
    """
    ref_cols = ['sample_id', 'job_id'] + FLAG_COLS + [
        'risk_score_all', 'single_pass_risk_score', 'fraudulent_agent'
    ]
    df_out = _keep(sample, ref_cols)
    df_out.to_csv(path, index=False, encoding='utf-8')


def write_summary(sample: pd.DataFrame,
                  coverage: dict,
                  target: int,
                  path: Path) -> None:
    """Plain-text summary of coverage and key statistics."""
    lines = [
        "MANUAL EVALUATION SAMPLE — SAMPLING SUMMARY",
        "=" * 60,
        f"Total postings selected : {len(sample)}",
        f"Target per flag         : {target}",
        "",
        "FLAG COVERAGE IN SAMPLE",
        "-" * 60,
        f"{'Flag':<36} {'AI pos':>7}  {'% of sample':>12}  {'Status':>10}",
    ]

    for col, label in FLAG_LABELS.items():
        if col not in sample.columns:
            lines.append(f"  {label:<34}  {'N/A':>7}")
            continue
        n   = int(sample[col].sum())
        pct = n / len(sample) * 100
        status = 'OK' if n >= target else f'short by {target - n}'
        lines.append(
            f"  {label:<34}  {n:>5}  {pct:>11.1f}%  {status:>10}"
        )

    # Fraud composition
    lines += ["", "FRAUD / LEGIT COMPOSITION", "-" * 60]
    if 'fraudulent_agent' in sample.columns:
        n_fraud = int(sample['fraudulent_agent'].sum())
        n_legit = len(sample) - n_fraud
        lines.append(f"  Fraudulent : {n_fraud} ({n_fraud/len(sample)*100:.1f}%)")
        lines.append(f"  Legitimate : {n_legit} ({n_legit/len(sample)*100:.1f}%)")

    # Flag count distribution
    avail = [c for c in FLAG_COLS if c in sample.columns]
    if avail:
        sample['_nflags'] = sample[avail].sum(axis=1)
        dist = sample['_nflags'].value_counts().sort_index()
        lines += ["", "FLAG COUNT DISTRIBUTION PER POSTING", "-" * 60]
        for n_flags, count in dist.items():
            lines.append(f"  {int(n_flags)} flags : {count} postings")

    lines += [
        "",
        "HOW TO USE THESE FILES",
        "-" * 60,
        "1. Open  sample_for_review.csv  in Excel / any spreadsheet app.",
        "2. Read each posting carefully.",
        "3. For each of the 8 manual_flag* columns, enter 1 (yes) or 0 (no).",
        "4. Save your completed file.",
        "5. Open  ai_flags_reference.csv  to compare your flags with the AI.",
        "",
        "Column mapping (manual → AI):",
    ]
    for mcol, acol in zip(MANUAL_FLAG_COLS.keys(), FLAG_COLS):
        lines.append(f"  {mcol}  ↔  {acol}")

    path.write_text("\n".join(lines), encoding='utf-8')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 62)
    print("  Generate Manual Evaluation Sample (n=50)")
    print("=" * 62)

    if not DATA_PATH.exists():
        sys.exit(f"\nData file not found:\n  {DATA_PATH}\n")

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"\nLoading {DATA_PATH.name} …")
    df = pd.read_csv(DATA_PATH, encoding='latin-1', low_memory=False)
    print(f"  {len(df):,} rows  |  {len(df.columns)} columns")

    # Normalise flag columns to int
    for col in FLAG_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # ── Print flag availability ───────────────────────────────────────────
    print("\n  Flag positives in full dataset:")
    for col in FLAG_COLS:
        if col in df.columns:
            n = int(df[col].sum())
            print(f"    {col:<30s}: {n:5d}  ({n/len(df)*100:.1f}%)")

    # ── Greedy sampling ───────────────────────────────────────────────────
    print(f"\n  Running greedy sampler "
          f"(n={N_TOTAL}, target={TARGET_PER_FLAG} per flag) …")
    rng = np.random.default_rng(RANDOM_SEED)
    selected_idx = greedy_sample(df, FLAG_COLS, N_TOTAL, TARGET_PER_FLAG, rng)
    sample = df.loc[selected_idx].copy().reset_index(drop=True)
    sample.insert(0, 'sample_id', range(1, len(sample) + 1))

    # ── Coverage report ───────────────────────────────────────────────────
    print(f"\n  Coverage achieved ({len(sample)} postings):")
    coverage = {}
    for col in FLAG_COLS:
        if col in sample.columns:
            n = int(sample[col].sum())
            coverage[col] = n
            label = FLAG_LABELS[col]
            status = 'OK' if n >= TARGET_PER_FLAG else f'⚠ short by {TARGET_PER_FLAG-n}'
            print(f"    {label:<36}: {n:2d}  {status}")

    if 'fraudulent_agent' in sample.columns:
        nf = int(sample['fraudulent_agent'].sum())
        print(f"\n  Fraud / Legit: {nf} / {len(sample)-nf}")

    # ── Write outputs ─────────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    review_path = OUT_DIR / 'sample_for_review.csv'
    ref_path    = OUT_DIR / 'ai_flags_reference.csv'
    summ_path   = OUT_DIR / 'sampling_summary.txt'

    write_review_file(sample, review_path)
    write_reference_file(sample, ref_path)
    write_summary(sample, coverage, TARGET_PER_FLAG, summ_path)

    print(f"\n  Output folder: {OUT_DIR.name}/")
    print(f"    sample_for_review.csv   ← open this for manual review")
    print(f"    ai_flags_reference.csv  ← open AFTER manual review to compare")
    print(f"    sampling_summary.txt    ← coverage and usage instructions")
    print()
    print("=" * 62)
    print("  Done.")
    print("=" * 62)


if __name__ == "__main__":
    main()
