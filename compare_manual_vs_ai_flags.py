#!/usr/bin/env python3
"""
compare_manual_vs_ai_flags.py

Compares manually labelled flags against AI-labelled flags for the
50-posting evaluation sample in flags_manual_eval/.

Inputs (read-only — never modified)
─────────────────────────────────────────────────────────────────────
  flags_manual_eval/sample_for_review.csv   manual labels (filled in by reviewer)
  flags_manual_eval/ai_flags_reference.csv  AI labels for the same postings

Rows are matched on sample_id. Any extra rows (e.g. a totals/sum row
added at the bottom of the spreadsheet) are ignored automatically.

For each of the 8 flags the comparison reports:
  AI Positive     — number of postings the AI flagged
  Human Positive  — number of postings the reviewer flagged
  TP  (true positive)   — both AI and human flagged
  TN  (true negative)   — neither flagged
  FP  (false positive)  — AI flagged, human did not
  FN  (false negative)  — human flagged, AI did not
  Agreement (%)         — (TP + TN) / N
  Cohen's kappa         — chance-corrected agreement

Convention: the human label is treated as the reference standard, so
"false positive" means the AI flagged something the human did not.

Output (saved to flags_manual_eval/)
─────────────────────────────────────────────────────────────────────
  manual_vs_ai_comparison.csv     per-flag summary table
  manual_vs_ai_disagreements.csv  row-level list of every disagreement
                                  (sample_id, job_id, flag, AI label, human label)

Usage
─────────────────────────────────────────────────────────────────────
  python compare_manual_vs_ai_flags.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
EVAL_DIR    = _HERE / "flags_manual_eval"
REVIEW_PATH = EVAL_DIR / "sample_for_review.csv"
AI_PATH     = EVAL_DIR / "ai_flags_reference.csv"
OUT_SUMMARY = EVAL_DIR / "manual_vs_ai_comparison.csv"
OUT_DETAIL  = EVAL_DIR / "manual_vs_ai_disagreements.csv"

# ── Flag mapping: AI column ↔ manual column ↔ display name ────────────────────
FLAGS = [
    ("flag1_sensitive_info",   "manual_flag1_sensitive_info",   "Sensitive Information Request"),
    ("flag2_unrealistic_comp", "manual_flag2_unrealistic_comp", "Unrealistic Compensation"),
    ("flag3_pressure_tactics", "manual_flag3_pressure_tactics", "Pressure Tactics"),
    ("flag4_anon_employer",    "manual_flag4_anon_employer",    "Unidentifiable Employer"),
    ("flag5_req_role_mismatch","manual_flag5_req_role_mismatch","Requirements Role Mismatch"),
    ("flag6_info_conflict",    "manual_flag6_info_conflict",    "Internal Information Conflict"),
    ("flag7_vague_info",       "manual_flag7_vague_info",       "Vague Information"),
    ("flag8_promo_template",   "manual_flag8_promo_template",   "Promotional Template Substitution"),
]


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV trying utf-8 first, then latin-1 (Excel re-saves change encoding)."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    sys.exit(f"Could not read {path.name} with utf-8 or latin-1 encoding.")


def _cohens_kappa(tp: int, tn: int, fp: int, fn: int) -> float:
    """Cohen's kappa from a 2x2 confusion matrix."""
    n = tp + tn + fp + fn
    if n == 0:
        return np.nan
    po = (tp + tn) / n                                   # observed agreement
    p_yes = ((tp + fp) / n) * ((tp + fn) / n)            # chance both say yes
    p_no  = ((tn + fn) / n) * ((tn + fp) / n)            # chance both say no
    pe = p_yes + p_no
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main() -> None:
    print("=" * 66)
    print("  Manual vs. AI Flag Comparison")
    print("=" * 66)

    for p in (REVIEW_PATH, AI_PATH):
        if not p.exists():
            sys.exit(f"\nRequired file not found:\n  {p}\n")

    # ── Load (read-only) ──────────────────────────────────────────────────
    manual = _read_csv(REVIEW_PATH)
    ai     = _read_csv(AI_PATH)
    print(f"\n  {REVIEW_PATH.name}: {len(manual)} rows")
    print(f"  {AI_PATH.name}: {len(ai)} rows")

    # ── Keep only valid sample rows (drops totals/sum rows etc.) ──────────
    valid_ids = set(ai["sample_id"].astype(int))
    manual = manual[pd.to_numeric(manual["sample_id"], errors="coerce")
                    .isin(valid_ids)].copy()
    manual["sample_id"] = manual["sample_id"].astype(int)
    if len(manual) != len(ai):
        print(f"  Note: kept {len(manual)} manual rows matching AI sample_ids "
              f"(extra rows such as totals were ignored)")

    # ── Merge on sample_id ────────────────────────────────────────────────
    merged = ai.merge(
        manual[["sample_id"] + [m for _, m, _ in FLAGS]],
        on="sample_id", how="inner",
    )
    n = len(merged)
    print(f"  Matched postings: {n}")
    if n == 0:
        sys.exit("No rows matched on sample_id — check the files.")

    # ── Per-flag comparison ───────────────────────────────────────────────
    summary_rows = []
    detail_rows  = []

    for ai_col, man_col, label in FLAGS:
        if ai_col not in merged.columns or man_col not in merged.columns:
            print(f"  Skipping {label}: column missing")
            continue

        a = pd.to_numeric(merged[ai_col],  errors="coerce")
        h = pd.to_numeric(merged[man_col], errors="coerce")
        ok = a.isin([0, 1]) & h.isin([0, 1])
        n_used = int(ok.sum())

        av = a[ok].astype(int).values
        hv = h[ok].astype(int).values

        tp = int(((av == 1) & (hv == 1)).sum())
        tn = int(((av == 0) & (hv == 0)).sum())
        fp = int(((av == 1) & (hv == 0)).sum())
        fn = int(((av == 0) & (hv == 1)).sum())

        agreement = (tp + tn) / n_used * 100 if n_used else np.nan
        kappa     = _cohens_kappa(tp, tn, fp, fn)

        summary_rows.append({
            "Flag":           label,
            "AI Column":      ai_col,
            "N Compared":     n_used,
            "AI Positive":    int((av == 1).sum()),
            "Human Positive": int((hv == 1).sum()),
            "True Positive":  tp,
            "True Negative":  tn,
            "False Positive": fp,
            "False Negative": fn,
            "Agreement (%)":  round(agreement, 1),
            "Cohens Kappa":   round(kappa, 3),
        })

        # Row-level disagreements
        dis = merged[ok.values][["sample_id", "job_id"]].copy()
        dis["ai"]    = av
        dis["human"] = hv
        dis = dis[dis["ai"] != dis["human"]]
        for _, r in dis.iterrows():
            detail_rows.append({
                "sample_id":   int(r["sample_id"]),
                "job_id":      r["job_id"],
                "flag":        label,
                "ai_label":    int(r["ai"]),
                "human_label": int(r["human"]),
                "type":        "AI flagged, human did not (FP)"
                               if r["ai"] == 1 else
                               "Human flagged, AI did not (FN)",
            })

    df_summary = pd.DataFrame(summary_rows)
    df_detail  = (pd.DataFrame(detail_rows)
                  .sort_values(["flag", "sample_id"])
                  if detail_rows else pd.DataFrame(
                      columns=["sample_id","job_id","flag",
                               "ai_label","human_label","type"]))

    # ── Save (inputs are never written) ───────────────────────────────────
    df_summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8")
    df_detail.to_csv(OUT_DETAIL,  index=False, encoding="utf-8")

    # ── Print results ─────────────────────────────────────────────────────
    print()
    print("=" * 66)
    print("  PER-FLAG COMPARISON  (human label = reference)")
    print("=" * 66)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(df_summary.drop(columns=["AI Column"]).to_string(index=False))

    total_dis = len(df_detail)
    print()
    print(f"  Total disagreements across all flags: {total_dis}")
    if total_dis:
        by_type = df_detail["type"].value_counts()
        for t, c in by_type.items():
            print(f"    {t}: {c}")

    print()
    print(f"  Saved → {OUT_SUMMARY.name}")
    print(f"  Saved → {OUT_DETAIL.name}")
    print("=" * 66)


if __name__ == "__main__":
    main()
