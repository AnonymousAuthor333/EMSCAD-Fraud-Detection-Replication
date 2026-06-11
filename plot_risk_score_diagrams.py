#!/usr/bin/env python3
"""
plot_risk_score_diagrams.py

Generates a 2x2 figure comparing the single-pass evaluation against the
multi-agent framework:

  Top row    — Single-Pass Evaluation (single_pass_risk_score)
  Bottom row — Multi-Agent Framework  (risk_score_all)

  Left column  — distribution of risk scores (how many postings per score)
  Right column — fraud rate at each score   (% of postings that are fraud)

Comparability design
─────────────────────────────────────────────────────────────────────
  • All four panels share the same x-axis: scores 0-10
  • The two distribution panels share the same y-axis maximum
  • The two fraud-rate panels both run 0-100 %
  This makes the two approaches directly visually comparable.

Panel titles (chosen to be self-explanatory)
─────────────────────────────────────────────────────────────────────
  (A) Single-Pass Evaluation: Risk Score Distribution
  (B) Single-Pass Evaluation: Fraud Rate by Risk Score
  (C) Multi-Agent Framework: Risk Score Distribution
  (D) Multi-Agent Framework: Fraud Rate by Risk Score

Output
─────────────────────────────────────────────────────────────────────
  diagram/risk_score_comparison.png   (300 dpi)
  diagram/risk_score_comparison.pdf   (vector, for publication)

Usage
─────────────────────────────────────────────────────────────────────
  python plot_risk_score_diagrams.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # no display needed; render to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent
DATA_PATH = _HERE / "EMSCAD_fraud_detection.csv"
OUT_DIR   = _HERE / "diagram"

# ── Column configuration ───────────────────────────────────────────────────────
SINGLE_COL = "single_pass_risk_score"
MULTI_COL  = "risk_score_all"
TARGET     = "fraudulent_agent"

SCORES = list(range(0, 11))          # 0 … 10 on every panel

# Colours: blue for counts, orange for fraud rates (matches the original mock-up)
BAR_BLUE   = "#5B84C4"
BAR_ORANGE = "#F0A258"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="latin-1", low_memory=False)
    if TARGET not in df.columns and "fraudulent" in df.columns:
        df = df.rename(columns={"fraudulent": TARGET})
    for col in (SINGLE_COL, MULTI_COL, TARGET):
        if col not in df.columns:
            sys.exit(f"Required column '{col}' not found in {DATA_PATH.name}.")
    return df


def score_stats(df: pd.DataFrame, score_col: str):
    """
    Return (counts, fraud_pct) aligned to SCORES.
      counts[i]    — number of postings with score == SCORES[i]
      fraud_pct[i] — % of those postings that are fraudulent (NaN if none)
    """
    s = pd.to_numeric(df[score_col], errors="coerce")
    y = df[TARGET].astype(int)
    valid = s.notna()
    s = s[valid].round().astype(int).clip(0, 10)
    y = y[valid]

    counts, fraud_pct = [], []
    for sc in SCORES:
        mask = s == sc
        n = int(mask.sum())
        counts.append(n)
        fraud_pct.append(y[mask].mean() * 100 if n > 0 else np.nan)
    return counts, fraud_pct


def main() -> None:
    print("=" * 60)
    print("  Risk Score Comparison Diagrams")
    print("=" * 60)

    if not DATA_PATH.exists():
        sys.exit(f"\nData file not found:\n  {DATA_PATH}\n")

    df = load_data()
    n_total = len(df)
    n_fraud = int(df[TARGET].sum())
    print(f"\n  {n_total:,} postings  |  {n_fraud:,} fraudulent "
          f"({n_fraud / n_total * 100:.2f} %)")

    sp_counts, sp_fraud = score_stats(df, SINGLE_COL)
    ma_counts, ma_fraud = score_stats(df, MULTI_COL)

    n_sp = sum(sp_counts)
    n_ma = sum(ma_counts)
    print(f"  Single-pass scores available : {n_sp:,}")
    print(f"  Multi-agent scores available : {n_ma:,}")

    # Shared y-limit for the two distribution panels (same scale = comparable)
    count_ymax = max(max(sp_counts), max(ma_counts)) * 1.08

    # ── Build the 2x2 figure ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    panels = [
        (ax_a, sp_counts, None,
         "(A) Single-Pass Evaluation: Risk Score Distribution",
         "Number of Job Postings", BAR_BLUE),
        (ax_b, None, sp_fraud,
         "(B) Single-Pass Evaluation: Fraud Rate by Risk Score",
         "Fraudulent Postings (%)", BAR_ORANGE),
        (ax_c, ma_counts, None,
         "(C) Multi-Agent Framework: Risk Score Distribution",
         "Number of Job Postings", BAR_BLUE),
        (ax_d, None, ma_fraud,
         "(D) Multi-Agent Framework: Fraud Rate by Risk Score",
         "Fraudulent Postings (%)", BAR_ORANGE),
    ]

    for ax, counts, fraud, title, ylabel, color in panels:
        vals = counts if counts is not None else fraud
        ax.bar(SCORES, vals, width=0.82, color=color,
               edgecolor="#3A3A3A", linewidth=0.4, zorder=3)
        ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
        ax.set_xlabel("Risk Score", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(SCORES)
        ax.set_xlim(-0.7, 10.7)
        ax.grid(axis="y", linestyle=":", color="#BBBBBB", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if counts is not None:
            ax.set_ylim(0, count_ymax)          # shared scale: panels A & C
        else:
            ax.set_ylim(0, 100)                 # shared scale: panels B & D

    fig.suptitle(
        "Fraud Risk Scores: Single-Pass Evaluation vs. Multi-Agent Framework",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    # ── Save ──────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    png_path = OUT_DIR / "risk_score_comparison.png"
    pdf_path = OUT_DIR / "risk_score_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Saved → {png_path.relative_to(_HERE)}")
    print(f"  Saved → {pdf_path.relative_to(_HERE)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
