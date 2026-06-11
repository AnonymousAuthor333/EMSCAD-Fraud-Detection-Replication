#!/usr/bin/env python3
"""
statistical_comparisons.py

Paired t-tests comparing Monte Carlo cross-validation results across
the six LLM-augmentation groups, for all six ML models.

Comparison groups (from model_evaluation.py)
─────────────────────────────────────────────────────────────────────
  base              — no LLM score (TF-IDF + salary only)
  single_pass       — base + single_pass_risk_score
  multi_agent_all   — base + risk_score_all  (all 3 sub-agents)
  multi_agent_lc    — base + risk_score_legit_context  (2 sub-agents)
  multi_agent_cc    — base + risk_score_context_consist (2 sub-agents)
  multi_agent_ls    — base + risk_score_legit_consist   (2 sub-agents)

Paired comparisons run for every model
─────────────────────────────────────────────────────────────────────
  PRIMARY (does LLM augmentation help?)
    base              → single_pass
    base              → multi_agent_all
    single_pass       → multi_agent_all

  ABLATION (does using all 3 agents matter vs. 2?)
    multi_agent_all   → multi_agent_lc
    multi_agent_all   → multi_agent_cc
    multi_agent_all   → multi_agent_ls

Each pair reports: mean_A, mean_B, mean_diff (B-A), paired t-statistic,
p-value, Cohen's d, 95% CI, and a significance flag (p < 0.05).

Reported metrics: accuracy, recall, f1, pr_auc

Usage
─────────────────────────────────────────────────────────────────────
  python statistical_comparisons.py

  If multiple result sets exist (different rs_labels), the program
  lists them and asks which to analyse.

Output (saved to results/)
─────────────────────────────────────────────────────────────────────
  statistical_comparisons_{rs_label}.csv   — all pair × metric × model rows
"""

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / "results"

# ── Metrics and model names ────────────────────────────────────────────────────
METRICS = ['accuracy', 'recall', 'f1', 'pr_auc']

# Canonical model prefixes (must match model_evaluation.py MODEL_FILE_NAMES values)
MODEL_FILE_LABELS = ['LR', 'ENLR', 'RF', 'XGBoost', 'MLP', 'BERT_MLP']

# Internal prefix used in column names (lower-case, underscore)
MODEL_COL_PREFIX = {
    'LR': 'lr', 'ENLR': 'enlr', 'RF': 'rf',
    'XGBoost': 'xgb', 'MLP': 'mlp', 'BERT_MLP': 'bert_mlp',
}

# ── Comparison pairs (group_A, group_B) — mean_diff = mean_B - mean_A ─────────
COMPARISONS = [
    # Primary: does any LLM score improve over base?
    ('base',            'single_pass'),
    ('base',            'multi_agent_all'),
    ('single_pass',     'multi_agent_all'),

    # Ablation: does using all 3 agents matter vs. each 2-agent subset?
    ('multi_agent_lc',  'multi_agent_all'),
    ('multi_agent_cc',  'multi_agent_all'),
    ('multi_agent_ls',  'multi_agent_all'),
]

COMPARISON_LABEL = {
    ('base',           'single_pass'):    'base → single_pass',
    ('base',           'multi_agent_all'):'base → multi_agent_all',
    ('single_pass',    'multi_agent_all'):'single_pass → multi_agent_all',
    ('multi_agent_lc', 'multi_agent_all'):'multi_agent_lc → multi_agent_all (ablation)',
    ('multi_agent_cc', 'multi_agent_all'):'multi_agent_cc → multi_agent_all (ablation)',
    ('multi_agent_ls', 'multi_agent_all'):'multi_agent_ls → multi_agent_all (ablation)',
}


# ══════════════════════════════════════════════════════════════════════════════
# File discovery
# ══════════════════════════════════════════════════════════════════════════════

def _find_rs_labels() -> list:
    """
    Scan results/ for files matching {MODEL}_results_{rs_label}.csv
    and return the unique rs_labels found.
    """
    pattern = re.compile(r'^[A-Za-z_]+_results_(rs\d+-\d+)\.csv$')
    labels  = set()
    for f in RESULTS_DIR.glob("*_results_rs*.csv"):
        m = pattern.match(f.name)
        if m:
            labels.add(m.group(1))
    return sorted(labels)


def _load_results(rs_label: str) -> dict:
    """
    Load all available model result CSVs for the given rs_label.
    Returns {file_label: DataFrame} for models that have a file.
    """
    loaded = {}
    for label in MODEL_FILE_LABELS:
        path = RESULTS_DIR / f"{label}_results_{rs_label}.csv"
        if path.exists():
            loaded[label] = pd.read_csv(path)
    return loaded


# ══════════════════════════════════════════════════════════════════════════════
# Paired t-test logic
# ══════════════════════════════════════════════════════════════════════════════

def _paired_test(data_a: np.ndarray, data_b: np.ndarray) -> dict:
    """
    Run a two-tailed paired t-test and compute Cohen's d + 95 % CI.
    mean_diff = mean(B) - mean(A).
    """
    diff     = data_b - data_a
    n        = len(diff)
    mean_a   = float(np.mean(data_a))
    mean_b   = float(np.mean(data_b))
    mean_d   = float(np.mean(diff))
    std_d    = float(np.std(diff, ddof=1))
    t_stat, p_val = stats.ttest_rel(data_b, data_a)
    cohens_d = mean_d / std_d if std_d > 0 else 0.0
    margin   = stats.t.ppf(0.975, n - 1) * (std_d / np.sqrt(n))
    return {
        'mean_A':      round(mean_a,  6),
        'mean_B':      round(mean_b,  6),
        'mean_diff':   round(mean_d,  6),
        't_statistic': round(float(t_stat), 4),
        'p_value':     round(float(p_val),  4),
        'cohens_d':    round(cohens_d,      4),
        'ci_95_lower': round(mean_d - margin, 6),
        'ci_95_upper': round(mean_d + margin, 6),
        'significant': 'Yes' if float(p_val) < 0.05 else 'No',
    }


def run_comparisons_for_model(file_label: str,
                               df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all COMPARISONS × METRICS for one model DataFrame.
    Returns a tidy result DataFrame.
    """
    prefix = MODEL_COL_PREFIX[file_label]
    rows   = []

    for (grp_a, grp_b) in COMPARISONS:
        comp_label = COMPARISON_LABEL[(grp_a, grp_b)]
        for metric in METRICS:
            col_a = f'{prefix}_{grp_a}_{metric}'
            col_b = f'{prefix}_{grp_b}_{metric}'
            if col_a not in df.columns or col_b not in df.columns:
                # Group or metric not present in this result file — skip
                continue
            result = _paired_test(df[col_a].values, df[col_b].values)
            rows.append({
                'model':      file_label,
                'comparison': comp_label,
                'group_A':    grp_a,
                'group_B':    grp_b,
                'metric':     metric,
                'n_iters':    len(df),
                **result,
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 66)
    print("  EMSCAD Fraud Detection — Statistical Comparisons")
    print("=" * 66)

    if not RESULTS_DIR.exists():
        sys.exit(f"\nResults folder not found:\n  {RESULTS_DIR}\n"
                 "Run model_evaluation.py first.\n")

    # ── Discover available rs_labels ──────────────────────────────────────
    labels = _find_rs_labels()
    if not labels:
        sys.exit("\nNo result files found in results/.\n"
                 "Run model_evaluation.py first.\n")

    if len(labels) == 1:
        rs_label = labels[0]
        print(f"\nUsing result set: {rs_label}")
    else:
        print(f"\nMultiple result sets found:")
        for i, lbl in enumerate(labels, 1):
            print(f"  [{i}] {lbl}")
        try:
            choice = int(input("Select a set to analyse (number): ").strip())
            rs_label = labels[choice - 1]
        except (ValueError, IndexError):
            sys.exit("Invalid selection.")

    # ── Load result files ─────────────────────────────────────────────────
    model_dfs = _load_results(rs_label)
    if not model_dfs:
        sys.exit(f"No result files found for rs_label={rs_label}.")

    print(f"\nLoaded result files:")
    for label, df in model_dfs.items():
        print(f"  {label:10s}: {len(df)} iterations × {len(df.columns)} columns")

    # ── Run comparisons ───────────────────────────────────────────────────
    print(f"\nRunning paired t-tests …")
    print(f"  Comparisons : {len(COMPARISONS)}")
    print(f"  Metrics     : {METRICS}")
    print()

    all_results = []
    for file_label, df in model_dfs.items():
        result_df = run_comparisons_for_model(file_label, df)
        n = len(result_df)
        sig = (result_df['significant'] == 'Yes').sum()
        print(f"  {file_label:10s}: {n:3d} tests  |  {sig:2d} significant (p < 0.05)")
        all_results.append(result_df)

    df_out = pd.concat(all_results, ignore_index=True)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"statistical_comparisons_{rs_label}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path.name}  ({df_out.shape})")

    # ── Print readable summary ────────────────────────────────────────────
    print()
    print("=" * 66)
    print("  SIGNIFICANT RESULTS (p < 0.05)")
    print("=" * 66)
    sig_df = df_out[df_out['significant'] == 'Yes'].copy()
    if sig_df.empty:
        print("  None found.")
    else:
        # Show condensed view
        display_cols = ['model', 'comparison', 'metric',
                        'mean_A', 'mean_B', 'mean_diff',
                        'p_value', 'cohens_d']
        with pd.option_context('display.width', 120,
                               'display.float_format', '{:.4f}'.format,
                               'display.max_rows', 200):
            print(sig_df[display_cols].to_string(index=False))

    # ── Summary counts by comparison type ─────────────────────────────────
    print()
    print("=" * 66)
    print("  SIGNIFICANCE COUNTS BY COMPARISON")
    print("  (significant / total tests across all models and metrics)")
    print("=" * 66)
    summary = (df_out.groupby('comparison')
               .apply(lambda g: pd.Series({
                   'significant': (g['significant'] == 'Yes').sum(),
                   'total':       len(g),
               }))
               .assign(pct=lambda d: (d['significant'] / d['total'] * 100).round(1))
               .reset_index())
    for _, row in summary.iterrows():
        print(f"  {row['comparison']}")
        print(f"    {int(row['significant'])}/{int(row['total'])} "
              f"significant ({row['pct']:.1f} %)")
        print()

    print(f"  Full results → results/{out_path.name}")
    print("=" * 66)


if __name__ == "__main__":
    main()
