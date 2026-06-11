#!/usr/bin/env python3
"""
model_evaluation.py
Monte Carlo cross-validation comparing baseline vs. LLM-augmented fraud classifiers.

Six comparison groups per model
─────────────────────────────────────────────────────────────────────
  base                  — no LLM score (salary features only)
  single_pass           — base + single_pass_risk_score
  multi_agent_all       — base + risk_score_all (all 3 sub-agents)
  multi_agent_lc        — base + risk_score_legit_context
  multi_agent_cc        — base + risk_score_context_consist
  multi_agent_ls        — base + risk_score_legit_consist

ML models evaluated (5 classical + 1 BERT)
─────────────────────────────────────────────────────────────────────
  LR        Logistic Regression        (sparse TF-IDF, std numeric)
  ENLR      Elastic-Net LR             (dense TF-IDF+SVD, std numeric)
  RF        Random Forest              (dense TF-IDF+SVD, raw numeric)
  XGBoost                              (dense TF-IDF+SVD, raw numeric)
  MLP       Multi-Layer Perceptron     (dense TF-IDF+SVD, std numeric)
  BERT-MLP  BERT CLS + MLP             (required — uses CPU if no GPU)

Experiment design
─────────────────────────────────────────────────────────────────────
  50 iterations of stratified 80/20 Monte Carlo CV
  SMOTE applied to training data only
  Parallel across iterations (joblib)

Reported metrics: accuracy, recall, F1, PR-AUC  (no other metrics)

Outputs (saved in results/ subfolder)
─────────────────────────────────────────────────────────────────────
  results/LR_results.csv
  results/ENLR_results.csv
  results/RF_results.csv
  results/XGBoost_results.csv
  results/MLP_results.csv
  results/BERT_MLP_results.csv
  results/summary_means.csv      (mean across 50 iterations, all models)
"""

import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Local imports (same folder only) ──────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import data_helpers as hdp
import feature_helpers as fh

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH       = _HERE / "EMSCAD_fraud_detection.csv"
RESULTS_DIR     = _HERE / "results"
BERT_CLS_PATH   = _HERE / "results" / "bert_cls_embeddings.npy"

# ── Experiment settings ────────────────────────────────────────────────────────
N_ITERATIONS      = 50
RANDOM_STATE_START = 1   # ← change this to use a different set of random states
                          #   e.g. 1 = states 1-50 (first run)
                          #        51 = states 51-100 (second run)
                          #        101 = states 101-150 (third run)
TEST_SIZE    = 0.20
N_JOBS       = -1         # parallel workers: -1 = all CPU cores

# ── Comparison groups ──────────────────────────────────────────────────────────
BASE_NUM = ['salary_lower', 'salary_upper']

GROUPS = {
    'base':               BASE_NUM,
    'single_pass':        BASE_NUM + ['single_pass_risk_score'],
    'multi_agent_all':    BASE_NUM + ['risk_score_all'],
    'multi_agent_lc':     BASE_NUM + ['risk_score_legit_context'],
    'multi_agent_cc':     BASE_NUM + ['risk_score_context_consist'],
    'multi_agent_ls':     BASE_NUM + ['risk_score_legit_consist'],
}

# ── Metrics (only the four requested) ─────────────────────────────────────────
METRICS = ['accuracy', 'recall', 'f1', 'pr_auc']

def eval_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'recall':   recall_score(y_true, y_pred, zero_division=0),
        'f1':       f1_score(y_true, y_pred, zero_division=0),
        'pr_auc':   average_precision_score(y_true, y_prob),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Data loading and salary preprocessing
# ══════════════════════════════════════════════════════════════════════════════

_MONTH_MAP = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
    'mar': 3, 'march': 3, 'apr': 4, 'april': 4, 'may': 5,
    'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
    'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
    'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
    'dec': 12, 'december': 12,
}

def _parse_salary(value):
    if pd.isna(value):
        return None, None
    s = str(value).strip().lower()
    for m, v in _MONTH_MAP.items():
        if m in s:
            return float(v), float(v)
    nums = re.findall(r'\d+', s)
    if not nums:
        return None, None
    if len(nums) == 1:
        n = float(nums[0]); return n, n
    return float(nums[0]), float(nums[1])

def load_and_preprocess(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='latin-1', low_memory=False)
    df = df.rename(columns={'fraudulent_agent': 'fraudulent'})

    df['salary_range_missing'] = df['salary_range'].isna().astype(int)
    parsed = df['salary_range'].apply(lambda x: pd.Series(_parse_salary(x),
                                                           index=['salary_lower', 'salary_upper']))
    df['salary_lower'] = parsed['salary_lower'].fillna(parsed['salary_lower'].mean())
    df['salary_upper'] = parsed['salary_upper'].fillna(parsed['salary_upper'].mean())

    df = df.reset_index(drop=True)
    return df

def _check_required_columns(df: pd.DataFrame) -> None:
    score_cols = ['single_pass_risk_score', 'risk_score_all',
                  'risk_score_legit_context', 'risk_score_context_consist',
                  'risk_score_legit_consist']
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        print(f"\n  WARNING: missing score columns: {missing}")
        print("  Run fraud_detector.py and single_pass_detector.py first.")
        print("  Groups depending on missing columns will be skipped.\n")


# ══════════════════════════════════════════════════════════════════════════════
# Optimised feature caching within one iteration
# ══════════════════════════════════════════════════════════════════════════════

def _build_base_features(train_df, test_df, text_mode, num_mode):
    """
    Fit text + boolean + categorical transformers on train (NOT numeric).
    Returns (X_text_train, X_text_test, X_bool_train, X_bool_test,
             X_cat_train,  X_cat_test,  bool_imp, cat_encoders, cat_meta).
    Numeric is handled separately per comparison group to avoid re-fitting.
    """
    if text_mode == 'dense':
        Xtr_d, vec, svd, norm = hdp.make_denseTFIDF_SVD(train_df)
        X_text_tr = csr_matrix(Xtr_d)
        X_text_te = csr_matrix(hdp.transform_denseTFIDF_SVD(test_df, vec, svd, norm))
    else:
        X_text_tr, vec = hdp.make_sparseTFIDF(train_df)
        X_text_te = hdp.transform_sparseTFIDF(test_df, vec)

    X_bool_tr, imp_bool = hdp.make_bool(train_df)
    X_bool_te = hdp.transform_bool(test_df, imp_bool)

    X_cat_tr, cat_enc, cat_meta = hdp.make_onehot_topk_train(train_df)
    X_cat_te = hdp.transform_onehot_topk_test(test_df, cat_enc, cat_meta)

    return (X_text_tr, X_text_te, X_bool_tr, X_bool_te,
            X_cat_tr,  X_cat_te, imp_bool, cat_enc, cat_meta)


def _add_num(train_df, test_df, num_cols, num_mode, base_num_cols):
    """Fit numeric block (impute/scale) for the given group's column list."""
    # Limit to columns that exist in the data
    valid = [c for c in num_cols if c in train_df.columns]
    if valid != num_cols:
        return None, None   # signal to skip this group

    if num_mode == 'raw':
        X_num_tr, imp = hdp.make_rawNum(train_df, valid)
        X_num_te = hdp.transform_rawNum(test_df, imp, valid)
    else:
        X_num_tr, imp, scl = hdp.make_stdNum(train_df, valid)
        X_num_te = hdp.transform_stdNum(test_df, imp, scl, valid)
    return X_num_tr, X_num_te


def _assemble(X_text, X_num, X_bool, X_cat):
    return hstack([X_text, X_num, X_bool, X_cat], format='csr')


# ══════════════════════════════════════════════════════════════════════════════
# Single-iteration worker (runs in parallel)
# ══════════════════════════════════════════════════════════════════════════════

def _one_iteration(rs: int, df: pd.DataFrame, models_cfg: list,
                   cls_all=None) -> dict:
    """
    Run one 80/20 split and evaluate all model × group combinations.
    Returns a flat dict of metric values (one row of the results CSV).
    """
    y = df['fraudulent'].astype(int).values
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=TEST_SIZE, stratify=y, random_state=rs
    )

    row = {'random_state': rs}

    # Group models by their (text_mode, num_mode) to share base features
    from itertools import groupby
    cfg_groups = {}
    for cfg in models_cfg:
        key = (cfg['text_mode'], cfg['num_mode'])
        cfg_groups.setdefault(key, []).append(cfg)

    for (text_mode, num_mode), cfgs in cfg_groups.items():
        # Build shared text + bool + cat base (fit once per feature config)
        if text_mode == 'bert':
            # BERT: pre-extracted CLS embeddings replace the text block
            tr_idx = train_df.index.values
            te_idx = test_df.index.values
            X_text_tr = csr_matrix(cls_all[tr_idx])
            X_text_te = csr_matrix(cls_all[te_idx])
            X_bool_tr, imp_bool = hdp.make_bool(train_df)
            X_bool_te = hdp.transform_bool(test_df, imp_bool)
            X_cat_tr, cat_enc, cat_meta = hdp.make_onehot_topk_train(train_df)
            X_cat_te = hdp.transform_onehot_topk_test(test_df, cat_enc, cat_meta)
        else:
            (X_text_tr, X_text_te,
             X_bool_tr, X_bool_te,
             X_cat_tr,  X_cat_te,
             imp_bool, cat_enc, cat_meta) = _build_base_features(
                train_df, test_df, text_mode, num_mode)

        for cfg in cfgs:
            model_name = cfg['name']
            num_mode_  = 'std' if text_mode == 'bert' else num_mode

            for group_name, num_cols in GROUPS.items():
                # Build numeric block for this group
                X_num_tr, X_num_te = _add_num(
                    train_df, test_df, num_cols, num_mode_, BASE_NUM)
                if X_num_tr is None:
                    continue   # required column not in data — skip group

                X_tr = _assemble(X_text_tr, X_num_tr, X_bool_tr, X_cat_tr)
                X_te = _assemble(X_text_te, X_num_te, X_bool_te, X_cat_te)

                # SMOTE
                sm = SMOTE(random_state=rs)
                X_bal, y_bal = sm.fit_resample(X_tr, y_train)

                # Train & predict
                clf = cfg['build'](rs)
                clf.fit(X_bal, y_bal)
                y_pred = clf.predict(X_te)
                y_prob = clf.predict_proba(X_te)[:, 1]

                m = eval_metrics(y_test, y_pred, y_prob)
                for mk in METRICS:
                    row[f'{model_name}_{group_name}_{mk}'] = m[mk]

    return row


# ══════════════════════════════════════════════════════════════════════════════
# Model configurations
# ══════════════════════════════════════════════════════════════════════════════

def _build_lr(rs):
    return LogisticRegression(
        penalty='l2', solver='liblinear', C=1.0,
        max_iter=2000, random_state=rs)

def _build_enlr(rs):
    return LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.5,
        C=1.0, max_iter=1000, tol=1e-3, n_jobs=-1, random_state=rs)

def _build_rf(rs):
    return RandomForestClassifier(
        n_estimators=100, random_state=rs, n_jobs=-1)

def _build_xgb(rs):
    # XGBoost runs on CPU (n_jobs=-1 = all cores).
    # GPU mode (device='cuda') was tested but caused severe miscalibration after SMOTE:
    # the GPU hist solver calibrates thresholds for the balanced training distribution
    # and produces ~99% recall / ~7% precision on the imbalanced test set.
    # CPU mode with default solver reproduces the notebook behaviour correctly.
    return xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=rs, eval_metric='logloss',
        n_jobs=-1)

def _build_mlp(rs):
    return MLPClassifier(
        hidden_layer_sizes=(100, 50), activation='relu',
        solver='adam', alpha=0.0001, learning_rate='adaptive',
        learning_rate_init=0.001, max_iter=200,
        early_stopping=True, validation_fraction=0.1,
        random_state=rs)

CLASSICAL_MODELS = [
    {'name': 'lr',   'text_mode': 'sparse', 'num_mode': 'std', 'build': _build_lr},
    {'name': 'enlr', 'text_mode': 'dense',  'num_mode': 'std', 'build': _build_enlr},
    {'name': 'rf',   'text_mode': 'dense',  'num_mode': 'raw', 'build': _build_rf},
    {'name': 'xgb',  'text_mode': 'dense',  'num_mode': 'raw', 'build': _build_xgb},
    {'name': 'mlp',  'text_mode': 'dense',  'num_mode': 'std', 'build': _build_mlp},
]

BERT_MODEL = [
    {'name': 'bert_mlp', 'text_mode': 'bert', 'num_mode': 'std', 'build': _build_mlp},
]

# Pretty names for output files
MODEL_FILE_NAMES = {
    'lr': 'LR', 'enlr': 'ENLR', 'rf': 'RF',
    'xgb': 'XGBoost', 'mlp': 'MLP', 'bert_mlp': 'BERT_MLP',
}


# ══════════════════════════════════════════════════════════════════════════════
# BERT CLS extraction — runs once, result cached to disk
# Uses GPU if available, falls back to CPU automatically.
# ══════════════════════════════════════════════════════════════════════════════

def _extract_bert_cls(df: pd.DataFrame) -> np.ndarray:
    import torch
    from transformers import BertModel, BertTokenizer
    from tqdm import tqdm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  BERT device: {device}")
    TEXT_COLS = ['description', 'requirements', 'benefits', 'company_profile']
    BATCH     = 64   # RTX 4080 SUPER (17 GB VRAM) handles 64 comfortably; drop to 32 if OOM

    def _text(row):
        parts = [str(row[c]).strip() for c in TEXT_COLS
                 if pd.notna(row[c]) and str(row[c]).strip()]
        return ' [SEP] '.join(parts) if parts else ''

    texts = df.apply(_text, axis=1).tolist()
    tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    bert_model.eval()

    all_cls = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH), desc='BERT CLS extraction'):
            batch = texts[i: i + BATCH]
            enc   = tokenizer(batch, return_tensors='pt', truncation=True,
                              max_length=512, padding=True, add_special_tokens=True)
            enc   = {k: v.to(device) for k, v in enc.items()}
            out   = bert_model(**enc)
            all_cls.append(out.last_hidden_state[:, 0, :].cpu().numpy())

    cls_arr = np.vstack(all_cls).astype(np.float32)
    np.save(str(BERT_CLS_PATH), cls_arr)
    print(f"  CLS saved → {BERT_CLS_PATH.name}  shape {cls_arr.shape}")
    return cls_arr


# ══════════════════════════════════════════════════════════════════════════════
# Results I/O
# ══════════════════════════════════════════════════════════════════════════════

def _save_results(rows: list, model_names: list,
                  rs_label: str = "rs1-50") -> pd.DataFrame:
    """
    Save per-model CSVs and a summary-means CSV.
    Filenames include the random-state range so multiple runs are kept separately,
    e.g. LR_results_rs1-50.csv, LR_results_rs51-100.csv.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    df_all = pd.DataFrame(rows)

    summary_rows = []

    for model_name in model_names:
        file_label = MODEL_FILE_NAMES.get(model_name, model_name.upper())
        # Collect columns for this model
        cols = ['random_state'] + [c for c in df_all.columns
                                    if c.startswith(f'{model_name}_')]
        if len(cols) <= 1:
            continue
        df_m = df_all[cols].copy()
        save_path = RESULTS_DIR / f"{file_label}_results_{rs_label}.csv"
        df_m.to_csv(save_path, index=False)
        print(f"  Saved {save_path.name}  ({df_m.shape})")

        # Compute means for summary
        means = df_m.drop(columns='random_state').mean()
        for col_name, val in means.items():
            # col_name = '{model}_{group}_{metric}'
            parts = col_name.split('_', 2)   # split into model + rest
            # re-split on last two underscores to separate group and metric
            suffix = col_name[len(model_name) + 1:]  # '{group}_{metric}'
            # find metric suffix (last token after the last _)
            for metric in METRICS:
                if suffix.endswith('_' + metric):
                    group = suffix[: -(len(metric) + 1)]
                    summary_rows.append({
                        'model':  file_label,
                        'group':  group,
                        'metric': metric,
                        'mean':   round(val, 6),
                    })
                    break

    # Pivot summary into wide format
    if summary_rows:
        df_sum = (pd.DataFrame(summary_rows)
                  .pivot_table(index=['model', 'group'],
                               columns='metric', values='mean')
                  .reset_index()[['model', 'group'] + METRICS])
        df_sum.to_csv(RESULTS_DIR / f'summary_means_{rs_label}.csv', index=False)
        print(f"\n  Summary saved → results/summary_means_{rs_label}.csv")
        return df_sum
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  EMSCAD Monte Carlo Fraud Detection Evaluation")
    print("=" * 64)

    if not DATA_PATH.exists():
        sys.exit(f"\nData file not found:\n  {DATA_PATH}\n")

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\nLoading {DATA_PATH.name} …")
    df = load_and_preprocess(DATA_PATH)
    print(f"  {len(df):,} rows  |  fraud rate = "
          f"{df['fraudulent'].mean():.3f} ({df['fraudulent'].sum()} fraud)")
    _check_required_columns(df)

    # ── BERT CLS (always required) ─────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    if BERT_CLS_PATH.exists():
        cls_all = np.load(str(BERT_CLS_PATH))
        print(f"\n  BERT CLS loaded from cache: {BERT_CLS_PATH.name}  "
              f"shape {cls_all.shape}")
    else:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n  Extracting BERT CLS embeddings on {device.upper()} "
              f"(runs once, then cached to {BERT_CLS_PATH.name}) …")
        cls_all = _extract_bert_cls(df)

    models_cfg = CLASSICAL_MODELS + BERT_MODEL
    model_names = [c['name'] for c in models_cfg]

    print(f"\n  Models   : {model_names}")
    print(f"  Groups   : {list(GROUPS.keys())}")
    print(f"  Metrics  : {METRICS}")
    print(f"  Iterations: {N_ITERATIONS}")
    print(f"  Parallel workers: {N_JOBS}")
    print()

    # ── Parallel Monte Carlo CV ────────────────────────────────────────────
    t0 = time.time()

    rs_end = RANDOM_STATE_START + N_ITERATIONS - 1
    rs_range_label = f"rs{RANDOM_STATE_START}-{rs_end}"
    print(f"  Random states: {RANDOM_STATE_START} to {rs_end} ({rs_range_label})")
    print()

    rows = Parallel(n_jobs=N_JOBS, backend='loky', verbose=5)(
        delayed(_one_iteration)(rs, df, models_cfg, cls_all)
        for rs in range(RANDOM_STATE_START, RANDOM_STATE_START + N_ITERATIONS)
    )

    elapsed = time.time() - t0
    print(f"\n  All {N_ITERATIONS} iterations complete in {elapsed/60:.1f} min")

    # ── Save results ───────────────────────────────────────────────────────
    print(f"\nSaving results to {RESULTS_DIR.name}/ …")
    df_summary = _save_results(rows, model_names, rs_range_label)

    # ── Print summary ──────────────────────────────────────────────────────
    if not df_summary.empty:
        print("\n" + "=" * 64)
        print(f"  Mean metrics — {rs_range_label}")
        print("=" * 64)
        with pd.option_context('display.float_format', '{:.4f}'.format,
                               'display.max_columns', 10,
                               'display.width', 120):
            print(df_summary.to_string(index=False))

    print(f"\n{'=' * 64}")
    print(f"  Done. Results in: {RESULTS_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    main()
