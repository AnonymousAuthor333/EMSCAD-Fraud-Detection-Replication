# EMSCAD-Fraud-Detection-Replication
This project is a self-contained pipeline for detecting fraudulent job postings in the EMSCAD dataset (job postings from 2013) using a multi-agent LLM framework, and for comparing it against a single-pass LLM baseline and classical machine-learning models.

All scripts use paths relative to this folder. The folder can be moved or
copied as a whole and everything will still run.

## Requirements

- Python 3.8+
- Packages: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `xgboost`,
  `scipy`, `joblib`, `openai`, `openpyxl`, `matplotlib`, `torch`,
  `transformers`, `tqdm`
- An OpenAI API key (entered at the prompt, or set the `OPENAI_API_KEY`
  environment variable). LLM scripts use `gpt-4o` with temperature 0.1.
- A CUDA GPU is used for BERT embedding extraction and XGBoost-GPU testing
  if available.

## Data

`EMSCAD_fraud_detection.csv` — the main dataset (8,781 postings, of which
556 / 6.3% are labelled fraudulent in the `fraudulent_agent` column).
Scripts append result columns to this file; the original posting fields are
never modified. Note that due to the 25 MB file size limitation, the EMSCAD
data is broken into 2 parts. Users should download both parts, combine them
and rename the combined dataset to `EMSCAD_fraud_detection.csv`.

`flag_definitions_agent.json` / `.xlsx` — definitions of the eight fraud
indicator flags. Each flag has a `detection_agent` section (rules a
sub-agent follows when reading a posting) and an `orchestrator` section
(calibration used when aggregating flags into a risk score).

The eight flags:

| # | Column | Name |
|---|--------|------|
| 1 | `flag1_sensitive_info` | Sensitive Information Request |
| 2 | `flag2_unrealistic_comp` | Unrealistic Compensation |
| 3 | `flag3_pressure_tactics` | Pressure Tactics |
| 4 | `flag4_anon_employer` | Unidentifiable Employer |
| 5 | `flag5_req_role_mismatch` | Requirements Role Mismatch |
| 6 | `flag6_info_conflict` | Internal Information Conflict |
| 7 | `flag7_vague_info` | Vague Information |
| 8 | `flag8_promo_template` | Promotional Template Substitution |

All flags are evaluated from the posting text only — no web search is
performed.

## Scripts

Run in roughly this order:

1. **`fraud_detector.py`** — multi-agent evaluation. Three sub-agents
   (legitimacy, context, consistency) mark the eight flags for each posting;
   an orchestrator that sees only the flag results assigns a 0–10 risk
   score. Also produces three ablation scores using two sub-agents at a
   time. If this program is executed after the evaluation has completed,
   existing values will be overwritten.
   Columns written: the eight flag columns plus `risk_score_all`,
   `risk_score_legit_context`, `risk_score_context_consist`,
   `risk_score_legit_consist`.

3. **`single_pass_detector.py`** — baseline. One API call per posting with
   the raw posting text and no flag framework. Writes
   `single_pass_risk_score`. The exact prompt is saved to
   `single_pass_query.txt`.

4. **`flag_analysis.py`** — flag statistics: per-flag prevalence and fraud
   rates (chi-square / Fisher), fraud rate by number of flags fired
   (Spearman), and risk-score distributions by fraud status
   (Mann-Whitney, AUROC). Outputs to `results/`.

5. **`model_evaluation.py`** — 50-iteration Monte Carlo cross-validation
   (stratified 80/20, SMOTE on training folds). Four models (LR, Random Forest, XGBoost, 
   BERT+MLP) x four feature groups (base, + single-pass score,
   + multi-agent score, + each of the three ablation scores).
   + Metrics: accuracy, recall, F1, PR-AUC. Iterations run in
   parallel. BERT CLS embeddings are extracted once and cached in
   `results/bert_cls_embeddings.npy`. Set `RANDOM_STATE_START` at the top
   to run a different batch of random states; output filenames include the
   range (e.g. `LR_results_rs1-50.csv`).

7. **`statistical_comparisons.py`** — paired t-tests on the Monte Carlo
   results: base vs. single-pass vs. multi-agent, plus ablation comparisons
   of each two-agent score against the full three-agent score. Reports mean
   difference, t, p, Cohen's d, 95% CI.

8. **`generate_manual_eval_sample.py`** — draws a 50-posting sample with at
   least 6 AI-positive cases per flag for human review. Writes a blind
   review sheet (no AI flags, no fraud labels) and an AI reference sheet to
   `flags_manual_eval/`.

9. **`compare_manual_vs_ai_flags.py`** — after the manual review is filled
   in, compares human vs. AI labels per flag (TP/TN/FP/FN, agreement,
   Cohen's kappa) and lists every disagreement.

10. **`plot_risk_score_diagrams.py`** — 2x2 figure comparing single-pass and
   multi-agent risk scores: score distributions and fraud rate by score,
   with shared axes. Saves PNG and PDF to `diagram/`.

Support modules (imported by the scripts above, not run directly):

- `agents.py` — sub-agent and orchestrator logic, flag-to-agent assignment,
  posting formatter.
- `data_helpers.py` / `feature_helpers.py` — feature engineering for the
  ML evaluation (TF-IDF/SVD, one-hot encoding, salary parsing, etc.).

## Folder layout

```
EMSCAD_fraud_detection_replication/
├── README.md
├── EMSCAD_fraud_detection.csv      main dataset + result columns
├── flag_definitions_agent.json     flag definitions (source of truth)
├── flag_definitions_agent.xlsx     same, human-readable
├── single_pass_query.txt           prompt record for the baseline
├── *.py                            scripts (see above)
├── results/                        analysis outputs, model results,
│                                   BERT embedding cache
├── diagram/                        figures
└── flags_manual_eval/              manual review sample and comparison
```

## Reproducibility notes

- LLM calls use `gpt-4o`, temperature 0.1, JSON-mode responses, and no
  conversation memory between rows, so re-running a range produces stable
  results.
- The ML evaluation fixes every random seed (splits, SMOTE, model inits),
  so a run with the same `RANDOM_STATE_START` is exactly reproducible.
- The fraud label (`fraudulent_agent`) is excluded from everything the
  LLM agents see and from all model features; it is used only for
  evaluation.
- Flags 4, 6, and 8 were renamed during the project (formerly
  `flag4_company_not_found`, `flag6_location_conflict`,
  `flag8_contents_overlap`); all current files use the new names.
