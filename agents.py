#!/usr/bin/env python3
"""
agents.py
Sub-agent and orchestrator logic for the EMSCAD fraud detection system.

Three sub-agents evaluate job postings against 8 flags:
  Legitimacy Agent  : Flags 1 (Sensitive Info), 2 (Unrealistic Comp),
                      3 (Pressure Tactics), 7 (Vague Info)
  Context Agent     : Flag 4 (Company Not Found)
  Consistency Agent : Flags 5 (Req/Role Mismatch), 6 (Location Conflict),
                      8 (Contents Overlap)

The Orchestrator Agent receives ONLY flag results (never the original posting)
and computes a fraud risk score 0-10.

Design principles:
  - GPT-4o, temperature=0.1 for high replicability
  - No message history between calls (each call is stateless)
  - fraudulent_agent column is NEVER passed to any agent
"""

import json
from typing import Dict, List, Optional

import openai

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL       = "gpt-4o"
TEMPERATURE = 0.1

# ── Flag-to-agent assignment ────────────────────────────────────────────────────
AGENT_FLAGS: Dict[str, List[int]] = {
    "legitimacy":  [1, 2, 3, 7],   # applicant-facing fraud signals
    "context":     [4],             # company identity / verifiability
    "consistency": [5, 6, 8],       # internal posting consistency
}

# Canonical column names used throughout the pipeline
FLAG_COL: Dict[int, str] = {
    1: "flag1_sensitive_info",
    2: "flag2_unrealistic_comp",
    3: "flag3_pressure_tactics",
    4: "flag4_anon_employer",
    5: "flag5_req_role_mismatch",
    6: "flag6_info_conflict",
    7: "flag7_vague_info",
    8: "flag8_promo_template",
}

# Ablation combinations: 2-agent subsets
ABLATION_COMBOS: Dict[str, List[str]] = {
    "legit_context":   ["legitimacy", "context"],
    "context_consist": ["context",    "consistency"],
    "legit_consist":   ["legitimacy", "consistency"],
}

_AGENT_LABEL: Dict[str, str] = {
    "legitimacy":  "Legitimacy Agent (applicant-facing fraud signals: "
                   "sensitive information requests, compensation, pressure tactics, vague content)",
    "context":     "Context Agent (company identity and internet verifiability)",
    "consistency": "Consistency Agent (internal posting consistency: "
                   "requirements-role fit, location, content overlap)",
}


# ══════════════════════════════════════════════════════════════════════════════
# Posting formatter
# ══════════════════════════════════════════════════════════════════════════════

def format_posting(row: dict) -> str:
    """
    Render one CSV row as readable text for agent input.
    Explicitly excludes 'fraudulent_agent' so no agent ever sees the label.
    """
    _EXCLUDE = {"fraudulent_agent", "job_id"}
    _FIELDS = [
        ("Title",                        "title"),
        ("Location",                     "location"),
        ("Department",                   "department"),
        ("Salary Range",                 "salary_range"),
        ("Employment Type",              "employment_type"),
        ("Required Experience",          "required_experience"),
        ("Required Education",           "required_education"),
        ("Industry",                     "industry"),
        ("Function",                     "function"),
        ("Telecommuting",                "telecommuting"),
        ("Has Company Logo",             "has_company_logo"),
        ("Has Questions",                "has_questions"),
        ("Company Profile",              "company_profile"),
        ("Description",                  "description"),
        ("Requirements",                 "requirements"),
        ("Benefits",                     "benefits"),
        ("QCEW Annual Avg Employment",   "qcew_annual_avg_emplvl"),
        ("QCEW Avg Annual Pay",          "qcew_avg_annual_pay"),
        ("QCEW OTY Employment Change",   "qcew_oty_annual_avg_emplvl_chg"),
        ("QCEW OTY Employment Pct Chg",  "qcew_oty_annual_avg_emplvl_pct_chg"),
    ]
    lines = []
    for label, key in _FIELDS:
        if key in _EXCLUDE:
            continue
        val = row.get(key)
        if val is None:
            continue
        s = str(val).strip()
        if s.lower() in ("", "nan", "none"):
            continue
        lines.append(f"{label}: {s[:700]}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Sub-agent
# ══════════════════════════════════════════════════════════════════════════════

def _detection_block(flag_defs: dict, flag_ids: List[int]) -> dict:
    """Extract detection_agent sections from the definitions for the given flag IDs."""
    return {
        FLAG_COL[f["flag_id"]]: f.get("detection_agent", {})
        for f in flag_defs["flags"]
        if f["flag_id"] in flag_ids
    }


def run_sub_agent(
    client: openai.OpenAI,
    agent_name: str,
    posting_text: str,
    flag_defs: dict,
) -> Dict[str, bool]:
    """
    Run one sub-agent on one posting.
    Returns {column_name: bool} for the agent's assigned flags.
    Each call is stateless — no prior conversation history.
    """
    flag_ids = AGENT_FLAGS[agent_name]
    det_block = _detection_block(flag_defs, flag_ids)
    # Template for the expected response
    expected = {FLAG_COL[fid]: False for fid in flag_ids}

    system_msg = (
        f"You are the {_AGENT_LABEL[agent_name]} in a fraud detection system "
        f"for job postings.\n"
        "Evaluate the posting against your assigned fraud indicator flags.\n\n"
        "RULES (follow exactly):\n"
        "  1. Each call is independent — no memory of any previous evaluation.\n"
        "  2. Do NOT infer or consider whether this posting is labelled as fraud.\n"
        "  3. Apply trigger_criteria precisely. Apply do_not_trigger guards strictly.\n"
        "  4. Return valid JSON only — no preamble, no explanation, no markdown."
    )

    user_msg = "\n".join([
        "=== JOB POSTING ===",
        posting_text,
        "",
        "=== YOUR FLAG DEFINITIONS (detection rules) ===",
        json.dumps(det_block, indent=2, ensure_ascii=False),
        "",
        "=== TASK ===",
        "For each flag in your definitions: return true if the flag is triggered",
        "according to its trigger_criteria, false otherwise.",
        "Apply every do_not_trigger guard before returning true.",
        "",
        "Return ONLY a JSON object with boolean values — no strings, no nulls:",
        json.dumps({k: "<bool>" for k in expected}, indent=2),
    ])

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )

    raw = json.loads(resp.choices[0].message.content)
    # Normalise: return all expected keys as bool, defaulting to False
    return {col: bool(raw.get(col, False)) for col in expected}


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator agent
# ══════════════════════════════════════════════════════════════════════════════

def run_orchestrator(
    client: openai.OpenAI,
    flag_results: Dict[str, bool],
    flag_defs: dict,
    included_agents: List[str],
) -> int:
    """
    Run the orchestrator on flag results from the specified agents.
    Does NOT receive the original posting text — only flag results + calibration.
    Returns an integer risk score 0-10.
    Each call is stateless — no prior conversation history.
    """
    included_ids = [fid for a in included_agents for fid in AGENT_FLAGS[a]]

    # Build calibration payload: triggered status + orchestrator context per flag
    calibration = {
        FLAG_COL[f["flag_id"]]: {
            "triggered":           flag_results.get(FLAG_COL[f["flag_id"]], False),
            "calibration":         f.get("orchestrator", {}),
        }
        for f in flag_defs["flags"]
        if f["flag_id"] in included_ids
    }

    system_msg = (
        "You are the Orchestrator Agent in a job-posting fraud detection system.\n"
        "You receive flag results from sub-agents and compute a risk score.\n\n"
        "RULES (follow exactly):\n"
        "  1. Do NOT access or consider the original job posting text — "
        "you never see it.\n"
        "  2. Base your score SOLELY on the flag results and calibration data below.\n"
        "  3. Each call is independent — no memory of any previous evaluation.\n"
        "  4. Return valid JSON only."
    )

    user_msg = "\n".join([
        "=== FLAG RESULTS AND CALIBRATION DATA ===",
        json.dumps(calibration, indent=2, ensure_ascii=False),
        "",
        "=== SCORING TASK ===",
        "Assign a fraud risk score 0-10 based only on which flags are triggered",
        "and their calibration data.",
        "",
        "Score scale:",
        "  0-1 : very likely legitimate  (no significant flags triggered)",
        "  2-3 : probably legitimate     (only weak or ambiguous flags)",
        "  4-5 : uncertain               (mixed signals)",
        "  6-7 : probably fraudulent     (moderate-to-strong flags triggered)",
        "  8-9 : likely fraudulent       (multiple strong flags triggered)",
        "  10  : very likely fraudulent  (strong convergent evidence)",
        "",
        "Weighting guidance (flags with triggered=false contribute nothing):",
        "  indicative_strength:  strong > moderate > weak",
        "  frequency_in_fraud:   common > occasional > rare",
        "  specificity_to_fraud: high > medium > low (higher = more diagnostic)",
        "  false_positive_risk:  low FP risk = higher weight",
        "  risk_interpretation:  follow the provided guidance for each triggered flag",
        "",
        'Return ONLY: {"risk_score": <integer 0-10>}',
    ])

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )

    raw = json.loads(resp.choices[0].message.content)
    score = raw.get("risk_score", 0)
    return max(0, min(10, int(round(float(score)))))
