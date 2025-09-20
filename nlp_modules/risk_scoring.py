from typing import List, Dict, Set
import re

RISKY_LABELS_ALL: Set[str] = {
    "termination_obligations", "non_compete", "indemnification",
    "limitation_of_liability", "termination_for_convenience",
    "termination_for_material_breach", "arbitration",
    "post_termination_services",
    "attorney_fees", "data_breach"
}

HIGH_IMPORT_CATS = {
    "arbitration", "indemnification", "limitation_of_liability",
    "termination_obligations", "attorney_fees", "data_breach",
    "non_compete",
}

AGREEMENT_THRESHOLD_CUAD = 0.55
AGREEMENT_THRESHOLD_ZS   = 0.60

def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def length_prior(text: str) -> float:
    n_tokens = len(re.findall(r"\w+", text))
    if n_tokens < 20:   return -0.10
    if n_tokens > 400:  return -0.05
    return 0.0

def heading_prior(text: str) -> float:
    head = text[:120].lower()
    hits = any(k in head for k in [
        "termination", "liability", "indemn", "arbitration",
        "non-compete", "noncompete", "attorney", "breach"
    ])
    return 0.05 if hits else 0.0

def regex_signal(regex_hits: int) -> float:
    if regex_hits <= 0: return 0.0
    if regex_hits == 1: return 0.5
    return 1.0  # 2+ hits

def compute_risk_features(
    clause_text: str,
    cuad_scores: List[Dict[str, float]],
    zs_labels: List[str],
    zs_scores: List[float],
    risky_set: Set[str] = RISKY_LABELS_ALL,
    regex_hits: int = 0,
    actionable: bool = True,
    matched_labels: Set[str] | None = None,
) -> Dict[str, float]:
    matched_labels = matched_labels or set()

    # --- CUAD features ---
    f1 = 0.0
    risk_mass = 0.0
    for s in cuad_scores:
        lab = s["label"].lower().replace(" ", "_")
        sc  = float(s["score"])
        if lab in risky_set:
            risk_mass += sc
            if sc > f1:
                f1 = sc
    f2 = clip01(risk_mass)

    # --- Zero-shot risky max ---
    f3 = 0.0
    for lab, sc in zip(zs_labels, zs_scores):
        labn = lab.lower().replace(" ", "_")
        if labn in risky_set:
            f3 = max(f3, float(sc))

    # --- Regex / Actionable ---
    f4 = regex_signal(regex_hits)
    f5 = 1.0 if actionable else 0.0

    # --- Priors ---
    len_pr = length_prior(clause_text) + heading_prior(clause_text)

    # --- Consensus checks ---
    cuad_ok = (f1 >= AGREEMENT_THRESHOLD_CUAD)
    zs_ok   = (f3 >= AGREEMENT_THRESHOLD_ZS)
    rx_ok   = (f4 > 0.0)

    # Agreement bonuses
    bonus = 0.0
    # CUAD + (Regex or ZS)
    if cuad_ok and (rx_ok or zs_ok):
        bonus += 0.10
    # ZS + Regex even if CUAD is not confident 
    if (not cuad_ok) and zs_ok and rx_ok:
        bonus += 0.10

    # --- Weights (adaptive if CUAD is weak but ZS+Regex strong) ---
    # Base weights 
    w1, w2, w3, w4, w5, w_len = 0.40, 0.20, 0.20, 0.15, 0.05, 0.05

    # If CUAD is weak but ZS is very strong and Regex present, lean toward ZS/Regex
    if f1 < 0.35 and f3 >= 0.85 and f4 >= 0.5:
        w1, w2, w3, w4 = 0.25, 0.15, 0.30, 0.20

    # Blend
    score_raw = (w1*f1 + w2*f2 + w3*f3 + w4*f4 + w5*f5 + w_len*len_pr)
    score = clip01(score_raw + bonus)

    # --- Category-aware gentle floors  ---
    # If regex hit one of the high-importance categories, nudge minimum
    if matched_labels & HIGH_IMPORT_CATS:
        if f4 >= 1.0:
            score = max(score, 0.65)  
        elif f4 >= 0.5:
            score = max(score, 0.60)  

    # --- Consensus floor  ---
    consensus_floor_applied = False
    if (not cuad_ok) and (f3 >= 0.85) and (f4 >= 0.5) and actionable:
        score = max(score, 0.70) 
        consensus_floor_applied = True

    return {
        "f1_cuad_max": f1,
        "f2_cuad_mass": f2,
        "f3_zs_max": f3,
        "f4_regex": f4,
        "f5_actionable": f5,
        "bonus_agreement": 1.0 if bonus > 0 else 0.0,
        "score": clip01(score),
        "consensus_floor_applied": consensus_floor_applied,
    }

def bucket(score: float) -> str:
    if score >= 0.65: return "High"
    if score >= 0.35: return "Medium"
    return "Low"
