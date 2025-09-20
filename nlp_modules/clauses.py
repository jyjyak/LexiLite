import re
from typing import Set

RISK_KEYWORDS = {
    "Termination": [
        r"\bterminate(?:s|d)?\b.*?\bwithout\s+(?:notice|cause)\b",
        r"\bmay\s+terminate\b.*?\bat\s+any\s+time\b",
        r"\btermination\s+for\s+convenience\b",
        r"\btermination\s+for\s+material\s+breach\b",
        r"\btermination\s+for\s+other\s+reasons?\b",
    ],
    "Indemnity": [
        r"\bindemnif(?:y|ication)\b",
        r"\bhold\s+harmless\b",
        r"\bdefend\b",
        r"\bthird[-\s]?party\s+claims?\b",
        r"\bfrom\s+and\s+against\b",
    ],
    "Non-compete": [
        r"\bnon[-\s]?compete\b",
        r"\bnot\b.*?\bengage\b.*?\bcompet(?:e|ing)\b",
        r"\brestrictive\s+covenant\b",
        r"\bcompeting\s+business\b", 
        r"\bfor\s+\d+\s*(?:months?|years?)\b.*?\b(after|following)\b",
    ],
    "Auto-renewal": [
        r"\bautomatically\s+renew\b",
        r"\brenewed\b.*?\bwithout\s+notice\b",
    ],
    "Arbitration": [
        r"\barbitration\b",
        r"\bbinding\s+arbitration\b",
        r"\bAAA\b|\bJAMS\b",
        r"\bexclusive\b.*\barbitration\b",
        r"\bwaive\b.*\b(court|jury)\b",
        r"\bforum\s+selection\b",
        r"\bwaive\b.*?\bright\b.*?\b(?:court|sue)\b",
    ],
    "Limitation of liability": [
        r"\blimitation\s+of\s+liabilit(?:y|ies)\b",
        r"\bin\s+no\s+event\b.*\bliabl?e\b",
        r"\b(cap|aggregate)\s+on?\s+liabilit(?:y|ies)\b",
        r"\bliability\s+shall\s+not\s+exceed\b",
        r"\b(consequential|incidental|indirect|special|punitive)\s+damages\b", 
        r"\baggregate\s+liability\b",
    ],
    "Damages": [
        r"\b(consequential|incidental|indirect|special|punitive)\s+damages\b",
    ],
    "Attorney fees": [ 
        r"\battorney'?s?\s+fees\b",
        r"\bprevailing\s+party\b.*?\bfees?\b",
        r"\bcosts\s+of\s+collection\b",
    ],
}


_RISK_KEYWORDS_COMPILED = {
    family: [re.compile(p, re.IGNORECASE | re.DOTALL) for p in pats]
    for family, pats in RISK_KEYWORDS.items()
}



# High-severity CUAD labels
HIGH_RISK_CUAD: Set[str] = {
    "indemnification",
    "limitation_of_liability",
    "consequential_damages",
    "non_compete",
    "termination_for_convenience",
    "termination_for_material_breach",
    "termination_for_other_reasons",
    "arbitration",
    "change_of_control",
    "intellectual_property_ownership",
    "license_grant",
    "most_favored_nation",
}

# Medium-severity CUAD labels
MEDIUM_RISK_CUAD: Set[str] = {
    "termination_obligations",
    "renewal_term",
    "assignment",
    "audit_rights",
    "non_solicit",
    "non_disparagement",
    "insurance",
    "export_control",
    "compliance_with_laws",
    "pricing",
    "post_termination_services",
}

RISKY_MAURO_LABELS: Set[str] = {
    "cap_on_liability",
    "uncapped_liability",
}

RISKY_LABELS_ALL: Set[str] = (
    HIGH_RISK_CUAD | MEDIUM_RISK_CUAD | RISKY_MAURO_LABELS
)

RISKY_CUAD_LABELS: Set[str] = HIGH_RISK_CUAD | MEDIUM_RISK_CUAD

# Regex to label mapping 
REGEX_TO_RISK_LABELS = {
    "Termination": "termination_obligations",
    "Indemnity": "indemnification",
    "Non-compete": "non_compete",
    "Arbitration": "arbitration",
    "Auto-renewal": "renewal_term",
    "Limitation of liability": "limitation_of_liability",
    "Damages": "consequential_damages",
    "Attorney Fees": "attorney_fees",
    "Data Breach": "data_breach",
}


# Clause detection
def detect_risk_clauses(text: str):
    findings = []
    for clause_type, patterns in _RISK_KEYWORDS_COMPILED.items():
        for rx in patterns:
            for m in rx.finditer(text):
                findings.append({
                    "type": clause_type,
                    "match_text": text[m.start():m.end()],
                    "start": m.start(),
                    "end": m.end(),
                })
    dedup = {(f["type"], f["start"], f["end"]): f for f in findings}
    return list(dedup.values())

# Actionable filtering
ACTION_KEYWORDS = [
    r"\bshall\b", r"\bmust\b", r"\bagree(?:s|d)?\b", r"\bterminate\b", r"\breturn\b",
    r"\bobligat(?:ed|ion)\b", r"\bliable\b", r"\bwaive\b", r"\bdisclos(?:e|ure)\b",
    r"\bengage\b", r"\bnot\b.*?\bcompete\b", r"\brestrain\b", r"\bprovide\b",
    r"\bexpire(?:s|d)?\b", r"\benforce\b", r"\bbind(?:ing)?\b", r"\bentitled\s+to\b",
    r"\bresponsible\s+for\b",
    r"\bindemnif(?:y|ication)\b",   
    r"\bhold\s+harmless\b",
    r"\bbinding\s+arbitration\b",
    r"\bmay\s+terminate\b|\bterminate\s+at\s+any\s+time\b",
    r"\bin\s+no\s+event\s+shall\b.*\bliab",
    r"\blimitation\s+of\s+liability\b|\bcap\s+on\s+liability\b",
    r"\bwaive\b.*\bright\b.*\b(?:court|sue)\b",
    r"\bwaive\b.*\b(court|jury)\b",
     r"\bin\s+no\s+event\b.*\bliabl?e\b",
    r"\bprevailing\s+party\b", r"\battorney'?s\s+fees\b",
    r"\bsecurity\s+incident\b", r"\bdata\s+breach\b",
]
NON_ACTIONABLE_PATTERNS = [
    r"\bconstitutes?\s+a\s+valid.*?\bagreement\b",
    r"\bshall\s+be\s+governed\s+by\b.*?\blaws\b",
    r"\bin\s+no\s+event\s+shall\b.*?\bbe\s+deemed\b",
    r"\bdefinitions?\b",
    r"\bcopyright\b",
    r"\bpage\s+\d+\s+of\s+\d+\b",
]
_ACTION_RE = re.compile("|".join(f"(?:{p})" for p in ACTION_KEYWORDS), re.IGNORECASE | re.DOTALL)
_EXCLUDE_RE = re.compile("|".join(f"(?:{p})" for p in NON_ACTIONABLE_PATTERNS), re.IGNORECASE | re.DOTALL)

def is_actionable(text: str) -> bool:
    if _EXCLUDE_RE.search(text):
        return False
    return bool(_ACTION_RE.search(text))

RISK_KEYWORDS.update({
    "Arbitration": RISK_KEYWORDS.get("Arbitration", []) + [
        r"\barbitration\b",
        r"\bAAA\b|\bJAMS\b",
        r"\bexclusive\b.*\barbitration\b",
        r"\bwaive\b.*\b(court|jury)\b",
        r"\bforum\s+selection\b",
    ],
    "Indemnity": RISK_KEYWORDS.get("Indemnity", []) + [
        r"\bdefend\b.*\bindemnif(?:y|ication)\b",
        r"\bhold\s+harmless\b",
        r"\bthird[-\s]?party\b.*\bclaims?\b",
        r"\bclaims?,\s*damages?,\s*losses?\b",
    ],
    "Limitation of liability": RISK_KEYWORDS.get("Limitation of liability", []) + [
        r"\blimitation\s+of\s+liabilit(?:y|ies)\b",
        r"\bin\s+no\s+event\b.*\bliabl?e\b",
        r"\bcap\b.*\bliabilit(?:y|ies)\b",
        r"\bconsequential|incidental|special\s+damages\b",
        r"\bindirect\s+damages\b",
    ],
    "Termination": RISK_KEYWORDS.get("Termination", []) + [
        r"\bupon\s+termination\b.*\b(return|destroy|cease)\b",
        r"\bsurvive\b.*\btermination\b",
        r"\bmay\s+terminate\b|\bterminate\s+at\s+any\s+time\b",
    ],
    "Attorney Fees": [
        r"\battorney'?s\s+fees\b",
        r"\bcounsel\s+fees\b",
        r"\bprevailing\s+party\b",
    ],
    "Data Breach": [
        r"\bsecurity\s+incident\b",
        r"\bunauthorized\s+access\b",
        r"\bdata\s+breach\b",
        r"\bnotify\b.*\bbreach\b",
        r"\bpersonal\s+data\b.*\bbreach\b",
    ],
})


