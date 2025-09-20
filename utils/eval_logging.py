# utils/eval_logging.py
import csv, os
from typing import Dict, Any

PRED_FIELDS = [
    "doc_id","clause_id","clause_text",
    "pred_score","pred_bucket",
    "pred_cuad_label","pred_cuad_score",
    # NEW: normalized label (canonical 11-class name)
    "pred_cuad_label_norm",
    "regex_labels","zs_top_label","zs_top_score",
    "f1_cuad_max","f2_cuad_mass","f3_zs_max","f4_regex","f5_actionable",
    # NEW: decisions for convenience (optional)
    "pred_is_risky_base",       # e.g., 1 if pred_score >= base_thresh
    "pred_is_risky_classaware", # e.g., 1 if pred_score >= CAT_THRESH[pred_cuad_label_norm] else base
    # (Optional) record threshold actually used for this row
    "used_thresh"
]

def append_eval_row(path: str, row: Dict[str, Any]) -> None:
    """Append one prediction row to CSV, creating header if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PRED_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)

def export_labels_template(path: str, rows: list[Dict[str, Any]]) -> None:
    """
    Write a labeling template (doc_id, clause_id, clause_text) for human gold labels.
    You’ll fill gold_is_risky and gold_primary_label later.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    FIELDNAMES = ["doc_id","clause_id","clause_text","gold_is_risky","gold_primary_label","notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({
                "doc_id": r["doc_id"],
                "clause_id": r["clause_id"],
                "clause_text": r["clause_text"],
                "gold_is_risky": "",           # fill manually (0/1)
                "gold_primary_label": "",      # fill manually (canonical name, e.g., non_compete)
                "notes": ""
            })
