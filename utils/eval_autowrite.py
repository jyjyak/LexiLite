# utils/eval_autowrite.py
import os
from typing import Dict, List
from utils.eval_logging import append_eval_row, export_labels_template

# NEW
import pandas as pd

DEFAULT_PREDS_CSV = "data/preds.csv"
DEFAULT_LABELS_CSV = "data/labels.csv"

def _rows_to_label_df(rows: List[Dict]) -> pd.DataFrame:
    """Convert eval_rows -> minimal label df with canonical dtypes."""
    df = pd.DataFrame([{
        "doc_id": r["doc_id"],
        "clause_id": r["clause_id"],
        "clause_text": r.get("clause_text", "")
    } for r in rows])
    if not df.empty:
        df["doc_id"] = df["doc_id"].astype(str)
        df["clause_id"] = pd.to_numeric(df["clause_id"], errors="coerce").astype("Int64")
    return df

def _upsert_labels(labels_csv: str, new_rows: List[Dict]) -> None:
    """
    UPSERT semantics:
    - If labels.csv exists: append only brand-new (doc_id, clause_id) rows; keep existing rows & gold_*.
    - If it doesn't exist: create a fresh template with blanks for gold_*.
    """
    new_df = _rows_to_label_df(new_rows)
    if new_df.empty:
        return

    key_cols = ["doc_id", "clause_id"]

    if os.path.exists(labels_csv):
        old = pd.read_csv(labels_csv)
        if old.empty:
            old = pd.DataFrame(columns=["doc_id","clause_id","clause_text","gold_is_risky","gold_primary_label","notes"])

        # enforce dtypes for join keys
        if "doc_id" in old:
            old["doc_id"] = old["doc_id"].astype(str)
        if "clause_id" in old:
            old["clause_id"] = pd.to_numeric(old["clause_id"], errors="coerce").astype("Int64")

        # identify brand-new keys
        if set(key_cols).issubset(old.columns):
            new_keys = set(map(tuple, new_df[key_cols].dropna().values.tolist()))
            old_keys = set(map(tuple, old[key_cols].dropna().values.tolist()))
            add_mask = new_df[key_cols].apply(tuple, axis=1).map(lambda k: k not in old_keys)
            add_df = new_df[add_mask].copy()
        else:
            add_df = new_df.copy()

        if add_df.empty:
            return

        # create empty gold columns for new rows
        for col in ["gold_is_risky","gold_primary_label","notes"]:
            add_df[col] = ""

        merged = pd.concat([old, add_df], ignore_index=True)

        # optional: stable sort for readability
        merged = merged.sort_values(key_cols, kind="mergesort").reset_index(drop=True)
        merged.to_csv(labels_csv, index=False)
    else:
        # first time: create full template for current rows
        tmpl_rows = [{
            "doc_id": r["doc_id"],
            "clause_id": r["clause_id"],
            "clause_text": r.get("clause_text","")
        } for r in new_rows]
        export_labels_template(labels_csv, tmpl_rows)

def write_eval_artifacts(
    eval_rows: List[Dict],
    preds_csv: str = DEFAULT_PREDS_CSV,
    labels_csv: str = DEFAULT_LABELS_CSV,
    overwrite_labels_if_exists: bool = False,
) -> None:
    """
    Append prediction rows to preds_csv and manage labels_csv.
    - preds_csv: always append (header auto-managed by append_eval_row)
    - labels_csv:
        * if overwrite_labels_if_exists=True → regenerate template for just these rows (not cumulative)
        * else (default) → UPSERT: append unique (doc_id, clause_id) while preserving existing gold_* fields
    """
    if not eval_rows:
        return

    # 1) Append predictions
    for r in eval_rows:
        append_eval_row(preds_csv, r)

    # 2) Manage labels cumulatively by default
    if overwrite_labels_if_exists:
        # Overwrite with a fresh template only for THIS run
        lab_rows = [{"doc_id": r["doc_id"], "clause_id": r["clause_id"], "clause_text": r.get("clause_text","")} for r in eval_rows]
        export_labels_template(labels_csv, lab_rows)
    else:
        # Cumulative upsert (recommended)
        _upsert_labels(labels_csv, eval_rows)
