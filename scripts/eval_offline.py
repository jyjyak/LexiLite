# scripts/eval_offline.py
import argparse
import numpy as np
import pandas as pd
from nlp_modules.label_normalizer import normalize_label
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)

# ----- Helpers ---------------------------------------------------------------

CAT_THRESH = {
    "arbitration": 0.58,
    "indemnification": 0.58,
    "limitation_of_liability": 0.58,
    "non_compete": 0.58,
    "data_breach": 0.58,
    "termination_obligations": 0.58,
    # keep the rest at 0.65 baseline:
    "attorney_fees": 0.65,
    "termination_for_convenience": 0.65,
    "termination_for_material_breach": 0.65,
    "post_termination_services": 0.65,
    "renewal_term": 0.65,
    "none": 0.65,  # not used, but safe
}

def coerce_float(series, default=0.0):
    try:
        return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)
    except Exception:
        return pd.Series([default]*len(series), index=series.index, dtype=float)

def load_inputs(preds_path: str, labels_path: str | None):
    """
     Supports either:
       A) Separate files: preds.csv + labels.csv (joined on doc_id, clause_id), or
       B) Single file: a labels.csv that already contains both predictions and gold.
    """
    preds = pd.read_csv(preds_path) if preds_path else None

    if labels_path:
        labels = pd.read_csv(labels_path)
        if preds is None:
            # single-file mode (already merged)
            df = labels.copy()
        else:
            # join on (doc_id, clause_id)
            key_cols = ["doc_id", "clause_id"]
            missing_in_preds  = [c for c in key_cols if c not in preds.columns]
            missing_in_labels = [c for c in key_cols if c not in labels.columns]
            if missing_in_preds or missing_in_labels:
                raise SystemExit(
                    f"Join keys missing. Need {key_cols} in both files.\n"
                    f"Missing in preds: {missing_in_preds}\nMissing in labels: {missing_in_labels}"
                )
            df = pd.merge(preds, labels, on=key_cols, how="inner")
            if df.empty:
                raise SystemExit("Joined dataframe is empty. Check doc_id/clause_id alignment.")
    else:
        if preds is None:
            raise SystemExit("Provide at least one CSV.")
        df = preds.copy()

    # normalize commonly used columns’ presence
    # prediction score can be 'pred_score' or 'risk_score'
    if "pred_score" in df.columns:
        df["__score__"] = coerce_float(df["pred_score"])
    elif "risk_score" in df.columns:
        df["__score__"] = coerce_float(df["risk_score"])
    else:
        # fallback: treat missing score as 0
        df["__score__"] = 0.0

    # predicted label column (prefer CUAD-style one)
    if "pred_cuad_label" in df.columns:
        raw_pred = df["pred_cuad_label"].fillna("").astype(str)
    elif "pred_bucket" in df.columns:
        raw_pred = df["pred_bucket"].fillna("").astype(str)
    else:
        raw_pred = pd.Series([""] * len(df), index=df.index, dtype=str)

    df["__pred_label__"] = raw_pred.apply(lambda s: normalize_label(s))

    # gold columns (keep your existing checks)
    if "gold_is_risky" not in df.columns:
        raise SystemExit("Missing column 'gold_is_risky' in labels (or merged) file.")
    if "gold_primary_label" not in df.columns:
        df["gold_primary_label"] = ""

    # standardize
    df["gold_is_risky"] = df["gold_is_risky"].fillna(0).astype(int)
    df["gold_primary_label"] = df["gold_primary_label"].fillna("").astype(str)
    # NEW: normalize gold labels too
    df["gold_primary_label"] = df["gold_primary_label"].apply(lambda s: normalize_label(s))


    return df

def eval_binary(df: pd.DataFrame, base_thresh: float = 0.65):
    y_true  = df["gold_is_risky"].values.astype(int)
    y_score = df["__score__"].values.astype(float)

    # Base threshold
    y_pred_base = (y_score >= base_thresh).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred_base, average="binary", zero_division=0)
    roc   = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    prauc = average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    brier = brier_score_loss(y_true, y_score)
    base = dict(thresh=base_thresh, precision=p, recall=r, f1=f1, roc_auc=roc, pr_auc=prauc, brier=brier)

    # Class-aware threshold (optional)
    def decide_row(row):
        lbl = str(row["__pred_label__"])
        t = CAT_THRESH.get(lbl, base_thresh)
        return int(float(row["__score__"]) >= t)
    y_pred_cat = df.apply(decide_row, axis=1)
    p2, r2, f12, _ = precision_recall_fscore_support(y_true, y_pred_cat, average="binary", zero_division=0)
    cat = dict(thresh="class-aware(0.55/0.65)", precision=p2, recall=r2, f1=f12)

    return base, cat

def eval_labels(df: pd.DataFrame, show_cm: bool = False, topn: int = 10):
    # choose a single predicted label per clause: CUAD (already in __pred_label__)
    y_true_lbl = df["gold_primary_label"].astype(str)
    mask = y_true_lbl != ""
    y_true_lbl = y_true_lbl[mask]
    y_pred_lbl = df.loc[mask, "__pred_label__"].astype(str)

    if y_true_lbl.empty:
        return None, None, None

    report_txt = classification_report(y_true_lbl, y_pred_lbl, zero_division=0, digits=3)
    report_obj = classification_report(y_true_lbl, y_pred_lbl, zero_division=0, digits=3, output_dict=True)

    cm_df = None
    if show_cm:
        by_support = y_true_lbl.value_counts().index.tolist()
        labsN = by_support[:topn]
        cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=labsN)
        cm_df = pd.DataFrame(cm, index=labsN, columns=labsN)

    return report_txt, report_obj, cm_df

# ----- Main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="data/preds.csv", help="Path to predictions CSV")
    ap.add_argument("--labels", default="data/labels.csv", help="Path to labels CSV (or merged CSV)")
    ap.add_argument("--thresh", type=float, default=0.65, help="Base threshold for binary risk")
    ap.add_argument("--sweep", action="store_true", help="Run threshold sweep 0.30..0.80 in steps of 0.05")
    ap.add_argument("--show_cm", action="store_true", help="Show top-N confusion matrix")
    ap.add_argument("--topn", type=int, default=10, help="Top-N labels by support for confusion matrix")
    args = ap.parse_args()

    df = load_inputs(args.preds, args.labels)

    # Binary risk metrics
    if args.sweep:
        print("=== Threshold sweep (binary risk) ===")
        best = None
        for t in np.arange(0.30, 0.81, 0.05):
            base, cat = eval_binary(df, base_thresh=float(t))
            print(f"t={t:.2f}  Base: P={base['precision']:.3f} R={base['recall']:.3f} F1={base['f1']:.3f}  "
                  f"ROC-AUC={base['roc_auc']:.3f} PR-AUC={base['pr_auc']:.3f} Brier={base['brier']:.3f} |  "
                  f"Class-aware: P={cat['precision']:.3f} R={cat['recall']:.3f} F1={cat['f1']:.3f}")
            if best is None or base["f1"] > best["f1"]:
                best = dict(t=t, **base)
        print("\nBest base-F1 operating point:", best, "\n")
    else:
        base, cat = eval_binary(df, base_thresh=args.thresh)
        print("=== RISK (binary) ===")
        print(f"[Base @{base['thresh']}]  P={base['precision']:.3f} R={base['recall']:.3f} F1={base['f1']:.3f}  "
              f"ROC-AUC={base['roc_auc']:.3f} PR-AUC={base['pr_auc']:.3f} Brier={base['brier']:.3f}")
        print(f"[Class-aware]           P={cat['precision']:.3f} R={cat['recall']:.3f} F1={cat['f1']:.3f}\n")

    # Per-label metrics
    report_txt, _, cm_df = eval_labels(df, show_cm=args.show_cm, topn=args.topn)
    if report_txt is None:
        print("No gold_primary_label values present; skipping per-label evaluation.")
    else:
        print("=== LABELS (macro report) ===")
        print(report_txt)
        if cm_df is not None:
            print("\nTop-N Confusion Matrix (rows=true, cols=pred):")
            print(cm_df)

if __name__ == "__main__":
    main()
