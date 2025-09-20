from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os, functools
from .model_paths import prefer_local, local_only

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_MODEL_ID = prefer_local("facebook/bart-large-mnli", "bart-large-mnli")

try:
    import torch
    _DEVICE = 0 if torch.cuda.is_available() else -1
except Exception:
    _DEVICE = -1

@functools.lru_cache(maxsize=1)
def _tok():
    return AutoTokenizer.from_pretrained(_MODEL_ID, local_files_only=local_only("bart-large-mnli"))

@functools.lru_cache(maxsize=1)
def _mdl():
    return AutoModelForSequenceClassification.from_pretrained(_MODEL_ID, local_files_only=local_only("bart-large-mnli"))

@functools.lru_cache(maxsize=1)
def _pipe():
    return pipeline(
        "zero-shot-classification",
        model=_mdl(),
        tokenizer=_tok(),
        device=_DEVICE,
        use_fast=True,
    )

CLAUSE_LABELS = [
    "termination obligations",
    "confidentiality",
    "non-compete",
    "arbitration",
    "governing law",
    "indemnification",
]

def _norm(lbl: str) -> str:
    return (lbl or "").lower().replace(" ", "_").replace("-", "_")

def zero_shot_classify(text: str, candidate_labels=None):
    text = (text or "").strip()
    if not text:
        return {"labels": [], "scores": []}

    labels = candidate_labels or CLAUSE_LABELS
    res = _pipe()(
        text,
        candidate_labels=labels,
        multi_label=True,
        hypothesis_template="This text is about {}.",
        truncation=True,
    )
    return {"labels": [_norm(l) for l in res["labels"]], "scores": res["scores"]}
