import os, re, functools
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from .model_paths import prefer_local, local_only

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_MODEL_ID = prefer_local("mauro/bert-base-uncased-finetuned-clause-type",
                         "mauro-bert-clause-type")

try:
    import torch
    _DEVICE = 0 if torch.cuda.is_available() else -1
except Exception:
    _DEVICE = -1

@functools.lru_cache(maxsize=1)
def _tok():
    return AutoTokenizer.from_pretrained(
        _MODEL_ID,
        local_files_only=local_only("mauro-bert-clause-type"),
    )

@functools.lru_cache(maxsize=1)
def _mdl():
    return AutoModelForSequenceClassification.from_pretrained(
        _MODEL_ID,
        local_files_only=local_only("mauro-bert-clause-type"),
    )

@functools.lru_cache(maxsize=1)
def _pipe():
    return pipeline(
        "text-classification",
        model=_mdl(),
        tokenizer=_tok(),
        device=_DEVICE,
    )

def _norm(lbl: str) -> str:
    lbl = (lbl or "").strip().replace("–", "-")
    lbl = re.sub(r"[^\w]+", "_", lbl)
    lbl = re.sub(r"_+", "_", lbl).strip("_")
    return lbl.lower()

def _normalize_scores(out):
    if isinstance(out, list):
        if not out:
            return []
        if isinstance(out[0], dict):
            return out
        if isinstance(out[0], list):
            return out[0]
    if isinstance(out, dict):
        return [out]
    return []

def classify_clause(text: str, threshold: float = 0.0, top_k: int | None = None):
    out = _pipe()(text or "", truncation=True, max_length=512, top_k=None)
    scores = _normalize_scores(out)
    pairs = [(_norm(d["label"]), float(d["score"])) for d in scores]
    pairs.sort(key=lambda x: x[1], reverse=True)

    if threshold > 0.0:
        pairs = [p for p in pairs if p[1] >= threshold]
    if top_k is not None:
        pairs = pairs[:top_k]
    return pairs
