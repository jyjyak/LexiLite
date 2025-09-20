from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os, functools
from .model_paths import prefer_local, local_only

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_FALLBACK = os.getenv("SUMMARY_MODEL", "facebook/bart-large-cnn")
_MODEL_ID = prefer_local(_FALLBACK, "bart-large-cnn")

try:
    import torch
    _DEVICE = 0 if torch.cuda.is_available() else -1
except Exception:
    _DEVICE = -1

@functools.lru_cache(maxsize=1)
def _tok():
    return AutoTokenizer.from_pretrained(_MODEL_ID, local_files_only=local_only("bart-large-cnn"))

@functools.lru_cache(maxsize=1)
def _mdl():
    return AutoModelForSeq2SeqLM.from_pretrained(_MODEL_ID, local_files_only=local_only("bart-large-cnn"))

@functools.lru_cache(maxsize=1)
def _smz():
    return pipeline(
        "summarization",
        model=_mdl(),
        tokenizer=_tok(),
        device=_DEVICE,
        use_fast=True,
    )

def _safe_lengths(n_tokens, min_len, max_len):
    target = max(30, min(220, int(0.2 * n_tokens)))
    max_len_eff = max(40, min(max_len, target + 40))
    min_len_eff = max(10, min(min_len, max_len_eff - 10))
    if min_len_eff >= max_len_eff:
        min_len_eff = max_len_eff - 5
    return min_len_eff, max_len_eff

def summarize_text(
    text: str,
    min_len: int = 30,
    max_len: int = 200,
    chunk_overlap: int = 30
    bullets: bool = True,
    combine: bool = False,
    debug: bool = False,
) -> str:
    text = (text or "").strip()
    if len(text) < 50:
        return "-"

    tok = _tok()
    smz = _smz()

    ids = tok(text, return_tensors="pt", truncation=False)["input_ids"][0]

    model_max = getattr(tok, "model_max_length", 1024)
    max_chunk_len = min(800, model_max - 24) if _DEVICE == -1 else min(1000, model_max - 24)

    step = max_chunk_len - max(0, chunk_overlap)
    chunks = []
    for start in range(0, len(ids), step):
        end = min(start + max_chunk_len, len(ids))
        chunk_ids = ids[start:end]
        if len(chunk_ids) >= 10:
            chunks.append(tok.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))

    if not chunks:
        return "Summary could not be generated."

    outputs = []
    beams = 2 if _DEVICE == -1 else 4

    for c in chunks:
        mn, mx = _safe_lengths(len(tok(c)["input_ids"]), min_len, max_len)
        result = smz(
            c,
            min_length=mn,
            max_length=mx,
            do_sample=False,
            num_beams=beams,
            no_repeat_ngram_size=3
        )
        outputs.append(result[0]["summary_text"].strip())

    if combine and len(outputs) > 1:
        combined = " ".join(outputs)
        mn, mx = _safe_lengths(len(tok(combined)["input_ids"]), max(40, min_len), min(220, max_len))
        outputs = [smz(
            combined, min_length=mn, max_length=mx, do_sample=False, num_beams=beams, no_repeat_ngram_size=3
        )[0]["summary_text"].strip()]

    diag = f"\n\n[chunks: {len(chunks)} | out: {len(outputs)} | tokens: {len(ids)} | device: {'CPU' if _DEVICE==-1 else 'GPU'}]" if debug else ""
    return ("\n\n".join(f"- {s}" for s in outputs) if bullets else "\n\n".join(outputs)) + diag
