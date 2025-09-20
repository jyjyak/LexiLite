from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import os, re, functools
from .model_paths import prefer_local, local_only

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_MODEL_ID = prefer_local("dslim/bert-base-NER", "dslim-bert-base-NER")

try:
    import torch
    _DEVICE = 0 if torch.cuda.is_available() else -1
except Exception:
    _DEVICE = -1

@functools.lru_cache(maxsize=1)
def _tok():
    return AutoTokenizer.from_pretrained(_MODEL_ID, local_files_only=local_only("dslim-bert-base-NER"))

@functools.lru_cache(maxsize=1)
def _mdl():
    return AutoModelForTokenClassification.from_pretrained(_MODEL_ID, local_files_only=local_only("dslim-bert-base-NER"))

@functools.lru_cache(maxsize=1)
def _ner_pipe():
    return pipeline(
        task="token-classification",
        model=_mdl(),
        tokenizer=_tok(),
        aggregation_strategy="simple",
        device=_DEVICE,
    )

GENERIC_PLACEHOLDERS = {"company", "interviewee", "information"}
ALLOWED_ACRONYMS = {"PTE","LTD","LLC","INC","PLC","LLP","PTE.","LTD.","LLC.","INC.","PLC.","LLP."}
GPE_MINI = {"singapore"}

_CITY_OF_RE = re.compile(r"\b(City of\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b")
_STATE_OF_RE = re.compile(r"\b(State of\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b")
_WORDPIECE = re.compile(r"##")

def _clean_wp(s: str) -> str:
    return _WORDPIECE.sub("", s).replace("“", "").replace("”", "").strip()

def _looks_like_noise(token: str) -> bool:
    if not token: return True
    if token.lower() in GENERIC_PLACEHOLDERS: return True
    if len(token) <= 3 and token.upper() not in ALLOWED_ACRONYMS: return True
    return False

def _augment_geo_spans(text: str):
    spans = []
    for m in _CITY_OF_RE.finditer(text):
        spans.append({"word": m.group(1), "entity_group": "LOC", "score": 1.0})
    for m in _STATE_OF_RE.finditer(text):
        spans.append({"word": m.group(1), "entity_group": "LOC", "score": 1.0})
    return spans

def _extend_person_plural(token: str, text: str) -> str:
    if not token or " " not in token: return token
    patt = re.compile(rf"\b{re.escape(token)}(?:'s|’s|s)?\b", re.IGNORECASE)
    cands = {m.group(0).strip() for m in patt.finditer(text)}
    if not cands: return token
    best = max(cands, key=len)
    exact = re.search(re.escape(best), text)
    return exact.group(0) if exact else best

def _expand_loc_with_number(text: str, entities: list[dict]) -> list[dict]:
    out = []
    for e in entities:
        if e["entity_group"] == "LOC":
            patt = rf"(?<!\w)(?:No\.?\s*)?\d+[A-Za-z]?(?:-\d+)?\s+{re.escape(e['word'])}(?!\w)"
            m = re.search(patt, text, flags=re.IGNORECASE)
            if m:
                expanded = m.group(0).strip()
                exact = re.search(re.escape(expanded), text)
                if exact:
                    expanded = exact.group(0)
                e = {**e, "word": expanded, "score": max(e["score"], 0.99)}
        out.append(e)
    # de-dup
    seen, dedup = set(), []
    for e in out:
        key = (e["word"].lower(), e["entity_group"])
        if key in seen: continue
        seen.add(key); dedup.append(e)
    return dedup

def extract_entities(
    text: str,
    max_tokens: int = 1200,
    stride: int = 300,
    min_score: float = 0.60,
    top_k_per_group: int | None = 30,
):
    text = (text or "").strip()
    if not text: return []

    tok = _tok()
    ner_pipeline = _ner_pipe()

    ids = tok(text, return_tensors="pt", truncation=False)["input_ids"][0]
    step = max_tokens - max(0, stride)
    chunks = []
    for start in range(0, len(ids), step):
        end = min(start + max_tokens, len(ids))
        chunks.append(tok.decode(ids[start:end], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        if end >= len(ids): break

    raw = []
    for ch in chunks:
        raw.extend(ner_pipeline(ch))
    raw.extend(_augment_geo_spans(text))

    items = []
    for e in raw:
        token = _clean_wp(e.get("word", ""))
        token = re.sub(r"^[^\w]+|[^\w]+$", "", token)
        if _looks_like_noise(token):
            continue
        if token.isupper() and len(token) <= 7 and token not in ALLOWED_ACRONYMS:
            if not re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text):
                continue
        if not re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text, flags=re.IGNORECASE):
            continue

        group = e.get("entity_group") or e.get("entity") or ""
        score = float(e.get("score", 0.0))
        if score < min_score: continue

        if token.lower() in GPE_MINI:
            group = "LOC"; score = max(score, 0.99)
        if group in ("PER","I-PER","B-PER"):
            token = _extend_person_plural(token, text)

        items.append({"word": token, "entity_group": group, "score": score})

    best = {}
    for e in items:
        k = (e["word"].lower(), e["entity_group"])
        if k not in best or e["score"] > best[k]["score"]:
            best[k] = e
    entities = list(best.values())

    entities = _expand_loc_with_number(text, entities)

    entities.sort(key=lambda x: (-len(x["word"]), -x["score"]))
    pruned = []
    for e in entities:
        if any(e["word"].lower() in o["word"].lower()
               and e["word"].lower() != o["word"].lower()
               and e["entity_group"] == o["entity_group"] for o in pruned):
            continue
        pruned.append(e)
    entities = pruned

    if top_k_per_group:
        per, trimmed = {}, []
        for e in entities:
            per.setdefault(e["entity_group"], []).append(e)
        for g, lst in per.items():
            lst.sort(key=lambda x: x["score"], reverse=True)
            trimmed.extend(lst[:top_k_per_group])
        entities = trimmed

    entities.sort(key=lambda x: x["score"], reverse=True)
    return entities
