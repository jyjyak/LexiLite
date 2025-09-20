import streamlit as st
import re, time, os, io, hashlib
import spacy
from pathlib import Path
from typing import Dict, Any, List

# --- local modules ---
from ui.upload import extract_text_from_pdf
from utils.text_cleaning import clean_contract_text, clean_summary_text
from utils.eval_autowrite import write_eval_artifacts

from nlp_modules.model_paths import SPACY_DIR
from nlp_modules.label_normalizer import normalize_label
from nlp_modules.summarizer import summarize_text
from nlp_modules.ner import extract_entities
from nlp_modules.clauses import (
    detect_risk_clauses,
    RISKY_LABELS_ALL,
    REGEX_TO_RISK_LABELS,
    is_actionable,
)
from nlp_modules.cuad_classifier import classify_clause
from nlp_modules.zero_shot import zero_shot_classify
from nlp_modules.risk_scoring import compute_risk_features, bucket


# =========================
# Page config + CSS
# =========================
st.set_page_config(page_title="AI Legal Document Assistant", page_icon="📄", layout="wide")

THEME_BG = "#0f172a"   # slate-900
CARD_BG  = "#111827"   # near-black
BORDER   = "#1f2937"   # slate-800
ACCENT   = "#22d3ee"   # cyan-400
WARN     = "#f59e0b"   # amber-500
DANGER   = "#ef4444"   # red-500
OK       = "#10b981"   # emerald-500

st.markdown(f"""
<style>
/* Constrain the central content to a comfortable max width */
.block-container {{
  max-width: 1080px;          /* <- change this number to taste (e.g., 1000–1280) */
  padding-top: 1rem;
  padding-bottom: 2rem;
  margin: 0 auto !important;  /* center on ultrawide monitors */
}}
@media (min-width: 1600px) {{
  .block-container {{ max-width: 1100px; }}    /* a bit narrower on very wide screens */
}}

/* Card look */
.card {{
  background: {CARD_BG};
  border: 1px solid {BORDER};
  border-radius: 14px;
  padding: 14px 16px;         /* balanced inner padding */
  margin: 12px 0;             /* consistent vertical rhythm */
}}

/* Badges */
.badge {{
  display:inline-block; padding:2px 8px; border-radius:10px; font-size:12px; font-weight:600;
  border:1px solid {BORDER}; background:#0b1220;
}}
.badge.high {{ color:{DANGER}; border-color:{DANGER}22; }}
.badge.med  {{ color:{WARN};   border-color:{WARN}22; }}
.badge.low  {{ color:{OK};     border-color:{OK}22; }}

/* Highlight */
mark {{ background: #fde68a66; color: inherit; padding: 0 3px; border-radius: 4px; }}

/* A narrow wrapper so legend/NER share the same exact width */
.narrow {{ max-width: 1180px; margin: 0 auto; }}   /* match .block-container max-width */

/* Sticky legend (but limited to the same width via .narrow parent) */
.legend-sticky {{
  position: sticky; top: 0; z-index: 5;
  
  background: linear-gradient(180deg, rgba(15,23,42,0.9) 0%, rgba(15,23,42,0.6) 100%);
  backdrop-filter: blur(6px); border-bottom: 1px solid {BORDER};
}}
</style>
""", unsafe_allow_html=True)


# =========================
# spaCy (local) loader
# =========================
@st.cache_resource(show_spinner=False)
def load_spacy():
    if not SPACY_DIR.exists():
        raise FileNotFoundError(f"spaCy model not found at: {SPACY_DIR}")
    return spacy.load(str(SPACY_DIR))

nlp = load_spacy()


# =========================
# Sidebar (controls)
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    ack = st.checkbox("I accept the disclaimer", value=False)
    uploaded_file = st.file_uploader("Upload contract (PDF)", type=["pdf"])
    run_now = st.button("▶ Run analysis")         # ← Only this triggers heavy work
    st.divider()

    st.markdown("### 🔎 Display & Filters")
    risk_threshold = st.slider("Minimum risk score to show", 0.0, 1.0, 0.50, 0.01)
    show_regex_only = st.checkbox("Show clauses with regex risk only", value=False)
    search_q = st.text_input("Search clause text", "")



# =========================
# Header + disclaimer
# =========================
st.title("AI Legal Document Assistant")
with st.expander("🔍 Disclaimer (Please Read)", expanded=True):
    st.markdown("""
    **⚠️ Important Notice:**

    This tool is an experimental prototype designed to assist users in understanding legal contract content. It is **not a substitute for professional legal advice**. The summaries generated are automated and may omit critical legal nuances or misrepresent terms.

    By using this tool, you acknowledge that:
    - You will **not rely** on this system as legal counsel.
    - You understand that the analysis is purely informational.
    - You accept full responsibility for any decisions made based on the output.

    Your document will be processed **locally on your device**. No data is uploaded, saved, or transmitted externally.
    """)
if not ack:
    st.warning("Please accept the disclaimer to proceed.")
    st.stop()


# =========================
# Helpers: caching keys & pure cached functions
# =========================
def _file_bytes(file) -> bytes:
    return file.getvalue() if file else b""

def _doc_key(file) -> str:
    b = _file_bytes(file)
    return hashlib.sha1(b).hexdigest() if b else "no_file"

@st.cache_data(show_spinner=False)
def cached_extract_text(pdf_bytes: bytes) -> str:
    return extract_text_from_pdf(io.BytesIO(pdf_bytes))

@st.cache_data(show_spinner=False)
def cached_clean_text(raw_text: str) -> str:
    return clean_contract_text(raw_text)

@st.cache_data(show_spinner=False)
def cached_split(text: str) -> List[str]:
    parts = re.split(r'(?m)^\s*(?=\d+\.\s+)', text)
    parts = [p.strip() for p in parts if p.strip()]
    paras = []
    for p in parts:
        paras.extend(s.strip() for s in re.split(r'(?=\n?[A-Z]\.\s)', p) if s.strip())
    return paras

def _new_bar(label: str):
    ph = st.empty()
    ph.markdown(f"**{label}**")
    bar = st.progress(0.0)
    def set_frac(frac: float):
        try:
            bar.progress(min(max(frac, 0.0), 1.0))
        except Exception:
            pass
    def close():
        ph.empty()
        bar.empty()
    return set_frac, close

@st.cache_data(show_spinner=False)
def cached_summaries(paragraphs: list[str], show_progress: bool = True) -> list[str]:
    setp, close = (lambda *_: None, lambda: None)
    if show_progress:
        setp, close = _new_bar("Summarizing…")
    outs, n = [], max(1, len(paragraphs))
    for i, para in enumerate(paragraphs):
        para_clean = re.sub(r'^[A-Z]\.\s+', '', para)
        s = summarize_text(para_clean, min_len=30, max_len=100)
        s = clean_summary_text(s)
        if s and s.strip() != "-" and len(s.strip()) > 10:
            outs.append(s.strip())
        if show_progress:
            setp((i + 1) / n)
    close()
    return outs

@st.cache_data(show_spinner=False)
def cached_entities(text: str, show_progress: bool = True, granular: bool = False) -> list[dict]:
    if not granular:
        setp, close = (lambda *_: None, lambda: None)
        if show_progress:
            setp, close = _new_bar("Extracting entities…")
            setp(0.05)
        ents = extract_entities(text, max_tokens=1200, stride=300, min_score=0.60, top_k_per_group=30)
        if show_progress:
            setp(1.0); close()
        return ents

    setp, close = (lambda *_: None, lambda: None)
    if show_progress:
        setp, close = _new_bar("Extracting entities (chunked)…")

    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    outs, seen = [], set()
    n = max(1, len(paras))
    for i, p in enumerate(paras):
        ents = extract_entities(
            p, max_tokens=1200, stride=300, min_score=0.60, top_k_per_group=30
        )
        for e in ents:
            key = (e["word"], e["entity_group"], round(e["score"], 2))
            if key not in seen:
                seen.add(key); outs.append(e)
        if show_progress:
            setp((i + 1) / n)
    close()
    return outs

@st.cache_data(show_spinner=False)
def cached_results(paragraphs: list[str], show_progress: bool = True) -> list[dict]:
    setp, close = (lambda *_: None, lambda: None)
    if show_progress:
        setp, close = _new_bar("Classifying clauses & scoring risk…")
    results = []
    n = max(1, len(paragraphs))
    for i, para in enumerate(paragraphs):
        para_clean = re.sub(r'^[A-Z]\.\s+', '', para).strip()
        if len(para_clean.split()) < 10: 
            if show_progress: setp((i + 1) / n)
            continue
        if re.match(r'^[IVXLCDM]+\.\s+.*$', para_clean): 
            if show_progress: setp((i + 1) / n)
            continue
        if not is_actionable(para_clean): 
            if show_progress: setp((i + 1) / n)
            continue

        local_hits = detect_risk_clauses(para_clean)
        local_regex_labels = {
            REGEX_TO_RISK_LABELS.get(m["type"]) for m in local_hits if REGEX_TO_RISK_LABELS.get(m["type"])
        }
        regex_hits = len(local_hits)

        raw_preds_raw = classify_clause(para_clean, threshold=0.0)
        raw_preds = [(normalize_label(lbl), sc) for lbl, sc in raw_preds_raw]
        cuad_scores_for_risk = [{"label": lbl, "score": sc} for (lbl, sc) in raw_preds]

        zs_raw = zero_shot_classify(para_clean)
        zs = {"labels": [normalize_label(lbl) for lbl in zs_raw["labels"]], "scores": zs_raw["scores"]}

        feats = compute_risk_features(
            clause_text=para_clean,
            cuad_scores=cuad_scores_for_risk,
            zs_labels=zs["labels"],
            zs_scores=zs["scores"],
            regex_hits=regex_hits,
            actionable=True,
            matched_labels=local_regex_labels,
        )
        results.append({
            "idx": i,
            "text": para_clean,
            "raw_preds": raw_preds,
            "zs": zs,
            "regex_labels": local_regex_labels,
            "regex_hits": regex_hits,
            "features": feats,
            "risk_bucket": bucket(feats["score"]),
        })
        if show_progress:
            setp((i + 1) / n)
    results.sort(key=lambda r: r["features"]["score"], reverse=True)
    close()
    return results

eval_log_path = "data/preds.csv"
labels_csv_path = "data/labels.csv"
doc_id = getattr(uploaded_file, "name", "uploaded.pdf") if uploaded_file else "uploaded.pdf"
if "metrics" not in st.session_state:
    st.session_state.metrics = {"summarize_sec": 0.0, "ner_sec": 0.0, "classify_sec": 0.0}
metrics = st.session_state.metrics

if uploaded_file:
    key = _doc_key(uploaded_file)

    if ("doc_key" not in st.session_state) or (st.session_state.doc_key != key) or run_now:
        updating_doc = ("doc_key" not in st.session_state) or (st.session_state.doc_key != key)

        raw_text     = cached_extract_text(_file_bytes(uploaded_file))
        cleaned_text = cached_clean_text(raw_text)
        paragraphs   = cached_split(cleaned_text)


        t_sum = time.perf_counter()
        summaries = cached_summaries(paragraphs, show_progress=True)
        summarize_sec = time.perf_counter() - t_sum

        t_ner = time.perf_counter()
        entities = cached_entities(cleaned_text, show_progress=True, granular=False)
        ner_sec = time.perf_counter() - t_ner

        t_cls = time.perf_counter()
        results = cached_results(paragraphs, show_progress=True)
        classify_sec = time.perf_counter() - t_cls


        st.session_state.doc_key     = key
        st.session_state.cleaned_text = cleaned_text
        st.session_state.paragraphs   = paragraphs
        st.session_state.summaries    = summaries
        st.session_state.entities     = entities
        st.session_state.results      = results

        if updating_doc or run_now:
            st.session_state.metrics.update({
                "summarize_sec": summarize_sec,
                "ner_sec": ner_sec,
                "classify_sec": classify_sec,
            })

        
    cleaned_text = st.session_state.cleaned_text
    paragraphs   = st.session_state.paragraphs
    summaries    = st.session_state.summaries
    entities     = st.session_state.entities
    results      = st.session_state.results

    # =========================
    # Regex highlight (uses cached cleaned_text only)
    # =========================
    st.subheader("⚠️ Risky Clauses (Regex-Matched Snippets Only)")
    risk_matches = detect_risk_clauses(cleaned_text)
    doc = nlp(cleaned_text)
    sentences = [s for s in doc.sents]

    matched_sentences, seen = [], set()
    for item in risk_matches:
        m_start, m_end = item["start"], item["end"]
        for sent in sentences:
            if sent.start_char <= m_start and m_end <= sent.end_char:
                key = (item["type"], sent.start_char, sent.end_char)
                if key in seen: break
                seen.add(key)
                rel_start, rel_end = m_start - sent.start_char, m_end - sent.start_char
                s_text = sent.text
                highlighted = s_text[:rel_start] + f"<mark>{s_text[rel_start:rel_end]}</mark>" + s_text[rel_end:]
                matched_sentences.append((item["type"], highlighted))
                break

    if matched_sentences:
        for clause_type, sentence in matched_sentences:
            st.markdown(f"**{clause_type} Clause:**")
            st.markdown(f"<div class='card' style='white-space: pre-wrap;'>{sentence}</div>", unsafe_allow_html=True)
        st.info(f"⚠️ {len(matched_sentences)} risky clause(s) shown.")
    else:
        st.success("No risky clauses detected.")

    # =========================
    # Summary (cached)
    # =========================
    st.subheader("Summary")
    if summaries:
        st.markdown('<div class="card"><b>Key Points</b><br>', unsafe_allow_html=True)
        for s in summaries:
            st.markdown(f"- {s}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No valid summaries were generated.")

    # =========================
    # NER (cached)
    # =========================
    st.subheader("Named Entities (General NER)")
    if entities:
        chips = " ".join(
            f'<span class="badge" title="score {e["score"]:.2f}">{e["word"]} — {e["entity_group"]}</span>'
            for e in entities
        )
        st.markdown(f'<div class="card">{chips}</div>', unsafe_allow_html=True)
    else:
        st.info("No entities found.")

    # =========================
    # Sticky legend + tabs
    # =========================
    st.markdown("""
    <div class="narrow">
    <div class="legend-sticky card">
        <b>Legend</b> —
        <span class="badge high">High risk</span>
        <span class="badge med">Medium</span>
        <span class="badge low">Low</span>
        &nbsp;|&nbsp; ✅ Confirmed Risk · ⚠️ Model-only Risk · 🔒 Regex Risk · 🧠 Informational
    </div>
    </div>
    """, unsafe_allow_html=True)

    tab_overview, tab_clauses, tab_exports = st.tabs(["Overview", "Clauses", "Exports"])

    metrics = st.session_state.metrics
    # Overview
    with tab_overview:
        classified_count = len(results)
        cols = st.columns(4)
        cols[0].metric("Actionable Clauses", value=classified_count)
        cols[1].metric("Summarization (s)", value=f"{metrics['summarize_sec']:.2f}")
        cols[2].metric("NER (s)", value=f"{metrics['ner_sec']:.2f}")
        cols[3].metric("Classification (s)", value=f"{metrics['classify_sec']:.2f}")
        hi = sum(r["risk_bucket"] == "High" for r in results)
        md = sum(r["risk_bucket"] == "Medium" for r in results)
        lo = sum(r["risk_bucket"] == "Low" for r in results)
        st.markdown(
            f'<div class="card">Risk distribution: '
            f'<span class="badge high">High {hi}</span> '
            f'<span class="badge med">Med {md}</span> '
            f'<span class="badge low">Low {lo}</span></div>',
            unsafe_allow_html=True
        )

    # Clauses — filter ON CACHED DATA ONLY (filter by unified risk score)
    def passes_filters(result, risk_threshold: float, show_rx_only: bool, q: str) -> bool:
        # text search
        txt_ok = (q.lower() in result["text"].lower()) if q else True
        # gate by overall risk score (from risk_scoring.py)
        risk_ok = result["features"]["score"] >= risk_threshold
        # regex-only filter
        rx_only_ok = (len(result["regex_labels"]) > 0) if show_rx_only else True
        return txt_ok and risk_ok and rx_only_ok

    with tab_clauses:
        # apply the filter using the sidebar's risk_threshold
        f_results = [r for r in results if passes_filters(r, risk_threshold, show_regex_only, search_q)]

        st.caption(f"Showing {len(f_results)} of {len(results)} clauses (risk ≥ {risk_threshold:.2f})")

        for r in f_results:
            rb = r["risk_bucket"]
            badge_cls = "high" if rb == "High" else ("med" if rb == "Medium" else "low")
            title = f'<span class="badge {badge_cls}">{rb}</span> <b>Score {r["features"]["score"]:.2f}</b>'
            st.markdown(f'<div class="card">{title}<br><small>{r["text"][:300]}...</small>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Clause Classifier**")
                if r["raw_preds"]:
                    lbl, sc = r["raw_preds"][0]
                    human = lbl.replace("_", " ").title()
                    risky = lbl in RISKY_LABELS_ALL
                    confirmed = risky and (lbl in r["regex_labels"])
                    if confirmed: st.error(f"✅ Confirmed Risk — {human} ({sc:.2f})")
                    elif risky:   st.warning(f"⚠️ Model-only Risk — {human} ({sc:.2f})")
                    else:         st.write(f"🧠 {human} ({sc:.2f})")
                else:
                    st.write("—")
                for rx_lbl in sorted(r["regex_labels"]):
                    if (not r["raw_preds"]) or rx_lbl != r["raw_preds"][0][0]:
                        st.info(f"🔒 Regex Risk — {rx_lbl.replace('_',' ').title()}")

            with c2:
                st.markdown("**Zero-Shot (fallback)**")
                # show ZS only if the main classifier is not confident
                CUAD_CUTOFF = 0.49 
                show_zs = (not r["raw_preds"]) or (r["raw_preds"][0][1] < CUAD_CUTOFF)

                if show_zs:
                    shown = False
                    for lbl, sc in list(zip(r["zs"]["labels"], r["zs"]["scores"]))[:3]:
                        if sc >= 0.50:
                            st.write(f"🔍 {lbl.replace('_',' ').title()} — {sc:.2f}")
                            shown = True
                    if not shown:
                        st.write("—")
                else:
                    st.caption("CUAD confident; zero-shot hidden")

            st.markdown("</div>", unsafe_allow_html=True)



    # Exports — build from cached results only
    with tab_exports:
        eval_rows, classified_count = [], 0
        for r in results:
            classified_count += 1
            para_clean = r["text"]
            feats = r["features"]
            pred_cuad_label = r["raw_preds"][0][0] if r["raw_preds"] else ""
            pred_cuad_score = r["raw_preds"][0][1] if r["raw_preds"] else 0.0
            zs_top_label = r["zs"]["labels"][0] if r["zs"]["labels"] else ""
            zs_top_score = r["zs"]["scores"][0] if r["zs"]["scores"] else 0.0

            eval_rows.append({
                "doc_id": doc_id,
                "clause_id": classified_count,
                "clause_text": para_clean,
                "pred_score": feats["score"],
                "pred_bucket": r["risk_bucket"],
                "pred_cuad_label": pred_cuad_label,
                "pred_cuad_score": pred_cuad_score,
                "regex_labels": ";".join(sorted(lbl for lbl in r["regex_labels"] if lbl)),
                "zs_top_label": zs_top_label,
                "zs_top_score": zs_top_score,
                "f1_cuad_max": feats["f1_cuad_max"],
                "f2_cuad_mass": feats["f2_cuad_mass"],
                "f3_zs_max": feats["f3_zs_max"],
                "f4_regex": feats["f4_regex"],
                "f5_actionable": feats["f5_actionable"],
            })

        if eval_rows:
            write_eval_artifacts(
                eval_rows,
                preds_csv=eval_log_path,
                labels_csv=labels_csv_path,
                overwrite_labels_if_exists=False,
            )
            st.success("Wrote predictions saved to data/preds.csv")
