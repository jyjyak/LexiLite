import re
import unicodedata

SMART_QUOTES = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
}

def _normalize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\xa0", " ")
    for k, v in SMART_QUOTES.items():
        s = s.replace(k, v)
    return s

def clean_contract_text(raw_text: str, lowercase: bool = False) -> str:
    if not raw_text:
        return ""

    cleaned = _normalize_unicode(raw_text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    footer_pattern = (
        r"(Interviewee['’]?s Signature[\s\S]+?)(Copyright|Page\s+\d+\s+of\s+\d+|$)"
    )
    cleaned = re.sub(footer_pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*Copyright\s*©?\s*\d{4}.*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

    cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)               # de-hyphenate at break
    cleaned = re.sub(r"(?<=[a-z,;:])\n(?=[a-z0-9])", " ", cleaned)  # join obvious wraps

    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    return cleaned.lower() if lowercase else cleaned


def clean_summary_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p[:1].upper() + p[1:] if p else "" for p in parts]
    return " ".join(parts).strip()
