
import os, sys
from pathlib import Path

def _base_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]

BASE_DIR   = _base_dir()
MODELS_DIR = BASE_DIR / "models"
SPACY_DIR  = BASE_DIR / "spacy_models" / "en_core_web_sm-3.7.1"

def _isdir(p: Path) -> bool:
    try:
        return p.is_dir()
    except Exception:
        return False

def prefer_local(hf_repo: str, local_subdir: str) -> str:
    local_path = MODELS_DIR / local_subdir
    return str(local_path) if _isdir(local_path) else hf_repo

def local_only(local_subdir: str) -> bool:
    return _isdir(MODELS_DIR / local_subdir)
