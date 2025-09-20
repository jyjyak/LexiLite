# download_models.py
"""
Download all required HuggingFace models into ./models for offline use.
"""

import os
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODELS = {
    # summarizer
    "facebook/bart-large-cnn": "bart-large-cnn",

    # zero-shot
    "facebook/bart-large-mnli": "bart-large-mnli",

    # NER
    "dslim/bert-base-NER": "dslim-bert-base-NER",

    # clause classifier (CUAD or Mauro)
    "mauro/bert-base-uncased-finetuned-clause-type": "mauro-bert-clause-type",
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_model(hf_id: str, local_name: str):
    local_path = MODELS_DIR / local_name
    ensure_dir(local_path)
    print(f"⬇️ Downloading {hf_id} → {local_path}")

    if "bart-large-cnn" in hf_id:
        AutoTokenizer.from_pretrained(hf_id).save_pretrained(local_path)
        AutoModelForSeq2SeqLM.from_pretrained(hf_id).save_pretrained(local_path)

    elif "mnli" in hf_id:
        AutoTokenizer.from_pretrained(hf_id).save_pretrained(local_path)
        AutoModelForSequenceClassification.from_pretrained(hf_id).save_pretrained(local_path)

    elif "NER" in hf_id:
        AutoTokenizer.from_pretrained(hf_id).save_pretrained(local_path)
        AutoModelForTokenClassification.from_pretrained(hf_id).save_pretrained(local_path)

    else:  # Mauro clause-type model
        AutoTokenizer.from_pretrained(hf_id).save_pretrained(local_path)
        AutoModelForSequenceClassification.from_pretrained(hf_id).save_pretrained(local_path)

    print(f"✅ Done: {hf_id}\n")


def main():
    for hf_id, local_name in MODELS.items():
        download_model(hf_id, local_name)
    print("🎉 All models downloaded into ./models/")


if __name__ == "__main__":
    main()
