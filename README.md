# ⚖️ AI Legal Document Assistant  

An **experimental prototype** that analyzes legal contracts using multiple NLP models. It can:  
- 📄 **Summarize** each clause of a contract  
- 🏷️ **Detect entities** (names, dates, organizations, etc.)  
- ⚠️ **Highlight risky clauses** (termination, indemnity, arbitration, etc.)  
- 🧠 **Classify clauses** using a fine-tuned CUAD/Mauro model and a zero-shot classifier  

---

## 🖥️ Requirements
- Windows 10 or 11 (64-bit)  
- At least **4 GB of free RAM** recommended  
- At least **19 GB of free disk space** recommended
---

## ⚠️ Disclaimer
This is an **experimental prototype**.  
It is **not a substitute for professional legal advice**.  
Always consult a qualified lawyer for legal matters.  

---

## 📂 Folder Structure
```
legal-assistant/
│
├── data/                   # Stores prediction results
├── models/                 # HuggingFace models (downloaded via script)
├── nlp_modules/            # NLP modules
│   ├── clauses.py
│   ├── cuad_classifier.py
│   ├── label_normalizer.py
│   ├── model_paths.py
│   ├── ner.py
│   ├── risk_scoring.py
│   ├── summarizer.py
│   └── zero_shot.py
│
├── scripts/                # Utility scripts
│   ├── download_models.py
│   └── eval_offline.py
│
├── spacy_models/           # Local spaCy model (en_core_web_sm)
│   └── en_core_web_sm-3.7.1/
│
├── ui/                     # UI modules
│   └── upload.py
│
├── utils/                  # Utilities
│   ├── eval_autowrite.py
│   ├── eval_logging.py
│   └── text_cleaning.py
│
├── run_app.bat             # Launches the app
├── download_models.bat     # Downloads HuggingFace models
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── main.py                 # Streamlit entrypoint
```

---

## 🚀 Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/legal-assistant.git
cd legal-assistant
```

### 2. Create a virtual environment  
(Must be named `offline_venv` because the `.bat` files depend on this name.)  
```bash
python -m venv offline_venv
```

### 3. Activate the environment
```bash
# PowerShell
.\offline_venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Download models (first time only)  
Double-click:  
```
download_models.bat
```
This will download:
- `facebook/bart-large-cnn` (Summarization)  
- `facebook/bart-large-mnli` (Zero-Shot Classification)  
- `dslim/bert-base-NER` (NER)  
- `mauro/bert-base-uncased-finetuned-clause-type` (Clause Classification)  

Models are stored in `./models/` and loaded locally, so the app works **fully offline**.

### 6. Run the app  
Double-click:  
```
run_app.bat
```
Or run manually:  
```bash
streamlit run main.py
```

---

## ✨ Features
- **Summarization**: Breaks contracts into clauses and generates concise summaries.  
- **Named Entity Recognition (NER)**: Detects parties, dates, monetary amounts, etc.  
- **Clause Classification**: Classifies each clause using CUAD/Mauro model + zero-shot fallback.  
- **Risk Detection**: Flags risky clauses (e.g., indemnity, arbitration, termination).  
- **Offline-ready**: All models are stored locally for private, offline use.  

---

