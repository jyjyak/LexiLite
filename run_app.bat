@echo off
setlocal

set "HF_HOME=%CD%\.hf_cache"
@REM set "TRANSFORMERS_CACHE=%CD%\.hf_cache"
set "TORCH_HOME=%CD%\.torch_cache"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

call ".\offline_venv\Scripts\activate.bat"
streamlit run main.py --server.fileWatcherType=none

endlocal
