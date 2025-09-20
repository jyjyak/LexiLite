@echo off
setlocal

echo ===============================
echo   AI Legal Assistant - Model Setup
echo ===============================

set "HF_HOME=%CD%\.hf_cache"
set "TORCH_HOME=%CD%\.torch_cache"

REM Activate offline_venv
call ".\offline_venv\Scripts\activate.bat"

echo.
echo Downloading models into ./models ...
python ".\scripts\download_models.py"

if %ERRORLEVEL% neq 0 (
    echo.
    echo  Error: Failed to download models.
    echo Make sure dependencies are installed via:
    echo     pip install -r requirements.txt
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo  Models downloaded successfully!
pause
endlocal
