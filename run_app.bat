@echo off
REM TomatoLeafAI v21 -- start the web app on Windows (run from this folder)
if not exist .venv (
  py -3.12 -m venv .venv || py -3.11 -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
python app.py
