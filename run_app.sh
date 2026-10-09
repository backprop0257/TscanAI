#!/usr/bin/env bash
# TomatoLeafAI v21 -- start the web app on Linux / macOS (run from this folder)
set -e
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
python app.py
