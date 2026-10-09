#!/usr/bin/env bash
# Container start: fetch the trained models / results (if a source is configured), then serve.
set -e
cd "$(dirname "$0")"
python tools/fetch_models.py || echo "WARNING: model download failed -- starting anyway (see the log above)"
exec gunicorn -c gunicorn.conf.py wsgi:app
