"""gunicorn settings for hosting (Docker / Hugging Face Spaces / Render / a VM).

ONE worker process on purpose: every worker would load all models into its own memory (the
EfficientNetB7-based models are several hundred MB each). Requests are served by threads; the
app itself queues the actual inference (one diagnosis at a time per worker)."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '7860')}"
workers = 1
worker_class = 'gthread'
threads = int(os.environ.get('THREADS', '8'))
timeout = int(os.environ.get('WORKER_TIMEOUT', '300'))   # a compare of 5 large models on CPU can take a while
graceful_timeout = 30
keepalive = 5
accesslog = '-'
errorlog = '-'
loglevel = os.environ.get('LOG_LEVEL', 'info')
forwarded_allow_ips = '*'                                 # behind the hosting proxy
