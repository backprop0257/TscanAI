# TomatoLeafAI v24 web app -- CPU image for Hugging Face Spaces (Docker SDK), Render, Railway, Cloud Run or any VM.
#   docker build -t tomatoleafai .
#   docker run -p 7860:7860 -v "$PWD/models:/home/user/app/models" -v "$PWD/results:/home/user/app/results" tomatoleafai
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=1 \
    PORT=7860 PUBLIC_MODE=1 AUTO_RESCAN=0 HOME=/home/user

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as user 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements-docker.txt ./
RUN pip install -r requirements-docker.txt

COPY --chown=user:user . .
RUN mkdir -p models results instance && chown -R user:user /home/user/app && chmod +x start.sh
USER user

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/health',timeout=4)" || exit 1
CMD ["bash", "start.sh"]
