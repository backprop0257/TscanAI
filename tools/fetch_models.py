"""Download the trained models and the notebook's results at container start (online hosting).

Sources (first one configured wins; nothing configured = use whatever is already in models/ and results/):

  HF_MODEL_REPO=<user>/<repo>     a Hugging Face model repo laid out as  models/...  and  results/...
                                  (tools/push_to_hf.py creates it). Private repo: also set HF_TOKEN (a secret).
  MODEL_URLS=<json>               [{"url": "https://.../Hybrid_CNN_ViT_final.keras", "path": "models/Hybrid_CNN_ViT_final.keras"}, ...]
                                  any direct-download links (GitHub release assets, S3, a web server ...)

Files that are already present with the same size are not downloaded again.
"""
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parent.parent


def from_hf(repo, token):
    from huggingface_hub import snapshot_download
    print(f'[fetch] Hugging Face repo {repo} -> models/ and results/')
    snapshot_download(repo_id=repo, repo_type=os.environ.get('HF_MODEL_REPO_TYPE', 'model'), local_dir=str(APP),
                      allow_patterns=['models/*', 'models/**', 'results/*', 'results/**'], token=token or None,
                      max_workers=4)


def from_urls(items):
    for it in items:
        dest = (APP / it['path']).resolve()
        if APP not in dest.parents:
            print(f"[fetch] skipped {it['path']}: outside the app folder"); continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(it['url'], headers={'User-Agent': 'TomatoLeafAI'})
        with urllib.request.urlopen(req, timeout=600) as r:
            size = int(r.headers.get('Content-Length') or 0)
            if dest.exists() and size and dest.stat().st_size == size:
                print(f"[fetch] {it['path']} up to date"); continue
            tmp = dest.with_suffix(dest.suffix + '.part')
            with open(tmp, 'wb') as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(tmp, dest)
        print(f"[fetch] {it['path']} ({dest.stat().st_size / 1e6:.1f} MB)")


def main():
    repo = os.environ.get('HF_MODEL_REPO', '').strip()
    urls = os.environ.get('MODEL_URLS', '').strip()
    if repo:
        from_hf(repo, os.environ.get('HF_TOKEN', ''))
    elif urls:
        from_urls(json.loads(urls))
    else:
        print('[fetch] no HF_MODEL_REPO / MODEL_URLS set -- using the files already in models/ and results/')
    n = len([p for p in (APP / 'models').glob('*') if p.suffix in ('.keras', '.h5') or p.is_dir()])
    print(f'[fetch] models/ now holds {n} model file(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
