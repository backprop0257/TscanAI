"""Publish TomatoLeafAI online on Hugging Face Spaces in one command (run it on the machine that has the
trained project folder; needs  pip install huggingface_hub  and a write token from
https://huggingface.co/settings/tokens).

    python tools/push_to_hf.py --project /path/to/TomatoLeafAI_Project_v5 \
        --space  <your-hf-name>/tomatoleafai \
        --models-repo <your-hf-name>/tomatoleafai-models  [--private-models] [--admin-token SOMETHING]

What it does
  1. models repo : uploads every trained classifier from <project>/models (found the same way the app finds
                   them -- checkpoints, *_pvonly and stage files are skipped) + ood_gate.keras,
                   feature_ood_v21.json, class_indices.json, and the notebook's results/ (*.json, *.csv,
                   eval/, plots/) -- no datasets, no caches.
  2. Space       : creates a Docker Space and uploads this app's code (with a Space README header).
  3. settings    : sets the Space variables PUBLIC_MODE=1 and HF_MODEL_REPO=<models repo>, and the secrets
                   HF_TOKEN (only if the models repo is private) and ADMIN_TOKEN (only if given).
The Space then builds (about 5-10 minutes the first time) and downloads the models when it starts.
Re-run any time to update models, results or code (only changed files are uploaded).
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
from utils import registry as REG          # noqa: E402

AUX = ['ood_gate.keras', 'feature_ood_v21.json', 'class_indices.json', 'model_info.json']
SPACE_HEADER = """---
title: TomatoLeafAI
emoji: 🍅
colorFrom: green
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Tomato leaf disease diagnosis with a hybrid CNN-ViT
---

"""


def model_files(project):
    """[(path_in_repo, local_path)] for every file the app needs -- found the same way the app finds them."""
    sm, sr = project / 'models', project / 'results'
    if not sm.exists():
        sm = project
    out = {}
    found = REG.discover(sm)
    for _, e in found.items():
        p = Path(e['path'])
        files = [x for x in p.rglob('*') if x.is_file()] if p.is_dir() else [p]
        for f in files:
            out[f'models/{f.relative_to(sm).as_posix()}'] = f
        for extra in sm.glob(f'{p.stem}*.json'):
            out[f'models/{extra.name}'] = extra
        print(f'  model  {p.name}  ({e["size_mb"]} MB)')
    for f in AUX + [x.name for x in sm.glob('*_classes.json')]:
        if (sm / f).exists():
            out.setdefault(f'models/{f}', sm / f)
    if 'models/model_info.json' not in out and (APP / 'model_info.json').exists():
        out['models/model_info.json'] = APP / 'model_info.json'
    skip = ('_pred_cache', '_ood_stress_tmp', '_archive')
    if sr.exists():
        for f in list(sr.glob('*.json')) + list(sr.glob('*.csv')):
            out[f'results/{f.name}'] = f
        for d in ('eval', 'plots'):
            for f in (sr / d).rglob('*') if (sr / d).exists() else []:
                rel = f.relative_to(sr).as_posix()
                if f.is_file() and f.suffix != '.npz' and not any(k in rel for k in skip):
                    out[f'results/{rel}'] = f
    return len(found), out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', required=True, help="the notebook's project folder (contains models/ and results/)")
    ap.add_argument('--space', required=True, help='<user>/<space-name>')
    ap.add_argument('--models-repo', required=True, help='<user>/<repo-name> for the model files')
    ap.add_argument('--private-models', action='store_true', help='keep the model repo private (the Space gets a read token)')
    ap.add_argument('--private-space', action='store_true')
    ap.add_argument('--admin-token', default='', help='optional: enables model upload / rescan on the site for whoever has it')
    ap.add_argument('--token', default=os.environ.get('HF_TOKEN', ''), help='HF write token (or set HF_TOKEN / run `huggingface-cli login`)')
    ap.add_argument('--skip-models', action='store_true', help='only update the app code')
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=a.token or None)
    project = Path(a.project).expanduser().resolve()

    if not a.skip_models:
        print(f'[1/3] model repo {a.models_repo}')
        api.create_repo(a.models_repo, repo_type='model', private=a.private_models, exist_ok=True)
        from huggingface_hub import CommitOperationAdd
        n, files = model_files(project)
        if not n:
            sys.exit(f'No trained models found in {project}/models -- check --project.')
        print(f'  uploading {len(files)} files ({sum(p.stat().st_size for p in files.values()) / 1e6:.0f} MB) ...')
        api.create_commit(repo_id=a.models_repo, repo_type='model', commit_message='TomatoLeafAI models + results',
                          operations=[CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in files.items()])
    print(f'[2/3] Space {a.space}')
    api.create_repo(a.space, repo_type='space', space_sdk='docker', private=a.private_space, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        readme = Path(tmp) / 'README.md'
        readme.write_text(SPACE_HEADER + (APP / 'README.md').read_text(encoding='utf-8'), encoding='utf-8')
        api.upload_folder(folder_path=str(APP), repo_id=a.space, repo_type='space', commit_message='TomatoLeafAI app',
                          ignore_patterns=['models/**', 'results/**', 'instance/**', '.venv/**', '**/__pycache__/**',
                                           '*.pyc', '*.zip', 'tests/**', 'README.md'])
        api.upload_file(path_or_fileobj=str(readme), path_in_repo='README.md', repo_id=a.space, repo_type='space')
    print('[3/3] Space settings')
    api.add_space_variable(a.space, 'PUBLIC_MODE', '1')
    api.add_space_variable(a.space, 'HF_MODEL_REPO', a.models_repo)
    if a.private_models:
        api.add_space_secret(a.space, 'HF_TOKEN', a.token or os.environ.get('HF_TOKEN', ''))
    if a.admin_token:
        api.add_space_secret(a.space, 'ADMIN_TOKEN', a.admin_token)
    user, name = a.space.split('/', 1)
    print(f'Done. The Space is building: https://huggingface.co/spaces/{a.space}\n'
          f'Direct app link once it is running: https://{user.lower()}-{name.lower().replace("_", "-")}.hf.space')


if __name__ == '__main__':
    main()
