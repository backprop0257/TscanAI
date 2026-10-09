"""Copy everything the web app needs from the notebook's project folder.

    python tools/collect_artifacts.py  /path/to/project_root
        (the folder that contains the notebook's models/ and results/ -- LOCAL_ROOT in Cell 1.2)

Copies into ./models : EVERY trained classifier found there (any *.keras / *.h5 / SavedModel folder --
                       found the same way the app finds them, so checkpoints, *_pvonly and stage files
                       are skipped), plus ood_gate.keras, feature_ood_v21.json, class_indices.json,
                       model_info.json and per-model *_classes.json / <model>.json files.
Copies into ./results: *.json, *.csv, eval/, plots/  (no image datasets, no checkpoints)
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import registry as REG          # noqa: E402

AUX = ['ood_gate.keras', 'feature_ood_v21.json', 'class_indices.json', 'model_info.json']


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1]).expanduser()
    app = Path(__file__).resolve().parent.parent
    sm, sr = src / 'models', src / 'results'
    if not sm.exists():
        sm = src                                   # also accept the models folder itself
    (app / 'models').mkdir(exist_ok=True); (app / 'results').mkdir(exist_ok=True)
    n = 0
    found = REG.discover(sm)
    if not found:
        print(f'No trained models found in {sm}.')
    for name, e in found.items():
        p = Path(e['path'])
        if p.is_dir():
            shutil.copytree(p, app / 'models' / p.name, dirs_exist_ok=True)
        else:
            shutil.copy2(p, app / 'models' / p.name)
        n += 1
        print(f'models/ <- {p.name}  ({e["size_mb"]} MB)')
        for extra in sm.glob(f'{p.stem}*.json'):
            shutil.copy2(extra, app / 'models' / extra.name)
    for f in AUX + [x.name for x in sm.glob('*_classes.json')]:
        if (sm / f).exists():
            shutil.copy2(sm / f, app / 'models' / f); n += 1
            print('models/ <-', f)
    if sr.exists():
        for f in list(sr.glob('*.json')) + list(sr.glob('*.csv')):
            shutil.copy2(f, app / 'results' / f.name); n += 1
        for d in ('eval', 'plots'):
            if (sr / d).exists():
                shutil.copytree(sr / d, app / 'results' / d, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('*.npz', '_pred_cache*', '_ood_stress_tmp', '_archive*'))
                print(f'results/{d}/ copied')
    print(f'Done: {n} files (+ eval/ and plots/). Start the app with: python app.py')


if __name__ == '__main__':
    main()
