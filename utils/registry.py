"""Dynamic model registry: finds every trained classifier in models/ -- nothing is hard-coded.

Drop a model into models/ and the app picks it up (at start, with "Rescan", or automatically
while running). Supported files:

    <Name>_final.keras / <Name>_best.keras / <Name>.keras    Keras 3 models (v21 or any other)
    <Name>.h5                                                 legacy Keras / tf.keras models
    <Name>/  (folder with saved_model.pb)                     TensorFlow SavedModel (prediction + saliency)

Skipped automatically: ood_gate*.keras (the gate CNN), *_pvonly, stage checkpoints (*_s1, *_s2 ...),
*.weights.h5, anything with 'checkpoint' / 'ckpt' / 'tmp' in the name, and names in EXCLUDE_MODELS.

Optional per-model settings (all optional; the app works without them):
    models/model_info.json     {"<Name>": {"label", "short", "kind", "desc", "order", "default",
                                            "preprocess", "class_names", "hidden"}}
    models/<file stem>.json    the same keys for one model, e.g. models/MyNet_final.json
    models/<file stem>_classes.json / <Name>_class_indices.json   class names for a model trained
                                on a different class list than models/class_indices.json
"""
import json
import os
import re
from pathlib import Path

MODEL_EXTS = ('.keras', '.h5', '.hdf5')
NAME_SUFFIXES = ('_final', '_best', '_model', '_trained')          # stripped to get the display name
_SUFFIX_RANK = {'_final': 0, '_best': 1, '': 2, '_model': 2, '_trained': 2}
_SKIP = re.compile(r'(^ood_gate)|(_pvonly)|(_s\d+$)|(_seed\d+)|(_s\d_(best|last))|(checkpoint)|(ckpt)|(^tmp)|(_tmp)|(\.weights$)|(_weights$)',
                   re.IGNORECASE)
_ABLATION_HINTS = ('nocam', 'no_cam', 'naive', 'branch', 'ablation', 'without', '_wo_', 'nobg', 'noreg')
KINDS = ('proposed', 'baseline', 'ablation', 'custom')
PREPROCESS_MODES = ('0-1', '0-255', '-1-1', 'caffe', 'torch', 'auto')


def norm(name):
    """Loose key for matching a model to its result files: 'VGG16' == 'vgg_16' == 'Vgg16'."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def split_name(stem):
    """'VGG16_final' -> ('VGG16', '_final')."""
    for suf in NAME_SUFFIXES:
        if stem.lower().endswith(suf) and len(stem) > len(suf):
            return stem[: -len(suf)], suf
    return stem, ''


def pretty_label(name):
    """'Hybrid_CNN_ViT_noCAMreg' -> 'Hybrid CNN ViT noCAMreg'; keeps the author's capitalisation."""
    s = re.sub(r'[_\-]+', ' ', str(name)).strip()
    return re.sub(r'\s+', ' ', s) or str(name)


def _read_json(p):
    try:
        return json.load(open(p, encoding='utf-8')) if Path(p).is_file() else None
    except Exception:  # noqa: BLE001
        return None


def _excluded():
    return {norm(s) for s in os.environ.get('EXCLUDE_MODELS', '').split(',') if s.strip()}


def discover(models_dir):
    """{name: {'name', 'path', 'format', 'size_mb', 'mtime', 'file'}} for every classifier found."""
    models_dir = Path(models_dir)
    found = {}
    if not models_dir.is_dir():
        return found
    excl = _excluded()
    cands = []
    for p in sorted(models_dir.iterdir()):
        if p.name.startswith('.'):
            continue
        if p.is_file() and p.suffix.lower() in MODEL_EXTS:
            stem = p.stem
            if p.name.lower().endswith('.weights.h5') or _SKIP.search(stem):
                continue
            name, suf = split_name(stem)
            fmt = 'keras' if p.suffix.lower() == '.keras' else 'h5'
            cands.append((name, suf, p, fmt))
        elif p.is_dir() and (p / 'saved_model.pb').is_file() and not _SKIP.search(p.name):
            name, suf = split_name(p.name)
            cands.append((name, suf, p, 'savedmodel'))
    for name, suf, p, fmt in cands:
        if norm(name) in excl or norm(p.stem) in excl:
            continue
        try:
            st = p.stat()
            size = st.st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
            mtime = st.st_mtime if p.is_file() else max([f.stat().st_mtime for f in p.rglob('*') if f.is_file()] or [0])
        except OSError:
            continue
        entry = {'name': name, 'path': p, 'file': p.name, 'format': fmt, 'suffix': suf,
                 'size_mb': round(size / 2 ** 20, 1), 'mtime': mtime, 'bytes': size}
        old = found.get(name)
        # several files for one model (e.g. X_final.keras and X_best.keras): _final wins, then newest
        if old is None or (_SUFFIX_RANK.get(suf, 2), -mtime) < (_SUFFIX_RANK.get(old['suffix'], 2), -old['mtime']):
            found[name] = entry
    return found


def signature(models_dir):
    """Cheap fingerprint of the models folder (names, sizes, times) for the auto-rescan watcher."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return ()
    out = []
    for p in sorted(models_dir.iterdir()):
        try:
            if p.is_file() and (p.suffix.lower() in MODEL_EXTS or p.suffix.lower() == '.json'):
                st = p.stat()
                out.append((p.name, st.st_size, int(st.st_mtime)))
            elif p.is_dir() and (p / 'saved_model.pb').is_file():
                out.append((p.name, sum(f.stat().st_size for f in p.rglob('*') if f.is_file()), 0))
        except OSError:
            pass
    return tuple(out)


def load_info(models_dir, app_root=None):
    """Optional display settings per model: <app>/model_info.json, then models/model_info.json (wins).
    Keys are matched loosely, so "vgg16" also applies to VGG16_final.keras."""
    out = {}
    for p in ([Path(app_root) / 'model_info.json'] if app_root else []) + [Path(models_dir) / 'model_info.json']:
        d = _read_json(p)
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) and not k.startswith('_'):
                    out.setdefault(norm(k), {}).update(v)
    return out


def sidecar(entry, models_dir):
    """Per-model JSON next to the file (models/<stem>.json or models/<name>.json), if any."""
    models_dir = Path(models_dir)
    p = Path(entry['path'])
    for c in (models_dir / f'{p.stem}.json', models_dir / f"{entry['name']}.json"):
        d = _read_json(c)
        if isinstance(d, dict):
            return d
    return {}


def class_file(entry, models_dir):
    """Class names for one model, if it ships its own list."""
    models_dir = Path(models_dir)
    stem, name = Path(entry['path']).stem, entry['name']
    for c in (f'{stem}_classes.json', f'{name}_classes.json', f'{name}_class_indices.json',
              f'{stem}_class_indices.json'):
        names = parse_class_names(_read_json(models_dir / c))
        if names:
            return names
    if entry['format'] == 'savedmodel':
        names = parse_class_names(_read_json(Path(entry['path']) / 'class_indices.json'))
        if names:
            return names
    return None


def parse_class_names(d):
    """Accepts every common class-list format: notebook Cell 1.4 {'class_names': [...]},
    Keras {'name': index}, {'index': 'name'}, or a plain list."""
    if isinstance(d, list) and d and all(isinstance(x, str) for x in d):
        return list(d)
    if isinstance(d, dict):
        if isinstance(d.get('class_names'), list):
            return list(d['class_names'])
        if isinstance(d.get('idx_to_class'), dict):
            d = {v: int(k) for k, v in d['idx_to_class'].items()}
        if isinstance(d.get('class_to_idx'), dict):
            d = d['class_to_idx']
        if d and all(isinstance(v, int) for v in d.values()):
            return [k for k, _ in sorted(d.items(), key=lambda kv: kv[1])]
        if d and all(str(k).isdigit() for k in d) and all(isinstance(v, str) for v in d.values()):
            return [v for _, v in sorted(d.items(), key=lambda kv: int(kv[0]))]
    return None


def guess_kind(name, gate_model=None):
    n = name.lower()
    if any(h in n for h in _ABLATION_HINTS):
        return 'ablation'
    if (gate_model and norm(name) == norm(gate_model)) or 'hybrid' in n or 'proposed' in n:
        return 'proposed'
    return 'baseline'


def build_meta(name, entry=None, info=None, gate_model=None):
    """Display metadata for one model: user settings first, sensible guesses otherwise."""
    info = dict(info or {})
    kind = str(info.get('kind') or guess_kind(name, gate_model)).lower()
    if kind not in KINDS:
        kind = 'custom'
    label = info.get('label') or pretty_label(name)
    return {
        'name': name,
        'label': label,
        'short': info.get('short') or label,
        'kind': kind,
        'desc': info.get('desc') or '',
        'order': info.get('order'),
        'default': bool(info.get('default')),
        'hidden': bool(info.get('hidden')),
        'file': entry['file'] if entry else None,
        'format': entry['format'] if entry else None,
        'size_mb': entry['size_mb'] if entry else None,
    }


def sort_key(meta):
    order = meta.get('order')
    return (order if isinstance(order, (int, float)) else 1e9,
            KINDS.index(meta['kind']) if meta['kind'] in KINDS else 9, meta['label'].lower())
