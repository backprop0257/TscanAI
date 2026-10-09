"""Reads the evaluation files the notebook writes into results/ (nothing is recomputed here).

    results/eval/<Model>/<Model>_PVTest_metrics.json      Cell 9.2 (generate_full_report, Cell 9.1)
    results/eval/<Model>/<Model>_FieldTest_metrics.json   Cell 9.2   (+ their PNG charts)
    results/final_comparison_table.csv                    Cell 9.3   (+ plots/model_comparison.png)
    results/statistical_validation.json                   Cell 9.4
    results/calibration_summary.json / calibration_comparison.csv   Cell 8.2
    results/gate_v21_config.json                          Cell 9.5
    results/xai_v21_summary.csv                           Cell 10.2  (+ plots/xai_v21/*.png)
    results/hypothesis_verdicts_v21.json                  Cell 10.3
    results/latency_model_size_v21.csv                    Cell 10.3
"""
import csv
import json
import re
from pathlib import Path

RESULTS_DIR = Path('results')
EVAL_DIRS = []
_SAFE_PNG = re.compile(r'^[\w\-. ()+,]+\.png$')


def configure(results_dir):
    global RESULTS_DIR, EVAL_DIRS
    RESULTS_DIR = Path(results_dir)
    EVAL_DIRS = [RESULTS_DIR / 'plots', RESULTS_DIR / 'plots' / 'xai_v21', RESULTS_DIR]
    ev = RESULTS_DIR / 'eval'
    if ev.exists():
        EVAL_DIRS += sorted(p for p in ev.iterdir() if p.is_dir())


def _json(p):
    try:
        return json.load(open(p, encoding='utf-8')) if Path(p).exists() else None
    except Exception:  # noqa: BLE001
        return None


def _csv(p, max_rows=200):
    p = Path(p)
    if not p.exists():
        return None
    try:
        with open(p, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))[:max_rows]
    except Exception:  # noqa: BLE001
        return None
    out = []
    for r in rows:
        row = {}
        for k, v in r.items():
            k = (k or '').strip()
            if not k:                      # pandas index column written by DataFrame.to_csv()
                continue
            try:
                row[k] = float(v) if v not in (None, '') and re.match(r'^-?[\d.eE+-]+$', v) else v
            except ValueError:
                row[k] = v
        out.append(row)
    return out


def _report(path):
    d = _json(path)
    if not d:
        return None
    return {'overall': d.get('overall', {}), 'per_class': d.get('per_class', {}),
            'confusion_matrix': (d.get('confusion_matrix') or {}).get('row_normalized'),
            'class_names': d.get('class_names'), 'n_samples': d.get('n_samples'),
            'plots': {k: v for k, v in (d.get('plots') or {}).items() if v}}


def split_label(split):
    """'PVTest' -> 'PV test', 'FieldTest' -> 'Field test', 'val' -> 'Val'."""
    s = re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|_|-', ' ', str(split)).strip()
    s = re.sub(r'\bTest\b', 'test', s)
    return s[:1].upper() + s[1:]


def _split_rank(split):
    s = split.lower()
    if 'pv' in s or 'plantvillage' in s or s in ('test', 'testset'):
        return 0
    if 'field' in s or 'real' in s or 'wild' in s:
        return 1
    return 2


def result_names():
    """Every model that has evaluation files: results/eval/<Model>/<Model>_<Split>_metrics.json."""
    ev = RESULTS_DIR / 'eval'
    if not ev.is_dir():
        return []
    return sorted(p.name for p in ev.iterdir() if p.is_dir() and any(p.glob('*_metrics.json')))


def _model_reports(name):
    """{split: report} for every *_metrics.json of one model -- splits are read from the file names."""
    d = RESULTS_DIR / 'eval' / name
    out = {}
    for f in sorted(d.glob('*_metrics.json')):
        stem = f.name[: -len('_metrics.json')]
        split = stem[len(name) + 1:] if stem.startswith(name + '_') else stem
        rep = _report(f)
        if rep:
            out[split or 'test'] = rep
    return dict(sorted(out.items(), key=lambda kv: (_split_rank(kv[0]), kv[0])))


def load_model_results(model_names=None):
    """{model: {'splits': [...], 'reports': {split: report}, 'pv', 'field', 'gap', 'pv_report', 'field_report'}}
    'pv' = the in-distribution test split, 'field' = the field / real-photo split, when present."""
    out = {}
    for n in (model_names if model_names is not None else result_names()):
        reps = _model_reports(n)
        if not reps:
            continue
        pv_key = next((k for k in reps if _split_rank(k) == 0), next(iter(reps)))
        fd_key = next((k for k in reps if _split_rank(k) == 1), None)
        pv, fd = reps.get(pv_key), reps.get(fd_key) if fd_key else None
        r = {'splits': [{'key': k, 'label': split_label(k)} for k in reps], 'reports': reps,
             'pv': (pv or {}).get('overall'), 'field': (fd or {}).get('overall'),
             'pv_split': pv_key, 'field_split': fd_key, 'pv_report': pv, 'field_report': fd}
        if r['pv'] and r['field'] and r['pv'].get('accuracy') is not None and r['field'].get('accuracy') is not None:
            r['gap'] = r['pv']['accuracy'] - r['field']['accuracy']
        out[n] = r
    return out


def signature():
    """Fingerprint of the results folder (for the auto-refresh watcher)."""
    out = []
    if RESULTS_DIR.is_dir():
        for pat in ('*.json', '*.csv', 'eval/*/*.json'):
            for p in RESULTS_DIR.glob(pat):
                try:
                    out.append((str(p.relative_to(RESULTS_DIR)), int(p.stat().st_mtime), p.stat().st_size))
                except OSError:
                    pass
    return tuple(sorted(out))


def _research_summary():
    """Compact view of results/research_questions_v23.json (notebook v23 Cell 10.3)."""
    d = _json(RESULTS_DIR / 'research_questions_v23.json')
    if not d:
        return None
    out = {}
    for k in ('H1', 'H2', 'H3'):
        if isinstance(d.get(k), dict):
            out[k] = {'verdict': d[k].get('verdict'), 'rule': d[k].get('rule')}
    for k in ('RQ2', 'RQ3'):
        if isinstance(d.get(k), dict):
            out[k] = {'answer': d[k].get('answer'), 'checks': d[k].get('checks')}
    out['settings'] = d.get('settings')
    return out


def load_evaluation():
    gate = _json(RESULTS_DIR / 'gate_v21_config.json') or {}
    plots = []
    for d in (RESULTS_DIR / 'plots',):
        if d.exists():
            plots += sorted(p.name for p in d.glob('*.png'))
    xai_examples = sorted(p.name for p in (RESULTS_DIR / 'plots' / 'xai_v21').glob('*.png'))[:24] \
        if (RESULTS_DIR / 'plots' / 'xai_v21').exists() else []
    return {
        'comparison': _csv(RESULTS_DIR / 'final_comparison_table.csv'),
        'calibration': _json(RESULTS_DIR / 'calibration_summary.json'),
        'calibration_comparison': _csv(RESULTS_DIR / 'calibration_comparison.csv'),
        'statistical': _json(RESULTS_DIR / 'statistical_validation.json'),
        'gate': {k: gate.get(k) for k in ('accept_threshold', 'deep_thresholds', 'component_auroc',
                                          'heldout', 'feature_model', 'temperature') if k in gate} or None,
        'hypotheses': _json(RESULTS_DIR / 'hypothesis_verdicts_v21.json'),
        'research': _research_summary(),
        'rq1_table': _csv(RESULTS_DIR / 'rq1_lab_field_table_v23.csv'),
        'xai': _csv(RESULTS_DIR / 'xai_v21_summary.csv'),
        'latency': _csv(RESULTS_DIR / 'latency_model_size_v21.csv'),
        'plots': plots,
        'xai_examples': xai_examples,
        'results_dir': str(RESULTS_DIR),
    }


def chart_path(filename):
    """Absolute path of an evaluation PNG, or None (only .png names, only inside results/)."""
    if not filename or not _SAFE_PNG.match(filename) or '..' in filename:
        return None
    for d in EVAL_DIRS:
        p = d / filename
        if p.is_file():
            return str(p)
    return None
