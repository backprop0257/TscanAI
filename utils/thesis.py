"""Thesis results dashboard: reads the files the v22-v24 notebook writes into results/ (nothing is recomputed,
nothing is typed in). Every section is optional -- a missing file just leaves its section out.

    research_questions_v23.json              Cell 10.3   RQ1-RQ3, H1-H3 (verdicts, rules, evidence)
    rq1_lab_field_table_v23.csv              Cell 10.3   lab vs field accuracy / macro-F1 / ECE with 95% CI
    hybrid_proof_scorecard_v24.json          Cell 10.3c  "is the hybrid the best model?" scorecard
    per_class_f1_hybrid_vs_best_backbone_v24.csv  Cell 10.3c
    robustness_field_v24.csv                 Cell 9.6    accuracy under corruptions, mCA
    gradcam_faithfulness_v24.csv             Cell 10.2b  deletion / insertion AUC, sanity check
    lfs_by_class_v24.csv                     Cell 10.2b  Leaf-Focus Score per class and model
    xai_v22_summary.csv (or xai_v21_summary.csv)  Cell 10.2  LFS / lesion focus per model and split
    ood_per_source_auroc_v24.json            Cell 9.5b   gate AUROC per non-tomato source
    ood_stress_test_v24.csv                  Cell 9.5b   gate under blur / dark / noise / ...
    gate_v21_config.json, ood_gate_metrics.json   Cells 9.5 / 4.4  held-out gate numbers
    data_validation_v24.json                 Cell 1.6c   duplicates, leakage fix, stratification
    segmentation_qa_v24.json                 Cell 2.5b   background-removal quality
    seed_runs_v24.json                       Cell 7.4    extra-seed runs
    latency_model_size_v21.csv               Cell 10.3   parameters, latency
"""
import csv
import json
import math
import re
from pathlib import Path

HYBRID = 'Hybrid_CNN_ViT'

FILES = {
    'research_questions_v23.json': 'RQ1-RQ3 and H1-H3 (Cell 10.3)',
    'rq1_lab_field_table_v23.csv': 'Lab vs field table (Cell 10.3)',
    'hybrid_proof_scorecard_v24.json': 'Hybrid-proof scorecard (Cell 10.3c)',
    'per_class_f1_hybrid_vs_best_backbone_v24.csv': 'Per-class F1, hybrid vs best backbone (Cell 10.3c)',
    'robustness_field_v24.csv': 'Robustness benchmark (Cell 9.6)',
    'gradcam_faithfulness_v24.csv': 'Grad-CAM faithfulness (Cell 10.2b)',
    'lfs_by_class_v24.csv': 'Leaf-Focus Score per class (Cell 10.2b)',
    'ood_per_source_auroc_v24.json': 'OOD gate per source (Cell 9.5b)',
    'ood_stress_test_v24.csv': 'OOD stress test (Cell 9.5b)',
    'gate_v21_config.json': 'Calibrated tomato-leaf gate (Cell 9.5)',
    'data_validation_v24.json': 'Data validation (Cell 1.6c)',
    'segmentation_qa_v24.json': 'Background-removal QA (Cell 2.5b)',
    'seed_runs_v24.json': 'Extra-seed runs (Cell 7.4)',
    'latency_model_size_v21.csv': 'Model size and latency (Cell 10.3)',
}


def _json(p):
    try:
        return json.load(open(p, encoding='utf-8')) if Path(p).is_file() else None
    except Exception:  # noqa: BLE001
        return None


def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
    s = str(v).strip()
    if re.fullmatch(r'-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?', s):
        f = float(s)
        return None if math.isnan(f) else f
    if s.lower() in ('true', 'false'):
        return s.lower() == 'true'
    return s if s not in ('', 'nan', 'None') else None


def _csv(p, index_name=None, max_rows=500):
    p = Path(p)
    if not p.is_file():
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
            if not k:                                   # unnamed pandas index column
                k = index_name or 'index'
            row[k] = _num(v)
        out.append(row)
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def _ranks(values, direction):
    """values {model: v}; direction +1 higher-is-better / -1 lower-is-better; ties share the best rank."""
    items = {m: v for m, v in values.items() if isinstance(v, (int, float))}
    keyed = sorted(items, key=lambda m: -direction * items[m])
    out = {}
    for m in keyed:
        out[m] = 1 + sum(1 for o in keyed if -direction * items[o] < -direction * items[m] - 1e-12)
    return out


def _research(d):
    if not d:
        return None
    out = {}
    for k in ('H1', 'H2', 'H3'):
        h = d.get(k)
        if isinstance(h, dict):
            out[k] = {'verdict': h.get('verdict'), 'rule': h.get('rule')}
    h1 = d.get('H1') or {}
    a = h1.get('a_field_finetuning') or {}
    if a:
        out['H1']['field'] = [{'model': m, 'field_acc': e.get('field_acc_final'),
                               'before': e.get('field_acc_before_field_finetune'),
                               'useful': e.get('useful'), 'significant_gain': e.get('significant_gain')}
                              for m, e in a.items()]
    b = h1.get('b_temperature_scaling') or {}
    if b:
        out['H1']['ece'] = [{'model': m, 'split': s, 'raw': v.get('ece_raw'), 'calibrated': v.get('ece_calibrated'),
                             'lowered': v.get('lowered')} for m, sp in b.items() for s, v in sp.items()]
    h2 = d.get('H2') or {}
    comp = h2.get('comparisons') or {}
    if comp:
        rows = []
        for split, c in comp.items():
            for rival, v in (c.get('vs') or {}).items():
                ci = v.get('ci95') or [None, None]
                rows.append({'split': split, 'vs': rival, 'diff_pp': (v.get('diff') or 0) * 100,
                             'ci_lo': ci[0] * 100 if ci[0] is not None else None,
                             'ci_hi': ci[1] * 100 if ci[1] is not None else None,
                             'mcnemar_p': v.get('mcnemar_p'), 'hybrid_f1': c.get('hybrid_macro_f1'),
                             'their_f1': v.get('their_macro_f1')})
        out['H2']['comparisons'] = rows
    h3 = d.get('H3') or {}
    if h3:
        out['H3'].update({'gate_auroc': h3.get('gate_auroc'), 'gate_fpr_at_95tpr': h3.get('gate_fpr_at_95tpr'),
                          'hybrid_lfs': (h3.get('hybrid_lfs') or {}).get('mean'),
                          'hybrid_lfs_ci': (h3.get('hybrid_lfs') or {}).get('ci95'),
                          'lfs_by_model': {m: (v or {}).get('mean') for m, v in (h3.get('lfs_by_model') or {}).items()},
                          'checks': h3.get('checks')})
    for k in ('RQ2', 'RQ3'):
        r = d.get(k)
        if isinstance(r, dict):
            out[k] = {'answer': r.get('answer'), 'checks': r.get('checks'), 'note': r.get('note'),
                      'trained_the_same_way': r.get('trained_the_same_way')}
    rq3 = d.get('RQ3') or {}
    if rq3.get('ood_gate'):
        out['RQ3']['ood_gate'] = rq3['ood_gate']
    rq1 = d.get('RQ1') or {}
    if rq1:
        rows = []
        for m, r in rq1.items():
            row = {'model': m}
            for split, key in (('PV-Test', 'pv'), ('Field-Test', 'field')):
                s = r.get(split) or {}
                for met in ('accuracy', 'macro_f1', 'ece_raw', 'ece_calibrated'):
                    v = s.get(met) or {}
                    row[f'{key}_{met}'] = v.get('value')
                    ci = v.get('ci95') or [None, None]
                    row[f'{key}_{met}_lo'], row[f'{key}_{met}_hi'] = ci[0], ci[1]
            gap = r.get('lab_to_field_gap_pp') or {}
            row['gap_pp'] = gap.get('value')
            row['gap_lo'], row['gap_hi'] = (gap.get('ci95') or [None, None])[:2]
            row['temperature'] = r.get('temperature')
            rows.append(row)
        out['RQ1'] = {'rows': rows}
    out['settings'] = d.get('settings')
    return _clean(out)


def _scorecard(d):
    if not d:
        return None
    crit = []
    models = []
    for name, c in (d.get('criteria') or {}).items():
        vals = {m: v for m, v in (c.get('values') or {}).items() if isinstance(v, (int, float))}
        direction = 1 if 'higher' in str(c.get('direction', 'higher')) else -1
        crit.append({'name': name, 'direction': direction, 'values': vals, 'ranks': _ranks(vals, direction)})
        for m in vals:
            if m not in models:
                models.append(m)
    mean_rank = d.get('mean_rank') or {}
    models.sort(key=lambda m: (mean_rank.get(m, 99), m))
    sig = [{'vs': b, **v} for b, v in (d.get('hybrid_vs_backbones_field_macro_f1') or {}).items()]
    seeds = d.get('seeds') or {}
    seed_rows = [{'model': m, 'per_seed': v.get('field_macro_f1_per_seed'), 'mean': v.get('mean'), 'std': v.get('std')}
                 for m, v in seeds.items() if isinstance(v, dict) and 'mean' in v]
    return _clean({'verdict': d.get('verdict'), 'best_mean_rank': d.get('best_mean_rank'),
                   'first_on_primary': d.get('first_on_primary'), 'mean_rank': mean_rank,
                   'first_places': d.get('first_places'), 'criteria': crit, 'models': models,
                   'significance': sig, 'seeds': seed_rows,
                   'seed_ttest': seeds.get('paired_t_test_field_macro_f1'), 'cost': d.get('cost')})


def _data_validation(d, moves_csv):
    if not d:
        return None
    audit = d.get('audit') or {}
    strat = {}
    for ds, s in (d.get('stratification') or {}).items():
        table = s.get('table') or {}                   # {split: {class: n}}
        classes = sorted({c for col in table.values() for c in col})
        strat[ds] = {'split_ratio': s.get('split_ratio'), 'chi2_p_value': s.get('chi2_p_value'),
                     'max_class_share_deviation_pp': s.get('max_class_share_deviation_pp'),
                     'min_per_class_in_eval': s.get('min_per_class_in_eval'),
                     'classes': classes,
                     'counts': {sp: [table.get(sp, {}).get(c, 0) for c in classes] for sp in ('train', 'val', 'test')
                                if sp in table}}
    moves = _csv(moves_csv) or []
    why = {}
    for m in moves:
        why[m.get('why') or '?'] = why.get(m.get('why') or '?', 0) + 1
    return _clean({'pairs_by_reason': audit.get('pairs_by_reason'), 'groups_with_copies': audit.get('groups_with_copies'),
                   'cross_split_groups': audit.get('cross_split_groups'), 'label_conflicts': audit.get('label_conflicts'),
                   'dedup_removed': audit.get('dedup_removed'), 'n_moved': d.get('n_moved'),
                   'moves_by_reason': why, 'fix_applied': d.get('fix_applied'),
                   'hamming_threshold': d.get('hamming_threshold'), 'stratification': strat})


def signature(results_dir):
    rd = Path(results_dir)
    sig = []
    for f in FILES:
        p = rd / f
        if p.is_file():
            st = p.stat()
            sig.append((f, int(st.st_mtime), st.st_size))
    return tuple(sig)


_CACHE = {'sig': None, 'data': None}


def load(results_dir):
    """Everything the Results tab shows, as plain JSON. Cached until a results file changes."""
    rd = Path(results_dir)
    sig = signature(rd)
    if _CACHE['sig'] == sig and _CACHE['data'] is not None:
        return _CACHE['data']
    gate = _json(rd / 'gate_v21_config.json') or {}
    ood_cnn = _json(rd / 'ood_gate_metrics.json') or {}
    xai = _csv(rd / 'xai_v22_summary.csv') or _csv(rd / 'xai_v21_summary.csv')
    data = {
        'files': [{'file': f, 'what': w, 'present': (rd / f).is_file()} for f, w in FILES.items()],
        'research': _research(_json(rd / 'research_questions_v23.json')),
        'rq1_table': _csv(rd / 'rq1_lab_field_table_v23.csv'),
        'scorecard': _scorecard(_json(rd / 'hybrid_proof_scorecard_v24.json')),
        'per_class': _csv(rd / 'per_class_f1_hybrid_vs_best_backbone_v24.csv'),
        'robustness': _csv(rd / 'robustness_field_v24.csv'),
        'faithfulness': _csv(rd / 'gradcam_faithfulness_v24.csv', index_name='model'),
        'lfs_by_class': _csv(rd / 'lfs_by_class_v24.csv', index_name='cls'),
        'xai_summary': xai,
        'ood': _clean({'per_source': _json(rd / 'ood_per_source_auroc_v24.json'),
                       'stress': _csv(rd / 'ood_stress_test_v24.csv'),
                       'heldout': gate.get('heldout'), 'feature_model': gate.get('feature_model'),
                       'accept_threshold': gate.get('accept_threshold'),
                       'component_auroc': gate.get('component_auroc'),
                       'ood_cnn': {k: ood_cnn.get(k) for k in ('auroc', 'auprc', 'fpr_at_95tpr', 'threshold')
                                   if k in ood_cnn} or None}),
        'data_validation': _data_validation(_json(rd / 'data_validation_v24.json'), rd / 'data_validation_moves_v24.csv'),
        'segmentation': _json(rd / 'segmentation_qa_v24.json'),
        'seed_runs': _json(rd / 'seed_runs_v24.json'),
        'latency': _csv(rd / 'latency_model_size_v21.csv'),
        'plots': sorted(p.name for p in (rd / 'plots').glob('*_v24.png')) if (rd / 'plots').is_dir() else [],
        'hybrid': HYBRID,
    }
    data['available'] = any(x['present'] for x in data['files'])
    data = _clean(data)
    _CACHE.update(sig=sig, data=data)
    return data
