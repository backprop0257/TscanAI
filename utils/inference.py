"""TomatoLeafAI v21 inference for the Flask app -- the SAME pipeline the notebook evaluated.

photo -> photo-quality check -> background removal V3 (Cell 2.1b, crop to leaf)
      -> tomato-leaf gate V21 (Cell 2.2b) with the deep tomato-identity stage
         calibrated in Cell 9.5 (OOD-gate CNN + Relative-Mahalanobis + energy + MSP)
      -> classifier(s) on the background-removed crop (exactly the input of Cell 9.2)
      -> temperature-calibrated confidence (Cell 8.2, results/calibration_summary.json)
      -> Grad-CAM on the native 14x14 lesion map + Leaf-Focus / Lesion-Focus scores (Cell 10.1b)
      -> disease regions + affected-area estimate -> treatment advice (Cell 11.1)

The code in utils/v21/ is copied unchanged from the notebook cells named in each file header.
Nothing about the models is hard-coded: every trained classifier found in models/ is loaded
(utils/registry.py) and wrapped in an adapter (utils/adapters.py) -- v21 trunk/head models keep the
notebook's exact Grad-CAM / embedding path, any other Keras/.h5/SavedModel classifier gets its input
size, preprocessing, outputs and Grad-CAM layer detected automatically. Models are loaded in a
background thread (mixed-precision ones rebuilt in float32 for CPU speed, same weights), and the
models/ and results/ folders are watched, so adding a model needs no code change and no restart.
"""
import base64
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'v21'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from seg_v21 import BackgroundAwarePreprocessorV3          # noqa: E402  (Cell 2.1b)
from gate_v21 import TomatoLeafGateV21                      # noqa: E402  (Cell 2.2b)
from ood_v21 import FeatureOODV21                           # noqa: E402  (Cell 9.5)
from . import treatment as T                                # noqa: E402  (Cell 11.1)
from . import metrics                                       # noqa: E402
from . import registry as REG                               # noqa: E402
from .adapters import V21Adapter, KerasAdapter, SavedModelAdapter   # noqa: E402

APP_ROOT = HERE.parent
MODELS_DIR = Path(os.environ.get('MODELS_DIR', APP_ROOT / 'models'))
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', APP_ROOT / 'results'))
CONF_THRESH = float(os.environ.get('CONF_THRESHOLD', '0.60'))   # below this: "Not sure"
OOD_THRESH = 0.62            # gate default; the calibrated value comes from results/gate_v21_config.json
TEMPERATURE = 1.0            # default when a model has no fitted temperature
CPU_FLOAT32 = os.environ.get('CPU_FLOAT32', '1') != '0'
AUTO_RESCAN = float(os.environ.get('AUTO_RESCAN', '5'))   # seconds between folder checks; 0 = off
CUSTOM_CODE_DIRS = [APP_ROOT / 'custom_layers']           # *.py here is imported before loading (custom layers)

DEFAULT_CLASSES = sorted([
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'])

GATE_REJECT_STAGE = {None: 'quality', 'no_leaf': 'leaf', 'skin': 'leaf', 'not_leaf_structure': 'leaf',
                     'featureless_object': 'leaf', 'geometric_shape': 'leaf', 'not_tomato': 'ood',
                     'low_confidence': 'ood'}


def tomato_name(cls):
    """'Tomato___Spider_mites Two-spotted_spider_mite' -> 'Spider mites (two-spotted)'."""
    s = str(cls).replace('Tomato___', '').replace('_', ' ').replace('  ', ' ').strip()
    s = s.replace('Spider mites Two-spotted spider mite', 'Spider mites (two-spotted)')
    return s[:1].upper() + s[1:]


_DB_KEYS = {}


def canon_class(cls):
    """Map a model's class name onto the treatment database's name, whatever naming the model was
    trained with: 'Early_blight', 'early blight', 'Tomato___Early_blight' -> 'Tomato___Early_blight'.
    Returns None when the class is not one of the known tomato classes."""
    if not _DB_KEYS:
        for k in T.TREATMENT_DATABASE:
            _DB_KEYS[REG.norm(k).replace('tomato', '')] = k
    n = REG.norm(cls).replace('tomato', '')
    if not n:
        return None
    if n in _DB_KEYS:
        return _DB_KEYS[n]
    for key, full in _DB_KEYS.items():
        if len(n) >= 5 and (key.startswith(n) or n.startswith(key)):
            return full
    return None


def _letterbox(img_bgr, size=(224, 224), pad=(230, 230, 230)):
    h, w = img_bgr.shape[:2]
    s = min(size[0] / w, size[1] / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.full((size[1], size[0], 3), pad, np.uint8)
    y0, x0 = (size[1] - nh) // 2, (size[0] - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def prepared_from_seg(img_bgr, seg, size=(224, 224)):
    """Exactly the notebook's classifier input (Cell 2.3 prepared_from_seg): the background-removed
    crop when background removal is reliable, otherwise a clean letterboxed resize of the photo."""
    if seg.get('is_reliable'):
        return _letterbox(seg['background_removed'], size)
    return _letterbox(img_bgr, size)


def _data_url(rgb, q=85, max_side=None):
    rgb = np.asarray(rgb, np.uint8)
    if max_side and max(rgb.shape[:2]) > max_side:
        s = max_side / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()


def _overlay(rgb, heat, alpha=0.45):
    """Standard Grad-CAM display: jet colour map blended over the whole image (honest view)."""
    col = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return np.clip((1 - alpha) * rgb + alpha * col, 0, 255).astype(np.uint8)


def _overlay_focus(rgb, heat, max_alpha=0.70):
    """Only strong values are tinted, so the leaf stays readable."""
    h = np.clip(heat, 0, 1).astype(np.float32)
    col = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1].astype(np.float32)
    a = (max_alpha * np.clip((h - 0.15) / 0.85, 0, 1))[..., None]
    return np.clip((1 - a) * rgb + a * col, 0, 255).astype(np.uint8)


def _upsample(m, hw):
    m = np.asarray(m, np.float32)
    mx = float(m.max())
    m = m / mx if mx > 0 else m
    return np.clip(cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR), 0, 1)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


class ModelBundle:
    """Finds, loads and runs every trained model in models/ through the v21 pipeline."""

    def __init__(self, models_dir=MODELS_DIR, results_dir=RESULTS_DIR, background=True, watch=True):
        self.models_dir, self.results_dir = Path(models_dir), Path(results_dir)
        self.lock = threading.RLock()                  # held while a diagnosis runs / models are swapped
        self._load_lock = threading.Lock()
        self.models = {}                               # name -> adapter (only successfully loaded models)
        self.model_classes, self.model_info, self.model_params = {}, {}, {}
        self.model_errors, self._loaded_sig = {}, {}
        self.registry, self.user_info = {}, {}
        self.load_errors, self.notices = [], []
        self.loading, self.load_progress = False, {'done': 0, 'total': 0, 'current': None}
        self.version = 0                               # bumps whenever the set of models / results changes
        self.load_seconds = None
        self.ood_model = None
        self.feature_ood = None
        self._aux_sig = None
        self.prep = BackgroundAwarePreprocessorV3(img_size=(224, 224))
        self.class_names = self._read_class_names()
        self.advice = self._advice_table()
        self.refresh_results()
        self.refresh_registry()
        self._rescan_again = False
        self.loading = True
        if background:
            threading.Thread(target=self._load_all, daemon=True, name='model-loader').start()
        else:
            self._load_all()
        if watch and background and AUTO_RESCAN > 0:
            threading.Thread(target=self._watch, daemon=True, name='model-watcher').start()

    # ------------------------------------------------------------------ files
    @staticmethod
    def _read_json(p):
        try:
            return json.load(open(p, encoding='utf-8')) if Path(p).exists() else None
        except Exception:  # noqa: BLE001
            return None

    def _read_class_names(self):
        names = REG.parse_class_names(self._read_json(self.models_dir / 'class_indices.json'))
        return names or list(DEFAULT_CLASSES)

    def _read_temperatures(self):
        cal = self._read_json(self.results_dir / 'calibration_summary.json') or {}
        out = {}
        for k, v in cal.items():
            try:
                out[k] = float(v['temperature'] if isinstance(v, dict) else v)
            except (KeyError, TypeError, ValueError):
                pass
        return out

    def _advice_table(self):
        out = {}
        for cn in self.class_names:
            key = canon_class(cn) or cn
            e = T.TREATMENT_DATABASE.get(key)
            if not e:
                continue
            first = (e.get('cultural_practices') or e.get('prevention_advice') or [''])[0]
            out[cn] = {'display_name': tomato_name(key), 'advice': f"{e.get('pathogen_type', '').capitalize()}. {first}".strip('. ') + '.',
                       **{k: v for k, v in e.items() if k != 'severity_note'}}
        return out

    def _tta_setting(self):
        """TTA env: '1' on, '0' off, 'auto' (default) = on when the notebook evaluated/calibrated with TTA
        (v23 writes "tta": true per model into results/eval_summary_v22.json)."""
        v = os.environ.get('TTA', 'auto').strip().lower()
        if v in ('1', 'true', 'on'):
            return True
        if v in ('0', 'false', 'off'):
            return False
        ev = self._read_json(self.results_dir / 'eval_summary_v22.json') or {}
        return any(isinstance(r, dict) and r.get('tta') for r in ev.values())

    def refresh_results(self):
        """(Re)reads the notebook's results/ folder: gate config, temperatures, evaluation files."""
        metrics.configure(self.results_dir)
        self.gate_cfg = self._read_json(self.results_dir / 'gate_v21_config.json') or {}
        self.gate_model_name = self.gate_cfg.get('feature_model')
        self.gate = TomatoLeafGateV21(self.prep, accept_threshold=self.gate_cfg.get('accept_threshold', OOD_THRESH),
                                      bands=self.gate_cfg.get('bands'),
                                      deep_thresholds=self.gate_cfg.get('deep_thresholds'),
                                      deep_weight=self.gate_cfg.get('deep_weight', 0.60))
        self._temps_raw = self._read_temperatures()
        self._results_raw = metrics.load_model_results()
        self.evaluation = metrics.load_evaluation()
        self.tta = self._tta_setting()
        self._results_sig = metrics.signature()
        self._index_results()

    def refresh_registry(self):
        self.registry = REG.discover(self.models_dir)
        self.user_info = REG.load_info(self.models_dir, APP_ROOT)
        self._models_sig = REG.signature(self.models_dir)
        self._index_results()

    def _match(self, raw):
        """Re-key result/temperature dicts by the model names in models/ (loose name matching)."""
        known = {REG.norm(n): n for n in list(self.registry) + list(self.models)}
        return {known.get(REG.norm(k), k): v for k, v in raw.items()}

    def _index_results(self):
        if not hasattr(self, '_temps_raw') or not hasattr(self, 'registry'):
            return
        self.temperatures = self._match(self._temps_raw)
        self.results = self._match(self._results_raw)
        if self.gate_model_name:
            self.gate_model_name = self._match({self.gate_model_name: 1}).popitem()[0]
        self.version += 1

    def info_for(self, name):
        info = dict(self.user_info.get(REG.norm(name)) or {})
        if name in self.registry:
            info.update(REG.sidecar(self.registry[name], self.models_dir))
        return info

    @property
    def effective_ood_threshold(self):
        return float(self.gate.accept_threshold)

    @property
    def ready(self):
        return bool(self.models)

    def ordered_names(self, loaded_only=True):
        names = list(self.models) if loaded_only else list(dict.fromkeys(list(self.registry) + list(self.models)))
        metas = [REG.build_meta(n, self.registry.get(n), self.info_for(n), self.gate_model_name) for n in names]
        return [m['name'] for m in sorted(metas, key=REG.sort_key) if not (m['hidden'] and loaded_only)]

    def label(self, name):
        return REG.build_meta(name, self.registry.get(name), self.info_for(name), self.gate_model_name)['label']

    @property
    def default_model_name(self):
        """DEFAULT_MODEL env > "default": true in model_info.json > the proposed model > the most
        accurate loaded model > the first loaded one."""
        loaded = self.ordered_names()
        env = os.environ.get('DEFAULT_MODEL')
        if env:
            m = {REG.norm(n): n for n in loaded}.get(REG.norm(env))
            if m:
                return m
        for n in loaded:
            if self.info_for(n).get('default'):
                return n
        prop = [n for n in loaded if REG.build_meta(n, self.registry.get(n), self.info_for(n),
                                                    self.gate_model_name)['kind'] == 'proposed']
        if self.gate_model_name in prop:
            return self.gate_model_name
        pool = prop or loaded
        scored = [n for n in pool if ((self.results.get(n) or {}).get('pv') or {}).get('accuracy') is not None]
        if scored:
            return max(scored, key=lambda n: self.results[n]['pv']['accuracy'])
        return pool[0] if pool else (env or '')

    # ---------------------------------------------------------------- loading
    @staticmethod
    def _to_float32(keras, model, js):
        js = js.replace('"mixed_float16"', '"float32"').replace('"dtype": "float16"', '"dtype": "float32"')
        m32 = keras.models.model_from_json(js)
        m32.set_weights([np.asarray(w, np.float32) for w in model.get_weights()])
        return m32

    def _import_custom_code(self):
        """Imports every *.py in custom_layers/ (and models/custom_layers/) so models with custom
        layers registered via @keras.saving.register_keras_serializable can be loaded."""
        import importlib.util
        for d in CUSTOM_CODE_DIRS + [self.models_dir / 'custom_layers']:
            for f in sorted(Path(d).glob('*.py')) if Path(d).is_dir() else []:
                key = f'custom_{f.stem}_{abs(hash(str(f)))}'
                if key in sys.modules:
                    continue
                try:
                    spec = importlib.util.spec_from_file_location(key, f)
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[key] = mod
                    spec.loader.exec_module(mod)
                except Exception as e:  # noqa: BLE001
                    self.notices.append(f'custom_layers/{f.name} could not be imported ({e})')

    def _classes_for(self, name, entry, n_out, info):
        names = REG.parse_class_names(info.get('class_names')) or REG.class_file(entry, self.models_dir)
        if names and len(names) == n_out:
            return names, 'model'
        if len(self.class_names) == n_out:
            return list(self.class_names), 'global'
        if len(DEFAULT_CLASSES) == n_out:
            return list(DEFAULT_CLASSES), 'default'
        raise ValueError(f'the model has {n_out} outputs but class_indices.json lists {len(self.class_names)} '
                         f'classes; add models/{name}_classes.json with its {n_out} class names')

    def _load_one(self, keras, name, entry):
        info = self.info_for(name)
        pre = info.get('preprocess') or 'auto'
        if entry['format'] == 'savedmodel':
            ad = SavedModelAdapter(name, entry['path'], pre)
            params = int(sum(np.prod(v.shape) for v in getattr(ad.model, 'variables', [])))
        else:
            try:
                m = keras.models.load_model(str(entry['path']), compile=False)
            except Exception as e:  # noqa: BLE001  (Lambda layers need safe_mode=False -- your own file)
                if 'safe_mode' not in str(e):
                    raise
                m = keras.models.load_model(str(entry['path']), compile=False, safe_mode=False)
            if CPU_FLOAT32:
                try:
                    js = m.to_json()
                    if 'mixed_float16' in js or '"float16"' in js:
                        m = self._to_float32(keras, m, js)
                except Exception as e:  # noqa: BLE001
                    self.notices.append(f'{name}: kept the mixed-precision model ({e.__class__.__name__}).')
            params = int(m.count_params())
            parts = V21Adapter.find_parts(m)
            if parts and pre in ('auto', '0-1'):
                ad = V21Adapter(name, m, *parts)
            else:
                ad = KerasAdapter(name, m, pre)
        out = ad.run(np.zeros((224, 224, 3), np.uint8))          # warm-up / trace + output check
        n_out = int(out['logits'].shape[-1])
        ad.num_classes = n_out
        classes, src = self._classes_for(name, entry, n_out, info)
        return ad, classes, src, params

    def _load_all(self):
        """Loads every model found in models/ that is not loaded yet (or whose file changed) and
        unloads models whose file was removed. Safe to call again at any time (rescan)."""
        if not self._load_lock.acquire(blocking=False):
            self._rescan_again = True
            return
        t0 = time.time()
        try:
            self.loading = True
            try:
                import tensorflow as tf
                import keras
                import model_v21             # noqa: F401  registers the v21 custom layers (Cell 5.1b)
                tf.get_logger().setLevel('ERROR')
            except Exception as e:  # noqa: BLE001
                self.load_errors = [f'TensorFlow/Keras could not be imported: {e}']
                return
            self._import_custom_code()
            self.refresh_registry()
            only = {REG.norm(s) for s in os.environ.get('LOAD_MODELS', '').split(',') if s.strip()}
            wanted = {n: e for n, e in self.registry.items() if not only or REG.norm(n) in only}
            sig = {n: (str(e['path']), e['mtime'], e['bytes']) for n, e in wanted.items()}
            todo = [n for n in wanted if self._loaded_sig.get(n) != sig[n]]
            gone = [n for n in list(self.models) + list(self.model_errors) if n not in wanted]
            todo.sort(key=lambda n: (n != self.gate_model_name, wanted[n]['bytes']))   # gate model first, small next
            self.load_progress = {'done': 0, 'total': len(todo), 'current': None}
            with self.lock:
                for n in gone:
                    for d in (self.models, self.model_classes, self.model_info, self.model_params,
                              self.model_errors, self._loaded_sig):
                        d.pop(n, None)
            for n in todo:
                self.load_progress['current'] = n
                try:
                    ad, classes, src, params = self._load_one(keras, n, wanted[n])
                    with self.lock:
                        self.models[n] = ad
                        self.model_classes[n] = classes
                        self.model_params[n] = params
                        self.model_info[n] = {**ad.info(), 'classes_source': src, 'summary': ad.describe()}
                        self.model_errors.pop(n, None)
                except Exception as e:  # noqa: BLE001
                    with self.lock:
                        self.models.pop(n, None)
                        self.model_errors[n] = f'{e.__class__.__name__}: {str(e)[:240]}'
                self._loaded_sig[n] = sig[n]
                self.load_progress['done'] += 1
                self.version += 1
            self._load_aux(keras)
            self._index_results()
            self._refresh_messages()
            self.load_seconds = round(time.time() - t0, 1)
        finally:
            self.load_progress['current'] = None
            self.loading = False
            self.version += 1
            self._load_lock.release()
        if self._rescan_again:
            self._rescan_again = False
            self._load_all()

    def _load_aux(self, keras):
        """ood_gate.keras (Cell 4.3) and feature_ood_v21.json (Cell 9.5) -- reloaded when they change."""
        files = [self.models_dir / 'ood_gate.keras', self.models_dir / 'feature_ood_v21.json']
        sig = tuple((f.name, f.stat().st_mtime) if f.exists() else (f.name, None) for f in files)
        if sig == self._aux_sig:
            return
        self._aux_sig = sig
        self.ood_model, self.feature_ood, self._aux_errors = None, None, []
        if files[0].exists():
            try:
                self.ood_model = keras.models.load_model(str(files[0]), compile=False)
            except Exception as e:  # noqa: BLE001
                self._aux_errors.append(f'ood_gate.keras could not be loaded ({e})')
        if files[1].exists():
            try:
                self.feature_ood = FeatureOODV21.from_state(json.load(open(files[1])))
            except Exception as e:  # noqa: BLE001
                self._aux_errors.append(f'feature_ood_v21.json could not be read ({e})')

    def _refresh_messages(self):
        errs = [f'{self.label(n)}: could not be loaded ({e})' for n, e in self.model_errors.items()]
        errs += getattr(self, '_aux_errors', [])
        if not self.registry:
            errs.append(f'No trained models found in {self.models_dir} (any .keras / .h5 classifier or '
                        'SavedModel folder; see README).')
        notes = [n for n in self.notices if n.startswith('custom_layers/') or 'mixed-precision' in n]
        if not self.gate_cfg:
            notes.append('results/gate_v21_config.json not found — the tomato-leaf gate uses its heuristic '
                         'stages only (copy the notebook results folder for the calibrated gate).')
        if self.gate_model_name and self.gate_model_name not in self.models and self.gate.deep_thresholds:
            notes.append(f'Gate feature model {self.gate_model_name} is not loaded — deep tomato-identity '
                         'check limited to the OOD-gate CNN.')
        for n in self.ordered_names():
            if n not in self.temperatures:
                notes.append(f'{self.label(n)}: no fitted temperature in calibration_summary.json (T = 1).')
            if self.model_info.get(n, {}).get('classes_source') == 'default':
                notes.append(f'{self.label(n)}: no class_indices.json — using the 10 PlantVillage tomato classes.')
        self.load_errors, self.notices = errs, list(dict.fromkeys(notes))

    def rescan(self):
        """Look for new / changed / removed model files and reload results (non-blocking)."""
        self.refresh_results()
        threading.Thread(target=self._load_all, daemon=True, name='model-rescan').start()

    def _watch(self):
        """Auto-rescan: a file must be unchanged for one interval (fully copied) before it is loaded."""
        last_m, last_r = None, None
        while True:
            time.sleep(AUTO_RESCAN)
            try:
                ms, rs = REG.signature(self.models_dir), metrics.signature()
                if rs != self._results_sig and rs == last_r:
                    self.refresh_results()
                    self._refresh_messages()
                if ms != self._models_sig and ms == last_m and not self.loading:
                    self._load_all()
                last_m, last_r = ms, rs
            except Exception:  # noqa: BLE001  (never let the watcher die)
                pass

    def status(self):
        return {'loading': self.loading, 'progress': dict(self.load_progress), 'ready': self.ready,
                'models_loaded': self.ordered_names(), 'load_errors': list(self.load_errors),
                'notices': list(self.notices), 'gate_calibrated': bool(self.gate.deep_thresholds),
                'gate_threshold': self.effective_ood_threshold, 'version': self.version, 'tta': self.tta,
                'models_found': len(self.registry), 'auto_rescan': AUTO_RESCAN}

    def models_meta(self):
        """Every model the app knows about -- found in models/, loaded, or only in results/ -- with its
        live state, architecture details and evaluation numbers, in display order."""
        names = list(dict.fromkeys(list(self.registry) + list(self.models) + list(self.results)))
        cur = self.load_progress.get('current')
        out = []
        for n in names:
            entry = self.registry.get(n)
            meta = REG.build_meta(n, entry, self.info_for(n), self.gate_model_name)
            if meta['hidden']:
                continue
            res = self.results.get(n) or {}
            pv, fd = res.get('pv') or {}, res.get('field') or {}
            ai = self.model_info.get(n, {})
            state = ('loaded' if n in self.models else 'error' if n in self.model_errors
                     else 'loading' if n == cur else 'queued' if entry and self.loading
                     else 'results' if not entry else 'queued')
            if not meta['desc']:
                meta['desc'] = (f"{ai['summary']}." if ai.get('summary') else
                                f"{entry['file']} ({entry['size_mb']} MB)." if entry else
                                'Evaluation results only — the model file is not in models/.')
            meta.update({
                'loaded': n in self.models, 'on_disk': bool(entry), 'state': state,
                'error': self.model_errors.get(n), 'params': self.model_params.get(n),
                'temperature': self.temperatures.get(n),
                'accuracy': pv.get('accuracy'), 'f1': pv.get('f1_macro'), 'ece': pv.get('ece'),
                'field_accuracy': fd.get('accuracy'),
                'pv_split': res.get('pv_split'), 'field_split': res.get('field_split'),
                'arch': ai.get('arch'), 'input': ai.get('input'), 'num_classes': ai.get('num_classes'),
                'cam_method': ai.get('cam_method'), 'cam_layer': ai.get('cam_layer'), 'cam_hw': ai.get('cam_hw'),
                'preprocess': ai.get('preprocess'), 'classes_source': ai.get('classes_source'),
                'summary': ai.get('summary'),
                'has_results': bool(res), 'is_gate_model': n == self.gate_model_name,
                'recommended': False, 'top': False})
            out.append(meta)
        out.sort(key=REG.sort_key)
        dflt = self.default_model_name
        for m in out:
            m['recommended'] = m['name'] == dflt
        scored = [m for m in out if m['loaded'] and m['accuracy'] is not None]
        if scored:
            max(scored, key=lambda m: m['accuracy'])['top'] = True
        return _jsonable(out)

    # ------------------------------------------------------------------ model
    def _run(self, name, rgb):
        """logits (+ embedding / lesion gate for v21 models) and Grad-CAM / Grad-CAM++ maps."""
        out = self.models[name].run(rgb)
        if self.tta:
            # notebook v23 Cell 8.0: mean probability over original + h-flip + v-flip (the temperature was
            # fitted on exactly these averaged probabilities); Grad-CAM stays from the original view
            def sm(z):
                e = np.exp(z - z.max()); return e / e.sum()
            ps = [sm(out['logits'])]
            for flipped in (rgb[:, ::-1], rgb[::-1]):
                ps.append(sm(self.models[name].run(np.ascontiguousarray(flipped))['logits']))
            out['logits_single'] = out['logits']
            out['logits'] = np.log(np.mean(ps, 0) + 1e-12)
        return out

    def _calibrated(self, name, logits):
        t = self.temperatures.get(name, TEMPERATURE)
        z = logits / max(t, 1e-3)
        p = np.exp(z - z.max())
        return p / p.sum(), t, name in self.temperatures

    # ------------------------------------------------------------------- gate
    def _deep_scores(self, img_bgr, gate_out):
        ds = {}
        if self.feature_ood is not None and gate_out is not None and 'embedding' in gate_out:
            _lg = gate_out.get('logits_single', gate_out['logits'])          # feature OOD was fitted on single-pass logits
            pv = self.feature_ood.pvalues(gate_out['embedding'][None], _lg[None])
            ds['maha_score'] = float(pv['maha_score'][0])
            ds['energy_score'] = float(pv['energy_score'][0])
        if self.ood_model is not None:
            h, w = self.ood_model.input_shape[1:3]
            x = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (w, h)).astype(np.float32)[None] / 255.0
            z = np.asarray(self.ood_model(x, training=False), np.float64)[0]
            e = np.exp(z - z.max())
            ds['ood_gate_prob'] = float(e[1] / e.sum())          # classes ['Other', 'Tomato'] (Cell 4.3)
        if gate_out is not None:
            t = float(self.gate_cfg.get('temperature') or self.temperatures.get(self.gate_model_name, 1.0))
            z = gate_out.get('logits_single', gate_out['logits']) / max(t, 1e-3)
            p = np.exp(z - z.max())
            ds['msp'] = float((p / p.sum()).max())
        keep = set((self.gate.deep_thresholds or {}).keys())
        return {k: v for k, v in ds.items() if k in keep}, ds

    # -------------------------------------------------------------- explain
    def _classify(self, name, out, seg, rgb, les_mask):
        p, t, calibrated = self._calibrated(name, out['logits'])
        order = np.argsort(-p)
        k = int(order[0])
        classes = self.model_classes.get(name) or self.class_names
        raw = classes[k]
        cls = canon_class(raw) or raw              # same name as the treatment database when it is a known class
        conf = float(p[k])
        healthy = 'healthy' in cls.lower()
        ad = self.models[name]
        leaf = seg['leaf_mask']
        cam = _upsample(out['gradcam'], rgb.shape[:2])
        from xai_v21 import focus_metrics, disease_regions        # Cell 10.1b (raw CAM metrics)
        met = focus_metrics(cam, leaf, None if healthy else les_mask)
        leaf_cam = cam * (leaf > 0)
        guided = leaf_cam * (0.25 + 0.75 * cv2.GaussianBlur(les_mask.astype(np.float32), (0, 0), 2))
        guided = guided / (guided.max() + 1e-8)
        regs = [] if healthy else disease_regions(cam, les_mask)
        boxed = _overlay_focus(rgb, guided)
        for r in regs[:3]:
            x, y, w, h = r['box']
            cv2.rectangle(boxed, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 255, 255), 2)
        if not healthy:
            cnts, _ = cv2.findContours(les_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(boxed, cnts, -1, (255, 0, 60), 1)
        imgs = {'gradcam': _data_url(_overlay(rgb, cam)),
                'gradcam_pp': _data_url(_overlay(rgb, _upsample(out['gradcam++'], rgb.shape[:2]))),
                'disease_regions': _data_url(boxed)}
        if out.get('lesion_gate') is not None:
            g = np.asarray(out['lesion_gate'], np.float32)
            g = (g - g.min()) / (g.max() - g.min() + 1e-8)
            imgs['lesion_attention'] = _data_url(_overlay_focus(rgb, _upsample(g, rgb.shape[:2])))
        top = [canon_class(classes[i]) or classes[i] for i in order[:5]]
        return {'model': name, 'label': self.label(name),
                'class': cls, 'raw_class': raw, 'known_class': cls in T.TREATMENT_DATABASE,
                'display_name': tomato_name(cls), 'confidence': conf,
                'calibrated': calibrated, 'temperature': t, 'confident': conf >= CONF_THRESH,
                'healthy': healthy, 'arch': ad.arch, 'cam_method': ad.cam_method, 'cam_layer': ad.cam_layer,
                'top5': [{'class': c, 'display_name': tomato_name(c), 'prob': float(p[i])}
                         for c, i in zip(top, order[:5])],
                'explain': {'lfs': met.get('lfs'), 'lefs': met.get('lefs'),
                            'pointing_leaf': met.get('pointing_leaf'), 'pointing_lesion': met.get('pointing_lesion'),
                            'top10_in_leaf': met.get('top10_in_leaf'),
                            'lesion_enrichment': met.get('lesion_enrichment'),
                            'regions': [{'box': list(r['box']), 'share': float(r['cam_share'])} for r in regs[:5]],
                            'images': imgs}}

    # ------------------------------------------------------------- pipeline
    @staticmethod
    def decode(image_bytes, max_side=1600):
        im = Image.open(io.BytesIO(image_bytes))
        im = ImageOps.exif_transpose(im).convert('RGB')
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    def _analyze(self, image_bytes, names, compare, progress_cb):
        cb = progress_cb or (lambda *a, **k: None)
        t0 = time.time()
        if not self.models:
            raise RuntimeError('Models are still loading.' if self.loading else 'No models are loaded.')
        cb('decode', 4, 'Reading the photo')
        img = self.decode(image_bytes)
        orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cb('segment', 12, 'Removing the background (soil, stones, shadows) — disease spots are kept')
        seg = self.prep.process(img)
        prep_bgr = prepared_from_seg(img, seg)            # notebook Cell 2.3: the classifier input
        rgb = cv2.cvtColor(prep_bgr, cv2.COLOR_BGR2RGB)
        preview = {'original': _data_url(orig_rgb, max_side=640),
                   'background_removed': _data_url(rgb),
                   'segmentation': _data_url(cv2.cvtColor(seg['stages']['overlay'], cv2.COLOR_BGR2RGB))}
        segm = {k: seg.get(k) for k in ('leaf_coverage', 'segmentation_confidence', 'is_reliable',
                                         'lesion_fraction', 'background_composition', 'rejection_reason')}
        res = {'mode': 'compare' if compare else 'predict', 'accepted': False, 'preview': preview,
               'segmentation': _jsonable(segm), 'pipeline': 'v21'}
        cb('segment', 20, 'Background removed', {'preview': preview})

        cb('leafcheck', 26, 'Checking photo quality and leaf shape, edges, veins and colour')
        g = self.gate.evaluate(img, seg=seg)
        if g['decision'] == 'REJECT' and g['veto'] != 'low_confidence':
            return self._finish(res, g, None, t0, cb)

        gname = self.gate_model_name if self.gate_model_name in self.models else None
        outs = {}
        cb('leafcheck', 34, 'Checking that it is a TOMATO leaf (deep features)')
        if gname:
            outs[gname] = self._run(gname, rgb)
        gate_in = prep_bgr if self.gate_cfg.get('ood_gate_input') == 'background_removed' else img
        deep, deep_all = self._deep_scores(gate_in, outs.get(gname))
        g = self.gate.evaluate(img, deep_scores=deep, seg=seg)
        if g['decision'] == 'REJECT':
            return self._finish(res, g, deep_all, t0, cb)
        res['gate'] = self._gate_summary(g, deep_all)
        cb('leafcheck', 40, 'Tomato leaf accepted', {'gate': res['gate']})

        from xai_v21 import lesion_proxy_mask                      # Cell 10.1b
        les = lesion_proxy_mask(rgb, seg['leaf_mask'], seg.get('lesion_mask'))
        per = {}
        for i, n in enumerate(names):
            cb('predict', 42 + int(46 * i / max(1, len(names))),
               f'{self.label(n)}: prediction + Grad-CAM')
            ts = time.time()
            out = outs.get(n) or self._run(n, rgb)
            r = self._classify(n, out, seg, rgb, les)
            r['time_ms'] = int((time.time() - ts) * 1000)
            per[n] = r
            cb('predict', 42 + int(46 * (i + 1) / max(1, len(names))), f'{r["label"]}: {r["display_name"]}',
               {'models': {n: r}} if compare else {'prediction': r})

        cb('advice', 92, 'Preparing treatment advice')
        main = per[names[0]]
        if compare:
            votes = {}                         # majority vote; ties broken by summed confidence
            for r in per.values():
                c, sc = votes.get(r['class'], (0, 0.0))
                votes[r['class']] = (c + 1, sc + r['confidence'])
            best = max(votes, key=votes.get)
            agree = sum(r['class'] == best for r in per.values())
            cands = [r for r in per.values() if r['class'] == best]
            main = next((r for r in cands if r['model'] == self.default_model_name),
                        max(cands, key=lambda r: r['confidence']))
            res['consensus'] = {'class': best, 'display_name': tomato_name(best), 'agree': agree,
                                'total': len(per), 'model': main['model']}
            res['models'] = per
        leaf_px = max(1, int((seg['leaf_mask'] > 0).sum()))
        area = None if main['healthy'] else float((les & (seg['leaf_mask'] > 0)).sum() / leaf_px)
        try:
            adv = T.get_treatment_recommendation(main['class'], float(np.clip(main['confidence'], 0, 1)),
                                                 calibrated=bool(main['calibrated']))
        except KeyError:                       # a class the treatment database does not cover
            adv = None
        res.update({'accepted': True, 'stage': 'done', 'model': main['model'], 'model_label': main['label'],
                    'prediction': main, 'display_name': main['display_name'], 'class': main['class'],
                    'confidence': main['confidence'], 'confident': main['confident'],
                    'status': 'ok' if main['confident'] else 'unsure',
                    'affected_area_estimate': area, 'treatment': _jsonable(adv),
                    'advice': (T.TREATMENT_DATABASE.get(main['class']) and
                               self._advice_text(main['class'])) or '',
                    'message': ('Diagnosis ready.' if main['confident'] else
                                f'Low confidence ({main["confidence"]:.0%}). The leaf may be unclear or the disease '
                                'unusual — treat this as a hint and confirm with an agronomist, or retake the photo.')})
        res['timing_ms'] = int((time.time() - t0) * 1000)
        cb('done', 100, 'Done', {'advice': res['treatment']} if res['treatment'] else None)
        return _jsonable(res)

    @staticmethod
    def _gate_summary(g, deep_all):
        return _jsonable({'decision': g['decision'], 'veto': g['veto'], 'confidence': g['confidence'],
                          'heuristic_score': g['heuristic_score'], 'deep_score': g['deep_score'],
                          'deep_scores': deep_all or {}, 'cue_scores': g.get('cue_scores') or {},
                          'reasons': g['reasons']})

    def _finish(self, res, g, deep_all, t0, cb):
        stage = GATE_REJECT_STAGE.get(g['veto'], 'ood')
        res.update({'accepted': False, 'stage': stage, 'status': 'ood', 'message': g['message'],
                    'reasons': g['reasons'], 'gate': self._gate_summary(g, deep_all),
                    'display_name': {'quality': 'Photo problem', 'leaf': 'No tomato leaf found',
                                     'ood': 'Not a tomato leaf'}.get(stage, 'Not diagnosed'),
                    'advice': g['message']})
        res['timing_ms'] = int((time.time() - t0) * 1000)
        cb('done', 100, 'Not accepted', {'gate': res['gate']})
        return _jsonable(res)

    def _advice_text(self, cls):
        e = T.TREATMENT_DATABASE.get(cls) or {}
        first = (e.get('cultural_practices') or e.get('prevention_advice') or [''])[0]
        return f"{e.get('pathogen_type', '').capitalize()}. {first}".strip('. ') + '.'

    def predict(self, image_bytes, model_name=None, progress_cb=None):
        with self.lock:
            name = model_name if model_name in self.models else self.default_model_name
            return self._analyze(image_bytes, [name], False, progress_cb)

    def compare(self, image_bytes, progress_cb=None, names=None):
        with self.lock:
            names = [n for n in self.ordered_names() if not names or n in names]
            return self._analyze(image_bytes, names, True, progress_cb)
