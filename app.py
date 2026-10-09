"""
TomatoLeafAI v24 -- web app (Proposal Section 3.11), Dammar Khadayat.

Runs on CPU. Training happens on the GPU box (v24 notebook); this app loads the finished .keras
files and runs EXACTLY the notebook's inference pipeline: background removal V3 -> tomato-leaf gate
(+ deep OOD, calibrated in Cell 9.5) -> classifier (+ TTA when the notebook used it) ->
temperature-calibrated confidence (Cell 8.2) -> Grad-CAM + Leaf-Focus Score (Cell 10.1b) ->
treatment advice (Cell 11.1).

New in v24: Results dashboard (RQ1-RQ3 / H1-H3, hybrid-proof scorecard, robustness, Grad-CAM
faithfulness, OOD stress test, data validation), batch diagnosis, CSV / PDF export, and an
online mode (PUBLIC_MODE=1) for hosting: per-visitor history, rate limits, admin-only model
management, Docker / Hugging Face Spaces ready.

    pip install -r requirements.txt
    python app.py                         # local: http://127.0.0.1:5000
    gunicorn -c gunicorn.conf.py wsgi:app # online (see DEPLOY.md)
"""
import hmac
import os
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque

from flask import (Flask, Response, abort, g, jsonify, render_template, request, send_file, url_for)
from werkzeug.utils import secure_filename

from utils.inference import ModelBundle, CONF_THRESH, OOD_THRESH, TEMPERATURE  # noqa: F401
from utils import registry as REG
from utils.history import HistoryStore, PIPELINE_VERSION
from utils.inference import tomato_name
from utils import metrics
from utils import reports
from utils import thesis


def _env_flag(name, default='0'):
    return os.environ.get(name, default).strip().lower() in ('1', 'true', 'yes', 'on')


# ------------------------------------------------------------------------------------------ settings
PUBLIC_MODE = _env_flag('PUBLIC_MODE')                       # online hosting: per-visitor history, limits, no model upload
ADMIN_TOKEN = os.environ.get('ADMIN_TOKEN', '').strip()      # online: enables model upload / rescan for whoever sends it
BATCH_MAX = int(os.environ.get('BATCH_MAX', '10' if PUBLIC_MODE else '30'))
RATE_LIMIT_PER_MIN = int(os.environ.get('RATE_LIMIT_PER_MIN', '12' if PUBLIC_MODE else '0'))   # 0 = off
MAX_QUEUE = int(os.environ.get('MAX_QUEUE', '6' if PUBLIC_MODE else '0'))                       # 0 = off
FRAME_ANCESTORS = os.environ.get('FRAME_ANCESTORS', "'self' https://huggingface.co https://*.hf.space")
IMAGE_MAX_BYTES = 12 * 1024 * 1024                                  # 12 MB per photo
MODEL_MAX_BYTES = int(float(os.environ.get('MODEL_UPLOAD_MAX_MB', '2048')) * 1024 * 1024)
_VID = re.compile(r'^[0-9a-f]{32}$')


def _port_in_use(port):
    """True if something already accepts connections on 127.0.0.1:port (Windows can share a port
    silently and the browser keeps reaching an OLD server -- refuse to start instead)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0


if __name__ == '__main__' and _port_in_use(int(os.environ.get('PORT', 5000))):
    raise SystemExit(
        f"Port {os.environ.get('PORT', 5000)} is already in use -- probably another TomatoLeafAI server is still "
        f"running. Stop it first (Ctrl+C in its terminal, or Task Manager > python.exe), or use another port: set PORT=5001")

app = Flask(__name__, static_folder='static', template_folder='templates')
# online the largest request is a batch of photos; locally a model upload can be large
app.config['MAX_CONTENT_LENGTH'] = (IMAGE_MAX_BYTES * max(BATCH_MAX, 1) + 1024 * 1024 if PUBLIC_MODE and not ADMIN_TOKEN
                                    else max(MODEL_MAX_BYTES, IMAGE_MAX_BYTES * BATCH_MAX) + 1024 * 1024)
app.json.sort_keys = False  # keep model order (proposed model first) in JSON responses
if _env_flag('TRUST_PROXY', '1' if PUBLIC_MODE else '0'):      # behind Hugging Face / Render / nginx
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

MODELS = ModelBundle()
HISTORY = HistoryStore()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# TensorFlow/Keras models are not guaranteed safe under overlapping inference calls, so jobs queue.
INFERENCE_LOCK = threading.Lock()
JOB_TTL_SECONDS = 900  # forget jobs nobody has polled for a while


class JobManager:
    """Runs a slow inference call on a background thread and exposes its live progress for the
    frontend to poll (progress bar + step list instead of a blocking form POST)."""

    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def _update(self, job_id, **fields):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.update(fields)

    def _merge_partial(self, job_id, data):
        """Shallow merge into job['partial'], one level deeper for dicts (compare() accumulates one
        model at a time into partial['models']; batch appends finished items to partial['items'])."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            partial = job.setdefault('partial', {})
            for key, value in data.items():
                if isinstance(value, dict) and isinstance(partial.get(key), dict):
                    partial[key].update(value)
                elif isinstance(value, list) and isinstance(partial.get(key), list):
                    partial[key] = partial[key] + value
                else:
                    partial[key] = value

    def running(self):
        with self._lock:
            return sum(1 for j in self._jobs.values() if j['state'] == 'running')

    def start(self, target_fn, owner=None):
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {'state': 'running', 'stage': 'queued', 'pct': 0,
                                  'detail': 'Waiting for a free worker' if INFERENCE_LOCK.locked() else 'Starting',
                                  'result': None, 'error': None, 'partial': {}, 'created': time.time(), 'owner': owner}

        def progress_cb(stage, pct, detail=None, data=None):
            self._update(job_id, stage=stage, pct=pct, detail=detail)
            if data:
                self._merge_partial(job_id, data)

        def run():
            try:
                with INFERENCE_LOCK:
                    result = target_fn(progress_cb)
                self._update(job_id, state='done', stage='done', pct=100, result=result)
            except Exception as exc:  # noqa: BLE001 -- surface a friendly message to the poller
                self._update(job_id, state='error', error=str(exc))

        threading.Thread(target=run, daemon=True).start()
        self._gc()
        return job_id

    def get(self, job_id, owner=None):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or (owner is not None and job.get('owner') != owner):
                return None
            out = dict(job)
        out.pop('owner', None)
        return out

    def _gc(self):
        cutoff = time.time() - JOB_TTL_SECONDS
        with self._lock:
            for jid in [j for j, job in self._jobs.items() if job['created'] < cutoff and job['state'] != 'running']:
                del self._jobs[jid]


JOBS = JobManager()


# ------------------------------------------------------------------------------ visitors, limits
@app.before_request
def _identify_visitor():
    """Anonymous visitor id (128-bit random): sent by the page as X-Visitor-Id (kept in localStorage, works
    inside the Hugging Face iframe) with a cookie as fallback. Only used to keep each visitor's history private."""
    vid = (request.headers.get('X-Visitor-Id') or request.cookies.get('tlai_vid') or '').strip().lower()
    g.new_vid = not _VID.match(vid)
    g.vid = secrets.token_hex(16) if g.new_vid else vid


def _owner():
    return g.vid if PUBLIC_MODE else None


_RATE = defaultdict(deque)
_RATE_LOCK = threading.Lock()


def _rate_limited(cost=1):
    """True if this visitor (or IP) started more than RATE_LIMIT_PER_MIN diagnoses in the last minute."""
    if RATE_LIMIT_PER_MIN <= 0:
        return False
    now = time.time()
    keys = [f'v:{g.vid}', f'ip:{request.remote_addr}']
    with _RATE_LOCK:
        for k in keys:
            q = _RATE[k]
            while q and q[0] < now - 60:
                q.popleft()
            if len(q) + cost > RATE_LIMIT_PER_MIN * (3 if k.startswith('ip:') else 1):
                return True
        for k in keys:
            _RATE[k].extend([now] * cost)
    return False


def _busy():
    return MAX_QUEUE > 0 and JOBS.running() >= MAX_QUEUE


def _limit_response(cost=1):
    if _busy():
        return jsonify({'error': 'The server is busy with other diagnoses -- please try again in a minute.'}), 503
    if _rate_limited(cost):
        return jsonify({'error': f'Too many diagnoses in a short time (limit {RATE_LIMIT_PER_MIN} photos per minute). '
                                 'Please wait a minute.'}), 429
    return None


def _is_admin():
    if not PUBLIC_MODE:
        return True
    tok = request.headers.get('X-Admin-Token') or request.form.get('admin_token') or ''
    return bool(ADMIN_TOKEN) and hmac.compare_digest(tok.encode(), ADMIN_TOKEN.encode())


def _admin_denied():
    return jsonify({'error': 'Model management is turned off on this public server.' if not ADMIN_TOKEN
                    else 'Admin token required for this action.'}), 403


# ------------------------------------------------------------------------------------- helpers
def _not_ready_message():
    if MODELS.loading:
        p = MODELS.load_progress
        return f"Models are still loading ({p['done']}/{p['total']}) -- try again in a few seconds."
    return 'No models are loaded. See the setup notes on the page.'


def _allowed(filename):
    return '.' in (filename or '') and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _cfg():
    return {'conf_threshold': CONF_THRESH, 'ood_threshold': MODELS.effective_ood_threshold,
            'temperature': TEMPERATURE, 'gate_calibrated': bool(MODELS.gate.deep_thresholds)}


def _models_meta():
    return MODELS.models_meta()


def _image_too_big(n_files=1):
    return request.content_length is not None and request.content_length > IMAGE_MAX_BYTES * n_files + 64 * 1024


def _wants_json():
    return request.headers.get('Accept', '').startswith('application/json') or request.args.get('format') == 'json' \
        or request.path.startswith('/api/')


def _security_headers(resp):
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('Referrer-Policy', 'same-origin')
    if PUBLIC_MODE:      # allow the Hugging Face Space page to show the app in its iframe, nobody else
        resp.headers.setdefault('Content-Security-Policy', f'frame-ancestors {FRAME_ANCESTORS}')
    else:
        resp.headers.setdefault('X-Frame-Options', 'SAMEORIGIN')
    if getattr(g, 'new_vid', False):
        secure = request.is_secure
        resp.set_cookie('tlai_vid', g.vid, max_age=365 * 24 * 3600, httponly=True, secure=secure,
                        samesite='None' if secure else 'Lax')
    return resp


app.after_request(_security_headers)
app.add_template_filter(tomato_name, 'tomato_name')


def _asset_url(filename):
    """Static URL with the file's modification time as ?v=..., so browsers never run a stale app.js."""
    try:
        version = int(os.path.getmtime(os.path.join(app.static_folder, filename)))
    except OSError:
        version = 0
    return url_for('static', filename=filename, v=version)


@app.context_processor
def _page_context():
    return {'evaluation': MODELS.evaluation, 'temperatures': MODELS.temperatures, 'author': 'Dammar Khadayat',
            'loading': MODELS.loading, 'notices': MODELS.notices, 'pipeline_version': PIPELINE_VERSION,
            'asset_url': _asset_url, 'model_version': MODELS.version, 'public_mode': PUBLIC_MODE,
            'admin_enabled': bool(ADMIN_TOKEN) or not PUBLIC_MODE, 'batch_max': BATCH_MAX}


def _render(status=200, **kw):
    ctx = dict(ready=MODELS.ready, load_errors=MODELS.load_errors if not PUBLIC_MODE else [],
               class_names=MODELS.class_names, results=None, compare_results=None, error=None, cfg=_cfg(),
               models_meta=_models_meta(), default_model=MODELS.default_model_name, model_results=MODELS.results,
               advice=MODELS.advice)
    ctx.update(kw)
    return render_template('index.html', **ctx), status


# ------------------------------------------------------------------------------------- errors
@app.errorhandler(413)
def too_large(_exc):
    message = ('That model file is too large (MODEL_UPLOAD_MAX_MB limit) -- copy it into the models/ folder instead.'
               if request.path.startswith('/api/models/') else
               f'The upload is too large (12 MB per photo, at most {BATCH_MAX} photos). Try smaller or fewer photos.')
    if _wants_json():
        return jsonify({'error': message}), 413
    return _render(413, error=message)


@app.errorhandler(404)
def not_found(_exc):
    if _wants_json():
        return jsonify({'error': 'Not found.'}), 404
    return _render(404)


@app.errorhandler(500)
def server_error(_exc):
    message = 'Something went wrong on the server. Check the server log for details.'
    if _wants_json():
        return jsonify({'error': message}), 500
    return _render(500, error=message)


# ------------------------------------------------------------------------------------- pages
@app.route('/', methods=['GET'])
def index():
    return _render()


@app.route('/eval-charts/<filename>', methods=['GET'])
def eval_chart(filename):
    """A notebook evaluation chart (confusion matrix, ROC, ...) -- only .png files inside results/."""
    path = metrics.chart_path(filename)
    if path is None:
        abort(404)
    return send_file(path, mimetype='image/png', max_age=3600)


def _check_upload(file, model_name=None):
    if not MODELS.ready:
        return _not_ready_message(), 503
    if file is None or file.filename == '':
        return 'Choose an image first.', 400
    if not _allowed(file.filename):
        return 'Unsupported file type -- use PNG, JPG, JPEG, WEBP, or BMP.', 400
    if model_name is not None and model_name not in MODELS.models:
        return f"Model '{model_name}' isn't loaded on this server.", 400
    return None


@app.route('/predict', methods=['POST'])
def predict():
    """No-JavaScript fallback (form POST)."""
    if _image_too_big():
        abort(413)
    file = request.files.get('leaf_image')
    model_name = request.form.get('model_name') or MODELS.default_model_name
    err = _check_upload(file, model_name)
    result, error = None, err[0] if err else None
    if not err:
        lim = _limit_response()
        if lim:
            error = lim[0].json['error']
        else:
            try:
                with INFERENCE_LOCK:
                    result = MODELS.predict(file.read(), model_name=model_name)
                HISTORY.add(mode='predict', filename=file.filename, result=result, owner=_owner())
            except Exception as exc:  # noqa: BLE001
                error = f"Couldn't process that image ({exc})."
    if request.headers.get('Accept', '').startswith('application/json') or request.args.get('format') == 'json':
        return jsonify({'error': error, 'result': result})
    return _render(results=result, error=error, default_model=model_name)


@app.route('/compare', methods=['POST'])
def compare():
    if _image_too_big():
        abort(413)
    file = request.files.get('leaf_image')
    err = _check_upload(file)
    result, error = None, err[0] if err else None
    if not err:
        lim = _limit_response()
        if lim:
            error = lim[0].json['error']
        else:
            try:
                with INFERENCE_LOCK:
                    result = MODELS.compare(file.read())
                HISTORY.add(mode='compare', filename=file.filename, result=result, owner=_owner())
            except Exception as exc:  # noqa: BLE001
                error = f"Couldn't process that image ({exc})."
    if request.headers.get('Accept', '').startswith('application/json') or request.args.get('format') == 'json':
        return jsonify({'error': error, 'result': result})
    return _render(compare_results=result, error=error)


# ------------------------------------------------------------------------------------- jobs
@app.route('/api/jobs/predict', methods=['POST'])
def start_predict_job():
    if _image_too_big():
        abort(413)
    file = request.files.get('leaf_image')
    model_name = request.form.get('model_name') or MODELS.default_model_name
    err = _check_upload(file, model_name)
    if err:
        return jsonify({'error': err[0]}), err[1]
    lim = _limit_response()
    if lim:
        return lim
    image_bytes, filename, owner = file.read(), file.filename, _owner()

    def task(progress_cb):
        result = MODELS.predict(image_bytes, model_name=model_name, progress_cb=progress_cb)
        result['history_id'] = HISTORY.add(mode='predict', filename=filename, result=result, owner=owner)
        return result

    return jsonify({'job_id': JOBS.start(task, owner=owner)})


@app.route('/api/jobs/compare', methods=['POST'])
def start_compare_job():
    if _image_too_big():
        abort(413)
    file = request.files.get('leaf_image')
    err = _check_upload(file)
    if err:
        return jsonify({'error': err[0]}), err[1]
    subset = [n.strip() for n in (request.form.get('models') or '').split(',') if n.strip() in MODELS.models] or None
    lim = _limit_response(cost=2)
    if lim:
        return lim
    image_bytes, filename, owner = file.read(), file.filename, _owner()

    def task(progress_cb):
        result = MODELS.compare(image_bytes, progress_cb=progress_cb, names=subset)
        result['history_id'] = HISTORY.add(mode='compare', filename=filename, result=result, owner=owner)
        return result

    return jsonify({'job_id': JOBS.start(task, owner=owner)})


@app.route('/api/jobs/batch', methods=['POST'])
def start_batch_job():
    """Many photos, one model: each photo runs the full pipeline and is saved to History with a common
    batch_id; finished items stream into the job's partial result so the table fills up live."""
    files = [f for f in request.files.getlist('leaf_images') if f and f.filename]
    if _image_too_big(max(len(files), 1)):
        abort(413)
    model_name = request.form.get('model_name') or MODELS.default_model_name
    if not MODELS.ready:
        return jsonify({'error': _not_ready_message()}), 503
    if not files:
        return jsonify({'error': 'Choose one or more photos first.'}), 400
    if len(files) > BATCH_MAX:
        return jsonify({'error': f'At most {BATCH_MAX} photos per batch on this server.'}), 400
    bad = [f.filename for f in files if not _allowed(f.filename)]
    if bad:
        return jsonify({'error': f"Unsupported file type: {', '.join(bad[:3])} -- use PNG, JPG, WEBP or BMP."}), 400
    if model_name not in MODELS.models:
        return jsonify({'error': f"Model '{model_name}' isn't loaded on this server."}), 400
    lim = _limit_response(cost=len(files))
    if lim:
        return lim
    payload = [(f.filename, f.read()) for f in files]
    owner, batch_id = _owner(), uuid.uuid4().hex[:10]
    n = len(payload)

    def task(progress_cb):
        items, counts = [], {'ok': 0, 'unsure': 0, 'ood': 0, 'error': 0}
        for i, (name, data) in enumerate(payload):
            def cb(stage, pct, detail=None, data=None, _i=i, _name=name):    # inner 0-100 -> overall
                progress_cb('batch', int(100 * (_i + (pct or 0) / 100) / n),
                            f'Photo {_i + 1}/{n} ({_name}): {detail or stage}')
            try:
                r = MODELS.predict(data, model_name=model_name, progress_cb=cb)
                hid = HISTORY.add(mode='predict', filename=name, result=r, owner=owner, batch_id=batch_id)
                s = HISTORY.list(limit=1, owner=owner, batch_id=batch_id)[0]
                s = {**s, 'id': hid, 'index': i}
            except Exception as exc:  # noqa: BLE001 -- one bad photo never stops the batch
                s = {'id': None, 'index': i, 'filename': name, 'status': 'error', 'title': 'Could not process',
                     'message': str(exc)}
            counts[s.get('status', 'error')] = counts.get(s.get('status', 'error'), 0) + 1
            items.append(s)
            progress_cb('batch', int(100 * (i + 1) / n), f'{i + 1}/{n} photos done', {'items': [s]})
        return {'mode': 'batch', 'batch_id': batch_id, 'model': model_name, 'model_label': MODELS.label(model_name),
                'items': items, 'counts': counts}

    return jsonify({'job_id': JOBS.start(task, owner=owner), 'batch_id': batch_id, 'n': n})


@app.route('/api/jobs/<job_id>', methods=['GET'])
def job_status(job_id):
    job = JOBS.get(job_id, owner=_owner())
    if job is None:
        return jsonify({'error': 'Unknown or expired job id.'}), 404
    return jsonify(job)


# ------------------------------------------------------------------------------------- history
@app.route('/api/history', methods=['GET'])
def api_history_list():
    limit = request.args.get('limit', default=50, type=int) or 50
    before = request.args.get('before', type=float)
    own = _owner()
    return jsonify({'items': HISTORY.list(limit=limit, before=before, owner=own,
                                          batch_id=request.args.get('batch_id')),
                    'total': HISTORY.count(own), 'stale': HISTORY.count_stale(own),
                    'pipeline_version': PIPELINE_VERSION})


@app.route('/api/history/export.csv', methods=['GET'])
def api_history_csv():
    items = HISTORY.list(limit=2000, owner=_owner(), batch_id=request.args.get('batch_id'))
    name = f"tomatoleafai_{'batch_' + request.args['batch_id'] if request.args.get('batch_id') else 'history'}.csv"
    return Response(reports.csv_bytes(items), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename="{secure_filename(name)}"'})


@app.route('/api/history/<item_id>', methods=['GET'])
def api_history_detail(item_id):
    item = HISTORY.get(item_id, owner=_owner())
    if item is None:
        return jsonify({'error': "That history entry doesn't exist (it may have been deleted)."}), 404
    item.pop('owner', None)
    return jsonify(item)


@app.route('/api/history/<item_id>/report.pdf', methods=['GET'])
def api_history_pdf(item_id):
    item = HISTORY.get(item_id, owner=_owner())
    if item is None:
        return jsonify({'error': "That history entry doesn't exist (it may have been deleted)."}), 404
    stem = os.path.splitext(secure_filename(item.get('filename') or 'photo'))[0] or 'photo'
    return Response(reports.pdf_single(item), mimetype='application/pdf',
                    headers={'Content-Disposition': f'attachment; filename="TomatoLeafAI_{stem}.pdf"'})


@app.route('/api/batch/<batch_id>/report.pdf', methods=['GET'])
def api_batch_pdf(batch_id):
    summaries = HISTORY.list(limit=2000, owner=_owner(), batch_id=batch_id)
    if not summaries:
        return jsonify({'error': 'No photos found for that batch.'}), 404
    items = [HISTORY.get(s['id'], owner=_owner()) for s in reversed(summaries)]
    pdf = reports.pdf_batch([i for i in items if i], title='TomatoLeafAI batch report',
                            detail_pages=request.args.get('details', '1') != '0')
    return Response(pdf, mimetype='application/pdf',
                    headers={'Content-Disposition': f'attachment; filename="TomatoLeafAI_batch_{secure_filename(batch_id)}.pdf"'})


@app.route('/api/history/<item_id>', methods=['DELETE'])
def api_history_delete(item_id):
    if not HISTORY.delete(item_id, owner=_owner()):
        return jsonify({'error': "That history entry doesn't exist (it may already be deleted)."}), 404
    return jsonify({'deleted': True})


@app.route('/api/history', methods=['DELETE'])
def api_history_clear():
    own = _owner()                       # online: a visitor only ever clears their own diagnoses
    if request.args.get('stale'):
        return jsonify({'cleared': True, 'removed': HISTORY.clear_stale(owner=own)})
    HISTORY.clear(owner=own)
    return jsonify({'cleared': True})


# ------------------------------------------------------------------------------------- models
@app.route('/api/models', methods=['GET'])
def api_models():
    return jsonify({'models': _models_meta(), 'default_model': MODELS.default_model_name, 'status': MODELS.status()})


@app.route('/api/models/rescan', methods=['POST'])
def api_models_rescan():
    if not _is_admin():
        return _admin_denied()
    MODELS.rescan()
    return jsonify({'started': True, 'status': MODELS.status()})


@app.route('/api/models/upload', methods=['POST'])
def api_models_upload():
    """Add a trained model from the browser (.keras / .h5 + optional class-names JSON). Admin only online:
    a model file can contain code, so strangers must never be able to upload one."""
    if not _is_admin():
        return _admin_denied()
    f = request.files.get('model_file')
    if f is None or not f.filename:
        return jsonify({'error': 'Choose a .keras or .h5 model file.'}), 400
    name = secure_filename(f.filename)
    ext = os.path.splitext(name)[1].lower()
    if ext not in REG.MODEL_EXTS or name.lower().endswith('.weights.h5'):
        return jsonify({'error': 'Only full models (.keras, .h5) can be added here -- not weights-only files. '
                                 'For a SavedModel folder, copy it into models/.'}), 400
    custom = (request.form.get('name') or '').strip()
    if custom:
        name = secure_filename(custom) + ext
    stem = os.path.splitext(name)[0]
    if REG._SKIP.search(stem):
        return jsonify({'error': f"'{name}' looks like a checkpoint or the gate model, which the app skips. Rename it."}), 400
    dest = MODELS.models_dir / name
    if dest.exists() and request.form.get('overwrite') != '1':
        return jsonify({'error': f'models/{name} already exists.', 'exists': True}), 409
    MODELS.models_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name('.' + name + '.part')
    f.save(str(tmp))
    os.replace(tmp, dest)
    classes = request.files.get('classes_file')
    if classes is not None and classes.filename:
        try:
            import json as _json
            names = REG.parse_class_names(_json.load(classes.stream))
            if not names:
                raise ValueError('no class names found')
            with open(MODELS.models_dir / f'{stem}_classes.json', 'w', encoding='utf-8') as fh:
                _json.dump(names, fh, indent=1)
        except Exception as exc:  # noqa: BLE001
            return jsonify({'error': f'Model saved, but the class-names file was not usable ({exc}).'}), 400
    info = {k: request.form.get(k) for k in ('label', 'kind', 'preprocess') if request.form.get(k)}
    if info:
        import json as _json
        with open(MODELS.models_dir / f'{stem}.json', 'w', encoding='utf-8') as fh:
            _json.dump(info, fh, indent=1)
    MODELS.rescan()
    return jsonify({'saved': name, 'model': REG.split_name(stem)[0], 'status': MODELS.status()})


@app.route('/api/status', methods=['GET'])
def api_status():
    st = MODELS.status()
    if PUBLIC_MODE:
        st['load_errors'] = []
    st.update(public_mode=PUBLIC_MODE, batch_max=BATCH_MAX, queue=JOBS.running())
    return jsonify(st)


@app.route('/api/predict', methods=['POST'])
def api_predict_sync():
    """Plain synchronous JSON API for scripts: form fields leaf_image, model_name, mode=single|compare."""
    if _image_too_big():
        abort(413)
    file = request.files.get('leaf_image') or request.files.get('image')
    if not MODELS.ready:
        return jsonify({'error': _not_ready_message()}), 503
    if file is None or file.filename == '' or not _allowed(file.filename):
        return jsonify({'error': 'Send a PNG/JPG/WEBP/BMP file as leaf_image.'}), 400
    lim = _limit_response()
    if lim:
        return lim
    data = file.read()
    with INFERENCE_LOCK:
        if request.form.get('mode') == 'compare':
            result = MODELS.compare(data)
        else:
            result = MODELS.predict(data, model_name=request.form.get('model_name') or MODELS.default_model_name)
    return jsonify(result)


@app.route('/api/diseases', methods=['GET'])
def api_diseases():
    from utils import treatment as _T
    return jsonify({'classes': MODELS.class_names, 'advice': MODELS.advice, 'disclaimer': _T.DISCLAIMER})


@app.route('/api/performance', methods=['GET'])
def api_performance():
    return jsonify({'results': MODELS.results, 'evaluation': MODELS.evaluation,
                    'temperatures': MODELS.temperatures, 'target_accuracy': 0.95})


@app.route('/api/thesis', methods=['GET'])
def api_thesis():
    """Everything the Results tab shows (research questions, hypotheses, scorecard, robustness,
    Grad-CAM faithfulness, OOD, data validation), read from results/."""
    data = dict(thesis.load(MODELS.results_dir))
    data['labels'] = {m['name']: m.get('short') or m.get('label') or m['name'] for m in _models_meta()}
    return jsonify(data)


@app.route('/health', methods=['GET'])
def health():
    out = {'status': 'ok' if MODELS.ready else ('loading' if MODELS.loading else 'models_missing'),
           'loading': MODELS.loading, 'progress': MODELS.load_progress,
           'gate_calibrated': bool(MODELS.gate.deep_thresholds), 'feature_ood_loaded': MODELS.feature_ood is not None,
           'calibrated_models': sorted(MODELS.temperatures), 'models_loaded': MODELS.ordered_names(),
           'ood_model_loaded': MODELS.ood_model is not None, 'num_classes': len(MODELS.class_names),
           'public_mode': PUBLIC_MODE, 'queue': JOBS.running()}
    if not PUBLIC_MODE:                # file paths and error details only for the local owner
        out.update(notices=MODELS.notices, models_found=sorted(MODELS.registry),
                   model_errors=MODELS.model_errors, load_errors=MODELS.load_errors)
    return jsonify(out)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host=os.environ.get('HOST', '127.0.0.1'), port=port, debug=False, use_reloader=False, threaded=True)
