"""Server-side diagnosis history: one JSON file per diagnosis in instance/history/.

Each entry stores the FULL result (images included) so the History tab can replay the
whole pipeline -- background removal, leaf check, prediction, Grad-CAM, advice.

Online (PUBLIC_MODE) every entry belongs to the visitor who made it (an anonymous id kept in a
cookie): visitors only ever see, export or delete their own diagnoses. Locally (single user)
the owner is ignored and everything is shown, as before.
"""
import json
import os
import threading
import time
import uuid
from pathlib import Path

PIPELINE_VERSION = 'v21'           # the inference pipeline (unchanged since v21); older entries are "stale"
MAX_ITEMS = int(os.environ.get('HISTORY_MAX_ITEMS', '1000'))
MAX_PER_OWNER = int(os.environ.get('HISTORY_MAX_PER_VISITOR', '200'))


def _summary(item):
    r = item.get('result') or {}
    pred = r.get('prediction') or {}
    if r.get('accepted'):
        status = 'ok' if r.get('confident', pred.get('confident')) else 'unsure'
        title = r.get('display_name') or pred.get('display_name')
    else:
        status = 'ood'
        title = r.get('display_name') or 'Not diagnosed'
    cons = r.get('consensus')
    return {
        'id': item['id'], 'created_at': item['created_at'], 'mode': item['mode'],
        'filename': item.get('filename'), 'pipeline_version': item.get('pipeline_version'),
        'owner': item.get('owner'), 'batch_id': item.get('batch_id'),
        'status': status, 'title': title, 'class': r.get('class'),
        'model': r.get('model_label') or r.get('model'),
        'confidence': r.get('confidence'),
        'agree': f"{cons['agree']}/{cons['total']}" if cons else None,
        'thumb': (r.get('preview') or {}).get('background_removed'),
        'message': r.get('message'),
        'lfs': ((pred.get('explain') or {}).get('lfs')),
        'affected_area': r.get('affected_area_estimate'),
        'timing_ms': r.get('timing_ms'),
    }


class HistoryStore:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get('HISTORY_DIR',
                                                 Path(__file__).resolve().parent.parent / 'instance' / 'history'))
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._index = {}
        for p in self.root.glob('*.json'):
            try:
                item = json.load(open(p, encoding='utf-8'))
                self._index[item['id']] = _summary(item)
            except Exception:  # noqa: BLE001 -- skip a corrupt file instead of failing startup
                continue

    @staticmethod
    def _mine(summary, owner):
        return owner is None or summary.get('owner') == owner

    def add(self, mode, filename, result, owner=None, batch_id=None):
        item = {'id': uuid.uuid4().hex[:12], 'created_at': time.time(), 'mode': mode,
                'filename': filename, 'pipeline_version': PIPELINE_VERSION, 'owner': owner,
                'batch_id': batch_id, 'result': result}
        with self._lock:
            tmp = self.root / f".{item['id']}.tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(item, f)
            os.replace(tmp, self.root / f"{item['id']}.json")
            self._index[item['id']] = _summary(item)
            if owner is not None:                          # cap per visitor (online)
                own = sorted((s for s in self._index.values() if s.get('owner') == owner),
                             key=lambda s: s['created_at'])
                for s in own[:-MAX_PER_OWNER]:
                    self._delete_nolock(s['id'])
            extra = sorted(self._index.values(), key=lambda s: s['created_at'])[:-MAX_ITEMS]
            for s in extra:
                self._delete_nolock(s['id'])
        return item['id']

    def list(self, limit=50, before=None, owner=None, batch_id=None):
        with self._lock:
            items = sorted((s for s in self._index.values() if self._mine(s, owner)), key=lambda s: -s['created_at'])
        if batch_id:
            items = [s for s in items if s.get('batch_id') == batch_id]
        if before is not None:
            items = [s for s in items if s['created_at'] < before]
        return items[:max(1, min(int(limit), 2000))]

    def count(self, owner=None):
        return sum(1 for s in self._index.values() if self._mine(s, owner))

    def count_stale(self, owner=None):
        return sum(1 for s in self._index.values()
                   if self._mine(s, owner) and s.get('pipeline_version') != PIPELINE_VERSION)

    def get(self, item_id, owner=None):
        s = self._index.get(item_id)
        if s is None or not self._mine(s, owner):
            return None
        try:
            return json.load(open(self.root / f'{item_id}.json', encoding='utf-8'))
        except Exception:  # noqa: BLE001
            return None

    def _delete_nolock(self, item_id):
        self._index.pop(item_id, None)
        try:
            (self.root / f'{item_id}.json').unlink()
            return True
        except FileNotFoundError:
            return False

    def delete(self, item_id, owner=None):
        s = self._index.get(item_id)
        if s is None or not self._mine(s, owner):
            return False
        with self._lock:
            return self._delete_nolock(item_id) or True

    def clear(self, owner=None):
        with self._lock:
            for i in [i for i, s in self._index.items() if self._mine(s, owner)]:
                self._delete_nolock(i)

    def clear_stale(self, owner=None):
        with self._lock:
            stale = [i for i, s in self._index.items()
                     if self._mine(s, owner) and s.get('pipeline_version') != PIPELINE_VERSION]
            for i in stale:
                self._delete_nolock(i)
        return len(stale)
