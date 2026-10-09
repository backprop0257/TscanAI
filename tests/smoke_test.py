"""Smoke test (no browser):  python tests/smoke_test.py [path/to/leaf.jpg]
Loads the models, runs one diagnosis, one compare and one non-leaf image through the job API."""
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np            # noqa: E402
from PIL import Image         # noqa: E402
import app as A               # noqa: E402


def jpeg(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, 'JPEG'); return b.getvalue()


def run(c, kind, data):
    j = c.post(f'/api/jobs/{kind}', data={'leaf_image': (io.BytesIO(data), 'x.jpg')},
               content_type='multipart/form-data').json
    if 'job_id' not in j:
        raise SystemExit(j)
    while True:
        s = c.get(f"/api/jobs/{j['job_id']}").json
        if s['state'] != 'running':
            return s
        time.sleep(0.3)


def main():
    c = A.app.test_client()
    t = time.time()
    while A.MODELS.loading:
        time.sleep(1)
    st = c.get('/api/status').json
    print(f'models loaded in {time.time() - t:.0f}s: {st["models_loaded"]}')
    for e in st['load_errors'] + st['notices']:
        print('  note:', e)
    if not st['models_loaded']:
        raise SystemExit('No models loaded -- check models/ (see README).')
    leaf = open(sys.argv[1], 'rb').read() if len(sys.argv) > 1 else None
    if leaf:
        for kind in ('predict', 'compare'):
            s = run(c, kind, leaf); r = s.get('result') or {}
            print(kind, s['state'], s.get('error') or '', '| accepted:', r.get('accepted'), '|',
                  r.get('display_name'), f"{(r.get('confidence') or 0):.1%}", '|', r.get('message', '')[:80])
    noise = (np.random.default_rng(0).random((300, 300, 3)) * 255).astype(np.uint8)
    s = run(c, 'predict', jpeg(noise)); r = s['result']
    print('random noise -> accepted:', r['accepted'], '|', r['message'][:80])
    print('history entries:', c.get('/api/history').json['total'])
    b = c.post('/api/jobs/batch', data={'leaf_images': [(io.BytesIO(jpeg(noise)), f'n{i}.jpg') for i in range(2)]},
               content_type='multipart/form-data').json
    while True:
        s = c.get(f"/api/jobs/{b['job_id']}").json
        if s['state'] != 'running':
            break
        time.sleep(0.3)
    print('batch', s['state'], (s.get('result') or {}).get('counts'))
    pdf = c.get(f"/api/batch/{b['batch_id']}/report.pdf")
    print('batch PDF', pdf.status_code, len(pdf.data), 'bytes | CSV', c.get('/api/history/export.csv').status_code)
    th = c.get('/api/thesis').json
    print('Results tab files found:', sum(f['present'] for f in th['files']), '/', len(th['files']))
    print('SMOKE TEST OK')


if __name__ == '__main__':
    main()
