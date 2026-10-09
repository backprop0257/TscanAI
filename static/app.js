/* TomatoLeafAI v24 web client -- no framework, no external requests. */
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const CFG = JSON.parse($('#app-config').textContent);
  let MODELS = CFG.models || [];
  let DEFAULT_MODEL = CFG.default_model;
  let STATUS_VERSION = null;
  const KIND_LABEL = { proposed: 'Proposed', baseline: 'Baseline', ablation: 'Ablation', custom: 'Custom' };
  const STATE_NOTE = { loading: ' (loading…)', queued: ' (waiting to load…)', error: ' (failed to load)' };

  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const num = (v) => (v === null || v === undefined || v === '' || Number.isNaN(Number(v))) ? null : Number(v);
  const pct = (v, d = 0) => num(v) === null ? '—' : `${(num(v) * 100).toFixed(d)}%`;
  const fx = (v, d = 3) => num(v) === null ? '—' : num(v).toFixed(d);
  const params = (n) => (!n ? '—' : n >= 1e6 ? `${(n / 1e6).toFixed(2)} M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)} K` : String(n));
  const smart = (v) => { const n = num(v); if (n === null) return '—'; if (Number.isInteger(n)) return String(n); const a = Math.abs(n); return n.toFixed(a >= 100 ? 0 : a >= 10 ? 2 : 4); };
  const modelLabel = (n) => (MODELS.find((m) => m.name === n) || {}).label || n;
  const pretty = (c) => {
    let s = String(c || '').replace('Tomato___', '').replace(/_/g, ' ').replace(/\s+/g, ' ').trim();
    s = s.replace('Spider mites Two-spotted spider mite', 'Spider mites (two-spotted)');
    return s.charAt(0).toUpperCase() + s.slice(1);
  };
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
  };

  /* ------------------------------------------------------------------ toasts */
  function toast(msg, kind = '') {
    const t = document.createElement('div');
    t.className = `toast ${kind}`; t.textContent = msg;
    $('#toast-stack').appendChild(t);
    setTimeout(() => t.remove(), kind === 'error' ? 6000 : 3200);
  }

  /* ------------------------------------------------------------------- theme */
  function applyTheme(t) {
    if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
    const dark = t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    $('#theme-icon-dark').hidden = dark; $('#theme-icon-light').hidden = !dark;
  }
  applyTheme(store.get('tla_theme', null));
  $('#theme-toggle-btn').addEventListener('click', () => {
    const cur = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark'; store.set('tla_theme', next); applyTheme(next);
  });

  /* -------------------------------------------------------------------- tabs */
  const TABS = ['diagnose', 'history', 'performance', 'results', 'diseases', 'about'];
  const loaded = {};
  function showTab(name, focus = false) {
    if (!TABS.includes(name)) name = 'diagnose';
    TABS.forEach((t) => {
      const on = t === name;
      $(`#tab-btn-${t}`).classList.toggle('active', on);
      $(`#tab-btn-${t}`).setAttribute('aria-selected', on);
      $(`#tab-btn-${t}`).tabIndex = on ? 0 : -1;
      $(`#tab-${t}`).classList.toggle('active', on);
    });
    if (focus) $(`#tab-btn-${name}`).focus();
    if (location.hash !== `#${name}`) history.replaceState(null, '', name === 'diagnose' ? location.pathname : `#${name}`);
    if (name === 'history') loadHistory();
    if (name === 'performance' && !loaded.perf) { loaded.perf = true; loadModelsTab(); }
    if (name === 'diseases' && !loaded.dis) { loaded.dis = true; loadDiseaseDisclaimer(); }
    if (name === 'results' && window.TLAI_RESULTS) window.TLAI_RESULTS.load();
  }
  TABS.forEach((t, i) => {
    const b = $(`#tab-btn-${t}`);
    b.addEventListener('click', () => showTab(t));
    b.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault(); showTab(TABS[(i + (e.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length], true);
      }
    });
  });

  /* ----------------------------------------------------------------- dialogs */
  $$('dialog').forEach((d) => {
    d.addEventListener('click', (e) => { if (e.target === d || e.target.closest('[data-close]')) d.close(); });
  });
  document.addEventListener('click', (e) => {
    const img = e.target.closest('img[data-zoom]');
    if (!img) return;
    $('#lightbox-img').src = img.dataset.zoom || img.src;
    $('#lightbox-caption').textContent = img.dataset.cap || img.alt || '';
    $('#lightbox').showModal();
  });

  /* ------------------------------------------------- model loading / status */
  function renderModelSelect() {
    const sel = $('#model-select'); const cur = sel.value || DEFAULT_MODEL;
    const usable = MODELS.filter((m) => m.loaded || m.on_disk);
    if (!usable.length) { sel.innerHTML = '<option value="">No models found</option>'; $('#compare-label').textContent = 'Compare all'; updateModelDesc(); return; }
    sel.innerHTML = usable.map((m) => `<option value="${esc(m.name)}" ${m.loaded ? '' : 'disabled'}>${esc(m.label)}${m.accuracy != null ? ` — ${(m.accuracy * 100).toFixed(1)}% test acc.` : ''}${m.loaded ? '' : (STATE_NOTE[m.state] || ' (not loaded)')}${m.recommended ? ' ★' : ''}</option>`).join('');
    const pick = MODELS.find((m) => m.name === cur && m.loaded) || MODELS.find((m) => m.name === DEFAULT_MODEL && m.loaded) || MODELS.find((m) => m.loaded);
    if (pick) sel.value = pick.name;
    updateModelDesc();
    const n = MODELS.filter((m) => m.loaded).length;
    $('#compare-label').textContent = `Compare (${n || usable.length})`;
  }
  function updateModelDesc() {
    const m = MODELS.find((x) => x.name === $('#model-select').value);
    $('#model-desc').textContent = m ? [m.desc, m.summary && m.summary !== (m.desc || '').replace(/\.$/, '') ? m.summary : '', m.temperature ? `Calibrated (T = ${m.temperature.toFixed(2)}).` : 'Not calibrated (T = 1).'].filter(Boolean).join(' ') : '';
  }
  $('#model-select').addEventListener('change', () => { updateModelDesc(); store.set('tla_model', $('#model-select').value); markFlow(); });

  function applyStatus(st) {
    const loadedN = (st.models_loaded || []).length;
    const pill = $('#status-pill');
    if (st.loading) {
      pill.className = 'status-pill busy';
      pill.textContent = st.progress.total ? `Loading models ${st.progress.done}/${st.progress.total}…` : 'Checking models…';
      $('#loading-banner').hidden = !!loadedN;
      $('#loading-detail').textContent = st.progress.current ? `Loading ${modelLabel(st.progress.current)} (${st.progress.done + 1} of ${st.progress.total})…` : 'Starting…';
      $('#loading-bar').style.width = `${st.progress.total ? (100 * st.progress.done / st.progress.total) : 5}%`;
    } else {
      $('#loading-banner').hidden = true;
      pill.className = `status-pill ${loadedN ? 'ok' : 'warn'}`;
      pill.textContent = loadedN ? `${loadedN} model${loadedN === 1 ? '' : 's'} ready` : 'Setup needed';
      $('#setup-banner').hidden = !!loadedN;
      const msgs = (st.load_errors || []).concat(st.notices || []);
      $('#notice-banner').hidden = !(loadedN && msgs.length);
      if (loadedN && msgs.length) {
        $('#notice-count').textContent = msgs.length;
        $('#notice-list').innerHTML = msgs.map((e) => `<div>· ${esc(e)}</div>`).join('');
      }
      if (!loadedN) $('#setup-errors').innerHTML = (st.load_errors || []).map((e) => `<div>· ${esc(e)}</div>`).join('');
    }
    $('#run-btn').disabled = !loadedN || busy;
  }
  // The model list is live: models added to / removed from models/ (or uploaded on the Models tab)
  // appear here without reloading the page. /api/status is cheap; the full list is fetched on change.
  let polling = false; let lastLoaded = null;
  async function pollStatus() {
    if (polling) return; polling = true;
    try {
      const r = await (await fetch('/api/models', { headers: { Accept: 'application/json' } })).json();
      MODELS = r.models; DEFAULT_MODEL = r.default_model; STATUS_VERSION = r.status.version;
      renderModelSelect(); applyStatus(r.status);
      const n = (r.status.models_loaded || []).length;
      if (!r.status.loading && lastLoaded !== null && n !== lastLoaded) toast(`${n} model${n === 1 ? '' : 's'} ready`);
      if (!r.status.loading) lastLoaded = n;
      if (loaded.perf) loadModelsTab(true);
      polling = false;
      if (r.status.loading) setTimeout(pollStatus, 1500);
    } catch { polling = false; setTimeout(pollStatus, 3000); }
  }
  async function watchStatus() {
    try {
      const st = await (await fetch('/api/status', { headers: { Accept: 'application/json' } })).json();
      if (st.version !== STATUS_VERSION || st.loading) await pollStatus();
    } catch { /* server restarting */ }
    setTimeout(watchStatus, document.hidden ? 15000 : 5000);
  }

  /* ------------------------------------------------------------ photo input */
  let photo = null; let busy = false;
  const fileInput = $('#leaf_image');
  function markFlow() {
    $('#flow-photo').classList.toggle('done', !!photo);
    $('#flow-model').classList.toggle('done', !!photo && !!$('#model-select').value);
  }
  async function shrink(file) {
    const max = 1280;
    let bmp;
    try { bmp = await createImageBitmap(file, { imageOrientation: 'from-image' }); } catch { return file; }
    const s = Math.min(1, max / Math.max(bmp.width, bmp.height));
    if (s === 1 && file.size < 3.5e6 && /jpe?g|png|webp/i.test(file.type)) { bmp.close?.(); return file; }
    const c = document.createElement('canvas');
    c.width = Math.round(bmp.width * s); c.height = Math.round(bmp.height * s);
    c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height); bmp.close?.();
    const blob = await new Promise((res) => c.toBlob(res, 'image/jpeg', 0.92));
    return new File([blob], (file.name || 'photo').replace(/\.\w+$/, '') + '.jpg', { type: 'image/jpeg' });
  }
  async function setPhoto(file) {
    if (!file) return;
    if (!/^image\//.test(file.type) && !/\.(jpe?g|png|webp|bmp)$/i.test(file.name || '')) { toast('Please choose an image file.', 'error'); return; }
    photo = await shrink(file);
    const url = URL.createObjectURL(photo);
    $('#upload-preview').src = url; $('#upload-preview').hidden = false;
    $('#dropzone-empty').hidden = true; $('#upload-change').hidden = false; $('#clear-photo-btn').hidden = false;
    const img = new Image();
    img.onload = () => { $('#upload-info').hidden = false; $('#upload-info').textContent = `${esc(file.name || 'photo')} · ${img.naturalWidth}×${img.naturalHeight}px · ${(photo.size / 1024).toFixed(0)} KB`; };
    img.src = url;
    markFlow();
  }
  fileInput.addEventListener('change', () => setPhoto(fileInput.files[0]));
  $('#clear-photo-btn').addEventListener('click', () => {
    photo = null; fileInput.value = '';
    $('#upload-preview').hidden = true; $('#dropzone-empty').hidden = false; $('#upload-change').hidden = true;
    $('#clear-photo-btn').hidden = true; $('#upload-info').hidden = true; markFlow();
  });
  const dz = $('#dropzone');
  ['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, () => dz.classList.remove('drag')));
  dz.addEventListener('drop', (e) => { e.preventDefault(); setPhoto(e.dataTransfer.files[0]); });
  document.addEventListener('paste', (e) => {
    const f = Array.from(e.clipboardData?.files || []).find((x) => x.type.startsWith('image/'));
    if (f) { showTab('diagnose'); setPhoto(f); }
  });

  // camera: live preview when supported, else the phone's own camera picker
  let stream = null;
  $('#camera-btn').addEventListener('click', async () => {
    if (!navigator.mediaDevices?.getUserMedia) { $('#camera-input').click(); return; }
    $('#camera-error').hidden = true; $('#camera-modal').showModal();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1920 } } });
      $('#camera-video').srcObject = stream;
    } catch (err) {
      $('#camera-error').hidden = false; $('#camera-error').textContent = `Camera not available (${err.name}). Use “Choose a leaf photo” instead.`;
    }
  });
  $('#camera-modal').addEventListener('close', () => { stream?.getTracks().forEach((t) => t.stop()); stream = null; });
  $('#camera-capture-btn').addEventListener('click', () => {
    const v = $('#camera-video'); if (!v.videoWidth) return;
    const c = $('#camera-canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    c.toBlob((b) => { setPhoto(new File([b], 'camera.jpg', { type: 'image/jpeg' })); $('#camera-modal').close(); }, 'image/jpeg', 0.92);
  });
  $('#camera-input').addEventListener('change', (e) => setPhoto(e.target.files[0]));

  // mode
  const mode = () => ($('input[name="run_mode"]:checked') || {}).value || 'single';
  function applyMode() {
    const m = mode(); const batch = m === 'batch';
    $('#model-field').hidden = m === 'compare';
    $('#dropzone').hidden = batch; $('.source-row').hidden = batch;
    fileInput.required = !batch;            // the hidden single-photo input must not block a batch submit
    $('#upload-info').hidden = batch || !photo;
    $('#batch-zone').hidden = !batch;
    $('#run-btn-label').textContent = m === 'compare' ? 'Compare all models' : batch ? (batchFiles.length ? `Diagnose ${batchFiles.length} photo${batchFiles.length > 1 ? 's' : ''}` : 'Diagnose photos') : 'Diagnose';
    store.set('tla_mode', m);
  }
  $$('input[name="run_mode"]').forEach((r) => r.addEventListener('change', applyMode));

  /* ------------------------------------------------------------ batch input */
  let batchFiles = [];
  const BATCH_MAX = CFG.batch_max || 10;
  async function addBatch(files) {
    const imgs = Array.from(files || []).filter((f) => /^image\//.test(f.type) || /\.(jpe?g|png|webp|bmp)$/i.test(f.name || ''));
    if (!imgs.length) { toast('Please choose image files.', 'error'); return; }
    const room = BATCH_MAX - batchFiles.length;
    if (imgs.length > room) toast(`At most ${BATCH_MAX} photos per batch — ${imgs.length - Math.max(room, 0)} not added.`, 'error');
    for (const f of imgs.slice(0, Math.max(room, 0))) { const s = await shrink(f); s.origName = f.name || s.name; batchFiles.push(s); }
    renderBatchThumbs(); applyMode();
  }
  function renderBatchThumbs() {
    const box = $('#batch-thumbs');
    box.hidden = !batchFiles.length; $('#batch-clear-btn').hidden = !batchFiles.length; $('#batch-info').hidden = !batchFiles.length;
    box.innerHTML = batchFiles.map((f, i) => `<figure class="bthumb"><img src="${URL.createObjectURL(f)}" alt=""><button type="button" class="bx" data-bdel="${i}" aria-label="Remove ${esc(f.origName)}">✕</button><figcaption>${esc(f.origName)}</figcaption></figure>`).join('');
    $('#batch-info').textContent = `${batchFiles.length} / ${BATCH_MAX} photos · ${(batchFiles.reduce((a, f) => a + f.size, 0) / 1048576).toFixed(1)} MB`;
    $('#batch-empty').querySelector('.dropzone-title').textContent = batchFiles.length ? 'Add more photos' : 'Choose several leaf photos';
  }
  $('#batch-input').addEventListener('change', (e) => { addBatch(e.target.files); e.target.value = ''; });
  $('#batch-thumbs').addEventListener('click', (e) => { const b = e.target.closest('[data-bdel]'); if (b) { batchFiles.splice(Number(b.dataset.bdel), 1); renderBatchThumbs(); applyMode(); } });
  $('#batch-clear-btn').addEventListener('click', () => { batchFiles = []; renderBatchThumbs(); applyMode(); });
  const bdz = $('#batch-drop');
  ['dragenter', 'dragover'].forEach((ev) => bdz.addEventListener(ev, (e) => { e.preventDefault(); bdz.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((ev) => bdz.addEventListener(ev, () => bdz.classList.remove('drag')));
  bdz.addEventListener('drop', (e) => { e.preventDefault(); addBatch(e.dataTransfer.files); });

  /* ---------------------------------------------------------------- running */
  const STAGE_ORDER = ['segment', 'leafcheck', 'predict', 'advice'];
  function progress(job) {
    $('#progress-panel').hidden = false;
    const p = Math.max(0, Math.min(100, job.pct || 0));
    $('#progress-bar').style.width = `${p}%`; $('#progress-pct').textContent = `${p}%`;
    $('#progress-detail').textContent = job.detail || '';
    $('#progress-title').textContent = job.state === 'done' ? 'Done' : 'Working…';
    const si = STAGE_ORDER.indexOf(job.stage === 'decode' ? 'segment' : job.stage);
    $$('#progress-steps li').forEach((li, i) => {
      li.classList.toggle('done', job.state === 'done' || (si > -1 && i < si));
      li.classList.toggle('active', job.state !== 'done' && i === si);
    });
  }
  function renderPartial(part) {
    if (!part) return;
    const bits = [];
    if (part.preview) bits.push(`<h3 class="section-title">Background removal</h3><div class="imgs">${fig(part.preview.original, 'Your photo')}${fig(part.preview.background_removed, 'Leaf after background removal')}${fig(part.preview.segmentation, 'Segmentation (green = leaf, red = lesions kept)')}</div>`);
    if (part.gate) bits.push(`<p class="muted small">Tomato-leaf check: <b>${esc(part.gate.decision)}</b> (confidence ${pct(part.gate.confidence)})</p>`);
    if (part.models) bits.push(`<p class="muted small">Finished: ${Object.values(part.models).map((m) => `${esc(m.label)} → ${esc(m.display_name)} (${pct(m.confidence)})`).join(' · ')}</p>`);
    if (part.prediction) bits.push(`<p class="muted small">${esc(part.prediction.label)} → <b>${esc(part.prediction.display_name)}</b> (${pct(part.prediction.confidence)})</p>`);
    $('#results-panel').innerHTML = bits.join('');
  }
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  $('#upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (busy) return;
    if (mode() === 'batch') { runBatch(); return; }
    if (!photo) { toast('Choose a leaf photo first.', 'error'); dz.focus(); return; }
    busy = true; $('#run-btn').disabled = true; $('#flow-run').classList.remove('done');
    const fd = new FormData();
    fd.append('leaf_image', photo, photo.name || 'photo.jpg');
    fd.append('model_name', $('#model-select').value);
    const kind = mode() === 'compare' ? 'compare' : 'predict';
    $('#results-panel').innerHTML = '';
    progress({ pct: 1, stage: 'segment', detail: 'Uploading…' });
    try {
      const r = await fetch(`/api/jobs/${kind}`, { method: 'POST', body: fd, headers: { Accept: 'application/json' } });
      const j = await r.json().catch(() => ({}));
      if (!r.ok || !j.job_id) throw new Error(j.error || `Server error ${r.status}`);
      let job;
      for (;;) {
        await sleep(450);
        const s = await fetch(`/api/jobs/${j.job_id}`, { headers: { Accept: 'application/json' } });
        job = await s.json();
        if (!s.ok) throw new Error(job.error || 'Lost the job.');
        progress(job);
        if (job.state === 'done' || job.state === 'error') break;
        renderPartial(job.partial);
      }
      if (job.state === 'error') throw new Error(job.error || 'Analysis failed.');
      $('#progress-panel').hidden = true;
      $('#results-panel').innerHTML = renderResult(job.result);
      $('#flow-run').classList.add('done');
      bumpHistoryCount();
      if (innerWidth < 900) $('#results-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
      $('#progress-panel').hidden = true;
      $('#results-panel').innerHTML = `<div class="banner error">${esc(err.message)}</div>`;
      toast(err.message, 'error');
    } finally {
      busy = false; $('#run-btn').disabled = !MODELS.some((m) => m.loaded);
    }
  });

  const STATUS_LABEL = { ok: 'Diagnosed', unsure: 'Not sure', ood: 'Not accepted', error: 'Error' };
  function renderBatch(res, running) {
    const items = (res.items || []).slice().sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
    const counts = res.counts || items.reduce((c, i) => { c[i.status] = (c[i.status] || 0) + 1; return c; }, {});
    const head = `<div class="verdict ${running ? 'unsure' : 'ok'}"><div class="verdict-top"><span class="badge brand">${esc(res.model_label || modelLabel(res.model) || 'Batch')}</span>
      ${Object.entries(counts).filter(([, n]) => n).map(([k, n]) => `<span class="badge ${k === 'ok' ? 'ok' : k === 'unsure' ? 'warn' : 'bad'}">${esc(STATUS_LABEL[k] || k)}: ${n}</span>`).join('')}</div>
      <h2>${running ? `Batch running — ${items.length} of ${res.n || '?'} done` : `Batch finished — ${items.length} photo${items.length === 1 ? '' : 's'}`}</h2>
      ${!running && res.batch_id ? `<div class="chip-row"><button type="button" class="btn btn-primary btn-small" data-dl="/api/batch/${esc(res.batch_id)}/report.pdf" data-name="TomatoLeafAI_batch.pdf">Download PDF report</button>
      <button type="button" class="btn btn-soft btn-small" data-dl="/api/history/export.csv?batch_id=${esc(res.batch_id)}" data-name="TomatoLeafAI_batch.csv">Download CSV</button></div>` : ''}</div>`;
    const rows = items.map((h) => `<tr class="${h.id ? 'clickable' : ''}" ${h.id ? `data-open="${esc(h.id)}" tabindex="0"` : ''}><td>${h.thumb ? `<img class="thumb" src="${h.thumb}" alt="">` : ''}</td>
      <td>${esc(h.filename || '')}</td><td><b>${esc(h.title || '—')}</b>${h.status === 'error' || h.status === 'ood' ? `<div class="muted small">${esc((h.message || '').slice(0, 120))}</div>` : ''}</td>
      <td class="num">${h.confidence != null ? pct(h.confidence, 1) : '—'}</td><td><span class="status-dot ${esc(h.status)}">${esc(STATUS_LABEL[h.status] || h.status)}</span></td>
      <td>${h.id ? `<button type="button" class="btn btn-soft btn-small" data-dl="/api/history/${esc(h.id)}/report.pdf" data-name="report.pdf">PDF</button>` : ''}</td></tr>`).join('');
    return `${head}<div class="table-wrap"><table class="data batch-table"><thead><tr><th></th><th>File</th><th>Diagnosis</th><th class="num">Confidence</th><th>Status</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
      ${!running ? '<p class="muted small">Click a row for the full result (Grad-CAM, advice). Every photo is also saved in History.</p>' : ''}`;
  }
  async function runBatch() {
    if (!batchFiles.length) { toast('Choose some leaf photos first.', 'error'); $('#batch-drop').focus(); return; }
    busy = true; $('#run-btn').disabled = true; $('#flow-run').classList.remove('done');
    const fd = new FormData();
    batchFiles.forEach((f) => fd.append('leaf_images', f, f.origName || f.name));
    fd.append('model_name', $('#model-select').value);
    $('#results-panel').innerHTML = '';
    progress({ pct: 1, stage: 'segment', detail: `Uploading ${batchFiles.length} photos…` });
    try {
      const r = await fetch('/api/jobs/batch', { method: 'POST', body: fd, headers: { Accept: 'application/json' } });
      const j = await r.json().catch(() => ({}));
      if (!r.ok || !j.job_id) throw new Error(j.error || `Server error ${r.status}`);
      let job;
      for (;;) {
        await sleep(600);
        const s2 = await fetch(`/api/jobs/${j.job_id}`, { headers: { Accept: 'application/json' } });
        job = await s2.json();
        if (!s2.ok) throw new Error(job.error || 'Lost the job.');
        progress({ ...job, stage: 'predict' });
        if (job.state === 'done' || job.state === 'error') break;
        $('#results-panel').innerHTML = renderBatch({ items: (job.partial || {}).items || [], n: j.n, model: $('#model-select').value }, true);
      }
      if (job.state === 'error') throw new Error(job.error || 'Batch failed.');
      $('#progress-panel').hidden = true;
      $('#results-panel').innerHTML = renderBatch(job.result, false);
      $('#flow-run').classList.add('done'); bumpHistoryCount();
    } catch (err) {
      $('#progress-panel').hidden = true;
      $('#results-panel').innerHTML = `<div class="banner error">${esc(err.message)}</div>`; toast(err.message, 'error');
    } finally { busy = false; $('#run-btn').disabled = !MODELS.some((m) => m.loaded); }
  }
  // downloads (PDF / CSV) and opening a batch row
  document.addEventListener('click', async (e) => {
    const d = e.target.closest('[data-dl]');
    if (d) {
      e.preventDefault(); e.stopPropagation(); d.disabled = true;
      try { await window.tlaiDownload(d.dataset.dl, d.dataset.name); } catch (err) { toast(err.message, 'error'); } finally { d.disabled = false; }
      return;
    }
    const o = e.target.closest('[data-open]');
    if (o && !e.target.closest('button')) openHistory(o.dataset.open);
  });
  document.addEventListener('keydown', (e) => { const o = e.target.closest && e.target.closest('[data-open]'); if (o && e.key === 'Enter') openHistory(o.dataset.open); });
  const reportBar = (id) => (id ? `<div class="report-bar"><button type="button" class="btn btn-soft btn-small" data-dl="/api/history/${esc(id)}/report.pdf" data-name="TomatoLeafAI_report.pdf">
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M12 16 7 11l1.4-1.4 2.6 2.6V4h2v8.2l2.6-2.6L17 11l-5 5Zm-7 4v-2h14v2H5Z"/></svg> Download PDF report</button></div>` : '');

  /* -------------------------------------------------------------- rendering */
  const fig = (src, cap) => (src ? `<figure><img src="${src}" data-zoom="${src}" data-cap="${esc(cap)}" alt="${esc(cap)}" loading="lazy"><figcaption>${esc(cap)}</figcaption></figure>` : '');
  const list = (arr) => (arr && arr.length ? `<ul>${arr.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>` : '');
  const acc = (title, arr, open = false) => (arr && arr.length ? `<details class="acc" ${open ? 'open' : ''}><summary>${esc(title)}</summary>${list(arr)}</details>` : '');
  const CATEGORY = { chemical_may_be_needed: 'Chemical control may be needed', biological_available: 'Biological / cultural control first',
    vector_control_for_viral: 'Viral — control the insect vector, remove infected plants', none_healthy: 'No treatment needed' };
  const ADVISORY = { high_confidence_guidance: 'High confidence', moderate_confidence_review_recommended: 'Moderate confidence — review recommended',
    low_confidence_seek_expert_opinion: 'Low confidence — seek expert opinion' };
  const DEEP_NAMES = { ood_gate_prob: 'OOD-gate CNN P(tomato)', maha_score: 'Mahalanobis p-value', energy_score: 'Energy p-value', msp: 'Max softmax (hybrid)' };

  function renderResult(r) {
    if (!r) return '';
    if (r.mode === 'batch') return renderBatch(r, false);
    const body = !r.accepted ? renderReject(r) : (r.mode === 'compare' || r.models ? renderCompare(r) : renderPredict(r));
    return reportBar(r.history_id) + body;
  }
  function gateBlock(g) {
    if (!g) return '';
    const ds = g.deep_scores || {};
    const cells = [
      `<div class="kpi"><b>${pct(g.confidence)}</b><small>Leaf-check confidence (accept ≥ ${pct(CFG.ood_threshold)})</small></div>`,
      g.heuristic_score != null ? `<div class="kpi"><b>${pct(g.heuristic_score)}</b><small>Shape / edge / vein / colour score</small></div>` : '',
      g.deep_score != null ? `<div class="kpi"><b>${pct(g.deep_score)}</b><small>Deep tomato-identity score</small></div>` : '',
      ...Object.entries(ds).map(([k, v]) => `<div class="kpi"><b>${fx(v, 3)}</b><small>${esc(DEEP_NAMES[k] || k)}</small></div>`),
    ].join('');
    return `<details class="acc"><summary>Tomato-leaf check details</summary><div style="padding:0 12px 12px"><div class="gate-grid">${cells}</div>${g.reasons && g.reasons.length ? `<ul class="reasons">${g.reasons.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>` : ''}</div></details>`;
  }
  function renderReject(r) {
    const p = r.preview || {};
    return `<div class="verdict bad"><div class="verdict-top"><span class="badge bad">${esc(r.stage === 'quality' ? 'Photo quality' : r.stage === 'leaf' ? 'Leaf check' : 'Tomato check')}</span></div>
      <h2>${esc(r.display_name || 'Not accepted')}</h2><p>${esc(r.message || '')}</p></div>
      <div class="imgs">${fig(p.original, 'Your photo')}${fig(p.segmentation, 'What the app found (green = leaf)')}${fig(p.background_removed, 'After background removal')}</div>
      ${gateBlock(r.gate)}
      <p class="muted small">Tips: one tomato leaf, filling most of the frame, in focus, in daylight. ${r.timing_ms ? `· ${(r.timing_ms / 1000).toFixed(1)} s` : ''}</p>`;
  }
  function explainImages(p, preview, withPreview = true) {
    const im = (p.explain || {}).images || {};
    return `<div class="imgs">${withPreview ? fig(preview.original, 'Your photo') + fig(preview.background_removed, 'Leaf after background removal') : ''}
      ${fig(im.gradcam, `${p.cam_method || 'Grad-CAM'}: where the model looked`)}${fig(im.disease_regions, 'Disease regions (CAM × lesion colour)')}
      ${fig(im.lesion_attention, 'Lesion attention (steers the ViT)')}${p.cam_method && /saliency/.test(p.cam_method) ? '' : fig(im.gradcam_pp, 'Grad-CAM++')}
      ${withPreview ? fig(preview.segmentation, 'Segmentation (green = leaf, red = lesions)') : ''}</div>`;
  }
  function kpis(r, p) {
    const x = p.explain || {};
    return `<div class="kpis">
      <div class="kpi"><b>${pct(x.lfs)}</b><small>Leaf-Focus Score — share of attention on the leaf</small></div>
      ${!p.healthy && x.lefs != null ? `<div class="kpi"><b>${pct(x.lefs)}</b><small>Lesion-Focus Score — attention on diseased tissue</small></div>` : ''}
      ${!p.healthy && x.pointing_lesion != null ? `<div class="kpi"><b>${x.pointing_lesion ? 'Yes' : 'No'}</b><small>Strongest attention point on a lesion</small></div>` : ''}
      ${r.affected_area_estimate != null ? `<div class="kpi"><b>${pct(r.affected_area_estimate)}</b><small>Leaf area with lesion colour (estimate, not a severity grade)</small></div>` : ''}
      <div class="kpi"><b>${pct(r.gate && r.gate.confidence)}</b><small>Tomato-leaf check confidence</small></div>
      ${r.timing_ms ? `<div class="kpi"><b>${(r.timing_ms / 1000).toFixed(1)} s</b><small>Processing time</small></div>` : ''}
    </div>`;
  }
  function top5(p) {
    return `<div class="bars">${(p.top5 || []).map((t) => `<div class="bar-row"><span>${esc(t.display_name)}</span><div class="bar"><span style="width:${(t.prob * 100).toFixed(1)}%"></span></div><span class="val">${pct(t.prob, 1)}</span></div>`).join('')}</div>`;
  }
  function verdict(r, p, extraBadges = '') {
    const cls = p.confident ? 'ok' : 'unsure';
    return `<div class="verdict ${cls}"><div class="verdict-top"><span class="badge brand">${esc(p.label || modelLabel(p.model))}</span>
      ${p.calibrated ? `<span class="badge">calibrated T = ${fx(p.temperature, 2)}</span>` : '<span class="badge warn">not calibrated</span>'}${extraBadges}
      <span class="badge ${p.confident ? 'ok' : 'warn'}">${p.confident ? 'Diagnosed' : 'Not sure'}</span></div>
      <h2>${p.healthy ? '🌿 ' : ''}${esc(p.display_name)}</h2>
      <div class="meter"><span style="width:${Math.max(2, (p.confidence || 0) * 100).toFixed(1)}%"></span></div>
      <p>Confidence ${pct(p.confidence, 1)}${p.confident ? ' (temperature-calibrated).' : ` — below the ${pct(CFG.conf_threshold)} threshold, so treat this as a hint: the leaf may be unclear or the disease unusual. Retake the photo or confirm with an agronomist.`}</p></div>`;
  }
  function treatment(t, p) {
    if (!t) return p && p.known_class === false ? `<div class="banner notice" style="margin-top:14px">“${esc(p.raw_class || p.class)}” is not one of the tomato classes in the treatment guide, so no management advice is shown.</div>` : '';
    return `<h3 class="section-title">Treatment &amp; management — ${esc(t.disease_name || '')}</h3>
      <div class="chip-row">${t.control_category ? `<span class="badge">${esc(CATEGORY[t.control_category] || t.control_category)}</span>` : ''}
      ${t.pathogen_type ? `<span class="badge">${esc(t.pathogen_type)}</span>` : ''}
      ${t.advisory_level ? `<span class="badge ${/low/.test(t.advisory_level) ? 'warn' : ''}">${esc(ADVISORY[t.advisory_level] || t.advisory_level)}</span>` : ''}</div>
      ${/low/.test(t.advisory_level || '') ? '<div class="banner warn" style="margin-top:10px">Confidence is low. Retake the photo (one leaf, in focus, daylight) or confirm with an agronomist before acting.</div>' : ''}
      ${acc('What to look for', t.symptoms_observed, true)}${acc('Cultural practices', t.cultural_practices, true)}
      ${acc('Sanitation', t.sanitation)}${acc('Irrigation & water', t.irrigation_water_management)}
      ${acc('Monitoring', t.monitoring_advice)}${acc('Prevention', t.prevention_advice)}
      ${t.severity_note ? `<p class="muted small">${esc(t.severity_note)}</p>` : ''}
      <p class="disclaimer">${esc(t.disclaimer || '')}</p>`;
  }
  function renderPredict(r) {
    const p = r.prediction || {};
    return `${verdict(r, p)}${explainImages(p, r.preview || {})}
      <h3 class="section-title">Explanation &amp; scores</h3>${kpis(r, p)}
      <h3 class="section-title">Other possibilities</h3>${top5(p)}
      ${gateBlock(r.gate)}${treatment(r.treatment, p)}`;
  }
  function renderCompare(r) {
    const p = r.prediction || {};
    const c = r.consensus || {};
    const rows = Object.values(r.models || {});
    const accOf = (n) => (MODELS.find((m) => m.name === n) || {}).accuracy;
    const table = `<div class="table-wrap"><table class="data"><thead><tr><th>Model</th><th>Prediction</th><th class="num">Confidence</th><th class="num">Leaf-Focus</th><th class="num">Lesion-Focus</th><th>Grad-CAM</th><th class="num" title="Accuracy on the model's in-distribution test split (results/eval)">Test acc.</th><th class="num">Time</th></tr></thead><tbody>
      ${rows.map((m) => `<tr class="${m.class === c.class ? 'best' : ''}"><td>${esc(m.label)}${m.cam_method && m.cam_method !== 'Grad-CAM' ? `<div class="muted small">${esc(m.cam_method)}</div>` : ''}</td><td>${esc(m.display_name)}${m.confident ? '' : ' <span class="badge warn">not sure</span>'}</td>
      <td class="num">${pct(m.confidence, 1)}</td><td class="num">${pct((m.explain || {}).lfs)}</td><td class="num">${pct((m.explain || {}).lefs)}</td>
      <td>${(m.explain || {}).images && m.explain.images.gradcam ? `<img class="thumb" src="${m.explain.images.gradcam}" data-zoom="${m.explain.images.gradcam}" data-cap="${esc(m.label)} ${esc(m.cam_method || 'Grad-CAM')}" alt="">` : ''}</td>
      <td class="num">${pct(accOf(m.model), 1)}</td><td class="num">${m.time_ms != null ? `${m.time_ms} ms` : '—'}</td></tr>`).join('')}
      </tbody></table></div>`;
    return `${verdict(r, p, ` <span class="badge">${c.agree}/${c.total} models agree</span>`)}
      ${explainImages(p, r.preview || {})}
      <h3 class="section-title">All models</h3>${table}
      <h3 class="section-title">Explanation &amp; scores (${esc(p.label)})</h3>${kpis(r, p)}
      <h3 class="section-title">Other possibilities (${esc(p.label)})</h3>${top5(p)}
      ${gateBlock(r.gate)}${treatment(r.treatment, p)}`;
  }

  /* ---------------------------------------------------------------- history */
  let HISTORY = []; let histFilter = 'all';
  function bumpHistoryCount(n) {
    if (n === undefined) { fetch('/api/history?limit=1').then((x) => x.json()).then((j) => bumpHistoryCount(j.total)).catch(() => {}); return; }
    $('#history-count').hidden = !n; $('#history-count').textContent = n;
  }
  async function loadHistory() {
    try {
      const j = await (await fetch('/api/history?limit=300', { headers: { Accept: 'application/json' } })).json();
      HISTORY = j.items || []; bumpHistoryCount(j.total);
      $('#history-clear-btn').hidden = !HISTORY.length; $('#history-export-btn').hidden = !HISTORY.length;
      $('#history-clear-stale-btn').hidden = !j.stale;
      $('#history-stale-note').hidden = !j.stale;
      if (j.stale) $('#history-stale-note').textContent = `${j.stale} result(s) come from an older pipeline version and may not match the current models.`;
      renderHistory();
    } catch { $('#history-list').innerHTML = '<p class="placeholder">Could not load history.</p>'; }
  }
  function renderHistory() {
    const q = ($('#history-search').value || '').toLowerCase();
    const items = HISTORY.filter((h) => (histFilter === 'all' || h.status === histFilter)
      && (!q || `${h.title} ${h.model} ${h.filename} ${h.mode}`.toLowerCase().includes(q)));
    $('#history-list').innerHTML = items.length ? items.map((h) => `<article class="history-item" data-id="${esc(h.id)}" tabindex="0">
      ${h.thumb ? `<img src="${h.thumb}" alt="">` : '<span class="noimg"></span>'}
      <div><b>${esc(h.title || '—')}</b><div class="meta">${esc([h.model, h.mode === 'compare' ? `compare${h.agree ? ` (${h.agree} agree)` : ''}` : '', h.filename, h.batch_id ? 'batch' : '', new Date(h.created_at * 1000).toLocaleString()].filter(Boolean).join(' · '))}</div></div>
      <div class="right"><span class="status-dot ${h.status}">${h.status === 'ok' ? pct(h.confidence) : h.status === 'unsure' ? `not sure ${pct(h.confidence)}` : 'rejected'}</span>
      <button type="button" class="btn btn-soft btn-small btn-danger" data-del="${esc(h.id)}" aria-label="Delete">✕</button></div></article>`).join('')
      : `<p class="placeholder">${HISTORY.length ? 'Nothing matches.' : 'No diagnoses yet. Your results appear here automatically.'}</p>`;
  }
  $$('.chip[data-filter]').forEach((c) => c.addEventListener('click', () => {
    histFilter = c.dataset.filter; $$('.chip[data-filter]').forEach((x) => x.classList.toggle('active', x === c)); renderHistory();
  }));
  $('#history-search').addEventListener('input', renderHistory);
  async function openHistory(id) {
    const r = await fetch(`/api/history/${id}`, { headers: { Accept: 'application/json' } });
    const item = await r.json();
    if (!r.ok) { toast(item.error || 'Not found', 'error'); return; }
    $('#history-modal-title').textContent = (item.result && item.result.display_name) || 'Diagnosis';
    $('#history-modal-meta').textContent = `${item.filename || ''} · ${new Date(item.created_at * 1000).toLocaleString()} · ${item.mode} · pipeline ${item.pipeline_version}`;
    $('#history-modal-body').innerHTML = renderResult({ ...item.result, history_id: item.id });
    $('#history-modal').showModal();
  }
  $('#history-list').addEventListener('click', async (e) => {
    const del = e.target.closest('[data-del]');
    if (del) {
      e.stopPropagation();
      const r = await fetch(`/api/history/${del.dataset.del}`, { method: 'DELETE' });
      if (r.ok) { HISTORY = HISTORY.filter((h) => h.id !== del.dataset.del); renderHistory(); bumpHistoryCount(HISTORY.length); toast('Deleted'); }
      return;
    }
    const it = e.target.closest('.history-item'); if (it) openHistory(it.dataset.id);
  });
  $('#history-list').addEventListener('keydown', (e) => { if (e.key === 'Enter' && e.target.classList.contains('history-item')) openHistory(e.target.dataset.id); });
  $('#history-clear-btn').addEventListener('click', async () => {
    if (!confirm('Delete all saved diagnoses?')) return;
    await fetch('/api/history', { method: 'DELETE' }); loadHistory(); toast('History cleared');
  });
  $('#history-export-btn').addEventListener('click', async () => {
    try { await window.tlaiDownload('/api/history/export.csv', 'TomatoLeafAI_history.csv'); } catch (err) { toast(err.message, 'error'); }
  });
  $('#history-clear-stale-btn').addEventListener('click', async () => {
    const j = await (await fetch('/api/history?stale=1', { method: 'DELETE' })).json(); loadHistory(); toast(`Removed ${j.removed} old result(s)`);
  });

  /* ----------------------------------------------------------------- models */
  const chart = (file, cap) => (file ? `<figure><img src="${CFG.chart_base}${encodeURIComponent(file)}" data-zoom="${CFG.chart_base}${encodeURIComponent(file)}" data-cap="${esc(cap)}" alt="${esc(cap)}" loading="lazy"><figcaption>${esc(cap)}</figcaption></figure>` : '');
  function tableFromRows(rows, cols, opts = {}) {
    if (!rows || !rows.length) return '';
    cols = cols || Object.keys(rows[0]);
    const isPct = opts.pct || (() => false);
    return `<div class="table-wrap"><table class="data"><thead><tr>${cols.map((c) => `<th>${esc(c.replace(/_/g, ' '))}</th>`).join('')}</tr></thead><tbody>
      ${rows.map((r) => `<tr>${cols.map((c) => { const v = r[c]; return typeof v === 'number' ? `<td class="num">${isPct(c) ? pct(v, 2) : smart(v)}</td>` : `<td>${esc(v ?? '—')}</td>`; }).join('')}</tr>`).join('')}
      </tbody></table></div>`;
  }
  const STATE_BADGE = { loaded: '', loading: '<span class="badge warn">loading…</span>', queued: '<span class="badge warn">waiting to load</span>',
    error: '<span class="badge bad">failed to load</span>', results: '<span class="badge">results only</span>' };
  const ARCH_LABEL = { v21: 'v21 trunk + head', keras: 'Keras model', savedmodel: 'TensorFlow SavedModel' };
  function modelCard(m, r) {
    const splits = (r.splits || []);
    const rep = (k) => ((r.reports || {})[k] || {}).overall || {};
    const pv = r.pv || {};
    const accRows = splits.map((sp) => `<dt>${esc(sp.label)} accuracy</dt><dd>${pct(rep(sp.key).accuracy, 2)}</dd>`).join('')
      || '<dt>Test accuracy</dt><dd>—</dd>';
    const arch = m.arch ? `<dt>Type</dt><dd>${esc(ARCH_LABEL[m.arch] || m.arch)}</dd>` : '';
    const inp = m.input ? `<dt>Input</dt><dd>${m.input[0]}×${m.input[1]}×${m.input[2]} · ${esc(m.preprocess || '')}</dd>` : '';
    const cls = m.num_classes ? `<dt>Classes</dt><dd>${m.num_classes}${m.classes_source === 'model' ? ' (own list)' : ''}</dd>` : '';
    const cam = m.cam_method ? `<dt>Explanation</dt><dd title="${esc(m.cam_layer || '')}">${esc(m.cam_method)}${m.cam_hw ? ` ${m.cam_hw[0]}×${m.cam_hw[1]}` : ''}</dd>` : '';
    const file = m.file ? `<dt>File</dt><dd title="${esc(m.file)}">${esc(m.file.length > 22 ? m.file.slice(0, 20) + '…' : m.file)}${m.size_mb != null ? ` · ${m.size_mb} MB` : ''}</dd>` : '';
    return `<article class="card model-card ${esc(m.kind)}" data-model="${esc(m.name)}" tabindex="0">
      <div class="model-badges">${m.recommended ? '<span class="badge brand">default</span>' : ''}${m.top ? '<span class="badge ok">top acc.</span>' : ''}${m.is_gate_model ? '<span class="badge" title="Its features power the tomato-leaf check">gate features</span>' : ''}${STATE_BADGE[m.state] || ''}</div>
      <span class="tag">${esc(KIND_LABEL[m.kind] || m.kind)}</span>
      <h3>${esc(m.label)}</h3><p class="muted small">${esc(m.desc || '')}</p>
      ${m.error ? `<p class="small error-text">${esc(m.error)}</p>` : ''}
      <dl>${accRows}${pv.f1_macro != null ? `<dt>Macro-F1 (${esc((splits[0] || {}).label || 'test')})</dt><dd>${pct(pv.f1_macro, 2)}</dd>` : ''}
      ${r.gap != null ? `<dt>Lab→field gap</dt><dd>${(r.gap * 100).toFixed(2)} pp</dd>` : ''}
      ${pv.ece != null ? `<dt>ECE</dt><dd>${pct(pv.ece, 2)}</dd>` : ''}<dt>Temperature</dt><dd>${m.temperature ? m.temperature.toFixed(3) : '—'}</dd>
      <dt>Parameters</dt><dd>${params(m.params)}</dd>${arch}${inp}${cls}${cam}${file}</dl></article>`;
  }
  async function loadModelsTab(silent = false) {
    const el = $('#models-view');
    try {
      const [perf, mm] = await Promise.all([
        fetch('/api/performance', { headers: { Accept: 'application/json' } }).then((x) => x.json()),
        fetch('/api/models', { headers: { Accept: 'application/json' } }).then((x) => x.json())]);
      MODELS = mm.models; window.__perf = perf;
      const res = perf.results || {}; const ev = perf.evaluation || {}; const st = mm.status || {};
      const loadedN = MODELS.filter((m) => m.loaded).length;
      $('#models-summary').textContent = `${MODELS.filter((m) => m.on_disk).length} model file(s) in models/ · ${loadedN} loaded${st.loading ? ' · loading…' : ''}${Object.keys(res).length ? ` · ${Object.keys(res).length} with evaluation results` : ''}`;
      const groups = ['proposed', 'baseline', 'custom', 'ablation'].map((k) => [k, MODELS.filter((m) => m.kind === k)]).filter(([, l]) => l.length);
      let html = MODELS.length ? `<div class="model-grid">${groups.flatMap(([, l]) => l).map((m) => modelCard(m, res[m.name] || {})).join('')}</div>`
        : '<div class="banner warn">No models yet. Copy trained model files into <code>models/</code> or use <b>Add a model</b> above.</div>';
      if (!Object.keys(res).length) html += '<div class="banner notice" style="margin-top:14px">No per-model evaluation files found in <code>results/eval/</code> (written by notebook Cell 9.2). Copy the notebook <code>results</code> folder next to <code>app.py</code>; the numbers appear automatically.</div>';
      const comp = ev.comparison;
      if (comp && comp.length) html += `<h2 class="section-title">Test-set comparison (Cell 9.3)</h2>${tableFromRows(comp, null, { pct: (c) => /accuracy|precision|recall|f1|ece/i.test(c) })}`;
      const g = ev.gate;
      if (g && g.heldout) {
        html += `<h2 class="section-title">Tomato-leaf gate on held-out images (Cell 9.5)</h2><div class="kpis">
          <div class="kpi"><b>${pct(g.heldout.tomato_accept_rate, 1)}</b><small>real tomato leaves accepted</small></div>
          <div class="kpi"><b>${pct(g.heldout.non_tomato_reject_rate, 1)}</b><small>non-tomato images rejected</small></div>
          ${Object.entries(g.heldout.per_category_reject_rate || {}).map(([k, v]) => `<div class="kpi"><b>${pct(v, 1)}</b><small>${esc(k)} rejected</small></div>`).join('')}
          ${Object.entries(g.component_auroc || {}).map(([k, v]) => `<div class="kpi"><b>${fx(v, 3)}</b><small>AUROC — ${esc(DEEP_NAMES[k] || k)}</small></div>`).join('')}</div>`;
      }
      if (ev.calibration) {
        const rows = Object.entries(ev.calibration).filter(([, v]) => v && typeof v === 'object').map(([k, v]) => ({ model: k, ...Object.fromEntries(Object.entries(v).filter(([, x]) => typeof x !== 'object')) }));
        if (rows.length) html += `<h2 class="section-title">Calibration (Cell 8.2)</h2>${tableFromRows(rows, null, { pct: (c) => /ece|mce/.test(c) })}`;
      }
      const rs = ev.research;
      if (rs) html += '<div class="banner notice" style="margin-top:14px">Research questions, hypotheses and the hybrid-proof scorecard are on the <a href="#results" data-goto="results">Results</a> tab.</div>';
      if (false) {
        html += `<h2 class="section-title">Research questions &amp; hypotheses (notebook Cell 10.3)</h2>`;
        if (ev.rq1_table && ev.rq1_table.length) html += `<p class="muted small">RQ1 — accuracy and reliability on lab and field images, lab-to-field gap (95% CI)</p>${tableFromRows(ev.rq1_table)}`;
        html += `<div class="rq-grid">${['H1', 'H2', 'H3'].filter((k) => rs[k]).map((k) => `<div class="kpi rq-card ${/^SUPPORTED/.test(rs[k].verdict || '') ? 'ok' : /PARTIAL/.test(rs[k].verdict || '') ? 'part' : 'no'}"><b>${esc(k)}: ${esc(rs[k].verdict || '')}</b><small>${esc(rs[k].rule || '')}</small></div>`).join('')}
          ${['RQ2', 'RQ3'].filter((k) => rs[k]).map((k) => `<div class="kpi rq-card ${/^(YES|SUPPORTED)/.test(rs[k].answer || '') ? 'ok' : /PART/.test(rs[k].answer || '') ? 'part' : 'no'}"><b>${esc(k)}: ${esc(rs[k].answer || '')}</b><small>${esc(Object.entries(rs[k].checks || {}).map(([a, b]) => `${a.replace(/_/g, ' ')}: ${b ? 'yes' : 'no'}`).join(' · '))}</small></div>`).join('')}</div>`;
      }
      const hyp = rs ? null : ev.hypotheses;
      if (hyp) html += `<h2 class="section-title">Proposal hypotheses (Cell 10.3)</h2><div class="kpis">${Object.keys(hyp).filter((k) => hyp[k] && typeof hyp[k] === 'object' && hyp[k].verdict).map((k) => `<div class="kpi"><b>${esc(k)}</b><small>${esc(hyp[k].verdict || '')}</small></div>`).join('')}</div>`;
      if (ev.statistical && typeof ev.statistical === 'object') {
        Object.entries(ev.statistical).filter(([, v]) => v && typeof v === 'object' && !Array.isArray(v)).slice(0, 4).forEach(([name, mc]) => {
          const cells = Object.entries(mc).filter(([, v]) => typeof v !== 'object').slice(0, 8);
          if (cells.length) html += `<h2 class="section-title">${esc(name.replace(/_/g, ' '))} (Cell 9.4)</h2><div class="kpis">${cells.map(([k, v]) => `<div class="kpi"><b>${typeof v === 'number' ? fx(v, 4) : esc(v)}</b><small>${esc(k.replace(/_/g, ' '))}</small></div>`).join('')}</div>`;
        });
      }
      if (ev.xai && ev.xai.length) html += `<h2 class="section-title">Explainability — Leaf-Focus / Lesion-Focus (Cell 10.2)</h2>${tableFromRows(ev.xai, null, { pct: (c) => /lfs|lefs|top10|pointing/i.test(c) })}`;
      if (ev.latency && ev.latency.length) html += `<h2 class="section-title">Speed &amp; size (Cell 10.3)</h2>${tableFromRows(ev.latency)}`;
      if ((ev.plots || []).length) html += `<h2 class="section-title">Project charts</h2><div class="chart-grid">${ev.plots.map((p) => chart(p, p.replace(/\.png$/, '').replace(/_/g, ' '))).join('')}</div>`;
      if ((ev.xai_examples || []).length) html += `<h2 class="section-title">Grad-CAM examples from the notebook</h2><div class="chart-grid">${ev.xai_examples.slice(0, 12).map((p) => chart(p, p.replace(/\.png$/, '').replace(/_/g, ' '))).join('')}</div>`;
      el.innerHTML = html;
    } catch (err) { if (!silent) el.innerHTML = `<p class="placeholder">Could not load model results (${esc(err.message)}).</p>`; }
  }

  // ---- add / rescan models (Models tab)
  $('#rescan-btn').addEventListener('click', async () => {
    $('#rescan-btn').disabled = true;
    try { await fetch('/api/models/rescan', { method: 'POST' }); toast('Looking for new models…'); await sleep(600); await pollStatus(); }
    catch { toast('Rescan failed', 'error'); } finally { $('#rescan-btn').disabled = false; }
  });
  $('#add-model-btn').addEventListener('click', () => { $('#add-model-form').hidden = !$('#add-model-form').hidden; });
  const adminBtn = $('#admin-btn');
  const showAdmin = (on) => { $('#rescan-btn').hidden = !on; $('#add-model-btn').hidden = !on; if (adminBtn) adminBtn.textContent = on ? 'Admin ✓' : 'Admin'; };
  if (CFG.public_mode) showAdmin(window.tlaiIsAdmin && window.tlaiIsAdmin());
  if (adminBtn) adminBtn.addEventListener('click', () => {
    if (window.tlaiIsAdmin()) { window.tlaiSetAdmin(''); showAdmin(false); toast('Admin mode off'); return; }
    const t = prompt('Admin token (set as ADMIN_TOKEN on the server):');
    if (t) { window.tlaiSetAdmin(t.trim()); showAdmin(true); toast('Admin mode on for this tab'); }
  });
  $('#add-model-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = $('#model-file').files[0];
    if (!f) { toast('Choose a .keras or .h5 file.', 'error'); return; }
    const fd = new FormData(e.target);
    const btn = $('#upload-model-btn'); btn.disabled = true;
    const send = (overwrite) => new Promise((resolve) => {
      if (overwrite) fd.set('overwrite', '1');
      try { const t = sessionStorage.getItem('tlai_admin'); if (t) fd.set('admin_token', t); } catch { /* none */ }
      const x = new XMLHttpRequest(); x.open('POST', '/api/models/upload');
      x.upload.onprogress = (ev) => { if (ev.lengthComputable) $('#upload-model-status').textContent = `Uploading ${(100 * ev.loaded / ev.total).toFixed(0)}%…`; };
      x.onload = () => { let j = {}; try { j = JSON.parse(x.responseText); } catch { /* */ } resolve({ status: x.status, j }); };
      x.onerror = () => resolve({ status: 0, j: { error: 'Upload failed (connection).' } });
      x.send(fd);
    });
    try {
      let r = await send(false);
      if (r.status === 409 && confirm(`${r.j.error} Replace it?`)) r = await send(true);
      if (r.status !== 200) throw new Error(r.j.error || `Upload failed (${r.status})`);
      $('#upload-model-status').textContent = `Saved models/${r.j.saved} — loading…`;
      toast(`Added ${r.j.model} — loading`); e.target.reset(); $('#add-model-form').hidden = true;
      $('#upload-model-status').textContent = ''; await sleep(600); await pollStatus();
    } catch (err) { $('#upload-model-status').textContent = err.message; toast(err.message, 'error'); }
    finally { btn.disabled = false; }
  });
  function openModel(name) {
    const perf = window.__perf || {}; const r = (perf.results || {})[name] || {};
    const m = MODELS.find((x) => x.name === name) || { label: name };
    $('#model-modal-title').textContent = m.label;
    $('#model-modal-meta').textContent = [KIND_LABEL[m.kind] || m.kind, m.arch ? (ARCH_LABEL[m.arch] || m.arch) : '', m.params ? `${params(m.params)} parameters` : '', m.temperature ? `T = ${m.temperature.toFixed(3)}` : ''].filter(Boolean).join(' · ');
    const section = (rep, title) => {
      if (!rep) return `<p class="muted">${esc(title)}: no evaluation file.</p>`;
      const o = rep.overall || {};
      const pc = Object.entries(rep.per_class || {}).map(([k, v]) => ({ class: pretty(k), precision: v.precision, recall: v.recall, f1: v.f1, support: v.support }));
      const pl = rep.plots || {};
      return `<h3 class="section-title">${esc(title)} <span class="muted small">(${rep.n_samples || '?'} images)</span></h3>
        <div class="kpis"><div class="kpi"><b>${pct(o.accuracy, 2)}</b><small>accuracy</small></div><div class="kpi"><b>${pct(o.f1_macro, 2)}</b><small>macro-F1</small></div>
        <div class="kpi"><b>${pct(o.balanced_accuracy, 2)}</b><small>balanced accuracy</small></div><div class="kpi"><b>${fx(o.mcc, 3)}</b><small>MCC</small></div>
        <div class="kpi"><b>${fx(o.roc_auc_macro, 3)}</b><small>ROC-AUC (macro)</small></div><div class="kpi"><b>${pct(o.ece, 2)}</b><small>ECE (uncalibrated)</small></div></div>
        <div style="margin-top:10px">${tableFromRows(pc, ['class', 'precision', 'recall', 'f1', 'support'], { pct: (c) => c !== 'support' })}</div>
        <div class="chart-grid" style="margin-top:10px">${chart(pl.confusion_matrix_normalized, 'Confusion matrix (normalised)')}${chart(pl.roc_curves, 'ROC curves')}
        ${chart(pl.pr_curves, 'Precision-recall curves')}${chart(pl.reliability_diagram, 'Reliability diagram')}${chart(pl.training_curves, 'Training curves')}</div>`;
    };
    const splits = r.splits || [];
    const arch = [m.summary, m.file ? `File: models/${m.file}` : '', m.error ? `Load error: ${m.error}` : ''].filter(Boolean).map((x) => `<p class="muted small">${esc(x)}</p>`).join('');
    $('#model-modal-body').innerHTML = `<p class="muted">${esc(m.desc || '')}</p>${arch}${splits.length ? splits.map((sp) => section((r.reports || {})[sp.key], sp.label)).join('') : '<p class="muted">No evaluation files for this model in results/eval/.</p>'}`;
    $('#model-modal').showModal();
  }
  $('#models-view').addEventListener('click', (e) => { const c = e.target.closest('.model-card'); if (c && !e.target.closest('img')) openModel(c.dataset.model); });
  $('#models-view').addEventListener('keydown', (e) => { const c = e.target.closest('.model-card'); if (c && e.key === 'Enter') openModel(c.dataset.model); });

  $('#results-reload-btn').addEventListener('click', () => window.TLAI_RESULTS && window.TLAI_RESULTS.reload());
  document.addEventListener('click', (e) => { const a = e.target.closest('[data-goto]'); if (a) { e.preventDefault(); showTab(a.dataset.goto); } });

  /* --------------------------------------------------------------- diseases */
  async function loadDiseaseDisclaimer() {
    try { const j = await (await fetch('/api/diseases')).json(); $('#disease-disclaimer').textContent = j.disclaimer || ''; } catch { /* optional */ }
  }
  $('#disease-search').addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase();
    $$('.disease-card').forEach((c) => { c.hidden = !!q && !c.dataset.search.includes(q); });
  });

  /* ------------------------------------------------------------------- init */
  renderModelSelect();
  const savedModel = store.get('tla_model', null);
  if (savedModel && MODELS.some((m) => m.name === savedModel && m.loaded)) { $('#model-select').value = savedModel; updateModelDesc(); }
  { const sm = store.get('tla_mode', 'single'); const r = $(`input[name="run_mode"][value="${sm === 'compare' || sm === 'batch' ? sm : 'single'}"]`); if (r) { r.checked = true; applyMode(); } }
  applyStatus({ loading: CFG.loading, progress: { done: 0, total: MODELS.length, current: null }, models_loaded: MODELS.filter((m) => m.loaded).map((m) => m.name) });
  STATUS_VERSION = CFG.version;
  if (CFG.loading) pollStatus();
  setTimeout(watchStatus, 4000);
  const init = $('#initial-result');
  if (init) { try { $('#results-panel').innerHTML = renderResult(JSON.parse(init.textContent)); } catch { /* noscript fallback only */ } }
  bumpHistoryCount();
  showTab((location.hash || '').slice(1) || 'diagnose');
  markFlow();
})();
