/* TomatoLeafAI v24 -- Results tab: the thesis evidence (RQ1-RQ3, H1-H3, hybrid-proof scorecard,
   robustness, Grad-CAM faithfulness, OOD gate, data validation) read from /api/thesis.
   Charts are plain inline SVG (no library, no external request); every chart has a hover/focus
   tooltip and a table view. Nothing is computed here that the notebook did not write. */
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const num = (v) => (v === null || v === undefined || v === '' || Number.isNaN(Number(v))) ? null : Number(v);
  const isPct = (v) => num(v) !== null && Math.abs(num(v)) <= 1.000001;
  const p1 = (v, d = 1) => (num(v) === null ? '—' : `${(num(v) * 100).toFixed(d)}%`);
  const f = (v, d = 2) => (num(v) === null ? '—' : num(v).toFixed(d));
  const pp = (v, d = 2) => (num(v) === null ? '—' : `${num(v) >= 0 ? '+' : ''}${num(v).toFixed(d)} pp`);
  const pval = (v) => (num(v) === null ? '—' : num(v) < 0.001 ? '< 0.001' : num(v).toFixed(3));
  let DATA = null; let LABELS = {};
  const HYB = () => (DATA && DATA.hybrid) || 'Hybrid_CNN_ViT';
  const lab = (m) => LABELS[m] || String(m || '').replace(/_/g, ' ');
  const pretty = (c) => String(c || '').replace('Tomato___', '').replace(/_/g, ' ').replace('Spider mites Two-spotted spider mite', 'Spider mites');

  /* --------------------------------------------------------------- status */
  const ICON = {
    good: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M6.4 11.2 3.2 8l1.1-1.1 2.1 2.1 5.3-5.3L12.8 4.8z"/></svg>',
    warning: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M7.2 3h1.6v6.4H7.2zm0 8h1.6v1.6H7.2z"/></svg>',
    critical: '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="m4.6 3.5 3.4 3.4 3.4-3.4 1.1 1.1L9.1 8l3.4 3.4-1.1 1.1L8 9.1l-3.4 3.4-1.1-1.1L6.9 8 3.5 4.6z"/></svg>',
    neutral: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="3" fill="currentColor"/></svg>',
  };
  function statusOf(text) {
    const t = String(text || '').toUpperCase();
    if (!t) return 'neutral';
    if (/^NOT|^NO\b|NOT THE BEST|NOT SUPPORTED/.test(t)) return 'critical';
    if (/PARTIAL|PARTLY|TIED|ONLY|NOT TESTED/.test(t)) return 'warning';
    if (/SUPPORTED|^YES|IS THE BEST/.test(t)) return 'good';
    return 'neutral';
  }
  const statusChip = (text, label) => {
    const s = statusOf(text);
    return `<span class="st st-${s}">${ICON[s]}<span>${esc(label || text || '—')}</span></span>`;
  };

  /* -------------------------------------------------------------- tooltip */
  let tip = null;
  function ensureTip() {
    if (tip) return tip;
    tip = document.createElement('div'); tip.className = 'viz-tip'; tip.setAttribute('role', 'status'); tip.hidden = true;
    document.body.appendChild(tip);
    return tip;
  }
  function showTip(el, x, y) {
    const t = ensureTip();
    const lines = String(el.getAttribute('data-tip') || '').split('\n');
    t.textContent = '';
    lines.forEach((l, i) => { const d = document.createElement(i === 0 ? 'strong' : 'div'); d.textContent = l; t.appendChild(d); });
    t.hidden = false;
    const r = t.getBoundingClientRect();
    let left = x + 14; let top = y + 14;
    if (left + r.width > innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > innerHeight - 8) top = y - r.height - 14;
    t.style.left = `${Math.max(8, left)}px`; t.style.top = `${Math.max(8, top)}px`;
  }
  function wireTips(root) {
    root.addEventListener('pointermove', (e) => { const el = e.target.closest('[data-tip]'); if (el) showTip(el, e.clientX, e.clientY); else if (tip) tip.hidden = true; });
    root.addEventListener('pointerleave', () => { if (tip) tip.hidden = true; });
    root.addEventListener('focusin', (e) => { const el = e.target.closest('[data-tip]'); if (!el) return; const r = el.getBoundingClientRect(); showTip(el, r.right, r.top); });
    root.addEventListener('focusout', () => { if (tip) tip.hidden = true; });
  }

  /* ------------------------------------------------------------- charts */
  const FONT = 12;
  const textW = (s, size = FONT) => String(s).length * size * 0.56;
  const niceMax = (v) => { if (v <= 0) return 1; const e = 10 ** Math.floor(Math.log10(v)); const m = v / e; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * e; };
  /** "nice" tick values covering [min, max] (3-6 ticks, steps 1/2/2.5/5 x 10^k). */
  function ticks(min, max) {
    const span = (max - min) || 1; const raw = span / 5; const e = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * e).find((st) => span / st <= 6) || 10 * e;
    const out = []; for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  // bar with rounded data-end, square at the baseline (horizontal)
  const hbarPath = (x0, y, w, h, r = 4) => {
    if (w <= 0) return '';
    const rr = Math.min(r, w, h / 2);
    return `M${x0},${y}H${x0 + w - rr}Q${x0 + w},${y} ${x0 + w},${y + rr}V${y + h - rr}Q${x0 + w},${y + h} ${x0 + w - rr},${y + h}H${x0}Z`;
  };
  const hbarPathNeg = (x0, y, w, h, r = 4) => {   // grows to the LEFT of x0
    if (w <= 0) return '';
    const rr = Math.min(r, w, h / 2);
    return `M${x0},${y}H${x0 - w + rr}Q${x0 - w},${y} ${x0 - w},${y + rr}V${y + h - rr}Q${x0 - w},${y + h} ${x0 - w + rr},${y + h}H${x0}Z`;
  };
  const vbarPath = (x, y0, w, h, r = 4) => {
    if (h <= 0) return '';
    const rr = Math.min(r, h, w / 2);
    return `M${x},${y0}V${y0 - h + rr}Q${x},${y0 - h} ${x + rr},${y0 - h}H${x + w - rr}Q${x + w},${y0 - h} ${x + w},${y0 - h + rr}V${y0}Z`;
  };

  /** Horizontal bars, one series. rows: [{label, value, lo?, hi?, highlight?, tip?}] */
  function hbars(rows, o = {}) {
    const W = o.width; const bh = 18; const gap = 12;
    const lw = Math.min(W * 0.42, Math.max(...rows.map((r) => textW(r.label))) + 10);
    const vw = 64; const x0 = lw; const plotW = Math.max(60, W - lw - vw);
    const vals = rows.flatMap((r) => [r.value, r.hi]).filter((v) => num(v) !== null);
    const max = o.max ?? niceMax(Math.max(...vals, 0));
    const min = o.min ?? 0;
    const sx = (v) => x0 + ((v - min) / (max - min || 1)) * plotW;
    const H = rows.length * (bh + gap) + 24;
    const fmt = o.fmt || ((v) => f(v));
    let s = `<svg class="viz-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria || '')}">`;
    for (const v of ticks(min, max)) {             // hairline grid + ticks
      const x = sx(v);
      s += `<line x1="${x}" x2="${x}" y1="0" y2="${H - 20}" class="grid"/><text x="${x}" y="${H - 6}" class="tick" text-anchor="middle">${esc(fmt(v))}</text>`;
    }
    rows.forEach((r, i) => {
      const y = i * (bh + gap) + 4; const v = num(r.value);
      s += `<text x="${lw - 8}" y="${y + bh / 2 + 4}" class="lbl${r.highlight ? ' strong' : ''}" text-anchor="end">${esc(r.label)}</text>`;
      if (v === null) return;
      const w = Math.max(0, sx(v) - x0);
      s += `<g class="mark" tabindex="0" data-tip="${esc(r.tip || `${fmt(v)}\n${r.label}`)}"><rect x="${x0}" y="${y - gap / 2}" width="${plotW + vw}" height="${bh + gap}" class="hit"/>`
        + `<path d="${hbarPath(x0, y, w, bh)}" class="${r.highlight ? 'b-hi' : (o.single ? 'b-1' : 'b-lo')}"/>`;
      if (num(r.lo) !== null && num(r.hi) !== null) {
        s += `<line x1="${sx(r.lo)}" x2="${sx(r.hi)}" y1="${y + bh / 2}" y2="${y + bh / 2}" class="ci"/><line x1="${sx(r.lo)}" x2="${sx(r.lo)}" y1="${y + 4}" y2="${y + bh - 4}" class="ci"/><line x1="${sx(r.hi)}" x2="${sx(r.hi)}" y1="${y + 4}" y2="${y + bh - 4}" class="ci"/>`;
      }
      const tx = Math.max(sx(v), num(r.hi) !== null ? sx(r.hi) : 0) + 6;
      s += `<text x="${tx}" y="${y + bh / 2 + 4}" class="val">${esc(fmt(v))}</text></g>`;
    });
    return `${s}</svg>`;
  }

  /** Diverging horizontal bars around 0 (positive = blue, negative = red). */
  function divbars(rows, o = {}) {
    const W = o.width; const bh = 16; const gap = 10;
    const lw = Math.min(W * 0.4, Math.max(...rows.map((r) => textW(r.label))) + 10);
    const plotW = Math.max(80, W - lw - 16);
    const m = niceMax(Math.max(...rows.map((r) => Math.abs(num(r.value) || 0)), 0.5));
    const zx = lw + plotW / 2; const sc = (plotW / 2 - 40) / m;
    const H = rows.length * (bh + gap) + 24;
    let s = `<svg class="viz-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria || '')}">`;
    [-m, -m / 2, 0, m / 2, m].forEach((v) => { const x = zx + v * sc; s += `<line x1="${x}" x2="${x}" y1="0" y2="${H - 20}" class="${v === 0 ? 'axis' : 'grid'}"/><text x="${x}" y="${H - 6}" class="tick" text-anchor="middle">${v > 0 ? '+' : ''}${Number.isInteger(v) ? v : v.toFixed(1)}</text>`; });
    rows.forEach((r, i) => {
      const y = i * (bh + gap) + 4; const v = num(r.value) || 0; const w = Math.abs(v) * sc;
      s += `<text x="${lw - 8}" y="${y + bh / 2 + 4}" class="lbl" text-anchor="end">${esc(r.label)}</text>`;
      s += `<g class="mark" tabindex="0" data-tip="${esc(r.tip || `${pp(v)}\n${r.label}`)}"><rect x="${lw}" y="${y - gap / 2}" width="${plotW}" height="${bh + gap}" class="hit"/>`
        + `<path d="${v >= 0 ? hbarPath(zx, y, w, bh) : hbarPathNeg(zx, y, w, bh)}" class="${v >= 0 ? 'b-pos' : 'b-neg'}"/>`
        + `<text x="${v >= 0 ? zx + w + 5 : zx - w - 5}" y="${y + bh / 2 + 4}" class="val" text-anchor="${v >= 0 ? 'start' : 'end'}">${esc(pp(v, 1))}</text></g>`;
    });
    return `${s}</svg>`;
  }

  /** Grouped vertical bars: cats [{label}], series [{name, cls, values[]}] (2-3 series). */
  function groupedBars(cats, series, o = {}) {
    const W = o.width; const H = o.height || 260; const left = 44; const bottom = 54; const top = 10;
    const plotW = W - left - 8; const plotH = H - bottom - top;
    const vals = series.flatMap((s) => s.values).filter((v) => num(v) !== null);
    const min = o.min ?? 0; const max = o.max ?? niceMax(Math.max(...vals));
    const sy = (v) => top + plotH - ((v - min) / (max - min || 1)) * plotH;
    const band = plotW / cats.length; const bw = Math.min(24, (band - 16) / series.length - 2);
    const fmt = o.fmt || ((v) => f(v));
    let s = `<svg class="viz-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria || '')}">`;
    for (const v of ticks(min, max)) { const y = sy(v); s += `<line x1="${left}" x2="${W - 8}" y1="${y}" y2="${y}" class="${Math.abs(v - min) < 1e-9 ? 'axis' : 'grid'}"/><text x="${left - 6}" y="${y + 4}" class="tick" text-anchor="end">${esc(fmt(v))}</text>`; }
    cats.forEach((c, i) => {
      const cx = left + band * i + band / 2; const gw = series.length * (bw + 2) - 2; let x = cx - gw / 2;
      const tips = series.map((se) => `${se.name}: ${fmt(se.values[i])}`).join('\n');
      s += `<g class="mark" tabindex="0" data-tip="${esc(`${c.label}\n${tips}`)}"><rect x="${left + band * i}" y="${top}" width="${band}" height="${plotH}" class="hit"/>`;
      series.forEach((se) => { const v = num(se.values[i]); if (v !== null) s += `<path d="${vbarPath(x, sy(min), bw, sy(min) - sy(v))}" class="${se.cls}"/>`; x += bw + 2; });
      s += '</g>';
      const words = String(c.label).split(' '); const l1 = words.slice(0, Math.ceil(words.length / 2)).join(' '); const l2 = words.slice(Math.ceil(words.length / 2)).join(' ');
      s += `<text x="${cx}" y="${H - bottom + 18}" class="lbl${c.highlight ? ' strong' : ''}" text-anchor="middle">${esc(band < textW(c.label) ? l1 : c.label)}</text>`;
      if (band < textW(c.label) && l2) s += `<text x="${cx}" y="${H - bottom + 32}" class="lbl${c.highlight ? ' strong' : ''}" text-anchor="middle">${esc(l2)}</text>`;
    });
    return `${s}</svg>${legend(series.map((se) => ({ name: se.name, cls: se.cls })))}`;
  }
  const legend = (items) => `<div class="viz-legend">${items.map((i) => `<span><i class="sw ${i.cls}"></i>${esc(i.name)}</span>`).join('')}</div>`;

  /** Heatmap table (HTML so it reflows): rows [{label, cells:[{v, text, tip}]}], cols [label]. step(v)->1..7 */
  function heat(cols, rows, step, o = {}) {
    return `<div class="table-wrap"><table class="heat"><thead><tr><th>${esc(o.corner || '')}</th>${cols.map((c) => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>
      ${rows.map((r) => `<tr><th class="${r.highlight ? 'strong' : ''}">${esc(r.label)}</th>${r.cells.map((c) => {
        const k = num(c.v) === null ? 0 : step(c.v);
        return `<td class="q${k}" tabindex="0" data-tip="${esc(c.tip || c.text)}">${esc(c.text)}</td>`;
      }).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  const linStep = (lo, hi) => (v) => { const t = (num(v) - lo) / ((hi - lo) || 1); return Math.max(1, Math.min(7, 1 + Math.round(t * 6))); };

  /** Scatter with direct labels; points [{x, y, label, highlight}] */
  function scatter(points, o = {}) {
    const W = o.width; const H = o.height || 280; const left = 48; const bottom = 40; const top = 12; const right = 16;
    const xs = points.map((p) => p.x); const ys = points.map((p) => p.y);
    const pad = (a, b) => { const d = (b - a) || 0.05; return [a - d * 0.25, b + d * 0.25]; };
    const [x0, x1] = pad(Math.min(...xs), Math.max(...xs)); const [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
    const sx = (v) => left + ((v - x0) / (x1 - x0)) * (W - left - right); const sy = (v) => top + (1 - (v - y0) / (y1 - y0)) * (H - top - bottom);
    let s = `<svg class="viz-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria || '')}">`;
    for (let t = 0; t <= 4; t++) {
      const vx = x0 + (t / 4) * (x1 - x0); const vy = y0 + (t / 4) * (y1 - y0);
      s += `<line x1="${sx(vx)}" x2="${sx(vx)}" y1="${top}" y2="${H - bottom}" class="grid"/><text x="${sx(vx)}" y="${H - bottom + 16}" class="tick" text-anchor="middle">${vx.toFixed(2)}</text>`;
      s += `<line x1="${left}" x2="${W - right}" y1="${sy(vy)}" y2="${sy(vy)}" class="grid"/><text x="${left - 6}" y="${sy(vy) + 4}" class="tick" text-anchor="end">${vy.toFixed(2)}</text>`;
    }
    s += `<text x="${(left + W - right) / 2}" y="${H - 6}" class="axt" text-anchor="middle">${esc(o.xlabel || '')}</text>`;
    s += `<text transform="translate(12 ${(top + H - bottom) / 2}) rotate(-90)" class="axt" text-anchor="middle">${esc(o.ylabel || '')}</text>`;
    points.slice().sort((a, b) => (a.highlight ? 1 : 0) - (b.highlight ? 1 : 0)).forEach((p) => {
      const cx = sx(p.x); const cy = sy(p.y); const toRight = cx + textW(p.label) + 12 < W - right;
      s += `<g class="mark" tabindex="0" data-tip="${esc(p.tip || p.label)}"><circle cx="${cx}" cy="${cy}" r="14" class="hit"/><circle cx="${cx}" cy="${cy}" r="${p.highlight ? 6 : 5}" class="${p.highlight ? 'd-hi' : 'd-lo'}"/>`
        + `<text x="${toRight ? cx + 10 : cx - 10}" y="${cy + 4}" class="lbl${p.highlight ? ' strong' : ''}" text-anchor="${toRight ? 'start' : 'end'}">${esc(p.label)}</text></g>`;
    });
    return `${s}</svg>`;
  }

  /** 100% stacked horizontal bars: rows [{label, parts:[{name, cls, v}]}] */
  function stack100(rows, o = {}) {
    const W = o.width; const bh = 22; const gap = 18; const lw = Math.min(140, Math.max(...rows.map((r) => textW(r.label))) + 10);
    const plotW = W - lw - 8; const H = rows.length * (bh + gap) + 6;
    let s = `<svg class="viz-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria || '')}">`;
    rows.forEach((r, i) => {
      const y = i * (bh + gap) + 4; let x = lw; const tot = r.parts.reduce((a, p) => a + (num(p.v) || 0), 0) || 1;
      s += `<text x="${lw - 8}" y="${y + bh / 2 + 4}" class="lbl" text-anchor="end">${esc(r.label)}</text>`;
      r.parts.forEach((p, k) => {
        const w = ((num(p.v) || 0) / tot) * plotW; const last = k === r.parts.length - 1;
        const segW = Math.max(0, w - (last ? 0 : 2));
        const d = last ? hbarPath(x, y, segW, bh) : `M${x},${y}h${segW}v${bh}h${-segW}Z`;
        const txt = `${((num(p.v) || 0) / tot * 100).toFixed(1)}%`;
        s += `<g class="mark" tabindex="0" data-tip="${esc(`${txt}\n${r.label} · ${p.name}`)}"><path d="${d}" class="${p.cls}"/>`;
        const ink = p.cls === 'b-1' ? 'inlbl' : 'inlbl dark';      // white on blue, ink on orange / aqua (contrast)
        if (segW > textW(`${p.name} ${txt}`, 11) + 10) s += `<text x="${x + 6}" y="${y + bh / 2 + 4}" class="${ink}">${esc(`${p.name} ${txt}`)}</text>`;
        else if (segW > textW(txt, 11) + 8) s += `<text x="${x + 4}" y="${y + bh / 2 + 4}" class="${ink}">${esc(txt)}</text>`;
        s += '</g>'; x += w;
      });
    });
    return `${s}</svg>${legend(rows[0].parts.map((p) => ({ name: p.name, cls: p.cls })))}`;
  }

  /* ----------------------------------------------------------- page parts */
  const card = (title, sub, body, table, id) => `<section class="card viz-card"${id ? ` id="${id}"` : ''}><h2 class="viz-title">${esc(title)}</h2>${sub ? `<p class="viz-sub">${sub}</p>` : ''}<div class="viz-body">${body}</div>${table ? `<details class="viz-table"><summary>Table view</summary>${table}</details>` : ''}</section>`;
  const tbl = (head, rows) => `<div class="table-wrap"><table class="data"><thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  const tile = (label, value, note) => `<div class="stat"><span class="stat-label">${esc(label)}</span><span class="stat-value">${esc(value)}</span>${note ? `<span class="stat-note">${esc(note)}</span>` : ''}</div>`;
  const W = () => Math.max(280, Math.min(($('#results-view') || document.body).clientWidth - 48, 900));

  function sectionVerdict(d) {
    const sc = d.scorecard; const R = d.research || {}; const rq1 = (R.RQ1 || {}).rows || [];
    const hyb = rq1.find((r) => r.model === HYB());
    const tiles = [];
    if (hyb) {
      tiles.push(tile('Hybrid field macro-F1', p1(hyb.field_macro_f1), `95% CI ${p1(hyb.field_macro_f1_lo)} – ${p1(hyb.field_macro_f1_hi)}`));
      tiles.push(tile('Hybrid lab (PV) accuracy', p1(hyb.pv_accuracy), 'PlantVillage test'));
      tiles.push(tile('Lab-to-field gap', `${f(hyb.gap_pp, 1)} pp`, `95% CI ${f(hyb.gap_lo, 1)} – ${f(hyb.gap_hi, 1)}`));
    }
    const h3 = R.H3 || {};
    if (num(h3.gate_auroc) !== null) tiles.push(tile('Leaf-gate AUROC', f(h3.gate_auroc, 3), 'tomato vs non-tomato, held out'));
    if (num(h3.hybrid_lfs) !== null) tiles.push(tile('Hybrid Leaf-Focus Score', f(h3.hybrid_lfs, 2), 'share of Grad-CAM on the leaf'));
    const v = sc ? sc.verdict : null;
    const banner = v ? `<div class="verdict-banner st-bg-${statusOf(v)}">${ICON[statusOf(v)]}<div><strong>${esc(v.split(' -- ')[0])}</strong><p>${esc(v.split(' -- ').slice(1).join(' — '))}</p>
      <p class="small">Best mean rank: <b>${esc((sc.best_mean_rank || []).map(lab).join(', ') || '—')}</b> · First on field macro-F1: <b>${esc((sc.first_on_primary || []).map(lab).join(', ') || '—')}</b></p></div></div>`
      : '<div class="banner notice">The hybrid-proof scorecard (results/hybrid_proof_scorecard_v24.json, notebook Cell 10.3c) is not in results/ yet.</div>';
    return `${banner}${tiles.length ? `<div class="stats">${tiles.join('')}</div>` : ''}`;
  }

  function sectionRQ(d) {
    const R = d.research; if (!R) return '';
    const items = [
      ['RQ1', 'How accurate and how reliable are the models on lab and field images, and how large is the lab-to-field gap?', null, 'Answered by the table and chart below (accuracy, macro-F1 and calibration error with 95% CIs).'],
      ['RQ2', 'Does the hybrid CNN–ViT beat single-backbone models trained the same way?', (R.RQ2 || {}).answer, (R.RQ2 || {}).trained_the_same_way === false ? 'Warning: not every model was trained with the same pipeline.' : 'All models: same data, splits, augmentation, schedule, EMA, TTA and calibration.'],
      ['RQ3', 'Do the OOD gate, temperature scaling and Leaf-Focus Score add trust without lowering accuracy?', (R.RQ3 || {}).answer, null, (R.RQ3 || {}).checks],
      ['H1', 'Field fine-tuning gives useful field accuracy, and temperature scaling lowers ECE for every model.', (R.H1 || {}).verdict, (R.H1 || {}).rule],
      ['H2', 'The hybrid reaches a higher macro-F1 than every single-backbone model.', (R.H2 || {}).verdict, (R.H2 || {}).rule],
      ['H3', 'The OOD gate rejects unknown inputs with high AUROC, and most of the hybrid’s Grad-CAM is on the leaf.', (R.H3 || {}).verdict, (R.H3 || {}).rule, (R.H3 || {}).checks],
    ];
    const cards = items.map(([k, q, ans, note, checks]) => `<article class="rq-card"><div class="rq-head"><span class="rq-key">${k}</span>${ans ? statusChip(ans) : statusChip('', 'see below')}</div><p class="rq-q">${esc(q)}</p>
      ${note ? `<p class="small muted">${esc(note)}</p>` : ''}${checks ? `<ul class="checks">${Object.entries(checks).map(([c, ok]) => `<li>${statusChip(ok ? 'YES' : 'NO', ok ? 'met' : 'not met')} ${esc(c.replace(/_/g, ' '))}</li>`).join('')}</ul>` : ''}</article>`).join('');
    return card('Research questions & hypotheses', 'Verdicts written by notebook Cell 10.3 from the thresholds in the thesis proposal — nothing here is re-judged by the app.', `<div class="rq-grid">${cards}</div>`);
  }

  function sectionRQ1(d) {
    const rows = ((d.research || {}).RQ1 || {}).rows || []; if (!rows.length) return '';
    const cats = rows.map((r) => ({ label: lab(r.model), highlight: r.model === HYB() }));
    const chart = groupedBars(cats, [{ name: 'Lab (PlantVillage test)', cls: 'b-1', values: rows.map((r) => r.pv_accuracy) },
      { name: 'Field test', cls: 'b-2', values: rows.map((r) => r.field_accuracy) }],
    { width: W(), min: Math.max(0, Math.floor((Math.min(...rows.map((r) => num(r.field_accuracy) ?? 1)) - 0.03) * 20) / 20), max: 1, fmt: (v) => p1(v, 0), aria: 'Accuracy on lab and field test sets per model' });
    const table = tbl(['Model', 'PV acc', 'PV macro-F1', 'Field acc', 'Field macro-F1', 'Gap (pp)', 'Field ECE raw → calibrated', 'T'],
      rows.map((r) => [esc(lab(r.model)), `${p1(r.pv_accuracy, 2)} <span class="muted small">[${p1(r.pv_accuracy_lo)}, ${p1(r.pv_accuracy_hi)}]</span>`, p1(r.pv_macro_f1, 2),
        `${p1(r.field_accuracy, 2)} <span class="muted small">[${p1(r.field_accuracy_lo)}, ${p1(r.field_accuracy_hi)}]</span>`, p1(r.field_macro_f1, 2),
        `${f(r.gap_pp, 2)} <span class="muted small">[${f(r.gap_lo, 1)}, ${f(r.gap_hi, 1)}]</span>`, `${p1(r.field_ece_raw, 2)} → ${p1(r.field_ece_calibrated, 2)}`, f(r.temperature, 2)]));
    return card('RQ1 · Lab vs field accuracy', 'Test-set accuracy per model (bootstrap 95% CIs in the table). The drop from lab to field is the domain gap the field fine-tuning reduces.', chart, table);
  }

  function sectionScorecard(d) {
    const sc = d.scorecard; if (!sc || !sc.criteria || !sc.criteria.length) return '';
    const models = sc.models; const n = models.length;
    const step = (r) => Math.max(1, Math.min(7, 1 + Math.round(((n - r) / Math.max(1, n - 1)) * 6)));
    const fmtVal = (name, v) => (/AUC|AUROC|Leaf-Focus|LFS/i.test(name) ? f(v, 3) : /gap|\(pp\)/i.test(name) ? f(v, 2) : isPct(v) ? p1(v, 2) : f(v, 2));
    const rows = sc.criteria.map((c) => ({ label: `${c.name} ${c.direction > 0 ? '↑' : '↓'}`, cells: models.map((m) => {
      const v = c.values[m]; const r = c.ranks[m];
      return { v: r ?? null, text: v === undefined ? '—' : `${fmtVal(c.name, v)} · #${r}`, tip: `${v === undefined ? '—' : fmtVal(c.name, v)} (rank ${r ?? '—'} of ${Object.keys(c.values).length})\n${lab(m)} · ${c.name}` };
    }) }));
    rows.push({ label: 'Mean rank (lower = better)', highlight: true, cells: models.map((m) => ({ v: null, text: f(sc.mean_rank[m], 2), tip: `${f(sc.mean_rank[m], 2)}\n${lab(m)} · mean rank` })) });
    rows.push({ label: 'First places', highlight: true, cells: models.map((m) => ({ v: null, text: String((sc.first_places || {})[m] ?? 0), tip: `${(sc.first_places || {})[m] ?? 0} criteria won\n${lab(m)}` })) });
    const heatHtml = heat(models.map(lab), rows, (r) => step(r), { corner: 'Criterion (↑ higher / ↓ lower is better)' });
    let sig = '';
    if ((sc.significance || []).length) {
      const div = divbars(sc.significance.map((s) => ({ label: `vs ${lab(s.vs)}`, value: s.diff_pp, tip: `${pp(s.diff_pp)} field macro-F1\nvs ${lab(s.vs)} · 95% CI ${pp(s.ci95_pp?.[0], 1)} to ${pp(s.ci95_pp?.[1], 1)} · McNemar p ${pval(s.mcnemar_p)}` })), { width: W(), aria: 'Hybrid minus each backbone, field macro-F1' });
      sig = `<h3 class="viz-h3">Hybrid minus each single backbone · Field-Test macro-F1</h3>${div}${tbl(['Versus', 'Difference', '95% CI', 'McNemar p', 'Significant'],
        sc.significance.map((s) => [esc(lab(s.vs)), pp(s.diff_pp), `${pp(s.ci95_pp?.[0], 1)} to ${pp(s.ci95_pp?.[1], 1)}`, pval(s.mcnemar_p), statusChip(s.significant ? 'YES' : 'PARTLY', s.significant ? 'yes' : 'no')]))}`;
    }
    let seeds = '';
    if ((sc.seeds || []).length) {
      seeds = `<h3 class="viz-h3">Repeated training (extra seeds)</h3>${tbl(['Model', 'Field macro-F1 per seed', 'Mean ± SD'], sc.seeds.map((s) => [esc(lab(s.model)), (s.per_seed || []).map((v) => p1(v, 2)).join(' · '), `${p1(s.mean, 2)} ± ${p1(s.std, 2)}`]))}
        ${sc.seed_ttest ? `<p class="small muted">Paired t-test across seeds vs ${esc(lab(sc.seed_ttest.vs))}: p = ${pval(sc.seed_ttest.p_value)}</p>` : ''}`;
    }
    return card('Is the hybrid the best model? · Evidence scorecard', 'Every model ranked on every criterion the notebook measured (the stronger the colour stands out from the page, the better the rank; ties share a rank). Cost (size, latency) is shown separately and not counted as “better”.', heatHtml + sig + seeds);
  }

  function sectionPerClass(d) {
    const rows = (d.per_class || []).filter((r) => /field/i.test(r.split || '')); if (!rows.length) return '';
    const rivalKey = Object.keys(rows[0]).find((k) => k.endsWith('_f1') && k !== 'hybrid_f1');
    const rival = rivalKey ? rivalKey.replace(/_f1$/, '') : 'best backbone';
    const ch = divbars(rows.map((r) => ({ label: pretty(r.class), value: r.diff_pp, tip: `${pp(r.diff_pp, 1)}\n${pretty(r.class)} · hybrid ${f(r.hybrid_f1, 1)} vs ${lab(rival)} ${f(r[rivalKey], 1)}` })), { width: W(), aria: 'Per-class F1 difference' });
    const better = rows.filter((r) => num(r.diff_pp) > 0).length;
    return card(`Per-class F1 · hybrid vs ${lab(rival)} (field test)`, `Positive = the hybrid is better on that disease. Better on ${better} of ${rows.length} classes.`, ch,
      tbl(['Class', 'Hybrid F1', `${lab(rival)} F1`, 'Difference'], rows.map((r) => [esc(pretty(r.class)), f(r.hybrid_f1, 2), f(r[rivalKey], 2), pp(r.diff_pp, 2)])));
  }

  function sectionRobust(d) {
    const rows = d.robustness || []; if (!rows.length) return '';
    const cols = Object.keys(rows[0]).filter((k) => !['model', 'mCA', 'relative_robustness'].includes(k));
    const all = rows.flatMap((r) => cols.map((c) => num(r[c]))).filter((v) => v !== null);
    const st = linStep(Math.min(...all), Math.max(...all));
    const hm = heat(cols, rows.map((r) => ({ label: lab(r.model), highlight: r.model === HYB(), cells: cols.map((c) => ({ v: r[c], text: f(r[c], 1), tip: `${f(r[c], 2)}% accuracy\n${lab(r.model)} · ${c}` })) })), st, { corner: 'Field-Test accuracy (%)' });
    const mca = hbars(rows.slice().sort((a, b) => b.mCA - a.mCA).map((r) => ({ label: lab(r.model), value: r.mCA, highlight: r.model === HYB(), tip: `${f(r.mCA, 2)}% mean corruption accuracy\n${lab(r.model)} · relative robustness ${f(r.relative_robustness, 3)}` })),
      { width: W(), fmt: (v) => `${f(v, 0)}%`, max: 100, aria: 'Mean corruption accuracy' });
    return card('Robustness to field-photo damage', 'Every model on the same Field-Test photos with noise, blur, under/over-exposure, low contrast, heavy JPEG and a hidden patch (Cell 9.6). Stronger colour = higher accuracy.',
      `${hm}<h3 class="viz-h3">Mean corruption accuracy (mCA)</h3>${mca}`, tbl(['Model', 'mCA', 'Relative robustness'], rows.map((r) => [esc(lab(r.model)), `${f(r.mCA, 2)}%`, f(r.relative_robustness, 3)])));
  }

  function sectionXAI(d) {
    const fr = d.faithfulness || []; const lc = d.lfs_by_class || []; let out = '';
    if (fr.length) {
      const sc = scatter(fr.map((r) => ({ x: num(r.deletion_auc), y: num(r.insertion_auc), label: lab(r.model), highlight: r.model === HYB(),
        tip: `${lab(r.model)}\ndeletion AUC ${f(r.deletion_auc, 3)} (random ${f(r.deletion_auc_random, 3)}) · insertion AUC ${f(r.insertion_auc, 3)}` })).filter((p) => p.x !== null && p.y !== null),
      { width: W(), xlabel: 'Deletion AUC (lower = more faithful)', ylabel: 'Insertion AUC (higher = better)', aria: 'Grad-CAM faithfulness per model' });
      out += card('Grad-CAM faithfulness', 'Does the heat-map show what the model really used? Removing the top-ranked pixels first should make the prediction collapse (low deletion AUC); putting them back first should restore it (high insertion AUC). Best = top-left. Sanity check: after randomising the classifier head the map should change (low correlation).',
        sc, tbl(['Model', 'Deletion AUC ↓', 'Random order', 'Gain vs random', 'Insertion AUC ↑', 'Sanity: correlation after randomising ↓'],
          fr.map((r) => [esc(lab(r.model)), f(r.deletion_auc, 3), f(r.deletion_auc_random, 3), f(r.deletion_gain_vs_random, 3), f(r.insertion_auc, 3), f(r.spearman_trained_vs_randomised_head, 3)])));
    }
    if (lc.length) {
      const models = Object.keys(lc[0]).filter((k) => k !== 'cls');
      const all = lc.flatMap((r) => models.map((m) => num(r[m]))).filter((v) => v !== null);
      out += card('Leaf-Focus Score per disease', 'Share of each model’s Grad-CAM attention that falls on the leaf (1.0 = all on the leaf), per class, on PV and field test photos. Stronger colour = more on the leaf.',
        heat(models.map(lab), lc.map((r) => ({ label: pretty(r.cls), cells: models.map((m) => ({ v: r[m], text: f(r[m], 2), tip: `${f(r[m], 3)} Leaf-Focus Score\n${lab(m)} · ${pretty(r.cls)}` })) })), linStep(Math.min(...all), Math.max(...all)), { corner: 'Class' }));
    }
    return out;
  }

  function sectionOOD(d) {
    const o = d.ood || {}; let body = ''; const tbls = [];
    const h = o.heldout || {};
    const tiles = [num(h.gate_auroc) !== null && tile('Whole-gate AUROC', f(h.gate_auroc, 3), 'held-out tomato vs non-tomato'),
      num(h.tomato_accept_rate) !== null && tile('Tomato leaves accepted', p1(h.tomato_accept_rate), `n = ${h.n_tomato ?? '—'}`),
      num(h.non_tomato_reject_rate) !== null && tile('Non-tomato rejected', p1(h.non_tomato_reject_rate), `n = ${h.n_other ?? '—'}`),
      num(h.classifier_acc_accepted) !== null && tile('Accuracy after the gate', p1(h.classifier_acc_accepted), `all photos ${p1(h.classifier_acc_all_tomato)}`)].filter(Boolean);
    if (tiles.length) body += `<div class="stats">${tiles.join('')}</div>`;
    const ps = o.per_source || {};
    if (Object.keys(ps).length) {
      const rows = Object.entries(ps).sort((a, b) => a[1].auroc - b[1].auroc);
      body += `<h3 class="viz-h3">AUROC per kind of non-tomato photo</h3>${hbars(rows.map(([k, v]) => ({ label: k.replace(/_/g, ' '), value: v.auroc, tip: `${f(v.auroc, 3)} AUROC\n${k} · rejected ${p1(v.reject_rate)} · n = ${v.n}` })), { width: W(), min: Math.min(0.5, ...rows.map((r) => r[1].auroc)), max: 1, fmt: (v) => f(v, 2), single: true, aria: 'Gate AUROC per source' })}`;
      tbls.push(tbl(['Source', 'AUROC', 'Rejected', 'n'], rows.map(([k, v]) => [esc(k), f(v.auroc, 3), p1(v.reject_rate), v.n])));
    }
    const st = o.stress || [];
    if (st.length) {
      body += `<h3 class="viz-h3">Stress test · tomato test photos degraded on purpose</h3>${hbars(st.map((r) => ({ label: r.condition, value: num(r['accepted_%']), tip: `${f(r['accepted_%'], 1)}% accepted\n${r.condition} · accuracy on accepted ${r['classifier_acc_on_accepted_%'] == null ? '—' : `${f(r['classifier_acc_on_accepted_%'], 1)}%`} · main reject reason: ${r.top_reject_reason || '—'}` })), { width: W(), max: 100, fmt: (v) => `${f(v, 0)}%`, single: true, aria: 'Share of degraded photos accepted' })}
        <p class="small muted">A good gate keeps accepting usable photos (rotation, mild blur) and asks for a retake on unusable ones (very dark, strong blur) instead of guessing.</p>`;
      tbls.push(tbl(['Condition', 'Accepted', 'Accuracy on accepted', 'Main reject reason', 'n'], st.map((r) => [esc(r.condition), `${f(r['accepted_%'], 1)}%`, r['classifier_acc_on_accepted_%'] == null ? '—' : `${f(r['classifier_acc_on_accepted_%'], 1)}%`, esc(r.top_reject_reason || '—'), r.n])));
    }
    if (!body) return '';
    return card('Tomato-leaf gate (out-of-distribution)', `Calibrated in Cell 9.5${o.feature_model ? ` with ${esc(lab(o.feature_model))} features` : ''}; stress-tested in Cell 9.5b.`, body, tbls.join(''));
  }

  function sectionData(d) {
    const v = d.data_validation; const sg = d.segmentation; let out = '';
    if (v) {
      const pairs = Object.values(v.pairs_by_reason || {}).reduce((a, b) => a + b, 0);
      const tiles = [tile('Duplicate pairs found', (pairs || 0).toLocaleString(), Object.entries(v.pairs_by_reason || {}).map(([k, n]) => `${k}: ${n.toLocaleString()}`).join(' · ')),
        tile('Groups of copies', (v.groups_with_copies ?? 0).toLocaleString(), 'same photo, possibly rotated / re-saved'),
        tile('Files moved by the fix', (v.n_moved ?? 0).toLocaleString(), v.fix_applied ? 'leakage fix applied' : 'report only'),
        tile('Label conflicts', String(v.label_conflicts ?? 0), 'same photo, different labels -> removed'),
        tile('Extra eval copies removed', String(v.dedup_removed ?? 0), 'duplicates inside val / test')];
      const strat = v.stratification || {};
      const rows = Object.entries(strat).filter(([, s]) => s.split_ratio).map(([ds, s]) => ({ label: ds === 'pv' ? 'PlantVillage' : ds === 'field' ? 'Field' : ds,
        parts: [{ name: 'Train', cls: 'b-1', v: s.split_ratio.train }, { name: 'Val', cls: 'b-2', v: s.split_ratio.val }, { name: 'Test', cls: 'b-3', v: s.split_ratio.test }] }));
      const cross = Object.entries(v.cross_split_groups || {}).sort((a, b) => b[1] - a[1]);
      out += card('Data validation · duplicates, leakage, stratification', 'Cell 1.6c checked PlantVillage and field splits (and PV ↔ field) for exact copies, rotated / mirrored copies and visually identical images, moved every group of copies into one split, and then checked whether the class mix is the same in every split.',
        `<div class="stats">${tiles.join('')}</div>${rows.length ? `<h3 class="viz-h3">Final split sizes</h3>${stack100(rows, { width: W(), aria: 'Train / val / test share per dataset' })}` : ''}
        ${Object.keys(strat).length ? tbl(['Dataset', 'Chi-square p (class × split)', 'Largest class-share difference', 'Smallest class in val/test', 'Reading'], Object.entries(strat).map(([ds, s]) => [esc(ds === 'pv' ? 'PlantVillage' : ds), pval(s.chi2_p_value), `${f(s.max_class_share_deviation_pp, 1)} pp`, s.min_per_class_in_eval ?? '—',
          num(s.chi2_p_value) > 0.05 ? statusChip('YES', 'class mix the same in every split') : statusChip('PARTLY', 'class mix differs — report it')])) : ''}`,
        cross.length ? tbl(['Copies found across', 'Groups'], cross.map(([k, n]) => [esc(k), n.toLocaleString()])) : '');
    }
    if (sg && (sg.by_split || []).length) {
      out += card('Background removal quality', 'Cell 2.5b: share of photos segmented reliably, leaf coverage, segmentation confidence and mask stability when the photo is mirrored (flip IoU, 1.0 = identical).',
        tbl(['Split', 'Photos', 'Reliable', 'Leaf coverage', 'Confidence', 'Flip IoU'], sg.by_split.map((r) => [esc(r.split), r.n, p1(r.reliable), p1(r.coverage), f(r.confidence, 2), f(r.flip_iou, 3)])));
    }
    return out;
  }

  function sectionCost(d) {
    const rows = d.latency || []; if (!rows.length) return '';
    const keys = Object.keys(rows[0]).filter((k) => k !== 'model');
    return card('Model size and speed', 'Reported for completeness (Cell 10.3); not counted as “better” in the scorecard.', tbl(['Model', ...keys.map((k) => k.replace(/_/g, ' '))], rows.map((r) => [esc(lab(r.model)), ...keys.map((k) => esc(num(r[k]) === null ? r[k] : f(r[k], 2)))])));
  }

  function sectionFiles(d) {
    const files = d.files || [];
    const missing = files.filter((x) => !x.present);
    return `<details class="card files-card"><summary>Result files used (${files.length - missing.length}/${files.length} found)</summary>
      <ul class="files">${files.map((x) => `<li>${x.present ? statusChip('YES', 'found') : statusChip('NO', 'missing')} <code>${esc(x.file)}</code> <span class="muted small">${esc(x.what)}</span></li>`).join('')}</ul>
      <p class="small muted">Copy the notebook’s <code>results/</code> folder next to the app (or run <code>tools/collect_artifacts.py</code>) — the tab updates by itself.</p></details>`;
  }

  function render() {
    const d = DATA; const root = $('#results-view');
    if (!d.available) {
      root.innerHTML = `<div class="card empty-state"><h2>No thesis results yet</h2><p>Copy the notebook’s <code>results/</code> folder into the app (see README) and this tab fills in by itself: research questions, hypotheses, the hybrid-proof scorecard, robustness, Grad-CAM faithfulness, the leaf gate and data validation.</p></div>${sectionFiles(d)}`;
      return;
    }
    root.innerHTML = [sectionVerdict(d), sectionRQ(d), sectionRQ1(d), sectionScorecard(d), sectionPerClass(d), sectionRobust(d), sectionXAI(d), sectionOOD(d), sectionData(d), sectionCost(d), sectionFiles(d)].join('');
  }

  async function load(force = false) {
    const root = $('#results-view'); if (!root) return;
    if (DATA && !force) { render(); return; }
    root.innerHTML = '<p class="placeholder">Loading results…</p>';
    try {
      const r = await fetch('/api/thesis', { headers: { Accept: 'application/json' } });
      DATA = await r.json(); LABELS = DATA.labels || {};
      if (!r.ok) throw new Error(DATA.error || r.status);
      render();
    } catch (e) { root.innerHTML = `<div class="banner error">Could not load the results (${esc(e.message)}).</div>`; }
  }
  let rt = null;
  addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { if (DATA && $('#tab-results')?.classList.contains('active')) render(); }, 200); });
  let wired = false;
  const wire = () => { const root = $('#results-view'); if (root && !wired) { wired = true; wireTips(root); } };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', wire); else wire();
  window.TLAI_RESULTS = { load, reload: () => load(true) };
  // opened directly on #results (app.js ran first and could not call us yet)
  const kick = () => { if ($('#tab-results')?.classList.contains('active') && !DATA) load(); };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', kick); else kick();
})();
