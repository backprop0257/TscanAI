/* TomatoLeafAI -- sends an anonymous visitor id with every request to this server (online, each visitor
   only sees their own history; the id is random and stays in this browser), plus the admin token for
   model management when one was entered. Loaded before app.js. */
(() => {
  'use strict';
  const KEY = 'tlai_vid';
  let vid = null;
  try { vid = localStorage.getItem(KEY); } catch { /* storage blocked */ }
  if (!/^[0-9a-f]{32}$/.test(vid || '')) {
    const b = new Uint8Array(16); crypto.getRandomValues(b);
    vid = Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
    try { localStorage.setItem(KEY, vid); } catch { /* cookie fallback on the server */ }
  }
  const adminToken = () => { try { return sessionStorage.getItem('tlai_admin') || ''; } catch { return ''; } };
  const orig = window.fetch.bind(window);
  window.fetch = (input, init = {}) => {
    const url = typeof input === 'string' ? input : input.url;
    const same = url.startsWith('/') || url.startsWith(location.origin);
    if (!same) return orig(input, init);
    const headers = new Headers(init.headers || (typeof input !== 'string' ? input.headers : undefined));
    headers.set('X-Visitor-Id', vid);
    if (/\/api\/models\//.test(url) && adminToken()) headers.set('X-Admin-Token', adminToken());
    return orig(input, { ...init, headers });
  };
  /** Download a same-origin file through fetch (so the visitor id is sent), keeping the server's file name. */
  window.tlaiDownload = async (url, fallbackName) => {
    const r = await window.fetch(url);
    if (!r.ok) { let m = `Download failed (${r.status})`; try { m = (await r.json()).error || m; } catch { /* not json */ } throw new Error(m); }
    const blob = await r.blob();
    const cd = r.headers.get('Content-Disposition') || '';
    const name = (cd.match(/filename="?([^"]+)"?/) || [])[1] || fallbackName || 'download';
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  };
  window.tlaiSetAdmin = (tok) => { try { if (tok) sessionStorage.setItem('tlai_admin', tok); else sessionStorage.removeItem('tlai_admin'); } catch { /* ignore */ } };
  window.tlaiIsAdmin = () => !!adminToken();
})();
