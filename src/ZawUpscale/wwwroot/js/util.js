// Small DOM and formatting helpers shared by all modules.

const PROPS = new Set(['value', 'checked', 'selected', 'disabled', 'readOnly', 'spellcheck', 'tabIndex', 'multiple', 'draggable']);

export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k === 'html') el.innerHTML = v;
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else if (PROPS.has(k)) el[k] = v;
      else el.setAttribute(k, v === true ? '' : v);
    }
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : String(kid));
  }
  return el;
}

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export const esc = s => String(s).replace(/[&<>"']/g, c => ESC[c]);

export function fmtBytes(n) {
  if (n == null || n === '') return '';
  n = Number(n);
  if (!isFinite(n)) return '';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return (i ? n.toFixed(n < 10 ? 2 : 1) : n) + ' ' + units[i];
}

export function fmtMs(ms) {
  if (ms == null) return '';
  return ms < 1000 ? Math.round(ms) + ' ms' : (ms / 1000).toFixed(2) + ' s';
}

/** "14:03:21" today, "2026-10-05 14:03" earlier. */
export function fmtTime(ms) {
  const d = new Date(ms), now = new Date();
  const p = n => String(n).padStart(2, '0');
  const time = `${p(d.getHours())}:${p(d.getMinutes())}`;
  return d.toDateString() === now.toDateString() ? `${time}:${p(d.getSeconds())}` : `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${time}`;
}

export function debounce(fn, ms) {
  let t;
  const f = (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  f.flush = (...a) => { clearTimeout(t); fn(...a); };
  return f;
}

/** Makes `handle` a drag handle that resizes something along one axis. */
export function makeSplitter(handle, { axis, get, set, onEnd }) {
  handle.addEventListener('mousedown', e => {
    e.preventDefault();
    const start = axis === 'x' ? e.clientX : e.clientY;
    const startVal = get();
    document.body.classList.add(axis === 'x' ? 'resizing-x' : 'resizing-y');
    const move = ev => set(startVal + ((axis === 'x' ? ev.clientX : ev.clientY) - start));
    const up = () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
      document.body.classList.remove('resizing-x', 'resizing-y');
      onEnd?.();
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  });
}

export function pickFile(accept = '') {
  return new Promise(resolve => {
    const input = h('input', { type: 'file', accept, style: { display: 'none' } });
    input.addEventListener('change', () => { resolve(input.files[0] || null); input.remove(); });
    input.addEventListener('cancel', () => { resolve(null); input.remove(); });
    document.body.append(input);
    input.click();
  });
}

/** Saves text through the native save dialog when available, else as a download. Resolves to the name, or null if cancelled. */
export async function saveTextFile(name, text, description = 'JSON files', ext = '.json', type = 'application/json') {
  if (window.showSaveFilePicker) {
    try {
      const fh = await window.showSaveFilePicker({ suggestedName: name, types: [{ description, accept: { [type]: [ext] } }] });
      const w = await fh.createWritable();
      await w.write(text);
      await w.close();
      return fh.name;
    } catch (e) {
      if (e.name === 'AbortError') return null;
    }
  }
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h('a', { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  return name;
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = h('textarea', { style: { position: 'fixed', opacity: '0' } }, text);
    document.body.append(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
  }
}

export const uid = () => (crypto.randomUUID ? crypto.randomUUID().replace(/-/g, '') : String(Math.random()).slice(2) + Date.now()).slice(0, 16);

export const clone = o => (o == null ? o : JSON.parse(JSON.stringify(o)));

/** CSS class for a method badge (GET, POST, …). */
export function methodClass(m) {
  m = String(m || 'GET').toUpperCase();
  return 'm-' + (['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS'].includes(m) ? m.toLowerCase() : 'other');
}

/** Short method label for narrow places (DELETE → DEL, OPTIONS → OPT). */
export const methodShort = m => ({ DELETE: 'DEL', OPTIONS: 'OPT', PATCH: 'PATCH' })[String(m).toUpperCase()] || String(m || 'GET').toUpperCase();

export function statusClass(s) {
  if (!s) return 's-err';
  return s < 200 ? 's-1' : s < 300 ? 's-2' : s < 400 ? 's-3' : s < 500 ? 's-4' : 's-5';
}
