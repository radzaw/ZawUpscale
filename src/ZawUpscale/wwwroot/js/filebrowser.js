// File and folder pickers. In the native window these are the system's own dialogs; in a browser window the page
// can't learn local paths from the browser's dialogs, so a picker backed by the local backend stands in.
// Modes: 'images' (one or more images), 'folder', 'onnx'.
import { get, post, urlWithToken } from './api.js';
import { h, fmtBytes } from './util.js';
import { icon } from './icons.js';
import { modal } from './dialogs.js';

let placesCache = null;
let native = false;

/** Called at start-up with whether the native open dialogs are available. */
export function useNativeDialogs(on) { native = !!on; }

/**
 * Resolves with an array of paths ('images', 'onnx') or a folder path ('folder'), or null when cancelled.
 * `start` is the folder to open in; `onDir` is told about every folder visited (to remember the last one).
 */
export async function browse(opts = {}) {
  if (native) {
    const { mode = 'images', title, start, onDir, multiple = true } = opts;
    const r = await post('/dialog/open', { mode, title, start, multiple });
    if (!r) return null;
    onDir?.(r.dir);
    return mode === 'folder' ? r.paths[0] : r.paths;
  }
  return browseHere(opts);
}

async function browseHere({ mode = 'images', title, start, onDir, multiple = true } = {}) {
  const places = placesCache ??= await get('/fs/places').catch(() => []);
  let cur = null, entries = [], selected = new Set(), anchor = -1;
  const pathInp = h('input', { class: 'inp no-enter', spellcheck: false, placeholder: 'Folder path' });
  const list = h('div', { class: 'fb-list', tabIndex: 0 });
  const foot = h('div', { class: 'fb-foot' });
  const placeEls = [];
  const placesEl = h('div', { class: 'fb-places' });
  let lastKind = null;
  for (const p of places) {
    if (lastKind && (p.kind === 'drive') !== (lastKind === 'drive')) placesEl.append(h('div', { class: 'fb-place-sep' }));
    lastKind = p.kind;
    const el = h('div', { class: 'fb-place', title: p.path, html: icon(p.kind) + `<span></span>`, onclick: () => go(p.path) });
    el.lastChild.textContent = p.name;
    el.dataset.path = p.path;
    placeEls.push(el);
    placesEl.append(el);
  }
  const upBtn = h('button', { class: 'iconbtn', title: 'Up one folder (Backspace)', html: icon('up'), onclick: () => cur?.parent && go(cur.parent) });
  const refreshBtn = h('button', { class: 'iconbtn', title: 'Refresh', html: icon('refresh'), onclick: () => cur && go(cur.path) });
  pathInp.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); go(pathInp.value); } });

  let ctx;
  const isPickable = en => !en.dir;

  async function go(path) {
    try {
      const r = await get('/fs/list', { path, models: mode === 'onnx' ? 'true' : '' });
      cur = r;
      entries = r.entries;
      selected = new Set();
      anchor = -1;
      pathInp.value = r.path;
      upBtn.disabled = !r.parent;
      onDir?.(r.path);
      for (const el of placeEls) el.classList.toggle('on', el.dataset.path.replace(/[\\/]+$/, '') === r.path.replace(/[\\/]+$/, ''));
      render();
      list.scrollTop = 0;
    } catch (e) {
      foot.replaceChildren(h('span', { html: icon('error') }), h('span', null, e.message));
    }
  }

  function render() {
    const rows = entries.map((en, i) => {
      const ic = en.dir
        ? h('span', { class: 'fb-ic dir', html: icon('folder') })
        : mode === 'images'
          ? h('span', { class: 'fb-ic' }, h('img', { loading: 'lazy', alt: '', src: urlWithToken('/fs/thumb', { path: en.path, size: 64 }) }))
          : h('span', { class: 'fb-ic', html: icon('model') });
      const row = h('div', { class: 'fb-row' + (selected.has(i) ? ' sel' : '') }, ic, h('span', { class: 'fb-name' }, en.name), h('span', { class: 'fb-size' }, en.dir ? '' : fmtBytes(en.size)));
      row.addEventListener('mousedown', e => select(i, e));
      row.addEventListener('dblclick', () => {
        if (en.dir) go(en.path);
        else if (mode !== 'folder') ctx.close([en.path]);
      });
      return row;
    });
    if (!rows.length) rows.push(h('div', { class: 'placeholder' }, mode === 'folder' ? 'No subfolders.' : mode === 'onnx' ? 'No folders or .onnx files here.' : 'No folders or images here.'));
    list.replaceChildren(...rows);
    updateFoot();
  }

  function select(i, e) {
    const en = entries[i];
    if (mode === 'folder' || !multiple || en.dir || !(e.ctrlKey || e.metaKey || e.shiftKey)) {
      selected = new Set([i]);
      anchor = i;
    } else if (e.shiftKey && anchor >= 0) {
      selected = new Set();
      for (let k = Math.min(anchor, i); k <= Math.max(anchor, i); k++) if (isPickable(entries[k])) selected.add(k);
    } else {
      if (selected.has(i)) selected.delete(i); else selected.add(i);
      for (const k of [...selected]) if (entries[k].dir) selected.delete(k);
      anchor = i;
    }
    for (const [k, row] of [...list.children].entries()) row.classList?.toggle('sel', selected.has(k));
    updateFoot();
  }

  function picked() {
    const files = [...selected].map(i => entries[i]).filter(isPickable);
    if (mode === 'folder') {
      const d = [...selected].map(i => entries[i]).find(en => en.dir);
      return d ? d.path : cur?.path;
    }
    return files.length ? files.map(f => f.path) : null;
  }

  function updateFoot() {
    const p = picked();
    if (mode === 'folder') foot.replaceChildren(h('span', null, 'Folder:'), h('b', null, p || ''));
    else if (!p) foot.replaceChildren(h('span', null, mode === 'images'
      ? (multiple ? 'Select one or more images (Ctrl or Shift + click for several). Double-click a folder to open it.' : 'Select an image.')
      : 'Select an .onnx model file.'));
    else foot.replaceChildren(h('b', null, p.length === 1 ? entries[[...selected][0]].name : `${p.length} images selected`));
    if (ctx) ctx.dialog.querySelector('.btn.primary').disabled = !p;
  }

  list.addEventListener('keydown', e => {
    if (e.key === 'Backspace' && cur?.parent) { e.preventDefault(); go(cur.parent); return; }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const i = Math.max(0, Math.min(entries.length - 1, anchor + (e.key === 'ArrowDown' ? 1 : -1)));
      select(i, {});
      list.children[i]?.scrollIntoView({ block: 'nearest' });
    } else if (e.key === 'Enter' && anchor >= 0 && entries[anchor]?.dir && mode !== 'folder') {
      e.preventDefault();
      e.stopPropagation();
      go(entries[anchor].path);
    }
  });

  const body = h('div', { class: 'fb' }, placesEl, h('div', { class: 'fb-main' }, h('div', { class: 'fb-bar' }, upBtn, pathInp, refreshBtn), list, foot));
  return modal({
    title: title || (mode === 'folder' ? 'Choose a folder' : mode === 'onnx' ? 'Choose an ONNX model' : 'Open images'),
    width: 820,
    className: 'fb-dialog',
    body,
    buttons: [
      { label: mode === 'folder' ? 'Select folder' : 'Open', primary: true, onClick: () => picked() ?? false },
      { label: 'Cancel', value: null },
    ],
    onOpen: c => {
      ctx = c;
      go(start || places[0]?.path || '/').then(() => list.focus());
    },
  });
}
