// ZawUpscale main module: layout, menus, the settings panel, the stage (viewer / batch grid), jobs, drop and paste.
import { get, post, put, startHeartbeat, token, urlWithToken } from './api.js';
import { h, debounce, makeSplitter, fmtBytes, fmtMs } from './util.js';
import { icon } from './icons.js';
import { contextMenu, closeMenus, alertError, confirmDlg } from './dialogs.js';
import { Viewer } from './viewer.js';
import { browse, useNativeDialogs } from './filebrowser.js';
import { modelsDialog, preferencesDialog, shortcutsDialog, aboutDialog } from './tools.js';

const DEFAULT_PREFS = { theme: 'system', panelWidth: 340, viewMode: 'slider', pixelated: true };
const DEFAULT_SETTINGS = {
  model: 'realesr-general-x4v3', scale: 4, format: 'auto', quality: 92, outputDir: '', nameTemplate: '{name}',
  existing: 'skip', tile: 0, device: 'auto', supir: {},
  faces: '', faceModel: 'gfpgan-v1.4', faceStrength: 0.8, // faces: the face model in use, or '' (off); faceModel: the last one chosen
  size: 'scale', width: 3840, height: 2160, longEdge: 4000, // size: 'scale', or a target size: 'width', 'height', 'fit' or 'long'
  detail: 1, sharpen: 0, keepColors: false, half: false,
};
// Running in the native window (Photino injects window.external.sendMessage) rather than a browser.
const NATIVE = typeof window.external?.sendMessage === 'function';
const native = msg => { if (NATIVE) window.external.sendMessage(msg); };
const SCALES = [2, 3, 4, 8];
const FORMATS = [['auto', 'Same as the original'], ['png', 'PNG'], ['jpg', 'JPEG'], ['webp', 'WebP'], ['avif', 'AVIF'], ['tif', 'TIFF'], ['jxl', 'JPEG XL']];
const SIZE_MODES = [['width', 'Width'], ['height', 'Height'], ['fit', 'Fit inside'], ['long', 'Long edge']];
// Target sizes: screens, and prints at 300 dpi (portrait; the box turns to match each image).
const SIZE_PRESETS = [
  ['Full HD screen', 1920, 1080], ['4K UHD screen', 3840, 2160], ['5K screen', 5120, 2880], ['8K screen', 7680, 4320],
  ['Print 10 × 15 cm, 300 dpi', 1181, 1772], ['Print 13 × 18 cm, 300 dpi', 1535, 2126], ['Print 20 × 30 cm, 300 dpi', 2362, 3543],
  ['Print A4, 300 dpi', 2480, 3508], ['Print A3, 300 dpi', 3508, 4961], ['Print US Letter, 300 dpi', 2550, 3300],
];
const UPLOAD_BATCH = 16;

const baseName = p => p.split(/[\\/]/).filter(Boolean).pop() || p;
const dirName = p => p.replace(/[\\/][^\\/]*$/, '');
const fmtScale = s => String(Number(Number(s).toFixed(2)));
const imageUrl = (path, v) => urlWithToken('/fs/image', { path, v });
const thumbUrl = (path, size, v) => urlWithToken('/fs/thumb', { path, size, v });
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;

// "About 3 minutes left", from the working time so far and the fraction done; nothing until there's enough to go on.
function eta(activeMs, frac) {
  if (!activeMs || frac < 0.02 || activeMs < 3000) return 'Working out the time left…';
  const left = activeMs * (1 - frac) / frac / 1000;
  if (left < 10) return 'Almost done';
  if (left < 60) return `About ${Math.ceil(left / 10) * 10} seconds left`;
  if (left < 3600) return `About ${plural(Math.round(left / 60), 'minute')} left`;
  return `About ${(left / 3600).toFixed(1)} hours left`;
}
// Models that upscale (face restoration models only run after one).
const isUpscaler = m => m.kind !== 'face';

// As the server's Jobs.FormatFor (an animated GIF stays a GIF, which only the server can tell).
function formatFor(path, format) {
  if (format !== 'auto') return format;
  const ext = (path.match(/\.([^.\\/]+)$/)?.[1] || '').toLowerCase();
  return { jpg: 'jpg', jpeg: 'jpg', jfif: 'jpg', heic: 'jpg', heif: 'jpg', webp: 'webp', avif: 'avif', tif: 'tif', tiff: 'tif', jxl: 'jxl' }[ext] || 'png';
}

// As the server's Jobs.SizeLabel / ScaleFor.
const sizeLabel = s => ({ width: `w${s.width}`, height: `h${s.height}`, fit: `${s.width}x${s.height}`, long: `long${s.longEdge}` })[s.size] || `x${fmtScale(s.scale)}`;
const scaleToken = s => (s.size && s.size !== 'scale' ? sizeLabel(s) : fmtScale(s.scale));
function scaleFor(s, w, h) {
  switch (s.size) {
    case 'width': return s.width / w;
    case 'height': return s.height / h;
    case 'long': return s.longEdge / Math.max(w, h);
    case 'fit': {
      const [bw, bh] = (w >= h) === (s.width >= s.height) ? [s.width, s.height] : [s.height, s.width];
      return Math.min(bw / w, bh / h);
    }
    default: return Number(s.scale);
  }
}

class App {
  constructor() {
    this.saveStateSoon = debounce(() => this.saveState(), 600);
    this.source = null;
    this.job = null;
    this.results = new Map(); // input path -> finished job item
    this.focus = null; // batch item shown in the viewer
  }

  async init() {
    try {
      const [state, system, models] = await Promise.all([get('/doc/state'), get('/system'), get('/models')]);
      this.state = state || {};
      this.system = system;
      useNativeDialogs(NATIVE);
      this.models = models;
    } catch (e) {
      document.body.textContent = e.message;
      return;
    }
    this.prefs = this.state.prefs = { ...DEFAULT_PREFS, ...this.state.prefs };
    this.settings = this.state.settings = { ...DEFAULT_SETTINGS, ...this.state.settings };
    this.settings.supir = { ...this.settings.supir };
    // Version 2: results keep the original's name (they go in a {model}_x{scale} folder); drop the old default pattern.
    if ((this.state.version || 1) < 2 && this.settings.nameTemplate === '{name}_x{scale}') this.settings.nameTemplate = '{name}';
    this.state.version = 2;
    if (!this.system.devices.some(d => d.id === this.settings.device)) this.settings.device = 'auto';
    this.applyPrefs();
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.applyPrefs());
    try { this.version = await get('/version'); } catch { this.version = null; }

    this.viewer = new Viewer({ onZoom: z => this.zoomChanged(z) });
    this.viewer.pixelated = this.prefs.pixelated;
    this.buildMenu();
    this.buildToolbar();
    this.buildPanel();
    this.buildStage();
    this.buildStatusbar();
    this.buildLayout();
    this.setupDrop();
    this.applyPrefs();
    if (!this.models.some(m => m.id === this.settings.model && m.installed && isUpscaler(m))) {
      const first = this.models.find(m => m.installed && isUpscaler(m));
      if (first) this.settings.model = first.id;
    }
    this.renderAll();
    if (this.models.some(m => m.download && !m.download.done)) this.pollModels();

    document.addEventListener('keydown', e => this.globalKey(e));
    document.addEventListener('paste', e => this.paste(e));
    window.addEventListener('beforeunload', () => this.saveStateSoon.flush());
    startHeartbeat();
    this.status('Ready.');
  }

  applyPrefs() {
    const p = this.prefs;
    const dark = p.theme === 'dark' || (p.theme === 'system' && matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    document.querySelector('meta[name=theme-color]').content = dark ? '#14161b' : '#f3f4f7';
    try { sessionStorage.setItem('zawupscale-theme', p.theme); } catch { /* storage unavailable */ }
    if (this.themeBtn) this.themeBtn.innerHTML = icon(dark ? 'sun' : 'moon');
    if (this.viewer) { this.viewer.pixelated = p.pixelated; this.viewer.layout(); }
    if (NATIVE) window.external.sendMessage(dark ? 'theme:dark' : 'theme:light'); // title bar to match
  }

  saveState() {
    put('/doc/state', this.state).catch(e => this.status('Could not save settings: ' + e.message));
  }

  settingsChanged() {
    this.saveStateSoon();
    this.renderAll();
  }

  status(msg) {
    if (this.sbMsg) this.sbMsg.textContent = msg;
  }

  // ------------------------------------------------------------------ chrome

  buildMenu() {
    const bar = document.getElementById('menubar');
    const hasResult = () => !!this.currentResult();
    const menus = [
      ['File', () => [
        { label: 'Open images…', icon: 'image', shortcut: 'Ctrl+O', onClick: () => this.openImages() },
        { label: 'Open folder…', icon: 'folderopen', shortcut: 'Ctrl+Shift+O', onClick: () => this.openFolder() },
        { label: 'Paste image', icon: 'paste', shortcut: 'Ctrl+V', onClick: () => this.pasteFromMenu() },
        '-',
        { label: 'Show result in folder', icon: 'external', shortcut: 'Ctrl+Shift+E', disabled: !hasResult(), onClick: () => this.revealResult() },
        { label: 'Open result', disabled: !hasResult(), onClick: () => this.openResult() },
        '-',
        { label: 'Close', icon: 'close', disabled: !this.source || this.running(), onClick: () => this.clearSource() },
        { label: 'Exit', onClick: () => this.exit() },
      ]],
      ['View', () => [
        { label: 'Compare slider', checked: this.prefs.viewMode === 'slider', shortcut: 'C', onClick: () => this.setViewMode('slider') },
        { label: 'Side by side', checked: this.prefs.viewMode === 'side', shortcut: 'S', onClick: () => this.setViewMode('side') },
        { label: 'Original', checked: this.prefs.viewMode === 'before', shortcut: 'B', onClick: () => this.setViewMode('before') },
        { label: 'Result', checked: this.prefs.viewMode === 'after', shortcut: 'A', onClick: () => this.setViewMode('after') },
        '-',
        { label: 'Zoom in', icon: 'zoomin', shortcut: '+', onClick: () => this.viewer.zoomBy(1.25) },
        { label: 'Zoom out', icon: 'zoomout', shortcut: '-', onClick: () => this.viewer.zoomBy(0.8) },
        { label: 'Fit to window', icon: 'fit', shortcut: 'F', onClick: () => this.viewer.fitView() },
        { label: 'Actual size', shortcut: '1', onClick: () => this.viewer.zoom100() },
      ]],
      ['Tools', () => [
        { label: 'Models…', icon: 'model', shortcut: 'Ctrl+M', onClick: () => modelsDialog(this) },
        { label: 'Preferences…', icon: 'settings', onClick: () => preferencesDialog(this) },
      ]],
      ['Help', () => [
        { label: 'Keyboard shortcuts', onClick: () => shortcutsDialog() },
        { label: 'About ZawUpscale', icon: 'info', onClick: () => aboutDialog(this) },
      ]],
    ];
    let openEl = null;
    for (const [label, items] of menus) {
      const el = h('div', { class: 'menubar-item' }, label);
      const open = () => {
        const r = el.getBoundingClientRect();
        el.classList.add('open');
        openEl = el;
        contextMenu(r.left, r.bottom, items(), { onClose: () => { el.classList.remove('open'); if (openEl === el) openEl = null; } });
      };
      el.addEventListener('mousedown', e => {
        e.preventDefault();
        if (el.classList.contains('open')) closeMenus(); else open();
      });
      el.addEventListener('mouseenter', () => { if (openEl && openEl !== el) { closeMenus(); open(); } });
      bar.append(el);
    }
  }

  buildToolbar() {
    const tb = document.getElementById('toolbar');
    const btn = (ic, title, onclick, label, cls = '') => h('button', { class: 'tbtn ' + cls, title, onclick, html: icon(ic) + (label ? `<span>${label}</span>` : '') });
    this.themeBtn = btn('moon', 'Switch between light and dark', () => {
      this.prefs.theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      this.applyPrefs();
      this.saveStateSoon();
    });
    this.devicePill = h('button', { class: 'tbtn device-pill', title: 'Processing device — click to change', onclick: () => preferencesDialog(this) });
    tb.append(
      btn('image', 'Open images (Ctrl+O)', () => this.openImages(), 'Open images', 'accent'),
      btn('folderopen', 'Open a folder (Ctrl+Shift+O)', () => this.openFolder(), 'Open folder'),
      h('span', { class: 'grow' }),
      this.devicePill,
      btn('model', 'Models (Ctrl+M)', () => modelsDialog(this), 'Models'),
      this.themeBtn);
  }

  buildLayout() {
    const panel = document.getElementById('panel');
    panel.style.width = this.prefs.panelWidth + 'px';
    makeSplitter(document.getElementById('split-left'), {
      axis: 'x',
      get: () => panel.offsetWidth,
      set: v => { panel.style.width = Math.max(290, Math.min(v, innerWidth - 420)) + 'px'; },
      onEnd: () => { this.prefs.panelWidth = panel.offsetWidth; this.saveStateSoon(); },
    });
  }

  buildStatusbar() {
    this.sbMsg = h('div', { class: 'sb-cell grow' });
    this.sbDevice = h('div', { class: 'sb-cell' });
    document.getElementById('statusbar').append(this.sbMsg, this.sbDevice, h('div', { class: 'sb-cell' }, 'ZawUpscale ' + (this.version?.version ?? '')));
  }

  renderDevice() {
    const d = this.system.devices.find(x => x.id === this.settings.device);
    const active = this.system.activeDevice;
    const name = active ? active.replace(/ \((DirectML|CUDA|CoreML)\)$/, '') : d?.id === 'auto' ? (d.detail?.split(',')[0] || 'Automatic') : d?.name;
    this.devicePill.innerHTML = icon('chip') + '<span></span>';
    this.devicePill.lastChild.textContent = name || 'Automatic';
    this.devicePill.title = `Processing on ${name || 'the automatic choice'}. Click to change.`;
    this.sbDevice.textContent = active ? `Running on ${active}` : '';
  }

  // ------------------------------------------------------------------ panel

  buildPanel() {
    const panel = document.getElementById('panel');
    this.secSource = h('div');
    this.secModel = h('div');
    this.secScale = h('div');
    this.secLook = h('div');
    this.secOutput = h('div');
    this.foot = h('div', { class: 'panel-foot' });
    panel.append(h('div', { class: 'panel-scroll' }, this.secSource, this.secModel, this.secScale, this.secLook, this.secOutput), this.foot);
    this.buildOutput();
  }

  renderAll() {
    this.renderSource();
    this.renderModels();
    this.renderScale();
    this.renderLook();
    this.renderOutputExtras();
    this.renderFoot();
    this.renderStage();
    this.renderDevice();
  }

  title(step, text, aside) {
    return h('div', { class: 'sec-title' }, h('span', { class: 'step' }, step), text, aside ? h('span', { class: 'aside' }, aside) : null);
  }

  renderSource() {
    const s = this.source;
    let box;
    if (!s) {
      box = h('div', { class: 'source' }, h('div', { class: 'source-empty' },
        h('div', { class: 'btns' },
          h('button', { class: 'btn', onclick: () => this.openImages(), html: icon('image') + '<span>Images…</span>' }),
          h('button', { class: 'btn', onclick: () => this.openFolder(), html: icon('folderopen') + '<span>Folder…</span>' })),
        h('div', { class: 'hint' }, 'or drop images or a folder anywhere, or paste with Ctrl+V')));
    } else {
      const close = h('button', { class: 'iconbtn', title: 'Close', html: icon('close'), disabled: this.running(), onclick: () => this.clearSource() });
      if (s.kind === 'image') {
        box = h('div', { class: 'source has' }, h('div', { class: 'source-card' },
          h('img', { class: 'source-thumb', alt: '', src: thumbUrl(s.path, 112) }),
          h('div', { class: 'source-info' },
            h('div', { class: 'source-name', title: s.path }, s.name),
            h('div', { class: 'source-meta' }, [s.width ? `${s.width} × ${s.height}` : null, fmtBytes(s.size)].filter(Boolean).join(' · ')),
            h('div', { class: 'source-meta', title: s.path }, dirName(s.path))),
          close));
      } else {
        const opts = s.folder ? h('div', { class: 'source-opts' }, h('label', { class: 'chk' },
          h('input', { type: 'checkbox', checked: !!this.state.recursive, disabled: this.running(), onchange: e => { this.state.recursive = e.target.checked; this.saveStateSoon(); this.rescan(); } }),
          'Include subfolders')) : null;
        box = h('div', { class: 'source has' }, h('div', { class: 'source-card' },
          h('div', { class: 'source-thumb', html: icon(s.folder ? 'folder' : 'images') }),
          h('div', { class: 'source-info' },
            h('div', { class: 'source-name', title: s.folder || '' }, s.label),
            h('div', { class: 'source-meta' }, `${plural(s.items.length, 'image')} · ${fmtBytes(s.bytes)}`),
            s.previous ? h('div', { class: 'source-meta', title: 'Earlier results (in model folders, or named like results of other images here) are not upscaled again' }, `${plural(s.previous, 'earlier result')} left out`) : null,
            s.folder ? h('div', { class: 'source-meta', title: s.folder }, dirName(s.folder)) : null),
          close), opts);
      }
    }
    this.secSource.replaceChildren(this.title(1, s?.kind === 'batch' ? 'Images' : 'Image'), box);
  }

  // The download button of a model that isn't installed, or its progress while it downloads.
  modelDownload(m) {
    const d = m.download;
    if (d && !d.done) {
      const f = d.total ? d.received / d.total : 0;
      return h('div', { class: 'mdl' }, h('div', { class: 'bar' }, h('i', { style: { width: f * 100 + '%' } })), h('span', { class: 'muted num', style: { fontSize: '12px' } }, `${Math.round(f * 100)}%`));
    }
    const half = this.wantsHalf(m);
    return h('div', { class: 'mdl' },
      h('button', { class: 'btn small', html: icon('download') + `<span>Download · ${fmtBytes(half ? m.halfSize : m.size)}${half ? ' (float16)' : ''}</span>`, onclick: e => { e.stopPropagation(); e.preventDefault(); this.downloadModel(m.id, half); } }),
      d?.error ? h('span', { class: 'err', title: d.error }, 'Failed') : null);
  }

  // Whether to fetch a model's float16 version: chosen in Preferences, the model has one, and the device isn't the CPU.
  wantsHalf(m) {
    return !!(this.settings.half && m.halfSize && this.settings.device !== 'cpu' && this.system.gpuSupport);
  }

  renderModels() {
    const list = this.models;
    // The main models always; the rest of the catalog once downloaded (all of them are in Models…).
    const cards = list.filter(m => isUpscaler(m) && (m.featured || m.installed)).map(m => {
      const on = m.installed && this.settings.model === m.id;
      const extra = m.installed ? null : this.modelDownload(m);
      const card = h('label', { class: 'mcard' + (on ? ' on' : '') + (m.installed ? '' : ' missing') },
        h('input', { type: 'radio', name: 'model', checked: on, disabled: !m.installed || this.running(), onchange: () => this.selectModel(m.id) }),
        h('div', { class: 'mbody' }, h('div', { class: 'mname' }, m.name, m.kind === 'supir' ? h('span', { class: 'tag' }, 'SUPIR') : null),
          h('div', { class: 'mdesc' }, m.description), extra));
      return card;
    });
    const sel = list.find(m => m.installed && m.id === this.settings.model);
    this.secModel.replaceChildren(
      this.title(2, 'AI model', h('button', { class: 'linkbtn', onclick: () => modelsDialog(this) }, 'Manage…')),
      h('div', { class: 'models' }, cards),
      sel?.kind === 'supir' ? this.supirOptions(sel) : this.facesOptions());
  }

  // Face restoration after upscaling (not with SUPIR, which rebuilds faces itself).
  facesOptions() {
    const s = this.settings;
    const list = this.models.filter(m => m.kind === 'face');
    if (!list.length) return null;
    const chosen = list.find(m => m.id === (s.faces || s.faceModel)) || list[0];
    const disabled = this.running();
    const toggle = h('input', { type: 'checkbox', checked: !!s.faces, disabled, onchange: e => {
      s.faces = e.target.checked ? chosen.id : '';
      s.faceModel = chosen.id;
      this.settingsChanged();
    } });
    const head = h('label', { class: 'chk', title: 'Finds the faces in each image and rebuilds them after upscaling' }, toggle, 'Restore faces');
    if (!s.faces) return h('div', { class: 'faces' }, head);

    const sel = h('select', { class: 'inp', disabled }, list.map(m => h('option', { value: m.id, selected: m.id === chosen.id }, m.name)));
    sel.addEventListener('change', () => { s.faces = s.faceModel = sel.value; this.settingsChanged(); });
    const pct = v => Math.round(v * 100) + '%';
    const val = h('span', { class: 'val' }, pct(s.faceStrength));
    const range = h('input', { type: 'range', min: 0.1, max: 1, step: 0.05, value: s.faceStrength, disabled });
    range.addEventListener('input', () => { s.faceStrength = Number(range.value); val.textContent = pct(s.faceStrength); this.saveStateSoon(); });
    return h('div', { class: 'faces' }, head,
      h('div', { class: 'faces-body' },
        h('div', { class: 'field' }, h('label', null, 'Model'), sel),
        h('div', { class: 'field', title: 'How much of the restored faces to blend over the upscaled image. Lower keeps more of the original look.' },
          h('label', null, 'Strength'), h('div', { class: 'quality' }, range, val)),
        h('div', { class: 'mdesc' }, chosen.description),
        chosen.installed ? null : this.modelDownload(chosen)));
  }

  // Whether this upscale restores faces: chosen, and the model isn't SUPIR.
  facesOn() {
    const sel = this.models.find(m => m.id === this.settings.model);
    return !!this.settings.faces && sel?.kind !== 'supir';
  }

  // SUPIR's options. Empty fields use the model's defaults, shown as placeholders. Typing doesn't re-render, so focus stays.
  supirOptions(m) {
    const s = this.settings.supir;
    const d = m.supir || {};
    const disabled = this.running();
    const set = (key, v) => {
      if (v === '' || v == null || Number.isNaN(v)) delete s[key]; else s[key] = v;
      this.saveStateSoon();
    };
    const text = (key, placeholder, rows) => {
      const t = h('textarea', { class: 'inp wide', rows, spellcheck: false, placeholder, disabled });
      t.value = s[key] ?? '';
      t.addEventListener('input', () => set(key, t.value.trim()));
      return t;
    };
    const num = (key, attrs, int) => {
      const i = h('input', { class: 'inp', type: 'number', placeholder: String(d[key] ?? ''), ...attrs, value: s[key] ?? '', disabled });
      i.addEventListener('change', () => set(key, i.value === '' ? '' : int ? Math.round(Number(i.value)) : Number(i.value)));
      return i;
    };
    const field = (label, input, title) => h('div', { class: 'field', title }, h('label', null, label), input);
    const fix = h('select', { class: 'inp', disabled }, [['', 'Default'], ['wavelet', 'Wavelet'], ['adain', 'AdaIN'], ['none', 'None']]
      .map(([v, l]) => h('option', { value: v, selected: (s.colorFix || '') === v }, l)));
    fix.addEventListener('change', () => set('colorFix', fix.value));
    const more = h('details', { class: 'adv', open: !!this.supirMore, ontoggle: () => { this.supirMore = more.open; } },
      h('summary', null, 'More SUPIR options'),
      h('div', { class: 'grid3' },
        field('Restoration', num('restoration', { min: 0, max: 20, step: 0.5, placeholder: d.restoration ? String(d.restoration) : 'Off' }),
          'Pulls the early steps towards the original: fewer invented details, a little less sharpness. 0 is off.'),
        field('Control', num('control', { min: 0, max: 2, step: 0.05 }), 'How closely the result follows the original image (0–2).'),
        field('Color fix', fix, 'Gives the result the colors of the original, removing color shifts.')),
      field('Quality prompt', text('qualityPrompt', d.qualityPrompt, 3)),
      field('Negative prompt', text('negativePrompt', d.negativePrompt, 3)));
    return h('div', { class: 'supir' },
      field('Describe the image (optional)', text('prompt', 'e.g. a red fox kit sitting in the grass', 2), 'A short description helps SUPIR rebuild the right details.'),
      h('div', { class: 'grid3' },
        field('Steps', num('steps', { min: 1, max: 200, step: 1 }, true), 'More steps are slower and a little more detailed.'),
        field('Guidance', num('guidance', { min: 0, max: 30, step: 0.5 }), 'How strongly the prompts steer the result.'),
        field('Seed', num('seed', { min: 0, step: 1, placeholder: 'Random' }, true), 'The same seed and settings give the same result.')),
      more,
      h('div', { class: 'result-size' }, 'SUPIR is slow: minutes per image on a fast GPU.'));
  }

  renderScale() {
    const s = this.settings;
    const disabled = this.running();
    const bySize = s.size && s.size !== 'scale';
    const modeSeg = h('div', { class: 'seg compact' },
      h('button', { class: bySize ? '' : 'on', disabled, onclick: () => { s.size = 'scale'; this.settingsChanged(); } }, 'Scale'),
      h('button', { class: bySize ? 'on' : '', disabled, title: 'A width, height, box or long edge in pixels', onclick: () => { if (!bySize) { s.size = 'fit'; this.settingsChanged(); } } }, 'Size'));
    let controls;
    if (!bySize) {
      const custom = h('input', { class: 'inp custom-scale', type: 'number', min: 1, max: 32, step: 0.5, value: SCALES.includes(Number(s.scale)) ? '' : s.scale, placeholder: 'Other', title: 'Any scale from 1 to 32', disabled });
      custom.addEventListener('change', () => {
        const v = Number(custom.value);
        if (v >= 1 && v <= 32) { s.scale = v; this.settingsChanged(); }
      });
      const seg = h('div', { class: 'seg grow' }, SCALES.map(n => h('button', { class: Number(s.scale) === n ? 'on' : '', disabled, onclick: () => { s.scale = n; this.settingsChanged(); } }, n + '×')));
      controls = h('div', { class: 'row' }, seg, custom);
    } else {
      const px = (key, title) => {
        const i = h('input', { class: 'inp px', type: 'number', min: 1, max: 65535, step: 1, value: s[key], title, disabled });
        i.addEventListener('change', () => {
          const v = Math.round(Number(i.value));
          if (v >= 1 && v <= 65535) { s[key] = v; this.settingsChanged(); } else i.value = s[key];
        });
        return i;
      };
      const mode = h('select', { class: 'inp', disabled }, SIZE_MODES.map(([v, l]) => h('option', { value: v, selected: s.size === v }, l)));
      mode.addEventListener('change', () => { s.size = mode.value; this.settingsChanged(); });
      const inputs = s.size === 'fit' ? [px('width', 'Width'), h('span', { class: 'muted' }, '×'), px('height', 'Height')]
        : s.size === 'width' ? [px('width', 'Width')] : s.size === 'height' ? [px('height', 'Height')] : [px('longEdge', 'Long edge')];
      const preset = h('select', { class: 'inp', disabled, title: 'Common screen and print sizes' },
        h('option', { value: '' }, 'Presets…'), SIZE_PRESETS.map(([l, w, hh], i) => h('option', { value: i }, `${l} (${w} × ${hh})`)));
      preset.addEventListener('change', () => {
        const p = SIZE_PRESETS[preset.value];
        if (p) { s.size = 'fit'; s.width = p[1]; s.height = p[2]; this.settingsChanged(); }
      });
      controls = h('div', { class: 'size-box' }, h('div', { class: 'row' }, mode, ...inputs, h('span', { class: 'muted' }, 'px')), preset);
    }
    let info;
    const src = this.source;
    if (src?.kind === 'image' && src.width) {
      const k = scaleFor(s, src.width, src.height);
      const w = Math.max(1, Math.round(src.width * k)), hh = Math.max(1, Math.round(src.height * k));
      info = h('div', { class: 'result-size' + (k > 32 ? ' err' : '') }, `${src.width} × ${src.height} → `, h('b', null, `${w} × ${hh}`),
        k > 32 ? ' — more than 32×, too large' : ` (${((w * hh) / 1e6).toFixed(1)} MP${bySize ? `, ${fmtScale(k)}×` : ''})`);
    } else {
      info = h('div', { class: 'result-size' }, {
        width: `Each image ${s.width} pixels wide.`, height: `Each image ${s.height} pixels high.`,
        fit: `Each image as large as fits inside ${s.width} × ${s.height}, the box turned to match portrait or landscape images.`,
        long: `Each image ${s.longEdge} pixels on its long edge.`,
      }[s.size] || `Each image ${fmtScale(s.scale)}× wider and taller.`);
    }
    this.secScale.replaceChildren(this.title(3, 'Size', modeSeg), controls, info);
  }

  // How the result looks: AI detail against a plain enlargement, sharpness, the original's colors. Sliders save without re-rendering.
  renderLook() {
    const s = this.settings;
    const disabled = this.running();
    const slider = (key, min, max, step, fmt, title) => {
      const val = h('span', { class: 'val' }, fmt(s[key]));
      const range = h('input', { type: 'range', min, max, step, value: s[key], disabled });
      range.addEventListener('input', () => { s[key] = Number(range.value); val.textContent = fmt(s[key]); this.saveStateSoon(); });
      range.addEventListener('dblclick', () => { s[key] = DEFAULT_SETTINGS[key]; range.value = s[key]; val.textContent = fmt(s[key]); this.saveStateSoon(); });
      return h('div', { class: 'quality', title: title + ' Double-click to reset.' }, range, val);
    };
    const pct = v => Math.round(v * 100) + '%';
    const sharp = v => (v === 0 ? 'Off' : v > 0 ? `+${Math.round(v * 100)}` : `−${Math.round(-v * 100)}`);
    const keep = h('input', { type: 'checkbox', checked: !!s.keepColors, disabled, onchange: e => { s.keepColors = e.target.checked; this.saveStateSoon(); } });
    this.secLook.replaceChildren(this.title(4, 'Look'),
      h('div', { class: 'field' }, h('label', null, 'AI detail'),
        slider('detail', 0, 1, 0.05, pct, 'How much of the AI\'s detail to keep. Lower blends in a plain enlargement of the original, for a more natural look.')),
      h('div', { class: 'field' }, h('label', null, 'Sharpness'),
        slider('sharpen', -1, 1, 0.05, sharp, 'Sharpen the result, or soften it (below zero).')),
      h('label', { class: 'chk', title: 'Gives the result the colors and tones of the original, removing any color shift' }, keep, 'Keep the original colors'));
  }

  buildOutput() {
    const s = this.settings;
    this.fmtSel = h('select', { class: 'inp', title: 'Same as the original keeps each image\'s format (HEIC becomes JPEG; BMP, ICO and still GIFs become PNG)' },
      FORMATS.map(([v, l]) => h('option', { value: v, selected: s.format === v }, l)));
    this.fmtSel.addEventListener('change', () => { s.format = this.fmtSel.value; this.settingsChanged(); });
    this.qualityRow = h('div', { class: 'field' });
    this.outDir = h('input', { class: 'inp', spellcheck: false, value: s.outputDir || '' });
    this.outDir.addEventListener('input', () => { s.outputDir = this.outDir.value.trim(); this.saveStateSoon(); this.renderOutputExtras(); });
    const pickDir = h('button', { class: 'iconbtn', title: 'Choose the output folder', html: icon('folderopen'), onclick: async () => {
      const d = await browse({ mode: 'folder', title: 'Save results in', start: s.outputDir || this.state.lastOutDir || this.state.lastDir });
      if (d) { s.outputDir = d; this.state.lastOutDir = d; this.outDir.value = d; this.saveStateSoon(); this.renderOutputExtras(); }
    } });
    this.name = h('input', { class: 'inp', spellcheck: false, value: s.nameTemplate, placeholder: '{name}' });
    this.name.addEventListener('input', () => { s.nameTemplate = this.name.value; this.saveStateSoon(); this.renderOutputExtras(); });
    this.namePreview = h('div', { class: 'result-size' });
    this.outHint = h('div', { class: 'result-size' });
    const existing = h('select', { class: 'inp' }, [['skip', 'Skip images already upscaled'], ['overwrite', 'Overwrite existing files'], ['rename', 'Keep both (add a number)']]
      .map(([v, l]) => h('option', { value: v, selected: s.existing === v }, l)));
    existing.addEventListener('change', () => { s.existing = existing.value; this.saveStateSoon(); });
    this.secOutput.append(
      this.title(5, 'Save as'),
      h('div', { class: 'field' }, h('label', null, 'Format'), this.fmtSel),
      this.qualityRow,
      h('div', { class: 'field' }, h('label', null, 'Folder'), h('div', { class: 'with-btn' }, this.outDir, pickDir), this.outHint),
      h('div', { class: 'field' }, h('label', null, 'File name'), this.name, this.namePreview),
      h('div', { class: 'field' }, h('label', null, 'If the file exists'), existing));
  }

  renderOutputExtras() {
    const s = this.settings;
    this.fmtSel.value = s.format;
    // PNG and TIFF are always lossless; WebP and JPEG XL become lossless at 100.
    if (s.format === 'png' || s.format === 'tif') this.qualityRow.replaceChildren();
    else {
      const label = () => (s.quality >= 100 && ['webp', 'jxl'].includes(s.format) ? 'Lossless' : s.quality);
      const val = h('span', { class: 'val' }, label());
      const range = h('input', { type: 'range', min: 50, max: 100, value: s.quality });
      range.addEventListener('input', () => { s.quality = Number(range.value); val.textContent = label(); this.saveStateSoon(); });
      this.qualityRow.replaceChildren(h('label', null, s.format === 'auto' ? 'Quality (for JPEG, WebP, AVIF and JPEG XL)' : 'Quality'), h('div', { class: 'quality' }, range, val));
    }
    const sample = this.source?.kind === 'image' ? this.source.path : this.source?.items?.[0]?.path;
    if (sample) {
      const base = baseName(sample).replace(/\.[^.]+$/, '');
      const tpl = s.nameTemplate?.trim() || '{name}';
      const name = tpl.replace(/\{name\}/gi, base).replace(/\{scale\}/gi, scaleToken(s)).replace(/\{model\}/gi, s.model) + '.' + formatFor(sample, s.format);
      this.namePreview.replaceChildren('e.g. ', h('b', null, name));
    } else {
      this.namePreview.replaceChildren('{name}, {scale} and {model} are filled in.');
    }
    const inbox = this.source?.uploaded;
    const sub = `${s.model}_${sizeLabel(s)}${this.facesOn() ? '_faces' : ''}`;
    this.outDir.placeholder = `Subfolder ${sub}`;
    this.outHint.textContent = s.outputDir ? '' : inbox ? `Dropped images are saved in ${this.system.defaultOutputDir}` : `Saved in a ${sub} folder inside the opened folder`;
    this.outHint.style.display = this.outHint.textContent ? '' : 'none';
  }

  renderFoot() {
    const job = this.job;
    const running = this.running();
    if (running) {
      const items = job.items;
      const doneCount = items.filter(i => !['queued', 'running'].includes(i.status)).length;
      const cur = items.find(i => i.status === 'running');
      const frac = (doneCount + (cur?.progress ?? 0)) / items.length;
      const what = cur ? baseName(cur.input) : job.status === 'queued' ? 'Waiting…' : 'Starting…';
      native(job.paused ? 'progress:paused' : `progress:${frac.toFixed(3)}`);
      const pause = job.paused
        ? h('button', { class: 'btn', title: 'Go on where it stopped', onclick: () => this.pause(false), html: icon('play') + '<span>Resume</span>' })
        : h('button', { class: 'btn', title: 'Stop at the next tile; nothing is lost', onclick: () => this.pause(true), html: icon('pause') + '<span>Pause</span>' });
      this.foot.replaceChildren(h('div', { class: 'run' + (job.paused ? ' paused' : '') },
        h('div', { class: 'run-top' }, job.paused ? h('span', { class: 'paused-dot', html: icon('pause') }) : h('span', { class: 'spin', style: { width: '14px', height: '14px' } }),
          h('span', { class: 'what' }, job.paused ? `Paused · ${Math.min(doneCount + 1, items.length)} of ${items.length}` : items.length > 1 ? `${what} · ${Math.min(doneCount + 1, items.length)} of ${items.length}` : `Upscaling ${what}`),
          h('span', { class: 'num muted' }, Math.floor(frac * 100) + '%')),
        h('div', { class: 'bar' }, h('i', { style: { width: frac * 100 + '%' } })),
        h('div', { class: 'foot-note' }, job.paused ? 'Nothing is lost: resume to go on where it stopped.' : eta(job.activeMs, frac)),
        job.notice ? h('div', { class: 'foot-note warn' }, job.notice) : null,
        h('div', { class: 'run-btns' }, pause,
          h('button', { class: 'btn', onclick: () => this.cancel(), html: icon('stop') + '<span>Cancel</span> <kbd>Esc</kbd>' }))));
      return;
    }
    const model = this.models.find(m => m.id === this.settings.model && m.installed);
    const n = this.source?.kind === 'batch' ? this.source.items.length : this.source ? 1 : 0;
    const go = h('button', { class: 'btn go', disabled: !n || !model, onclick: () => this.upscale(), html: icon('sparkle') + `<span>${n > 1 ? `Upscale ${n} images` : 'Upscale'}</span>` });
    go.title = 'Ctrl+Enter';
    let note = null;
    if (!model) note = h('div', { class: 'foot-note warn' }, this.models.some(m => m.installed) ? 'Choose a model.' : 'Download a model to start.');
    else if (this.lastSummary) note = h('div', { class: 'foot-note' }, this.lastSummary);
    this.foot.replaceChildren(...[go, note].filter(Boolean));
  }

  // ------------------------------------------------------------------ stage

  buildStage() {
    const stage = document.getElementById('stage');
    this.stageBar = h('div', { class: 'stage-bar' });
    this.stageBody = h('div', { class: 'stage-body' });
    stage.append(this.stageBar, this.stageBody);
    this.zoomVal = h('span', { class: 'zoom-val' });
  }

  zoomChanged(z) {
    this.zoomVal.textContent = Math.round(z * 100) + '%';
  }

  currentResult() {
    if (this.source?.kind === 'image') return this.results.get(this.source.path);
    if (this.focus) return this.results.get(this.focus);
    return null;
  }

  renderStage() {
    const s = this.source;
    if (!s) return this.renderEmpty();
    if (s.kind === 'batch' && !this.focus) return this.renderBatch();
    const path = s.kind === 'image' ? s.path : this.focus;
    const res = this.results.get(path);
    const after = res?.status === 'done' ? imageUrl(res.output, res.finished) : null;
    this.viewer.show({
      before: imageUrl(path),
      after,
      beforeLabel: 'Original' + (res?.inWidth ? ` · ${res.inWidth}×${res.inHeight}` : ''),
      afterLabel: res?.outWidth ? `Upscaled · ${res.outWidth}×${res.outHeight}` : 'Upscaled',
    });
    this.viewer.setMode(this.prefs.viewMode);
    if (this.viewer.el.parentNode !== this.stageBody) this.stageBody.replaceChildren(this.viewer.el);
    this.renderViewerBar(path, res);
    this.updateViewerBusy();
  }

  renderViewerBar(path, res) {
    const hasAfter = res?.status === 'done';
    const mode = this.prefs.viewMode;
    const modes = [['slider', 'compare', 'Compare (C)'], ['side', 'side', 'Side by side (S)'], ['before', null, 'Original (B)', 'Original'], ['after', null, 'Result (A)', 'Result']];
    const seg = h('div', { class: 'seg compact' }, modes.map(([m, ic, title, label]) => h('button', {
      class: mode === m ? 'on' : '', title, disabled: !hasAfter, onclick: () => this.setViewMode(m),
      html: (ic ? icon(ic) : '') + (label ? `<span>${label}</span>` : ''),
    })));
    const back = this.source.kind === 'batch'
      ? h('button', { class: 'tbtn', title: 'Back to all images', onclick: () => { this.focus = null; this.renderStage(); }, html: icon('back') + '<span>All images</span>' })
      : null;
    const faces = res?.faces != null ? ` · ${res.faces ? plural(res.faces, 'face') + ' restored' : 'no faces found'}` : '';
    const meta = res?.status === 'done' ? `${fmtMs(res.ms)}${res.frames ? ` · ${res.frames} frames` : ''}${faces} · ${baseName(res.output)}` : res?.status === 'failed' ? 'Failed' : res?.status === 'skipped' ? 'Skipped: the result already exists' : '';
    this.stageBar.replaceChildren(...[
      back, back ? h('span', { class: 'sep' }) : null,
      h('div', { class: 'stage-title' }, h('b', { title: path }, baseName(path)), h('span', null, meta)),
      h('span', { class: 'grow' }),
      seg,
      h('span', { class: 'sep' }),
      h('button', { class: 'iconbtn', title: 'Zoom out (-)', html: icon('zoomout'), onclick: () => this.viewer.zoomBy(0.8) }),
      this.zoomVal,
      h('button', { class: 'iconbtn', title: 'Zoom in (+)', html: icon('zoomin'), onclick: () => this.viewer.zoomBy(1.25) }),
      h('button', { class: 'iconbtn', title: 'Fit (F)', html: icon('fit'), onclick: () => this.viewer.fitView() }),
      h('button', { class: 'tbtn', title: 'Actual size (1)', onclick: () => this.viewer.zoom100() }, '1:1'),
      hasAfter ? h('span', { class: 'sep' }) : null,
      hasAfter ? h('button', { class: 'tbtn', title: 'Open the result in the default app', onclick: () => this.openResult(), html: icon('image') + '<span>Open</span>' }) : null,
      hasAfter ? h('button', { class: 'tbtn', title: 'Show in folder (Ctrl+Shift+E)', onclick: () => this.revealResult(), html: icon('external') + '<span>Show in folder</span>' }) : null].filter(Boolean));
    this.viewer.setError(res?.status === 'failed' ? res.error : null);
  }

  updateViewerBusy() {
    const s = this.source;
    if (!s || (s.kind === 'batch' && !this.focus)) return;
    const path = s.kind === 'image' ? s.path : this.focus;
    const it = this.running() ? this.job.items.find(i => i.input === path) : null;
    if (it && (it.status === 'running' || it.status === 'queued')) {
      this.viewer.setBusy({
        text: it.status === 'queued' ? 'Waiting…' : it.step === 'faces' ? 'Restoring faces…' : it.progress > 0 ? 'Upscaling…' : 'Loading the model…',
        detail: it.status === 'running' ? Math.floor(it.progress * 100) + '%' : '',
        pct: it.status === 'running' && it.progress > 0 ? it.progress : null,
      });
    } else this.viewer.setBusy(null);
  }

  renderEmpty() {
    this.stageBar.replaceChildren(h('div', { class: 'stage-title' }, h('b', null, 'No image'), h('span', null, 'Open or drop images to begin')));
    const anyModel = this.models.some(m => m.installed && isUpscaler(m));
    const general = this.models.find(m => m.featured && !m.installed && isUpscaler(m));
    const dl = general?.download && !general.download.done;
    this.stageBody.replaceChildren(h('div', { class: 'empty' }, h('div', { class: 'drop-card' },
      h('div', { class: 'big', html: icon('sparkle') }),
      h('h2', null, 'Upscale images with AI'),
      h('p', null, 'Drop images or a folder here, paste an image, or open files from your computer. Results are saved in a subfolder named after the model and scale.'),
      h('div', { class: 'btns' },
        h('button', { class: 'btn primary', onclick: () => this.openImages(), html: icon('image') + '<span>Open images…</span>' }),
        h('button', { class: 'btn', onclick: () => this.openFolder(), html: icon('folderopen') + '<span>Open folder…</span>' })),
      h('div', { class: 'kbd-hint' }, h('kbd', null, 'Ctrl'), '+', h('kbd', null, 'V'), ' pastes an image or a copied path'),
      !anyModel && general ? h('div', { class: 'first-run' },
        h('div', null, h('b', null, 'First, get a model. '), `${general.name} is ${fmtBytes(general.size)} and works well for most pictures.`),
        dl ? h('span', { class: 'spin' }) : h('button', { class: 'btn primary small', onclick: () => this.downloadModel(general.id, this.wantsHalf(general)), html: icon('download') + '<span>Download</span>' })) : null)));
  }

  renderBatch() {
    const s = this.source;
    const doneN = s.items.filter(i => this.results.get(i.path)?.status === 'done').length;
    this.stageBar.replaceChildren(
      h('div', { class: 'stage-title' }, h('b', { title: s.folder || '' }, s.label), h('span', null, `${plural(s.items.length, 'image')}${doneN ? ` · ${doneN} upscaled` : ''}`)),
      h('span', { class: 'grow' }),
      h('button', { class: 'tbtn', title: 'Show the output folder', disabled: !doneN, onclick: () => this.revealBatch(), html: icon('external') + '<span>Show results</span>' }));
    this.cards = new Map();
    const grid = h('div', { class: 'grid' }, s.items.slice(0, 2000).map(it => {
      const img = h('img', { loading: 'lazy', alt: '', src: thumbUrl(it.path, 320) });
      const badge = h('span', { class: 'card-badge', style: { display: 'none' } });
      const bar = h('div', { class: 'card-bar', style: { display: 'none' } }, h('div', { class: 'bar' }, h('i')));
      const meta = h('div', { class: 'card-meta' }, it.rel && it.rel !== baseName(it.path) ? dirName(it.rel) || '' : '');
      const card = h('div', { class: 'card', title: it.path }, h('div', { class: 'card-img' }, img, badge, bar), h('div', { class: 'card-body' }, h('div', { class: 'card-name' }, baseName(it.path)), meta));
      card.addEventListener('click', () => { this.focus = it.path; this.renderStage(); });
      const c = { card, img, badge, bar, meta, path: it.path, shown: null };
      this.cards.set(it.path, c);
      this.updateCard(c);
      return card;
    }));
    const more = s.items.length > 2000 ? h('p', { class: 'hint' }, `Showing the first 2000 of ${s.items.length} images.`) : null;
    this.stageBody.replaceChildren(h('div', { class: 'batch' }, grid, more));
  }

  updateCard(c) {
    const live = this.running() ? this.job.items.find(i => i.input === c.path) : null;
    const r = live && live.status !== 'queued' ? live : this.results.get(c.path) || live;
    const st = r?.status;
    const label = { queued: 'Queued', running: 'Upscaling', done: 'Done', skipped: 'Skipped', failed: 'Failed', cancelled: 'Cancelled' }[st];
    c.badge.style.display = st ? '' : 'none';
    c.badge.className = 'card-badge ' + (st || '');
    c.badge.innerHTML = (st === 'done' ? icon('check') : st === 'failed' ? icon('error') : st === 'skipped' ? icon('skip') : '') + `<span>${label || ''}</span>`;
    c.bar.style.display = st === 'running' ? '' : 'none';
    if (st === 'running') c.bar.querySelector('i').style.width = (r.progress || 0) * 100 + '%';
    c.card.classList.toggle('done', st === 'done');
    if (st === 'done' && c.shown !== r.output) {
      c.shown = r.output;
      c.img.src = thumbUrl(r.output, 320, r.finished);
      c.meta.className = 'card-meta';
      c.meta.textContent = `${r.outWidth} × ${r.outHeight} · ${fmtMs(r.ms)}${r.frames ? ` · ${r.frames} frames` : ''}${r.faces ? ` · ${plural(r.faces, 'face')}` : ''}`;
    } else if (st === 'failed') {
      c.meta.className = 'card-meta err';
      c.meta.textContent = r.error || 'Failed';
    } else if (st === 'skipped') {
      c.meta.className = 'card-meta';
      c.meta.textContent = 'Already upscaled';
    }
  }

  setViewMode(m) {
    this.prefs.viewMode = m;
    this.saveStateSoon();
    if (this.source && (this.source.kind === 'image' || this.focus)) this.renderStage();
  }

  // ------------------------------------------------------------------ sources

  running() {
    return !!this.job && (this.job.status === 'queued' || this.job.status === 'running');
  }

  async openImages() {
    if (this.running()) return;
    const paths = await browse({ mode: 'images', start: this.state.lastDir, onDir: d => { this.state.lastDir = d; this.saveStateSoon(); } });
    if (paths) await this.openPaths(paths);
  }

  async openFolder() {
    if (this.running()) return;
    const dir = await browse({ mode: 'folder', title: 'Upscale all images in a folder', start: this.state.lastDir, onDir: d => { this.state.lastDir = d; this.saveStateSoon(); } });
    if (dir) await this.openPaths([dir], { folder: dir });
  }

  /** Opens files and folders: one image becomes a single-image source, anything else a batch. */
  async openPaths(paths, { folder = null, uploaded = false } = {}) {
    if (this.running()) { this.status('Wait for the current job to finish.'); return; }
    try {
      this.status('Looking for images…');
      const scan = await post('/fs/scan', { inputs: paths, recursive: !!this.state.recursive, nameTemplate: this.settings.nameTemplate });
      if (!scan.items.length) {
        await alertError(scan.skipped ? 'None of these files is a supported image (PNG, JPEG, WebP, BMP, GIF, ICO).' : 'No images found here.' + (this.state.recursive ? '' : ' Subfolders are not searched; turn on “Include subfolders” to look in them.'), 'Nothing to upscale');
        if (!folder) { this.status('Ready.'); return; }
      }
      // A single folder argument is a folder batch even if it holds one image.
      if (!folder && paths.length === 1 && scan.items.length === 1) {
        const info = await get('/fs/info', { path: scan.items[0].path });
        this.source = { kind: 'image', path: info.path, name: info.name, size: info.size, width: info.width, height: info.height, uploaded };
      } else {
        const onlyFolder = folder || (paths.length === 1 && !/\.[a-z0-9]{2,5}$/i.test(paths[0]) ? paths[0] : null);
        this.source = {
          kind: 'batch', items: scan.items, bytes: scan.bytes, folder: onlyFolder, uploaded, previous: scan.previousResults,
          label: onlyFolder ? baseName(onlyFolder) : `${plural(scan.items.length, 'image')}`, inputs: paths,
        };
      }
      this.focus = null;
      if (this.source.kind === 'image' || this.source.items.length) this.status(this.source.kind === 'image' ? `Opened ${this.source.name}.` : `${plural(this.source.items.length, 'image')} ready.`);
      this.renderAll();
    } catch (e) {
      alertError(e.message);
      this.status('Ready.');
    }
  }

  async rescan() {
    const s = this.source;
    if (s?.kind !== 'batch' || !s.folder) return;
    await this.openPaths(s.inputs, { folder: s.folder, uploaded: s.uploaded });
  }

  clearSource() {
    if (this.running()) return;
    this.source = null;
    this.focus = null;
    this.viewer.clear();
    this.renderAll();
    this.status('Ready.');
  }

  // ------------------------------------------------------------------ drop & paste

  setupDrop() {
    const overlay = h('div', { class: 'drop-overlay' }, h('div', { html: icon('download') + '<span>Drop to upscale</span>' }));
    document.body.append(overlay);
    let depth = 0;
    const hasFiles = e => [...(e.dataTransfer?.types || [])].includes('Files');
    window.addEventListener('dragenter', e => { if (!hasFiles(e)) return; e.preventDefault(); depth++; overlay.classList.add('on'); });
    window.addEventListener('dragleave', e => { if (!hasFiles(e)) return; if (--depth <= 0) { depth = 0; overlay.classList.remove('on'); } });
    window.addEventListener('dragover', e => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = this.running() ? 'none' : 'copy'; });
    window.addEventListener('drop', async e => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      overlay.classList.remove('on');
      if (this.running() || document.querySelector('.modal-bg')) return;
      const files = await collectDropped(e.dataTransfer);
      if (files.length) this.uploadAndOpen(files);
    });
  }

  async uploadAndOpen(files) {
    const exts = this.system.inputExtensions;
    const ok = files.filter(f => exts.includes((f.name.match(/\.[^.]+$/)?.[0] || '').toLowerCase()));
    if (!ok.length) { alertError('None of the dropped files is a supported image (PNG, JPEG, WebP, AVIF, HEIC, TIFF, JPEG XL, BMP, GIF, ICO).'); return; }
    const paths = [];
    try {
      for (let i = 0; i < ok.length; i += UPLOAD_BATCH) {
        this.status(ok.length > 1 ? `Copying images… ${Math.min(i + UPLOAD_BATCH, ok.length)} of ${ok.length}` : 'Copying the image…');
        const fd = new FormData();
        for (const f of ok.slice(i, i + UPLOAD_BATCH)) fd.append('files', f, f.name);
        const res = await fetch('/api/upload', { method: 'POST', headers: { 'X-Token': token }, body: fd });
        const j = await res.json();
        if (!j.ok) throw new Error(j.error);
        paths.push(...j.data);
      }
    } catch (e) {
      alertError('Could not read the dropped files: ' + e.message);
      this.status('Ready.');
      return;
    }
    await this.openPaths(paths, { uploaded: true });
  }

  async paste(e) {
    if (e.target.closest?.('input, textarea')) return;
    const files = [...(e.clipboardData?.files || [])];
    if (files.length) {
      e.preventDefault();
      const stamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '-');
      const named = files.map((f, i) => (f.name && f.name !== 'image.png' ? f : new File([f], `pasted-${stamp}${files.length > 1 ? '-' + (i + 1) : ''}.${(f.type.split('/')[1] || 'png').replace('jpeg', 'jpg')}`, { type: f.type })));
      this.uploadAndOpen(named);
      return;
    }
    // A copied path ("Copy as path" in Explorer puts it in quotes), one per line.
    const text = e.clipboardData?.getData('text/plain')?.trim();
    if (text && /^("?)([a-zA-Z]:[\\/]|\/|\\\\)/.test(text)) {
      e.preventDefault();
      const paths = text.split(/\r?\n/).map(l => l.trim().replace(/^"|"$/g, '')).filter(Boolean);
      const folder = paths.length === 1 && !/\.[a-z0-9]{2,5}$/i.test(paths[0]) ? paths[0] : null;
      this.openPaths(paths, { folder });
    }
  }

  async pasteFromMenu() {
    try {
      const items = await navigator.clipboard.read();
      const files = [];
      for (const it of items) {
        const type = it.types.find(t => t.startsWith('image/'));
        if (type) files.push(new File([await it.getType(type)], `pasted-${Date.now()}.${type.split('/')[1].replace('jpeg', 'jpg')}`, { type }));
      }
      if (files.length) this.uploadAndOpen(files);
      else this.status('There is no image on the clipboard.');
    } catch {
      this.status('Press Ctrl+V to paste an image.');
    }
  }

  // ------------------------------------------------------------------ models

  selectModel(id) {
    if (this.running()) return;
    this.settings.model = id;
    this.settingsChanged();
  }

  setModels(list) {
    const before = new Set(this.models.filter(m => m.installed).map(m => m.id));
    this.models = list;
    const ready = list.filter(m => m.installed && isUpscaler(m));
    // A model that just finished downloading becomes the choice if none was usable.
    const fresh = ready.find(m => !before.has(m.id));
    if (fresh && !ready.some(m => m.id === this.settings.model && before.has(m.id))) this.settings.model = fresh.id;
    if (!ready.some(m => m.id === this.settings.model)) this.settings.model = ready[0]?.id || this.settings.model;
    this.renderModels();
    this.renderFoot();
    if (!this.source) this.renderEmpty();
  }

  async downloadModel(id, half = false) {
    try {
      await post(`/models/${id}/download${half ? '?half=true' : ''}`);
      this.pollModels();
    } catch (e) {
      alertError(e.message);
    }
  }

  async pollModels() {
    clearTimeout(this.modelTimer);
    try {
      const list = await get('/models');
      this.setModels(list);
      const busy = list.find(m => m.download && !m.download.done);
      if (busy) {
        this.status(`Downloading ${busy.name}… ${Math.round(busy.download.received / busy.download.total * 100)}%`);
        this.modelTimer = setTimeout(() => this.pollModels(), 400);
      } else {
        const failed = list.find(m => m.download?.error);
        this.status(failed ? `Download of ${failed.name} failed: ${failed.download.error}` : 'Model ready.');
      }
    } catch (e) {
      this.status(e.message);
    }
  }

  // ------------------------------------------------------------------ jobs

  async upscale() {
    if (this.running() || !this.source) return;
    const model = this.models.find(m => m.id === this.settings.model && m.installed);
    if (!model) { modelsDialog(this); return; }
    const faceModel = this.facesOn() ? this.models.find(m => m.id === this.settings.faces) : null;
    if (faceModel && !faceModel.installed) {
      alertError(`Download ${faceModel.name} first (under Restore faces), or turn face restoration off.`);
      return;
    }
    const items = this.source.kind === 'image' ? [{ path: this.source.path }] : this.source.items;
    if (this.settings.existing === 'overwrite' && items.length > 20
      && !await confirmDlg(`Upscale ${items.length} images and overwrite results that already exist?`, { title: 'Upscale', ok: 'Upscale' })) return;
    // In a browser, ask now (a click) whether a notification may say when a batch is done.
    if (!NATIVE && items.length > 1 && window.Notification?.permission === 'default') Notification.requestPermission().catch(() => {});
    try {
      this.lastSummary = null;
      // A single image is always upscaled again when asked: the point is to see the new result.
      const settings = { ...this.settings };
      if (this.source.kind === 'image' && settings.existing === 'skip') settings.existing = 'overwrite';
      this.job = await post('/jobs', { items, settings });
      if (this.source.kind === 'batch' && this.focus) this.focus = null;
      this.status(`Upscaling with ${model.name}${faceModel ? ` and restoring faces with ${faceModel.name}` : ''}…`);
      this.renderAll();
      this.poll();
    } catch (e) {
      alertError(e.message);
    }
  }

  async poll() {
    clearTimeout(this.jobTimer);
    if (!this.job) return;
    try {
      const prev = this.job;
      this.job = await get(`/jobs/${prev.id}`);
    } catch (e) {
      this.status(e.message);
      this.jobTimer = setTimeout(() => this.poll(), 1000);
      return;
    }
    const job = this.job;
    for (const it of job.items) {
      if (['done', 'failed', 'skipped'].includes(it.status) && this.results.get(it.input)?.finishedJob !== job.id) {
        this.results.set(it.input, { ...it, finished: Date.now(), finishedJob: job.id });
        if (this.source?.kind === 'image' && this.source.path === it.input) this.renderStage();
      }
    }
    if (job.device && job.device !== this.system.activeDevice) { this.system.activeDevice = job.device; this.renderDevice(); }
    if (this.source?.kind === 'batch' && !this.focus && this.cards) {
      for (const c of this.cards.values()) this.updateCard(c);
      const doneN = this.source.items.filter(i => this.results.get(i.path)?.status === 'done').length;
      const t = this.stageBar.querySelector('.stage-title span');
      if (t) t.textContent = `${plural(this.source.items.length, 'image')}${doneN ? ` · ${doneN} upscaled` : ''}`;
    } else {
      this.updateViewerBusy();
    }

    if (this.running()) {
      this.renderFoot();
      this.jobTimer = setTimeout(() => this.poll(), 250);
      return;
    }
    // Finished.
    const c = s => job.items.filter(i => i.status === s).length;
    const took = fmtMs(job.activeMs || (job.finished ?? Date.now()) - (job.started ?? job.created)); // without pauses
    const parts = [c('done') && `${c('done')} upscaled`, c('skipped') && `${c('skipped')} skipped`, c('failed') && `${c('failed')} failed`, c('cancelled') && `${c('cancelled')} cancelled`].filter(Boolean);
    this.lastSummary = job.items.length === 1
      ? (c('done') ? `Done in ${took}.` : c('failed') ? 'Failed.' : c('skipped') ? 'Skipped: the result already exists.' : 'Cancelled.')
      : `${parts.join(', ')} in ${took}.`;
    this.status(job.status === 'cancelled' ? 'Cancelled.' : this.lastSummary);
    const failed = job.items.find(i => i.status === 'failed');
    if (failed && job.items.length === 1) { /* shown in the viewer */ } else if (failed && c('done') === 0) alertError(failed.error || 'Upscaling failed.');
    this.renderAll();
    // The taskbar button goes back to normal, or red when images failed; away from the window, say it's done.
    native(c('failed') ? 'progress:error' : 'progress:none');
    if (job.status === 'done' && (c('done') || c('failed')) && (document.hidden || !document.hasFocus())) {
      if (NATIVE) {
        native(`notify:ZawUpscale\n${this.lastSummary}`);
        native('attention');
      } else if (window.Notification?.permission === 'granted') {
        try { new Notification('ZawUpscale', { body: this.lastSummary, icon: 'icon-256.png' }); } catch { /* not allowed */ }
      }
    }
  }

  async pause(on) {
    if (!this.running()) return;
    try {
      this.job = { ...this.job, paused: on };
      this.renderFoot();
      await post(`/jobs/${this.job.id}/${on ? 'pause' : 'resume'}`);
      this.status(on ? 'Paused. The current image stops at its next tile.' : 'Resumed.');
    } catch (e) {
      alertError(e.message);
    }
  }

  async cancel() {
    if (!this.running()) return;
    try {
      await post(`/jobs/${this.job.id}/cancel`);
      this.status('Cancelling…');
    } catch (e) {
      alertError(e.message);
    }
  }

  revealResult() {
    const r = this.currentResult();
    if (r?.output) post('/fs/reveal', { path: r.output }).catch(e => alertError(e.message));
  }

  openResult() {
    const r = this.currentResult();
    if (r?.output) post('/fs/open', { path: r.output }).catch(e => alertError(e.message));
  }

  revealBatch() {
    const first = this.source?.items.map(i => this.results.get(i.path)).find(r => r?.status === 'done');
    if (first) post('/fs/reveal', { path: this.settings.outputDir || first.output }).catch(e => alertError(e.message));
  }

  async exit() {
    if (this.running() && !await confirmDlg('An upscale is running. Quit anyway?', { title: 'Exit', ok: 'Quit', danger: true })) return;
    this.saveStateSoon.flush();
    try { await post('/exit'); } catch { /* already gone */ }
    window.close();
  }

  // ------------------------------------------------------------------ keys

  globalKey(e) {
    const ctrl = e.ctrlKey || e.metaKey;
    const k = e.key.toLowerCase();
    if (document.querySelector('.modal-bg')) return;
    const typing = e.target.closest?.('input, textarea, select');
    if (ctrl && k === 'o') { e.preventDefault(); if (e.shiftKey) this.openFolder(); else this.openImages(); }
    else if (ctrl && k === 'enter') { e.preventDefault(); this.upscale(); }
    else if (ctrl && k === 'm') { e.preventDefault(); modelsDialog(this); }
    else if (ctrl && e.shiftKey && k === 'e') { e.preventDefault(); this.revealResult(); }
    else if (k === 'escape' && this.running()) { e.preventDefault(); this.cancel(); }
    else if (k === 'escape' && this.focus) { e.preventDefault(); this.focus = null; this.renderStage(); }
    else if (!ctrl && !e.altKey && !typing && this.currentResult()?.status === 'done' && { c: 'slider', s: 'side', b: 'before', a: 'after' }[k]) {
      e.preventDefault();
      this.setViewMode({ c: 'slider', s: 'side', b: 'before', a: 'after' }[k]);
    } else if (!ctrl && !e.altKey && !typing && this.viewer.el.isConnected && ['+', '=', '-', '0', '1', 'f'].includes(k) && e.target === document.body) {
      this.viewer.key(e);
    }
  }
}

/** Files from a drop, including the contents of dropped folders. */
async function collectDropped(dt) {
  const entries = [...(dt.items || [])].map(i => i.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...(dt.files || [])];
  const out = [];
  const readAll = reader => new Promise(resolve => {
    const acc = [];
    const next = () => reader.readEntries(batch => { if (!batch.length) resolve(acc); else { acc.push(...batch); next(); } }, () => resolve(acc));
    next();
  });
  const walk = async entry => {
    if (out.length > 20000) return;
    if (entry.isFile) out.push(await new Promise(resolve => entry.file(resolve, () => resolve(null))));
    else if (entry.isDirectory) for (const e of await readAll(entry.createReader())) await walk(e);
  };
  for (const e of entries) await walk(e);
  return out.filter(Boolean);
}

new App().init();
