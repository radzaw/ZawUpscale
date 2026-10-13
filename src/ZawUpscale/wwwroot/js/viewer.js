// Before / after image viewer: a slider that wipes between the original and the result, or both side by side,
// with shared zoom and pan. Content coordinates are pixels of the result (or of the original while there is none),
// so 100% means one screen pixel per output pixel.
import { h } from './util.js';
import { icon } from './icons.js';

const MIN_Z = 0.02, MAX_Z = 32;

export class Viewer {
  constructor({ onZoom } = {}) {
    this.onZoom = onZoom;
    this.mode = 'slider'; // slider | side | before | after
    this.split = 0.5;
    this.z = 1;
    this.tx = 0;
    this.ty = 0;
    this.fit = true;
    this.cw = 0;
    this.ch = 0;
    this.after = null;

    this.imgA = h('img', { class: 'vw-img', alt: '', draggable: false });
    this.imgB = h('img', { class: 'vw-img', alt: '', draggable: false });
    this.paneA = h('div', { class: 'vw-pane vw-a' }, this.imgA);
    this.paneB = h('div', { class: 'vw-pane vw-b' }, this.imgB);
    this.labelL = h('span', { class: 'vw-label l' });
    this.labelR = h('span', { class: 'vw-label r' });
    this.handle = h('div', { class: 'vw-handle', title: 'Drag to compare' }, h('div', { class: 'vw-knob', html: icon('compare') }));
    this.divider = h('div', { class: 'vw-divider' });
    this.busy = h('div', { class: 'vw-busy', style: { display: 'none' } });
    this.error = h('div', { class: 'vw-error', style: { display: 'none' } });
    this.el = h('div', { class: 'vw', tabIndex: 0 }, this.paneA, this.paneB, this.divider, this.handle, this.labelL, this.labelR, this.busy, this.error);

    for (const img of [this.imgA, this.imgB]) img.addEventListener('load', () => this.contentChanged());
    this.el.addEventListener('wheel', e => this.wheel(e), { passive: false });
    this.el.addEventListener('pointerdown', e => this.pointerDown(e));
    this.el.addEventListener('dblclick', e => {
      if (e.target.closest('.vw-handle')) return;
      if (this.fit) this.zoomAt(1, ...this.paneXY(e)); else this.fitView();
    });
    this.el.addEventListener('keydown', e => this.key(e));
    new ResizeObserver(() => this.layout()).observe(this.el);
  }

  /** Shows an original and, optionally, its upscaled result. Labels describe each side. */
  show({ before, after = null, beforeLabel = 'Original', afterLabel = 'Upscaled' }) {
    const sameBefore = this.imgA.dataset.src === before;
    this.after = after;
    this.labels = { before: beforeLabel, after: afterLabel };
    if (!sameBefore) { this.imgA.dataset.src = before; this.imgA.src = before; }
    if (after) { if (this.imgB.dataset.src !== after) { this.imgB.dataset.src = after; this.imgB.src = after; } }
    else { this.imgB.removeAttribute('src'); this.imgB.dataset.src = ''; }
    if (!sameBefore) this.fit = true;
    this.contentChanged();
  }

  clear() {
    for (const img of [this.imgA, this.imgB]) { img.removeAttribute('src'); img.dataset.src = ''; }
    this.after = null;
    this.setBusy(null);
    this.setError(null);
  }

  setMode(m) {
    this.mode = m;
    this.layout();
  }

  /** Progress overlay while the image is being upscaled: {text, detail, pct} (pct null = indeterminate), or null. */
  setBusy(b) {
    if (!b) { this.busy.style.display = 'none'; return; }
    this.busy.style.display = '';
    const bar = h('div', { class: 'bar' + (b.pct == null ? ' indet' : '') }, h('i', { style: { width: (b.pct ?? 0) * 100 + '%' } }));
    this.busy.replaceChildren(h('div', null, h('div', { class: 't' }, h('b', null, b.text), h('span', { class: 'num' }, b.detail ?? '')), bar));
  }

  setError(msg) {
    this.error.style.display = msg ? '' : 'none';
    if (msg) this.error.replaceChildren(h('span', { html: icon('error') }), h('span', null, msg));
  }

  contentChanged() {
    const ref = this.after && this.imgB.naturalWidth ? this.imgB : this.imgA;
    const w = ref.naturalWidth, hh = ref.naturalHeight;
    if (!w || !hh) { this.layout(); return; }
    if (this.cw && this.cw !== w && !this.fit) {
      // The result replaced the original (or the other way round): keep the same view of the picture.
      const k = w / this.cw;
      this.z /= k;
    }
    this.cw = w;
    this.ch = hh;
    this.layout();
  }

  /** Pane rectangles in viewer coordinates. */
  panes() {
    const W = this.el.clientWidth;
    if (this.mode === 'side' && this.after) {
      const half = Math.floor(W / 2);
      return { a: [0, half - 1], b: [half + 1, W - half - 1] };
    }
    return { a: [0, W], b: [0, W] };
  }

  layout() {
    const W = this.el.clientWidth, H = this.el.clientHeight;
    const side = this.mode === 'side' && this.after;
    const { a, b } = this.panes();
    const pw = a[1];
    if (this.cw && this.ch && W && H) {
      if (this.fit) {
        this.z = Math.min(pw / this.cw, H / this.ch, 8);
        this.tx = (pw - this.cw * this.z) / 2;
        this.ty = (H - this.ch * this.z) / 2;
      }
      this.clamp(pw, H);
    }
    const showA = !this.after || this.mode !== 'after';
    const showB = !!this.after && this.mode !== 'before';
    Object.assign(this.paneA.style, { left: a[0] + 'px', width: a[1] + 'px', display: showA ? '' : 'none' });
    Object.assign(this.paneB.style, { left: b[0] + 'px', width: b[1] + 'px', display: showB ? '' : 'none' });
    for (const img of [this.imgA, this.imgB]) {
      img.style.width = this.cw * this.z + 'px';
      img.style.height = this.ch * this.z + 'px';
      img.style.transform = `translate(${this.tx}px, ${this.ty}px)`;
    }
    const slider = this.mode === 'slider' && this.after;
    this.paneB.style.clipPath = slider ? `inset(0 0 0 ${Math.round(this.split * W)}px)` : '';
    this.handle.style.display = slider ? '' : 'none';
    this.handle.style.left = Math.round(this.split * W) + 'px';
    this.divider.style.display = side ? '' : 'none';
    this.divider.style.left = Math.floor(W / 2) + 'px';
    this.el.classList.toggle('pixelated', this.pixelated !== false && this.z >= 2);

    const lb = this.labels || { before: 'Original', after: 'Upscaled' };
    const L = this.labelL, R = this.labelR;
    if (!this.after) { L.textContent = lb.before; L.style.display = this.cw ? '' : 'none'; R.style.display = 'none'; }
    else if (this.mode === 'before') { L.textContent = lb.before; L.style.display = ''; R.style.display = 'none'; }
    else if (this.mode === 'after') { L.textContent = lb.after; L.style.display = ''; R.style.display = 'none'; }
    else {
      L.textContent = lb.before;
      R.textContent = lb.after;
      L.style.display = R.style.display = '';
      R.style.left = side ? b[0] + 12 + 'px' : '';
      R.style.right = side ? 'auto' : '';
    }
    this.onZoom?.(this.z, this.fit);
  }

  clamp(pw, H) {
    const w = this.cw * this.z, hh = this.ch * this.z;
    this.tx = w <= pw ? (pw - w) / 2 : Math.min(0, Math.max(pw - w, this.tx));
    this.ty = hh <= H ? (H - hh) / 2 : Math.min(0, Math.max(H - hh, this.ty));
  }

  /** Mouse position relative to the pane under it (side by side, both panes share coordinates). */
  paneXY(e) {
    const r = this.el.getBoundingClientRect();
    let x = e.clientX - r.left;
    const { b } = this.panes();
    if (this.mode === 'side' && this.after && x >= b[0]) x -= b[0];
    return [x, e.clientY - r.top];
  }

  zoomAt(z, px, py) {
    if (!this.cw) return;
    z = Math.min(MAX_Z, Math.max(MIN_Z, z));
    const cx = (px - this.tx) / this.z, cy = (py - this.ty) / this.z;
    this.z = z;
    this.tx = px - cx * z;
    this.ty = py - cy * z;
    this.fit = false;
    this.layout();
  }

  /** Zoom about the centre of the view. */
  zoomBy(k) {
    const { a } = this.panes();
    this.zoomAt(this.z * k, a[1] / 2, this.el.clientHeight / 2);
  }

  zoom100() {
    const { a } = this.panes();
    this.zoomAt(1, a[1] / 2, this.el.clientHeight / 2);
  }

  fitView() {
    this.fit = true;
    this.layout();
  }

  wheel(e) {
    e.preventDefault();
    const [x, y] = this.paneXY(e);
    this.zoomAt(this.z * Math.exp(-e.deltaY * 0.0015), x, y);
  }

  pointerDown(e) {
    if (e.button !== 0 || !this.cw) return;
    this.el.focus({ preventScroll: true });
    const onHandle = e.target.closest('.vw-handle');
    const r = this.el.getBoundingClientRect();
    const sx = e.clientX, sy = e.clientY, tx0 = this.tx, ty0 = this.ty;
    this.el.setPointerCapture(e.pointerId);
    if (!onHandle) this.el.classList.add('panning');
    const move = ev => {
      if (onHandle) {
        this.split = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
      } else {
        this.tx = tx0 + ev.clientX - sx;
        this.ty = ty0 + ev.clientY - sy;
        this.fit = false;
      }
      this.layout();
    };
    const up = () => {
      this.el.removeEventListener('pointermove', move);
      this.el.removeEventListener('pointerup', up);
      this.el.removeEventListener('pointercancel', up);
      this.el.classList.remove('panning');
    };
    this.el.addEventListener('pointermove', move);
    this.el.addEventListener('pointerup', up);
    this.el.addEventListener('pointercancel', up);
  }

  key(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const step = 40;
    switch (e.key) {
      case '+': case '=': this.zoomBy(1.25); break;
      case '-': this.zoomBy(0.8); break;
      case '0': case 'f': this.fitView(); break;
      case '1': this.zoom100(); break;
      case 'ArrowLeft': if (this.mode === 'slider' && this.after) { this.split = Math.max(0, this.split - 0.05); this.layout(); } else { this.tx += step; this.fit = false; this.layout(); } break;
      case 'ArrowRight': if (this.mode === 'slider' && this.after) { this.split = Math.min(1, this.split + 0.05); this.layout(); } else { this.tx -= step; this.fit = false; this.layout(); } break;
      case 'ArrowUp': this.ty += step; this.fit = false; this.layout(); break;
      case 'ArrowDown': this.ty -= step; this.fit = false; this.layout(); break;
      default: return;
    }
    e.preventDefault();
  }
}
