// Dialogs: models, preferences, keyboard shortcuts, about.
import { get, post, del } from './api.js';
import { h, fmtBytes } from './util.js';
import { icon } from './icons.js';
import { modal, alertError, confirmDlg } from './dialogs.js';
import { browse } from './filebrowser.js';

// ---------------------------------------------------------------- models

export function modelsDialog(app) {
  const tbody = h('tbody');
  let timer = null;
  const render = list => {
    tbody.replaceChildren(...list.map(m => {
      const d = m.download;
      const busy = d && !d.done;
      let act;
      if (busy) {
        act = h('div', { class: 'row' },
          h('div', { class: 'bar', style: { width: '120px' } }, h('i', { style: { width: (d.total ? d.received / d.total : 0) * 100 + '%' } })),
          h('span', { class: 'muted num' }, fmtBytes(d.received)),
          h('button', { class: 'iconbtn', title: 'Cancel download', html: icon('close'), onclick: () => post(`/models/${m.id}/cancel`).then(load) }));
      } else if (m.installed) {
        const what = m.kind === 'supir' ? 'folder' : 'file';
        act = h('button', { class: 'iconbtn danger', title: `Delete the model ${what}`, html: icon('trash'), onclick: async () => {
          if (!await confirmDlg(`Delete the model “${m.name}”?${m.builtIn ? ' You can download it again later.' : ` The ${what} will be removed from the models folder.`}`, { title: 'Delete model', ok: 'Delete', danger: true })) return;
          try { await del(`/models/${m.id}`); } catch (e) { alertError(e.message); }
          load();
        } });
      } else if (m.kind === 'supir') {
        act = h('button', { class: 'iconbtn danger', title: 'Delete the model folder', html: icon('trash'), onclick: async () => {
          if (!await confirmDlg(`Delete the incomplete model “${m.name}”? Its folder will be removed from the models folder.`, { title: 'Delete model', ok: 'Delete', danger: true })) return;
          try { await del(`/models/${m.id}`); } catch (e) { alertError(e.message); }
          load();
        } });
      } else {
        act = h('button', { class: 'btn small', html: icon('download') + `<span>Download</span>`, onclick: () => post(`/models/${m.id}/download`).then(load).catch(e => alertError(e.message)) });
      }
      return h('tr', null,
        h('td', null,
          h('div', null, h('b', null, m.name), m.installed ? h('span', { class: 'tag ok' }, 'Installed') : null, m.kind === 'supir' ? h('span', { class: 'tag' }, 'SUPIR') : m.kind === 'face' ? h('span', { class: 'tag' }, 'Faces') : !m.builtIn ? h('span', { class: 'tag' }, 'Custom') : null),
          h('div', { class: 'mdesc' }, m.description),
          d?.error ? h('div', { class: 'mdesc', style: { color: 'var(--err)' } }, 'Download failed: ' + d.error) : null,
          m.error ? h('div', { class: 'mdesc', style: { color: 'var(--err)' } }, m.error) : null,
          m.builtIn ? h('div', { class: 'mdesc' }, `${m.license} · ${fmtBytes(m.size)} · `, h('span', { class: 'mono' }, m.id)) : h('div', { class: 'mdesc' }, fmtBytes(m.size), ' · ', h('span', { class: 'mono' }, m.id + (m.kind === 'supir' ? '/' : '.onnx')))),
        h('td', { class: 'act' }, act));
    }));
  };
  const load = async () => {
    try {
      const list = await get('/models');
      render(list);
      app.setModels(list);
      clearTimeout(timer);
      if (list.some(m => m.download && !m.download.done)) timer = setTimeout(load, 400);
    } catch (e) {
      tbody.replaceChildren(h('tr', null, h('td', { colspan: 2, class: 'placeholder' }, e.message)));
    }
  };
  load();
  return modal({
    title: 'Models',
    width: 720,
    body: h('div', { class: 'form' },
      h('p', { class: 'hint', style: { margin: '0 0 6px' } },
        'Models are downloaded once and kept in the models folder. You can also add your own: any ESRGAN-style super-resolution network exported to ONNX (RGB input [1, 3, H, W] in 0–1, float32 or float16). The scale is detected automatically. SUPIR models are folders made from the original weights by tools/supir/export_supir.py. Face models (GFPGAN, CodeFormer) rebuild faces after upscaling; turn them on under Restore faces.'),
      h('table', { class: 'mtable' }, h('thead', null, h('tr', null, h('th', null, 'Model'), h('th', null, ''))), tbody)),
    buttons: [
      { label: 'Add model file…', align: 'left', onClick: async () => {
        const r = await browse({ mode: 'onnx', title: 'Add an ONNX model', multiple: false, start: app.state.lastModelDir, onDir: d => { app.state.lastModelDir = d; } });
        if (!r) return false;
        const { id } = await post('/models/add', { path: r[0] });
        await load();
        app.selectModel(id);
        return false;
      } },
      { label: 'Open models folder', align: 'left', onClick: async () => { await post('/fs/reveal', { path: app.system.modelsDir }); return false; } },
      { label: 'Close', primary: true, value: null },
    ],
  }).finally(() => clearTimeout(timer));
}

// ---------------------------------------------------------------- preferences

const GPU_NOTES = {
  DirectML: 'GPUs are used through DirectML (any DirectX 12 graphics card).',
  CUDA: 'NVIDIA GPUs are used through CUDA; it needs the NVIDIA driver, CUDA 12 and cuDNN 9.',
  CoreML: 'The GPU and Neural Engine are used through Core ML.',
};

export async function preferencesDialog(app) {
  const p = app.prefs;
  const s = app.settings;
  const theme = h('select', { class: 'inp' }, [['system', 'Same as the system'], ['light', 'Light'], ['dark', 'Dark']].map(([v, l]) => h('option', { value: v, selected: p.theme === v }, l)));
  const device = h('select', { class: 'inp' }, app.system.devices.map(d => h('option', { value: d.id, selected: s.device === d.id }, d.name + (d.detail ? ` — ${d.detail}` : ''))));
  const tile = h('select', { class: 'inp' }, [0, 128, 192, 256, 384, 512, 768, 1024].map(t => h('option', { value: t, selected: (s.tile || 0) === t }, t ? `${t} px` : 'Automatic')));
  const pixel = h('input', { type: 'checkbox', checked: p.pixelated });
  const r = await modal({
    title: 'Preferences',
    width: 560,
    body: h('div', { class: 'form2' },
      h('label', { class: 'frow' }, h('span', null, 'Theme'), theme),
      h('div', { class: 'frow' }, h('span', null, 'Viewer'), h('label', { class: 'chk' }, pixel, 'Show pixels sharply when zoomed in to 200% and more')),
      h('div', { class: 'sm-sep' }, 'Processing'),
      h('label', { class: 'frow' }, h('span', null, 'Device'), device),
      h('label', { class: 'frow' }, h('span', null, 'Tile size'), tile),
      h('div', { class: 'hint' }, 'Images are processed in tiles. Larger tiles are a little faster but need more memory; if the GPU runs out, the tile size is halved automatically. SUPIR has its own tiles (1024 px, smaller if memory runs out). ' +
        (GPU_NOTES[app.system.gpuBackend] || 'This build runs on the CPU. For an NVIDIA GPU on Linux, use the linux-x64-cuda build.')),
      app.system.gpuError ? h('div', { class: 'hint', style: { color: 'var(--err)' } }, 'The GPU could not be used: ' + app.system.gpuError) : null),
    buttons: [{ label: 'OK', primary: true, value: true }, { label: 'Cancel', value: false }],
  });
  if (!r) return;
  p.theme = theme.value;
  p.pixelated = pixel.checked;
  s.device = device.value;
  s.tile = parseInt(tile.value, 10) || 0;
  app.applyPrefs();
  app.settingsChanged();
}

// ---------------------------------------------------------------- help

export function shortcutsDialog() {
  const rows = [
    ['Ctrl+O', 'Open images'], ['Ctrl+Shift+O', 'Open a folder'], ['Ctrl+V', 'Paste an image'], ['Ctrl+Enter', 'Upscale'], ['Esc', 'Cancel'],
    ['Ctrl+M', 'Models'], ['Ctrl+Shift+E', 'Show the result in its folder'],
    ['C / S / B / A', 'Viewer: compare slider, side by side, original, result'],
    ['F or 0 / 1', 'Viewer: fit / 100%'], ['+ / -', 'Viewer: zoom in / out'], ['← / →', 'Viewer: move the slider'],
  ];
  return modal({
    title: 'Keyboard shortcuts',
    width: 480,
    body: h('table', { class: 'kbd-table' }, rows.map(([k, d]) => h('tr', null, h('td', null, k.split(' / ').map((x, i) => [i ? ' / ' : '', ...x.split('+').flatMap((y, j) => [j ? '+' : '', h('kbd', null, y)])])), h('td', null, d)))),
  });
}

export function aboutDialog(app) {
  const v = app.version;
  return modal({
    title: 'About ZawUpscale',
    width: 460,
    body: h('div', { class: 'about' }, h('img', { src: 'icon-256.png', width: 64, height: 64, alt: '' }),
      h('div', null, h('h2', null, 'ZawUpscale'),
        h('p', null, `Version ${v?.version ?? '?'}`),
        h('p', null, 'AI image upscaling for single images and whole folders, with Real-ESRGAN models running on your GPU or CPU, and SUPIR on your GPU.'),
        h('p', { class: 'muted' }, `ONNX Runtime ${v?.onnxRuntime ?? '?'} · .NET ${v?.runtime ?? ''}`),
        h('p', { class: 'muted' }, v?.os ?? ''))),
  });
}
