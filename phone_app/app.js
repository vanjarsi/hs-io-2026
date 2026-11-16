/* Leaf Check UI. All model settings come from model/config.json (written by export_model.sh). */
(function () {
  'use strict';

  // onnxruntime-web, pinned to an exact released version. The .wasm files load from the same folder.
  const ORT_VERSION = '1.20.1';
  const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  const ORT_URL = ORT_BASE + 'ort.wasm.min.js';

  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  // The final model: Run D, YOLO11n-seg trained at 1024 px.
  const MODEL_DIR = 'model_final/';
  const now = () => performance.now();

  let cfg = null;
  let engine = null;      // { run(msg, transfer) }
  let modelInfo = null;
  let busy = false;
  let last = null;        // { bitmap, crop, res, sample }
  let lastInput = null;   // { source, sample } for re-running when an option changes

  // ---------- small helpers ----------
  const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)}<small>s</small>` : `${ms < 10 ? ms.toFixed(1) : Math.round(ms)}<small>ms</small>`);
  const fmtMB = (bytes) => `${(bytes / 1e6).toFixed(1)}<small>MB</small>`;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  function showError(msg) { $('err').textContent = msg; console.error(msg); }

  // ---------- engine: Web Worker first, main thread as a fallback ----------
  function workerEngine() {
    return new Promise((resolve, reject) => {
      let w;
      try { w = new Worker('worker.js'); } catch (e) { reject(e); return; }
      let seq = 0;
      const pending = new Map();
      w.onmessage = (e) => {
        const p = pending.get(e.data.id);
        if (!p) return;
        pending.delete(e.data.id);
        if (e.data.ok) p.resolve(e.data.result); else p.reject(new Error(e.data.error));
      };
      w.onerror = (e) => {
        const err = new Error('worker error: ' + (e.message || 'failed to start'));
        pending.forEach((p) => p.reject(err));
        pending.clear();
        reject(err);
      };
      const call = (msg, transfer) => new Promise((res, rej) => {
        const id = ++seq;
        pending.set(id, { resolve: res, reject: rej });
        w.postMessage(Object.assign({ id }, msg), transfer || []);
      });
      resolve({ kind: 'worker', terminate: () => w.terminate(), init: (m) => call(Object.assign({ type: 'init' }, m)), run: (m, t) => call(Object.assign({ type: 'run' }, m), t) });
    });
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src;
      s.onload = resolve;
      s.onerror = () => reject(new Error('could not load ' + src));
      document.head.appendChild(s);
    });
  }

  async function mainThreadEngine() {
    if (!window.ort) await loadScript(ORT_URL);
    if (!window.LeafEngine) await loadScript('engine.js');
    return {
      kind: 'main',
      init: (m) => window.LeafEngine.init(m),
      run: async (m) => { const r = await window.LeafEngine.run(m); delete r.transfer; return r; },
    };
  }

  // ---------- model loading ----------
  async function fetchModel(url, expectedBytes) {
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`model download failed (${res.status})`);
    const total = Number(res.headers.get('content-length')) || expectedBytes || 0;
    if (!res.body || !res.body.getReader) return await res.arrayBuffer();
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      if (total) $('modelBar').style.width = Math.min(100, (got / total) * 100) + '%';
    }
    const buf = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) { buf.set(c, o); o += c.length; }
    return buf.buffer;
  }

  async function loadModel() {
    const t0 = now();
    cfg = await (await fetch(MODEL_DIR + 'config.json', { cache: 'no-cache' })).json();
    $('modelName').textContent = cfg.model_name || 'Segmentation model';
    applyConfigToUi();

    // ?model=name.onnx loads another file from model/ (used to compare FP16 against FP32)
    const modelFile = (params.get('model') || cfg.model_file).replace(/[^\w.-]/g, '');
    let model = await fetchModel(MODEL_DIR + modelFile, modelFile === cfg.model_file ? cfg.model_bytes : 0);
    let bytes = model.byteLength;
    let precision = modelFile === cfg.fallback_model_file ? cfg.fallback_precision : cfg.precision;
    const tDownload = now() - t0;
    $('mSize').innerHTML = fmtMB(bytes);
    $('modelState').textContent = 'starting';

    const wantThreads = params.has('threads') ? Math.max(1, parseInt(params.get('threads'), 10) || 1)
      : (self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1);
    const init = { config: cfg, ortUrl: ORT_URL, ortBase: ORT_BASE, threads: wantThreads, backend: 'wasm' };

    const t1 = now();
    let info;
    let fallbackNote = null;
    // A start that hangs (seen with threaded WASM on some phones) must not leave the page stuck on "starting".
    const withTimeout = (p, ms, what) => new Promise((res, rej) => {
      const t = setTimeout(() => rej(new Error(what + ' timed out')), ms);
      p.then((v) => { clearTimeout(t); res(v); }, (e) => { clearTimeout(t); rej(e); });
    });
    const inWorker = async (buf, threads) => {
      engine = await workerEngine();
      try {
        return await withTimeout(engine.init(Object.assign({ model: buf.slice(0) }, init, { threads })), 45000, 'start-up');
      } catch (err) {
        engine.terminate();
        throw err;
      }
    };
    const start = async (buf) => {
      try {
        if (params.get('worker') === '0') throw new Error('worker disabled by ?worker=0');
        try {
          return await inWorker(buf, wantThreads);
        } catch (err) {
          if (wantThreads === 1) throw err;
          const info1 = await inWorker(buf, 1); // fresh worker, single thread
          info1.note = `${wantThreads}-thread start failed (${String(err.message).slice(0, 60)}); using 1 thread`;
          return info1;
        }
      } catch (err) {
        fallbackNote = `worker unavailable (${String(err.message).slice(0, 60)}); running on the main thread`;
        engine = await mainThreadEngine();
        return await engine.init(Object.assign({ model: buf }, init));
      }
    };
    try {
      info = await start(model);
    } catch (err) {
      // e.g. a device that cannot run the FP16 file: use the FP32 copy exported next to it.
      if (!cfg.fallback_model_file || modelFile === cfg.fallback_model_file) throw err;
      const why = String(err.message || err).slice(0, 80);
      model = await fetchModel(MODEL_DIR + cfg.fallback_model_file, cfg.fallback_model_bytes);
      bytes = model.byteLength;
      precision = cfg.fallback_precision || precision;
      $('mSize').innerHTML = fmtMB(bytes);
      info = await start(model);
      fallbackNote = `${modelFile} did not start (${why}); using ${cfg.fallback_model_file}`;
    }
    const tInit = now() - t1;

    modelInfo = { bytes, file: modelFile, precision, downloadMs: tDownload, initMs: tInit, createMs: info.createMs, warmupMs: info.warmupMs, backend: info.backend,
      threads: info.threads, engine: engine.kind, outputs: info.outputs, inputName: info.inputName, note: info.note || fallbackNote };
    $('mLoad').innerHTML = fmtMs(tDownload + tInit);
    $('mBackend').innerHTML = `WASM<small> ×${info.threads}</small>`;
    $('modelState').textContent = 'ready';
    $('modelState').classList.add('ready');
    $('modelProgress').classList.add('done');
    const outs = info.outputs.map((o) => `${o.name} [${o.shape.join('×')}]`).join(', ');
    $('modelDetail').textContent =
      `Download ${Math.round(tDownload)} ms, start-up ${Math.round(info.createMs)} ms, warm-up run ${Math.round(info.warmupMs)} ms. ` +
      `${String(precision || '').toUpperCase()} ONNX, onnxruntime-web ${ORT_VERSION}, ${info.threads} thread${info.threads > 1 ? 's' : ''}${engine.kind === 'worker' ? ', in a background worker' : ''}. ` +
      `Outputs: ${outs}.` + (modelInfo.note ? ` Note: ${modelInfo.note}.` : '');
    setEnabled(true);
    return modelInfo;
  }

  function applyConfigToUi() {
    const pp = cfg.phone_photo || {};
    $('optMatch').checked = params.has('match') ? params.get('match') !== '0' : pp.match_training_resolution !== false;
    $('optCrop').checked = params.has('crop') ? params.get('crop') !== '0' : pp.center_crop_square !== false;
    const px = pp.train_image_px;
    $('optMatchHint').textContent = px
      ? `The model learned from ${px}×${px} px close-ups. On: the photo is first shrunk to ${px} px, so spots appear at the size the model knows. Off: the photo goes in at ${cfg.imgsz} px.`
      : 'No training image size in the model config, so this has no effect.';
    const parts = [];
    if (cfg.model_name) parts.push(cfg.model_name);
    if (cfg.training_run) parts.push(cfg.training_run);
    parts.push(`input ${cfg.imgsz}×${cfg.imgsz}`);
    parts.push(`confidence ≥ ${cfg.conf}`);
    parts.push(`NMS IoU ${cfg.iou}`);
    parts.push(`${String(cfg.precision || '').toUpperCase()} ONNX`);
    if (cfg.params) parts.push(`${(cfg.params / 1e6).toFixed(2)} M parameters`);
    if (cfg.export_date) parts.push(`exported ${cfg.export_date.slice(0, 10)}` + (cfg.ultralytics_version ? ` with Ultralytics ${cfg.ultralytics_version}` : ''));
    $('about').textContent = 'Model: ' + parts.join(' · ') + '. Classes: ' + cfg.classes.map((c) => c.label).join(', ') + '.';
  }

  function setEnabled(on) {
    $('camBtn').classList.toggle('disabled', !on);
    $('fileBtn').classList.toggle('disabled', !on);
    $('camBtn2').classList.toggle('disabled', !on);
    document.querySelectorAll('.sample').forEach((b) => { b.disabled = !on; });
  }

  // ---------- image handling ----------
  async function decodeImage(source) {
    let blob = source;
    if (typeof source === 'string') blob = await (await fetch(source)).blob();
    try {
      // Applies the EXIF rotation that phone cameras store instead of rotating the pixels.
      return await createImageBitmap(blob, { imageOrientation: 'from-image' });
    } catch (e) {
      const url = URL.createObjectURL(blob);
      try {
        const img = new Image();
        img.decoding = 'async';
        img.src = url;
        await img.decode();
        return img; // <img> also honours EXIF orientation in current browsers
      } finally { setTimeout(() => URL.revokeObjectURL(url), 30000); }
    }
  }
  const dims = (im) => ({ w: im.naturalWidth || im.width, h: im.naturalHeight || im.height });

  function canvasOf(w, h) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    return c;
  }

  // Crop (optional), then shrink to the working size. Shrinks in halving steps so big photos are
  // averaged rather than aliased. Never enlarges: the engine's letterbox does the upscaling.
  function makeSource(bitmap, opts) {
    const d = dims(bitmap);
    let cx = 0, cy = 0, cw = d.w, ch = d.h;
    if (opts.crop && cw !== ch) {
      const s = Math.min(cw, ch);
      cx = Math.floor((cw - s) / 2); cy = Math.floor((ch - s) / 2); cw = ch = s;
    }
    const pp = cfg.phone_photo || {};
    const longTarget = opts.match && pp.train_image_px ? pp.train_image_px : cfg.imgsz;
    const scale = Math.min(1, longTarget / Math.max(cw, ch));
    const tw = Math.max(1, Math.round(cw * scale)), th = Math.max(1, Math.round(ch * scale));

    let cur = bitmap, sx = cx, sy = cy, sw = cw, sh = ch;
    while (sw / 2 >= tw * 1.0001 && sh / 2 >= th * 1.0001) {
      const nw = Math.max(tw, Math.ceil(sw / 2)), nh = Math.max(th, Math.ceil(sh / 2));
      const c = canvasOf(nw, nh);
      const x = c.getContext('2d');
      x.imageSmoothingEnabled = true; x.imageSmoothingQuality = 'high';
      x.drawImage(cur, sx, sy, sw, sh, 0, 0, nw, nh);
      cur = c; sx = 0; sy = 0; sw = nw; sh = nh;
    }
    const c = canvasOf(tw, th);
    const x = c.getContext('2d', { willReadFrequently: true });
    x.imageSmoothingEnabled = !(sw === tw && sh === th);
    x.imageSmoothingQuality = 'high';
    x.drawImage(cur, sx, sy, sw, sh, 0, 0, tw, th);
    const data = x.getImageData(0, 0, tw, th).data;
    return { rgba: data, sw: tw, sh: th, crop: { x: cx, y: cy, w: cw, h: ch }, scale };
  }

  // ---------- run ----------
  async function analyze(source, sample, overrides) {
    if (!engine || busy) return null;
    busy = true;
    lastInput = { source, sample };
    const opts = Object.assign({ match: $('optMatch').checked, crop: $('optCrop').checked }, overrides || {});
    $('result').hidden = false;
    $('busy').hidden = false;
    setVerdict('pending', '…', 'Analysing', '');
    try {
      const t0 = now();
      const bitmap = await decodeImage(source);
      const src = makeSource(bitmap, opts);
      const tMain = now() - t0;
      const res = await engine.run({ rgba: src.rgba.buffer, sw: src.sw, sh: src.sh, conf: opts.conf, iou: opts.iou }, [src.rgba.buffer]);
      res.timing.preMs = tMain + res.timing.tensorMs;
      last = { bitmap, crop: src.crop, res, sample, opts, original: dims(bitmap) };
      const td = now();
      draw();
      res.timing.drawMs = now() - td;
      res.timing.totalMs = now() - t0;
      report();
      return res;
    } catch (err) {
      setVerdict('failed', '!', 'Could not analyse this image', String(err.message || err));
      showError(String(err.message || err));
      return null;
    } finally {
      busy = false;
      $('busy').hidden = true;
    }
  }

  function setVerdict(kind, icon, title, sub) {
    $('verdict').className = 'verdict ' + kind;
    $('verdict').innerHTML = `<span class="icon" aria-hidden="true">${esc(icon)}</span><span><b>${esc(title)}</b><small>${esc(sub)}</small></span>`;
  }

  function report() {
    const { res, sample, opts, original, crop } = last;
    const n = res.dets.length;
    const counts = cfg.classes.map(() => 0);
    res.dets.forEach((d) => { counts[d.cls]++; });
    let truth = '';
    if (sample && sample.true_spots != null) truth = sample.true_spots === 0 ? 'Test image labelled healthy (no spots).' : `Test image with ${sample.true_spots} labelled spots.`;
    if (n === 0) {
      setVerdict('healthy', '✓', 'Healthy, no disease spots found', truth || `Nothing above ${Math.round(cfg.conf * 100)}% confidence.`);
    } else {
      const kinds = cfg.classes.filter((c, i) => counts[i]).map((c) => c.label).join(', ');
      setVerdict('found', n > 99 ? '99+' : String(n), `${n} disease spot${n === 1 ? '' : 's'} found: ${kinds}`, truth);
    }
    $('counts').innerHTML = cfg.classes.map((c, i) => (counts[i]
      ? `<span class="chip"><i class="swatch" style="background:${esc(c.color)}"></i>${esc(c.label)} × ${counts[i]}</span>` : '')).join('');

    const t = res.timing;
    $('tPre').innerHTML = fmtMs(t.preMs);
    $('tInf').innerHTML = fmtMs(t.inferMs);
    $('tPost').innerHTML = fmtMs(t.postMs);
    $('tTotal').innerHTML = fmtMs(t.totalMs);
    const cropped = crop.w !== original.w || crop.h !== original.h;
    $('runDetail').textContent =
      `Photo ${original.w}×${original.h} px` + (cropped ? `, centre-cropped to ${crop.w}×${crop.h}` : '') +
      (res.sw !== crop.w ? `, shrunk to ${res.sw}×${res.sh}` : '') +
      `, letterboxed to ${cfg.imgsz}×${cfg.imgsz}. Total includes decoding and drawing (${Math.round(t.drawMs)} ms).`;

    $('detBox').hidden = n === 0;
    $('detSummary').textContent = `Spots found (${n})`;
    $('detList').innerHTML = res.dets.map((d) => {
      const c = cfg.classes[d.cls];
      const pct = Math.round(d.score * 100);
      return `<li><i class="swatch" style="background:${esc(c.color)}"></i>` +
        `<span>${esc(c.label)} <small>· ${Math.max(1, Math.round(Math.sqrt(d.areaPx)))} px across</small><div class="bar"><i style="width:${pct}%"></i></div></span>` +
        `<span class="pct">${pct}%</span></li>`;
    }).join('');
  }

  // ---------- drawing ----------
  function hexToRgb(hex) {
    const m = /^#?([0-9a-f]{6})$/i.exec(hex || '');
    const v = m ? parseInt(m[1], 16) : 0xffffff;
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }

  function draw() {
    if (!last) return;
    const { bitmap, crop, res } = last;
    const canvas = $('view');
    const long = Math.max(crop.w, crop.h);
    const k0 = Math.min(1600, Math.max(960, long)) / long;
    const cw = Math.round(crop.w * k0), chh = Math.round(crop.h * k0);
    canvas.width = cw; canvas.height = chh;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bitmap, crop.x, crop.y, crop.w, crop.h, 0, 0, cw, chh);
    if (!$('showMasks').checked) return;

    // model-input pixel -> canvas pixel
    const k = cw / (res.lb.r * res.sw);
    const X = (x) => (x - res.lb.left) * k;
    const Y = (y) => (y - res.lb.top) * k;
    const lineW = Math.max(2, cw / 260);

    for (const d of res.dets) {
      const m = d.mask;
      if (!m.w || !m.h) continue;
      const [r, g, b] = hexToRgb(cfg.classes[d.cls].color);
      const patch = canvasOf(m.w, m.h);
      const pctx = patch.getContext('2d');
      const img = pctx.createImageData(m.w, m.h);
      const path = new Path2D();
      for (let y = 0; y < m.h; y++) {
        for (let x = 0; x < m.w; x++) {
          const i = y * m.w + x;
          if (!m.data[i]) continue;
          const o = i * 4;
          img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 105;
          // Outline: one segment for each pixel side that borders a non-mask pixel.
          const x0 = X(m.x + x), x1 = X(m.x + x + 1), y0 = Y(m.y + y), y1 = Y(m.y + y + 1);
          if (y === 0 || !m.data[i - m.w]) { path.moveTo(x0, y0); path.lineTo(x1, y0); }
          if (y === m.h - 1 || !m.data[i + m.w]) { path.moveTo(x0, y1); path.lineTo(x1, y1); }
          if (x === 0 || !m.data[i - 1]) { path.moveTo(x0, y0); path.lineTo(x0, y1); }
          if (x === m.w - 1 || !m.data[i + 1]) { path.moveTo(x1, y0); path.lineTo(x1, y1); }
        }
      }
      pctx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(patch, X(m.x), Y(m.y), m.w * k, m.h * k);
      ctx.lineCap = 'round';
      ctx.strokeStyle = 'rgba(0,0,0,0.75)';
      ctx.lineWidth = lineW + 2;
      ctx.stroke(path);
      ctx.strokeStyle = `rgb(${r},${g},${b})`;
      ctx.lineWidth = lineW;
      ctx.stroke(path);
    }
  }

  // ---------- samples ----------
  async function loadSamples() {
    let list = [];
    try { list = await (await fetch('samples/samples.json', { cache: 'no-cache' })).json(); } catch (e) { /* none bundled */ }
    const box = $('samples');
    if (!list.length) { box.closest('section').hidden = true; return list; }
    const pick = async (s, scroll) => {
      document.querySelectorAll('.sample').forEach((x) => x.classList.toggle('active', x.dataset.file === s.file));
      if (scroll) scrollToResult();
      await analyze('samples/' + encodeURIComponent(s.file), s);
    };
    list.forEach((s) => {
      // Once in the main grid, once as a small thumbnail under the result (no scrolling needed between leaves).
      [[box, true], [$('quick'), false]].forEach(([parent, scroll]) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'sample';
        b.dataset.file = s.file;
        b.disabled = !engine;
        b.setAttribute('aria-label', 'Test leaf: ' + s.label);
        b.innerHTML = `<img src="samples/${encodeURIComponent(s.file)}" alt="" loading="lazy"><span>${esc(s.label)}</span>`;
        b.addEventListener('click', () => pick(s, scroll));
        parent.appendChild(b);
      });
    });
    return list;
  }

  function scrollToResult() {
    $('result').hidden = false;
    requestAnimationFrame(() => $('result').scrollIntoView({ behavior: 'smooth', block: 'start' }));
  }

  function onFile(e) {
    const f = e.target.files && e.target.files[0];
    e.target.value = '';
    if (!f) return;
    document.querySelectorAll('.sample').forEach((x) => x.classList.remove('active'));
    scrollToResult();
    analyze(f, null);
  }

  // ---------- start ----------
  $('camInput').addEventListener('change', onFile);
  $('fileInput').addEventListener('change', onFile);
  $('showMasks').addEventListener('change', draw);
  const rerun = () => { if (lastInput) analyze(lastInput.source, lastInput.sample); };
  $('optMatch').addEventListener('change', rerun);
  $('optCrop').addEventListener('change', rerun);
  setEnabled(false);

  const samplesReady = loadSamples();
  const ready = loadModel().catch((err) => {
    $('modelState').textContent = 'failed';
    $('modelState').classList.add('error');
    showError('Model failed to load: ' + String(err.message || err));
    throw err;
  });

  // Small API for automated checks (parity_check.py drives the page through this).
  function b64(u8) {
    let s = '';
    for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    return btoa(s);
  }
  window.leafApp = {
    ready,
    samplesReady,
    info: () => modelInfo,
    config: () => cfg,
    async analyzeUrl(url, overrides) {
      const res = await analyze(url, null, overrides);
      if (!res) return null;
      return {
        sw: res.sw, sh: res.sh, lb: res.lb, timing: res.timing, candidates: res.candidates,
        dets: res.dets.map((d) => ({
          cls: d.cls, name: cfg.classes[d.cls].name, score: d.score, box: d.box, boxInput: d.boxInput, areaPx: d.areaPx,
          mask: { x: d.mask.x, y: d.mask.y, w: d.mask.w, h: d.mask.h, b64: b64(d.mask.data) },
        })),
      };
    },
  };
})();
