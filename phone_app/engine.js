/* Inference engine for YOLO-seg ONNX models.
 * Runs inside a Web Worker (preferred, keeps the UI responsive) or on the main
 * thread (fallback). Every model setting comes from model/config.json, and the
 * real output names/shapes are read from the session, never assumed.
 *
 * Pipeline, matching Ultralytics predict():
 *   letterbox (bilinear, pad 114) -> /255 NCHW float32 -> model
 *   -> score = max class score, keep score > conf -> per-class NMS
 *   -> mask logits = coeffs . prototypes, bilinear-upsampled to input size,
 *      thresholded, then cropped to the box.
 */
(function (g) {
  'use strict';

  let session = null;
  let cfg = null;
  let io = null;

  const now = () => performance.now();
  // Python-style round (half to even), so letterbox sizes match Ultralytics exactly.
  function pyRound(v) {
    const f = Math.floor(v);
    const d = v - f;
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1;
    return f % 2 === 0 ? f : f + 1;
  }

  async function init(msg) {
    cfg = msg.config;
    const ort = g.ort;
    ort.env.wasm.wasmPaths = msg.ortBase;
    ort.env.wasm.numThreads = msg.threads;
    const bytes = new Uint8Array(msg.model);
    let backend = msg.backend || 'wasm';
    let threads = msg.threads;
    let note = null;

    const t0 = now();
    const create = () => ort.InferenceSession.create(bytes, { executionProviders: [backend], graphOptimizationLevel: 'all' });
    try {
      session = await create();
    } catch (err) {
      if (backend === 'wasm' && threads === 1) throw err;
      note = `${backend}${threads > 1 ? ' x' + threads : ''} failed (${String(err && err.message || err).slice(0, 80)}); using single-thread WASM`;
      backend = 'wasm';
      threads = 1;
      ort.env.wasm.numThreads = 1;
      session = await create();
    }
    const createMs = now() - t0;

    // Warm-up run on a blank input: reads the real output shapes and takes the one-off first-run cost.
    const S = cfg.imgsz;
    const inputName = session.inputNames[0];
    const t1 = now();
    const out = await session.run({ [inputName]: new ort.Tensor('float32', new Float32Array(3 * S * S), [1, 3, S, S]) });
    const warmupMs = now() - t1;

    const outputs = session.outputNames.map((n) => ({ name: n, shape: Array.from(out[n].dims) }));
    const det = outputs.find((o) => o.shape.length === 3);
    const proto = outputs.find((o) => o.shape.length === 4);
    if (!det || !proto) throw new Error('This is not a YOLO segmentation model: expected one 3-D and one 4-D output, got ' + JSON.stringify(outputs));
    const nm = proto.shape[1];
    const nc = Math.min(det.shape[1], det.shape[2]) - 4 - nm;
    if (nc !== cfg.classes.length) throw new Error(`Model has ${nc} classes but config.json lists ${cfg.classes.length}`);
    io = { inputName, detName: det.name, protoName: proto.name };
    return { backend, threads, note, createMs, warmupMs, inputName, outputs, numClasses: nc, numMaskCoeffs: nm };
  }

  // Bilinear letterbox with OpenCV INTER_LINEAR sampling (half-pixel centres), straight into a CHW float tensor.
  function preprocess(rgba, sw, sh) {
    const S = cfg.imgsz;
    const r = Math.min(S / sh, S / sw);
    const nw = pyRound(sw * r);
    const nh = pyRound(sh * r);
    const left = pyRound((S - nw) / 2 - 0.1);
    const top = pyRound((S - nh) / 2 - 0.1);
    const area = S * S;
    const out = new Float32Array(3 * area).fill(cfg.pad_value / 255);

    const xa = new Int32Array(nw), xb = new Int32Array(nw), xf = new Float32Array(nw);
    const scx = sw / nw, scy = sh / nh;
    for (let x = 0; x < nw; x++) {
      let f = (x + 0.5) * scx - 0.5;
      let i = Math.floor(f);
      f -= i;
      if (i < 0) { i = 0; f = 0; }
      if (i >= sw - 1) { i = sw - 1; f = 0; }
      xa[x] = i * 4;
      xb[x] = Math.min(i + 1, sw - 1) * 4;
      xf[x] = f;
    }
    const same = nw === sw && nh === sh;
    for (let y = 0; y < nh; y++) {
      let f = (y + 0.5) * scy - 0.5;
      let i = Math.floor(f);
      f -= i;
      if (i < 0) { i = 0; f = 0; }
      if (i >= sh - 1) { i = sh - 1; f = 0; }
      const r0 = i * sw * 4;
      const r1 = Math.min(i + 1, sh - 1) * sw * 4;
      const o = (top + y) * S + left;
      if (same) {
        for (let x = 0; x < nw; x++) {
          const p = r0 + x * 4;
          out[o + x] = rgba[p] / 255;
          out[area + o + x] = rgba[p + 1] / 255;
          out[2 * area + o + x] = rgba[p + 2] / 255;
        }
        continue;
      }
      for (let x = 0; x < nw; x++) {
        const a = r0 + xa[x], b = r0 + xb[x], c = r1 + xa[x], d = r1 + xb[x];
        const fx = xf[x];
        for (let ch = 0; ch < 3; ch++) {
          const t = rgba[a + ch] + (rgba[b + ch] - rgba[a + ch]) * fx;
          const u = rgba[c + ch] + (rgba[d + ch] - rgba[c + ch]) * fx;
          out[ch * area + o + x] = Math.round(t + (u - t) * f) / 255;
        }
      }
    }
    return { data: out, lb: { r, left, top, nw, nh, S } };
  }

  function decode(out, lb, sw, sh, conf, iouThr) {
    const det = out[io.detName];
    const proto = out[io.protoName];
    const nm = proto.dims[1], mh = proto.dims[2], mw = proto.dims[3];
    // Layout is [1, C, N] (Ultralytics default) or [1, N, C]; C = 4 + classes + mask coeffs.
    let C = det.dims[1], N = det.dims[2], cs, is;
    if (C <= N) { cs = N; is = 1; } else { const t = C; C = N; N = t; cs = 1; is = C; }
    const nc = C - 4 - nm;
    const d = det.data;
    const S = lb.S;

    const cand = [];
    for (let i = 0; i < N; i++) {
      const base = i * is;
      let best = 0, bc = 0;
      for (let c = 0; c < nc; c++) {
        const s = d[(4 + c) * cs + base];
        if (s > best) { best = s; bc = c; }
      }
      if (best > conf) {
        const cx = d[base], cy = d[cs + base], w = d[2 * cs + base], h = d[3 * cs + base];
        cand.push({ i, c: bc, s: best, x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 });
      }
    }
    cand.sort((a, b) => b.s - a.s);
    if (cand.length > 30000) cand.length = 30000;

    // Per-class NMS: a box is dropped if it overlaps a kept, higher-scoring box of the same class by more than iouThr.
    const keep = [];
    const maxDet = cfg.max_det || 300;
    for (const a of cand) {
      let ok = true;
      const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
      for (const k of keep) {
        if (k.c !== a.c) continue;
        const iw = Math.min(a.x2, k.x2) - Math.max(a.x1, k.x1);
        const ih = Math.min(a.y2, k.y2) - Math.max(a.y1, k.y1);
        if (iw <= 0 || ih <= 0) continue;
        const inter = iw * ih;
        if (inter / (areaA + (k.x2 - k.x1) * (k.y2 - k.y1) - inter) > iouThr) { ok = false; break; }
      }
      if (ok) {
        keep.push(a);
        if (keep.length >= maxDet) break;
      }
    }

    const t = cfg.mask_threshold == null ? 0.5 : cfg.mask_threshold;
    const logitThr = Math.log(t / (1 - t)); // sigmoid(x) > t  <=>  x > logit(t); 0 for t = 0.5
    const pd = proto.data;
    const plane = mh * mw;
    const sx = mw / S, sy = mh / S;
    const coef = new Float32Array(nm);
    const dets = [];
    const transfer = [];

    for (const a of keep) {
      // Pixels inside the box, as Ultralytics crop_mask: X >= x1 and X < x2.
      const X0 = Math.max(0, Math.ceil(a.x1)), X1 = Math.min(S - 1, Math.ceil(a.x2) - 1);
      const Y0 = Math.max(0, Math.ceil(a.y1)), Y1 = Math.min(S - 1, Math.ceil(a.y2) - 1);
      const w = Math.max(0, X1 - X0 + 1), h = Math.max(0, Y1 - Y0 + 1);
      const data = new Uint8Array(w * h);
      let area = 0;
      if (w > 0 && h > 0) {
        // Bilinear sampling tables, PyTorch F.interpolate(align_corners=False) convention.
        const xa = new Int32Array(w), xb = new Int32Array(w), xf = new Float32Array(w);
        const ya = new Int32Array(h), yb = new Int32Array(h), yf = new Float32Array(h);
        for (let x = 0; x < w; x++) {
          const f = Math.max(0, (X0 + x + 0.5) * sx - 0.5), i = Math.floor(f);
          xa[x] = i; xb[x] = Math.min(i + 1, mw - 1); xf[x] = f - i;
        }
        for (let y = 0; y < h; y++) {
          const f = Math.max(0, (Y0 + y + 0.5) * sy - 0.5), i = Math.floor(f);
          ya[y] = i; yb[y] = Math.min(i + 1, mh - 1); yf[y] = f - i;
        }
        const px0 = xa[0], px1 = xb[w - 1], py0 = ya[0], py1 = yb[h - 1];
        const pw = px1 - px0 + 1, ph = py1 - py0 + 1;
        // Mask logits in prototype space, only for the region this box needs.
        for (let k = 0; k < nm; k++) coef[k] = d[(4 + nc + k) * cs + a.i * is];
        const L = new Float32Array(pw * ph);
        for (let k = 0; k < nm; k++) {
          const ck = coef[k];
          const off = k * plane + px0;
          for (let y = 0; y < ph; y++) {
            const ro = off + (py0 + y) * mw, lo = y * pw;
            for (let x = 0; x < pw; x++) L[lo + x] += ck * pd[ro + x];
          }
        }
        for (let y = 0; y < h; y++) {
          const ra = (ya[y] - py0) * pw, rb = (yb[y] - py0) * pw, fy = yf[y];
          for (let x = 0; x < w; x++) {
            const ia = xa[x] - px0, ib = xb[x] - px0, fx = xf[x];
            const top = L[ra + ia] + (L[ra + ib] - L[ra + ia]) * fx;
            const bot = L[rb + ia] + (L[rb + ib] - L[rb + ia]) * fx;
            if (top + (bot - top) * fy > logitThr) { data[y * w + x] = 1; area++; }
          }
        }
      }
      const clampX = (v) => Math.min(sw, Math.max(0, (v - lb.left) / lb.r));
      const clampY = (v) => Math.min(sh, Math.max(0, (v - lb.top) / lb.r));
      dets.push({
        cls: a.c,
        score: a.s,
        box: [clampX(a.x1), clampY(a.y1), clampX(a.x2), clampY(a.y2)], // source-image pixels
        boxInput: [a.x1, a.y1, a.x2, a.y2],                            // model-input pixels
        mask: { x: X0, y: Y0, w, h, data },                            // model-input pixels
        areaPx: area / (lb.r * lb.r),                                  // source-image pixels squared
      });
      transfer.push(data.buffer);
    }
    return { dets, transfer, candidates: cand.length };
  }

  async function run(msg) {
    const ort = g.ort;
    const rgba = new Uint8ClampedArray(msg.rgba);
    const t0 = now();
    const pre = preprocess(rgba, msg.sw, msg.sh);
    const S = cfg.imgsz;
    const tensor = new ort.Tensor('float32', pre.data, [1, 3, S, S]);
    const t1 = now();
    const out = await session.run({ [io.inputName]: tensor });
    const t2 = now();
    const res = decode(out, pre.lb, msg.sw, msg.sh, msg.conf == null ? cfg.conf : msg.conf, msg.iou == null ? cfg.iou : msg.iou);
    const t3 = now();
    return {
      dets: res.dets,
      candidates: res.candidates,
      lb: pre.lb,
      sw: msg.sw,
      sh: msg.sh,
      timing: { tensorMs: t1 - t0, inferMs: t2 - t1, postMs: t3 - t2 },
      transfer: res.transfer,
    };
  }

  g.LeafEngine = { init, run };
})(typeof self !== 'undefined' ? self : window);
