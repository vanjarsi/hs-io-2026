/* Web Worker wrapper: keeps model loading and inference off the UI thread. */
'use strict';
let loaded = false;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      if (!loaded) {
        importScripts(m.ortUrl, 'engine.js');
        loaded = true;
      }
      const info = await self.LeafEngine.init(m);
      self.postMessage({ id: m.id, ok: true, result: info });
    } else if (m.type === 'run') {
      const res = await self.LeafEngine.run(m);
      const transfer = res.transfer;
      delete res.transfer;
      self.postMessage({ id: m.id, ok: true, result: res }, transfer);
    }
  } catch (err) {
    self.postMessage({ id: m.id, ok: false, error: String((err && err.message) || err) });
  }
};
