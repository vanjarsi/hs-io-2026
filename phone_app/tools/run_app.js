#!/usr/bin/env node
/* Drives the real app in headless Chrome and saves its detections as JSON.
 * Used by parity_check.py and for speed measurements.
 *
 *   node tools/run_app.js --url http://127.0.0.1:8000/ --images a.jpg,b.jpg --out out.json
 *        [--query "threads=1"] [--shot page.png] [--opts '{"match":false}'] [--repeat 5]
 *        [--mobile] [--headed]
 *
 * Needs the `playwright` npm package and Google Chrome. If playwright is not
 * installed locally, point PLAYWRIGHT_DIR at a folder that contains it
 * (for example the npx cache of @playwright/mcp).
 */
const fs = require('fs');
const path = require('path');

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  if (i < 0) return dflt;
  const v = process.argv[i + 1];
  return v === undefined || v.startsWith('--') ? true : v;
}

function loadPlaywright() {
  const tries = ['playwright', 'playwright-core'];
  if (process.env.PLAYWRIGHT_DIR) tries.unshift(path.join(process.env.PLAYWRIGHT_DIR, 'playwright'));
  for (const t of tries) {
    try { return require(t); } catch (e) { /* next */ }
  }
  console.error('playwright not found: npm i playwright, or set PLAYWRIGHT_DIR');
  process.exit(2);
}

(async () => {
  const { chromium } = loadPlaywright();
  const base = arg('url', 'http://127.0.0.1:8000/');
  const query = arg('query', '');
  const images = String(arg('images', '')).split(',').filter(Boolean);
  const opts = JSON.parse(arg('opts', '{}'));
  const repeat = parseInt(arg('repeat', '1'), 10);
  const mobile = arg('mobile', false);

  const browser = await chromium.launch({ channel: 'chrome', headless: !arg('headed', false) });
  const ctx = await browser.newContext(mobile
    ? { viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true }
    : { viewport: { width: 1100, height: 900 } });
  const page = await ctx.newPage();
  const logs = [];
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') logs.push(`${m.type()}: ${m.text()}`.slice(0, 300)); });
  page.on('pageerror', (e) => logs.push('pageerror: ' + String(e.message).slice(0, 300)));

  await page.goto(base + (query ? '?' + query : ''), { waitUntil: 'load' });
  let info;
  try {
    info = await page.evaluate(async () => { await window.leafApp.ready; return window.leafApp.info(); });
  } catch (e) {
    console.error('model failed to load:', String(e.message).split('\n')[0]);
    console.error(logs.join('\n'));
    await browser.close();
    process.exit(1);
  }
  const isolated = await page.evaluate(() => self.crossOriginIsolated);
  const results = {};
  for (const img of images) {
    let url = img;
    if (fs.existsSync(img)) {
      const ext = path.extname(img).toLowerCase() === '.png' ? 'png' : 'jpeg';
      url = `data:image/${ext};base64,` + fs.readFileSync(img).toString('base64');
    }
    const runs = [];
    let res = null;
    for (let i = 0; i < repeat; i++) {
      res = await page.evaluate(([u, o]) => window.leafApp.analyzeUrl(u, o), [url, opts]);
      if (res) runs.push(res.timing);
    }
    if (res) res.runs = runs;
    results[path.basename(img)] = res;
  }
  const shot = arg('shot', null);
  if (shot) await page.screenshot({ path: shot, fullPage: true });
  const out = { url: base, query, opts, crossOriginIsolated: isolated, userAgent: await page.evaluate(() => navigator.userAgent), model: info, results, logs };
  const outFile = arg('out', null);
  if (outFile) fs.writeFileSync(outFile, JSON.stringify(out));
  const summary = Object.entries(results).map(([k, r]) => r
    ? `${k}: ${r.dets.length} dets, infer ${r.runs.map((t) => Math.round(t.inferMs)).join('/')} ms, pre ${r.timing.preMs.toFixed(1)} post ${r.timing.postMs.toFixed(1)} ms`
    : `${k}: FAILED`);
  console.log(`model: ${(info.bytes / 1e6).toFixed(2)} MB, engine=${info.engine}, threads=${info.threads}, isolated=${isolated}, download ${Math.round(info.downloadMs)} ms, create ${Math.round(info.createMs)} ms, warm-up ${Math.round(info.warmupMs)} ms${info.note ? ', note: ' + info.note : ''}`);
  console.log(summary.join('\n'));
  if (logs.length) console.log('console:\n' + logs.join('\n'));
  await browser.close();
})();
