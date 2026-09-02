import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const playwrightRoot = process.env.PLAYWRIGHT_PACKAGE ?? '/tmp/uf-pw/node_modules/playwright/index.mjs';
const { chromium } = await import(pathToFileURL(playwrightRoot).href);

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const frontend = process.env.URBAN_FLIGHTER_FRONTEND ?? 'http://127.0.0.1:5173';
const outDir = join(root, 'docs/showcase/landmarks');

const landmarks = [
  { id: 'louvre', name: 'Louvre, Paris', lat: 48.8606, lon: 2.3376 },
  { id: 'colosseum', name: 'Colosseum, Rome', lat: 41.8902, lon: 12.4922 },
  { id: 'westminster', name: 'Westminster, London', lat: 51.5007, lon: -0.1246 },
  { id: 'shibuya', name: 'Shibuya Crossing, Tokyo', lat: 35.6595, lon: 139.7004 },
  { id: 'sagrada', name: 'Sagrada Família, Barcelona', lat: 41.4036, lon: 2.1744 },
];

mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({
  headless: true,
  channel: process.env.URBAN_FLIGHTER_CHROME_CHANNEL ?? 'chrome',
});
const page = await browser.newPage({
  viewport: { width: 1600, height: 900 },
  deviceScaleFactor: 2,
});

const results = [];
for (const place of landmarks) {
  const url = `${frontend}/?shot=3d&lat=${place.lat}&lon=${place.lon}`;
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await page.addStyleTag({ content: '* { cursor: none !important; }' });
  try {
    await page.waitForSelector('[data-shot-ready="1"]', { timeout: 90_000 });
  } catch (error) {
    const debug = join(outDir, `${place.id}-timeout.png`);
    await page.screenshot({ path: debug, type: 'png' }).catch(() => {});
    results.push({ ...place, ok: false, error: error.message, file: debug });
    continue;
  }
  await page.waitForTimeout(3800);
  const file = `${place.id}_3d.png`;
  await page.screenshot({ path: join(outDir, file), type: 'png' });
  const buildings = Number((await page.getAttribute('[data-shot-ready]', 'data-building-count')) ?? 0);
  results.push({ ...place, ok: true, file, buildings });
  console.log(`${place.id}: ${buildings} buildings → ${file}`);
}

writeFileSync(join(outDir, 'meta.json'), `${JSON.stringify({
  captured_kst: new Date().toISOString().slice(0, 10),
  note: 'Trial 3D Lite stills at named landmarks. OSM prisms only — not photogrammetry.',
  results,
}, null, 2)}\n`);

await browser.close();
console.log(JSON.stringify(results, null, 2));
