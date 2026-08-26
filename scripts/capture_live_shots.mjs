import { copyFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

const playwrightRoot = process.env.PLAYWRIGHT_PACKAGE ?? '/tmp/uf-pw/node_modules/playwright/index.mjs';
const { chromium } = await import(pathToFileURL(playwrightRoot).href);

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const frontend = process.env.URBAN_FLIGHTER_FRONTEND ?? 'http://127.0.0.1:5173';
const outDir = join(root, 'docs/showcase/components');
const paperDir = join(root, 'paper/figures');

const shots = [
  { kind: '2d', file: 'simplecfd_2d_field_inha.png', paper: 'fig_live_2d.png', settleMs: 2200 },
  { kind: '3d', file: 'cockpit_3d_lite_inha.png', paper: 'fig_live_3d.png', settleMs: 3800 },
  { kind: 'map', file: 'geometry_loader_map_inha.png', paper: 'fig_live_map.png', selector: '.panel-map', settleMs: 1800 },
  { kind: 'radar', file: 'radar_3d_inha.png', paper: 'fig_live_radar.png', selector: '.slam-window', settleMs: 4200 },
  { kind: 'cockpit', file: 'cockpit_2d_inha.png', settleMs: 1800 },
];

async function waitReady(page, shot) {
  try {
    await page.waitForSelector('[data-shot-ready="1"]', { timeout: 90_000 });
  } catch (error) {
    const debugDir = join(outDir, 'debug');
    mkdirSync(debugDir, { recursive: true });
    await page.screenshot({ path: join(debugDir, `${shot.kind}-timeout.png`), type: 'png' }).catch(() => {});
    const ready = await page.getAttribute('.app-shell', 'data-shot-ready').catch(() => null);
    const html = await page.content().catch(() => '');
    writeFileSync(join(debugDir, `${shot.kind}-timeout.html`), html);
    throw new Error(`shot=${shot.kind} never became ready (data-shot-ready=${ready}): ${error.message}`);
  }
}

async function capture(page, shot) {
  await page.goto(`${frontend}/?shot=${shot.kind}`, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await page.addStyleTag({ content: '* { cursor: none !important; }' });
  await waitReady(page, shot);
  if (shot.kind === 'map') {
    await page.waitForSelector('.leaflet-tile-loaded', { timeout: 30_000 }).catch(() => {});
  }
  await page.waitForTimeout(shot.settleMs);
  const target = join(outDir, shot.file);
  if (shot.selector) {
    const handle = page.locator(shot.selector).first();
    await handle.waitFor({ state: 'visible', timeout: 15_000 });
    await handle.screenshot({ path: target, type: 'png' });
  } else {
    await page.screenshot({ path: target, type: 'png' });
  }
  if (shot.paper) {
    copyFileSync(target, join(paperDir, shot.paper));
  }
  const buildings = await page.getAttribute('[data-shot-ready]', 'data-building-count');
  return { file: shot.file, paper: shot.paper ?? null, buildings: Number(buildings ?? 0) };
}

mkdirSync(outDir, { recursive: true });
mkdirSync(paperDir, { recursive: true });

const browser = await chromium.launch({
  headless: true,
  channel: process.env.URBAN_FLIGHTER_CHROME_CHANNEL ?? 'chrome',
});
const page = await browser.newPage({
  viewport: { width: 1600, height: 900 },
  deviceScaleFactor: 2,
});
await page.addStyleTag({ content: '* { cursor: none !important; }' }).catch(() => {});

const captured = [];
for (const shot of shots) {
  captured.push(await capture(page, shot));
}

const meta = {
  captured_kst: new Date().toISOString().slice(0, 10),
  frontend,
  backend: 'http://127.0.0.1:8000',
  location: { name: 'Inha/Incheon', lat: 37.451448, lon: 126.651542 },
  viewport: { width: 1600, height: 900, deviceScaleFactor: 2 },
  buildings: captured[0]?.buildings ?? 0,
  files: captured.map((row) => row.file),
  paper_figures: captured.flatMap((row) => (row.paper ? [row.paper] : [])),
  note: 'Fullscreen 2D/3D shots hide chrome. Map and radar are element crops so floating windows are not cut.',
};
writeFileSync(join(outDir, 'capture_meta.json'), `${JSON.stringify(meta, null, 2)}\n`);

await browser.close();
console.log(JSON.stringify(meta, null, 2));
