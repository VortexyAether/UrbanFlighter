import { mkdirSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { pathToFileURL, fileURLToPath } from 'node:url';

const playwrightRoot = process.env.PLAYWRIGHT_PACKAGE ?? '/tmp/uf-pw/node_modules/playwright/index.mjs';
const { chromium } = await import(pathToFileURL(playwrightRoot).href);

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const frontend = process.env.URBAN_FLIGHTER_FRONTEND ?? 'http://127.0.0.1:5173';
const frameDir = join(root, 'docs/showcase/components/.radar-gif-frames');
const outGif = join(root, 'docs/showcase/components/radar_3d_nyc.gif');
const frameCount = 36;
const intervalMs = 140;

rmSync(frameDir, { recursive: true, force: true });
mkdirSync(frameDir, { recursive: true });

const browser = await chromium.launch({
  headless: true,
  channel: process.env.URBAN_FLIGHTER_CHROME_CHANNEL ?? 'chrome',
});
const page = await browser.newPage({
  viewport: { width: 1440, height: 860 },
  deviceScaleFactor: 1,
});

await page.goto(`${frontend}/?shot=radar`, { waitUntil: 'domcontentloaded', timeout: 60_000 });
await page.waitForSelector('[data-shot-ready="1"]', { timeout: 90_000 });
await page.waitForTimeout(2200);
await page.locator('canvas').first().click({ timeout: 10_000 }).catch(() => {});
await page.evaluate(() => document.body.focus());

const radar = page.locator('.slam-window').first();
await radar.waitFor({ state: 'visible', timeout: 15_000 });

await page.keyboard.down('KeyW');
await page.keyboard.down('Space');

for (let i = 0; i < frameCount; i += 1) {
  if (i === 12) await page.keyboard.down('KeyD');
  if (i === 22) await page.keyboard.up('KeyD');
  if (i === 22) await page.keyboard.down('KeyA');
  if (i === 30) await page.keyboard.up('KeyA');
  await radar.screenshot({
    path: join(frameDir, `frame-${String(i).padStart(2, '0')}.png`),
    type: 'png',
  });
  await page.waitForTimeout(intervalMs);
}

await page.keyboard.up('KeyW');
await page.keyboard.up('Space');
await page.keyboard.up('KeyD');
await page.keyboard.up('KeyA');
await browser.close();

const frames = readdirSync(frameDir).filter((name) => name.endsWith('.png')).sort().map((name) => join(frameDir, name));
const encoded = spawnSync('magick', [
  '-delay', '12',
  '-loop', '0',
  ...frames,
  '-resize', '720x',
  '-dither', 'FloydSteinberg',
  '-colors', '48',
  '-layers', 'OptimizePlus',
  outGif,
], { cwd: root });

if (encoded.status !== 0) {
  throw new Error(encoded.stderr.toString() || encoded.stdout.toString() || 'magick failed');
}

rmSync(frameDir, { recursive: true, force: true });
writeFileSync(join(root, 'docs/showcase/components/radar_gif_meta.json'), `${JSON.stringify({
  captured_kst: new Date().toISOString().slice(0, 10),
  file: 'radar_3d_nyc.gif',
  frames: frameCount,
  interval_ms: intervalMs,
  note: '3D rolling sensor map while holding W/Space and a short A/D turn. SIM odometry only.',
}, null, 2)}\n`);
console.log(outGif);
