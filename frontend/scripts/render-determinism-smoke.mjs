import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const vite = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  appType: 'custom',
  server: { middlewareMode: true, hmr: false },
});

try {
  const {
    appendBoundedSample,
    deterministicUnit,
  } = await vite.ssrLoadModule('/src/utils/deterministicSampling.ts');

  const seed = 0x51f15e5d;
  const firstPass = Array.from(
    { length: 800 },
    (_, index) => Array.from({ length: 7 }, (_, channel) => deterministicUnit(seed, index, channel)),
  );
  const secondPass = Array.from(
    { length: 800 },
    (_, index) => Array.from({ length: 7 }, (_, channel) => deterministicUnit(seed, index, channel)),
  );

  if (JSON.stringify(firstPass) !== JSON.stringify(secondPass)) {
    throw new Error('Deterministic samples changed between identical passes.');
  }
  if (firstPass.flat().some((value) => !Number.isFinite(value) || value < 0 || value >= 1)) {
    throw new Error('Deterministic samples must stay in the [0, 1) interval.');
  }
  if (new Set(firstPass.flat()).size < 5_500) {
    throw new Error('Deterministic samples do not provide enough visual variation.');
  }

  const initialHistory = [0, 0, 0, 0];
  let history = initialHistory;
  for (let sample = 1; sample <= 10; sample += 1) {
    history = appendBoundedSample(history, sample, 4);
  }
  if (history.join(',') !== '7,8,9,10' || initialHistory.join(',') !== '0,0,0,0') {
    throw new Error(`Unexpected bounded history: ${history.join(',')}.`);
  }

  console.log(JSON.stringify({
    status: 'ok',
    tests: 4,
    contract: 'repeatable hash samples, unit bounds, visual variation, immutable bounded history',
  }));
} finally {
  await vite.close();
}
