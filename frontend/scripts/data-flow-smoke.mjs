import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const vite = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  appType: 'custom',
  server: { middlewareMode: true, hmr: false },
});

try {
  const {
    LatestFlowRequestCoordinator,
    TimedLruCache,
    createFlowCacheKey,
  } = await vite.ssrLoadModule('/src/data/flowLoadCoordinator.ts');

  let now = 0;
  const cache = new TimedLruCache({ maxEntries: 2, ttlMs: 10, now: () => now });
  cache.set('a', 'alpha');
  cache.set('b', 'bravo');
  if (cache.get('a') !== 'alpha') throw new Error('Expected a live cache hit.');
  now = 1;
  cache.set('c', 'charlie');
  if (cache.get('b') !== undefined || cache.get('a') !== 'alpha' || cache.size !== 2) {
    throw new Error('Expected bounded LRU eviction to retain the recently read entry.');
  }
  now = 10;
  if (cache.get('a') !== undefined || cache.get('c') !== 'charlie' || cache.size !== 1) {
    throw new Error('Expected absolute TTL expiry without extending expiry on reads.');
  }
  now = 11;
  if (cache.get('c') !== undefined || cache.size !== 0) {
    throw new Error('Expected expired cache entries to be removed.');
  }
  cache.set('registered-live-scenario', 'snapshot');
  if (!cache.delete('registered-live-scenario') || cache.get('registered-live-scenario') !== undefined) {
    throw new Error('Expected stale live-scenario cache entries to be explicitly removable before refetch.');
  }

  const cacheKey = createFlowCacheKey(37.451448, 126.6515423, 400, 400, 2.5);
  if (cacheKey !== '37.45145,126.65154,400,400,2.5') {
    throw new Error(`Unexpected stable flow cache key: ${cacheKey}`);
  }

  const scheduler = createManualScheduler();
  const coordinator = new LatestFlowRequestCoordinator(scheduler);
  let timeoutCount = 0;
  const first = coordinator.start(30_000, () => { timeoutCount += 1; });
  const second = coordinator.start(30_000, () => { timeoutCount += 1; });
  if (!first.signal.aborted || coordinator.canApply(first) || !coordinator.canApply(second)) {
    throw new Error('Expected a newer flow request to abort and supersede the older request.');
  }
  if (coordinator.complete(first) || !coordinator.complete(second) || timeoutCount !== 0) {
    throw new Error('Expected only the current request to complete UI ownership.');
  }

  let loading = true;
  const pending = coordinator.start(30_000, () => { timeoutCount += 1; });
  const cacheHit = coordinator.start(30_000, () => { timeoutCount += 1; });
  if (coordinator.complete(cacheHit)) loading = false;
  if (!pending.signal.aborted || loading || coordinator.complete(pending)) {
    throw new Error('Expected a cache hit to supersede a pending request and settle loading.');
  }

  const timed = coordinator.start(30_000, () => { timeoutCount += 1; });
  scheduler.fireNext();
  if (
    timeoutCount !== 1
    || !timed.signal.aborted
    || !coordinator.didTimeOut(timed)
    || coordinator.canApply(timed)
    || !coordinator.complete(timed)
  ) {
    throw new Error('Expected timeout to abort the fetch and permanently suppress its response.');
  }

  console.log(JSON.stringify({
    status: 'ok',
    tests: 7,
    contract: 'TTL/LRU bounds/delete, stable keys, supersession, cache-hit settling, timeout, stale suppression',
  }));
} finally {
  await vite.close();
}

function createManualScheduler() {
  let nextId = 0;
  const callbacks = new Map();
  return {
    schedule(callback) {
      const id = ++nextId;
      callbacks.set(id, callback);
      return id;
    },
    cancel(handle) {
      callbacks.delete(handle);
    },
    fireNext() {
      const entry = callbacks.entries().next();
      if (entry.done) throw new Error('No scheduled timeout to fire.');
      const [id, callback] = entry.value;
      callbacks.delete(id);
      callback();
    },
  };
}
