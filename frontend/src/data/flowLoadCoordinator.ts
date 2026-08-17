export const FLOW_CACHE_MAX_ENTRIES = 3;
export const FLOW_CACHE_TTL_MS = 5 * 60 * 1_000;
export const FLOW_REQUEST_TIMEOUT_MS = 30_000;

interface CacheEntry<Value> {
  value: Value;
  expiresAt: number;
}

interface TimedLruCacheOptions {
  maxEntries: number;
  ttlMs: number;
  now?: () => number;
}

/**
 * A small absolute-TTL/LRU cache. Reads update recency without extending the
 * expiry time, so current-condition metadata cannot remain fresh indefinitely through use.
 */
export class TimedLruCache<Key, Value> {
  private readonly entries = new Map<Key, CacheEntry<Value>>();
  private readonly maxEntries: number;
  private readonly ttlMs: number;
  private readonly now: () => number;

  constructor(options: TimedLruCacheOptions) {
    if (!Number.isInteger(options.maxEntries) || options.maxEntries < 1) {
      throw new Error('TimedLruCache maxEntries must be a positive integer.');
    }
    if (!Number.isFinite(options.ttlMs) || options.ttlMs <= 0) {
      throw new Error('TimedLruCache ttlMs must be a positive finite number.');
    }

    this.maxEntries = options.maxEntries;
    this.ttlMs = options.ttlMs;
    this.now = options.now ?? Date.now;
  }

  get size() {
    this.pruneExpired();
    return this.entries.size;
  }

  get(key: Key): Value | undefined {
    const entry = this.entries.get(key);
    if (!entry) return undefined;

    if (entry.expiresAt <= this.now()) {
      this.entries.delete(key);
      return undefined;
    }

    // Map insertion order is the LRU order, so reinsert a successful read.
    this.entries.delete(key);
    this.entries.set(key, entry);
    return entry.value;
  }

  set(key: Key, value: Value) {
    this.pruneExpired();
    this.entries.delete(key);
    this.entries.set(key, {
      value,
      expiresAt: this.now() + this.ttlMs,
    });

    while (this.entries.size > this.maxEntries) {
      const leastRecentlyUsed = this.entries.keys().next().value;
      if (leastRecentlyUsed === undefined) break;
      this.entries.delete(leastRecentlyUsed);
    }
  }

  delete(key: Key) {
    return this.entries.delete(key);
  }

  clear() {
    this.entries.clear();
  }

  private pruneExpired() {
    const now = this.now();
    for (const [key, entry] of this.entries) {
      if (entry.expiresAt <= now) this.entries.delete(key);
    }
  }
}

export interface FlowRequestToken {
  readonly id: number;
  readonly signal: AbortSignal;
}

export interface TimeoutScheduler {
  schedule(callback: () => void, delayMs: number): unknown;
  cancel(handle: unknown): void;
}

interface ActiveRequest {
  id: number;
  controller: AbortController;
  timeoutHandle: unknown;
  timedOut: boolean;
}

const defaultTimeoutScheduler: TimeoutScheduler = {
  schedule: (callback, delayMs) => setTimeout(callback, delayMs),
  cancel: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
};

/**
 * Owns the one request whose result may still update the UI. Starting a new
 * request aborts the previous fetch; the identity check remains authoritative
 * even if a mocked or non-standard fetch ignores AbortSignal.
 */
export class LatestFlowRequestCoordinator {
  private readonly scheduler: TimeoutScheduler;
  private active: ActiveRequest | null = null;
  private nextId = 0;

  constructor(scheduler: TimeoutScheduler = defaultTimeoutScheduler) {
    this.scheduler = scheduler;
  }

  start(timeoutMs: number, onTimeout: () => void): FlowRequestToken {
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new Error('Flow request timeout must be a positive finite number.');
    }

    this.cancel();
    const id = ++this.nextId;
    const controller = new AbortController();
    const active: ActiveRequest = {
      id,
      controller,
      timeoutHandle: undefined,
      timedOut: false,
    };
    this.active = active;
    active.timeoutHandle = this.scheduler.schedule(() => {
      if (this.active?.id !== id) return;
      active.timedOut = true;
      controller.abort();
      onTimeout();
    }, timeoutMs);

    return { id, signal: controller.signal };
  }

  isCurrent(token: FlowRequestToken) {
    return this.active?.id === token.id;
  }

  canApply(token: FlowRequestToken) {
    return this.active?.id === token.id && !this.active.timedOut;
  }

  didTimeOut(token: FlowRequestToken) {
    return this.active?.id === token.id && this.active.timedOut;
  }

  complete(token: FlowRequestToken) {
    if (this.active?.id !== token.id) return false;
    this.scheduler.cancel(this.active.timeoutHandle);
    this.active = null;
    return true;
  }

  cancel() {
    if (!this.active) return;
    const active = this.active;
    this.active = null;
    this.scheduler.cancel(active.timeoutHandle);
    active.controller.abort();
  }
}

export function createFlowCacheKey(
  lat: number,
  lon: number,
  geometryRadiusM: number,
  solveRadiusM: number,
  gridSizeM: number,
) {
  return `${lat.toFixed(5)},${lon.toFixed(5)},${geometryRadiusM},${solveRadiusM},${gridSizeM}`;
}
