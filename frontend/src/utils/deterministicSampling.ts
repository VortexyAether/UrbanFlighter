const UINT32_RANGE = 0x1_0000_0000;

/** Stable 32-bit FNV-1a hash for presentation identities and test fixtures. */
export function deterministicStringSeed(value: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

/**
 * Returns a reproducible value in [0, 1) for an integer sample/channel pair.
 * This is stateless so render order cannot change the generated layout.
 */
export function deterministicUnit(seed: number, sampleIndex: number, channel = 0): number {
  let hash = (
    seed
    ^ Math.imul(sampleIndex + 1, 0x9e3779b1)
    ^ Math.imul(channel + 1, 0x85ebca77)
  ) >>> 0;

  hash = Math.imul(hash ^ (hash >>> 16), 0x7feb352d);
  hash = Math.imul(hash ^ (hash >>> 15), 0x846ca68b);
  return ((hash ^ (hash >>> 16)) >>> 0) / UINT32_RANGE;
}

/** Appends without mutating the input and retains at most capacity entries. */
export function appendBoundedSample<T>(
  history: readonly T[],
  sample: T,
  capacity: number,
): T[] {
  if (!Number.isInteger(capacity) || capacity < 1) {
    throw new RangeError('Sample history capacity must be a positive integer.');
  }

  const start = Math.max(0, history.length - capacity + 1);
  return [...history.slice(start), sample];
}
