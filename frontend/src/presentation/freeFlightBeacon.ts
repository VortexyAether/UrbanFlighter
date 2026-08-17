import type { BuildingData } from '../api';
import { deterministicStringSeed, deterministicUnit } from '../utils/deterministicSampling';
import {
  dressingPointBuildingClearance,
  type DressingBounds,
  type DressingIdentity,
  type DressingPoint,
} from './urbanDressing';

export interface FreeFlightBeacon extends DressingPoint {
  altitude: number;
  rotation: number;
  contract: 'display_only_non_physical';
}

function distanceSquared(a: DressingPoint, b: DressingPoint) {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
}

/** Selects one bounded, deterministic display waypoint; it has no reward or collision role. */
export function selectFreeFlightBeacon(
  bounds: DressingBounds,
  buildings: readonly BuildingData[],
  identity: DressingIdentity,
  start: DressingPoint,
) {
  const seed = deterministicStringSeed([
    'urban-flighter.free-flight-beacon.v1',
    identity.scenarioId ?? `${identity.lat.toFixed(6)},${identity.lon.toFixed(6)}`,
  ].join('|'));
  const width = bounds.max_x - bounds.min_x;
  const depth = bounds.max_y - bounds.min_y;
  const pad = Math.min(30, width * 0.12, depth * 0.12);
  const minimumDistance = Math.min(width, depth) * 0.32;
  let best: { point: DressingPoint; score: number; sample: number } | null = null;
  for (let sample = 0; sample < 160; sample += 1) {
    const point = {
      x: bounds.min_x + pad + deterministicUnit(seed, sample, 0) * Math.max(0, width - pad * 2),
      y: bounds.min_y + pad + deterministicUnit(seed, sample, 1) * Math.max(0, depth - pad * 2),
    };
    const startDistance = Math.sqrt(distanceSquared(point, start));
    if (startDistance < minimumDistance) continue;
    const clearance = dressingPointBuildingClearance(point, buildings);
    if (clearance < 8) continue;
    const score = startDistance + Math.min(clearance, 30) * 1.8;
    if (!best || score > best.score) best = { point, score, sample };
  }
  const fallback = {
    x: Math.max(bounds.min_x + pad, Math.min(bounds.max_x - pad, bounds.max_x * 0.58)),
    y: Math.max(bounds.min_y + pad, Math.min(bounds.max_y - pad, bounds.max_y * 0.58)),
  };
  const selected = best?.point ?? fallback;
  const sample = best?.sample ?? 0;
  return {
    ...selected,
    altitude: 24 + deterministicUnit(seed, sample, 2) * 16,
    rotation: Math.atan2(selected.x - start.x, selected.y - start.y),
    contract: 'display_only_non_physical',
  } satisfies FreeFlightBeacon;
}
