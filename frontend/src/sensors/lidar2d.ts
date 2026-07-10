import type { BuildingData } from '../api';
import {
  IDENTITY_LIDAR_DISPLAY_POSE,
  type LidarDisplayPose,
  type LidarTelemetry,
} from './lidar';

export interface Lidar2DPoint {
  x: number;
  y: number;
}

export interface Lidar2DConfig {
  maxRange: number;
  horizontalSamples: number;
}

export interface Lidar2DPose {
  position: Lidar2DPoint;
  heading: number;
}

export interface Lidar2DBounds {
  min_x: number;
  max_x: number;
  min_y: number;
  max_y: number;
}

export interface Lidar2DSample {
  directionLocal: Lidar2DPoint;
  directionWorld: Lidar2DPoint;
  distance: number;
  normalizedDistance: number;
  hit: boolean;
  point: Lidar2DPoint | null;
}

export interface Lidar2DScan {
  config: Readonly<Lidar2DConfig>;
  samples: Lidar2DSample[];
  observation: number[];
}

export const DEFAULT_LIDAR_2D_CONFIG: Readonly<Lidar2DConfig> = Object.freeze({
  // A 2-degree horizontal step remains inexpensive at 10 Hz while resolving
  // narrow, distant urban facades much more reliably than the old 7.5-degree scan.
  maxRange: 180,
  horizontalSamples: 180,
});

function cross(a: Lidar2DPoint, b: Lidar2DPoint) {
  return a.x * b.y - a.y * b.x;
}

function raySegmentDistance(
  origin: Lidar2DPoint,
  direction: Lidar2DPoint,
  start: number[],
  end: number[],
) {
  const segment = { x: end[0] - start[0], y: end[1] - start[1] };
  const denominator = cross(direction, segment);
  if (Math.abs(denominator) < 1e-9) return null;

  const offset = { x: start[0] - origin.x, y: start[1] - origin.y };
  const distance = cross(offset, segment) / denominator;
  const edgeFraction = cross(offset, direction) / denominator;
  return distance >= 0 && edgeFraction >= 0 && edgeFraction <= 1 ? distance : null;
}

function validateConfig(config: Lidar2DConfig) {
  if (!Number.isFinite(config.maxRange) || config.maxRange <= 0) {
    throw new Error('2D LiDAR maxRange must be a positive finite number.');
  }
  if (!Number.isInteger(config.horizontalSamples) || config.horizontalSamples <= 0) {
    throw new Error('2D LiDAR horizontalSamples must be a positive integer.');
  }
}

export function createLidar2DLocalDirections(config: Lidar2DConfig): Lidar2DPoint[] {
  validateConfig(config);
  return Array.from({ length: config.horizontalSamples }, (_, index) => {
    const angle = (index / config.horizontalSamples) * Math.PI * 2;
    return { x: Math.cos(angle), y: Math.sin(angle) };
  });
}

export function scanLidar2D(
  pose: Lidar2DPose,
  buildings: readonly BuildingData[],
  config: Lidar2DConfig = DEFAULT_LIDAR_2D_CONFIG,
  localDirections = createLidar2DLocalDirections(config),
): Lidar2DScan {
  validateConfig(config);
  if (localDirections.length !== config.horizontalSamples) {
    throw new Error(`2D LiDAR direction count ${localDirections.length} does not match configured sample count ${config.horizontalSamples}.`);
  }

  const cosHeading = Math.cos(pose.heading);
  const sinHeading = Math.sin(pose.heading);
  const samples = localDirections.map((directionLocal) => {
    const directionWorld = {
      x: directionLocal.x * cosHeading - directionLocal.y * sinHeading,
      y: directionLocal.x * sinHeading + directionLocal.y * cosHeading,
    };
    let distance = config.maxRange;
    let hit = false;

    for (const building of buildings) {
      const footprint = building.footprint;
      if (footprint.length < 2) continue;
      for (let edgeIndex = 0; edgeIndex < footprint.length; edgeIndex += 1) {
        const edgeDistance = raySegmentDistance(
          pose.position,
          directionWorld,
          footprint[edgeIndex],
          footprint[(edgeIndex + 1) % footprint.length],
        );
        if (edgeDistance !== null && edgeDistance <= distance && edgeDistance <= config.maxRange) {
          distance = edgeDistance;
          hit = true;
        }
      }
    }

    return {
      directionLocal: { ...directionLocal },
      directionWorld,
      distance,
      normalizedDistance: Math.max(0, Math.min(1, distance / config.maxRange)),
      hit,
      point: hit
        ? {
            x: pose.position.x + directionWorld.x * distance,
            y: pose.position.y + directionWorld.y * distance,
          }
        : null,
    };
  });

  return {
    config: { ...config },
    samples,
    observation: samples.flatMap((sample) => [
      sample.directionLocal.x,
      sample.directionLocal.y,
      0,
      sample.normalizedDistance,
      sample.hit ? 1 : 0,
    ]),
  };
}

function pointInFootprint(point: Lidar2DPoint, footprint: number[][]) {
  let inside = false;
  for (let index = 0, previous = footprint.length - 1; index < footprint.length; previous = index, index += 1) {
    const [x, y] = footprint[index];
    const [previousX, previousY] = footprint[previous];
    const crossesY = (y > point.y) !== (previousY > point.y);
    const edgeX = ((previousX - x) * (point.y - y)) / (previousY - y || Number.EPSILON) + x;
    if (crossesY && point.x < edgeX) inside = !inside;
  }
  return inside;
}

function distanceToFootprint(point: Lidar2DPoint, footprint: number[][]) {
  if (footprint.length < 2) return Infinity;
  if (pointInFootprint(point, footprint)) return 0;

  let nearest = Infinity;
  for (let index = 0; index < footprint.length; index += 1) {
    const start = footprint[index];
    const end = footprint[(index + 1) % footprint.length];
    const edgeX = end[0] - start[0];
    const edgeY = end[1] - start[1];
    const edgeLengthSquared = edgeX * edgeX + edgeY * edgeY;
    const fraction = edgeLengthSquared > 0
      ? Math.max(0, Math.min(1, ((point.x - start[0]) * edgeX + (point.y - start[1]) * edgeY) / edgeLengthSquared))
      : 0;
    nearest = Math.min(nearest, Math.hypot(
      point.x - (start[0] + edgeX * fraction),
      point.y - (start[1] + edgeY * fraction),
    ));
  }
  return nearest;
}

/**
 * Select a deterministic, collision-clear reset point whose real 2D LiDAR scan
 * contains useful returns. Building geometry is used only to validate the spawn;
 * the local map continues to receive scan samples, never footprint polygons.
 */
export function selectLidar2DSpawn(
  bounds: Lidar2DBounds,
  buildings: readonly BuildingData[],
  minimumClearance: number,
  config: Lidar2DConfig = DEFAULT_LIDAR_2D_CONFIG,
  localDirections = createLidar2DLocalDirections(config),
): Lidar2DPoint {
  const center = {
    x: (bounds.min_x + bounds.max_x) * 0.5,
    y: (bounds.min_y + bounds.max_y) * 0.5,
  };
  const spanX = bounds.max_x - bounds.min_x;
  const spanY = bounds.max_y - bounds.min_y;
  const candidates: Lidar2DPoint[] = [];
  const seen = new Set<string>();
  const addCandidate = (xFraction: number, yFraction: number) => {
    const point = {
      x: center.x + spanX * xFraction,
      y: center.y + spanY * yFraction,
    };
    const key = `${point.x.toFixed(6)},${point.y.toFixed(6)}`;
    if (!seen.has(key)) {
      seen.add(key);
      candidates.push(point);
    }
  };

  [
    [0, 0], [0, 0.18], [0, -0.18], [-0.18, 0], [0.18, 0],
    [-0.18, 0.18], [0.18, 0.18], [-0.18, -0.18], [0.18, -0.18],
    [0, 0.32], [0, -0.32], [-0.32, 0], [0.32, 0],
  ].forEach(([x, y]) => addCandidate(x, y));
  for (let yIndex = 1; yIndex <= 7; yIndex += 1) {
    for (let xIndex = 1; xIndex <= 7; xIndex += 1) {
      addCandidate(xIndex / 8 - 0.5, yIndex / 8 - 0.5);
    }
  }

  const minimumHits = Math.max(6, Math.ceil(config.horizontalSamples * 0.04));
  let bestSafe: { point: Lidar2DPoint; hitCount: number; clearance: number } | null = null;
  let bestClear: { point: Lidar2DPoint; hitCount: number; clearance: number } | null = null;
  for (const point of candidates) {
    if (
      point.x < bounds.min_x + minimumClearance
      || point.x > bounds.max_x - minimumClearance
      || point.y < bounds.min_y + minimumClearance
      || point.y > bounds.max_y - minimumClearance
    ) {
      continue;
    }
    const clearance = Math.min(
      Infinity,
      ...buildings.map((building) => distanceToFootprint(point, building.footprint)),
    );
    if (clearance <= 0) continue;

    const hitCount = scanLidar2D(
      { position: point, heading: 0 },
      buildings,
      config,
      localDirections,
    ).samples.reduce((count, sample) => count + Number(sample.hit), 0);
    if (!bestClear || hitCount > bestClear.hitCount || (hitCount === bestClear.hitCount && clearance > bestClear.clearance)) {
      bestClear = { point, hitCount, clearance };
    }
    if (clearance < minimumClearance) continue;
    if (hitCount >= minimumHits) return point;
    if (!bestSafe || hitCount > bestSafe.hitCount || (hitCount === bestSafe.hitCount && clearance > bestSafe.clearance)) {
      bestSafe = { point, hitCount, clearance };
    }
  }

  return bestSafe?.point ?? bestClear?.point ?? center;
}

export function summarizeLidar2DScan(
  scan: Lidar2DScan,
  scanPose: LidarDisplayPose = IDENTITY_LIDAR_DISPLAY_POSE,
): LidarTelemetry {
  const hits = scan.samples.filter((sample) => sample.hit);
  return {
    sampleCount: scan.samples.length,
    nearestDistance: hits.length > 0 ? Math.min(...hits.map((sample) => sample.distance)) : null,
    hitRatio: scan.samples.length > 0 ? hits.length / scan.samples.length : 0,
    observation: scan.observation,
    ui: {
      maxRange: scan.config.maxRange,
      hitCount: hits.length,
      scanPose: { ...scanPose },
      // Display telemetry includes max-range misses and uses the drone-local
      // frame. The stable five-values-per-ray observation above is unchanged.
      returns: scan.samples.map((sample) => ({
        relativeX: sample.directionLocal.x * sample.distance,
        relativeY: 0,
        relativeZ: sample.directionLocal.y * sample.distance,
        distance: sample.distance,
        normalizedDistance: sample.normalizedDistance,
        hit: sample.hit,
      })),
    },
  };
}
