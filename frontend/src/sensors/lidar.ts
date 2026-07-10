import * as THREE from 'three';

export interface LidarConfig {
  maxRange: number;
  sampleCount: number;
}

export interface DronePose {
  position: THREE.Vector3;
  yaw: number;
  pitch: number;
  roll: number;
}

export interface LidarSample {
  origin: THREE.Vector3;
  directionWorld: THREE.Vector3;
  directionLocal: THREE.Vector3;
  endpoint: THREE.Vector3;
  hitPoint: THREE.Vector3 | null;
  distance: number;
  normalizedDistance: number;
  hit: boolean;
}

export interface LidarScan {
  config: Readonly<LidarConfig>;
  samples: LidarSample[];
  observation: number[];
}

export interface LidarTelemetry {
  sampleCount: number;
  nearestDistance: number | null;
  hitRatio: number;
  observation: number[];
  ui: LidarUiTelemetry;
}

export interface LidarUiReturn {
  relativeX: number;
  relativeY: number;
  relativeZ: number;
  distance: number;
  normalizedDistance: number;
  hit: boolean;
}

/** Simulator pose exposed only to the rolling sensor-map displays. */
export interface LidarDisplayPose {
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  roll: number;
}

/** Display-only scan data; intentionally separate from the stable RL observation. */
export interface LidarUiTelemetry {
  maxRange: number;
  hitCount: number;
  scanPose: LidarDisplayPose;
  returns: LidarUiReturn[];
}

export const IDENTITY_LIDAR_DISPLAY_POSE: Readonly<LidarDisplayPose> = Object.freeze({
  x: 0,
  y: 0,
  z: 0,
  yaw: 0,
  pitch: 0,
  roll: 0,
});

/** Per-ray order in the flattened observation vector. */
export const LIDAR_OBSERVATION_LAYOUT = [
  'directionLocalX',
  'directionLocalY',
  'directionLocalZ',
  'normalizedDistance',
  'hit',
] as const;

export const LIDAR_OBSERVATION_VALUES_PER_SAMPLE = LIDAR_OBSERVATION_LAYOUT.length;

export const DEFAULT_LIDAR_CONFIG: Readonly<LidarConfig> = Object.freeze({
  // 120 m is a defensible long-range 360° proxy for a small drone-mounted LiDAR.
  maxRange: 120,
  // A Fibonacci sphere spreads deterministic samples across the complete shell.
  sampleCount: 600,
});

const GOLDEN_ANGLE_RAD = Math.PI * (3 - Math.sqrt(5));

const JET_DISTANCE_STOPS = [
  { at: 0, color: new THREE.Color('#ff0000') },
  { at: 1 / 3, color: new THREE.Color('#ffff00') },
  { at: 2 / 3, color: new THREE.Color('#00ffff') },
  { at: 1, color: new THREE.Color('#0000ff') },
] as const;

/** JET distance scale used only by displays: near red -> yellow -> cyan -> far blue. */
export function getLidarJetColor(normalizedDistance: number, target = new THREE.Color()) {
  const distance = THREE.MathUtils.clamp(normalizedDistance, 0, 1);
  const upperIndex = JET_DISTANCE_STOPS.findIndex((stop) => distance <= stop.at);
  if (upperIndex <= 0) return target.copy(JET_DISTANCE_STOPS[0].color);

  const lower = JET_DISTANCE_STOPS[upperIndex - 1];
  const upper = JET_DISTANCE_STOPS[upperIndex];
  return target.copy(lower.color).lerp(upper.color, (distance - lower.at) / (upper.at - lower.at));
}

export function getLidarJetCss(normalizedDistance: number) {
  return `#${getLidarJetColor(normalizedDistance).getHexString()}`;
}

function validateLidarConfig(config: LidarConfig) {
  if (!Number.isFinite(config.maxRange) || config.maxRange <= 0) {
    throw new Error('LiDAR maxRange must be a positive finite number.');
  }
  if (!Number.isInteger(config.sampleCount) || config.sampleCount <= 0) {
    throw new Error('LiDAR sampleCount must be a positive integer.');
  }
}

export function createLidarLocalDirections(config: LidarConfig): THREE.Vector3[] {
  validateLidarConfig(config);
  return Array.from({ length: config.sampleCount }, (_, index) => {
    // Midpoint y values avoid duplicated/over-weighted pole samples. Advancing
    // by the golden angle then yields stable, near-uniform full-sphere coverage.
    const y = 1 - (2 * (index + 0.5)) / config.sampleCount;
    const horizontalRadius = Math.sqrt(Math.max(0, 1 - y * y));
    const azimuth = index * GOLDEN_ANGLE_RAD;
    return new THREE.Vector3(
      Math.sin(azimuth) * horizontalRadius,
      y,
      -Math.cos(azimuth) * horizontalRadius,
    );
  });
}

export function scanLidar(
  pose: DronePose,
  buildingCollisionMeshes: THREE.Object3D[],
  config: LidarConfig = DEFAULT_LIDAR_CONFIG,
  localDirections = createLidarLocalDirections(config),
): LidarScan {
  validateLidarConfig(config);
  if (localDirections.length !== config.sampleCount) {
    throw new Error(`LiDAR direction count ${localDirections.length} does not match configured sample count ${config.sampleCount}.`);
  }

  const origin = pose.position.clone();
  const orientation = new THREE.Euler(pose.pitch, pose.yaw, pose.roll, 'YXZ');
  const raycaster = new THREE.Raycaster();
  const samples = localDirections.map((directionLocal) => {
    const directionWorld = directionLocal.clone().applyEuler(orientation).normalize();
    raycaster.set(origin, directionWorld);
    raycaster.near = 0.25;
    raycaster.far = config.maxRange;
    const buildingHit = raycaster.intersectObjects(buildingCollisionMeshes, false)[0];
    const groundDistance = directionWorld.y < -1e-9
      ? -origin.y / directionWorld.y
      : Number.POSITIVE_INFINITY;
    const groundHit = groundDistance >= raycaster.near && groundDistance <= config.maxRange;
    const buildingDistance = buildingHit?.distance ?? Number.POSITIVE_INFINITY;
    const distance = groundHit && groundDistance < buildingDistance
      ? groundDistance
      : Math.min(buildingDistance, config.maxRange);
    const hit = Boolean(buildingHit) || groundHit;
    const endpoint = origin.clone().add(directionWorld.clone().multiplyScalar(distance));
    const hitPoint = hit ? endpoint.clone() : null;

    return {
      origin: origin.clone(),
      directionWorld,
      directionLocal: directionLocal.clone(),
      endpoint,
      hitPoint,
      distance,
      normalizedDistance: THREE.MathUtils.clamp(distance / config.maxRange, 0, 1),
      hit,
    };
  });

  return {
    config: { ...config },
    samples,
    observation: getLidarObservationVector(samples),
  };
}

export function getLidarObservationVector(samples: LidarSample[]) {
  return samples.flatMap((sample) => [
    sample.directionLocal.x,
    sample.directionLocal.y,
    sample.directionLocal.z,
    sample.normalizedDistance,
    sample.hit ? 1 : 0,
  ]);
}

export function summarizeLidarScan(
  scan: LidarScan,
  scanPose: LidarDisplayPose = IDENTITY_LIDAR_DISPLAY_POSE,
): LidarTelemetry {
  const hits = scan.samples.filter((sample) => sample.hit);
  const nearestDistance = hits.length > 0
    ? Math.min(...hits.map((sample) => sample.distance))
    : null;

  return {
    sampleCount: scan.samples.length,
    nearestDistance,
    hitRatio: scan.samples.length > 0 ? hits.length / scan.samples.length : 0,
    observation: scan.observation,
    ui: {
      maxRange: scan.config.maxRange,
      hitCount: hits.length,
      scanPose: { ...scanPose },
      // Display-only local endpoints retain the complete spherical scan. Hits
      // stop at their measured range; misses lie on the max-range shell. Using
      // directionLocal keeps this UI view centered on the drone's sensor frame.
      returns: scan.samples.map((sample) => ({
        relativeX: sample.directionLocal.x * sample.distance,
        relativeY: sample.directionLocal.y * sample.distance,
        relativeZ: sample.directionLocal.z * sample.distance,
        distance: sample.distance,
        normalizedDistance: sample.normalizedDistance,
        hit: sample.hit,
      })),
    },
  };
}
