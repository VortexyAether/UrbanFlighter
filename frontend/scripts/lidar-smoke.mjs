import * as THREE from 'three';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const vite = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  appType: 'custom',
  server: { middlewareMode: true, hmr: false },
});
const {
  buildBuildingCollisionMeshes,
  isScenePointInsideBuilding,
} = await vite.ssrLoadModule('/src/geometry/buildingGeometry.ts');
const {
  DEFAULT_LIDAR_CONFIG,
  LIDAR_OBSERVATION_LAYOUT,
  LIDAR_OBSERVATION_VALUES_PER_SAMPLE,
  createLidarLocalDirections,
  getLidarJetColor,
  getLidarJetCss,
  scanLidar,
  summarizeLidarScan,
} = await vite.ssrLoadModule('/src/sensors/lidar.ts');
const {
  DEFAULT_LIDAR_2D_CONFIG,
  createLidar2DLocalDirections,
  scanLidar2D,
  selectLidar2DSpawn,
  summarizeLidar2DScan,
} = await vite.ssrLoadModule('/src/sensors/lidar2d.ts');
const {
  ROLLING_SENSOR_MAP_KEYFRAMES,
  appendRollingSensorKeyframe,
  createRollingSensorKeyframe,
  getPoseInCurrentFrame,
  getRecordedToCurrentMatrix,
} = await vite.ssrLoadModule('/src/sensors/rollingSensorMap.ts');

const EXPECTED_OBSERVATION_LAYOUT = [
  'directionLocalX',
  'directionLocalY',
  'directionLocalZ',
  'normalizedDistance',
  'hit',
];

function assertObservationSlots(observation, expectedSlots, label) {
  const expectedLength = expectedSlots.length * EXPECTED_OBSERVATION_LAYOUT.length;
  if (observation.length !== expectedLength) {
    throw new Error(`${label}: expected ${expectedLength} observation values, got ${observation.length}.`);
  }

  expectedSlots.forEach((expectedSlot, rayIndex) => {
    EXPECTED_OBSERVATION_LAYOUT.forEach((field, fieldIndex) => {
      const actual = observation[rayIndex * EXPECTED_OBSERVATION_LAYOUT.length + fieldIndex];
      const expected = expectedSlot[field];
      const matches = field === 'hit'
        ? actual === expected
        : Number.isFinite(actual) && Math.abs(actual - expected) <= 1e-9;
      if (!matches) {
        throw new Error(`${label}: ray ${rayIndex} field ${field} expected ${expected}, got ${actual}.`);
      }
    });
  });
}

const config = {
  maxRange: 100,
  sampleCount: 4,
};
const defaultDirections = createLidarLocalDirections(DEFAULT_LIDAR_CONFIG);
if (
  DEFAULT_LIDAR_CONFIG.maxRange !== 120
  || defaultDirections.length !== 600
  || DEFAULT_LIDAR_CONFIG.sampleCount !== 600
  || defaultDirections.some((direction) => Math.abs(direction.length() - 1) > 1e-12)
) {
  throw new Error(`Expected spherical 120 m / 600-ray default LiDAR, got ${DEFAULT_LIDAR_CONFIG.maxRange} m / ${defaultDirections.length} rays.`);
}
const expectedFirstY = 1 - 1 / DEFAULT_LIDAR_CONFIG.sampleCount;
const expectedLastY = -expectedFirstY;
const uniqueRoundedElevations = new Set(defaultDirections.map((direction) => direction.y.toFixed(12)));
const repeatedDefaultDirections = createLidarLocalDirections(DEFAULT_LIDAR_CONFIG);
if (
  LIDAR_OBSERVATION_VALUES_PER_SAMPLE !== 5
  || LIDAR_OBSERVATION_LAYOUT.length !== EXPECTED_OBSERVATION_LAYOUT.length
  || LIDAR_OBSERVATION_LAYOUT.some((field, index) => field !== EXPECTED_OBSERVATION_LAYOUT[index])
  || Math.abs(defaultDirections[0].y - expectedFirstY) > 1e-12
  || Math.abs(defaultDirections.at(-1).y - expectedLastY) > 1e-12
  || uniqueRoundedElevations.size !== DEFAULT_LIDAR_CONFIG.sampleCount
  || Math.abs(defaultDirections[1].y - defaultDirections[0].y + 2 / DEFAULT_LIDAR_CONFIG.sampleCount) > 1e-12
  || defaultDirections.some((direction, index) => !direction.equals(repeatedDefaultDirections[index]))
) {
  throw new Error('Expected a stable five-value observation layout and midpoint Fibonacci ordering with one near-uniform elevation per 3D LiDAR sample.');
}
const building = {
  height: 20,
  footprint: [[-5, 29], [5, 29], [5, 31], [-5, 31]],
};
const collisionMeshes = buildBuildingCollisionMeshes([building]);
const scanDirections = [
  new THREE.Vector3(0, 0, -1),
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(0, 0, 1),
  new THREE.Vector3(-1, 0, 0),
];
const scan = scanLidar(
  { position: new THREE.Vector3(0, 10, 0), yaw: 0, pitch: 0, roll: 0 },
  collisionMeshes,
  config,
  scanDirections,
);
const forwardSample = scan.samples[0];

if (!forwardSample.hit) {
  throw new Error('Expected LiDAR smoke ray to hit the building mesh.');
}

if (Math.abs(forwardSample.distance - 29) > 1e-6) {
  throw new Error(`Expected first wall face at 29 m, got ${forwardSample.distance.toFixed(6)} m.`);
}

const expectedObservationLength = scan.samples.length * LIDAR_OBSERVATION_VALUES_PER_SAMPLE;
if (scan.observation.length !== expectedObservationLength || scan.observation.some((value) => !Number.isFinite(value))) {
  throw new Error(`Expected ${expectedObservationLength} finite observation values, got ${scan.observation.length}.`);
}
assertObservationSlots(scan.observation, [
  { directionLocalX: 0, directionLocalY: 0, directionLocalZ: -1, normalizedDistance: 0.29, hit: 1 },
  { directionLocalX: 1, directionLocalY: 0, directionLocalZ: 0, normalizedDistance: 1, hit: 0 },
  { directionLocalX: 0, directionLocalY: 0, directionLocalZ: 1, normalizedDistance: 1, hit: 0 },
  { directionLocalX: -1, directionLocalY: 0, directionLocalZ: 0, normalizedDistance: 1, hit: 0 },
], '3D wall hit/miss observation');
const scanPose = { x: 12, y: 10, z: -4, yaw: 0, pitch: 0, roll: 0 };
const telemetry = summarizeLidarScan(scan, scanPose);
if (
  telemetry.observation !== scan.observation
  || telemetry.ui.maxRange !== config.maxRange
  || telemetry.ui.hitCount !== 1
  || telemetry.ui.scanPose.x !== scanPose.x
  || telemetry.ui.scanPose.y !== scanPose.y
  || telemetry.ui.scanPose.z !== scanPose.z
  || telemetry.ui.returns.length !== config.sampleCount
  || Math.abs(telemetry.ui.returns[0].relativeZ + 29) > 1e-6
  || Math.abs(telemetry.ui.returns[1].relativeX - config.maxRange) > 1e-6
  || telemetry.ui.returns[0].hit !== true
  || telemetry.ui.returns[1].hit !== false
) {
  throw new Error('Expected display-only scan pose and local XYZ returns to include every hit and max-range miss without changing the stable observation.');
}
const decimatedKeyframe = createRollingSensorKeyframe(telemetry, 2);
const syntheticReturns = Array.from({ length: 10 }, (_, index) => ({
  relativeX: index,
  relativeY: index + 0.25,
  relativeZ: -index,
  distance: index + 1,
  normalizedDistance: (index + 1) / 10,
  hit: index === 1 || index === 8,
}));
const syntheticTelemetry = {
  sampleCount: syntheticReturns.length,
  nearestDistance: 2,
  hitRatio: 0.2,
  observation: [],
  ui: {
    maxRange: 10,
    hitCount: 2,
    scanPose: { x: 3, y: 4, z: 5, yaw: 0.1, pitch: 0.2, roll: 0.3 },
    returns: syntheticReturns,
  },
};
const hitAwareKeyframe = createRollingSensorKeyframe(syntheticTelemetry, 4);
const hitOnlyCappedKeyframe = createRollingSensorKeyframe(syntheticTelemetry, 1);
const clonedKeyframe = createRollingSensorKeyframe(syntheticTelemetry, syntheticReturns.length);
const selectedSyntheticIndices = hitAwareKeyframe.returns.map((sample) => sample.relativeX);
if (
  JSON.stringify(selectedSyntheticIndices) !== JSON.stringify([1, 3, 7, 8])
  || hitAwareKeyframe.returns.filter((sample) => sample.hit).length !== 2
  || hitAwareKeyframe.returns.length !== 4
  || hitOnlyCappedKeyframe.returns.length !== 1
  || hitOnlyCappedKeyframe.returns[0].hit !== true
  || createRollingSensorKeyframe(syntheticTelemetry, 0).returns.length !== 0
  || clonedKeyframe.pose === syntheticTelemetry.ui.scanPose
  || clonedKeyframe.returns.some((sample, index) => sample === syntheticReturns[index])
) {
  throw new Error('Expected deterministic hit-aware rolling selection to preserve sparse hits, obey caps, and clone selected data.');
}
const clonedPoseX = clonedKeyframe.pose.x;
const clonedFirstX = clonedKeyframe.returns[0].relativeX;
syntheticTelemetry.ui.scanPose.x = 999;
syntheticReturns[0].relativeX = 999;
if (clonedKeyframe.pose.x !== clonedPoseX || clonedKeyframe.returns[0].relativeX !== clonedFirstX) {
  throw new Error('Expected rolling keyframes to remain independent after source telemetry mutation.');
}
let boundedKeyframes = [];
for (let index = 0; index < ROLLING_SENSOR_MAP_KEYFRAMES + 5; index += 1) {
  boundedKeyframes = appendRollingSensorKeyframe(boundedKeyframes, {
    ...decimatedKeyframe,
    pose: { ...decimatedKeyframe.pose, x: index },
  });
}
const translatedPoint = new THREE.Vector3(1, 0, 0).applyMatrix4(getRecordedToCurrentMatrix(
  { ...scanPose, x: 10, y: 0, z: 0 },
  { ...scanPose, x: 4, y: 0, z: 0 },
));
const translatedPose = getPoseInCurrentFrame(
  { ...scanPose, x: 10, y: 0, z: 0 },
  { ...scanPose, x: 4, y: 0, z: 0 },
);
if (
  decimatedKeyframe.returns.length !== 2
  || boundedKeyframes.length !== ROLLING_SENSOR_MAP_KEYFRAMES
  || boundedKeyframes[0].pose.x !== 5
  || Math.abs(translatedPoint.x - 7) > 1e-9
  || Math.abs(translatedPose.x - 6) > 1e-9
) {
  throw new Error('Expected deterministic rolling-map decimation, 30-keyframe memory bounds, and current-frame pose transforms.');
}
const noHitSamples = scan.samples.filter((sample) => !sample.hit);
if (
  noHitSamples.length !== 3
  || noHitSamples.some((sample) => Math.abs(sample.endpoint.distanceTo(sample.origin) - config.maxRange) > 1e-6)
) {
  throw new Error('Expected every no-hit sample to retain a max-range spherical-envelope endpoint.');
}
const nearJet = getLidarJetColor(0);
const farJet = getLidarJetColor(1);
if (
  nearJet.r < 0.99
  || nearJet.g > 0.01
  || farJet.b < 0.99
  || farJet.r > 0.01
  || getLidarJetCss(0) !== '#ff0000'
  || getLidarJetCss(1) !== '#0000ff'
) {
  throw new Error('Expected JET distance endpoints to map near=red and far=blue.');
}

const groundScan = scanLidar(
  { position: new THREE.Vector3(0, 12, 0), yaw: 0, pitch: 0, roll: 0 },
  [],
  { maxRange: 40, sampleCount: 2 },
  [new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 1, 0)],
);
assertObservationSlots(groundScan.observation, [
  { directionLocalX: 0, directionLocalY: -1, directionLocalZ: 0, normalizedDistance: 0.3, hit: 1 },
  { directionLocalX: 0, directionLocalY: 1, directionLocalZ: 0, normalizedDistance: 1, hit: 0 },
], '3D ground hit/miss observation');
if (
  !groundScan.samples[0].hit
  || Math.abs(groundScan.samples[0].distance - 12) > 1e-6
  || Math.abs(groundScan.samples[0].hitPoint?.y ?? Infinity) > 1e-6
  || groundScan.samples[1].hit
  || groundScan.observation.length !== 2 * LIDAR_OBSERVATION_VALUES_PER_SAMPLE
) {
  throw new Error('Expected downward 3D rays to return the y=0 ground plane through the unchanged five-value sample contract.');
}

const roofMeshes = buildBuildingCollisionMeshes([{ height: 20, footprint: [[-5, -5], [5, -5], [5, 5], [-5, 5]] }]);
const roofBeforeGround = scanLidar(
  { position: new THREE.Vector3(0, 30, 0), yaw: 0, pitch: 0, roll: 0 },
  roofMeshes,
  { maxRange: 50, sampleCount: 1 },
  [new THREE.Vector3(0, -1, 0)],
).samples[0];
if (!roofBeforeGround.hit || Math.abs(roofBeforeGround.distance - 10) > 1e-6) {
  throw new Error(`Expected nearest OSM roof before ground at 10 m, got ${roofBeforeGround.distance.toFixed(6)} m.`);
}

if (!isScenePointInsideBuilding(building, new THREE.Vector3(0, 10, -30))) {
  throw new Error('Expected flight collision and rendered prism to share scene-Z coordinates.');
}
if (isScenePointInsideBuilding(building, new THREE.Vector3(0, 10, 30))) {
  throw new Error('Building collision was mirrored across scene Z.');
}

const config2D = { maxRange: 100, horizontalSamples: 4 };
const defaultDirections2D = createLidar2DLocalDirections(DEFAULT_LIDAR_2D_CONFIG);
if (
  DEFAULT_LIDAR_2D_CONFIG.maxRange !== 180
  || DEFAULT_LIDAR_2D_CONFIG.horizontalSamples !== 180
  || defaultDirections2D.length !== 180
) {
  throw new Error(`Expected physical 180 m / 180-ray default 2D LiDAR, got ${DEFAULT_LIDAR_2D_CONFIG.maxRange} m / ${defaultDirections2D.length} rays.`);
}
const directions2D = createLidar2DLocalDirections(config2D);
if (
  directions2D.length !== 4
  || Math.abs(directions2D[0].x - 1) > 1e-12
  || Math.abs(directions2D[0].y) > 1e-12
  || Math.abs(directions2D[1].x) > 1e-12
  || Math.abs(directions2D[1].y - 1) > 1e-12
) {
  throw new Error('Expected deterministic counter-clockwise 2D LiDAR sample order starting at local +X.');
}
const scan2D = scanLidar2D(
  { position: { x: 0, y: 0 }, heading: 0 },
  [{ height: 20, footprint: [[29, -5], [31, -5], [31, 5], [29, 5]] }],
  config2D,
  directions2D,
);
const forward2D = scan2D.samples[0];
if (!forward2D.hit || !forward2D.point || Math.abs(forward2D.distance - 29) > 1e-6) {
  throw new Error(`Expected 2D footprint edge at 29 m, got ${forward2D.distance.toFixed(6)} m.`);
}
if (Math.abs(forward2D.point.x - 29) > 1e-6 || Math.abs(forward2D.point.y) > 1e-6) {
  throw new Error(`Expected 2D hit point (29, 0), got (${forward2D.point.x}, ${forward2D.point.y}).`);
}
const expected2DObservationLength = scan2D.samples.length * LIDAR_OBSERVATION_VALUES_PER_SAMPLE;
if (scan2D.observation.length !== expected2DObservationLength || scan2D.observation[2] !== 0) {
  throw new Error(`Expected ${expected2DObservationLength} 2D observation values with local z=0.`);
}
assertObservationSlots(scan2D.observation, [
  { directionLocalX: 1, directionLocalY: 0, directionLocalZ: 0, normalizedDistance: 0.29, hit: 1 },
  { directionLocalX: 0, directionLocalY: 1, directionLocalZ: 0, normalizedDistance: 1, hit: 0 },
  { directionLocalX: -1, directionLocalY: 0, directionLocalZ: 0, normalizedDistance: 1, hit: 0 },
  { directionLocalX: 0, directionLocalY: -1, directionLocalZ: 0, normalizedDistance: 1, hit: 0 },
], '2D footprint hit/miss observation');
const boundedScan2D = scanLidar2D(
  { position: { x: 0, y: 0 }, heading: 0 },
  [],
  config2D,
  directions2D,
  { min_x: -40, max_x: 40, min_y: -30, max_y: 30 },
);
if (
  boundedScan2D.samples.some((sample) => !sample.hit)
  || JSON.stringify(boundedScan2D.samples.map((sample) => Math.round(sample.distance))) !== JSON.stringify([40, 30, 40, 30])
) {
  throw new Error('Expected browser 2D LiDAR to share the Gym solid rectangular-domain boundary semantics.');
}
const telemetry2D = summarizeLidar2DScan(scan2D, {
  x: 0,
  y: 0,
  z: 0,
  yaw: 0,
  pitch: 0,
  roll: 0,
});
if (
  telemetry2D.ui.returns.length !== config2D.horizontalSamples
  || telemetry2D.ui.returns[0].hit !== true
  || telemetry2D.ui.returns[1].hit !== false
  || Math.abs(telemetry2D.ui.returns[1].relativeZ - config2D.maxRange) > 1e-6
  || telemetry2D.observation !== scan2D.observation
) {
  throw new Error('Expected 2D display telemetry to include drone-local hit/miss samples without changing its RL observation.');
}

const representativeInhaBuildings = [
  { height: 24, footprint: [[-18, -18], [18, -18], [18, 18], [-18, 18]] },
  { height: 18, footprint: [[-28, 92], [28, 92], [28, 108], [-28, 108]] },
  { height: 15, footprint: [[112, 116], [124, 116], [124, 142], [112, 142]] },
];
const representativeBounds = { min_x: -200, max_x: 200, min_y: -200, max_y: 200 };
const spawn2D = selectLidar2DSpawn(
  representativeBounds,
  representativeInhaBuildings,
  16,
  DEFAULT_LIDAR_2D_CONFIG,
);
const spawnScan2D = scanLidar2D(
  { position: spawn2D, heading: 0 },
  representativeInhaBuildings,
  DEFAULT_LIDAR_2D_CONFIG,
);
const spawnHitCount = spawnScan2D.samples.filter((sample) => sample.hit).length;
if (
  Math.abs(spawn2D.x) > 1e-12
  || Math.abs(spawn2D.y - 72) > 1e-12
  || spawnHitCount < Math.ceil(DEFAULT_LIDAR_2D_CONFIG.horizontalSamples * 0.04)
) {
  throw new Error(`Expected deterministic clear default-like 2D spawn with visible real returns, got (${spawn2D.x}, ${spawn2D.y}) and ${spawnHitCount} hits.`);
}

collisionMeshes.forEach((mesh) => {
  mesh.geometry.dispose();
  if (Array.isArray(mesh.material)) mesh.material.forEach((material) => material.dispose());
  else mesh.material.dispose();
});
roofMeshes.forEach((mesh) => {
  mesh.geometry.dispose();
  if (Array.isArray(mesh.material)) mesh.material.forEach((material) => material.dispose());
  else mesh.material.dispose();
});

console.log(`LiDAR smoke passed: default=${defaultDirections.length} rays @ ${DEFAULT_LIDAR_CONFIG.maxRange}m 3D=${forwardSample.distance.toFixed(1)}m 2D=${forward2D.distance.toFixed(1)}m spawn=${spawnHitCount}/${DEFAULT_LIDAR_2D_CONFIG.horizontalSamples} observations=${scan.observation.length}/${scan2D.observation.length}`);
await vite.close();
