import * as THREE from 'three';
import type {
  LidarDisplayPose,
  LidarTelemetry,
  LidarUiReturn,
} from './lidar';

export const ROLLING_SENSOR_MAP_KEYFRAMES = 30;
export const ROLLING_SENSOR_MAP_3D_POINTS_PER_KEYFRAME = 200;
export const ROLLING_SENSOR_MAP_2D_POINTS_PER_KEYFRAME = 60;

export interface RollingSensorKeyframe {
  pose: LidarDisplayPose;
  returns: LidarUiReturn[];
}

interface IndexedReturn {
  index: number;
  sample: LidarUiReturn;
}

function selectEvenly(items: IndexedReturn[], count: number) {
  if (count <= 0) return [];
  if (count >= items.length) return items;

  // Pick the midpoint of each equal-width stratum. This fills every requested
  // slot without favoring the beginning or end of the deterministic scan order.
  return Array.from({ length: count }, (_, slot) => (
    items[Math.floor(((slot + 0.5) * items.length) / count)]
  ));
}

export function createRollingSensorKeyframe(
  lidar: LidarTelemetry,
  pointLimit: number,
): RollingSensorKeyframe {
  const limit = Number.isFinite(pointLimit) ? Math.max(0, Math.floor(pointLimit)) : 0;
  const indexedReturns = lidar.ui.returns.map((sample, index) => ({ index, sample }));
  const hits = indexedReturns.filter(({ sample }) => sample.hit);
  const misses = indexedReturns.filter(({ sample }) => !sample.hit);
  const selectedHits = selectEvenly(hits, Math.min(limit, hits.length));
  const selectedMisses = selectEvenly(misses, Math.min(limit - selectedHits.length, misses.length));
  const selected = [...selectedHits, ...selectedMisses].sort((a, b) => a.index - b.index);

  return {
    pose: { ...lidar.ui.scanPose },
    returns: selected.map(({ sample }) => ({ ...sample })),
  };
}

export function appendRollingSensorKeyframe(
  keyframes: RollingSensorKeyframe[],
  next: RollingSensorKeyframe,
  limit = ROLLING_SENSOR_MAP_KEYFRAMES,
) {
  return [...keyframes, next].slice(-Math.max(1, limit));
}

export function getLidarPoseMatrix(pose: LidarDisplayPose) {
  const rotation = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(pose.pitch, pose.yaw, pose.roll, 'YXZ'),
  );
  return new THREE.Matrix4().compose(
    new THREE.Vector3(pose.x, pose.y, pose.z),
    rotation,
    new THREE.Vector3(1, 1, 1),
  );
}

/** Converts coordinates from a recorded sensor frame into the live sensor frame. */
export function getRecordedToCurrentMatrix(
  recordedPose: LidarDisplayPose,
  currentPose: LidarDisplayPose,
) {
  return getLidarPoseMatrix(currentPose).invert().multiply(getLidarPoseMatrix(recordedPose));
}

export function getPoseInCurrentFrame(
  pose: LidarDisplayPose,
  currentPose: LidarDisplayPose,
  target = new THREE.Vector3(),
) {
  return target.set(pose.x, pose.y, pose.z).applyMatrix4(getLidarPoseMatrix(currentPose).invert());
}
