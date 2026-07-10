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

export function createRollingSensorKeyframe(
  lidar: LidarTelemetry,
  pointLimit: number,
): RollingSensorKeyframe {
  const stride = Math.max(1, Math.ceil(lidar.ui.returns.length / Math.max(1, pointLimit)));
  return {
    pose: { ...lidar.ui.scanPose },
    returns: lidar.ui.returns
      .filter((_, index) => index % stride === 0)
      .slice(0, pointLimit)
      .map((sample) => ({ ...sample })),
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
