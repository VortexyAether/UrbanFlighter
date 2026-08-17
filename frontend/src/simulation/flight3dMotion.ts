import type { BuildingData } from '../api';
import type { Flight3DCommand } from './flight3dControls';

export interface Flight3DVector {
  x: number;
  y: number;
  z: number;
}

export interface Flight3DState {
  position: Flight3DVector;
  velocity: Flight3DVector;
  yaw: number;
}

export interface Flight3DBounds {
  min_x: number;
  max_x: number;
  min_y: number;
  max_y: number;
}

export interface Flight3DObstacle {
  footprint: number[][];
  height: number;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface Flight3DMotionOptions {
  safetyRadiusM: number;
  verticalSafetyClearanceM?: number;
  minAltitudeM?: number;
  maxAltitudeM?: number;
  maxHorizontalSpeedMps?: number;
  boostedHorizontalSpeedMps?: number;
  maxVerticalSpeedMps?: number;
  boostedVerticalSpeedMps?: number;
}

export interface Flight3DSpawnOptions {
  baseAltitudeM?: number;
  desiredBuildingClearanceM?: number;
  favorCityAhead?: boolean;
  verticalSafetyClearanceM?: number;
}

export const FLIGHT_3D_FIXED_STEP_S = 1 / 120;
export const FLIGHT_3D_MAX_FRAME_DELTA_S = 0.25;
export const FLIGHT_3D_MAX_CATCH_UP_STEPS = 30;

const FORWARD_ACCEL_MPS2 = 24;
const REVERSE_ACCEL_MPS2 = 18;
const STRAFE_ACCEL_MPS2 = 21;
const VERTICAL_ACCEL_MPS2 = 19;
const YAW_RATE_RAD_S = 1.8;
const HORIZONTAL_DAMPING_PER_S = 1.18;
const VERTICAL_DAMPING_PER_S = 1.55;
const BRAKE_DAMPING_PER_S = 8.5;
const BOOST_ACCELERATION_SCALE = 1.55;
const WIND_ACCELERATION_SCALE = 0.03;
const COLLISION_EPSILON = 1e-9;

const DEFAULT_MOTION_LIMITS = Object.freeze({
  minAltitudeM: 5,
  maxAltitudeM: 300,
  maxHorizontalSpeedMps: 24,
  boostedHorizontalSpeedMps: 36,
  maxVerticalSpeedMps: 12,
  boostedVerticalSpeedMps: 17,
});

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function clampAxis(value: number) {
  return clamp(Number.isFinite(value) ? value : 0, -1, 1);
}

function pointInPolygon(pointX: number, pointY: number, footprint: readonly number[][]) {
  let inside = false;
  for (let index = 0, previous = footprint.length - 1; index < footprint.length; previous = index, index += 1) {
    const [x, y] = footprint[index];
    const [previousX, previousY] = footprint[previous];
    const crossesY = (y > pointY) !== (previousY > pointY);
    const edgeX = ((previousX - x) * (pointY - y)) / (previousY - y || Number.EPSILON) + x;
    if (crossesY && pointX < edgeX) inside = !inside;
  }
  return inside;
}

function pointSegmentDistanceSquared(
  pointX: number,
  pointY: number,
  start: readonly number[],
  end: readonly number[],
) {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= COLLISION_EPSILON) return (pointX - start[0]) ** 2 + (pointY - start[1]) ** 2;
  const fraction = clamp(((pointX - start[0]) * dx + (pointY - start[1]) * dy) / lengthSquared, 0, 1);
  const nearestX = start[0] + dx * fraction;
  const nearestY = start[1] + dy * fraction;
  return (pointX - nearestX) ** 2 + (pointY - nearestY) ** 2;
}

function horizontalPointBlocked(
  x: number,
  fieldY: number,
  obstacle: Flight3DObstacle,
  radius: number,
) {
  if (
    x < obstacle.minX - radius
    || x > obstacle.maxX + radius
    || fieldY < obstacle.minY - radius
    || fieldY > obstacle.maxY + radius
  ) return false;
  if (pointInPolygon(x, fieldY, obstacle.footprint)) return true;
  const radiusSquared = radius * radius;
  return obstacle.footprint.some((start, index) => pointSegmentDistanceSquared(
    x,
    fieldY,
    start,
    obstacle.footprint[(index + 1) % obstacle.footprint.length],
  ) <= radiusSquared);
}

export function buildFlight3DObstacles(buildings: readonly BuildingData[]): Flight3DObstacle[] {
  return buildings.flatMap((building) => {
    if (!building.footprint || building.footprint.length < 3 || building.height <= 0) return [];
    const xs = building.footprint.map((point) => point[0]);
    const ys = building.footprint.map((point) => point[1]);
    return [{
      footprint: building.footprint,
      height: building.height,
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    }];
  });
}

export function isFlight3DPositionBlocked(
  position: Flight3DVector,
  obstacles: readonly Flight3DObstacle[],
  safetyRadiusM: number,
  verticalSafetyClearanceM = safetyRadiusM,
) {
  const radius = Math.max(0, safetyRadiusM);
  const verticalClearance = Math.max(0, verticalSafetyClearanceM);
  const fieldY = -position.z;
  return obstacles.some((obstacle) => (
    position.y - verticalClearance <= obstacle.height
    && position.y + verticalClearance >= 0
    && horizontalPointBlocked(position.x, fieldY, obstacle, radius)
  ));
}

export function flight3DSegmentHitsObstacle(
  start: Flight3DVector,
  end: Flight3DVector,
  obstacles: readonly Flight3DObstacle[],
  safetyRadiusM: number,
  verticalSafetyClearanceM = safetyRadiusM,
) {
  const distance = Math.hypot(end.x - start.x, end.y - start.y, end.z - start.z);
  const sampleSpacing = Math.max(0.18, Math.max(0, safetyRadiusM) * 0.38);
  const samples = Math.max(1, Math.ceil(distance / sampleSpacing));
  for (let index = 1; index <= samples; index += 1) {
    const fraction = index / samples;
    if (isFlight3DPositionBlocked({
      x: start.x + (end.x - start.x) * fraction,
      y: start.y + (end.y - start.y) * fraction,
      z: start.z + (end.z - start.z) * fraction,
    }, obstacles, safetyRadiusM, verticalSafetyClearanceM)) return true;
  }
  return false;
}

function normalizeYaw(yaw: number) {
  return Math.atan2(Math.sin(yaw), Math.cos(yaw));
}

function copyVector(vector: Flight3DVector): Flight3DVector {
  return { x: vector.x, y: vector.y, z: vector.z };
}

function resolveCollisionAxes(
  start: Flight3DVector,
  end: Flight3DVector,
  velocity: Flight3DVector,
  obstacles: readonly Flight3DObstacle[],
  safetyRadiusM: number,
  verticalSafetyClearanceM: number,
) {
  if (!flight3DSegmentHitsObstacle(start, end, obstacles, safetyRadiusM, verticalSafetyClearanceM)) return end;
  const resolved = copyVector(start);
  for (const axis of ['x', 'z', 'y'] as const) {
    const candidate = copyVector(resolved);
    candidate[axis] = end[axis];
    if (flight3DSegmentHitsObstacle(resolved, candidate, obstacles, safetyRadiusM, verticalSafetyClearanceM)) {
      velocity[axis] = 0;
    } else {
      resolved[axis] = candidate[axis];
    }
  }
  return resolved;
}

export function stepFlight3DMotion(
  state: Flight3DState,
  rawCommand: Flight3DCommand,
  wind: Flight3DVector,
  obstacles: readonly Flight3DObstacle[],
  bounds: Flight3DBounds,
  options: Flight3DMotionOptions,
  deltaSeconds = FLIGHT_3D_FIXED_STEP_S,
): Flight3DState {
  if (!Number.isFinite(deltaSeconds) || deltaSeconds <= 0) return state;
  const limits = { ...DEFAULT_MOTION_LIMITS, ...options };
  const safetyRadius = Math.max(0, limits.safetyRadiusM);
  const verticalSafetyClearance = Math.max(0, options.verticalSafetyClearanceM ?? safetyRadius);
  const command = {
    forward: clampAxis(rawCommand.forward),
    strafe: clampAxis(rawCommand.strafe),
    yaw: clampAxis(rawCommand.yaw),
    lift: clampAxis(rawCommand.lift),
    boost: Boolean(rawCommand.boost),
    brake: Boolean(rawCommand.brake),
  };
  const yaw = normalizeYaw(state.yaw + command.yaw * YAW_RATE_RAD_S * deltaSeconds);
  const forward = { x: -Math.sin(yaw), z: -Math.cos(yaw) };
  const right = { x: Math.cos(yaw), z: -Math.sin(yaw) };
  const horizontalInputMagnitude = Math.hypot(command.forward, command.strafe);
  const horizontalScale = horizontalInputMagnitude > 1 ? 1 / horizontalInputMagnitude : 1;
  const accelerationScale = command.boost ? BOOST_ACCELERATION_SCALE : 1;
  const forwardAcceleration = command.forward >= 0 ? FORWARD_ACCEL_MPS2 : REVERSE_ACCEL_MPS2;
  const velocity = copyVector(state.velocity);

  velocity.x += (
    forward.x * command.forward * forwardAcceleration * horizontalScale * accelerationScale
    + right.x * command.strafe * STRAFE_ACCEL_MPS2 * horizontalScale * accelerationScale
    + wind.x * WIND_ACCELERATION_SCALE
  ) * deltaSeconds;
  velocity.z += (
    forward.z * command.forward * forwardAcceleration * horizontalScale * accelerationScale
    + right.z * command.strafe * STRAFE_ACCEL_MPS2 * horizontalScale * accelerationScale
    + wind.z * WIND_ACCELERATION_SCALE
  ) * deltaSeconds;
  velocity.y += (
    command.lift * VERTICAL_ACCEL_MPS2 * accelerationScale
    + wind.y * WIND_ACCELERATION_SCALE
  ) * deltaSeconds;

  let horizontalDamping = HORIZONTAL_DAMPING_PER_S;
  let verticalDamping = VERTICAL_DAMPING_PER_S;
  if (command.brake) {
    horizontalDamping += BRAKE_DAMPING_PER_S;
    verticalDamping += BRAKE_DAMPING_PER_S * 0.7;
  }
  velocity.x *= Math.exp(-horizontalDamping * deltaSeconds);
  velocity.z *= Math.exp(-horizontalDamping * deltaSeconds);
  velocity.y *= Math.exp(-verticalDamping * deltaSeconds);

  const maxHorizontalSpeed = command.boost
    ? limits.boostedHorizontalSpeedMps
    : limits.maxHorizontalSpeedMps;
  const horizontalSpeed = Math.hypot(velocity.x, velocity.z);
  if (horizontalSpeed > maxHorizontalSpeed) {
    const speedScale = maxHorizontalSpeed / horizontalSpeed;
    velocity.x *= speedScale;
    velocity.z *= speedScale;
  }
  const maxVerticalSpeed = command.boost ? limits.boostedVerticalSpeedMps : limits.maxVerticalSpeedMps;
  velocity.y = clamp(velocity.y, -maxVerticalSpeed, maxVerticalSpeed);

  const rawEnd = {
    x: state.position.x + velocity.x * deltaSeconds,
    y: state.position.y + velocity.y * deltaSeconds,
    z: state.position.z + velocity.z * deltaSeconds,
  };
  const minX = Math.min(bounds.max_x, bounds.min_x + safetyRadius);
  const maxX = Math.max(bounds.min_x, bounds.max_x - safetyRadius);
  const minZ = Math.min(-bounds.min_y, -bounds.max_y + safetyRadius);
  const maxZ = Math.max(-bounds.max_y, -bounds.min_y - safetyRadius);
  const boundedEnd = {
    x: clamp(rawEnd.x, minX, maxX),
    y: clamp(rawEnd.y, limits.minAltitudeM, limits.maxAltitudeM),
    z: clamp(rawEnd.z, minZ, maxZ),
  };
  if (boundedEnd.x !== rawEnd.x) velocity.x = 0;
  if (boundedEnd.y !== rawEnd.y) velocity.y = 0;
  if (boundedEnd.z !== rawEnd.z) velocity.z = 0;

  return {
    position: resolveCollisionAxes(
      state.position,
      boundedEnd,
      velocity,
      obstacles,
      safetyRadius,
      verticalSafetyClearance,
    ),
    velocity,
    yaw,
  };
}

export function consumeFlight3DFixedSteps(
  accumulatorSeconds: number,
  frameDeltaSeconds: number,
  step: (deltaSeconds: number) => void,
) {
  const accumulator = Number.isFinite(accumulatorSeconds) ? Math.max(0, accumulatorSeconds) : 0;
  const frameDelta = Number.isFinite(frameDeltaSeconds)
    ? clamp(frameDeltaSeconds, 0, FLIGHT_3D_MAX_FRAME_DELTA_S)
    : 0;
  let remaining = accumulator + frameDelta;
  let steps = 0;
  while (remaining + COLLISION_EPSILON >= FLIGHT_3D_FIXED_STEP_S && steps < FLIGHT_3D_MAX_CATCH_UP_STEPS) {
    step(FLIGHT_3D_FIXED_STEP_S);
    remaining -= FLIGHT_3D_FIXED_STEP_S;
    steps += 1;
  }
  if (steps === FLIGHT_3D_MAX_CATCH_UP_STEPS && remaining >= FLIGHT_3D_FIXED_STEP_S) {
    remaining %= FLIGHT_3D_FIXED_STEP_S;
  }
  return {
    accumulatorSeconds: Math.max(0, remaining),
    steps,
    simulatedSeconds: steps * FLIGHT_3D_FIXED_STEP_S,
  };
}

export function selectFlight3DSpawn(
  bounds: Flight3DBounds,
  obstacles: readonly Flight3DObstacle[],
  safetyRadiusM: number,
  options: Flight3DSpawnOptions = {},
): Flight3DVector {
  const baseAltitudeM = options.baseAltitudeM ?? 28;
  const desiredBuildingClearanceM = options.desiredBuildingClearanceM ?? 22;
  const favorCityAhead = options.favorCityAhead ?? true;
  const verticalSafetyClearanceM = options.verticalSafetyClearanceM ?? safetyRadiusM;
  const minimumX = bounds.min_x + safetyRadiusM + 5;
  const maximumX = bounds.max_x - safetyRadiusM - 5;
  const minimumY = bounds.min_y + safetyRadiusM + 5;
  const maximumY = bounds.max_y - safetyRadiusM - 5;
  const centerX = (bounds.min_x + bounds.max_x) / 2;
  const centerY = (bounds.min_y + bounds.max_y) / 2;
  const horizontalClearance = (x: number, fieldY: number) => {
    let clearance = Number.POSITIVE_INFINITY;
    for (const obstacle of obstacles) {
      if (pointInPolygon(x, fieldY, obstacle.footprint)) return -1;
      obstacle.footprint.forEach((start, index) => {
        clearance = Math.min(
          clearance,
          Math.sqrt(pointSegmentDistanceSquared(
            x,
            fieldY,
            start,
            obstacle.footprint[(index + 1) % obstacle.footprint.length],
          )),
        );
      });
    }
    return clearance;
  };
  let selected: { x: number; fieldY: number; score: number } | null = null;
  const gridSize = 31;
  for (let row = 0; row < gridSize; row += 1) {
    for (let column = 0; column < gridSize; column += 1) {
      const x = minimumX + (column / (gridSize - 1)) * Math.max(0, maximumX - minimumX);
      const fieldY = minimumY + (row / (gridSize - 1)) * Math.max(0, maximumY - minimumY);
      const position = { x, y: baseAltitudeM, z: -fieldY };
      const cameraPosition = { x: x + 0.7, y: baseAltitudeM + 3.2, z: -fieldY + 7.6 };
      if (
        isFlight3DPositionBlocked(position, obstacles, safetyRadiusM, verticalSafetyClearanceM)
        || isFlight3DPositionBlocked(cameraPosition, obstacles, 0.35)
      ) continue;
      const clearance = horizontalClearance(x, fieldY);
      if (clearance < safetyRadiusM + 3) continue;
      const frontClearance = horizontalClearance(x, fieldY + 24);
      const cityAheadBonus = favorCityAhead && Number.isFinite(frontClearance)
        ? Math.max(0, 16 - Math.abs(frontClearance - 4))
        : 0;
      const centerDistance = Math.hypot(x - centerX, fieldY - centerY);
      const score = -Math.abs(clearance - desiredBuildingClearanceM) * 1.15 + cityAheadBonus - centerDistance * 0.012;
      if (!selected || score > selected.score) selected = { x, fieldY, score };
    }
  }
  const x = selected?.x ?? clamp(0, bounds.min_x + safetyRadiusM, bounds.max_x - safetyRadiusM);
  const fieldY = selected?.fieldY ?? clamp(0, bounds.min_y + safetyRadiusM, bounds.max_y - safetyRadiusM);
  const base = { x, y: baseAltitudeM, z: -fieldY };
  let requiredAltitude = base.y;
  obstacles.forEach((obstacle) => {
    if (horizontalPointBlocked(x, fieldY, obstacle, safetyRadiusM)) {
      requiredAltitude = Math.max(requiredAltitude, obstacle.height + verticalSafetyClearanceM + 5);
    }
  });
  return { ...base, y: clamp(requiredAltitude, baseAltitudeM, 220) };
}
