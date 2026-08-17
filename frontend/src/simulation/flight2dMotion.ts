export interface Flight2DVector {
  x: number;
  y: number;
}

export interface Flight2DMotionState {
  position: Flight2DVector;
  velocity: Flight2DVector;
  heading: number;
}

export interface Flight2DBounds {
  min_x: number;
  max_x: number;
  min_y: number;
  max_y: number;
}

export interface Flight2DCollisionObstacle {
  footprint: number[][];
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface Flight2DMotionOptions {
  maxSpeed?: number;
  thrust?: number;
  dragPerSecond?: number;
  windAccelerationScale?: number;
  collisionRadius?: number;
}

export const FLIGHT_2D_FIXED_STEP_S = 1 / 120;
export const FLIGHT_2D_MAX_FRAME_DELTA_S = 0.25;
export const FLIGHT_2D_MAX_CATCH_UP_STEPS = 30;

const DEFAULT_MOTION_OPTIONS: Required<Flight2DMotionOptions> = {
  maxSpeed: 32,
  thrust: 34,
  dragPerSecond: 0.9,
  windAccelerationScale: 0.22,
  collisionRadius: 12,
};
const COLLISION_EPSILON = 1e-9;
const SAFE_FRACTION_ITERATIONS = 18;

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function normalize(vector: Flight2DVector) {
  const magnitude = Math.hypot(vector.x, vector.y);
  return magnitude > 1e-9
    ? { x: vector.x / magnitude, y: vector.y / magnitude }
    : { x: 0, y: 0 };
}

function pointInPolygon(point: Flight2DVector, polygon: number[][]) {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index, index += 1) {
    const [x, y] = polygon[index];
    const [previousX, previousY] = polygon[previous];
    const crossesY = (y > point.y) !== (previousY > point.y);
    const edgeX = ((previousX - x) * (point.y - y)) / (previousY - y || Number.EPSILON) + x;
    if (crossesY && point.x < edgeX) inside = !inside;
  }
  return inside;
}

function pointSegmentDistanceSquared(point: Flight2DVector, start: Flight2DVector, end: Flight2DVector) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= COLLISION_EPSILON) {
    return (point.x - start.x) ** 2 + (point.y - start.y) ** 2;
  }
  const fraction = clamp(
    ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared,
    0,
    1,
  );
  const closestX = start.x + dx * fraction;
  const closestY = start.y + dy * fraction;
  return (point.x - closestX) ** 2 + (point.y - closestY) ** 2;
}

function cross(origin: Flight2DVector, a: Flight2DVector, b: Flight2DVector) {
  return (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);
}

function pointOnSegment(point: Flight2DVector, start: Flight2DVector, end: Flight2DVector) {
  return point.x >= Math.min(start.x, end.x) - COLLISION_EPSILON
    && point.x <= Math.max(start.x, end.x) + COLLISION_EPSILON
    && point.y >= Math.min(start.y, end.y) - COLLISION_EPSILON
    && point.y <= Math.max(start.y, end.y) + COLLISION_EPSILON;
}

function segmentsIntersect(
  firstStart: Flight2DVector,
  firstEnd: Flight2DVector,
  secondStart: Flight2DVector,
  secondEnd: Flight2DVector,
) {
  const c1 = cross(firstStart, firstEnd, secondStart);
  const c2 = cross(firstStart, firstEnd, secondEnd);
  const c3 = cross(secondStart, secondEnd, firstStart);
  const c4 = cross(secondStart, secondEnd, firstEnd);
  if (
    ((c1 > COLLISION_EPSILON && c2 < -COLLISION_EPSILON) || (c1 < -COLLISION_EPSILON && c2 > COLLISION_EPSILON))
    && ((c3 > COLLISION_EPSILON && c4 < -COLLISION_EPSILON) || (c3 < -COLLISION_EPSILON && c4 > COLLISION_EPSILON))
  ) {
    return true;
  }
  return (Math.abs(c1) <= COLLISION_EPSILON && pointOnSegment(secondStart, firstStart, firstEnd))
    || (Math.abs(c2) <= COLLISION_EPSILON && pointOnSegment(secondEnd, firstStart, firstEnd))
    || (Math.abs(c3) <= COLLISION_EPSILON && pointOnSegment(firstStart, secondStart, secondEnd))
    || (Math.abs(c4) <= COLLISION_EPSILON && pointOnSegment(firstEnd, secondStart, secondEnd));
}

function segmentDistanceSquared(
  firstStart: Flight2DVector,
  firstEnd: Flight2DVector,
  secondStart: Flight2DVector,
  secondEnd: Flight2DVector,
) {
  if (segmentsIntersect(firstStart, firstEnd, secondStart, secondEnd)) return 0;
  return Math.min(
    pointSegmentDistanceSquared(firstStart, secondStart, secondEnd),
    pointSegmentDistanceSquared(firstEnd, secondStart, secondEnd),
    pointSegmentDistanceSquared(secondStart, firstStart, firstEnd),
    pointSegmentDistanceSquared(secondEnd, firstStart, firstEnd),
  );
}

function forEachObstacleEdge(
  obstacle: Flight2DCollisionObstacle,
  callback: (start: Flight2DVector, end: Flight2DVector) => boolean,
) {
  for (let index = 0; index < obstacle.footprint.length; index += 1) {
    const start = obstacle.footprint[index];
    const end = obstacle.footprint[(index + 1) % obstacle.footprint.length];
    if (callback({ x: start[0], y: start[1] }, { x: end[0], y: end[1] })) return true;
  }
  return false;
}

export function isFlight2DPositionBlocked(
  point: Flight2DVector,
  obstacles: readonly Flight2DCollisionObstacle[],
  collisionRadius: number,
) {
  const radius = Math.max(0, collisionRadius);
  const radiusSquared = radius * radius;
  return obstacles.some((obstacle) => {
    if (
      point.x < obstacle.minX - radius
      || point.x > obstacle.maxX + radius
      || point.y < obstacle.minY - radius
      || point.y > obstacle.maxY + radius
    ) {
      return false;
    }
    if (pointInPolygon(point, obstacle.footprint)) return true;
    return forEachObstacleEdge(
      obstacle,
      (start, end) => pointSegmentDistanceSquared(point, start, end) <= radiusSquared,
    );
  });
}

export function flight2DSweepHitsObstacle(
  start: Flight2DVector,
  end: Flight2DVector,
  obstacles: readonly Flight2DCollisionObstacle[],
  collisionRadius: number,
) {
  const radius = Math.max(0, collisionRadius);
  const radiusSquared = radius * radius;
  const minX = Math.min(start.x, end.x) - radius;
  const maxX = Math.max(start.x, end.x) + radius;
  const minY = Math.min(start.y, end.y) - radius;
  const maxY = Math.max(start.y, end.y) + radius;

  return obstacles.some((obstacle) => {
    if (maxX < obstacle.minX || minX > obstacle.maxX || maxY < obstacle.minY || minY > obstacle.maxY) {
      return false;
    }
    if (pointInPolygon(start, obstacle.footprint) || pointInPolygon(end, obstacle.footprint)) return true;
    return forEachObstacleEdge(
      obstacle,
      (edgeStart, edgeEnd) => segmentDistanceSquared(start, end, edgeStart, edgeEnd) <= radiusSquared,
    );
  });
}

function moveToSafeFraction(
  start: Flight2DVector,
  end: Flight2DVector,
  obstacles: readonly Flight2DCollisionObstacle[],
  collisionRadius: number,
) {
  if (!flight2DSweepHitsObstacle(start, end, obstacles, collisionRadius)) return end;
  if (isFlight2DPositionBlocked(start, obstacles, collisionRadius)) return start;

  let safeFraction = 0;
  let blockedFraction = 1;
  for (let iteration = 0; iteration < SAFE_FRACTION_ITERATIONS; iteration += 1) {
    const candidateFraction = (safeFraction + blockedFraction) / 2;
    const candidate = {
      x: start.x + (end.x - start.x) * candidateFraction,
      y: start.y + (end.y - start.y) * candidateFraction,
    };
    if (flight2DSweepHitsObstacle(start, candidate, obstacles, collisionRadius)) {
      blockedFraction = candidateFraction;
    } else {
      safeFraction = candidateFraction;
    }
  }
  return {
    x: start.x + (end.x - start.x) * safeFraction,
    y: start.y + (end.y - start.y) * safeFraction,
  };
}

export function stepFlight2DMotion(
  state: Flight2DMotionState,
  input: Flight2DVector,
  wind: Flight2DVector,
  obstacles: readonly Flight2DCollisionObstacle[],
  bounds: Flight2DBounds,
  deltaSeconds = FLIGHT_2D_FIXED_STEP_S,
  options: Flight2DMotionOptions = {},
): Flight2DMotionState {
  if (!Number.isFinite(deltaSeconds) || deltaSeconds <= 0) return state;
  const resolvedOptions = { ...DEFAULT_MOTION_OPTIONS, ...options };
  const direction = normalize(input);
  const heading = Math.hypot(direction.x, direction.y) > 0
    ? Math.atan2(direction.y, direction.x)
    : state.heading;
  const damping = Math.exp(-resolvedOptions.dragPerSecond * deltaSeconds);
  const velocity = {
    x: (
      state.velocity.x
      + (direction.x * resolvedOptions.thrust + wind.x * resolvedOptions.windAccelerationScale) * deltaSeconds
    ) * damping,
    y: (
      state.velocity.y
      + (direction.y * resolvedOptions.thrust + wind.y * resolvedOptions.windAccelerationScale) * deltaSeconds
    ) * damping,
  };
  const speed = Math.hypot(velocity.x, velocity.y);
  if (speed > resolvedOptions.maxSpeed) {
    const speedScale = resolvedOptions.maxSpeed / speed;
    velocity.x *= speedScale;
    velocity.y *= speedScale;
  }

  const radius = Math.max(0, resolvedOptions.collisionRadius);
  const minX = Math.min(bounds.max_x, bounds.min_x + radius);
  const maxX = Math.max(bounds.min_x, bounds.max_x - radius);
  const minY = Math.min(bounds.max_y, bounds.min_y + radius);
  const maxY = Math.max(bounds.min_y, bounds.max_y - radius);
  const rawEnd = {
    x: state.position.x + velocity.x * deltaSeconds,
    y: state.position.y + velocity.y * deltaSeconds,
  };
  const boundedEnd = {
    x: clamp(rawEnd.x, minX, maxX),
    y: clamp(rawEnd.y, minY, maxY),
  };
  if (boundedEnd.x !== rawEnd.x) velocity.x = 0;
  if (boundedEnd.y !== rawEnd.y) velocity.y = 0;

  let position: Flight2DVector;
  if (!flight2DSweepHitsObstacle(state.position, boundedEnd, obstacles, radius)) {
    position = boundedEnd;
  } else {
    position = { ...state.position };
    const xTarget = { x: boundedEnd.x, y: position.y };
    const safeX = moveToSafeFraction(position, xTarget, obstacles, radius);
    if (Math.abs(safeX.x - xTarget.x) > COLLISION_EPSILON) velocity.x = 0;
    position = safeX;

    const yTarget = { x: position.x, y: boundedEnd.y };
    const safeY = moveToSafeFraction(position, yTarget, obstacles, radius);
    if (Math.abs(safeY.y - yTarget.y) > COLLISION_EPSILON) velocity.y = 0;
    position = safeY;
  }

  return { position, velocity, heading };
}

export function consumeFlight2DFixedSteps(
  accumulatorSeconds: number,
  frameDeltaSeconds: number,
  step: (deltaSeconds: number) => void,
) {
  const accumulator = Number.isFinite(accumulatorSeconds) ? Math.max(0, accumulatorSeconds) : 0;
  const frameDelta = Number.isFinite(frameDeltaSeconds)
    ? clamp(frameDeltaSeconds, 0, FLIGHT_2D_MAX_FRAME_DELTA_S)
    : 0;
  let pendingSeconds = accumulator + frameDelta;
  const requestedSteps = Math.floor(
    (pendingSeconds + FLIGHT_2D_FIXED_STEP_S * 1e-9) / FLIGHT_2D_FIXED_STEP_S,
  );
  const steps = Math.min(requestedSteps, FLIGHT_2D_MAX_CATCH_UP_STEPS);
  for (let index = 0; index < steps; index += 1) step(FLIGHT_2D_FIXED_STEP_S);
  pendingSeconds -= steps * FLIGHT_2D_FIXED_STEP_S;

  // If a throttled tab exceeds the catch-up budget, drop whole stale steps and
  // retain only interpolation remainder. Normal frame schedules lose no time.
  if (requestedSteps > FLIGHT_2D_MAX_CATCH_UP_STEPS) {
    pendingSeconds %= FLIGHT_2D_FIXED_STEP_S;
  }
  if (Math.abs(pendingSeconds) < FLIGHT_2D_FIXED_STEP_S * 1e-9) pendingSeconds = 0;
  return {
    accumulatorSeconds: pendingSeconds,
    simulatedSeconds: steps * FLIGHT_2D_FIXED_STEP_S,
    steps,
  };
}
