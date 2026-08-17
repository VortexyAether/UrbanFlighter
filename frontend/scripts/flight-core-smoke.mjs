import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const vite = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  appType: 'custom',
  server: { middlewareMode: true, hmr: false },
});

function assertNear(actual, expected, label, tolerance = 1e-9) {
  if (!Number.isFinite(actual) || Math.abs(actual - expected) > tolerance) {
    throw new Error(`${label}: expected ${expected}, got ${actual}.`);
  }
}

function rectangularObstacle(minX, maxX, minY, maxY) {
  return {
    footprint: [
      [minX, minY],
      [maxX, minY],
      [maxX, maxY],
      [minX, maxY],
    ],
    minX,
    maxX,
    minY,
    maxY,
  };
}

try {
  const {
    FLIGHT_2D_MAX_CATCH_UP_STEPS,
    consumeFlight2DFixedSteps,
    flight2DSweepHitsObstacle,
    isFlight2DPositionBlocked,
    stepFlight2DMotion,
  } = await vite.ssrLoadModule('/src/simulation/flight2dMotion.ts');
  const {
    eastNorthToCompassBearingDeg,
    mapAngleToCompassBearingDeg,
    sceneVectorToCompassBearingDeg,
  } = await vite.ssrLoadModule('/src/utils/bearings.ts');

  const cardinalBearings = [
    [eastNorthToCompassBearingDeg(0, 1), 0, 'north vector'],
    [eastNorthToCompassBearingDeg(1, 0), 90, 'east vector'],
    [eastNorthToCompassBearingDeg(0, -1), 180, 'south vector'],
    [eastNorthToCompassBearingDeg(-1, 0), 270, 'west vector'],
    [mapAngleToCompassBearingDeg(Math.PI / 2), 0, 'north-up map heading'],
    [mapAngleToCompassBearingDeg(0), 90, 'east map heading'],
    [sceneVectorToCompassBearingDeg(0, -1), 0, 'scene -Z north'],
    [sceneVectorToCompassBearingDeg(1, 0), 90, 'scene +X east'],
  ];
  cardinalBearings.forEach(([actual, expected, label]) => assertNear(actual, expected, label));

  const thinWall = rectangularObstacle(0, 0.2, -10, 10);
  if (!flight2DSweepHitsObstacle({ x: -20, y: 0 }, { x: 20, y: 0 }, [thinWall], 1)) {
    throw new Error('Swept-circle collision missed a thin wall crossed between endpoints.');
  }
  const stopped = stepFlight2DMotion(
    { position: { x: -20, y: 0 }, velocity: { x: 100, y: 0 }, heading: 0 },
    { x: 0, y: 0 },
    { x: 0, y: 0 },
    [thinWall],
    { min_x: -100, max_x: 100, min_y: -100, max_y: 100 },
    0.5,
    { maxSpeed: 200, thrust: 0, dragPerSecond: 0, collisionRadius: 1 },
  );
  if (
    stopped.position.x < -1.01
    || stopped.position.x > -0.999
    || Math.abs(stopped.velocity.x) > 1e-12
    || isFlight2DPositionBlocked(stopped.position, [thinWall], 1)
  ) {
    throw new Error(`Collision resolution was not safe at the thin wall: ${JSON.stringify(stopped)}.`);
  }

  const block = rectangularObstacle(0, 10, -10, 10);
  const sliding = stepFlight2DMotion(
    { position: { x: -5, y: -20 }, velocity: { x: 20, y: 20 }, heading: 0 },
    { x: 0, y: 0 },
    { x: 0, y: 0 },
    [block],
    { min_x: -100, max_x: 100, min_y: -100, max_y: 100 },
    0.5,
    { maxSpeed: 200, thrust: 0, dragPerSecond: 0, collisionRadius: 1 },
  );
  if (
    sliding.position.x < 4.99
    || sliding.position.y < -11.01
    || sliding.position.y > -10.999
    || Math.abs(sliding.velocity.x - 20) > 1e-12
    || Math.abs(sliding.velocity.y) > 1e-12
    || isFlight2DPositionBlocked(sliding.position, [block], 1)
  ) {
    throw new Error(`Expected deterministic facade sliding with only the blocked velocity removed: ${JSON.stringify(sliding)}.`);
  }

  function simulate(frameRate) {
    let state = {
      position: { x: 0, y: 0 },
      velocity: { x: 0, y: 0 },
      heading: 0,
    };
    let accumulatorSeconds = 0;
    let totalSteps = 0;
    for (let frame = 0; frame < frameRate * 2; frame += 1) {
      const result = consumeFlight2DFixedSteps(accumulatorSeconds, 1 / frameRate, (fixedDelta) => {
        state = stepFlight2DMotion(
          state,
          { x: 1, y: 0 },
          { x: 0, y: 0 },
          [],
          { min_x: -1000, max_x: 1000, min_y: -1000, max_y: 1000 },
          fixedDelta,
          { collisionRadius: 0 },
        );
      });
      accumulatorSeconds = result.accumulatorSeconds;
      totalSteps += result.steps;
    }
    return { state, accumulatorSeconds, totalSteps };
  }

  const reference = simulate(30);
  for (const frameRate of [60, 144]) {
    const candidate = simulate(frameRate);
    assertNear(candidate.state.position.x, reference.state.position.x, `${frameRate} Hz position`);
    assertNear(candidate.state.velocity.x, reference.state.velocity.x, `${frameRate} Hz velocity`);
    assertNear(candidate.state.heading, reference.state.heading, `${frameRate} Hz heading`);
    assertNear(candidate.accumulatorSeconds, 0, `${frameRate} Hz accumulator`);
    if (candidate.totalSteps !== reference.totalSteps || candidate.totalSteps !== 240) {
      throw new Error(`Expected 240 identical fixed steps at ${frameRate} Hz, got ${candidate.totalSteps}.`);
    }
  }

  let boundedSteps = 0;
  const catchUp = consumeFlight2DFixedSteps(0, 10, () => { boundedSteps += 1; });
  if (boundedSteps !== FLIGHT_2D_MAX_CATCH_UP_STEPS || catchUp.steps !== FLIGHT_2D_MAX_CATCH_UP_STEPS) {
    throw new Error('Long-frame catch-up exceeded or failed to reach its explicit step budget.');
  }

  console.log(JSON.stringify({
    status: 'ok',
    tests: 5,
    contract: 'cardinal bearings, swept thin-wall collision, facade sliding, 30/60/144 Hz equivalence, bounded catch-up',
  }));
} finally {
  await vite.close();
}
