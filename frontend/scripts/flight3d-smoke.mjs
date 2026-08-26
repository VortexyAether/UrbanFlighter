import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const vite = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  appType: 'custom',
  server: { middlewareMode: true, hmr: false },
});

function assertNear(actual, expected, label, tolerance = 1e-8) {
  if (!Number.isFinite(actual) || Math.abs(actual - expected) > tolerance) {
    throw new Error(`${label}: expected ${expected}, got ${actual}.`);
  }
}

try {
  const {
    mapFlight3DControls,
    toggleFlightCameraMode,
  } = await vite.ssrLoadModule('/src/simulation/flight3dControls.ts');
  const {
    FLIGHT_3D_MAX_CATCH_UP_STEPS,
    buildFlight3DObstacles,
    consumeFlight3DFixedSteps,
    flight3DSegmentHitsObstacle,
    isFlight3DPositionBlocked,
    selectFlight3DSpawn,
    stepFlight3DMotion,
  } = await vite.ssrLoadModule('/src/simulation/flight3dMotion.ts');
  const {
    DRONE_SCALE_CONTRACT,
    resolveDroneSafetyRadius,
  } = await vite.ssrLoadModule('/src/simulation/droneScale.ts');

  const keys = new Set(['KeyW', 'KeyD', 'KeyE', 'Space', 'KeyR']);
  const arcade = mapFlight3DControls(keys, 'arcade');
  const pilot = mapFlight3DControls(keys, 'pilot');
  if (arcade.forward !== 1 || arcade.strafe !== 1 || arcade.yaw !== 1 || arcade.lift !== 1 || !arcade.boost) {
    throw new Error(`Unexpected Arcade mapping: ${JSON.stringify(arcade)}.`);
  }
  if (pilot.forward !== 1 || pilot.strafe !== 1 || pilot.yaw !== -1 || pilot.lift !== 1 || !pilot.boost) {
    throw new Error(`Unexpected Pilot mapping: ${JSON.stringify(pilot)}.`);
  }
  if (toggleFlightCameraMode('chase') !== 'orbit' || toggleFlightCameraMode('orbit') !== 'chase') {
    throw new Error('Chase/orbit camera mapping is not reversible.');
  }

  if (
    DRONE_SCALE_CONTRACT.visualBody.spanM < 0.45
    || DRONE_SCALE_CONTRACT.visualBody.spanM > 0.7
    || DRONE_SCALE_CONTRACT.visualBody.heightM >= DRONE_SCALE_CONTRACT.visualBody.spanM
  ) throw new Error(`Implausible compact quad visual scale: ${JSON.stringify(DRONE_SCALE_CONTRACT.visualBody)}.`);
  const safetyRadius = resolveDroneSafetyRadius(1.25);
  if (safetyRadius !== 1.25 || safetyRadius <= DRONE_SCALE_CONTRACT.visualBody.spanM / 2) {
    throw new Error('The research safety envelope was not explicit and larger than the visual body radius.');
  }
  if (DRONE_SCALE_CONTRACT.researchSafetyEnvelope.verticalClearanceM !== 2) {
    throw new Error('The original 2m browser roof-clearance safety semantics were not retained.');
  }
  if (resolveDroneSafetyRadius(Number.NaN) !== DRONE_SCALE_CONTRACT.researchSafetyEnvelope.fallbackRadiusM) {
    throw new Error('Invalid live safety radius did not resolve to the labeled fallback.');
  }

  const building = {
    building_id: 'wall',
    height: 30,
    footprint: [[0, -10], [0.2, -10], [0.2, 10], [0, 10]],
  };
  const obstacles = buildFlight3DObstacles([building]);
  if (!flight3DSegmentHitsObstacle(
    { x: -5, y: 10, z: 0 },
    { x: 5, y: 10, z: 0 },
    obstacles,
    safetyRadius,
  )) throw new Error('Swept 3D safety envelope missed a thin OSM wall.');
  if (!isFlight3DPositionBlocked({ x: -1, y: 10, z: 0 }, obstacles, safetyRadius)) {
    throw new Error('Research envelope did not block a body-clear but safety-unsafe position.');
  }
  if (isFlight3DPositionBlocked({ x: -1, y: 33, z: 0 }, obstacles, safetyRadius)) {
    throw new Error('Safety envelope remained blocked above the building roof.');
  }
  if (!isFlight3DPositionBlocked({ x: -1, y: 31.8, z: 0 }, obstacles, safetyRadius, 2)) {
    throw new Error('The retained 2m vertical research clearance did not block a roof-unsafe position.');
  }

  const bounds = { min_x: -200, max_x: 200, min_y: -200, max_y: 200 };
  const zeroCommand = { forward: 0, strafe: 0, yaw: 0, lift: 0, boost: false, brake: false };
  const crossing = stepFlight3DMotion(
    { position: { x: -3, y: 10, z: 0 }, velocity: { x: 60, y: 0, z: 0 }, yaw: 0 },
    zeroCommand,
    { x: 0, y: 0, z: 0 },
    obstacles,
    bounds,
    { safetyRadiusM: safetyRadius, maxHorizontalSpeedMps: 100, boostedHorizontalSpeedMps: 100, quadraticAirDragPerM: 0, linearAirDragPerS: 0 },
    0.2,
  );
  if (crossing.position.x >= -1.25 || Math.abs(crossing.velocity.x) > 1e-12) {
    throw new Error(`Swept collision failed to stop unsafe wall crossing: ${JSON.stringify(crossing)}.`);
  }

  function simulate(frameRate) {
    let state = {
      position: { x: -80, y: 30, z: 70 },
      velocity: { x: 0, y: 0, z: 0 },
      yaw: 0,
    };
    let accumulatorSeconds = 0;
    let totalSteps = 0;
    const command = { forward: 1, strafe: 0.35, yaw: -0.25, lift: 0.1, boost: false, brake: false };
    for (let frame = 0; frame < frameRate * 2; frame += 1) {
      const result = consumeFlight3DFixedSteps(accumulatorSeconds, 1 / frameRate, (fixedDelta) => {
        state = stepFlight3DMotion(
          state,
          command,
          { x: 3.2, y: 0, z: -1.4 },
          [],
          bounds,
          { safetyRadiusM: safetyRadius },
          fixedDelta,
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
    for (const axis of ['x', 'y', 'z']) {
      assertNear(candidate.state.position[axis], reference.state.position[axis], `${frameRate} Hz position ${axis}`);
      assertNear(candidate.state.velocity[axis], reference.state.velocity[axis], `${frameRate} Hz velocity ${axis}`);
    }
    assertNear(candidate.state.yaw, reference.state.yaw, `${frameRate} Hz yaw`);
    assertNear(candidate.accumulatorSeconds, 0, `${frameRate} Hz accumulator`);
    if (candidate.totalSteps !== 240 || candidate.totalSteps !== reference.totalSteps) {
      throw new Error(`Expected 240 fixed 3D steps at ${frameRate} Hz, got ${candidate.totalSteps}.`);
    }
  }

  let boundedSteps = 0;
  const catchUp = consumeFlight3DFixedSteps(0, 10, () => { boundedSteps += 1; });
  if (boundedSteps !== FLIGHT_3D_MAX_CATCH_UP_STEPS || catchUp.steps !== FLIGHT_3D_MAX_CATCH_UP_STEPS) {
    throw new Error('3D long-frame catch-up exceeded its explicit step budget.');
  }

  const tallCenter = buildFlight3DObstacles([{
    building_id: 'center-tower',
    height: 86,
    footprint: [[-8, -8], [8, -8], [8, 8], [-8, 8]],
  }]);
  const spawn = selectFlight3DSpawn(bounds, tallCenter, safetyRadius);
  const stayedAtCenter = Math.hypot(spawn.x, spawn.z) < 1e-6;
  if (
    isFlight3DPositionBlocked(spawn, tallCenter, safetyRadius)
    || (stayedAtCenter && spawn.y < 86 + safetyRadius)
  ) {
    throw new Error(`Spawn did not clear the OSM roof and safety envelope: ${JSON.stringify(spawn)}.`);
  }

  const { integrateQuadraticAirDrag3 } = await vite.ssrLoadModule('/src/simulation/quadraticAirDrag.ts');
  let drifted = { x: 0, y: 0, z: 0 };
  for (let step = 0; step < 720; step += 1) {
    drifted = integrateQuadraticAirDrag3(drifted, { x: 5, y: 0, z: -2 }, 1 / 120);
  }
  if (Math.hypot(drifted.x - 5, drifted.z + 2) > 1.2) {
    throw new Error(`3D zero-thrust hover must drift toward local wind: ${JSON.stringify(drifted)}.`);
  }

  console.log(JSON.stringify({
    status: 'ok',
    tests: 9,
    contract: 'selectable controls/camera, 0.58m body vs 1.25m horizontal/2m vertical research envelope, swept OSM collision, fixed-step equivalence, bounded catch-up, roof-safe spawn, quadratic wind-follow',
  }));
} finally {
  await vite.close();
}
