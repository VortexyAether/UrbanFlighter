import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const vite = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  appType: 'custom',
  server: { middlewareMode: true, hmr: false },
});

const scenarioId = 'urbanflow-live-v1-1234567890abcdef12345678';
const sessionId = 'ufi_deterministic_session';
const world = {
  schema_id: 'urbanflow.episode_inspector_world.v1',
  scenario_id: scenarioId,
  content_hash_sha256: 'a'.repeat(64),
  coordinate_frame: {
    horizontal_units: 'm',
    x_axis: 'east',
    y_axis: 'north',
    display_orientation: 'north_up',
  },
  bounds: { min_x_m: -100, max_x_m: 100, min_y_m: -100, max_y_m: 100 },
  start_goal_source: 'deterministic_safe_route_in_registered_geometry',
  structure_count: 1,
  buildings: [{
    building_id: 'osm-a',
    height_m: 24,
    height_source: 'osm:height',
    footprint_xy_m: [[-20, -10], [10, -10], [10, 20], [-20, 20]],
  }],
  known_inlet: {
    velocity_xy_mps: [3, 0],
    speed_mps: 3,
    direction_from_north_deg: 270,
    timestamp: null,
    source: {},
    fallback: { used: false },
  },
  source: 'exact_registered_live_osm_scenario',
  synthetic_fixture: false,
};
const frame = {
  schema_id: 'urbanflow.episode_inspector_frame.v1',
  scenario_id: scenarioId,
  seed: 10007,
  baseline: { baseline_id: 'shortest_path', label: 'Shortest path', uses_full_flow: false },
  world_bounds: world.bounds,
  drone: { position_xy_m: [0, 0], heading_rad: Math.PI / 2, ground_velocity_xy_mps: [0, 1] },
  start_xy_m: [-80, -80],
  goal_xy_m: [80, 80],
  trajectory_xy_m: [[-80, -80], [0, 0]],
  actor_lidar: {
    ray_count: 2,
    max_range_m: 35,
    frame: 'vehicle_local_counter_clockwise_from_forward',
    rays: [
      { local_angle_rad: 0, distance_m: 20, endpoint_xy_m: [0, 20], hit: true },
      { local_angle_rad: Math.PI, distance_m: 35, endpoint_xy_m: [0, -35], hit: false },
    ],
  },
  local_guidance_action: {
    schema_id: 'urbanflow.local_guidance_action.v1',
    frame: 'vehicle_local_forward_left',
    vector: [1, 0],
    forward: 1,
    left: 0,
    phase: 'executed',
    source: 'deterministic_baseline',
  },
  actor_observation: { schema_id: 'urbanflow.actor_observation.v1', vector: [], fields: [] },
  air_relative_velocity_xy_mps: [-3, 1],
  reward: { schema_id: 'urbanflow.reward_terms.v1', components: {}, step_total: 1, episode_total: 2 },
  clearance_m: 4,
  collision: false,
  terminated: false,
  truncated: false,
  status: 'running',
  termination_reason: null,
  step_index: 2,
  max_steps: 1200,
  dt_s: 0.25,
  simulated_elapsed_s: 0.5,
  simulated_max_s: 300,
  distance_to_goal_m: Math.hypot(80, 80),
  estimated_minimum_steps: 51,
  estimated_minimum_time_s: 12.58,
  flags: {
    policy_status: 'not_trained',
    policy_had_privileged_flow_access: false,
    full_flow_access: false,
    training_executed: false,
    browser_motor_training: false,
    navier_stokes_cfd: false,
    real_cfd_validation_run: false,
    real_cfd_adapter_status: 'interface_only_not_executed',
    synthetic_fixture: false,
  },
};
const responsePayload = {
  session_id: sessionId,
  session_active: true,
  scenario_id: scenarioId,
  seed: 10007,
  baseline_id: 'shortest_path',
  limits: {
    max_steps: 1200,
    max_sessions: 12,
    ttl_s: 900,
    reset_count: 0,
    dt_s: 0.25,
    simulated_max_s: 300,
    max_batch_steps: 64,
  },
  world,
  frame,
  requested_steps: 16,
  executed_steps: 16,
  batch_reward: 16,
};
const calls = [];

globalThis.fetch = async (url, options = {}) => {
  calls.push({ url: String(url), options });
  const isDelete = options.method === 'DELETE';
  return new Response(JSON.stringify(
    isDelete ? { status: 'deleted', session_id: sessionId } : responsePayload,
  ), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
};

try {
  const {
    createUrbanFlowInspectorSession,
    deleteUrbanFlowInspectorSession,
    resetUrbanFlowInspectorSession,
    stepUrbanFlowInspectorSession,
  } = await vite.ssrLoadModule('/src/api.ts');
  const {
    buildEpisodeInspectorRenderModel,
    createInspectorViewport,
    inspectorResponseMatchesSelection,
    inspectorSessionIsStale,
    localActionToWorldVector,
    worldToInspectorPoint,
  } = await vite.ssrLoadModule('/src/data/episodeInspectorModel.ts');
  const {
    INSPECTOR_MAX_BATCH_STEPS,
    INSPECTOR_PLAY_RATES,
    inspectorStepResponseIsCurrent,
    scheduleInspectorPlaybackTick,
  } = await vite.ssrLoadModule('/src/data/episodeInspectorScheduler.ts');

  await createUrbanFlowInspectorSession(scenarioId, 10007, 'shortest_path');
  await resetUrbanFlowInspectorSession(sessionId);
  await stepUrbanFlowInspectorSession(sessionId, { repeat: 16 });
  await stepUrbanFlowInspectorSession(sessionId);
  await deleteUrbanFlowInspectorSession(sessionId);
  const createBody = JSON.parse(calls[0].options.body);
  const stepBody = JSON.parse(calls[2].options.body);
  const manualStepBody = JSON.parse(calls[3].options.body);
  if (
    !calls[0].url.endsWith('/urbanflow-gym/inspector/sessions')
    || calls[0].options.method !== 'POST'
    || createBody.scenario_id !== scenarioId
    || createBody.seed !== 10007
    || createBody.baseline !== 'shortest_path'
    || createBody.max_steps !== 1200
    || !calls[1].url.endsWith(`/inspector/sessions/${sessionId}/reset`)
    || !calls[2].url.endsWith(`/inspector/sessions/${sessionId}/step`)
    || JSON.stringify(stepBody) !== JSON.stringify({ repeat: 16 })
    || JSON.stringify(manualStepBody) !== JSON.stringify({ repeat: 1 })
    || calls[4].options.method !== 'DELETE'
  ) {
    throw new Error('Inspector API client did not preserve scenario/session binding and bounded replay controls.');
  }

  const callsBeforeInvalidRepeat = calls.length;
  for (const repeat of [0, 65, true]) {
    let rejected = false;
    try {
      await stepUrbanFlowInspectorSession(sessionId, { repeat });
    } catch (reason) {
      rejected = reason instanceof Error && reason.message.includes('integer from 1 through 64');
    }
    if (!rejected) throw new Error(`Inspector API client accepted invalid repeat ${String(repeat)}.`);
  }
  if (calls.length !== callsBeforeInvalidRepeat) {
    throw new Error('Invalid inspector repeats reached fetch instead of being rejected client-side.');
  }

  const expectedFourTickSchedules = new Map([
    [1, [0, 0, 0, 1]],
    [2, [0, 1, 0, 1]],
    [4, [1, 1, 1, 1]],
    [8, [2, 2, 2, 2]],
    [16, [4, 4, 4, 4]],
    [32, [8, 8, 8, 8]],
    [64, [16, 16, 16, 16]],
  ]);
  for (const rate of INSPECTOR_PLAY_RATES) {
    let creditQuanta = 0;
    const repeats = [];
    for (let tickIndex = 0; tickIndex < 4; tickIndex += 1) {
      const tick = scheduleInspectorPlaybackTick(rate, creditQuanta, false);
      creditQuanta = tick.creditQuanta;
      repeats.push(tick.repeat);
    }
    if (
      JSON.stringify(repeats) !== JSON.stringify(expectedFourTickSchedules.get(rate))
      || repeats.reduce((total, repeat) => total + repeat, 0) !== rate
      || repeats.some((repeat) => repeat > INSPECTOR_MAX_BATCH_STEPS)
    ) {
      throw new Error(`Inspector ${rate}x accumulator did not schedule exactly ${rate} bounded steps/s.`);
    }
  }

  let blockedCreditQuanta = 0;
  for (let tickIndex = 0; tickIndex < 8; tickIndex += 1) {
    const tick = scheduleInspectorPlaybackTick(64, blockedCreditQuanta, true);
    blockedCreditQuanta = tick.creditQuanta;
    if (tick.repeat !== 0) throw new Error('Inspector scheduler issued an overlapping request.');
  }
  const catchUpTick = scheduleInspectorPlaybackTick(64, blockedCreditQuanta, false);
  if (catchUpTick.repeat !== INSPECTOR_MAX_BATCH_STEPS || catchUpTick.creditQuanta !== 0) {
    throw new Error('Inspector scheduler did not bound and drain its slow-request backlog.');
  }
  if (
    !inspectorStepResponseIsCurrent(sessionId, sessionId, 7, 7, true, false)
    || inspectorStepResponseIsCurrent(sessionId, sessionId, 7, 8, true, false)
    || inspectorStepResponseIsCurrent(sessionId, 'replacement-session', 7, 7, true, false)
    || inspectorStepResponseIsCurrent(sessionId, sessionId, 7, 7, false, false)
    || inspectorStepResponseIsCurrent(sessionId, sessionId, 7, 7, true, true)
  ) {
    throw new Error('Inspector request ownership guard accepted a stale or aborted batch response.');
  }

  const viewport = createInspectorViewport(world.bounds, 1000, 620, 26);
  const center = worldToInspectorPoint([0, 0], world.bounds, viewport);
  const north = worldToInspectorPoint([0, 100], world.bounds, viewport);
  const east = worldToInspectorPoint([100, 0], world.bounds, viewport);
  if (
    Math.abs(center.x - 500) > 1e-9
    || Math.abs(center.y - 310) > 1e-9
    || north.y >= center.y
    || east.x <= center.x
  ) {
    throw new Error('North-up world-to-frame transform changed orientation or centering.');
  }

  const forwardNorth = localActionToWorldVector([1, 0], Math.PI / 2);
  const leftFromNorth = localActionToWorldVector([0, 1], Math.PI / 2);
  if (
    Math.abs(forwardNorth[0]) > 1e-9
    || Math.abs(forwardNorth[1] - 1) > 1e-9
    || Math.abs(leftFromNorth[0] + 1) > 1e-9
    || Math.abs(leftFromNorth[1]) > 1e-9
  ) {
    throw new Error('Vehicle-local forward/left action did not transform into the expected east/north vector.');
  }

  const model = buildEpisodeInspectorRenderModel(world, frame, 1000, 620);
  if (
    model.buildings.length !== 1
    || model.buildings[0].points.length !== 4
    || model.trajectory.length !== 2
    || model.lidarRays.length !== 2
    || !model.lidarRays[0].hit
    || model.actionEnd.y >= model.drone.y
  ) {
    throw new Error('Episode render model lost OSM polygons, trajectory, actor rays, or action direction.');
  }

  const expectedIdentity = '37.45144800,126.65154230:scenario:ready';
  if (
    !inspectorResponseMatchesSelection(responsePayload, expectedIdentity, expectedIdentity, scenarioId, sessionId)
    || inspectorResponseMatchesSelection(responsePayload, expectedIdentity, `${expectedIdentity}:stale`, scenarioId, sessionId)
    || inspectorResponseMatchesSelection(responsePayload, expectedIdentity, expectedIdentity, scenarioId, 'wrong-session')
    || inspectorSessionIsStale(scenarioId, scenarioId, expectedIdentity, expectedIdentity)
    || !inspectorSessionIsStale(scenarioId, null, expectedIdentity, expectedIdentity)
    || !inspectorSessionIsStale(scenarioId, `${scenarioId}-changed`, expectedIdentity, expectedIdentity)
    || !inspectorSessionIsStale(scenarioId, scenarioId, expectedIdentity, `${expectedIdentity}:loading`)
  ) {
    throw new Error('Inspector stale-session guards accepted a changed selection or rejected the pinned selection.');
  }

  let mismatchRejected = false;
  try {
    buildEpisodeInspectorRenderModel({ ...world, scenario_id: `${scenarioId}-other` }, frame);
  } catch {
    mismatchRejected = true;
  }
  if (!mismatchRejected) throw new Error('Render model accepted mismatched world/frame scenario identities.');

  console.log(JSON.stringify({
    status: 'ok',
    tests: 8,
    contract: '1200-step/batch API, bounded 1x-64x accumulator, no-overlap scheduling, north-up/action transforms, OSM/LiDAR render model, stale-session guards',
  }));
} finally {
  await vite.close();
}
