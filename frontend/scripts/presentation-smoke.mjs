import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import * as THREE from 'three';

const vite = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  appType: 'custom',
  server: { middlewareMode: true, hmr: false },
});

function rectangle(id, minX, maxX, minY, maxY, height) {
  return {
    building_id: id,
    height,
    height_source: 'smoke_fixture',
    footprint: [
      [minX, minY],
      [maxX, minY],
      [maxX, maxY],
      [minX, maxY],
    ],
  };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function disposeMeshes(meshes) {
  meshes.forEach((mesh) => {
    mesh.geometry.dispose();
    if (Array.isArray(mesh.material)) mesh.material.forEach((material) => material.dispose());
    else mesh.material.dispose();
  });
}

try {
  const {
    URBAN_DRESSING_LIMITS,
    URBAN_DRESSING_MECHANICS_CONTRACT,
    createUrbanDressingLayout,
    isDressingGroundPointSafe,
    isDressingPointInsideFootprint,
  } = await vite.ssrLoadModule('/src/presentation/urbanDressing.ts');
  const { buildBuildingCollisionMeshes } = await vite.ssrLoadModule('/src/geometry/buildingGeometry.ts');
  const { buildFlight3DObstacles, isFlight3DPositionBlocked } = await vite.ssrLoadModule('/src/simulation/flight3dMotion.ts');
  const { createLidarLocalDirections, scanLidar } = await vite.ssrLoadModule('/src/sensors/lidar.ts');
  const { sampleFlowField2D } = await vite.ssrLoadModule('/src/utils/flowFieldSampling.ts');

  const bounds = { min_x: -120, max_x: 120, min_y: -120, max_y: 120 };
  const buildings = [
    rectangle('osm-a', -35, -8, -28, 24, 28),
    rectangle('osm-b', 12, 42, -36, -4, 46),
    rectangle('osm-c', 18, 58, 24, 62, 19),
    rectangle('osm-d', -72, -42, 38, 74, 35),
    rectangle('osm-e', 62, 88, -90, -54, 24),
    rectangle('osm-f', -96, -68, -88, -55, 16),
  ];
  const zones = [
    { x: -100, y: -100, radius: 17, kind: 'start' },
    { x: 100, y: 100, radius: 19, kind: 'goal' },
  ];
  const options = {
    bounds,
    buildings,
    identity: { scenarioId: 'urbanflow-live-v1-smoke', lat: 37.45, lon: 126.65 },
    exclusionZones: zones,
  };
  const sourceSnapshot = JSON.stringify({ bounds, buildings });
  const first = createUrbanDressingLayout(options);
  const second = createUrbanDressingLayout(options);
  if (JSON.stringify(first) !== JSON.stringify(second)) {
    throw new Error('Identical scenario/building/location identity did not reproduce the same dressing layout.');
  }
  const changed = createUrbanDressingLayout({
    ...options,
    identity: { ...options.identity, scenarioId: 'urbanflow-live-v1-different' },
  });
  if (changed.seed === first.seed || JSON.stringify(changed.trees) === JSON.stringify(first.trees)) {
    throw new Error('A different scenario identity did not produce a different bounded presentation layout.');
  }
  if (JSON.stringify({ bounds, buildings }) !== sourceSnapshot) {
    throw new Error('Presentation generation mutated the live-world source object.');
  }

  const collections = [
    ['trees', first.trees, URBAN_DRESSING_LIMITS.trees],
    ['streetlights', first.streetlights, URBAN_DRESSING_LIMITS.streetlights],
    ['rooftopUnits', first.rooftopUnits, URBAN_DRESSING_LIMITS.rooftopUnits],
    ['roadBands', first.roadBands, URBAN_DRESSING_LIMITS.roadBands],
    ['roadMarkings', first.roadMarkings, URBAN_DRESSING_LIMITS.roadMarkings],
    ['facadePanels', first.facadePanels, URBAN_DRESSING_LIMITS.facadePanels],
  ];
  collections.forEach(([label, items, limit]) => {
    if (items.length > limit) throw new Error(`${label} exceeded its cap: ${items.length} > ${limit}.`);
  });
  if (first.trees.length < 8 || first.streetlights.length < 4 || first.roadMarkings.length < 8) {
    throw new Error(`Fixture dressing was unexpectedly empty: ${JSON.stringify(collections.map(([label, items]) => [label, items.length]))}.`);
  }

  first.trees.forEach((tree) => {
    if (!isDressingGroundPointSafe(tree, bounds, buildings, zones, 4.8, 7)) {
      throw new Error(`Unsafe tree placement at ${JSON.stringify(tree)}.`);
    }
  });
  first.streetlights.forEach((light) => {
    if (!isDressingGroundPointSafe(light, bounds, buildings, zones, 2.4, 4)) {
      throw new Error(`Unsafe streetlight placement at ${JSON.stringify(light)}.`);
    }
  });
  first.roadMarkings.forEach((marking) => {
    if (!isDressingGroundPointSafe(marking, bounds, buildings, zones, 1.3, 2)) {
      throw new Error(`Unsafe road-marking placement at ${JSON.stringify(marking)}.`);
    }
  });
  [...first.trees, ...first.streetlights, ...first.roadMarkings].forEach((prop) => {
    zones.forEach((zone) => {
      if (distance(prop, zone) < zone.radius - 1e-9) {
        throw new Error(`Ground prop entered the ${zone.kind} exclusion zone.`);
      }
    });
  });
  const buildingsById = new Map(buildings.map((building) => [building.building_id, building]));
  first.rooftopUnits.forEach((unit) => {
    const building = buildingsById.get(unit.buildingIdentity);
    if (!building || !isDressingPointInsideFootprint(unit, building.footprint) || unit.roofHeight !== building.height) {
      throw new Error(`Rooftop unit lost its source-building roof contract: ${JSON.stringify(unit)}.`);
    }
  });

  if (Object.values(URBAN_DRESSING_MECHANICS_CONTRACT).some(Boolean)) {
    throw new Error('Presentation mechanics contract exposed a physical/policy-visible channel.');
  }
  const flow = {
    buildings,
    weather: { wind_speed: 3, wind_deg: 270, description: 'fixture' },
    inlet: { ux: 3, uy: 0, speed_mps: 3 },
    domain: { geometry_radius_m: 120, solve_radius_m: 120 },
    field: {
      nx: 2,
      ny: 2,
      cell_size_m: 240,
      bounds,
      ux: [2, 2, 4, 4],
      uy: [1, 1, -1, -1],
      mask: [0, 0, 0, 0],
      stats: { mean_speed_mps: 3, max_speed_mps: 4.2, blocked_fraction: 0 },
    },
  };
  const windBefore = sampleFlowField2D(flow, -84, 84).toArray();
  createUrbanDressingLayout(options);
  const windAfter = sampleFlowField2D(flow, -84, 84).toArray();
  if (JSON.stringify(windBefore) !== JSON.stringify(windAfter)) {
    throw new Error('Dressing generation altered live flow sampling.');
  }

  const pose = { position: new THREE.Vector3(-50, 12, 0), yaw: 0, pitch: 0, roll: 0 };
  const lidarConfig = { maxRange: 80, sampleCount: 32 };
  const directions = createLidarLocalDirections(lidarConfig);
  const meshesBefore = buildBuildingCollisionMeshes(buildings);
  const scanBefore = scanLidar(pose, meshesBefore, lidarConfig, directions);
  disposeMeshes(meshesBefore);
  createUrbanDressingLayout(options);
  const meshesAfter = buildBuildingCollisionMeshes(buildings);
  const scanAfter = scanLidar(pose, meshesAfter, lidarConfig, directions);
  disposeMeshes(meshesAfter);
  if (JSON.stringify(scanBefore.observation) !== JSON.stringify(scanAfter.observation)) {
    throw new Error('Presentation props leaked into the physical building/ground LiDAR observation.');
  }

  const obstacles = buildFlight3DObstacles(buildings);
  const collisionBefore = isFlight3DPositionBlocked({ x: -7, y: 10, z: 0 }, obstacles, 1.25);
  createUrbanDressingLayout(options);
  const collisionAfter = isFlight3DPositionBlocked({ x: -7, y: 10, z: 0 }, obstacles, 1.25);
  if (collisionBefore !== collisionAfter || obstacles.length !== buildings.length) {
    throw new Error('Presentation generation altered the building-only flight collision contract.');
  }

  console.log(JSON.stringify({
    status: 'ok',
    tests: 9,
    counts: Object.fromEntries(collections.map(([label, items]) => [label, items.length])),
    contract: 'deterministic/capped/bounded placement, start-goal/building exclusion, roof identity, no source/flow/collision/LiDAR influence',
  }));
} finally {
  await vite.close();
}
