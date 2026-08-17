import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const frontendRoot = fileURLToPath(new URL('..', import.meta.url));
const vite = await createServer({
  root: frontendRoot,
  appType: 'custom',
  server: { middlewareMode: true, hmr: false },
});

function assertTriplet(actual, expected, label) {
  if (
    !Array.isArray(actual)
    || actual.length !== 3
    || actual.some((value, index) => !Number.isFinite(value) || Math.abs(value - expected[index]) > 1e-12)
  ) {
    throw new Error(`${label}: expected [${expected.join(', ')}], got ${JSON.stringify(actual)}.`);
  }
}

try {
  const {
    isFiniteTrue3DTriplet,
    mapTrue3DPointToScene,
    mapTrue3DVectorToScene,
    mapTrue3DVelocityToScene,
  } = await vite.ssrLoadModule('/src/geometry/true3dCoordinates.ts');

  assertTriplet(mapTrue3DPointToScene([12, 34, 56]), [12, 56, -34], 'point mapping');
  assertTriplet(mapTrue3DVectorToScene([-7, 8, -9]), [-7, -9, -8], 'vector mapping');
  assertTriplet(mapTrue3DVelocityToScene([4, -5, 6]), [4, 6, 5], 'velocity [u,v,w] mapping');

  const invalidTriplets = [null, [1, 2], [1, 2, 3, 4], [1, 2, Number.POSITIVE_INFINITY], ['1', 2, 3]];
  if (!isFiniteTrue3DTriplet([1, 2, 3]) || invalidTriplets.some(isFiniteTrue3DTriplet)) {
    throw new Error('Finite-triplet guard accepted malformed or non-finite source coordinates.');
  }

  const datasetUrl = new URL('../public/data/true3d_potential_gangnam/true3d_streamlines.json', import.meta.url);
  const dataset = JSON.parse(await readFile(datasetUrl, 'utf8'));
  if (!Array.isArray(dataset.streamlines) || dataset.streamlines.length === 0) {
    throw new Error('Bundled True 3D dataset has no streamlines.');
  }

  const sourceMin = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const sourceMax = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  const sceneMin = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const sceneMax = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  let mappedPoints = 0;

  for (const line of dataset.streamlines) {
    if (!Array.isArray(line)) throw new Error('Bundled True 3D dataset contains a malformed streamline.');
    for (const point of line) {
      if (!isFiniteTrue3DTriplet(point)) throw new Error('Bundled True 3D dataset contains a malformed or non-finite point.');
      const mapped = mapTrue3DPointToScene(point);
      if (!mapped.every(Number.isFinite)) throw new Error('Mapped True 3D dataset contains a non-finite scene point.');
      for (let axis = 0; axis < 3; axis += 1) {
        sourceMin[axis] = Math.min(sourceMin[axis], point[axis]);
        sourceMax[axis] = Math.max(sourceMax[axis], point[axis]);
        sceneMin[axis] = Math.min(sceneMin[axis], mapped[axis]);
        sceneMax[axis] = Math.max(sceneMax[axis], mapped[axis]);
      }
      mappedPoints += 1;
    }
  }

  const boundsMatch = Math.abs(sceneMin[0] - sourceMin[0]) < 1e-12
    && Math.abs(sceneMax[0] - sourceMax[0]) < 1e-12
    && Math.abs(sceneMin[1] - sourceMin[2]) < 1e-12
    && Math.abs(sceneMax[1] - sourceMax[2]) < 1e-12
    && Math.abs(sceneMin[2] + sourceMax[1]) < 1e-12
    && Math.abs(sceneMax[2] + sourceMin[1]) < 1e-12;
  if (mappedPoints < 100 || !boundsMatch) {
    throw new Error(`Bundled dataset bounds do not follow source [x,y,z] -> scene [x,z,-y] across ${mappedPoints} points.`);
  }

  const inletVelocity = [
    dataset.meta?.inlet_ux_mps,
    dataset.meta?.inlet_uy_mps,
    dataset.meta?.inlet_uz_mps,
  ];
  if (!isFiniteTrue3DTriplet(inletVelocity)) throw new Error('Bundled True 3D dataset is missing finite inlet [u,v,w] components.');
  const sceneInlet = mapTrue3DVelocityToScene(inletVelocity);
  if (!sceneInlet.every(Number.isFinite)) throw new Error('Bundled inlet velocity maps to non-finite scene components.');

  console.log(
    `True 3D smoke passed: ${dataset.streamlines.length} streamlines / ${mappedPoints} finite points, `
    + `source +Y -> scene -Z, inlet [${inletVelocity.join(', ')}] -> [${sceneInlet.join(', ')}].`,
  );
} finally {
  await vite.close();
}
