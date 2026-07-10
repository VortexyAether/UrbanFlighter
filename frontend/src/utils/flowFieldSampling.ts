import * as THREE from 'three';
import type { FlowField2DResponse } from '../api';

export function hasResolvedFlowGrid(flow: FlowField2DResponse | null): flow is FlowField2DResponse {
  if (!flow) return false;
  const expected = flow.field.nx * flow.field.ny;
  return flow.field.nx > 1 && flow.field.ny > 1 && flow.field.ux.length === expected && flow.field.uy.length === expected;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

export function sampleFlowField2D(flow: FlowField2DResponse | null, x: number, z: number): THREE.Vector3 | null {
  if (!hasResolvedFlowGrid(flow)) return null;

  const { field } = flow;
  const { bounds } = field;
  // CityModel extrudes XY footprints then rotateX(-90), so source +Y becomes scene -Z.
  // Convert scene Z back to field Y for sampling, and convert sampled field VY to scene -Z.
  const fieldY = -z;
  if (x < bounds.min_x || x > bounds.max_x || fieldY < bounds.min_y || fieldY > bounds.max_y) {
    return new THREE.Vector3(flow.inlet.ux, 0, -flow.inlet.uy);
  }

  const gx = ((x - bounds.min_x) / Math.max(1e-6, bounds.max_x - bounds.min_x)) * (field.nx - 1);
  const gy = ((fieldY - bounds.min_y) / Math.max(1e-6, bounds.max_y - bounds.min_y)) * (field.ny - 1);
  const x0 = clamp(Math.floor(gx), 0, field.nx - 1);
  const y0 = clamp(Math.floor(gy), 0, field.ny - 1);
  const x1 = clamp(x0 + 1, 0, field.nx - 1);
  const y1 = clamp(y0 + 1, 0, field.ny - 1);
  const tx = gx - x0;
  const ty = gy - y0;
  const idx = (ix: number, iy: number) => ix * field.ny + iy;

  const nearest = idx(Math.round(gx), Math.round(gy));
  if ((field.mask[nearest] ?? 0) > 0) return new THREE.Vector3(0, 0, 0);

  const bilerp = (values: number[]) => {
    const c00 = values[idx(x0, y0)] ?? 0;
    const c10 = values[idx(x1, y0)] ?? c00;
    const c01 = values[idx(x0, y1)] ?? c00;
    const c11 = values[idx(x1, y1)] ?? c10;
    const c0 = c00 * (1 - tx) + c10 * tx;
    const c1 = c01 * (1 - tx) + c11 * tx;
    return c0 * (1 - ty) + c1 * ty;
  };

  return new THREE.Vector3(bilerp(field.ux), 0, -bilerp(field.uy));
}

export function flowSpeedAtIndex(flow: FlowField2DResponse, ix: number, iy: number) {
  const idx = ix * flow.field.ny + iy;
  return Math.hypot(flow.field.ux[idx] ?? 0, flow.field.uy[idx] ?? 0);
}
