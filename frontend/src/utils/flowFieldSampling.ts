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

  const sampledUx = sampleMaskedBilinear(field.ux, field.mask, idx, x0, y0, x1, y1, tx, ty);
  const sampledUy = sampleMaskedBilinear(field.uy, field.mask, idx, x0, y0, x1, y1, tx, ty);
  if (sampledUx === null || sampledUy === null) return new THREE.Vector3(0, 0, 0);

  return new THREE.Vector3(sampledUx, 0, -sampledUy);
}

export function isSolidMaskCell(mask: number[] | undefined, index: number): boolean {
  return (mask?.[index] ?? 0) > 0;
}

export function sampleMaskedBilinear(
  values: number[],
  mask: number[] | undefined,
  idx: (ix: number, iy: number) => number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  tx: number,
  ty: number,
): number | null {
  const corners = [
    { w: (1 - tx) * (1 - ty), index: idx(x0, y0) },
    { w: tx * (1 - ty), index: idx(x1, y0) },
    { w: (1 - tx) * ty, index: idx(x0, y1) },
    { w: tx * ty, index: idx(x1, y1) },
  ];
  let weight = 0;
  let value = 0;
  corners.forEach((corner) => {
    if (isSolidMaskCell(mask, corner.index) || corner.w <= 0) return;
    weight += corner.w;
    value += (values[corner.index] ?? 0) * corner.w;
  });
  if (weight <= 1e-6) return null;
  return value / weight;
}

export function fieldCellIndex(flow: FlowField2DResponse, x: number, fieldY: number): number | null {
  const { field } = flow;
  const { bounds } = field;
  if (x < bounds.min_x || x > bounds.max_x || fieldY < bounds.min_y || fieldY > bounds.max_y) return null;
  const ix = clamp(Math.round(((x - bounds.min_x) / Math.max(1e-6, bounds.max_x - bounds.min_x)) * (field.nx - 1)), 0, field.nx - 1);
  const iy = clamp(Math.round(((fieldY - bounds.min_y) / Math.max(1e-6, bounds.max_y - bounds.min_y)) * (field.ny - 1)), 0, field.ny - 1);
  return ix * field.ny + iy;
}

export function maskBlocksSegment(flow: FlowField2DResponse, x0: number, fieldY0: number, x1: number, fieldY1: number): boolean {
  const span = Math.hypot(x1 - x0, fieldY1 - fieldY0);
  const steps = Math.max(2, Math.ceil(span / Math.max(flow.field.cell_size_m * 0.45, 0.8)));
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps;
    const index = fieldCellIndex(flow, x0 + (x1 - x0) * t, fieldY0 + (fieldY1 - fieldY0) * t);
    if (index === null) continue;
    if (isSolidMaskCell(flow.field.mask, index)) return true;
  }
  return false;
}

export function flowSpeedAtIndex(flow: FlowField2DResponse, ix: number, iy: number) {
  const idx = ix * flow.field.ny + iy;
  return Math.hypot(flow.field.ux[idx] ?? 0, flow.field.uy[idx] ?? 0);
}
