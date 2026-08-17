import React, { useEffect, useMemo, useRef } from 'react';
import { Line } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { FlowField2DResponse } from '../api';
import { flowSpeedAtIndex, hasResolvedFlowGrid, sampleFlowField2D } from '../utils/flowFieldSampling';

interface CFDLiteWindLayerProps {
  flow: FlowField2DResponse | null;
  height?: number;
  altitudeLevels?: number[];
  showContour?: boolean;
  showArrows?: boolean;
  showStreamlines?: boolean;
}

interface StreamlinePath {
  points: THREE.Vector3[];
  speed: number;
  offset: number;
  layerIndex: number;
}

function makeColor(speed: number, maxSpeed: number) {
  const t = THREE.MathUtils.clamp(speed / Math.max(maxSpeed, 1), 0, 1);
  return new THREE.Color().setHSL(0.58 - t * 0.50, 0.95, 0.52);
}

function fieldYToSceneZ(fieldY: number) {
  return -fieldY;
}

function sceneZToFieldY(sceneZ: number) {
  return -sceneZ;
}

function samplePolyline(points: THREE.Vector3[], t: number, target: THREE.Vector3) {
  if (points.length === 0) return target.set(0, 0, 0);
  if (points.length === 1) return target.copy(points[0]);
  const wrapped = ((t % 1) + 1) % 1;
  const scaled = wrapped * (points.length - 1);
  const idx = Math.min(points.length - 2, Math.floor(scaled));
  const localT = scaled - idx;
  return target.copy(points[idx]).lerp(points[idx + 1], localT);
}

const FlowDashes: React.FC<{ lines: StreamlinePath[]; maxSpeed: number }> = ({ lines, maxSpeed }) => {
  const refs = useRef<Array<THREE.Line | null>>([]);
  const tmpA = useMemo(() => new THREE.Vector3(), []);
  const tmpB = useMemo(() => new THREE.Vector3(), []);
  const dashes = useMemo(() => {
    const items: { line: StreamlinePath; offset: number; dashLength: number }[] = [];
    lines.forEach((line, lineIdx) => {
      const count = line.points.length > 80 ? 2 : 1;
      for (let i = 0; i < count; i += 1) {
        items.push({
          line,
          offset: (i / count + line.offset + lineIdx * 0.017) % 1,
          dashLength: THREE.MathUtils.clamp(5 / Math.max(line.points.length, 1), 0.018, 0.055),
        });
      }
    });
    return items;
  }, [lines]);

  useFrame(({ clock }) => {
    const elapsed = clock.getElapsedTime();
    dashes.forEach((dash, idx) => {
      const line = refs.current[idx];
      if (!line) return;
      const flowRate = 0.032 + 0.115 * THREE.MathUtils.clamp(dash.line.speed / Math.max(maxSpeed, 1), 0, 1);
      const headT = elapsed * flowRate + dash.offset;
      const tailT = headT - dash.dashLength;
      samplePolyline(dash.line.points, tailT, tmpA);
      samplePolyline(dash.line.points, headT, tmpB);
      const geometry = line.geometry;
      const position = geometry.getAttribute('position');
      if (!(position instanceof THREE.BufferAttribute)) return;
      position.setXYZ(0, tmpA.x, tmpA.y, tmpA.z);
      position.setXYZ(1, tmpB.x, tmpB.y, tmpB.z);
      position.needsUpdate = true;
      geometry.computeBoundingSphere();
    });
  });

  return (
    <group name="white-flow-dash-segments">
      {dashes.map((dash, idx) => (
        <line
          key={`flow-dash-${idx}-${dash.offset.toFixed(3)}`}
          ref={(node: unknown) => {
            refs.current[idx] = node instanceof THREE.Line ? node : null;
          }}
        >
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[new Float32Array(6), 3]} />
          </bufferGeometry>
          <lineBasicMaterial color="#f4feff" transparent opacity={0.96} depthWrite={false} toneMapped={false} />
        </line>
      ))}
    </group>
  );
};

const CFDLiteWindLayer: React.FC<CFDLiteWindLayerProps> = ({
  flow,
  height = 22,
  altitudeLevels,
  showContour = false,
  showArrows = false,
  showStreamlines = true,
}) => {
  const maxSpeed = Math.max(flow?.field.stats.max_speed_mps ?? flow?.inlet.speed_mps ?? 8, 1);
  const levels = useMemo(() => altitudeLevels && altitudeLevels.length > 0 ? altitudeLevels : [height], [altitudeLevels, height]);

  const contourGeometry = useMemo(() => {
    if (!hasResolvedFlowGrid(flow) || !showContour) return null;
    const { field } = flow;
    const vertices: number[] = [];
    const indices: number[] = [];
    const colors = new Float32Array(field.nx * field.ny * 3);

    for (let ix = 0; ix < field.nx; ix += 1) {
      for (let iy = 0; iy < field.ny; iy += 1) {
        const idx = ix * field.ny + iy;
        const x = field.bounds.min_x + (ix / (field.nx - 1)) * (field.bounds.max_x - field.bounds.min_x);
        const fieldY = field.bounds.min_y + (iy / (field.ny - 1)) * (field.bounds.max_y - field.bounds.min_y);
        vertices.push(x, height, fieldYToSceneZ(fieldY));
        const color = (field.mask[idx] ?? 0) > 0 ? new THREE.Color('#101820') : makeColor(flowSpeedAtIndex(flow, ix, iy), maxSpeed);
        colors[idx * 3] = color.r;
        colors[idx * 3 + 1] = color.g;
        colors[idx * 3 + 2] = color.b;
      }
    }

    for (let ix = 0; ix < field.nx - 1; ix += 1) {
      for (let iy = 0; iy < field.ny - 1; iy += 1) {
        const a = ix * field.ny + iy;
        const b = (ix + 1) * field.ny + iy;
        const c = (ix + 1) * field.ny + iy + 1;
        const d = ix * field.ny + iy + 1;
        indices.push(a, b, d, b, c, d);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    return geo;
  }, [flow, height, maxSpeed, showContour]);

  useEffect(() => () => {
    contourGeometry?.dispose();
  }, [contourGeometry]);

  const arrows = useMemo(() => {
    if (!hasResolvedFlowGrid(flow) || !showArrows) return [];
    const { field } = flow;
    const stride = Math.max(4, Math.round(Math.max(field.nx, field.ny) / 28));
    const items: { position: THREE.Vector3; direction: THREE.Vector3; speed: number; color: THREE.Color }[] = [];
    for (let ix = 1; ix < field.nx - 1; ix += stride) {
      for (let iy = 1; iy < field.ny - 1; iy += stride) {
        const idx = ix * field.ny + iy;
        if ((field.mask[idx] ?? 0) > 0) continue;
        const x = field.bounds.min_x + (ix / (field.nx - 1)) * (field.bounds.max_x - field.bounds.min_x);
        const fieldY = field.bounds.min_y + (iy / (field.ny - 1)) * (field.bounds.max_y - field.bounds.min_y);
        const z = fieldYToSceneZ(fieldY);
        const vec = sampleFlowField2D(flow, x, z);
        const speed = vec?.length() ?? 0;
        if (!vec || speed < 0.2) continue;
        items.push({ position: new THREE.Vector3(x, levels[0] + 4, z), direction: vec.clone().normalize(), speed, color: makeColor(speed, maxSpeed) });
      }
    }
    return items;
  }, [flow, levels, maxSpeed, showArrows]);

  const streamlines = useMemo<StreamlinePath[]>(() => {
    if (!hasResolvedFlowGrid(flow)) return [];
    const { field } = flow;
    const lines: StreamlinePath[] = [];
    const seedsPerLayer = 22;
    const direction = new THREE.Vector2(flow.inlet.ux, flow.inlet.uy).normalize();
    const useXEdge = Math.abs(direction.x) >= Math.abs(direction.y);
    const step = Math.max(field.cell_size_m * 1.15, 3.0);

    levels.forEach((layerHeight, layerIndex) => {
      for (let s = 0; s < seedsPerLayer; s += 1) {
        let x = useXEdge
          ? (direction.x >= 0 ? field.bounds.min_x + field.cell_size_m : field.bounds.max_x - field.cell_size_m)
          : field.bounds.min_x + ((s + 0.5) / seedsPerLayer) * (field.bounds.max_x - field.bounds.min_x);
        let fieldY = useXEdge
          ? field.bounds.min_y + ((s + 0.5) / seedsPerLayer) * (field.bounds.max_y - field.bounds.min_y)
          : (direction.y >= 0 ? field.bounds.min_y + field.cell_size_m : field.bounds.max_y - field.cell_size_m);
        const pts: THREE.Vector3[] = [];
        let meanSpeed = 0;

        for (let i = 0; i < 280; i += 1) {
          if (x < field.bounds.min_x || x > field.bounds.max_x || fieldY < field.bounds.min_y || fieldY > field.bounds.max_y) break;
          const z = fieldYToSceneZ(fieldY);
          const vec = sampleFlowField2D(flow, x, z);
          const speed = vec?.length() ?? 0;
          if (!vec || speed < 0.12) break;
          const wave = Math.sin((i * 0.23) + layerIndex * 0.8) * 1.8;
          pts.push(new THREE.Vector3(x, layerHeight + wave, z));
          meanSpeed += speed;
          const dir = vec.clone().normalize();
          x += dir.x * step;
          fieldY = sceneZToFieldY(z + dir.z * step);
        }

        if (pts.length > 8) {
          const avg = meanSpeed / pts.length;
          lines.push({ points: pts, speed: avg, offset: (s / seedsPerLayer + layerIndex * 0.11) % 1, layerIndex });
        }
      }
    });
    return lines;
  }, [flow, levels]);

  if (!hasResolvedFlowGrid(flow)) return null;

  return (
    <group name="CFD-lite B wind streamlines">
      {showContour && contourGeometry && (
        <mesh geometry={contourGeometry}>
          <meshBasicMaterial vertexColors transparent opacity={0.42} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      )}
      {showArrows && arrows.map((arrow, idx) => (
        <mesh key={`arrow-${idx}`} position={arrow.position} quaternion={new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), arrow.direction)}>
          <coneGeometry args={[1.6, THREE.MathUtils.clamp(arrow.speed * 1.2, 3, 10), 8]} />
          <meshBasicMaterial color={arrow.color} transparent opacity={0.82} toneMapped={false} />
        </mesh>
      ))}
      {showStreamlines && streamlines.map((line, idx) => (
        <Line
          key={`stream-${idx}`}
          points={line.points}
          color={`#${makeColor(line.speed, maxSpeed).offsetHSL(0, -0.08, line.layerIndex % 2 === 0 ? 0.08 : -0.02).getHexString()}`}
          lineWidth={1.2}
          transparent
          opacity={0.56}
          depthWrite={false}
        />
      ))}
      {showStreamlines && <FlowDashes lines={streamlines} maxSpeed={maxSpeed} />}
    </group>
  );
};

export default CFDLiteWindLayer;
