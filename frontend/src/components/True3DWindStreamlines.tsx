import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Line } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface True3DPayload {
  kind: string;
  label: string;
  meta: {
    max_speed_mps?: number;
    mean_abs_w_mps?: number;
    max_abs_w_mps?: number;
    nx?: number;
    ny?: number;
    nz?: number;
    buildings?: number;
  };
  streamlines: number[][][];
}

interface PathLine {
  points: THREE.Vector3[];
  color: string;
  offset: number;
}

interface True3DWindStreamlinesProps {
  url: string;
  visible?: boolean;
}

function toThreePoint(point: number[]) {
  // Solver coordinates are [x, y, z_altitude]. Three.js scene is [x, y_altitude, z].
  return new THREE.Vector3(point[0], point[2], point[1]);
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

const True3DFlowParticles: React.FC<{ lines: PathLine[] }> = ({ lines }) => {
  const refs = useRef<(THREE.Mesh | null)[]>([]);
  const tmp = useMemo(() => new THREE.Vector3(), []);
  const up = useMemo(() => new THREE.Vector3(0, 1, 0), []);
  const particles = useMemo(() => {
    const items: { line: PathLine; offset: number; color: string; scale: number }[] = [];
    lines.forEach((line, lineIdx) => {
      const count = line.points.length > 80 ? 4 : 3;
      for (let i = 0; i < count; i += 1) {
        items.push({
          line,
          offset: (i / count + line.offset + lineIdx * 0.017) % 1,
          color: line.color,
          scale: 1.4 + (lineIdx % 3) * 0.35,
        });
      }
    });
    return items;
  }, [lines]);

  useFrame(({ clock }) => {
    const elapsed = clock.getElapsedTime();
    particles.forEach((particle, idx) => {
      const mesh = refs.current[idx];
      if (!mesh) return;
      const t = elapsed * 0.075 + particle.offset;
      samplePolyline(particle.line.points, t, tmp);
      mesh.position.copy(tmp);
      const next = samplePolyline(particle.line.points, t + 0.006, new THREE.Vector3());
      const direction = next.sub(tmp).normalize();
      if (direction.lengthSq() > 0.0001) mesh.quaternion.setFromUnitVectors(up, direction);
    });
  });

  return (
    <group name="true-3d-uvw-flow-particles">
      {particles.map((particle, idx) => (
        <mesh key={`uvw-particle-${idx}`} ref={(node) => { refs.current[idx] = node; }} scale={particle.scale}>
          <coneGeometry args={[0.9, 4.2, 12]} />
          <meshBasicMaterial color={particle.color} transparent opacity={0.96} toneMapped={false} />
        </mesh>
      ))}
    </group>
  );
};

const True3DWindStreamlines: React.FC<True3DWindStreamlinesProps> = ({ url, visible = true }) => {
  const [payload, setPayload] = useState<True3DPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    fetch(url)
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return response.json() as Promise<True3DPayload>;
      })
      .then((data) => { if (!cancelled) setPayload(data); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)); });
    return () => { cancelled = true; };
  }, [url]);

  const lines = useMemo<PathLine[]>(() => {
    if (!payload) return [];
    return payload.streamlines
      .map((line, idx) => {
        const points = line.map(toThreePoint).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z));
        const hue = 0.57 - (idx / Math.max(payload.streamlines.length - 1, 1)) * 0.48;
        const color = `#${new THREE.Color().setHSL(hue, 0.95, 0.58).getHexString()}`;
        return { points, color, offset: idx * 0.031 };
      })
      .filter((line) => line.points.length > 5);
  }, [payload]);

  if (!visible || !payload || error) return null;

  return (
    <group name="true-3d-uvw-potential-flow-layer">
      {lines.map((line, idx) => (
        <Line key={`uvw-line-${idx}`} points={line.points} color={line.color} lineWidth={1.35} transparent opacity={0.66} />
      ))}
      <True3DFlowParticles lines={lines} />
    </group>
  );
};

export default True3DWindStreamlines;
