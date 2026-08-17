import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Html, Line } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { isFiniteTrue3DTriplet, mapTrue3DPointToScene } from '../geometry/true3dCoordinates';

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
  streamlines: unknown[][];
}

interface PathLine {
  points: THREE.Vector3[];
  color: string;
  offset: number;
}

interface True3DWindStreamlinesProps {
  url: string;
  visible?: boolean;
  onLoadStateChange?: (status: 'loading' | 'ready' | 'error') => void;
}

interface True3DLoadState {
  url: string;
  payload: True3DPayload | null;
  error: string | null;
}

function isTrue3DPayload(value: unknown): value is True3DPayload {
  if (!value || typeof value !== 'object') return false;
  const candidate = value as Partial<True3DPayload>;
  return typeof candidate.kind === 'string'
    && typeof candidate.label === 'string'
    && !!candidate.meta
    && typeof candidate.meta === 'object'
    && Array.isArray(candidate.streamlines)
    && candidate.streamlines.every((line) => Array.isArray(line));
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
  const tmpNext = useMemo(() => new THREE.Vector3(), []);
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
      samplePolyline(particle.line.points, t + 0.006, tmpNext).sub(tmp);
      if (tmpNext.lengthSq() > 0.0001) mesh.quaternion.setFromUnitVectors(up, tmpNext.normalize());
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

const True3DWindStreamlines: React.FC<True3DWindStreamlinesProps> = ({
  url,
  visible = true,
  onLoadStateChange,
}) => {
  const [loadState, setLoadState] = useState<True3DLoadState>(() => ({ url, payload: null, error: null }));
  const payload = loadState.url === url ? loadState.payload : null;
  const error = loadState.url === url ? loadState.error : null;

  useEffect(() => {
    const controller = new AbortController();
    onLoadStateChange?.('loading');
    fetch(url, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return response.json() as Promise<unknown>;
      })
      .then((data) => {
        if (controller.signal.aborted) return;
        if (!isTrue3DPayload(data)) throw new Error('Invalid dataset: expected labeled streamline arrays.');
        setLoadState({ url, payload: data, error: null });
        onLoadStateChange?.('ready');
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setLoadState({ url, payload: null, error: reason instanceof Error ? reason.message : String(reason) });
        onLoadStateChange?.('error');
      });
    return () => { controller.abort(); };
  }, [onLoadStateChange, url]);

  const lines = useMemo<PathLine[]>(() => {
    if (!payload) return [];
    return payload.streamlines
      .map((line, idx) => {
        const points = line
          .filter(isFiniteTrue3DTriplet)
          .map(mapTrue3DPointToScene)
          .map(([x, y, z]) => new THREE.Vector3(x, y, z));
        const hue = 0.57 - (idx / Math.max(payload.streamlines.length - 1, 1)) * 0.48;
        const color = `#${new THREE.Color().setHSL(hue, 0.95, 0.58).getHexString()}`;
        return { points, color, offset: idx * 0.031 };
      })
      .filter((line) => line.points.length > 5);
  }, [payload]);

  if (!visible) return null;

  if (error) {
    return (
      <Html fullscreen zIndexRange={[30, 0]}>
        <div
          role="alert"
          style={{
            background: 'rgba(30, 6, 10, 0.92)',
            border: '1px solid #ff657a',
            borderRadius: 6,
            boxShadow: '0 8px 28px rgba(0, 0, 0, 0.45)',
            color: '#ffe8ec',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            fontSize: 12,
            left: '50%',
            maxWidth: 420,
            padding: '10px 14px',
            pointerEvents: 'none',
            position: 'absolute',
            textAlign: 'center',
            top: 72,
            transform: 'translateX(-50%)',
          }}
        >
          <strong style={{ display: 'block', marginBottom: 3 }}>True 3D wind layer unavailable</strong>
          <span>{error}</span>
        </div>
      </Html>
    );
  }

  if (!payload) return null;

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
