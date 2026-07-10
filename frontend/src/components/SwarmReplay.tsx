import { useEffect, useMemo, useRef, useState } from 'react';
import { Line } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { BuildingData } from '../api';

export interface DroneTrajectory {
  drone_id: string;
  start: number[];
  goal: number[];
  trajectory: number[][];
  waypoints: number[][];
}

interface SwarmReplayPayload {
  world?: {
    buildings?: Array<{
      center: number[];
      size: number[];
      height_m: number;
    }>;
  };
  drones: DroneTrajectory[];
}

export interface SwarmReplayStatus {
  loaded: boolean;
  droneCount: number;
  frame: number;
  totalFrames: number;
  sourceUrl: string;
  error?: string;
}

interface SwarmReplayProps {
  url: string;
  speed?: number;
  playing?: boolean;
  frameOverride?: number;
  resetToken?: number;
  onFocusUpdate?: (focus: THREE.Vector3) => void;
  onReplayBuildingsUpdate?: (buildings: BuildingData[]) => void;
  onStatusUpdate?: (status: SwarmReplayStatus) => void;
}

const COLORS = ['#ff4d6d', '#4cc9f0', '#80ed99', '#ffd166', '#c77dff', '#f77f00', '#06d6a0', '#f72585'];

function simPointToThree(point: number[]) {
  // Python simulator uses [x, y, z_altitude]. Three scene uses [x, y_altitude, z].
  return new THREE.Vector3(point[0], point[2], point[1]);
}

function makeLinePoints(points: number[][]) {
  return points.map(simPointToThree);
}

function makeGroundLinePoints(points: number[][]) {
  return points.map((point) => new THREE.Vector3(point[0], 0.45, point[1]));
}

function makeTetherPairs(points: number[][]) {
  const stride = Math.max(1, Math.ceil(points.length / 12));
  return points
    .filter((_, index) => index % stride === 0)
    .map((point) => [simPointToThree(point), new THREE.Vector3(point[0], 0.45, point[1])]);
}

function maxFrames(payload: SwarmReplayPayload | null) {
  return Math.max(0, ...(payload?.drones.map((drone) => drone.trajectory.length) ?? [0]));
}

function replayWorldToBuildings(payload: SwarmReplayPayload): BuildingData[] {
  return payload.world?.buildings?.map((building) => {
    const cx = building.center[0];
    const cy = building.center[1];
    const sx = building.size[0] / 2;
    const sy = building.size[1] / 2;
    return {
      height: building.height_m,
      footprint: [
        [cx - sx, cy - sy],
        [cx + sx, cy - sy],
        [cx + sx, cy + sy],
        [cx - sx, cy + sy],
      ],
    };
  }) ?? [];
}

export default function SwarmReplay({
  url,
  speed = 18,
  playing = true,
  frameOverride,
  resetToken = 0,
  onFocusUpdate,
  onReplayBuildingsUpdate,
  onStatusUpdate,
}: SwarmReplayProps) {
  const [payload, setPayload] = useState<SwarmReplayPayload | null>(null);
  const droneRefs = useRef<Array<THREE.Group | null>>([]);
  const clockRef = useRef(0);
  const lastStatusRef = useRef('');

  useEffect(() => {
    let cancelled = false;
    lastStatusRef.current = '';
    onStatusUpdate?.({ loaded: false, droneCount: 0, frame: 0, totalFrames: 0, sourceUrl: url });

    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load swarm replay ${url}: ${res.status}`);
        return res.json();
      })
      .then((data: SwarmReplayPayload) => {
        if (cancelled) return;
        setPayload(data);
        onReplayBuildingsUpdate?.(replayWorldToBuildings(data));
        const totalFrames = maxFrames(data);
        onStatusUpdate?.({ loaded: true, droneCount: data.drones.length, frame: 0, totalFrames, sourceUrl: url });
      })
      .catch((err: Error) => {
        console.error(err);
        if (!cancelled) {
          onStatusUpdate?.({ loaded: false, droneCount: 0, frame: 0, totalFrames: 0, sourceUrl: url, error: err.message });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [onReplayBuildingsUpdate, onStatusUpdate, url]);

  useEffect(() => {
    clockRef.current = 0;
  }, [resetToken, url]);

  useEffect(() => {
    if (frameOverride !== undefined) {
      clockRef.current = frameOverride;
    }
  }, [frameOverride]);

  const linePoints = useMemo(() => {
    return payload?.drones.map((drone) => makeLinePoints(drone.trajectory)) ?? [];
  }, [payload]);
  const groundLinePoints = useMemo(() => {
    return payload?.drones.map((drone) => makeGroundLinePoints(drone.trajectory)) ?? [];
  }, [payload]);
  const waypointLinePoints = useMemo(() => {
    return payload?.drones.map((drone) => makeLinePoints(drone.waypoints)) ?? [];
  }, [payload]);
  const tetherPairs = useMemo(() => {
    return payload?.drones.map((drone) => makeTetherPairs(drone.trajectory)) ?? [];
  }, [payload]);

  useFrame((_, delta) => {
    if (!payload) return;
    if (playing) {
      clockRef.current += delta * speed;
    }

    const totalFrames = maxFrames(payload);
    const frame = totalFrames > 0 ? Math.floor(clockRef.current) % totalFrames : 0;
    const focus = new THREE.Vector3();
    let activeCount = 0;

    payload.drones.forEach((drone, idx) => {
      const ref = droneRefs.current[idx];
      if (!ref || drone.trajectory.length === 0) return;
      const localFrame = frame % drone.trajectory.length;
      const nextFrame = Math.min(localFrame + 1, drone.trajectory.length - 1);
      const alpha = clockRef.current - Math.floor(clockRef.current);
      const p0 = simPointToThree(drone.trajectory[localFrame]);
      const p1 = simPointToThree(drone.trajectory[nextFrame]);
      ref.position.copy(p0.lerp(p1, alpha));
      focus.add(ref.position);
      activeCount += 1;

      if (nextFrame !== localFrame) {
        const dir = simPointToThree(drone.trajectory[nextFrame]).sub(simPointToThree(drone.trajectory[localFrame]));
        if (dir.lengthSq() > 1e-6) {
          ref.lookAt(ref.position.clone().add(dir));
        }
      }
    });

    if (activeCount > 0) {
      focus.divideScalar(activeCount);
      focus.y += 28;
      onFocusUpdate?.(focus);
    }

    const statusKey = `${frame}:${totalFrames}:${playing}`;
    if (statusKey !== lastStatusRef.current) {
      lastStatusRef.current = statusKey;
      onStatusUpdate?.({ loaded: true, droneCount: payload.drones.length, frame, totalFrames, sourceUrl: url });
    }
  });

  if (!payload) return null;

  return (
    <group name="deterministic-swarm-replay">
      {payload.drones.map((drone, idx) => {
        const color = COLORS[idx % COLORS.length];
        const start = simPointToThree(drone.start);
        const goal = simPointToThree(drone.goal);
        return (
          <group key={drone.drone_id}>
            <Line points={groundLinePoints[idx]} color="#050607" lineWidth={1.2} transparent opacity={0.34} />
            {waypointLinePoints[idx].length > 1 && (
              <Line points={waypointLinePoints[idx]} color="#ffffff" lineWidth={1.1} transparent opacity={0.24} />
            )}
            <Line points={linePoints[idx]} color={color} lineWidth={3.1} transparent opacity={0.9} />
            {tetherPairs[idx].map((points, tetherIndex) => (
              <Line
                key={`${drone.drone_id}-tether-${tetherIndex}`}
                points={points}
                color={color}
                lineWidth={0.65}
                transparent
                opacity={0.22}
              />
            ))}

            <mesh position={start}>
              <sphereGeometry args={[3.2, 16, 16]} />
              <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.15} />
            </mesh>
            <mesh position={goal}>
              <octahedronGeometry args={[5.5, 0]} />
              <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.35} />
            </mesh>

            <group ref={(el) => { droneRefs.current[idx] = el; }} position={start}>
              <mesh rotation={[Math.PI / 2, 0, 0]}>
                <coneGeometry args={[3.8, 10, 16]} />
                <meshStandardMaterial color={color} metalness={0.25} roughness={0.35} emissive={color} emissiveIntensity={0.3} />
              </mesh>
              <mesh position={[0, 0, 4.5]}>
                <sphereGeometry args={[2.2, 12, 12]} />
                <meshStandardMaterial color="#f8f9fa" metalness={0.15} roughness={0.3} />
              </mesh>
            </group>
          </group>
        );
      })}
    </group>
  );
}
