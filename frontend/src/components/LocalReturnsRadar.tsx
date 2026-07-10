import { Line, OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import {
  DEFAULT_LIDAR_CONFIG,
  getLidarJetColor,
  type LidarDisplayPose,
  type LidarTelemetry,
  type LidarUiReturn,
} from '../sensors/lidar';
import {
  ROLLING_SENSOR_MAP_3D_POINTS_PER_KEYFRAME,
  ROLLING_SENSOR_MAP_KEYFRAMES,
  appendRollingSensorKeyframe,
  createRollingSensorKeyframe,
  getLidarPoseMatrix,
  type RollingSensorKeyframe,
} from '../sensors/rollingSensorMap';

interface LocalReturnsRadarProps {
  lidar?: LidarTelemetry;
  currentPose: LidarDisplayPose;
  enabled: boolean;
  compact?: boolean;
}

interface PointBuffers {
  positions: Float32Array;
  colors: Float32Array;
}

const EMPTY_POINT_BUFFERS: PointBuffers = {
  positions: new Float32Array(),
  colors: new Float32Array(),
};
const DEFAULT_CAMERA_POSITION = new THREE.Vector3(168, 118, 168);
const DEFAULT_CAMERA_DISTANCE = DEFAULT_CAMERA_POSITION.length();
const MIN_CAMERA_DISTANCE = 155;
const MAX_CAMERA_DISTANCE = 390;

function buildCurrentPointSets(lidar: LidarTelemetry | undefined, enabled: boolean) {
  const sets: LidarUiReturn[][] = [[], []];
  if (enabled) lidar?.ui.returns.forEach((sample) => sets[sample.hit ? 1 : 0].push(sample));
  if (!lidar) return [EMPTY_POINT_BUFFERS, EMPTY_POINT_BUFFERS];

  const transform = getLidarPoseMatrix(lidar.ui.scanPose);
  const point = new THREE.Vector3();
  const color = new THREE.Color();
  return sets.map<PointBuffers>((set) => {
    const positions = new Float32Array(set.length * 3);
    const colors = new Float32Array(set.length * 3);
    set.forEach((sample, index) => {
      point.set(sample.relativeX, sample.relativeY, sample.relativeZ).applyMatrix4(transform);
      positions.set(point.toArray(), index * 3);
      getLidarJetColor(sample.normalizedDistance, color);
      colors.set(color.toArray(), index * 3);
    });
    return { positions, colors };
  });
}

function RollingMapGeometry({
  lidar,
  currentPose,
  enabled,
  keyframes,
}: Pick<LocalReturnsRadarProps, 'lidar' | 'currentPose' | 'enabled'> & { keyframes: RollingSensorKeyframe[] }) {
  const currentPointSets = useMemo(
    () => buildCurrentPointSets(lidar, enabled),
    [enabled, lidar],
  );
  const historicalPointSets = useMemo(() => {
    const sets: Array<Array<{ point: THREE.Vector3; sample: LidarUiReturn; age: number }>> = [[], []];
    const historicalFrames = keyframes.slice(0, -1);
    historicalFrames.forEach((keyframe, frameIndex) => {
      const transform = getLidarPoseMatrix(keyframe.pose);
      const age = (frameIndex + 1) / Math.max(1, historicalFrames.length);
      keyframe.returns.forEach((sample) => {
        sets[sample.hit ? 1 : 0].push({
          point: new THREE.Vector3(sample.relativeX, sample.relativeY, sample.relativeZ).applyMatrix4(transform),
          sample,
          age,
        });
      });
    });

    const color = new THREE.Color();
    const subdued = new THREE.Color('#66828a');
    return sets.map<PointBuffers>((set) => {
      const positions = new Float32Array(set.length * 3);
      const colors = new Float32Array(set.length * 3);
      set.forEach(({ point, sample, age }, index) => {
        positions.set(point.toArray(), index * 3);
        getLidarJetColor(sample.normalizedDistance, color)
          .lerp(subdued, 0.72)
          .multiplyScalar(0.28 + age * 0.3);
        colors.set(color.toArray(), index * 3);
      });
      return { positions, colors };
    });
  }, [keyframes]);
  const trajectory = useMemo(() => {
    return keyframes.map((keyframe) => new THREE.Vector3(keyframe.pose.x, keyframe.pose.y, keyframe.pose.z));
  }, [keyframes]);
  const markerPositions = useMemo(() => {
    const positions = new Float32Array(keyframes.length * 3);
    keyframes.forEach((keyframe, index) => {
      positions.set([keyframe.pose.x, keyframe.pose.y, keyframe.pose.z], index * 3);
    });
    return positions;
  }, [keyframes]);
  const worldToCurrent = useMemo(() => getLidarPoseMatrix(currentPose).invert(), [currentPose]);

  return (
    <group matrix={worldToCurrent} matrixAutoUpdate={false} visible={enabled}>
      {historicalPointSets.map((pointSet, index) => (
        <points key={`history-${index}`} frustumCulled={false} renderOrder={1}>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[pointSet.positions, 3]} />
            <bufferAttribute attach="attributes-color" args={[pointSet.colors, 3]} />
          </bufferGeometry>
          <pointsMaterial size={index === 0 ? 1.4 : 2.2} sizeAttenuation={false} vertexColors transparent opacity={index === 0 ? 0.3 : 0.62} depthWrite={false} fog={false} />
        </points>
      ))}
      {currentPointSets.map((pointSet, index) => (
        <points key={`current-${index}`} frustumCulled={false} renderOrder={3}>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[pointSet.positions, 3]} />
            <bufferAttribute attach="attributes-color" args={[pointSet.colors, 3]} />
          </bufferGeometry>
          <pointsMaterial size={index === 0 ? 2.2 : 3.5} sizeAttenuation={false} vertexColors transparent opacity={index === 0 ? 0.52 : 1} depthWrite={false} fog={false} />
        </points>
      ))}
      {trajectory.length >= 2 && <Line points={trajectory} color="#6ff7ff" lineWidth={2.8} transparent opacity={0.95} depthTest={false} renderOrder={5} />}
      <points frustumCulled={false} renderOrder={6}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[markerPositions, 3]} />
        </bufferGeometry>
        <pointsMaterial color="#fff4a8" size={5} sizeAttenuation={false} depthTest={false} depthWrite={false} />
      </points>
    </group>
  );
}

function DroneOriginMarker() {
  return (
    <group>
      <axesHelper args={[34]} />
      <mesh rotation={[Math.PI / 2, 0, 0]}><octahedronGeometry args={[3.2, 0]} /><meshBasicMaterial color="#ffffff" /></mesh>
      <mesh position={[0, 0, -7]} rotation={[-Math.PI / 2, 0, 0]}><coneGeometry args={[2.4, 10, 3]} /><meshBasicMaterial color="#e9fbff" wireframe /></mesh>
      <mesh scale={[11, 0.7, 1.8]}><boxGeometry /><meshBasicMaterial color="#93eaff" wireframe /></mesh>
    </group>
  );
}

function LocalSlamScene({
  cameraDistance,
  resetViewToken,
  onCameraDistanceChange,
  ...props
}: Pick<LocalReturnsRadarProps, 'lidar' | 'currentPose' | 'enabled'> & {
  keyframes: RollingSensorKeyframe[];
  cameraDistance: number;
  resetViewToken: number;
  onCameraDistanceChange: (distance: number) => void;
}) {
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);
  const controlsRef = useRef<OrbitControlsImpl>(null);

  useEffect(() => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    const direction = camera.position.clone().sub(controls.target).normalize();
    camera.position.copy(controls.target).addScaledVector(direction, cameraDistance);
    controls.update();
  }, [cameraDistance]);

  useEffect(() => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    controls.target.set(0, 0, 0);
    camera.position.copy(DEFAULT_CAMERA_POSITION);
    camera.up.set(0, 1, 0);
    controls.update();
  }, [resetViewToken]);

  return (
    <>
      <color attach="background" args={['#050b10']} />
      <PerspectiveCamera ref={cameraRef} makeDefault position={DEFAULT_CAMERA_POSITION.toArray()} fov={46} near={0.5} far={700} />
      <ambientLight intensity={0.9} />
      <gridHelper args={[240, 12, '#315f69', '#17313a']} />
      <mesh rotation={[Math.PI / 2, 0, 0]} renderOrder={0}><ringGeometry args={[119.5, 120, 96]} /><meshBasicMaterial color="#8bdde9" transparent opacity={0.16} side={THREE.DoubleSide} depthWrite={false} /></mesh>
      <DroneOriginMarker />
      <RollingMapGeometry {...props} />
      <OrbitControls
        ref={controlsRef}
        target={[0, 0, 0]}
        enablePan={false}
        enableRotate
        enableZoom
        autoRotate
        autoRotateSpeed={0.35}
        minDistance={MIN_CAMERA_DISTANCE}
        maxDistance={MAX_CAMERA_DISTANCE}
        minPolarAngle={0.12}
        maxPolarAngle={Math.PI - 0.12}
        onChange={() => {
          const camera = cameraRef.current;
          const controls = controlsRef.current;
          if (camera && controls) onCameraDistanceChange(camera.position.distanceTo(controls.target));
        }}
      />
    </>
  );
}

export default function LocalReturnsRadar({ lidar, currentPose, enabled, compact = false }: LocalReturnsRadarProps) {
  const [history, setHistory] = useState<{ lastLidar?: LidarTelemetry; keyframes: RollingSensorKeyframe[] }>({ keyframes: [] });
  const [cameraDistance, setCameraDistance] = useState(DEFAULT_CAMERA_DISTANCE);
  const [resetViewToken, setResetViewToken] = useState(0);
  let displayedHistory = history;
  if (enabled && lidar && history.lastLidar !== lidar) {
    displayedHistory = {
      lastLidar: lidar,
      keyframes: appendRollingSensorKeyframe(
        history.keyframes,
        createRollingSensorKeyframe(lidar, ROLLING_SENSOR_MAP_3D_POINTS_PER_KEYFRAME),
      ),
    };
    setHistory(displayedHistory);
  }
  const { keyframes } = displayedHistory;

  const sampleCount = lidar?.sampleCount ?? DEFAULT_LIDAR_CONFIG.sampleCount;
  const range = lidar?.ui.maxRange ?? DEFAULT_LIDAR_CONFIG.maxRange;
  const hitCount = enabled ? lidar?.ui.hitCount ?? 0 : 0;
  const state = enabled ? (lidar ? `${hitCount} HITS` : 'SCANNING') : 'LiDAR OFF';
  const setClampedCameraDistance = (distance: number) => {
    const clampedDistance = Math.min(MAX_CAMERA_DISTANCE, Math.max(MIN_CAMERA_DISTANCE, distance));
    setCameraDistance((current) => Math.abs(current - clampedDistance) < 0.05 ? current : clampedDistance);
  };
  const resetView = () => {
    setCameraDistance(DEFAULT_CAMERA_DISTANCE);
    setResetViewToken((token) => token + 1);
  };

  return (
    <section className={`local-radar local-slam-viewer${compact ? ' local-radar--compact' : ''}`} aria-label="Interactive three-dimensional rolling LiDAR sensor-map viewer">
      <div className="local-radar__header">
        <span>ROLLING SENSOR MAP · SIM ODOMETRY · NO LOOP CLOSURE</span>
        <strong>{state}</strong>
      </div>
      <div className="local-radar__canvas" role="img" aria-label="Rotatable rolling sensor point cloud and simulated odometry trajectory centered on the drone">
        <Canvas dpr={[1, 2]} gl={{ antialias: true, alpha: false }}>
          <LocalSlamScene lidar={lidar} currentPose={currentPose} enabled={enabled} keyframes={keyframes} cameraDistance={cameraDistance} resetViewToken={resetViewToken} onCameraDistanceChange={setClampedCameraDistance} />
        </Canvas>
        <div className="local-slam-viewer__controls" aria-label="3D map view controls">
          <button type="button" onClick={() => setClampedCameraDistance(cameraDistance * 1.2)} aria-label="Zoom out">−</button>
          <button type="button" onClick={() => setClampedCameraDistance(cameraDistance / 1.2)} aria-label="Zoom in">+</button>
          <button type="button" className="local-slam-viewer__reset" onClick={resetView}>RESET VIEW</button>
        </div>
        <div className="local-slam-viewer__axis" aria-hidden="true"><span className="axis-x">X</span><span className="axis-y">Y</span><span className="axis-z">Z</span></div>
      </div>
      <div className="local-radar__footer">
        <span>{range} M · {sampleCount} LIVE PTS · {keyframes.length}/{ROLLING_SENSOR_MAP_KEYFRAMES} KF</span>
        <button type="button" className="local-radar__clear" onClick={() => setHistory({ lastLidar: lidar, keyframes: [] })}>CLEAR MAP</button>
      </div>
      <div className="local-slam-viewer__hint">{compact ? 'DRAG TITLE TO MOVE · CORNER TO RESIZE · DRAG ORBIT · SCROLL TO ZOOM' : 'DRAG ORBIT · SCROLL TO ZOOM'} · RETURNS + DISPLAY ODOMETRY ONLY</div>
    </section>
  );
}
