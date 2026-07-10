import { useMemo, useState } from 'react';
import * as THREE from 'three';
import { getLidarJetCss, type LidarDisplayPose, type LidarTelemetry } from '../sensors/lidar';
import { DEFAULT_LIDAR_2D_CONFIG } from '../sensors/lidar2d';
import {
  ROLLING_SENSOR_MAP_2D_POINTS_PER_KEYFRAME,
  ROLLING_SENSOR_MAP_KEYFRAMES,
  appendRollingSensorKeyframe,
  createRollingSensorKeyframe,
  getLidarPoseMatrix,
  type RollingSensorKeyframe,
} from '../sensors/rollingSensorMap';

interface LocalReturns2DProps {
  lidar?: LidarTelemetry;
  currentPose: LidarDisplayPose;
  enabled: boolean;
  headingDeg: number;
}

const SIZE = 210;
const CENTER = SIZE / 2;
const RADIUS = 84;
const MIN_VIEW_ZOOM = 0.5;
const MAX_VIEW_ZOOM = 3;
const VIEW_ZOOM_STEP = 1.25;

export default function LocalReturns2D({ lidar, currentPose, enabled, headingDeg }: LocalReturns2DProps) {
  const [history, setHistory] = useState<{ lastLidar?: LidarTelemetry; keyframes: RollingSensorKeyframe[] }>({ keyframes: [] });
  const [viewZoom, setViewZoom] = useState(1);
  let displayedHistory = history;
  if (enabled && lidar && history.lastLidar !== lidar) {
    displayedHistory = {
      lastLidar: lidar,
      keyframes: appendRollingSensorKeyframe(
        history.keyframes,
        createRollingSensorKeyframe(lidar, ROLLING_SENSOR_MAP_2D_POINTS_PER_KEYFRAME),
      ),
    };
    setHistory(displayedHistory);
  }
  const { keyframes } = displayedHistory;

  const maxRange = lidar?.ui.maxRange ?? DEFAULT_LIDAR_2D_CONFIG.maxRange;
  const scale = (RADIUS / Math.max(1, maxRange)) * viewZoom;
  const currentReturns = useMemo(() => {
    if (!enabled || !lidar) return [];
    const transform = getLidarPoseMatrix(lidar.ui.scanPose);
    return lidar.ui.returns.map((sample) => ({
      sample,
      point: new THREE.Vector3(sample.relativeX, sample.relativeY, sample.relativeZ).applyMatrix4(transform),
    }));
  }, [enabled, lidar]);
  const historicalReturns = useMemo(() => {
    const frames = keyframes.slice(0, -1);
    return frames.flatMap((keyframe, frameIndex) => {
      const transform = getLidarPoseMatrix(keyframe.pose);
      const age = (frameIndex + 1) / Math.max(1, frames.length);
      return keyframe.returns.map((sample) => ({
        sample,
        age,
        point: new THREE.Vector3(sample.relativeX, sample.relativeY, sample.relativeZ).applyMatrix4(transform),
      }));
    });
  }, [keyframes]);
  const trajectory = useMemo(() => {
    return keyframes.map((keyframe) => new THREE.Vector3(keyframe.pose.x, keyframe.pose.y, keyframe.pose.z));
  }, [keyframes]);
  const hitCount = enabled ? lidar?.ui.hitCount ?? 0 : 0;
  const state = enabled ? (lidar ? `${hitCount} HITS` : 'SCANNING') : 'LiDAR OFF';
  const setClampedViewZoom = (zoom: number) => {
    setViewZoom(Math.min(MAX_VIEW_ZOOM, Math.max(MIN_VIEW_ZOOM, zoom)));
  };

  const worldToCurrent = getLidarPoseMatrix(currentPose).invert().elements;
  const sensorFrameTransform = `matrix(${scale * worldToCurrent[0]} ${-scale * worldToCurrent[2]} ${scale * worldToCurrent[8]} ${-scale * worldToCurrent[10]} ${CENTER + scale * worldToCurrent[12]} ${CENTER - scale * worldToCurrent[14]})`;
  const trajectoryPoints = trajectory.map((point) => `${point.x},${point.z}`).join(' ');

  return (
    <section className="local-radar local-radar-2d" aria-label="Two-dimensional rolling LiDAR sensor-map viewer">
      <div className="local-radar__header">
        <span>ROLLING SENSOR MAP · SIM ODOMETRY · NO LOOP CLOSURE</span>
        <strong>{state}</strong>
      </div>
      <div
        className="local-radar-2d__viewport"
        onWheel={(event) => {
          event.preventDefault();
          setClampedViewZoom(viewZoom * (event.deltaY < 0 ? VIEW_ZOOM_STEP : 1 / VIEW_ZOOM_STEP));
        }}
      >
        <svg className="local-radar-2d__plot" viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label="Drone-centered rolling LiDAR returns and simulated odometry trajectory">
        <rect width={SIZE} height={SIZE} fill="#050b10" />
        <circle cx={CENTER} cy={CENTER} r={RADIUS} fill="none" stroke="#75dceb" strokeOpacity="0.28" />
        {[0.25, 0.5, 0.75].map((fraction) => <circle key={fraction} cx={CENTER} cy={CENTER} r={RADIUS * fraction} fill="none" stroke="#315f69" strokeOpacity="0.36" />)}
        <line x1={CENTER - RADIUS} y1={CENTER} x2={CENTER + RADIUS} y2={CENTER} stroke="#ff625c" strokeOpacity="0.34" />
        <line x1={CENTER} y1={CENTER - RADIUS} x2={CENTER} y2={CENTER + RADIUS} stroke="#58a7ff" strokeOpacity="0.34" />
        <text x={CENTER + RADIUS - 11} y={CENTER - 5} fill="#ff8782" fontSize="8">+X</text>
        <text x={CENTER + 5} y={CENTER - RADIUS + 9} fill="#77b5ff" fontSize="8">+Y</text>
        {enabled && <g transform={sensorFrameTransform}>
          {historicalReturns.map(({ sample, point, age }, index) => (
            <circle key={`history-${index}`} cx={point.x} cy={point.z} r={(sample.hit ? 1.65 : 0.9) / scale} fill={getLidarJetCss(sample.normalizedDistance)} fillOpacity={(sample.hit ? 0.18 : 0.08) + age * (sample.hit ? 0.3 : 0.12)} />
          ))}
          {trajectory.length >= 2 && <polyline points={trajectoryPoints} fill="none" stroke="#6ff7ff" strokeWidth={2.6 / scale} strokeOpacity="0.96" strokeLinejoin="round" strokeLinecap="round" />}
          {trajectory.map((point, index) => <circle key={`pose-${index}`} cx={point.x} cy={point.z} r={2.25 / scale} fill="#fff4a8" stroke="#071016" strokeWidth={0.8 / scale} />)}
          {currentReturns.map(({ sample, point }, index) => (
            <circle key={`current-${index}`} cx={point.x} cy={point.z} r={(sample.hit ? 2.4 : 1.35) / scale} fill={getLidarJetCss(sample.normalizedDistance)} fillOpacity={sample.hit ? 1 : 0.42} />
          ))}
        </g>}
        <circle cx={CENTER} cy={CENTER} r="4" fill="#ffffff" />
        <path d={`M ${CENTER + 7} ${CENTER} L ${CENTER - 2} ${CENTER - 5} L ${CENTER - 2} ${CENTER + 5} Z`} fill="#bff6ff" />
        </svg>
        <div className="local-slam-viewer__controls" aria-label="2D map view controls">
          <button type="button" onClick={() => setClampedViewZoom(viewZoom / VIEW_ZOOM_STEP)} aria-label="Zoom out">−</button>
          <button type="button" onClick={() => setClampedViewZoom(viewZoom * VIEW_ZOOM_STEP)} aria-label="Zoom in">+</button>
          <button type="button" className="local-slam-viewer__reset" onClick={() => setViewZoom(1)}>RESET VIEW</button>
        </div>
      </div>
      <div className="local-radar__footer">
        <span>{maxRange} M · {lidar?.sampleCount ?? DEFAULT_LIDAR_2D_CONFIG.horizontalSamples} LIVE · {keyframes.length}/{ROLLING_SENSOR_MAP_KEYFRAMES} KF</span>
        <button type="button" className="local-radar__clear" onClick={() => setHistory({ lastLidar: lidar, keyframes: [] })}>CLEAR MAP</button>
      </div>
      <div className="local-slam-viewer__hint">DRAG TITLE TO MOVE · CORNER TO RESIZE · SCROLL OR BUTTONS TO ZOOM · HDG {headingDeg.toFixed(0)}° · RETURNS + DISPLAY ODOMETRY ONLY · NO WORLD POLYGONS</div>
    </section>
  );
}
