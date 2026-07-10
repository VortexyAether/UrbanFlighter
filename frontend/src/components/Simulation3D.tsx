import { useEffect, useMemo, useState } from 'react';
import * as THREE from 'three';
import type { FlowField2DResponse } from '../api';
import { TRUE_3D_WIND_URL } from '../appModel';
import Aircraft, { type AircraftMetrics } from './Aircraft';
import CameraFollow from './CameraFollow';
import CFDLiteWindLayer from './CFDLiteWindLayer';
import CircularBoundary from './CircularBoundary';
import CityModel from './CityModel';
import ThreeCanvas from './ThreeCanvas';
import True3DWindStreamlines from './True3DWindStreamlines';
import type { Telemetry } from './TopDownGame';
import { hasResolvedFlowGrid } from '../utils/flowFieldSampling';
import { buildBuildingCollisionMeshes } from '../geometry/buildingGeometry';
import type { LidarTelemetry } from '../sensors/lidar';

interface Simulation3DProps {
  flow: FlowField2DResponse | null;
  showFlowAnimation: boolean;
  onTelemetry: (telemetry: Telemetry) => void;
  followCamera?: boolean;
  showLidar?: boolean;
  onLidarTelemetry?: (telemetry: LidarTelemetry | null) => void;
  true3DWind?: boolean;
}

const EMPTY_TARGET = new THREE.Vector3(0, 55, 0);

function inletDirectionDeg(flow: FlowField2DResponse | null) {
  if (!flow) return 45;
  return (Math.atan2(flow.inlet.uy, flow.inlet.ux) * 180) / Math.PI;
}

function toTelemetry(metrics: AircraftMetrics): Telemetry {
  const horizontalWind = new THREE.Vector2(metrics.windDirection.x, metrics.windDirection.z)
    .multiplyScalar(metrics.windSpeed);

  return {
    droneSpeed: metrics.velocity.length(),
    localWindSpeed: metrics.windSpeed,
    localWindDirDeg: ((Math.atan2(horizontalWind.y, horizontalWind.x) * 180) / Math.PI + 360) % 360,
    energyRate: metrics.energyMetrics.consumptionRate,
    energyUsed: metrics.energy,
    headingDeg: ((-metrics.yaw * 180) / Math.PI + 360) % 360,
    position: {
      x: metrics.position.x,
      y: metrics.position.z,
    },
    displayPose: {
      x: metrics.position.x,
      y: metrics.position.y,
      z: metrics.position.z,
      yaw: metrics.yaw,
      pitch: metrics.pitch,
      roll: metrics.roll,
    },
    lidar: metrics.lidar ?? undefined,
  };
}

export default function Simulation3D({
  flow,
  showFlowAnimation,
  onTelemetry,
  followCamera = true,
  showLidar = true,
  onLidarTelemetry,
  true3DWind = false,
}: Simulation3DProps) {
  const [aircraftMetrics, setAircraftMetrics] = useState<AircraftMetrics | null>(null);

  const windSpeed = flow?.inlet.speed_mps ?? flow?.weather.wind_speed ?? 8;
  const windDir = useMemo(() => inletDirectionDeg(flow), [flow]);
  const buildings = useMemo(() => flow?.buildings ?? [], [flow]);
  const hasBackendWindGrid = hasResolvedFlowGrid(flow);
  const buildingCollisionMeshes = useMemo(() => buildBuildingCollisionMeshes(buildings), [buildings]);

  useEffect(() => () => {
    buildingCollisionMeshes.forEach((mesh) => {
      if (mesh instanceof THREE.Mesh) {
        mesh.geometry.dispose();
        if (Array.isArray(mesh.material)) {
          mesh.material.forEach((material) => material.dispose());
        } else {
          mesh.material.dispose();
        }
      }
    });
  }, [buildingCollisionMeshes]);

  const handleMetricsUpdate = (metrics: AircraftMetrics) => {
    setAircraftMetrics(metrics);
    onTelemetry(toTelemetry(metrics));
  };

  return (
    <div className="simulation-3d">
      <ThreeCanvas>
        <CircularBoundary />
        <CityModel buildings={buildings} />
        {true3DWind && <True3DWindStreamlines url={TRUE_3D_WIND_URL} visible={showFlowAnimation} />}
        {!true3DWind && showFlowAnimation && hasBackendWindGrid && (
          <CFDLiteWindLayer
            flow={flow}
            showContour={false}
            showArrows={false}
            showStreamlines
            altitudeLevels={[22, 38, 56, 78, 106]}
          />
        )}
        <Aircraft
          globalWindSpeed={windSpeed}
          globalWindDir={windDir}
          buildings={buildings}
          flow={flow}
          buildingCollisionMeshes={buildingCollisionMeshes}
          lidarVisible={showLidar}
          onMetricsUpdate={handleMetricsUpdate}
          onLidarUpdate={onLidarTelemetry ?? undefined}
        />
        <CameraFollow
          target={aircraftMetrics?.position ?? EMPTY_TARGET}
          yaw={aircraftMetrics?.yaw ?? 0}
          pitch={aircraftMetrics?.pitch ?? 0}
          enabled={followCamera}
          distance={118}
          height={52}
          lookAhead={88}
        />
      </ThreeCanvas>
      {!flow && (
        <div className="simulation-3d-loading">
          Loading 3D field...
        </div>
      )}
    </div>
  );
}
