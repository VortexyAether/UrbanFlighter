import { useEffect, useMemo, useState } from 'react';
import * as THREE from 'three';
import type { FlowField2DResponse } from '../api';
import { TRUE_3D_WIND_URL } from '../appModel';
import Aircraft, { type AircraftMetrics } from './Aircraft';
import CameraFollow from './CameraFollow';
import CFDLiteWindLayer from './CFDLiteWindLayer';
import CityModel from './CityModel';
import FlightDomainBoundary from './FlightDomainBoundary';
import FlightGameHud from './FlightGameHud';
import FreeFlightBeacon from './FreeFlightBeacon';
import ThreeCanvas from './ThreeCanvas';
import True3DWindStreamlines from './True3DWindStreamlines';
import UrbanDressing from './UrbanDressing';
import type { Telemetry } from './TopDownGame';
import { hasResolvedFlowGrid } from '../utils/flowFieldSampling';
import { buildBuildingCollisionMeshes } from '../geometry/buildingGeometry';
import type { LidarTelemetry } from '../sensors/lidar';
import { normalizeBearingDeg, sceneVectorToCompassBearingDeg } from '../utils/bearings';
import { createUrbanDressingLayout } from '../presentation/urbanDressing';
import { selectFreeFlightBeacon } from '../presentation/freeFlightBeacon';
import { DRONE_SCALE_CONTRACT, resolveDroneSafetyRadius } from '../simulation/droneScale';
import {
  buildFlight3DObstacles,
  selectFlight3DSpawn,
  type Flight3DBounds,
} from '../simulation/flight3dMotion';
import type { DroneControlPreset, FlightCameraMode } from '../simulation/flight3dControls';

interface Simulation3DProps {
  flow: FlowField2DResponse | null;
  selectedLocation: { lat: number; lon: number };
  showFlowAnimation: boolean;
  onTelemetry: (telemetry: Telemetry) => void;
  followCamera?: boolean;
  onFollowCameraChange: (enabled: boolean) => void;
  showLidar?: boolean;
  onLidarTelemetry?: (telemetry: LidarTelemetry | null) => void;
  true3DWind?: boolean;
  controlPreset: DroneControlPreset;
  onControlPresetChange: (preset: DroneControlPreset) => void;
  presentationMode: boolean;
  onPresentationModeChange: (enabled: boolean) => void;
  hideFlightHud?: boolean;
  showcaseFraming?: boolean;
}

const DEFAULT_BOUNDS: Flight3DBounds = { min_x: -400, max_x: 400, min_y: -400, max_y: 400 };
const TRUE_3D_PRESENTATION_SPAWN_BOUNDS: Flight3DBounds = { min_x: -190, max_x: 190, min_y: -190, max_y: 190 };
const CFD_ALTITUDE_LEVELS = [28, 56, 92];

function inletDirectionDeg(flow: FlowField2DResponse | null) {
  if (!flow) return 45;
  return (Math.atan2(flow.inlet.uy, flow.inlet.ux) * 180) / Math.PI;
}

function toTelemetry(metrics: AircraftMetrics): Telemetry {
  return {
    droneSpeed: metrics.velocity.length(),
    localWindSpeed: metrics.windSpeed,
    localWindDirDeg: sceneVectorToCompassBearingDeg(
      metrics.windDirection.x,
      metrics.windDirection.z,
    ),
    energyRate: metrics.energyMetrics.consumptionRate,
    energyUsed: metrics.energy,
    headingDeg: normalizeBearingDeg((-metrics.yaw * 180) / Math.PI),
    position: { x: metrics.position.x, y: metrics.position.z },
    displayPose: {
      x: metrics.position.x,
      y: metrics.position.y,
      z: metrics.position.z,
      yaw: metrics.yaw,
      pitch: metrics.pitch,
      roll: metrics.roll,
    },
    lidar: metrics.lidar ?? undefined,
    relativeAirSpeed: metrics.energyMetrics.relativeAirSpeed,
    dragForceN: metrics.energyMetrics.dragForceN,
    dragPowerW: metrics.energyMetrics.dragPowerW,
    inducedPowerW: metrics.energyMetrics.inducedPowerW,
    totalPowerW: metrics.energyMetrics.totalPowerW,
    flowType: metrics.energyMetrics.flowType,
  };
}

function isInteractiveTarget(target: EventTarget | null) {
  if (!(target instanceof Element)) return false;
  if (target instanceof HTMLElement && target.isContentEditable) return true;
  return target.closest('button,input,select,textarea,a,[contenteditable]:not([contenteditable="false"]),[tabindex]:not([tabindex="-1"])') !== null;
}

export default function Simulation3D({
  flow,
  selectedLocation,
  showFlowAnimation,
  onTelemetry,
  followCamera = true,
  onFollowCameraChange,
  showLidar = true,
  onLidarTelemetry,
  true3DWind = false,
  controlPreset,
  onControlPresetChange,
  presentationMode,
  onPresentationModeChange,
  hideFlightHud = false,
  showcaseFraming = false,
}: Simulation3DProps) {
  const [aircraftMetrics, setAircraftMetrics] = useState<AircraftMetrics | null>(null);
  const [true3DOverlayStatus, setTrue3DOverlayStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const windSpeed = flow?.inlet.speed_mps ?? flow?.weather.wind_speed ?? 8;
  const windDir = useMemo(() => inletDirectionDeg(flow), [flow]);
  const buildings = useMemo(() => flow?.buildings ?? [], [flow]);
  const bounds = flow?.field.bounds ?? DEFAULT_BOUNDS;
  const hasBackendWindGrid = hasResolvedFlowGrid(flow);
  const worldIdentity = flow?.live_scenario?.scenario_id
    ?? `${selectedLocation.lat.toFixed(6)}:${selectedLocation.lon.toFixed(6)}:${buildings.length}`;
  const safetyRadiusM = resolveDroneSafetyRadius(
    flow?.live_scenario?.collision_lidar_semantics.agent_radius_m,
  );
  const verticalSafetyClearanceM = DRONE_SCALE_CONTRACT.researchSafetyEnvelope.verticalClearanceM;
  const flightObstacles = useMemo(() => buildFlight3DObstacles(buildings), [buildings]);
  const spawnBounds = useMemo(() => true3DWind ? {
    min_x: Math.max(bounds.min_x, TRUE_3D_PRESENTATION_SPAWN_BOUNDS.min_x),
    max_x: Math.min(bounds.max_x, TRUE_3D_PRESENTATION_SPAWN_BOUNDS.max_x),
    min_y: Math.max(bounds.min_y, TRUE_3D_PRESENTATION_SPAWN_BOUNDS.min_y),
    max_y: Math.min(bounds.max_y, TRUE_3D_PRESENTATION_SPAWN_BOUNDS.max_y),
  } : bounds, [bounds, true3DWind]);
  const spawnPosition = useMemo(
    () => selectFlight3DSpawn(
      spawnBounds,
      flightObstacles,
      safetyRadiusM,
      true3DWind
        ? {
            baseAltitudeM: 62,
            desiredBuildingClearanceM: 34,
            favorCityAhead: false,
            verticalSafetyClearanceM,
          }
        : { verticalSafetyClearanceM },
    ),
    [flightObstacles, safetyRadiusM, spawnBounds, true3DWind, verticalSafetyClearanceM],
  );
  const dressingIdentity = useMemo(() => ({
    scenarioId: flow?.live_scenario?.scenario_id,
    lat: selectedLocation.lat,
    lon: selectedLocation.lon,
  }), [flow?.live_scenario?.scenario_id, selectedLocation.lat, selectedLocation.lon]);
  const beacon = useMemo(() => selectFreeFlightBeacon(
    bounds,
    buildings,
    dressingIdentity,
    { x: spawnPosition.x, y: -spawnPosition.z },
  ), [bounds, buildings, dressingIdentity, spawnPosition.x, spawnPosition.z]);
  const dressing = useMemo(() => createUrbanDressingLayout({
    bounds,
    buildings,
    identity: dressingIdentity,
    exclusionZones: [
      { x: spawnPosition.x, y: -spawnPosition.z, radius: 22, kind: 'start' },
      { x: beacon.x, y: beacon.y, radius: 22, kind: 'goal' },
    ],
  }), [beacon.x, beacon.y, bounds, buildings, dressingIdentity, spawnPosition.x, spawnPosition.z]);
  const buildingCollisionMeshes = useMemo(
    () => showLidar ? buildBuildingCollisionMeshes(buildings) : [],
    [buildings, showLidar],
  );
  const cameraMode: FlightCameraMode = followCamera ? 'chase' : 'orbit';
  const cameraTarget = aircraftMetrics?.position ?? vectorFromSpawn(spawnPosition);

  useEffect(() => () => {
    buildingCollisionMeshes.forEach((mesh) => {
      if (!(mesh instanceof THREE.Mesh)) return;
      mesh.geometry.dispose();
      if (Array.isArray(mesh.material)) mesh.material.forEach((material) => material.dispose());
      else mesh.material.dispose();
    });
  }, [buildingCollisionMeshes]);

  useEffect(() => {
    const handleCameraShortcut = (event: KeyboardEvent) => {
      if (event.code !== 'KeyC' || event.repeat || isInteractiveTarget(event.target)) return;
      event.preventDefault();
      onFollowCameraChange(!followCamera);
    };
    window.addEventListener('keydown', handleCameraShortcut);
    return () => window.removeEventListener('keydown', handleCameraShortcut);
  }, [followCamera, onFollowCameraChange]);

  const handleMetricsUpdate = (metrics: AircraftMetrics) => {
    setAircraftMetrics(metrics);
    onTelemetry(toTelemetry(metrics));
  };

  return (
    <div className="simulation-3d">
      <ThreeCanvas
        cameraMode={cameraMode}
        orbitTarget={cameraTarget}
        presentationMode={presentationMode}
      >
        <FlightDomainBoundary bounds={bounds} />
        <CityModel buildings={buildings} bounds={bounds} presentationMode={presentationMode} />
        {presentationMode && <UrbanDressing layout={dressing} />}
        {presentationMode && <FreeFlightBeacon beacon={beacon} />}
        {true3DWind && (
          <True3DWindStreamlines
            url={TRUE_3D_WIND_URL}
            visible={showFlowAnimation}
            onLoadStateChange={setTrue3DOverlayStatus}
          />
        )}
        {!true3DWind && showFlowAnimation && hasBackendWindGrid && (
          <CFDLiteWindLayer
            flow={flow}
            height={2.4}
            showContour
            showArrows={false}
            showStreamlines
            altitudeLevels={showcaseFraming ? [18, 28, 42] : CFD_ALTITUDE_LEVELS}
            showcase={showcaseFraming}
          />
        )}
        <Aircraft
          globalWindSpeed={windSpeed}
          globalWindDir={windDir}
          buildings={buildings}
          flow={flow}
          bounds={bounds}
          flightObstacles={flightObstacles}
          spawnPosition={spawnPosition}
          worldIdentity={worldIdentity}
          safetyRadiusM={safetyRadiusM}
          verticalSafetyClearanceM={verticalSafetyClearanceM}
          controlPreset={controlPreset}
          buildingCollisionMeshes={buildingCollisionMeshes}
          lidarVisible={showLidar}
          lidarVisualizationVisible={showLidar && !presentationMode}
          showSafetyEnvelope={!presentationMode}
          visualScale={showcaseFraming ? 11 : 1}
          motionTimeScale={showcaseFraming ? 4 : 1}
          onMetricsUpdate={handleMetricsUpdate}
          onLidarUpdate={onLidarTelemetry}
        />
        <CameraFollow
          target={cameraTarget}
          yaw={aircraftMetrics?.yaw ?? 0}
          pitch={aircraftMetrics?.pitch ?? 0}
          enabled={followCamera}
          buildings={buildings}
          distance={showcaseFraming ? 22 : undefined}
          height={showcaseFraming ? 9.2 : undefined}
          lookAhead={showcaseFraming ? 14 : undefined}
        />
      </ThreeCanvas>
      {!hideFlightHud && (
      <FlightGameHud
        metrics={aircraftMetrics}
        safetyRadiusM={safetyRadiusM}
        verticalSafetyClearanceM={verticalSafetyClearanceM}
        controlPreset={controlPreset}
        cameraMode={cameraMode}
        presentationMode={presentationMode}
        dressing={dressing}
        beacon={beacon}
        true3DOverlayStatus={true3DWind ? true3DOverlayStatus : undefined}
        onControlPresetChange={onControlPresetChange}
        onCameraModeChange={(mode) => onFollowCameraChange(mode === 'chase')}
        onPresentationModeChange={onPresentationModeChange}
      />
      )}
      {!flow && <div className="simulation-3d-loading">Loading 3D field...</div>}
    </div>
  );
}

function vectorFromSpawn(spawn: { x: number; y: number; z: number }) {
  return new THREE.Vector3(spawn.x, spawn.y, spawn.z);
}
