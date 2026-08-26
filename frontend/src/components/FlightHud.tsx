import type { FlowField2DResponse } from '../api';
import type { ViewMode } from '../appModel';
import type { EnergyGraphScale } from './EnergyGraph';
import EnergyGraph from './EnergyGraph';
import Metric from './Metric';
import MissionIntelligence from './MissionIntelligence';
import type { Telemetry } from './TopDownGame';

interface FlightHudProps {
  flow: FlowField2DResponse | null;
  telemetry: Telemetry;
  energyHistory: number[];
  viewMode: ViewMode;
  energyGraphScale: EnergyGraphScale;
  onEnergyGraphScaleChange: (scale: EnergyGraphScale) => void;
}

export default function FlightHud({ flow, telemetry, energyHistory, viewMode, energyGraphScale, onEnergyGraphScaleChange }: FlightHudProps) {
  const lidarTitle = viewMode === '3d'
    ? '3D SENSOR MAP / SIM ODOMETRY'
    : '2D SENSOR MAP / SIM ODOMETRY';
  const lidarAriaLabel = viewMode === '3d'
    ? 'Three-dimensional rolling sensor-map telemetry using simulator odometry'
    : 'Two-dimensional rolling sensor-map telemetry using simulator odometry';

  return (
    <>
      <div className="hud-grid">
        <Metric label="Drone Speed" value={`${telemetry.droneSpeed.toFixed(1)} m/s`} />
        <Metric label="Air Rel" value={`${(telemetry.relativeAirSpeed ?? 0).toFixed(1)} m/s`} />
        <Metric label="Local Wind" value={`${telemetry.localWindSpeed.toFixed(1)} m/s`} />
        <Metric label="Wind Dir" value={`${telemetry.localWindDirDeg.toFixed(0)}°`} />
        <Metric label="Heading" value={`${telemetry.headingDeg.toFixed(0)}°`} />
        <Metric label="Flow" value={telemetry.flowType ?? 'CROSS'} />
        <Metric label="Parasite" value={`${(telemetry.dragPowerW ?? 0).toFixed(0)} W`} />
        <Metric label="Induced" value={`${(telemetry.inducedPowerW ?? 0).toFixed(0)} W`} />
        <Metric label="Energy Burn" value={`${telemetry.energyRate.toFixed(1)} u/s`} wide />
        <Metric label="Total Energy" value={`${telemetry.energyUsed.toFixed(0)} u`} wide accent />
      </div>
      {telemetry.lidar && (
        <div className="lidar-readout" aria-label={lidarAriaLabel}>
          <div className="section-header">
            <span>{lidarTitle} / {telemetry.lidar.ui.maxRange} M / {telemetry.lidar.sampleCount} PTS</span>
            <strong>{telemetry.lidar.ui.hitCount} HITS</strong>
          </div>
          <div className="lidar-metrics">
            <Metric
              label="Nearest"
              value={telemetry.lidar.nearestDistance === null ? 'No hit' : `${telemetry.lidar.nearestDistance.toFixed(1)} m`}
            />
            <Metric label="Hit Ratio" value={`${(telemetry.lidar.hitRatio * 100).toFixed(0)}%`} />
            <Metric label="Observation" value={`${telemetry.lidar.observation.length} values`} wide />
          </div>
          <p className="lidar-disclaimer">Simulator sensor returns against OSM collision meshes{viewMode === '3d' ? ' and the ground plane' : ''}; display odometry only, with no loop closure or full SLAM localization.</p>
        </div>
      )}
      <EnergyGraph history={energyHistory} scale={energyGraphScale} adjustable onScaleChange={onEnergyGraphScaleChange} />
      <MissionIntelligence flow={flow} telemetry={telemetry} energyHistory={energyHistory} viewMode={viewMode} />
    </>
  );
}
