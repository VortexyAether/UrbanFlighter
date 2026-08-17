import type { FlowField2DResponse } from '../api';
import { type ViewMode } from '../appModel';
import type { EnergyGraphScale } from './EnergyGraph';
import FlightHud from './FlightHud';
import HudLegend from './HudLegend';
import type { Telemetry } from './TopDownGame';
import UrbanFlowGymPanel from './UrbanFlowGymPanel';
import type { DroneControlPreset } from '../simulation/flight3dControls';

interface HudPanelProps {
  viewMode: ViewMode;
  flow: FlowField2DResponse | null;
  selectedLocation: { lat: number; lon: number };
  worldLoading: boolean;
  telemetry: Telemetry;
  energyHistory: number[];
  showFlowAnimation: boolean;
  showLidar: boolean;
  followCamera: boolean;
  controlPreset: DroneControlPreset;
  presentationMode: boolean;
  energyGraphScale: EnergyGraphScale;
  onClose: () => void;
  onShowFlowAnimationChange: (enabled: boolean) => void;
  onShowLidarChange: (enabled: boolean) => void;
  onFollowCameraChange: (enabled: boolean) => void;
  onControlPresetChange: (preset: DroneControlPreset) => void;
  onPresentationModeChange: (enabled: boolean) => void;
  onEnergyGraphScaleChange: (scale: EnergyGraphScale) => void;
}

export default function HudPanel({
  viewMode,
  flow,
  selectedLocation,
  worldLoading,
  telemetry,
  energyHistory,
  showFlowAnimation,
  showLidar,
  followCamera,
  controlPreset,
  presentationMode,
  energyGraphScale,
  onClose,
  onShowFlowAnimationChange,
  onShowLidarChange,
  onFollowCameraChange,
  onControlPresetChange,
  onPresentationModeChange,
  onEnergyGraphScaleChange,
}: HudPanelProps) {
  return (
    <aside className="panel panel-hud">
      <div className="panel-window-title panel-window-title--draggable" title="Drag to move window">
        <span>Telemetry / Controls <small>⠿ Move</small></span>
        <button type="button" onPointerDown={(event) => event.stopPropagation()} onClick={onClose} aria-label="Hide Telemetry and Controls window">×</button>
      </div>
      <div className="panel-eyebrow">Aero Command</div>
      <label className="flow-toggle">
        <input type="checkbox" checked={showFlowAnimation} onChange={(event) => onShowFlowAnimationChange(event.target.checked)} />
        <span>{viewMode === '2d' ? 'Flow Animation' : 'Wind Layer'}</span>
      </label>
      {viewMode === '3d' && (
        <>
          <label className="flow-toggle">
            <span>Camera mode</span>
            <select
              value={followCamera ? 'chase' : 'orbit'}
              onChange={(event) => onFollowCameraChange(event.target.value === 'chase')}
            >
              <option value="chase">Chase</option>
              <option value="orbit">Orbit / inspect</option>
            </select>
          </label>
          <label className="flow-toggle">
            <span>3D controls</span>
            <select
              value={controlPreset}
              onChange={(event) => onControlPresetChange(event.target.value as DroneControlPreset)}
            >
              <option value="arcade">Arcade: A/D strafe</option>
              <option value="pilot">Pilot: A/D yaw</option>
            </select>
          </label>
          <label className="flow-toggle">
            <input
              type="checkbox"
              checked={presentationMode}
              onChange={(event) => onPresentationModeChange(event.target.checked)}
            />
            <span>Presentation-only city dressing</span>
          </label>
        </>
      )}
      <label className="flow-toggle">
        <input type="checkbox" checked={showLidar} onChange={(event) => onShowLidarChange(event.target.checked)} />
        <span>{viewMode === '3d' ? '3D LiDAR + ground returns' : '2D LiDAR returns'}</span>
      </label>
      <UrbanFlowGymPanel
        flow={flow}
        selectedLocation={selectedLocation}
        worldLoading={worldLoading}
      />
      <FlightHud
        flow={flow}
        telemetry={telemetry}
        energyHistory={energyHistory}
        viewMode={viewMode}
        energyGraphScale={energyGraphScale}
        onEnergyGraphScaleChange={onEnergyGraphScaleChange}
      />
      <HudLegend viewMode={viewMode} flow={flow} />
    </aside>
  );
}
