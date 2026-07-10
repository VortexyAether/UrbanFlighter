import type { FlowField2DResponse } from '../api';
import { type ViewMode } from '../appModel';
import type { EnergyGraphScale } from './EnergyGraph';
import FlightHud from './FlightHud';
import HudLegend from './HudLegend';
import type { Telemetry } from './TopDownGame';

interface HudPanelProps {
  viewMode: ViewMode;
  flow: FlowField2DResponse | null;
  telemetry: Telemetry;
  energyHistory: number[];
  showFlowAnimation: boolean;
  showLidar: boolean;
  followCamera: boolean;
  energyGraphScale: EnergyGraphScale;
  onClose: () => void;
  onShowFlowAnimationChange: (enabled: boolean) => void;
  onShowLidarChange: (enabled: boolean) => void;
  onFollowCameraChange: (enabled: boolean) => void;
  onEnergyGraphScaleChange: (scale: EnergyGraphScale) => void;
}

export default function HudPanel({
  viewMode,
  flow,
  telemetry,
  energyHistory,
  showFlowAnimation,
  showLidar,
  followCamera,
  energyGraphScale,
  onClose,
  onShowFlowAnimationChange,
  onShowLidarChange,
  onFollowCameraChange,
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
        <label className="flow-toggle">
          <input type="checkbox" checked={followCamera} onChange={(event) => onFollowCameraChange(event.target.checked)} />
          <span>Camera Follow</span>
        </label>
      )}
      <label className="flow-toggle">
        <input type="checkbox" checked={showLidar} onChange={(event) => onShowLidarChange(event.target.checked)} />
        <span>{viewMode === '3d' ? '3D LiDAR + ground returns' : '2D LiDAR returns'}</span>
      </label>
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
