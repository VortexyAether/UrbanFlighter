import {
  DEFAULT_LAT,
  DEFAULT_LON,
  modeHelp,
  type SimulationMode,
  type ViewMode,
} from '../appModel';
import LocationPicker from './LocationPicker';

interface RegionPanelProps {
  location: { lat: number; lon: number };
  simulationMode: SimulationMode;
  viewMode: ViewMode;
  backendLabel: string;
  backendDetail: string;
  loading: boolean;
  status: string;
  isTrue3DMode: boolean;
  onClose: () => void;
  onLocationSelect: (lat: number, lon: number) => void;
  onPreset: (lat: number, lon: number) => void;
  onModeSelect: (mode: SimulationMode) => void;
  onReload: () => void;
}

export default function RegionPanel({
  location,
  simulationMode,
  viewMode,
  backendLabel,
  backendDetail,
  loading,
  status,
  isTrue3DMode,
  onClose,
  onLocationSelect,
  onPreset,
  onModeSelect,
  onReload,
}: RegionPanelProps) {
  const presetsDisabled = isTrue3DMode;

  return (
    <aside className="panel panel-map">
      <div className="panel-window-title panel-window-title--draggable" title="Drag to move window">
        <span>Map / Mode <small>⠿ Move</small></span>
        <button type="button" onPointerDown={(event) => event.stopPropagation()} onClick={onClose} aria-label="Hide Map and Mode window">×</button>
      </div>
      <div className="panel-eyebrow">Region Selector</div>
      <h1>Urban Flighter {viewMode.toUpperCase()}</h1>
      <p className="mode-help">{modeHelp(simulationMode)}</p>
      <div className="mode-selector" aria-label="Simulation mode">
        <ModeOption mode="2d" active={simulationMode === '2d'} onModeSelect={onModeSelect} title="2D" caption="top-down sensor flight" />
        <ModeOption mode="3d" active={simulationMode === '3d'} onModeSelect={onModeSelect} title="3D Lite" caption="Three.js + grid wind" />
        <ModeOption mode="true3d" active={simulationMode === 'true3d'} onModeSelect={onModeSelect} title="True 3D Wind" caption="u/v/w streamlines" />
      </div>
      <div className="map-container-wrapper">
        <LocationPicker initialLat={location.lat} initialLon={location.lon} onLocationSelect={onLocationSelect} />
      </div>
      <div className="city-presets">
        <button type="button" onClick={() => onPreset(40.7128, -74.006)} disabled={presetsDisabled}>NYC</button>
        <button type="button" onClick={() => onPreset(48.8566, 2.3522)} disabled={presetsDisabled}>Paris</button>
        <button type="button" onClick={() => onPreset(35.6762, 139.6503)} disabled={presetsDisabled}>Tokyo</button>
        <button type="button" onClick={() => onPreset(DEFAULT_LAT, DEFAULT_LON)} disabled={presetsDisabled}>Inha</button>
      </div>
      <div className="coords">
        <div>
          <span>Lat</span>
          <strong>{location.lat.toFixed(6)}</strong>
        </div>
        <div>
          <span>Lon</span>
          <strong>{location.lon.toFixed(6)}</strong>
        </div>
      </div>
      <div className="source-card">
        <span>Backend</span>
        <strong>{backendLabel}</strong>
        <p>{backendDetail}</p>
      </div>
      <button type="button" className="reload-btn" onClick={onReload} disabled={loading}>
        {loading ? 'Solving...' : 'Reload Flow'}
      </button>
      <p className="status-bar">{status}</p>
    </aside>
  );
}

interface ModeOptionProps {
  mode: SimulationMode;
  active: boolean;
  title: string;
  caption: string;
  onModeSelect: (mode: SimulationMode) => void;
}

function ModeOption({ mode, active, title, caption, onModeSelect }: ModeOptionProps) {
  return (
    <label className={active ? 'active' : ''}>
      <input type="radio" name="simulationMode" checked={active} onChange={() => onModeSelect(mode)} />
      <span>
        <strong>{title}</strong>
        <small>{caption}</small>
      </span>
    </label>
  );
}
