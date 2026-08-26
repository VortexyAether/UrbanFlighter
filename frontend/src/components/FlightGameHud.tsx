import type { AircraftMetrics } from './Aircraft';
import type { UrbanDressingLayout } from '../presentation/urbanDressing';
import type { FreeFlightBeacon } from '../presentation/freeFlightBeacon';
import { DRONE_SCALE_CONTRACT } from '../simulation/droneScale';
import {
  flight3DControlHint,
  type DroneControlPreset,
  type FlightCameraMode,
} from '../simulation/flight3dControls';

interface FlightGameHudProps {
  metrics: AircraftMetrics | null;
  safetyRadiusM: number;
  verticalSafetyClearanceM: number;
  controlPreset: DroneControlPreset;
  cameraMode: FlightCameraMode;
  presentationMode: boolean;
  dressing: UrbanDressingLayout;
  beacon: FreeFlightBeacon;
  true3DOverlayStatus?: 'loading' | 'ready' | 'error';
  onControlPresetChange: (preset: DroneControlPreset) => void;
  onCameraModeChange: (mode: FlightCameraMode) => void;
  onPresentationModeChange: (enabled: boolean) => void;
}

export default function FlightGameHud({
  metrics,
  safetyRadiusM,
  verticalSafetyClearanceM,
  controlPreset,
  cameraMode,
  presentationMode,
  dressing,
  beacon,
  true3DOverlayStatus,
  onControlPresetChange,
  onCameraModeChange,
  onPresentationModeChange,
}: FlightGameHudProps) {
  const distanceToBeacon = metrics
    ? Math.hypot(
        metrics.position.x - beacon.x,
        metrics.position.y - beacon.altitude,
        metrics.position.z + beacon.y,
      )
    : null;
  return (
    <section className="flight-game-hud" aria-label="3D flight game controls and presentation status">
      <div className="flight-game-hud__readouts">
        <span><small>ALT</small>{metrics ? `${metrics.position.y.toFixed(1)} m` : '--'}</span>
        <span><small>SPD</small>{metrics ? `${metrics.velocity.length().toFixed(1)} m/s` : '--'}</span>
        <span><small>VAIR</small>{metrics ? `${metrics.energyMetrics.relativeAirSpeed.toFixed(1)} m/s` : '--'}</span>
        <span><small>DRAG</small>{metrics ? `${metrics.energyMetrics.dragForceN.toFixed(1)} N` : '--'}</span>
        <span><small>FLOW</small>{metrics ? metrics.energyMetrics.flowType : '--'}</span>
        <span className="flight-game-hud__beacon"><small>FREE-FLIGHT BEACON</small>{distanceToBeacon === null ? '--' : `${distanceToBeacon.toFixed(0)} m`}</span>
        {true3DOverlayStatus && (
          <span className={`flight-game-hud__uvw flight-game-hud__uvw--${true3DOverlayStatus}`}>
            <small>U/V/W OVERLAY</small>{true3DOverlayStatus.toUpperCase()}
          </span>
        )}
      </div>
      <div className="flight-game-hud__controls">
        <label>
          <span>Control preset</span>
          <select
            value={controlPreset}
            onChange={(event) => onControlPresetChange(event.target.value as DroneControlPreset)}
          >
            <option value="arcade">Arcade strafe</option>
            <option value="pilot">Pilot yaw</option>
          </select>
        </label>
        <button
          type="button"
          onClick={() => onCameraModeChange(cameraMode === 'chase' ? 'orbit' : 'chase')}
          aria-pressed={cameraMode === 'orbit'}
        >
          Camera: {cameraMode === 'chase' ? 'Chase' : 'Orbit'} <kbd>C</kbd>
        </button>
        <button
          type="button"
          onClick={() => onPresentationModeChange(!presentationMode)}
          aria-pressed={presentationMode}
        >
          City dressing: {presentationMode ? 'On' : 'Off'}
        </button>
      </div>
      <p className="flight-game-hud__hint">
        {flight3DControlHint(controlPreset)}
        {cameraMode === 'chase'
          ? ' · scroll zoom chase'
          : ' · drag orbit / scroll zoom / right-drag pan'}
      </p>
      <p className="flight-game-hud__contract">
        <strong>SCENERY ONLY</strong> — {dressing.trees.length} trees, {dressing.streetlights.length} lamps and rooftop/road detail are non-physical: no wind, collision, LiDAR, rolling-map, Gym, observation, or reward effect.
      </p>
      <p className="flight-game-hud__scale">
        Visual quad {DRONE_SCALE_CONTRACT.visualBody.spanM.toFixed(2)} m span · research safety {safetyRadiusM.toFixed(2)} m horizontal / {verticalSafetyClearanceM.toFixed(2)} m roof clearance{presentationMode ? '' : ' (shown)'}.
        Drag is quadratic air-relative; stick-off follows local wind. Not blade-element / not NS.
      </p>
    </section>
  );
}
