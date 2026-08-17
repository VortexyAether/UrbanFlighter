export type DroneControlPreset = 'arcade' | 'pilot';
export type FlightCameraMode = 'chase' | 'orbit';

export interface Flight3DCommand {
  forward: number;
  strafe: number;
  yaw: number;
  lift: number;
  boost: boolean;
  brake: boolean;
}

export const FLIGHT_3D_CONTROL_CODES = new Set([
  'KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE',
  'KeyR', 'KeyF', 'Space', 'ShiftLeft', 'ShiftRight',
  'ArrowUp', 'ArrowDown',
]);

function axis(keys: ReadonlySet<string>, positive: string, negative: string) {
  return Number(keys.has(positive)) - Number(keys.has(negative));
}

export function mapFlight3DControls(
  keys: ReadonlySet<string>,
  preset: DroneControlPreset,
): Flight3DCommand {
  const aD = axis(keys, 'KeyD', 'KeyA');
  const qE = axis(keys, 'KeyE', 'KeyQ');
  return {
    forward: axis(keys, 'KeyW', 'KeyS'),
    strafe: preset === 'arcade' ? aD : qE,
    yaw: preset === 'arcade' ? qE : -aD,
    lift: Number(keys.has('Space') || keys.has('ArrowUp'))
      - Number(keys.has('ShiftLeft') || keys.has('ShiftRight') || keys.has('ArrowDown')),
    boost: keys.has('KeyR'),
    brake: keys.has('KeyF'),
  };
}

export function flight3DControlHint(preset: DroneControlPreset) {
  return preset === 'arcade'
    ? 'W/S drive · A/D strafe · Q/E yaw · Space/Shift altitude · R boost · F brake · C camera (Chase/Orbit)'
    : 'W/S drive · A/D yaw · Q/E strafe · Space/Shift altitude · R boost · F brake · C camera (Chase/Orbit)';
}

export function toggleFlightCameraMode(mode: FlightCameraMode): FlightCameraMode {
  return mode === 'chase' ? 'orbit' : 'chase';
}
