const FULL_TURN_DEG = 360;

export function normalizeBearingDeg(value: number) {
  if (!Number.isFinite(value)) return 0;
  return ((value % FULL_TURN_DEG) + FULL_TURN_DEG) % FULL_TURN_DEG;
}

/** Converts an east/north vector into a compass flow-to bearing. */
export function eastNorthToCompassBearingDeg(east: number, north: number) {
  if (!Number.isFinite(east) || !Number.isFinite(north) || Math.hypot(east, north) < 1e-9) {
    return 0;
  }
  return normalizeBearingDeg((Math.atan2(east, north) * 180) / Math.PI);
}

/** 2D map angles are mathematical: zero east, positive counter-clockwise. */
export function mapAngleToCompassBearingDeg(angleRad: number) {
  if (!Number.isFinite(angleRad)) return 0;
  return normalizeBearingDeg(90 - (angleRad * 180) / Math.PI);
}

/** Three.js uses -Z as north in the shared city coordinate mapping. */
export function sceneVectorToCompassBearingDeg(x: number, z: number) {
  return eastNorthToCompassBearingDeg(x, -z);
}
