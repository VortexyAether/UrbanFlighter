export const SIM_RADAR_MODEL_ID = 'sim-range-doppler-proxy-v1';

export interface SimRadarConfig {
  rayCount: number;
  halfFovDeg: number;
  maxRangeM: number;
}

export const DEFAULT_SIM_RADAR_CONFIG: SimRadarConfig = {
  rayCount: 8,
  halfFovDeg: 60,
  maxRangeM: 40,
};

export function simRadarLocalAnglesRad(config: SimRadarConfig = DEFAULT_SIM_RADAR_CONFIG): number[] {
  if (config.rayCount <= 1) return [0];
  const half = (config.halfFovDeg * Math.PI) / 180;
  return Array.from({ length: config.rayCount }, (_, index) => {
    const t = index / (config.rayCount - 1);
    return -half + t * 2 * half;
  });
}

export function simulatedRadarRangeRate(
  groundVelocity: { x: number; y: number },
  worldAngleRad: number,
): number {
  return -(groundVelocity.x * Math.cos(worldAngleRad) + groundVelocity.y * Math.sin(worldAngleRad));
}
