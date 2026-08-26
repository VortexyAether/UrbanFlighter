export interface AirVector3 {
  x: number;
  y: number;
  z: number;
}

export interface AirVector2 {
  x: number;
  y: number;
}

export interface QuadraticAirDragConfig {
  airDensityKgM3: number;
  dragCoefficient: number;
  frontalAreaM2: number;
  massKg: number;
  gravityMps2: number;
  rotorCount: number;
  propellerDiameterM: number;
  inducedPowerFactor: number;
  avionicsPowerW: number;
  sensorPowerW: number;
  energyUnitScale: number;
  linearAirDragPerS: number;
  modelId: string;
  honesty: string;
}

export const QUAD_AIR_DRAG: QuadraticAirDragConfig = Object.freeze({
  airDensityKgM3: 1.225,
  dragCoefficient: 1.05,
  frontalAreaM2: 0.18,
  massKg: 2.5,
  gravityMps2: 9.81,
  rotorCount: 4,
  propellerDiameterM: 0.15,
  inducedPowerFactor: 1.15,
  avionicsPowerW: 18,
  sensorPowerW: 8,
  energyUnitScale: 0.03,
  linearAirDragPerS: 0.28,
  modelId: 'quadratic-air-relative-v1',
  honesty: 'QUADRATIC AIR-RELATIVE DRAG · MOMENTUM-THEORY INDUCED · NOT BLADE-ELEMENT / NOT NS',
});

export function parasiteDragPerM(config: QuadraticAirDragConfig = QUAD_AIR_DRAG) {
  return (0.5 * config.airDensityKgM3 * config.dragCoefficient * config.frontalAreaM2)
    / Math.max(config.massKg, 1e-9);
}

export function rotorDiskAreaM2(config: QuadraticAirDragConfig = QUAD_AIR_DRAG) {
  const radius = config.propellerDiameterM * 0.5;
  return config.rotorCount * Math.PI * radius * radius;
}

export function hoverInducedVelocityMps(config: QuadraticAirDragConfig = QUAD_AIR_DRAG) {
  const weightN = config.massKg * config.gravityMps2;
  return Math.sqrt(weightN / (2 * config.airDensityKgM3 * Math.max(rotorDiskAreaM2(config), 1e-9)));
}

export function relativeAirVelocity3(ground: AirVector3, wind: AirVector3): AirVector3 {
  return {
    x: ground.x - wind.x,
    y: ground.y - wind.y,
    z: ground.z - wind.z,
  };
}

export function integrateQuadraticAirDrag3(
  ground: AirVector3,
  wind: AirVector3,
  dt: number,
  options: { kPerM?: number; linearPerS?: number } = {},
): AirVector3 {
  if (!Number.isFinite(dt) || dt <= 0) return { ...ground };
  const k = options.kPerM ?? parasiteDragPerM();
  const linear = options.linearPerS ?? QUAD_AIR_DRAG.linearAirDragPerS;
  const safeK = Number.isFinite(k) && k > 0 ? k : 0;
  const safeLinear = Number.isFinite(linear) && linear > 0 ? linear : 0;
  if (safeK === 0 && safeLinear === 0) return { ...ground };
  const air = relativeAirVelocity3(ground, wind);
  const airSpeed = Math.hypot(air.x, air.y, air.z);
  const denom = 1 + dt * (safeK * airSpeed + safeLinear);
  return {
    x: wind.x + air.x / denom,
    y: wind.y + air.y / denom,
    z: wind.z + air.z / denom,
  };
}

export function integrateQuadraticAirDrag2(
  ground: AirVector2,
  wind: AirVector2,
  dt: number,
  options: { kPerM?: number; linearPerS?: number } = {},
): AirVector2 {
  const next = integrateQuadraticAirDrag3(
    { x: ground.x, y: 0, z: ground.y },
    { x: wind.x, y: 0, z: wind.y },
    dt,
    options,
  );
  return { x: next.x, y: next.z };
}

export interface PhysicalDragPower {
  relativeAirSpeed: number;
  dragForceN: number;
  parasitePowerW: number;
  inducedPowerW: number;
  climbPowerW: number;
  totalPowerW: number;
  hoverInducedVelocityMps: number;
}

export function evaluatePhysicalDragPower(
  ground: AirVector3,
  wind: AirVector3,
  config: QuadraticAirDragConfig = QUAD_AIR_DRAG,
): PhysicalDragPower {
  const air = relativeAirVelocity3(ground, wind);
  const relativeAirSpeed = Math.hypot(air.x, air.y, air.z);
  const dragForceN = 0.5
    * config.airDensityKgM3
    * config.dragCoefficient
    * config.frontalAreaM2
    * relativeAirSpeed
    * relativeAirSpeed;
  const parasitePowerW = dragForceN * relativeAirSpeed;
  const weightN = config.massKg * config.gravityMps2;
  const inducedHover = hoverInducedVelocityMps(config);
  const inducedPowerHoverW = config.inducedPowerFactor * weightN * inducedHover;
  const inducedPowerW = inducedPowerHoverW
    / Math.sqrt(1 + (relativeAirSpeed / Math.max(inducedHover, 1e-6)) ** 2);
  const climbPowerW = Math.max(0, ground.y) * weightN;
  const totalPowerW = config.avionicsPowerW
    + config.sensorPowerW
    + parasitePowerW
    + inducedPowerW
    + climbPowerW;
  return {
    relativeAirSpeed,
    dragForceN,
    parasitePowerW,
    inducedPowerW,
    climbPowerW,
    totalPowerW,
    hoverInducedVelocityMps: inducedHover,
  };
}
