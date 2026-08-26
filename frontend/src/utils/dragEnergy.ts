import {
  QUAD_AIR_DRAG,
  evaluatePhysicalDragPower,
  type AirVector3,
} from '../simulation/quadraticAirDrag';

export type FlowType = 'COUNTER' | 'CROSS' | 'TAIL';

export interface VectorLike {
  x: number;
  y?: number;
  z?: number;
}

export interface DragEnergyMetrics {
  consumptionRate: number;
  windAlignment: number;
  flowType: FlowType;
  efficiency: number;
  relativeAirSpeed: number;
  dragForceN: number;
  dragPowerW: number;
  inducedPowerW: number;
  climbPowerW: number;
  totalPowerW: number;
  modelId: string;
}

export interface DragEnergyConfig {
  airDensityKgM3?: number;
  dragCoefficient?: number;
  frontalAreaM2?: number;
  vehicleWeightN?: number;
  rotorSpanM?: number;
  hoverPowerW?: number;
  sensorPowerW?: number;
  inducedPowerW?: number;
  energyUnitScale?: number;
  minCruiseSpeedMps?: number;
  optimalCruiseSpeedMps?: number;
}

function componentY(vec: VectorLike) {
  return vec.y ?? 0;
}

function componentZ(vec: VectorLike) {
  return vec.z ?? 0;
}

function magnitude(vec: VectorLike) {
  return Math.hypot(vec.x, componentY(vec), componentZ(vec));
}

function toAirVector(vec: VectorLike): AirVector3 {
  return { x: vec.x, y: componentY(vec), z: componentZ(vec) };
}

function dot(a: VectorLike, b: VectorLike) {
  return a.x * b.x + componentY(a) * componentY(b) + componentZ(a) * componentZ(b);
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

export function getFlowType(alignmentAngle: number): FlowType {
  if (alignmentAngle > 120) return 'COUNTER';
  if (alignmentAngle < 60) return 'TAIL';
  return 'CROSS';
}

export function calculateWindAlignment(groundVelocity: VectorLike, windVelocity: VectorLike): number {
  const groundSpeed = magnitude(groundVelocity);
  const windSpeed = magnitude(windVelocity);
  if (groundSpeed < 0.1 || windSpeed < 0.1) {
    return 90;
  }

  const cosine = clamp(dot(groundVelocity, windVelocity) / (groundSpeed * windSpeed), -1, 1);
  return (Math.acos(cosine) * 180) / Math.PI;
}

export function calculateDragEnergy(
  groundVelocity: VectorLike,
  windVelocity: VectorLike,
  config: DragEnergyConfig = {},
): DragEnergyMetrics {
  const energyUnitScale = config.energyUnitScale ?? QUAD_AIR_DRAG.energyUnitScale;
  const physics = evaluatePhysicalDragPower(toAirVector(groundVelocity), toAirVector(windVelocity), {
    ...QUAD_AIR_DRAG,
    airDensityKgM3: config.airDensityKgM3 ?? QUAD_AIR_DRAG.airDensityKgM3,
    dragCoefficient: config.dragCoefficient ?? QUAD_AIR_DRAG.dragCoefficient,
    frontalAreaM2: config.frontalAreaM2 ?? QUAD_AIR_DRAG.frontalAreaM2,
    massKg: config.vehicleWeightN
      ? config.vehicleWeightN / QUAD_AIR_DRAG.gravityMps2
      : QUAD_AIR_DRAG.massKg,
    sensorPowerW: config.sensorPowerW ?? QUAD_AIR_DRAG.sensorPowerW,
    energyUnitScale,
  });
  const alignment = calculateWindAlignment(groundVelocity, windVelocity);
  const flowType = getFlowType(alignment);
  const cruiseRef = config.optimalCruiseSpeedMps ?? 11;
  const cruiseError = Math.abs(physics.relativeAirSpeed - cruiseRef) / cruiseRef;
  const efficiency = clamp(1 - cruiseError * 0.72 - Math.max(0, alignment - 120) / 240, 0, 1);

  return {
    consumptionRate: physics.totalPowerW * energyUnitScale,
    windAlignment: alignment,
    flowType,
    efficiency,
    relativeAirSpeed: physics.relativeAirSpeed,
    dragForceN: physics.dragForceN,
    dragPowerW: physics.parasitePowerW,
    inducedPowerW: physics.inducedPowerW,
    climbPowerW: physics.climbPowerW,
    totalPowerW: physics.totalPowerW,
    modelId: QUAD_AIR_DRAG.modelId,
  };
}
