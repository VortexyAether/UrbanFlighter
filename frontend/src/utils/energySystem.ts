import * as THREE from 'three';
import type { WindDragInfo } from './windDrag';
import {
    calculateDragEnergy,
    calculateWindAlignment as calculateDragWindAlignment,
    getFlowType as getDragFlowType,
} from './dragEnergy';
import type { FlowType } from './dragEnergy';

/**
 * Energy System for Aircraft
 * Calculates energy consumption based on wind resistance
 */

export interface EnergyMetrics {
    consumptionRate: number;  // Energy units per second
    windAlignment: number;    // Angle between velocity and wind (0-180°)
    flowType: FlowType;
    efficiency: number;       // 0-1, higher is better
    relativeAirSpeed: number;
    dragForceN: number;
    dragPowerW: number;
    inducedPowerW: number;
    climbPowerW: number;
    totalPowerW: number;
    modelId: string;
}

/**
 * Calculate angle between aircraft velocity and wind direction
 * Returns angle in degrees (0-180)
 */
export function calculateWindAlignment(
    velocity: THREE.Vector3,
    windDirection: THREE.Vector3
): number {
    return calculateDragWindAlignment(velocity, windDirection);
}

/**
 * Determine flow type based on alignment angle
 */
export function getFlowType(alignmentAngle: number): FlowType {
    return getDragFlowType(alignmentAngle);
}

/**
 * Calculate energy consumption rate
 * 
 * Formula:
 * v_air = v_ground - w_local
 * F_para = 0.5 * rho * C_d * A * |v_air|^2
 * P_para = F_para * |v_air|
 * P_ind = momentum-theory hover induced / sqrt(1 + (|v_air|/v_h)^2)
 * Not blade-element and not Navier-Stokes.
 */
export function calculateEnergyConsumption(
    velocity: THREE.Vector3,
    windInfo: WindDragInfo
): EnergyMetrics {
    return calculateDragEnergy(velocity, windInfo.force);
}

/**
 * Update energy level based on consumption
 */
export function updateEnergy(
    currentEnergy: number,
    consumptionRate: number,
    deltaTime: number,
    maxEnergy: number = 100
): number {
    const newEnergy = currentEnergy - (consumptionRate * deltaTime);
    return THREE.MathUtils.clamp(newEnergy, 0, maxEnergy);
}
