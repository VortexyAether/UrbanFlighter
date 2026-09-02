import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { calculateWindDrag, type WindDragInfo } from '../utils/windDrag';
import { calculateEnergyConsumption, type EnergyMetrics } from '../utils/energySystem';
import type { BuildingData, FlowField2DResponse } from '../api';
import { sampleFlowField2D } from '../utils/flowFieldSampling';
import LidarSensorVisualization from './LidarSensorVisualization';
import CompactQuadcopter from './CompactQuadcopter';
import {
    DEFAULT_LIDAR_CONFIG,
    createLidarLocalDirections,
    scanLidar,
    summarizeLidarScan,
    type LidarScan,
    type LidarTelemetry,
} from '../sensors/lidar';
import {
    consumeFlight3DFixedSteps,
    stepFlight3DMotion,
    type Flight3DBounds,
    type Flight3DObstacle,
    type Flight3DState,
    type Flight3DVector,
} from '../simulation/flight3dMotion';
import {
    FLIGHT_3D_CONTROL_CODES,
    mapFlight3DControls,
    type DroneControlPreset,
} from '../simulation/flight3dControls';

interface AircraftProps {
    globalWindSpeed: number;
    globalWindDir: number;
    buildings: BuildingData[];
    flow?: FlowField2DResponse | null;
    bounds: Flight3DBounds;
    flightObstacles: Flight3DObstacle[];
    spawnPosition: Flight3DVector;
    worldIdentity: string;
    safetyRadiusM: number;
    verticalSafetyClearanceM: number;
    controlPreset: DroneControlPreset;
    buildingCollisionMeshes: THREE.Object3D[];
    lidarVisible: boolean;
    lidarVisualizationVisible?: boolean;
    showSafetyEnvelope?: boolean;
    visualScale?: number;
    motionTimeScale?: number;
    onMetricsUpdate?: (metrics: AircraftMetrics) => void;
    onLidarUpdate?: (telemetry: LidarTelemetry | null) => void;
}

export interface AircraftMetrics {
    position: THREE.Vector3;
    velocity: THREE.Vector3;
    windSpeed: number;
    windDirection: THREE.Vector3;
    energy: number;
    energyMetrics: EnergyMetrics;
    yaw: number;
    pitch: number;
    roll: number;
    lidar: LidarTelemetry | null;
}

const MAX_HORIZONTAL_SPEED = 36;
const MAX_VERTICAL_SPEED = 17;
// Keep the 600-ray physical-building/ground scan bounded at about 6.7 Hz.
const LIDAR_INTERVAL_S = 0.15;

function stateAt(position: Flight3DVector): Flight3DState {
    return {
        position: { ...position },
        velocity: { x: 0, y: 0, z: 0 },
        yaw: 0,
    };
}

function vector3(value: Flight3DVector) {
    return new THREE.Vector3(value.x, value.y, value.z);
}

function isInteractiveTarget(target: EventTarget | null) {
    if (!(target instanceof Element)) return false;
    if (target instanceof HTMLElement && target.isContentEditable) return true;
    return target.closest([
        'a', 'area[href]', 'button', 'input', 'label', 'select', 'textarea',
        'summary', 'iframe', 'object', 'embed', 'audio[controls]', 'video[controls]',
        '[contenteditable]:not([contenteditable="false"])',
        '[role="button"]', '[role="link"]', '[role="checkbox"]', '[role="radio"]',
        '[role="switch"]', '[role="textbox"]', '[role="combobox"]',
        '[role="slider"]', '[role="spinbutton"]', '[role="menuitem"]',
        '[tabindex]:not([tabindex="-1"])',
    ].join(',')) !== null;
}

const Aircraft: React.FC<AircraftProps> = ({
    globalWindSpeed,
    globalWindDir,
    buildings,
    flow,
    bounds,
    flightObstacles,
    spawnPosition,
    worldIdentity,
    safetyRadiusM,
    verticalSafetyClearanceM,
    controlPreset,
    buildingCollisionMeshes,
    lidarVisible,
    lidarVisualizationVisible = lidarVisible,
    showSafetyEnvelope = false,
    visualScale = 1,
    motionTimeScale = 1,
    onMetricsUpdate,
    onLidarUpdate,
}) => {
    const groupRef = useRef<THREE.Group>(null);
    const stateRef = useRef<Flight3DState>(stateAt(spawnPosition));
    const physicsAccumulatorRef = useRef(0);
    const totalEnergyUsedRef = useRef(0);
    const pitchRef = useRef(0);
    const rollRef = useRef(0);
    const [lidarScan, setLidarScan] = useState<LidarScan | null>(null);
    const lidarClockRef = useRef(0);
    const lidarNeedsImmediateScanRef = useRef(true);
    const lidarDisabledClearedRef = useRef(false);
    const lidarTelemetryRef = useRef<LidarTelemetry | null>(null);
    const onLidarUpdateRef = useRef(onLidarUpdate);
    const keysPressed = useRef<Set<string>>(new Set());
    const lidarDirections = useMemo(
        () => lidarVisible ? createLidarLocalDirections(DEFAULT_LIDAR_CONFIG) : [],
        [lidarVisible],
    );

    useEffect(() => {
        onLidarUpdateRef.current = onLidarUpdate;
    }, [onLidarUpdate]);

    useEffect(() => {
        stateRef.current = stateAt(spawnPosition);
        physicsAccumulatorRef.current = 0;
        totalEnergyUsedRef.current = 0;
        pitchRef.current = 0;
        rollRef.current = 0;
        lidarClockRef.current = 0;
        lidarNeedsImmediateScanRef.current = true;
        lidarTelemetryRef.current = null;
        keysPressed.current.clear();
        queueMicrotask(() => setLidarScan((current) => current === null ? current : null));
        onLidarUpdateRef.current?.(null);
        if (groupRef.current) {
            groupRef.current.position.copy(vector3(spawnPosition));
            groupRef.current.rotation.set(0, 0, 0);
        }
    }, [spawnPosition, worldIdentity]);

    useEffect(() => {
        lidarClockRef.current = 0;
        lidarNeedsImmediateScanRef.current = lidarVisible;
        if (lidarVisible) {
            lidarDisabledClearedRef.current = false;
            return;
        }
        // A disabled sensor owns no live scan data. Decorative instances never
        // enter buildingCollisionMeshes, regardless of this display toggle.
        lidarTelemetryRef.current = null;
        queueMicrotask(() => setLidarScan((current) => current === null ? current : null));
        if (!lidarDisabledClearedRef.current) {
            lidarDisabledClearedRef.current = true;
            onLidarUpdateRef.current?.(null);
        }
    }, [lidarVisible]);

    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (isInteractiveTarget(event.target) || !FLIGHT_3D_CONTROL_CODES.has(event.code)) return;
            event.preventDefault();
            keysPressed.current.add(event.code);
        };
        const handleKeyUp = (event: KeyboardEvent) => {
            if (!FLIGHT_3D_CONTROL_CODES.has(event.code)) return;
            if (!isInteractiveTarget(event.target)) event.preventDefault();
            keysPressed.current.delete(event.code);
        };
        const clearControls = () => keysPressed.current.clear();
        const handleVisibilityChange = () => {
            if (document.visibilityState !== 'visible') clearControls();
        };

        window.addEventListener('keydown', handleKeyDown);
        window.addEventListener('keyup', handleKeyUp);
        window.addEventListener('blur', clearControls);
        document.addEventListener('visibilitychange', handleVisibilityChange);
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
            window.removeEventListener('blur', clearControls);
            document.removeEventListener('visibilitychange', handleVisibilityChange);
            clearControls();
        };
    }, []);

    const windAt = (position: Flight3DVector): WindDragInfo => {
        const sampledGridWind = sampleFlowField2D(flow ?? null, position.x, position.z);
        if (sampledGridWind) {
            return {
                speed: sampledGridWind.length(),
                direction: sampledGridWind.length() > 0.01
                    ? sampledGridWind.clone().normalize()
                    : new THREE.Vector3(0, 0, 0),
                force: sampledGridWind,
            };
        }
        const fallbackWind = calculateWindDrag(
            new THREE.Vector3(position.x, position.y, -position.z),
            globalWindSpeed,
            globalWindDir,
            buildings,
        );
        return {
            speed: fallbackWind.speed,
            direction: new THREE.Vector3(fallbackWind.direction.x, 0, -fallbackWind.direction.z),
            force: new THREE.Vector3(fallbackWind.force.x, 0, -fallbackWind.force.z),
        };
    };

    useFrame((_, frameDelta) => {
        if (!groupRef.current) return;
        const command = mapFlight3DControls(keysPressed.current, controlPreset);
        let windInfo = windAt(stateRef.current.position);
        let energyMetrics = calculateEnergyConsumption(vector3(stateRef.current.velocity), windInfo);
        const fixedStep = consumeFlight3DFixedSteps(
            physicsAccumulatorRef.current,
            frameDelta * Math.max(0.1, motionTimeScale),
            (deltaSeconds) => {
                windInfo = windAt(stateRef.current.position);
                stateRef.current = stepFlight3DMotion(
                    stateRef.current,
                    command,
                    { x: windInfo.force.x, y: windInfo.force.y, z: windInfo.force.z },
                    flightObstacles,
                    bounds,
                    { safetyRadiusM, verticalSafetyClearanceM },
                    deltaSeconds,
                );
                energyMetrics = calculateEnergyConsumption(vector3(stateRef.current.velocity), windInfo);
                totalEnergyUsedRef.current += energyMetrics.consumptionRate * deltaSeconds;
            },
        );
        physicsAccumulatorRef.current = fixedStep.accumulatorSeconds;

        const state = stateRef.current;
        const position = vector3(state.position);
        const velocity = vector3(state.velocity);
        windInfo = windAt(state.position);
        energyMetrics = calculateEnergyConsumption(velocity, windInfo);
        groupRef.current.position.copy(position);

        const forward = new THREE.Vector3(-Math.sin(state.yaw), 0, -Math.cos(state.yaw));
        const right = new THREE.Vector3(Math.cos(state.yaw), 0, -Math.sin(state.yaw));
        const forwardSpeed = velocity.dot(forward);
        const sideSpeed = velocity.dot(right);
        const visualDelta = Math.min(Math.max(frameDelta, 0), 0.05);
        const targetPitch = THREE.MathUtils.clamp(
            (velocity.y / MAX_VERTICAL_SPEED) * 0.16 - (forwardSpeed / MAX_HORIZONTAL_SPEED) * 0.28,
            -0.42,
            0.42,
        );
        const targetRoll = THREE.MathUtils.clamp(
            -(sideSpeed / MAX_HORIZONTAL_SPEED) * 0.42 - command.strafe * 0.18 - command.yaw * 0.1,
            -0.52,
            0.52,
        );
        pitchRef.current = THREE.MathUtils.lerp(pitchRef.current, targetPitch, 1 - Math.exp(-7 * visualDelta));
        rollRef.current = THREE.MathUtils.lerp(rollRef.current, targetRoll, 1 - Math.exp(-9 * visualDelta));
        groupRef.current.rotation.set(pitchRef.current, state.yaw, rollRef.current, 'YXZ');

        if (lidarVisible) lidarClockRef.current += fixedStep.simulatedSeconds;
        if (lidarVisible && (lidarNeedsImmediateScanRef.current || lidarClockRef.current >= LIDAR_INTERVAL_S)) {
            lidarClockRef.current %= LIDAR_INTERVAL_S;
            lidarNeedsImmediateScanRef.current = false;
            const nextScan = scanLidar(
                {
                    position,
                    yaw: state.yaw,
                    pitch: pitchRef.current,
                    roll: rollRef.current,
                },
                buildingCollisionMeshes,
                DEFAULT_LIDAR_CONFIG,
                lidarDirections,
            );
            const nextTelemetry = summarizeLidarScan(nextScan, {
                x: position.x,
                y: position.y,
                z: position.z,
                yaw: state.yaw,
                pitch: pitchRef.current,
                roll: rollRef.current,
            });
            lidarTelemetryRef.current = nextTelemetry;
            setLidarScan(nextScan);
            onLidarUpdateRef.current?.(nextTelemetry);
        }

        onMetricsUpdate?.({
            position,
            velocity,
            windSpeed: windInfo.speed,
            windDirection: windInfo.direction.clone(),
            energy: totalEnergyUsedRef.current,
            energyMetrics,
            yaw: state.yaw,
            pitch: pitchRef.current,
            roll: rollRef.current,
            lidar: lidarVisible ? lidarTelemetryRef.current : null,
        });
    });

    return (
        <>
            <group ref={groupRef}>
                <CompactQuadcopter
                    safetyRadiusM={safetyRadiusM}
                    verticalSafetyClearanceM={verticalSafetyClearanceM}
                    showSafetyEnvelope={showSafetyEnvelope}
                    visualScale={visualScale}
                />
            </group>
            <LidarSensorVisualization
                scan={lidarVisualizationVisible ? lidarScan : null}
                visible={lidarVisualizationVisible}
            />
        </>
    );
};

export default Aircraft;
