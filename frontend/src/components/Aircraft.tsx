import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { calculateWindDrag } from '../utils/windDrag';
import { calculateEnergyConsumption, type EnergyMetrics } from '../utils/energySystem';
import type { BuildingData, FlowField2DResponse } from '../api';
import { sampleFlowField2D } from '../utils/flowFieldSampling';
import { isScenePointInsideBuilding } from '../geometry/buildingGeometry';
import LidarSensorVisualization from './LidarSensorVisualization';
import {
    DEFAULT_LIDAR_CONFIG,
    createLidarLocalDirections,
    scanLidar,
    summarizeLidarScan,
    type LidarScan,
    type LidarTelemetry,
} from '../sensors/lidar';

interface AircraftProps {
    globalWindSpeed: number;
    globalWindDir: number;
    buildings: BuildingData[];
    flow?: FlowField2DResponse | null;
    buildingCollisionMeshes: THREE.Object3D[];
    lidarVisible: boolean;
    onMetricsUpdate?: (metrics: AircraftMetrics) => void;
    onLidarUpdate?: (telemetry: LidarTelemetry) => void;
}

export interface AircraftMetrics {
    position: THREE.Vector3;
    velocity: THREE.Vector3;
    windSpeed: number;
    windDirection: THREE.Vector3;
    energy: number;
    energyMetrics: EnergyMetrics;
    yaw: number;   // Aircraft heading for camera
    pitch: number; // Aircraft pitch for camera
    roll: number;
    lidar: LidarTelemetry | null;
}

const MAX_HORIZONTAL_SPEED = 44;
const MAX_VERTICAL_SPEED = 18;
const FORWARD_ACCEL = 34;
const REVERSE_ACCEL = 18;
const VERTICAL_ACCEL = 26;
const HORIZONTAL_DRAG = 0.82;
const VERTICAL_DRAG = 1.15;
const TURN_SPEED = 1.65;
// Keep the 600-ray building-mesh scan bounded at about 6.7 Hz.
const LIDAR_INTERVAL_S = 0.15;
const DRONE_VERTICAL_CLEARANCE = 2;
const CONTROL_CODES = new Set([
    'KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE',
    'ArrowUp', 'ArrowDown', 'Space', 'ShiftLeft', 'ShiftRight',
]);

const Aircraft: React.FC<AircraftProps> = ({
    globalWindSpeed,
    globalWindDir,
    buildings,
    flow,
    buildingCollisionMeshes,
    lidarVisible,
    onMetricsUpdate,
    onLidarUpdate,
}) => {
    const groupRef = useRef<THREE.Group>(null);

    // Aircraft state
    const positionRef = useRef(new THREE.Vector3(0, 50, 0));
    const velocityRef = useRef(new THREE.Vector3(0, 0, 0));
    const totalEnergyUsedRef = useRef(0);
    const yawRef = useRef(0);
    const pitchRef = useRef(0);
    const rollRef = useRef(0);
    const [lidarScan, setLidarScan] = useState<LidarScan | null>(null);
    const lidarClockRef = useRef(0);
    const lidarTelemetryRef = useRef<LidarTelemetry | null>(null);
    const lidarDirections = useMemo(() => createLidarLocalDirections(DEFAULT_LIDAR_CONFIG), []);

    // Controls state
    const keysPressed = useRef<Set<string>>(new Set());

    useEffect(() => {
        const isTypingTarget = (target: EventTarget | null) => (
            target instanceof HTMLElement
            && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))
        );

        const handleKeyDown = (e: KeyboardEvent) => {
            if (isTypingTarget(e.target)) return;
            if (CONTROL_CODES.has(e.code)) e.preventDefault();
            keysPressed.current.add(e.code.toLowerCase());
        };

        const handleKeyUp = (e: KeyboardEvent) => {
            if (isTypingTarget(e.target)) return;
            if (CONTROL_CODES.has(e.code)) e.preventDefault();
            keysPressed.current.delete(e.code.toLowerCase());
        };

        const handleBlur = () => keysPressed.current.clear();

        window.addEventListener('keydown', handleKeyDown);
        window.addEventListener('keyup', handleKeyUp);
        window.addEventListener('blur', handleBlur);

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
            window.removeEventListener('blur', handleBlur);
        };
    }, []);

    const buildingAt = (candidate: THREE.Vector3) => buildings.some((building) => (
        isScenePointInsideBuilding(building, candidate, DRONE_VERTICAL_CLEARANCE)
    ));

    const segmentHitsBuilding = (start: THREE.Vector3, end: THREE.Vector3) => {
        const distance = start.distanceTo(end);
        const samples = Math.max(1, Math.ceil(distance));
        for (let index = 1; index <= samples; index += 1) {
            const alpha = index / samples;
            if (buildingAt(start.clone().lerp(end, alpha))) return true;
        }
        return false;
    };

    useFrame((_, delta) => {
        if (!groupRef.current) return;
        const dt = Math.min(delta, 0.05);
        const position = positionRef.current;
        const velocity = velocityRef.current;
        const previousPosition = position.clone();

        const forwardInput = Number(keysPressed.current.has('keyw')) - Number(keysPressed.current.has('keys'));
        const turnInput = Number(keysPressed.current.has('keya')) - Number(keysPressed.current.has('keyd'));
        const liftInput = Number(keysPressed.current.has('space') || keysPressed.current.has('keye') || keysPressed.current.has('arrowup'))
            - Number(keysPressed.current.has('shiftleft') || keysPressed.current.has('shiftright') || keysPressed.current.has('keyq') || keysPressed.current.has('arrowdown'));

        const horizontalSpeed = Math.hypot(velocity.x, velocity.z);
        yawRef.current += turnInput * TURN_SPEED * (0.45 + Math.min(horizontalSpeed / 28, 1)) * dt;

        const forward = new THREE.Vector3(-Math.sin(yawRef.current), 0, -Math.cos(yawRef.current)).normalize();
        const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();
        const acceleration = new THREE.Vector3();
        if (forwardInput > 0) acceleration.add(forward.clone().multiplyScalar(FORWARD_ACCEL * forwardInput));
        if (forwardInput < 0) acceleration.add(forward.clone().multiplyScalar(REVERSE_ACCEL * forwardInput));
        acceleration.y += liftInput * VERTICAL_ACCEL;

        velocity.add(acceleration.multiplyScalar(dt));
        const horizontalDamping = Math.exp(-HORIZONTAL_DRAG * dt);
        velocity.x *= horizontalDamping;
        velocity.z *= horizontalDamping;
        velocity.y *= Math.exp(-VERTICAL_DRAG * dt);

        const nextHorizontalSpeed = Math.hypot(velocity.x, velocity.z);
        if (nextHorizontalSpeed > MAX_HORIZONTAL_SPEED) {
            const scale = MAX_HORIZONTAL_SPEED / nextHorizontalSpeed;
            velocity.x *= scale;
            velocity.z *= scale;
        }
        velocity.y = THREE.MathUtils.clamp(velocity.y, -MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED);

        // Get wind force at current position. Prefer backend CFD-lite B grid when present;
        // fall back to the old local polygon heuristic if the grid has not loaded yet.
        const sampledGridWind = sampleFlowField2D(flow ?? null, position.x, position.z);
        const fallbackWind = sampledGridWind ? null : calculateWindDrag(
            new THREE.Vector3(position.x, position.y, -position.z),
            globalWindSpeed,
            globalWindDir,
            buildings,
        );
        const windInfo = sampledGridWind
            ? {
                speed: sampledGridWind.length(),
                direction: sampledGridWind.length() > 0.01 ? sampledGridWind.clone().normalize() : new THREE.Vector3(0, 0, 0),
                force: sampledGridWind,
            }
            : {
                speed: fallbackWind!.speed,
                direction: new THREE.Vector3(fallbackWind!.direction.x, 0, -fallbackWind!.direction.z),
                force: new THREE.Vector3(fallbackWind!.force.x, 0, -fallbackWind!.force.z),
            };

        // Apply wind force (subtle influence)
        const windForce = windInfo.force.clone().multiplyScalar(0.03);
        velocity.add(windForce.clone().multiplyScalar(dt));

        // Update position
        const proposedPosition = position.clone().add(velocity.clone().multiplyScalar(dt));
        position.copy(proposedPosition);

        // Boundary constraints (stay within domain)
        const maxDist = 1500;
        const dist2D = Math.sqrt(position.x ** 2 + position.z ** 2);
        if (dist2D > maxDist) {
            const angle = Math.atan2(position.z, position.x);
            position.x = Math.cos(angle) * maxDist;
            position.z = Math.sin(angle) * maxDist;
            velocity.multiplyScalar(0.5); // Slow down at boundary
        }

        // Height constraints
        position.y = THREE.MathUtils.clamp(position.y, 5, 300);
        if (position.y <= 5 || position.y >= 300) {
            velocity.y *= -0.3; // Bounce
        }

        if (segmentHitsBuilding(previousPosition, position)) {
            const intendedDelta = position.clone().sub(previousPosition);
            const resolvedPosition = previousPosition.clone();
            const resolveAxis = (axis: 'x' | 'y' | 'z') => {
                const candidate = resolvedPosition.clone();
                candidate[axis] += intendedDelta[axis];
                if (segmentHitsBuilding(resolvedPosition, candidate)) {
                    velocity[axis] = 0;
                } else {
                    resolvedPosition.copy(candidate);
                }
            };

            // Resolve independently so blocked motion loses its inward component while
            // unblocked components continue along the facade or over the roof.
            resolveAxis('x');
            resolveAxis('z');
            resolveAxis('y');
            position.copy(resolvedPosition);
        }

        // Calculate energy consumption and accumulate total used
        const energyMetrics = calculateEnergyConsumption(velocity, windInfo);
        const energyConsumed = energyMetrics.consumptionRate * dt;
        totalEnergyUsedRef.current += energyConsumed;

        // Update mesh position
        groupRef.current.position.copy(position);

        const forwardSpeed = velocity.dot(forward);
        const sideSpeed = velocity.dot(right);
        const targetPitch = THREE.MathUtils.clamp((velocity.y / MAX_VERTICAL_SPEED) * 0.34 - (forwardSpeed / MAX_HORIZONTAL_SPEED) * 0.18, -0.52, 0.52);
        const targetRoll = THREE.MathUtils.clamp(turnInput * 0.5 + (sideSpeed / MAX_HORIZONTAL_SPEED) * 0.55, -0.68, 0.68);
        pitchRef.current = THREE.MathUtils.lerp(pitchRef.current, targetPitch, 1 - Math.exp(-7 * dt));
        rollRef.current = THREE.MathUtils.lerp(rollRef.current, targetRoll, 1 - Math.exp(-9 * dt));

        groupRef.current.rotation.set(pitchRef.current, yawRef.current, rollRef.current, 'YXZ');

        lidarClockRef.current += dt;
        if (lidarClockRef.current >= LIDAR_INTERVAL_S) {
            lidarClockRef.current = 0;
            const nextScan = scanLidar(
                {
                    position,
                    yaw: yawRef.current,
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
                yaw: yawRef.current,
                pitch: pitchRef.current,
                roll: rollRef.current,
            });
            lidarTelemetryRef.current = nextTelemetry;
            setLidarScan(nextScan);
            onLidarUpdate?.(nextTelemetry);
        }

        // Send metrics to parent
        if (onMetricsUpdate) {
            onMetricsUpdate({
                position: position.clone(),
                velocity: velocity.clone(),
                windSpeed: windInfo.speed,
                windDirection: windInfo.direction.clone(),
                energy: totalEnergyUsedRef.current,
                energyMetrics,
                yaw: yawRef.current,
                pitch: pitchRef.current,
                roll: rollRef.current,
                lidar: lidarTelemetryRef.current,
            });
        }
    });

    return (
        <>
        <group ref={groupRef} scale={0.42}>
            {/* Compact sci-fi UAV silhouette */}
            <mesh position={[0, 0, -1.5]} rotation={[Math.PI / 2, 0, 0]}>
                <capsuleGeometry args={[1.6, 9.5, 8, 18]} />
                <meshStandardMaterial color="#e9ece8" metalness={0.55} roughness={0.28} />
            </mesh>

            {/* Sharp nose */}
            <mesh position={[0, 0, -8.2]} rotation={[-Math.PI / 2, 0, 0]}>
                <coneGeometry args={[1.6, 4.2, 24]} />
                <meshStandardMaterial color="#ffffff" metalness={0.62} roughness={0.24} />
            </mesh>

            {/* Delta wing */}
            <mesh position={[0, -0.05, -1.2]} rotation={[0, 0, Math.PI / 4]}>
                <boxGeometry args={[15.5, 0.42, 3.1]} />
                <meshStandardMaterial color="#bfc7c9" metalness={0.52} roughness={0.34} />
            </mesh>
            <mesh position={[0, -0.05, -1.2]} rotation={[0, 0, -Math.PI / 4]}>
                <boxGeometry args={[15.5, 0.42, 3.1]} />
                <meshStandardMaterial color="#bfc7c9" metalness={0.52} roughness={0.34} />
            </mesh>

            {/* Wingtip pods */}
            {[-1, 1].map((side) => (
                <group key={side} position={[side * 6.2, 0, -1.2]}>
                    <mesh rotation={[Math.PI / 2, 0, 0]}>
                        <cylinderGeometry args={[0.75, 0.75, 2.4, 18]} />
                        <meshStandardMaterial color="#22282d" metalness={0.65} roughness={0.3} />
                    </mesh>
                    <mesh position={[0, 0, -1.35]} rotation={[Math.PI / 2, 0, 0]}>
                        <torusGeometry args={[0.9, 0.06, 8, 22]} />
                        <meshBasicMaterial color="#f5f7f4" transparent opacity={0.75} />
                    </mesh>
                </group>
            ))}

            {/* Tail fins */}
            <mesh position={[-1.15, 1.1, 5.0]} rotation={[0.35, 0, -0.35]}>
                <boxGeometry args={[0.35, 2.8, 3.2]} />
                <meshStandardMaterial color="#8e989d" metalness={0.45} roughness={0.38} />
            </mesh>
            <mesh position={[1.15, 1.1, 5.0]} rotation={[0.35, 0, 0.35]}>
                <boxGeometry args={[0.35, 2.8, 3.2]} />
                <meshStandardMaterial color="#8e989d" metalness={0.45} roughness={0.38} />
            </mesh>

            {/* Cockpit / sensor glow */}
            <mesh position={[0, 0.85, -5.1]}>
                <sphereGeometry args={[0.75, 16, 10]} />
                <meshStandardMaterial
                    color="#83d8ff"
                    metalness={0.35}
                    roughness={0.18}
                    emissive="#20aeea"
                    emissiveIntensity={0.8}
                />
            </mesh>
        </group>
        <LidarSensorVisualization scan={lidarScan} visible={lidarVisible} />
        </>
    );
};

export default Aircraft;
