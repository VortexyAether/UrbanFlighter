import React, { useEffect, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import type { BuildingData } from '../api';
import { isScenePointInsideBuilding } from '../geometry/buildingGeometry';

interface CameraFollowProps {
    target: THREE.Vector3;
    yaw: number;
    pitch: number;
    enabled: boolean;
    buildings?: readonly BuildingData[];
    distance?: number;
    height?: number;
    lookAhead?: number;
}

const MIN_CHASE_DISTANCE = 3.2;
const MAX_CHASE_DISTANCE = 48;
const MIN_CHASE_HEIGHT = 1.4;
const MAX_CHASE_HEIGHT = 22;

const CameraFollow: React.FC<CameraFollowProps> = ({
    target,
    yaw,
    pitch,
    enabled,
    buildings = [],
    distance = 7.6,
    height = 3.2,
    lookAhead = 5.2,
}) => {
    const { camera, gl } = useThree();
    const smoothTarget = useRef(target.clone());
    const smoothCamera = useRef(target.clone().add(new THREE.Vector3(0, height, distance)));
    const followedLastFrame = useRef(false);
    const chaseDistance = useRef(distance);
    const chaseHeight = useRef(height);

    // Scroll / trackpad zoom while chase is active (OrbitControls owns zoom in orbit mode).
    useEffect(() => {
        if (!enabled) return undefined;
        const element = gl.domElement;
        const onWheel = (event: WheelEvent) => {
            event.preventDefault();
            const delta = Math.sign(event.deltaY);
            const zoomFactor = event.deltaY > 0 ? 1.08 : 1 / 1.08;
            // Prefer pixel-magnitude when available so trackpads feel less jumpy.
            const magnitude = Math.min(1.2, Math.max(0.55, Math.abs(event.deltaY) / 120));
            const factor = delta >= 0 ? 1 + 0.08 * magnitude : 1 / (1 + 0.08 * magnitude);
            chaseDistance.current = THREE.MathUtils.clamp(
                chaseDistance.current * (event.deltaMode === 0 ? factor : zoomFactor),
                MIN_CHASE_DISTANCE,
                MAX_CHASE_DISTANCE,
            );
            chaseHeight.current = THREE.MathUtils.clamp(
                chaseHeight.current * (0.65 + 0.35 * (chaseDistance.current / distance)),
                MIN_CHASE_HEIGHT,
                MAX_CHASE_HEIGHT,
            );
        };
        element.addEventListener('wheel', onWheel, { passive: false });
        return () => element.removeEventListener('wheel', onWheel);
    }, [distance, enabled, gl.domElement]);

    useFrame((_, delta) => {
        if (!enabled) {
            followedLastFrame.current = false;
            return;
        }
        const forward = new THREE.Vector3(
            -Math.sin(yaw) * Math.cos(pitch),
            Math.sin(pitch) * 0.42,
            -Math.cos(yaw) * Math.cos(pitch),
        ).normalize();
        const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();
        const rawDesiredPosition = target.clone()
            .addScaledVector(forward, -chaseDistance.current)
            .add(new THREE.Vector3(0, chaseHeight.current, 0))
            .addScaledVector(right, 0.7);
        // Keep the chase camera on the target side of physical OSM facades.
        // Decorative dressing is intentionally absent from this camera check.
        const cameraAnchor = target.clone().add(new THREE.Vector3(0, 0.35, 0));
        const desiredPosition = cameraAnchor.clone();
        for (let sample = 1; sample <= 20; sample += 1) {
            const candidate = cameraAnchor.clone().lerp(rawDesiredPosition, sample / 20);
            if (buildings.some((building) => isScenePointInsideBuilding(building, candidate, 0.3))) break;
            desiredPosition.copy(candidate);
        }
        const desiredLookAt = target.clone()
            .addScaledVector(forward, lookAhead)
            .add(new THREE.Vector3(0, 0.7, 0));

        if (!followedLastFrame.current) {
            smoothCamera.current.copy(desiredPosition);
            smoothTarget.current.copy(desiredLookAt);
            followedLastFrame.current = true;
        } else {
            smoothCamera.current.lerp(desiredPosition, 1 - Math.exp(-5.8 * Math.min(delta, 0.1)));
            smoothTarget.current.lerp(desiredLookAt, 1 - Math.exp(-7.5 * Math.min(delta, 0.1)));
        }
        camera.position.copy(smoothCamera.current);
        camera.lookAt(smoothTarget.current);
    });

    return null;
};

export default CameraFollow;
