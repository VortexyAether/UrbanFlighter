import React, { useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

interface CameraFollowProps {
    target: THREE.Vector3;
    yaw: number;
    pitch: number;
    enabled: boolean;
    distance?: number;
    height?: number;
    lookAhead?: number;
}

const CameraFollow: React.FC<CameraFollowProps> = ({
    target,
    yaw,
    pitch,
    enabled,
    distance = 68,
    height = 28,
    lookAhead = 118,
}) => {
    const { camera } = useThree();
    const smoothTarget = useRef(new THREE.Vector3(0, 50, 0));
    const smoothCamera = useRef(new THREE.Vector3(0, 68, 68));

    useFrame(() => {
        if (!enabled) return;

        const forward = new THREE.Vector3(
            -Math.sin(yaw) * Math.cos(pitch),
            Math.sin(pitch),
            -Math.cos(yaw) * Math.cos(pitch)
        ).normalize();

        const side = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), forward).normalize();
        const cameraOffset = forward.clone().multiplyScalar(-distance)
            .add(new THREE.Vector3(0, height, 0))
            .add(side.multiplyScalar(2.0));

        const desiredPosition = target.clone().add(cameraOffset);
        const desiredLookAt = target.clone()
            .add(forward.clone().multiplyScalar(lookAhead))
            .add(new THREE.Vector3(0, 3.5, 0));

        smoothCamera.current.lerp(desiredPosition, 0.085);
        smoothTarget.current.lerp(desiredLookAt, 0.12);

        camera.position.copy(smoothCamera.current);
        camera.lookAt(smoothTarget.current);
    });

    return null;
};

export default CameraFollow;
