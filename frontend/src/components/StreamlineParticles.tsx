import React, { useEffect, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { getWindVector } from '../systems/WindSystem';
import type { BuildingData } from '../api';
import { deterministicUnit } from '../utils/deterministicSampling';

interface StreamlineParticlesProps {
    globalWindSpeed: number;
    globalWindDir: number;
    buildings: BuildingData[];
}

const STREAMLINE_COUNT = 800;
const STREAMLINE_SAMPLE_SEED = 0x51f15e5d;
const TUBE_RADIUS = 0.5;

// Custom shader for flowing line effect
const FlowShader = {
    uniforms: {
        uTime: { value: 0 },
        uColor: { value: new THREE.Color('#d4af37') },
        uEmissive: { value: new THREE.Color('#ffd700') },
        uSpeed: { value: 1.0 },
        uOffset: { value: 0.0 },
        uLength: { value: 0.2 },
        uOpacity: { value: 0.8 },
    },
    vertexShader: `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        uniform float uTime;
        uniform vec3 uColor;
        uniform vec3 uEmissive;
        uniform float uSpeed;
        uniform float uOffset;
        uniform float uLength;
        uniform float uOpacity;
        varying vec2 vUv;

        void main() {
            // Calculate flowing segment position
            float progress = mod(vUv.x - uTime * uSpeed + uOffset, 1.0);
            
            // Soft edge for the segment
            float edge = 0.05;
            float alpha = smoothstep(0.0, edge, progress) * (1.0 - smoothstep(uLength - edge, uLength, progress));
            
            // If outside segment, make very dim
            float finalAlpha = mix(0.1, uOpacity, alpha);
            vec3 finalColor = mix(uColor * 0.2, uColor, alpha);
            vec3 finalEmissive = uEmissive * alpha;

            gl_FragColor = vec4(finalColor + finalEmissive, finalAlpha);
        }
    `
};

const StreamlineParticles: React.FC<StreamlineParticlesProps> = ({
    globalWindSpeed,
    globalWindDir,
    buildings
}) => {
    // Convert global wind to vector
    const globalWindVector = useMemo(() => {
        const rad = (globalWindDir * Math.PI) / 180;
        return new THREE.Vector3(Math.cos(rad), 0, Math.sin(rad)).multiplyScalar(globalWindSpeed);
    }, [globalWindSpeed, globalWindDir]);

    // Generate streamlines (pre-calculated curves)
    const streamlines = useMemo(() => {
        const lines: THREE.CatmullRomCurve3[] = [];

        for (let i = 0; i < STREAMLINE_COUNT; i++) {
            // Deterministic starting position (expanded to 1500m)
            const angle = deterministicUnit(STREAMLINE_SAMPLE_SEED, i, 0) * Math.PI * 2;
            const radius = deterministicUnit(STREAMLINE_SAMPLE_SEED, i, 1) * 1500;
            const startPos = new THREE.Vector3(
                Math.cos(angle) * radius,
                5 + deterministicUnit(STREAMLINE_SAMPLE_SEED, i, 2) * 250, // Higher ceiling
                Math.sin(angle) * radius
            );

            // Trace wind flow
            const points: THREE.Vector3[] = [];
            const currentPos = startPos.clone();
            const stepSize = 15;
            const numSteps = 35; // Trace longer distances

            for (let j = 0; j < numSteps; j++) {
                points.push(currentPos.clone());

                const wind = getWindVector(currentPos, globalWindVector, buildings);
                const direction = wind.clone().normalize();
                currentPos.add(direction.multiplyScalar(stepSize));

                // Keep within expanded bounds
                const dist2D = Math.sqrt(currentPos.x ** 2 + currentPos.z ** 2);
                if (dist2D > 1650) break;
                if (currentPos.y < 2 || currentPos.y > 350) break;
            }

            if (points.length >= 4) {
                lines.push(new THREE.CatmullRomCurve3(points));
            }
        }

        return lines;
    }, [globalWindVector, buildings]);

    // Create streamline mesh data
    const meshes = useMemo(() => {
        return streamlines.map((curve, index) => {
            const geometry = new THREE.TubeGeometry(curve, 32, TUBE_RADIUS, 6, false);
            const isGolden = deterministicUnit(STREAMLINE_SAMPLE_SEED, index, 3) > 0.2;

            const material = new THREE.ShaderMaterial({
                uniforms: THREE.UniformsUtils.clone(FlowShader.uniforms),
                vertexShader: FlowShader.vertexShader,
                fragmentShader: FlowShader.fragmentShader,
                transparent: true,
                depthWrite: false,
                blending: THREE.AdditiveBlending,
            });

            material.uniforms.uColor.value = isGolden ? new THREE.Color('#d4af37') : new THREE.Color('#555555');
            material.uniforms.uEmissive.value = isGolden ? new THREE.Color('#ffaa00') : new THREE.Color('#222222');
            material.uniforms.uSpeed.value = 0.4 + deterministicUnit(STREAMLINE_SAMPLE_SEED, index, 4) * 0.4;
            material.uniforms.uOffset.value = deterministicUnit(STREAMLINE_SAMPLE_SEED, index, 5);
            material.uniforms.uLength.value = 0.15 + deterministicUnit(STREAMLINE_SAMPLE_SEED, index, 6) * 0.2;
            material.uniforms.uOpacity.value = isGolden ? 0.7 : 0.3;

            return { geometry, material };
        });
    }, [streamlines]);

    useEffect(() => () => {
        meshes.forEach(({ geometry, material }) => {
            geometry.dispose();
            material.dispose();
        });
    }, [meshes]);

    // Update uTime uniform
    useFrame(({ clock }) => {
        meshes.forEach(m => {
            m.material.uniforms.uTime.value = clock.getElapsedTime();
        });
    });

    return (
        <group>
            {meshes.map((m, i) => (
                <mesh key={i} geometry={m.geometry} material={m.material} />
            ))}
        </group>
    );
};

export default StreamlineParticles;
