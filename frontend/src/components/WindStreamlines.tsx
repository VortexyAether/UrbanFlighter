import React, { useMemo } from 'react';
import * as THREE from 'three';
import { getWindVector } from '../systems/WindSystem';
import type { BuildingData } from '../api';
import { Line } from '@react-three/drei';
import { deterministicUnit } from '../utils/deterministicSampling';

interface WindStreamlinesProps {
    globalWindSpeed?: number;
    globalWindDir?: number;
    buildings?: BuildingData[];
    streamlineCount?: number;
    maxSteps?: number;
    stepSize?: number;
}

const EMPTY_BUILDINGS: BuildingData[] = [];
const WIND_STREAMLINE_SAMPLE_SEED = 0x7f4a7c15;

// Catmull-Rom spline interpolation helper
function catmullRom(p0: number, p1: number, p2: number, p3: number, t: number): number {
    const t2 = t * t;
    const t3 = t2 * t;
    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
        (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    );
}

const WindStreamlines: React.FC<WindStreamlinesProps> = ({
    globalWindSpeed = 10,
    globalWindDir = 45,
    buildings = EMPTY_BUILDINGS,
    streamlineCount = 200,
    maxSteps = 150,
    stepSize = 4.0
}) => {
    const globalWindVector = useMemo(() => {
        const rad = (globalWindDir * Math.PI) / 180;
        return new THREE.Vector3(Math.cos(rad), 0, Math.sin(rad)).multiplyScalar(globalWindSpeed);
    }, [globalWindSpeed, globalWindDir]);

    // Generate streamlines using fast Euler + Catmull-Rom smoothing
    const streamlines = useMemo(() => {
        const lines: { points: THREE.Vector3[]; colors: THREE.Color[] }[] = [];

        const gridDensity = Math.ceil(Math.sqrt(streamlineCount));
        const spacing = 3000 / gridDensity; // Cover 3000m x 3000m area

        for (let ix = 0; ix < gridDensity; ix++) {
            for (let iz = 0; iz < gridDensity; iz++) {
                if (lines.length >= streamlineCount) break;

                const sampleIndex = ix * gridDensity + iz;
                const startX = -1500 + ix * spacing
                    + (deterministicUnit(WIND_STREAMLINE_SAMPLE_SEED, sampleIndex, 0) - 0.5) * spacing * 0.3;
                const startZ = -1500 + iz * spacing
                    + (deterministicUnit(WIND_STREAMLINE_SAMPLE_SEED, sampleIndex, 1) - 0.5) * spacing * 0.3;
                const startY = 5 + deterministicUnit(WIND_STREAMLINE_SAMPLE_SEED, sampleIndex, 2) * 25; // 5-30m height

                const rawPoints: THREE.Vector3[] = [];
                const currentPos = new THREE.Vector3(startX, startY, startZ);

                // Fast Euler integration (coarse sampling)
                for (let step = 0; step < maxSteps; step++) {
                    // Circular boundary check (1500m radius)
                    const distFromCenter = Math.sqrt(currentPos.x * currentPos.x + currentPos.z * currentPos.z);
                    if (distFromCenter > 1500 || currentPos.y < 2 || currentPos.y > 200) {
                        break;
                    }

                    rawPoints.push(currentPos.clone());

                    const windVec = getWindVector(currentPos, globalWindVector, buildings);
                    const speed = windVec.length();

                    if (speed < 0.5) break;

                    // Simple Euler step
                    const direction = windVec.normalize();
                    currentPos.add(direction.multiplyScalar(stepSize));
                }

                if (rawPoints.length < 4) continue;

                // Apply Catmull-Rom smoothing
                const smoothPoints: THREE.Vector3[] = [];
                const colors: THREE.Color[] = [];

                for (let i = 0; i < rawPoints.length - 1; i++) {
                    const p0 = rawPoints[Math.max(0, i - 1)];
                    const p1 = rawPoints[i];
                    const p2 = rawPoints[i + 1];
                    const p3 = rawPoints[Math.min(rawPoints.length - 1, i + 2)];

                    // Interpolate 3 points between each pair
                    for (let t = 0; t < 1; t += 0.33) {
                        const point = new THREE.Vector3();
                        point.x = catmullRom(p0.x, p1.x, p2.x, p3.x, t);
                        point.y = catmullRom(p0.y, p1.y, p2.y, p3.y, t);
                        point.z = catmullRom(p0.z, p1.z, p2.z, p3.z, t);

                        smoothPoints.push(point);

                        // Color based on local wind speed
                        const windVec = getWindVector(point, globalWindVector, buildings);
                        const speed = windVec.length();
                        const hue = THREE.MathUtils.clamp(0.7 - (speed / 25.0) * 0.7, 0.0, 0.7);
                        colors.push(new THREE.Color().setHSL(hue, 1.0, 0.6));
                    }
                }

                if (smoothPoints.length > 5) {
                    lines.push({ points: smoothPoints, colors });
                }
            }
        }

        return lines;
    }, [globalWindVector, buildings, streamlineCount, maxSteps, stepSize]);

    return (
        <group>
            {streamlines.map((streamline, idx) => {
                // Alternate between golden and dark for Elden Ring aesthetic
                const isGolden = idx % 3 !== 0; // 2/3 golden, 1/3 dark
                const color = isGolden ? '#d4af37' : '#1a1a1a'; // Deep gold or black

                return (
                    <Line
                        key={idx}
                        points={streamline.points}
                        color={color}
                        lineWidth={isGolden ? 1.5 : 2.0}
                        transparent
                        opacity={isGolden ? 0.4 : 0.3}
                    />
                );
            })}
        </group>
    );
};

export default WindStreamlines;
