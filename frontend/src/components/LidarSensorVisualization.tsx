import { useMemo } from 'react';
import * as THREE from 'three';
import { getLidarJetColor, type LidarScan } from '../sensors/lidar';

interface LidarSensorVisualizationProps {
  scan: LidarScan | null;
  visible: boolean;
}

export default function LidarSensorVisualization({ scan, visible }: LidarSensorVisualizationProps) {
  const pointSets = useMemo(() => {
    const sets = [[], []] as Array<Array<{ point: THREE.Vector3; normalizedDistance: number }>>;
    scan?.samples.forEach((sample) => {
      // Hits stop on simulator sensing surfaces (OSM meshes or y=0 ground);
      // misses complete the max-range shell. Every sample is displayed.
      sets[sample.hit ? 1 : 0].push({
        point: sample.hitPoint ?? sample.endpoint,
        normalizedDistance: sample.normalizedDistance,
      });
    });

    const color = new THREE.Color();
    return sets.map((set) => {
      const positions = new Float32Array(set.length * 3);
      const colors = new Float32Array(set.length * 3);
      set.forEach(({ point, normalizedDistance }, index) => {
        getLidarJetColor(normalizedDistance, color);
        positions.set(point.toArray(), index * 3);
        colors.set(color.toArray(), index * 3);
      });
      return { positions, colors };
    });
  }, [scan]);

  if (!visible || !scan) return null;

  return (
    <group>
      {pointSets.map((pointSet, index) => (
        <points key={index} frustumCulled={false} renderOrder={index + 1}>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[pointSet.positions, 3]} />
            <bufferAttribute attach="attributes-color" args={[pointSet.colors, 3]} />
          </bufferGeometry>
          <pointsMaterial
            size={index === 0 ? 2.55 : 3.2}
            sizeAttenuation
            vertexColors
            transparent
            opacity={index === 0 ? 0.78 : 0.96}
            blending={THREE.NormalBlending}
            depthTest
            depthWrite={false}
            fog={false}
          />
        </points>
      ))}
    </group>
  );
}
