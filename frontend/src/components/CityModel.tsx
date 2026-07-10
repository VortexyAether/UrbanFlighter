import React, { useEffect, useMemo } from 'react';
import * as THREE from 'three';
import type { BuildingData } from '../api';
import { buildBuildingMeshData } from '../geometry/buildingGeometry';

interface CityModelProps {
  buildings: BuildingData[];
  showClearanceHalo?: boolean;
}

const CityModel: React.FC<CityModelProps> = ({ buildings, showClearanceHalo = false }) => {
  const buildingMeshes = useMemo(() => buildBuildingMeshData(buildings), [buildings]);

  useEffect(() => () => {
    buildingMeshes.forEach((building) => {
      building.edgeGeometry.dispose();
      building.geometry.dispose();
    });
  }, [buildingMeshes]);

  return (
    <group>
      {buildingMeshes.map((building, index) => (
        <group key={`${index}-${building.height.toFixed(1)}`}>
          <mesh geometry={building.geometry} receiveShadow castShadow>
            <meshStandardMaterial
              color={building.color}
              metalness={0.1}
              roughness={0.72}
              transparent
              opacity={0.86}
            />
          </mesh>
          <lineSegments geometry={building.edgeGeometry}>
            <lineBasicMaterial color="#15191d" transparent opacity={0.2} />
          </lineSegments>
          {showClearanceHalo && (
            <mesh position={[building.center.x, building.height / 2, building.center.y]} renderOrder={1}>
              <cylinderGeometry args={[building.haloRadius, building.haloRadius, building.height + 6, 40, 1, true]} />
              <meshBasicMaterial color="#ff8a2a" transparent opacity={0.075} depthWrite={false} side={THREE.DoubleSide} />
            </mesh>
          )}
        </group>
      ))}

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.08, 0]} receiveShadow>
        <planeGeometry args={[5200, 5200]} />
        <meshStandardMaterial color="#252b2e" roughness={0.97} metalness={0.02} />
      </mesh>
      <gridHelper args={[5200, 104, '#485158', '#2d3439']} position={[0, 0.025, 0]} />
    </group>
  );
};

export default CityModel;
