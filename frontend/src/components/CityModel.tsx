import React, { useEffect, useMemo } from 'react';
import * as THREE from 'three';
import type { BuildingData } from '../api';
import { buildBuildingMeshData } from '../geometry/buildingGeometry';

interface CityModelProps {
  buildings: BuildingData[];
  bounds?: {
    min_x: number;
    max_x: number;
    min_y: number;
    max_y: number;
  };
  presentationMode?: boolean;
  showClearanceHalo?: boolean;
}

const CityModel: React.FC<CityModelProps> = ({
  buildings,
  bounds = { min_x: -400, max_x: 400, min_y: -400, max_y: 400 },
  presentationMode = true,
  showClearanceHalo = false,
}) => {
  const buildingMeshes = useMemo(() => buildBuildingMeshData(buildings), [buildings]);
  const groundWidth = Math.max(1, bounds.max_x - bounds.min_x);
  const groundDepth = Math.max(1, bounds.max_y - bounds.min_y);
  const groundCenterX = (bounds.min_x + bounds.max_x) / 2;
  const groundCenterZ = -(bounds.min_y + bounds.max_y) / 2;

  useEffect(() => () => {
    buildingMeshes.forEach((building) => {
      building.edgeGeometry.dispose();
      building.geometry.dispose();
      building.roofGeometry.dispose();
    });
  }, [buildingMeshes]);

  return (
    <group>
      {buildingMeshes.map((building, index) => (
        <group key={building.identity || `${index}-${building.height.toFixed(1)}`}>
          <mesh geometry={building.geometry} receiveShadow castShadow>
            <meshStandardMaterial
              color={presentationMode ? building.facadeColor : '#a3aaab'}
              metalness={presentationMode ? 0.12 : 0.05}
              roughness={presentationMode ? building.facadeRoughness : 0.82}
            />
          </mesh>
          <mesh geometry={building.roofGeometry} receiveShadow castShadow>
            <meshStandardMaterial
              color={presentationMode ? building.roofColor : '#81898a'}
              metalness={presentationMode ? 0.18 : 0.04}
              roughness={presentationMode ? building.roofRoughness : 0.88}
            />
          </mesh>
          <lineSegments geometry={building.edgeGeometry}>
            <lineBasicMaterial color="#263033" transparent opacity={presentationMode ? 0.18 : 0.28} />
          </lineSegments>
          {showClearanceHalo && (
            <mesh position={[building.center.x, building.height / 2, building.center.y]} renderOrder={1}>
              <cylinderGeometry args={[building.haloRadius, building.haloRadius, building.height + 6, 40, 1, true]} />
              <meshBasicMaterial color="#ff8a2a" transparent opacity={0.075} depthWrite={false} side={THREE.DoubleSide} />
            </mesh>
          )}
        </group>
      ))}

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.16, 0]} receiveShadow>
        <planeGeometry args={[5200, 5200]} />
        <meshStandardMaterial color={presentationMode ? '#536057' : '#242b2e'} roughness={1} metalness={0} />
      </mesh>
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[groundCenterX, -0.075, groundCenterZ]}
        receiveShadow
      >
        <planeGeometry args={[groundWidth + 40, groundDepth + 40]} />
        <meshStandardMaterial
          color={presentationMode ? '#68716b' : '#252d30'}
          roughness={0.99}
          metalness={0.01}
        />
      </mesh>
      {!presentationMode && (
        <gridHelper
          args={[Math.max(groundWidth, groundDepth), 64, '#5f7379', '#344348']}
          position={[groundCenterX, 0.025, groundCenterZ]}
        />
      )}
    </group>
  );
};

export default CityModel;
