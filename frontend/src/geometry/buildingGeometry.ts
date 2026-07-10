import * as THREE from 'three';
import type { BuildingData } from '../api';

export interface BuildingMeshData {
  center: THREE.Vector2;
  color: string;
  edgeGeometry: THREE.EdgesGeometry;
  geometry: THREE.ExtrudeGeometry;
  height: number;
  haloRadius: number;
}

function getHeightColor(height: number) {
  if (height > 110) return '#ffffff';
  if (height > 70) return '#e1e4de';
  if (height > 35) return '#c8ccc7';
  return '#929995';
}

export function createBuildingGeometry(building: BuildingData) {
  const shape = new THREE.Shape();
  shape.moveTo(building.footprint[0][0], building.footprint[0][1]);
  for (let i = 1; i < building.footprint.length; i += 1) {
    shape.lineTo(building.footprint[i][0], building.footprint[i][1]);
  }
  shape.closePath();

  const geometry = new THREE.ExtrudeGeometry(shape, {
    steps: 1,
    depth: building.height,
    bevelEnabled: false,
  });
  geometry.rotateX(-Math.PI / 2);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

/**
 * Building footprints arrive in OSM-local XY coordinates. Extruding the shape and
 * rotating it into Three.js Y-up maps source +Y to scene -Z.
 */
export function isScenePointInsideBuilding(building: BuildingData, point: THREE.Vector3, verticalClearance = 0) {
  if (!building.footprint || building.footprint.length < 3 || point.y < 0 || point.y > building.height + verticalClearance) {
    return false;
  }

  const footprintY = -point.z;
  let inside = false;
  for (let i = 0, j = building.footprint.length - 1; i < building.footprint.length; j = i, i += 1) {
    const [xi, yi] = building.footprint[i];
    const [xj, yj] = building.footprint[j];
    const crossesY = (yi > footprintY) !== (yj > footprintY);
    const edgeX = ((xj - xi) * (footprintY - yi)) / (yj - yi || Number.EPSILON) + xi;
    if (crossesY && point.x < edgeX) inside = !inside;
  }
  return inside;
}

export function buildBuildingMeshData(buildings: BuildingData[]): BuildingMeshData[] {
  return buildings.flatMap((building) => {
    if (!building.footprint || building.footprint.length < 3) return [];
    const footprint = building.footprint.map((point) => new THREE.Vector2(point[0], -point[1]));
    const center = footprint.reduce((sum, point) => sum.add(point), new THREE.Vector2()).divideScalar(footprint.length);
    const haloRadius = Math.max(...footprint.map((point) => point.distanceTo(center))) + 8;
    const geometry = createBuildingGeometry(building);

    return [{
      center,
      color: getHeightColor(building.height),
      edgeGeometry: new THREE.EdgesGeometry(geometry, 28),
      geometry,
      height: building.height,
      haloRadius,
    }];
  });
}

export function buildBuildingCollisionMeshes(buildings: BuildingData[]): THREE.Mesh[] {
  return buildings.flatMap((building) => {
    if (!building.footprint || building.footprint.length < 3) return [];
    const mesh = new THREE.Mesh(createBuildingGeometry(building));
    mesh.updateMatrixWorld(true);
    return [mesh];
  });
}
