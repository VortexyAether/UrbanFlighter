import * as THREE from 'three';
import type { BuildingData } from '../api';
import { deterministicStringSeed, deterministicUnit } from '../utils/deterministicSampling';

export interface BuildingMeshData {
  center: THREE.Vector2;
  edgeGeometry: THREE.EdgesGeometry;
  facadeColor: string;
  facadeRoughness: number;
  geometry: THREE.ExtrudeGeometry;
  height: number;
  haloRadius: number;
  identity: string;
  roofColor: string;
  roofGeometry: THREE.ShapeGeometry;
  roofRoughness: number;
}

const FACADE_PALETTE = ['#a9afb0', '#b8b5aa', '#9fa9ac', '#b0a49a', '#8f9ca0', '#b8bab4'];
const ROOF_PALETTE = ['#737d7f', '#85877f', '#687477', '#7d736b', '#8b8f8a'];

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

export function createBuildingRoofGeometry(building: BuildingData) {
  const shape = new THREE.Shape();
  shape.moveTo(building.footprint[0][0], building.footprint[0][1]);
  for (let index = 1; index < building.footprint.length; index += 1) {
    shape.lineTo(building.footprint[index][0], building.footprint[index][1]);
  }
  shape.closePath();
  const geometry = new THREE.ShapeGeometry(shape);
  geometry.rotateX(-Math.PI / 2);
  geometry.translate(0, building.height + 0.018, 0);
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
  return buildings.flatMap((building, index) => {
    if (!building.footprint || building.footprint.length < 3) return [];
    const footprint = building.footprint.map((point) => new THREE.Vector2(point[0], -point[1]));
    const center = footprint.reduce((sum, point) => sum.add(point), new THREE.Vector2()).divideScalar(footprint.length);
    const haloRadius = Math.max(...footprint.map((point) => point.distanceTo(center))) + 8;
    const geometry = createBuildingGeometry(building);
    const identity = building.building_id ?? `building-${index}-${building.footprint[0].join('-')}`;
    const seed = deterministicStringSeed(identity);
    const facadeIndex = Math.floor(deterministicUnit(seed, index, 0) * FACADE_PALETTE.length);
    const roofIndex = Math.floor(deterministicUnit(seed, index, 1) * ROOF_PALETTE.length);

    return [{
      center,
      edgeGeometry: new THREE.EdgesGeometry(geometry, 28),
      facadeColor: FACADE_PALETTE[facadeIndex],
      facadeRoughness: 0.68 + deterministicUnit(seed, index, 2) * 0.2,
      geometry,
      height: building.height,
      haloRadius,
      identity,
      roofColor: ROOF_PALETTE[roofIndex],
      roofGeometry: createBuildingRoofGeometry(building),
      roofRoughness: 0.74 + deterministicUnit(seed, index, 3) * 0.18,
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
