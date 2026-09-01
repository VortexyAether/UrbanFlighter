import * as THREE from 'three';
import type { BuildingData } from '../api';
import { deterministicStringSeed, deterministicUnit } from '../utils/deterministicSampling';

export interface BuildingMeshData {
  center: THREE.Vector2;
  edgeGeometry: THREE.EdgesGeometry;
  facadeColor: string;
  facadeMetalness: number;
  facadeRoughness: number;
  geometry: THREE.ExtrudeGeometry;
  height: number;
  haloRadius: number;
  identity: string;
  roofColor: string;
  roofGeometry: THREE.ShapeGeometry;
  roofRoughness: number;
}

const FACADE_PALETTE = [
  { color: '#8d8476', metalness: 0.08, roughness: 0.78 },
  { color: '#6f767c', metalness: 0.42, roughness: 0.28 },
  { color: '#c2b6a4', metalness: 0.06, roughness: 0.72 },
  { color: '#5a5048', metalness: 0.1, roughness: 0.86 },
  { color: '#7b8389', metalness: 0.38, roughness: 0.32 },
  { color: '#9a8b78', metalness: 0.07, roughness: 0.8 },
];
const ROOF_PALETTE = ['#3c3f41', '#45423d', '#2f3335', '#3a3632', '#41464a'];

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
    const facade = FACADE_PALETTE[facadeIndex];

    return [{
      center,
      edgeGeometry: new THREE.EdgesGeometry(geometry, 22),
      facadeColor: facade.color,
      facadeMetalness: facade.metalness,
      facadeRoughness: facade.roughness + deterministicUnit(seed, index, 2) * 0.08,
      geometry,
      height: building.height,
      haloRadius,
      identity,
      roofColor: ROOF_PALETTE[roofIndex],
      roofGeometry: createBuildingRoofGeometry(building),
      roofRoughness: 0.86 + deterministicUnit(seed, index, 3) * 0.1,
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
