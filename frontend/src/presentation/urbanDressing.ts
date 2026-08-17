import type { BuildingData } from '../api';
import { deterministicStringSeed, deterministicUnit } from '../utils/deterministicSampling';

export interface DressingBounds {
  min_x: number;
  max_x: number;
  min_y: number;
  max_y: number;
}

export interface DressingPoint {
  x: number;
  y: number;
}

export interface DressingExclusionZone extends DressingPoint {
  radius: number;
  kind: 'start' | 'goal' | 'protected';
}

export interface DressingIdentity {
  scenarioId?: string | null;
  lat: number;
  lon: number;
}

export interface TreeProp extends DressingPoint {
  height: number;
  canopyRadius: number;
  rotation: number;
  tone: number;
}

export interface StreetlightProp extends DressingPoint {
  height: number;
  rotation: number;
}

export interface RooftopUnitProp extends DressingPoint {
  buildingIdentity: string;
  roofHeight: number;
  width: number;
  depth: number;
  height: number;
  rotation: number;
  tone: number;
}

export interface RoadBandProp extends DressingPoint {
  length: number;
  width: number;
  rotation: number;
  tone: number;
}

export interface RoadMarkingProp extends DressingPoint {
  length: number;
  width: number;
  rotation: number;
}

export interface FacadePanelProp extends DressingPoint {
  elevation: number;
  width: number;
  height: number;
  rotation: number;
  tone: number;
}

export interface UrbanDressingLayout {
  readonly contract: 'presentation_only_non_physical';
  readonly seed: number;
  readonly trees: TreeProp[];
  readonly streetlights: StreetlightProp[];
  readonly rooftopUnits: RooftopUnitProp[];
  readonly roadBands: RoadBandProp[];
  readonly roadMarkings: RoadMarkingProp[];
  readonly facadePanels: FacadePanelProp[];
}

export interface UrbanDressingOptions {
  bounds: DressingBounds;
  buildings: readonly BuildingData[];
  identity: DressingIdentity;
  exclusionZones?: readonly DressingExclusionZone[];
}

export const URBAN_DRESSING_LIMITS = Object.freeze({
  trees: 88,
  streetlights: 72,
  rooftopUnits: 56,
  roadBands: 4,
  roadMarkings: 176,
  facadePanels: 420,
  candidateAttempts: 1_600,
});

export const URBAN_DRESSING_MECHANICS_CONTRACT = Object.freeze({
  collision: false,
  lidar: false,
  wind: false,
  gym: false,
  observations: false,
  reward: false,
  scenarioHash: false,
} as const);

const POINT_EPSILON = 1e-9;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function distanceSquared(a: DressingPoint, b: DressingPoint) {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
}

function pointSegmentDistance(point: DressingPoint, start: number[], end: number[]) {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= POINT_EPSILON) return Math.hypot(point.x - start[0], point.y - start[1]);
  const fraction = clamp(
    ((point.x - start[0]) * dx + (point.y - start[1]) * dy) / lengthSquared,
    0,
    1,
  );
  return Math.hypot(
    point.x - (start[0] + dx * fraction),
    point.y - (start[1] + dy * fraction),
  );
}

export function isDressingPointInsideFootprint(point: DressingPoint, footprint: readonly number[][]) {
  if (footprint.length < 3) return false;
  let inside = false;
  for (let index = 0, previous = footprint.length - 1; index < footprint.length; previous = index, index += 1) {
    const [x, y] = footprint[index];
    const [previousX, previousY] = footprint[previous];
    const crossesY = (y > point.y) !== (previousY > point.y);
    const edgeX = ((previousX - x) * (point.y - y)) / (previousY - y || Number.EPSILON) + x;
    if (crossesY && point.x < edgeX) inside = !inside;
  }
  return inside;
}

export function dressingPointBuildingClearance(
  point: DressingPoint,
  buildings: readonly BuildingData[],
) {
  let clearance = Number.POSITIVE_INFINITY;
  for (const building of buildings) {
    if (building.footprint.length < 3) continue;
    if (isDressingPointInsideFootprint(point, building.footprint)) return -1;
    for (let index = 0; index < building.footprint.length; index += 1) {
      clearance = Math.min(
        clearance,
        pointSegmentDistance(
          point,
          building.footprint[index],
          building.footprint[(index + 1) % building.footprint.length],
        ),
      );
    }
  }
  return clearance;
}

export function isDressingGroundPointSafe(
  point: DressingPoint,
  bounds: DressingBounds,
  buildings: readonly BuildingData[],
  exclusionZones: readonly DressingExclusionZone[],
  buildingClearance: number,
  boundaryClearance: number,
) {
  if (
    point.x < bounds.min_x + boundaryClearance
    || point.x > bounds.max_x - boundaryClearance
    || point.y < bounds.min_y + boundaryClearance
    || point.y > bounds.max_y - boundaryClearance
  ) return false;
  if (dressingPointBuildingClearance(point, buildings) < buildingClearance) return false;
  return exclusionZones.every((zone) => distanceSquared(point, zone) >= zone.radius ** 2);
}

function mixSeed(seed: number, value: number) {
  let hash = (seed ^ value) >>> 0;
  hash = Math.imul(hash ^ (hash >>> 16), 0x7feb352d);
  hash = Math.imul(hash ^ (hash >>> 15), 0x846ca68b);
  return (hash ^ (hash >>> 16)) >>> 0;
}

function quantizedCoordinate(value: number) {
  return Math.round(value * 100);
}

function buildingIdentity(building: BuildingData, index: number) {
  if (building.building_id) return building.building_id;
  const first = building.footprint[0] ?? [0, 0];
  return `building-${index}-${quantizedCoordinate(first[0])}-${quantizedCoordinate(first[1])}-${quantizedCoordinate(building.height)}`;
}

export function urbanDressingSeed(options: UrbanDressingOptions) {
  const { bounds, buildings, identity } = options;
  const scenarioIdentity = identity.scenarioId
    ?? `${identity.lat.toFixed(6)},${identity.lon.toFixed(6)}`;
  let seed = deterministicStringSeed([
    'urban-flighter.presentation.v1',
    scenarioIdentity,
    bounds.min_x.toFixed(2),
    bounds.max_x.toFixed(2),
    bounds.min_y.toFixed(2),
    bounds.max_y.toFixed(2),
  ].join('|'));
  buildings.forEach((building, index) => {
    seed = mixSeed(seed, deterministicStringSeed(buildingIdentity(building, index)));
    seed = mixSeed(seed, quantizedCoordinate(building.height));
    building.footprint.forEach(([x, y]) => {
      seed = mixSeed(seed, quantizedCoordinate(x));
      seed = mixSeed(seed, quantizedCoordinate(y));
    });
  });
  return seed >>> 0;
}

function polygonArea(footprint: readonly number[][]) {
  let twiceArea = 0;
  for (let index = 0; index < footprint.length; index += 1) {
    const [x, y] = footprint[index];
    const [nextX, nextY] = footprint[(index + 1) % footprint.length];
    twiceArea += x * nextY - nextX * y;
  }
  return Math.abs(twiceArea) / 2;
}

function polygonSignedArea(footprint: readonly number[][]) {
  let twiceArea = 0;
  for (let index = 0; index < footprint.length; index += 1) {
    const [x, y] = footprint[index];
    const [nextX, nextY] = footprint[(index + 1) % footprint.length];
    twiceArea += x * nextY - nextX * y;
  }
  return twiceArea / 2;
}

function polygonCentroid(footprint: readonly number[][]): DressingPoint | null {
  let twiceArea = 0;
  let xSum = 0;
  let ySum = 0;
  for (let index = 0; index < footprint.length; index += 1) {
    const [x, y] = footprint[index];
    const [nextX, nextY] = footprint[(index + 1) % footprint.length];
    const cross = x * nextY - nextX * y;
    twiceArea += cross;
    xSum += (x + nextX) * cross;
    ySum += (y + nextY) * cross;
  }
  if (Math.abs(twiceArea) <= POINT_EPSILON) return null;
  return { x: xSum / (3 * twiceArea), y: ySum / (3 * twiceArea) };
}

function pointFootprintEdgeClearance(point: DressingPoint, footprint: readonly number[][]) {
  let clearance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < footprint.length; index += 1) {
    clearance = Math.min(
      clearance,
      pointSegmentDistance(point, footprint[index], footprint[(index + 1) % footprint.length]),
    );
  }
  return clearance;
}

function findRooftopPoint(footprint: readonly number[][], seed: number) {
  const xs = footprint.map((point) => point[0]);
  const ys = footprint.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const candidates: DressingPoint[] = [];
  const centroid = polygonCentroid(footprint);
  if (centroid) candidates.push(centroid);
  for (let row = 1; row <= 5; row += 1) {
    for (let column = 1; column <= 5; column += 1) {
      const index = (row - 1) * 5 + column - 1;
      const jitterX = (deterministicUnit(seed, index, 0) - 0.5) * 0.08;
      const jitterY = (deterministicUnit(seed, index, 1) - 0.5) * 0.08;
      candidates.push({
        x: minX + clamp(column / 6 + jitterX, 0.08, 0.92) * (maxX - minX),
        y: minY + clamp(row / 6 + jitterY, 0.08, 0.92) * (maxY - minY),
      });
    }
  }
  return candidates
    .filter((point) => isDressingPointInsideFootprint(point, footprint))
    .map((point) => ({ point, clearance: pointFootprintEdgeClearance(point, footprint) }))
    .sort((a, b) => b.clearance - a.clearance)[0] ?? null;
}

function makeRoadBands(seed: number, bounds: DressingBounds): RoadBandProp[] {
  const width = bounds.max_x - bounds.min_x;
  const depth = bounds.max_y - bounds.min_y;
  const fractions = [0.34, 0.66];
  const bands: RoadBandProp[] = [];
  fractions.forEach((fraction, index) => {
    const horizontalFraction = clamp(
      fraction + (deterministicUnit(seed, index, 10) - 0.5) * 0.07,
      0.15,
      0.85,
    );
    const verticalFraction = clamp(
      fraction + (deterministicUnit(seed, index, 11) - 0.5) * 0.07,
      0.15,
      0.85,
    );
    bands.push({
      x: (bounds.min_x + bounds.max_x) / 2,
      y: bounds.min_y + horizontalFraction * depth,
      length: width,
      width: 8 + deterministicUnit(seed, index, 12) * 3.5,
      rotation: 0,
      tone: deterministicUnit(seed, index, 13),
    });
    bands.push({
      x: bounds.min_x + verticalFraction * width,
      y: (bounds.min_y + bounds.max_y) / 2,
      length: depth,
      width: 8 + deterministicUnit(seed, index, 14) * 3.5,
      rotation: Math.PI / 2,
      tone: deterministicUnit(seed, index, 15),
    });
  });
  return bands.slice(0, URBAN_DRESSING_LIMITS.roadBands);
}

function candidateAlongBand(
  band: RoadBandProp,
  distanceAlong: number,
  lateralOffset: number,
): DressingPoint {
  const cosine = Math.cos(band.rotation);
  const sine = Math.sin(band.rotation);
  return {
    x: band.x + cosine * distanceAlong - sine * lateralOffset,
    y: band.y + sine * distanceAlong + cosine * lateralOffset,
  };
}

function makeRoadMarkings(
  seed: number,
  bounds: DressingBounds,
  buildings: readonly BuildingData[],
  zones: readonly DressingExclusionZone[],
  bands: readonly RoadBandProp[],
) {
  const markings: RoadMarkingProp[] = [];
  bands.forEach((band, bandIndex) => {
    const spacing = 12;
    const count = Math.floor(band.length / spacing);
    const phase = deterministicUnit(seed, bandIndex, 20) * spacing;
    for (let index = 0; index < count && markings.length < URBAN_DRESSING_LIMITS.roadMarkings; index += 1) {
      const distanceAlong = -band.length / 2 + phase + index * spacing;
      const point = candidateAlongBand(band, distanceAlong, 0);
      if (!isDressingGroundPointSafe(point, bounds, buildings, zones, 1.3, 2)) continue;
      markings.push({ ...point, length: 4.8, width: 0.18, rotation: band.rotation });
    }
  });
  return markings;
}

function makeStreetlights(
  seed: number,
  bounds: DressingBounds,
  buildings: readonly BuildingData[],
  zones: readonly DressingExclusionZone[],
  bands: readonly RoadBandProp[],
) {
  const lights: StreetlightProp[] = [];
  bands.forEach((band, bandIndex) => {
    const spacing = 30;
    const count = Math.floor(band.length / spacing);
    const phase = deterministicUnit(seed, bandIndex, 30) * spacing;
    for (let index = 0; index < count; index += 1) {
      for (const side of [-1, 1]) {
        if (lights.length >= URBAN_DRESSING_LIMITS.streetlights) return;
        const point = candidateAlongBand(
          band,
          -band.length / 2 + phase + index * spacing,
          side * (band.width / 2 + 1.8),
        );
        if (!isDressingGroundPointSafe(point, bounds, buildings, zones, 2.4, 4)) continue;
        if (lights.some((light) => distanceSquared(light, point) < 9 ** 2)) continue;
        lights.push({
          ...point,
          height: 5.8 + deterministicUnit(seed, lights.length, 31) * 1.8,
          rotation: band.rotation + (side < 0 ? Math.PI : 0),
        });
      }
    }
  });
  return lights;
}

function makeTrees(
  seed: number,
  bounds: DressingBounds,
  buildings: readonly BuildingData[],
  zones: readonly DressingExclusionZone[],
  lights: readonly StreetlightProp[],
) {
  const width = Math.max(0, bounds.max_x - bounds.min_x);
  const depth = Math.max(0, bounds.max_y - bounds.min_y);
  const area = width * depth;
  const targetCount = clamp(Math.round(area / 8_500), 18, URBAN_DRESSING_LIMITS.trees);
  const trees: TreeProp[] = [];
  const pad = Math.min(10, width / 8, depth / 8);
  const usableWidth = Math.max(0, width - pad * 2);
  const usableDepth = Math.max(0, depth - pad * 2);
  for (let index = 0; index < URBAN_DRESSING_LIMITS.candidateAttempts; index += 1) {
    if (trees.length >= targetCount) break;
    const point = {
      x: bounds.min_x + pad + deterministicUnit(seed, index, 40) * usableWidth,
      y: bounds.min_y + pad + deterministicUnit(seed, index, 41) * usableDepth,
    };
    if (!isDressingGroundPointSafe(point, bounds, buildings, zones, 4.8, 7)) continue;
    if (trees.some((tree) => distanceSquared(tree, point) < 9.5 ** 2)) continue;
    if (lights.some((light) => distanceSquared(light, point) < 3.5 ** 2)) continue;
    const maturity = deterministicUnit(seed, index, 42);
    trees.push({
      ...point,
      height: 4.4 + maturity * 4.6,
      canopyRadius: 1.15 + deterministicUnit(seed, index, 43) * 1.15,
      rotation: deterministicUnit(seed, index, 44) * Math.PI * 2,
      tone: deterministicUnit(seed, index, 45),
    });
  }
  return trees;
}

function makeRooftopUnits(seed: number, buildings: readonly BuildingData[]) {
  return buildings
    .map((building, index) => {
      const identity = buildingIdentity(building, index);
      const buildingSeed = mixSeed(seed, deterministicStringSeed(identity));
      if (building.height < 7 || polygonArea(building.footprint) < 45) return null;
      if (deterministicUnit(buildingSeed, index, 50) < 0.28) return null;
      const rooftop = findRooftopPoint(building.footprint, buildingSeed);
      if (!rooftop) return null;
      const maxUnitSpan = Math.min(3.2, Math.max(0, rooftop.clearance * 1.25));
      if (maxUnitSpan < 1.1) return null;
      const width = Math.min(maxUnitSpan, 1.15 + deterministicUnit(buildingSeed, index, 51) * 1.8);
      const depth = Math.min(maxUnitSpan, 1.0 + deterministicUnit(buildingSeed, index, 52) * 1.6);
      return {
        ...rooftop.point,
        buildingIdentity: identity,
        roofHeight: building.height,
        width,
        depth,
        height: 0.65 + deterministicUnit(buildingSeed, index, 53) * 1.05,
        rotation: deterministicUnit(buildingSeed, index, 54) * Math.PI,
        tone: deterministicUnit(buildingSeed, index, 55),
        sort: deterministicUnit(buildingSeed, index, 56),
      };
    })
    .filter((unit): unit is NonNullable<typeof unit> => unit !== null)
    .sort((a, b) => a.sort - b.sort)
    .slice(0, URBAN_DRESSING_LIMITS.rooftopUnits)
    .map((unit) => ({
      x: unit.x,
      y: unit.y,
      buildingIdentity: unit.buildingIdentity,
      roofHeight: unit.roofHeight,
      width: unit.width,
      depth: unit.depth,
      height: unit.height,
      rotation: unit.rotation,
      tone: unit.tone,
    }));
}

function makeFacadePanels(seed: number, buildings: readonly BuildingData[]): FacadePanelProp[] {
  const panels: Array<FacadePanelProp & { sort: number }> = [];
  buildings.forEach((building, buildingIndex) => {
    if (building.height < 6 || building.footprint.length < 3) return;
    const identitySeed = mixSeed(seed, deterministicStringSeed(buildingIdentity(building, buildingIndex)));
    const counterClockwise = polygonSignedArea(building.footprint) > 0;
    building.footprint.forEach((start, edgeIndex) => {
      const end = building.footprint[(edgeIndex + 1) % building.footprint.length];
      const dx = end[0] - start[0];
      const dy = end[1] - start[1];
      const edgeLength = Math.hypot(dx, dy);
      if (edgeLength < 5) return;
      const outwardX = (counterClockwise ? dy : -dy) / edgeLength;
      const outwardY = (counterClockwise ? -dx : dx) / edgeLength;
      const levelCount = Math.min(3, Math.max(1, Math.floor((building.height - 2) / 10)));
      for (let level = 0; level < levelCount; level += 1) {
        const sample = edgeIndex * 4 + level;
        if (deterministicUnit(identitySeed, sample, 60) < 0.14) continue;
        const alongJitter = (deterministicUnit(identitySeed, sample, 61) - 0.5) * Math.min(2, edgeLength * 0.12);
        const midpointX = (start[0] + end[0]) / 2 + (dx / edgeLength) * alongJitter;
        const midpointY = (start[1] + end[1]) / 2 + (dy / edgeLength) * alongJitter;
        panels.push({
          x: midpointX + outwardX * 0.045,
          y: midpointY + outwardY * 0.045,
          elevation: 3.2 + ((level + 0.5) / levelCount) * Math.max(1.5, building.height - 5),
          width: Math.min(edgeLength * 0.72, 16),
          height: 1.0 + deterministicUnit(identitySeed, sample, 62) * 0.75,
          rotation: Math.atan2(dy, dx),
          tone: deterministicUnit(identitySeed, sample, 63),
          sort: deterministicUnit(identitySeed, sample, 64),
        });
      }
    });
  });
  return panels
    .sort((a, b) => a.sort - b.sort)
    .slice(0, URBAN_DRESSING_LIMITS.facadePanels)
    .map((panel) => ({
      x: panel.x,
      y: panel.y,
      elevation: panel.elevation,
      width: panel.width,
      height: panel.height,
      rotation: panel.rotation,
      tone: panel.tone,
    }));
}

/**
 * Generates a bounded visual layout in the OSM-local east/north metre frame.
 * The returned data is consumed only by UrbanDressing; it is deliberately not
 * accepted by flight, collision, LiDAR, flow, rolling-map, or Gym interfaces.
 */
export function createUrbanDressingLayout(options: UrbanDressingOptions): UrbanDressingLayout {
  const zones = options.exclusionZones ?? [];
  const seed = urbanDressingSeed(options);
  const roadBands = makeRoadBands(seed, options.bounds);
  const roadMarkings = makeRoadMarkings(seed, options.bounds, options.buildings, zones, roadBands);
  const streetlights = makeStreetlights(seed, options.bounds, options.buildings, zones, roadBands);
  const trees = makeTrees(seed, options.bounds, options.buildings, zones, streetlights);
  const rooftopUnits = makeRooftopUnits(seed, options.buildings);
  const facadePanels = makeFacadePanels(seed, options.buildings);

  return {
    contract: 'presentation_only_non_physical',
    seed,
    trees,
    streetlights,
    rooftopUnits,
    roadBands,
    roadMarkings,
    facadePanels,
  };
}
