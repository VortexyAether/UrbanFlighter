import { useEffect, useMemo, useRef } from 'react';
import type { BuildingData, FlowField2DResponse } from '../api';
import { calculateDragEnergy } from '../utils/dragEnergy';
import { getLidarJetCss, type LidarDisplayPose, type LidarTelemetry } from '../sensors/lidar';
import {
  DEFAULT_LIDAR_2D_CONFIG,
  createLidar2DLocalDirections,
  scanLidar2D,
  selectLidar2DSpawn,
  summarizeLidar2DScan,
  type Lidar2DScan,
} from '../sensors/lidar2d';
import {
  consumeFlight2DFixedSteps,
  isFlight2DPositionBlocked,
  stepFlight2DMotion,
} from '../simulation/flight2dMotion';
import {
  eastNorthToCompassBearingDeg,
  mapAngleToCompassBearingDeg,
} from '../utils/bearings';

interface Vec2 {
  x: number;
  y: number;
}

interface TopDownGameProps {
  flow: FlowField2DResponse | null;
  showFlowAnimation: boolean;
  flowVisualization?: FlowVisualization;
  windScale?: number;
  showLidar?: boolean;
  onTelemetry: (telemetry: Telemetry) => void;
}

export type FlowVisualization = 'arrows' | 'streamlines' | 'colormap' | 'both';

export interface Telemetry {
  droneSpeed: number;
  localWindSpeed: number;
  localWindDirDeg: number;
  energyRate: number;
  energyUsed: number;
  headingDeg: number;
  position: Vec2;
  displayPose: LidarDisplayPose;
  lidar?: LidarTelemetry;
}

interface WorldState {
  position: Vec2;
  velocity: Vec2;
  heading: number;
  energyUsed: number;
}

interface FlowParticle {
  position: Vec2;
  age: number;
  life: number;
  hue: number;
  width: number;
}

interface FlowObstacle {
  footprint: number[][];
  center: Vec2;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  span: number;
}

interface Streamline {
  points: Vec2[];
  speed: number;
}

const LEGACY_DRONE_RADIUS_M = 12;
const VIEW_RADIUS_M = 200;
const PARTICLE_COUNT = 900;
const LIDAR_INTERVAL_S = 0.1;
const FLIGHT_CONTROL_CODES = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD']);

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function length(vec: Vec2) {
  return Math.hypot(vec.x, vec.y);
}

function normalize(vec: Vec2): Vec2 {
  const len = length(vec);
  return len > 1e-6 ? { x: vec.x / len, y: vec.y / len } : { x: 0, y: 0 };
}

function hasResolvedGrid(flow: FlowField2DResponse) {
  const { field } = flow;
  const expected = field.nx * field.ny;
  return field.nx > 1 && field.ny > 1 && field.ux.length === expected && field.uy.length === expected;
}

function isInsideBounds(point: Vec2, bounds: FlowField2DResponse['field']['bounds']) {
  return (
    point.x >= bounds.min_x && point.x <= bounds.max_x
    && point.y >= bounds.min_y && point.y <= bounds.max_y
  );
}

function sampleResolvedGrid(flow: FlowField2DResponse, point: Vec2): Vec2 | null {
  if (!hasResolvedGrid(flow)) {
    return null;
  }

  const { field } = flow;
  const { bounds } = field;
  if (
    point.x < bounds.min_x || point.x > bounds.max_x
    || point.y < bounds.min_y || point.y > bounds.max_y
  ) {
    return { x: flow.inlet.ux, y: flow.inlet.uy };
  }

  const gx = ((point.x - bounds.min_x) / Math.max(1e-6, bounds.max_x - bounds.min_x)) * (field.nx - 1);
  const gy = ((point.y - bounds.min_y) / Math.max(1e-6, bounds.max_y - bounds.min_y)) * (field.ny - 1);
  const x0 = clamp(Math.floor(gx), 0, field.nx - 1);
  const y0 = clamp(Math.floor(gy), 0, field.ny - 1);
  const x1 = clamp(x0 + 1, 0, field.nx - 1);
  const y1 = clamp(y0 + 1, 0, field.ny - 1);
  const tx = gx - x0;
  const ty = gy - y0;
  const idx = (ix: number, iy: number) => ix * field.ny + iy;

  const nearest = idx(Math.round(gx), Math.round(gy));
  if ((field.mask[nearest] ?? 0) > 0) {
    return { x: 0, y: 0 };
  }

  const bilerp = (values: number[]) => {
    const c00 = values[idx(x0, y0)] ?? 0;
    const c10 = values[idx(x1, y0)] ?? c00;
    const c01 = values[idx(x0, y1)] ?? c00;
    const c11 = values[idx(x1, y1)] ?? c10;
    const c0 = c00 * (1 - tx) + c10 * tx;
    const c1 = c01 * (1 - tx) + c11 * tx;
    return c0 * (1 - ty) + c1 * ty;
  };

  return {
    x: bilerp(field.ux),
    y: bilerp(field.uy),
  };
}

function pointInPolygon(point: Vec2, polygon: number[][]) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0];
    const yi = polygon[i][1];
    const xj = polygon[j][0];
    const yj = polygon[j][1];
    const intersect = ((yi > point.y) !== (yj > point.y))
      && (point.x < ((xj - xi) * (point.y - yi)) / ((yj - yi) || 1e-6) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function distanceToSegment(point: Vec2, a: number[], b: number[]) {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const apx = point.x - a[0];
  const apy = point.y - a[1];
  const ab2 = abx * abx + aby * aby || 1e-6;
  const t = clamp((apx * abx + apy * aby) / ab2, 0, 1);
  const closest = { x: a[0] + abx * t, y: a[1] + aby * t };
  const dx = point.x - closest.x;
  const dy = point.y - closest.y;
  return {
    dist: Math.hypot(dx, dy),
    point: closest,
  };
}

function buildObstacles(buildings: BuildingData[]): FlowObstacle[] {
  return buildings
    .filter((building) => building.footprint.length >= 3)
    .map((building) => {
      let centerX = 0;
      let centerY = 0;
      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;

      for (const [x, y] of building.footprint) {
        centerX += x;
        centerY += y;
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }

      centerX /= building.footprint.length;
      centerY /= building.footprint.length;

      return {
        footprint: building.footprint,
        center: { x: centerX, y: centerY },
        minX,
        maxX,
        minY,
        maxY,
        span: Math.max(8, maxX - minX, maxY - minY),
      };
    });
}

function sampleField(flow: FlowField2DResponse, obstacles: FlowObstacle[], point: Vec2): Vec2 {
  const resolved = sampleResolvedGrid(flow, point);
  if (resolved) {
    return resolved;
  }

  const wind = { x: flow.inlet.ux, y: flow.inlet.uy };
  const speed = Math.max(0.1, length(wind));
  const windHat = normalize(wind);
  const crossHat = { x: -windHat.y, y: windHat.x };
  const velocity = { ...wind };

  for (const obstacle of obstacles) {
    const influencePad = Math.max(18, obstacle.span * 1.6);
    if (
      point.x < obstacle.minX - influencePad
      || point.x > obstacle.maxX + influencePad
      || point.y < obstacle.minY - influencePad
      || point.y > obstacle.maxY + influencePad
    ) {
      continue;
    }

    const rel = { x: point.x - obstacle.center.x, y: point.y - obstacle.center.y };
    const along = rel.x * windHat.x + rel.y * windHat.y;
    const cross = rel.x * crossHat.x + rel.y * crossHat.y;

    if (pointInPolygon(point, obstacle.footprint)) {
      return { x: 0, y: 0 };
    }

    let nearest = { dist: Infinity, point: obstacle.center };
    for (let i = 0; i < obstacle.footprint.length; i += 1) {
      const a = obstacle.footprint[i];
      const b = obstacle.footprint[(i + 1) % obstacle.footprint.length];
      const candidate = distanceToSegment(point, a, b);
      if (candidate.dist < nearest.dist) nearest = candidate;
    }

    if (nearest.dist < 10) {
      const normal = normalize({ x: point.x - nearest.point.x, y: point.y - nearest.point.y });
      const wallFactor = 1 - nearest.dist / 10;
      velocity.x += normal.x * speed * wallFactor * 0.95;
      velocity.y += normal.y * speed * wallFactor * 0.95;
      velocity.x += crossHat.x * Math.sign(cross) * speed * wallFactor * 0.12;
      velocity.y += crossHat.y * Math.sign(cross) * speed * wallFactor * 0.12;
    }

    if (along > 0 && Math.abs(cross) < Math.max(3, obstacle.span * 0.45)) {
      const wakeFalloff = Math.exp(-along / Math.max(14, obstacle.span * 1.6));
      const wakeWidth = Math.exp(-Math.abs(cross) / Math.max(2.5, obstacle.span * 0.3));
      const deficit = wakeFalloff * wakeWidth * 0.5;
      velocity.x -= windHat.x * speed * deficit;
      velocity.y -= windHat.y * speed * deficit;
    }
  }

  return velocity;
}

function scaledWind(wind: Vec2, windScale: number): Vec2 {
  return {
    x: wind.x * windScale,
    y: wind.y * windScale,
  };
}

function computeEnergyRate(velocity: Vec2, wind: Vec2): number {
  return calculateDragEnergy(
    { x: velocity.x, y: 0, z: velocity.y },
    { x: wind.x, y: 0, z: wind.y },
  ).consumptionRate;
}

function lerpColor(
  a: [number, number, number],
  b: [number, number, number],
  t: number,
): [number, number, number] {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

function speedToRgb(speed: number, maxSpeed: number): [number, number, number] {
  const t = clamp(speed / Math.max(0.1, maxSpeed), 0, 1);
  if (t < 0.5) {
    return lerpColor([10, 35, 52], [34, 184, 255], t / 0.5);
  }
  if (t < 0.82) {
    return lerpColor([34, 184, 255], [255, 138, 42], (t - 0.5) / 0.32);
  }
  return lerpColor([255, 138, 42], [255, 246, 220], (t - 0.82) / 0.18);
}

function buildSpeedColormap(flow: FlowField2DResponse): HTMLCanvasElement | null {
  if (!hasResolvedGrid(flow) || typeof document === 'undefined') {
    return null;
  }

  const { field } = flow;
  const canvas = document.createElement('canvas');
  canvas.width = field.nx;
  canvas.height = field.ny;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return null;
  }

  const maxSpeed = Math.max(0.1, field.stats.max_speed_mps);
  const image = ctx.createImageData(field.nx, field.ny);
  for (let ix = 0; ix < field.nx; ix += 1) {
    for (let iy = 0; iy < field.ny; iy += 1) {
      const sourceIndex = ix * field.ny + iy;
      const targetIndex = ((field.ny - 1 - iy) * field.nx + ix) * 4;
      if ((field.mask[sourceIndex] ?? 0) > 0) {
        image.data[targetIndex] = 255;
        image.data[targetIndex + 1] = 255;
        image.data[targetIndex + 2] = 255;
        image.data[targetIndex + 3] = 0;
        continue;
      }

      const ux = field.ux[sourceIndex] ?? 0;
      const uy = field.uy[sourceIndex] ?? 0;
      const [r, g, b] = speedToRgb(Math.hypot(ux, uy), maxSpeed);
      image.data[targetIndex] = r;
      image.data[targetIndex + 1] = g;
      image.data[targetIndex + 2] = b;
      image.data[targetIndex + 3] = 188;
    }
  }

  ctx.putImageData(image, 0, 0);
  return canvas;
}

function worldToCanvas(
  point: Vec2,
  width: number,
  height: number,
  bounds: FlowField2DResponse['field']['bounds'],
) {
  const worldWidth = bounds.max_x - bounds.min_x;
  const worldHeight = bounds.max_y - bounds.min_y;
  const scale = Math.min(width / worldWidth, height / worldHeight);
  const offsetX = (width - worldWidth * scale) * 0.5;
  const offsetY = (height - worldHeight * scale) * 0.5;
  const x = offsetX + (point.x - bounds.min_x) * scale;
  const y = height - offsetY - (point.y - bounds.min_y) * scale;
  return { x, y };
}

function canvasToWorld(
  point: Vec2,
  width: number,
  height: number,
  bounds: FlowField2DResponse['field']['bounds'],
) {
  const worldWidth = bounds.max_x - bounds.min_x;
  const worldHeight = bounds.max_y - bounds.min_y;
  const scale = Math.min(width / worldWidth, height / worldHeight);
  const offsetX = (width - worldWidth * scale) * 0.5;
  const offsetY = (height - worldHeight * scale) * 0.5;
  const x = bounds.min_x + (point.x - offsetX) / scale;
  const y = bounds.min_y + (height - offsetY - point.y) / scale;
  return { x, y };
}

function drawDrone(ctx: CanvasRenderingContext2D, canvasPoint: Vec2, heading: number) {
  ctx.save();
  ctx.translate(canvasPoint.x, canvasPoint.y);
  ctx.rotate(-heading + Math.PI / 2);
  ctx.beginPath();
  ctx.moveTo(0, -14);
  ctx.lineTo(10, 10);
  ctx.lineTo(0, 5);
  ctx.lineTo(-10, 10);
  ctx.closePath();
  ctx.fillStyle = '#ffffff';
  ctx.strokeStyle = '#0f1114';
  ctx.lineWidth = 2;
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function createParticle(bounds: FlowField2DResponse['field']['bounds'], flow: FlowField2DResponse): FlowParticle {
  const windHat = normalize({ x: flow.inlet.ux, y: flow.inlet.uy });
  const edgeChoice = Math.random();
  let point: Vec2;
  if (Math.abs(windHat.x) > Math.abs(windHat.y)) {
    const x = windHat.x >= 0 ? bounds.min_x + 8 : bounds.max_x - 8;
    point = { x, y: bounds.min_y + Math.random() * (bounds.max_y - bounds.min_y) };
  } else {
    const y = windHat.y >= 0 ? bounds.min_y + 8 : bounds.max_y - 8;
    point = { x: bounds.min_x + Math.random() * (bounds.max_x - bounds.min_x), y };
  }
  if (edgeChoice > 0.8) {
    point = {
      x: bounds.min_x + Math.random() * (bounds.max_x - bounds.min_x),
      y: bounds.min_y + Math.random() * (bounds.max_y - bounds.min_y),
    };
  }
  return {
    position: point,
    age: Math.random() * 4,
    life: 4 + Math.random() * 5,
    hue: 185 + Math.random() * 35,
    width: 0.6 + Math.random() * 1.8,
  };
}

function traceStreamline(
  flow: FlowField2DResponse,
  obstacles: FlowObstacle[],
  seed: Vec2,
  bounds: FlowField2DResponse['field']['bounds'],
  direction: 1 | -1,
  stepLength: number,
  maxSteps: number,
) {
  const points: Vec2[] = [];
  const speeds: number[] = [];
  let current = seed;

  for (let step = 0; step < maxSteps; step += 1) {
    const local = sampleField(flow, obstacles, current);
    const speed = length(local);
    if (speed < 0.035) {
      break;
    }

    const dir = normalize(local);
    const next = {
      x: current.x + dir.x * stepLength * direction,
      y: current.y + dir.y * stepLength * direction,
    };

    if (!isInsideBounds(next, bounds)) {
      break;
    }
    if (obstacles.some((obstacle) => pointInPolygon(next, obstacle.footprint))) {
      break;
    }

    points.push(next);
    speeds.push(speed);
    current = next;
  }

  return { points, speeds };
}

function buildStreamlines(
  flow: FlowField2DResponse,
  obstacles: FlowObstacle[],
  bounds: FlowField2DResponse['field']['bounds'],
): Streamline[] {
  const spanX = bounds.max_x - bounds.min_x;
  const spanY = bounds.max_y - bounds.min_y;
  const majorSpan = Math.max(spanX, spanY);
  const columns = 10;
  const rows = 13;
  const stepLength = Math.max(3.5, majorSpan / 155);
  const maxSteps = 92;
  const lines: Streamline[] = [];

  for (let ix = 0; ix < columns; ix += 1) {
    for (let iy = 0; iy < rows; iy += 1) {
      const jitterX = (((ix * 37 + iy * 17) % 19) - 9) / 19;
      const jitterY = (((ix * 13 + iy * 29) % 23) - 11) / 23;
      const seed = {
        x: bounds.min_x + ((ix + 0.5 + jitterX * 0.24) / columns) * spanX,
        y: bounds.min_y + ((iy + 0.5 + jitterY * 0.24) / rows) * spanY,
      };

      if (obstacles.some((obstacle) => pointInPolygon(seed, obstacle.footprint))) {
        continue;
      }

      const seedWind = sampleField(flow, obstacles, seed);
      if (length(seedWind) < 0.04) {
        continue;
      }

      const backward = traceStreamline(flow, obstacles, seed, bounds, -1, stepLength, maxSteps);
      const forward = traceStreamline(flow, obstacles, seed, bounds, 1, stepLength, maxSteps);
      const points = [...backward.points.reverse(), seed, ...forward.points];
      if (points.length < 10) {
        continue;
      }

      const speeds = [...backward.speeds, length(seedWind), ...forward.speeds];
      const meanSpeed = speeds.reduce((sum, value) => sum + value, 0) / Math.max(1, speeds.length);
      lines.push({ points, speed: meanSpeed });
    }
  }

  return lines;
}

function resetWorld(flow: FlowField2DResponse | null): WorldState {
  if (!flow) {
    return { position: { x: 0, y: 0 }, velocity: { x: 0, y: 0 }, heading: 0, energyUsed: 0 };
  }

  const found = selectLidar2DSpawn(
    flow.field.bounds,
    flow.buildings,
    collisionRadiusForFlow(flow) + 4,
    DEFAULT_LIDAR_2D_CONFIG,
  );

  return {
    position: found,
    velocity: { x: 0, y: 0 },
    heading: 0,
    energyUsed: 0,
  };
}

function collisionRadiusForFlow(flow: FlowField2DResponse | null) {
  return flow?.live_scenario?.collision_lidar_semantics.agent_radius_m ?? LEGACY_DRONE_RADIUS_M;
}

export default function TopDownGame({
  flow,
  showFlowAnimation,
  flowVisualization = 'arrows',
  windScale = 1,
  showLidar = true,
  onTelemetry,
}: TopDownGameProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const physicsAccumulatorRef = useRef(0);
  const worldRef = useRef<WorldState>(resetWorld(flow));
  const keysRef = useRef<Set<string>>(new Set());
  const particlesRef = useRef<FlowParticle[]>([]);
  const trailRef = useRef<Vec2[]>([]);
  const flowLayerRef = useRef<HTMLCanvasElement | null>(null);
  const lidarClockRef = useRef(LIDAR_INTERVAL_S);
  const lidarScanRef = useRef<Lidar2DScan | null>(null);
  const lidarTelemetryRef = useRef<LidarTelemetry | null>(null);

  const buildings = useMemo(() => flow?.buildings ?? [], [flow]);
  const obstacles = useMemo(() => buildObstacles(buildings), [buildings]);
  const collisionRadius = collisionRadiusForFlow(flow);
  const lidarDirections = useMemo(
    () => showLidar ? createLidar2DLocalDirections(DEFAULT_LIDAR_2D_CONFIG) : [],
    [showLidar],
  );
  const bounds = flow?.field.bounds;
  const speedColormap = useMemo(() => (flow ? buildSpeedColormap(flow) : null), [flow]);
  const viewportBounds = useMemo(() => {
    if (!flow) {
      return null;
    }
    if (flow.field.bounds) {
      return flow.field.bounds;
    }
    return {
      min_x: -VIEW_RADIUS_M,
      max_x: VIEW_RADIUS_M,
      min_y: -VIEW_RADIUS_M,
      max_y: VIEW_RADIUS_M,
    };
  }, [flow]);

  const fieldPreview = useMemo(() => {
    if (!flow || !viewportBounds || flowVisualization !== 'arrows') {
      return [];
    }

    const span = Math.max(
      viewportBounds.max_x - viewportBounds.min_x,
      viewportBounds.max_y - viewportBounds.min_y,
    );
    const step = Math.max(12, span / 46);
    const arrows: Array<{ x: number; y: number; vx: number; vy: number }> = [];

    for (let x = viewportBounds.min_x + step * 0.5; x <= viewportBounds.max_x; x += step) {
      for (let y = viewportBounds.min_y + step * 0.5; y <= viewportBounds.max_y; y += step) {
        if (obstacles.some((obstacle) => pointInPolygon({ x, y }, obstacle.footprint))) continue;
        const local = scaledWind(sampleField(flow, obstacles, { x, y }), windScale);
        arrows.push({
          x,
          y,
          vx: local.x,
          vy: local.y,
        });
      }
    }

    return arrows;
  }, [flow, flowVisualization, obstacles, viewportBounds, windScale]);

  const streamlines = useMemo(() => {
    if (!flow || !viewportBounds || (flowVisualization !== 'streamlines' && flowVisualization !== 'both')) {
      return [];
    }
    return buildStreamlines(flow, obstacles, viewportBounds);
  }, [flow, flowVisualization, obstacles, viewportBounds]);

  useEffect(() => {
    worldRef.current = resetWorld(flow);
    trailRef.current = [worldRef.current.position];
    particlesRef.current = flow && viewportBounds
      ? Array.from({ length: PARTICLE_COUNT }, () => createParticle(viewportBounds, flow))
      : [];
    lidarClockRef.current = LIDAR_INTERVAL_S;
    lidarScanRef.current = null;
    lidarTelemetryRef.current = null;
  }, [flow, viewportBounds]);

  useEffect(() => {
    const isInteractiveTarget = (target: EventTarget | null) => {
      if (!(target instanceof Element)) return false;
      if (target instanceof HTMLElement && target.isContentEditable) return true;
      return target.closest([
        'a', 'area[href]', 'button', 'input', 'label', 'select', 'textarea',
        'summary', 'iframe', 'object', 'embed', 'audio[controls]', 'video[controls]',
        '[contenteditable]:not([contenteditable="false"])',
        '[role="button"]', '[role="link"]', '[role="checkbox"]', '[role="radio"]',
        '[role="switch"]', '[role="textbox"]', '[role="combobox"]',
        '[role="slider"]', '[role="spinbutton"]', '[role="menuitem"]',
        '[tabindex]:not([tabindex="-1"])',
      ].join(',')) !== null;
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (!FLIGHT_CONTROL_CODES.has(event.code) || isInteractiveTarget(event.target)) return;
      event.preventDefault();
      keysRef.current.add(event.code);
    };
    const handleKeyUp = (event: KeyboardEvent) => {
      if (!FLIGHT_CONTROL_CODES.has(event.code)) return;
      if (!isInteractiveTarget(event.target)) event.preventDefault();
      keysRef.current.delete(event.code);
    };
    const clearControls = () => keysRef.current.clear();
    const handleVisibilityChange = () => {
      if (document.visibilityState !== 'visible') clearControls();
    };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('blur', clearControls);
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('blur', clearControls);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      clearControls();
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext('2d');
    if (!context) {
      return;
    }
    const flowLayer = document.createElement('canvas');
    const flowContext = flowLayer.getContext('2d');
    if (!flowContext) {
      return;
    }
    flowLayerRef.current = flowLayer;
    lastTimeRef.current = 0;
    physicsAccumulatorRef.current = 0;

    const resize = () => {
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(canvas.clientWidth * ratio);
      canvas.height = Math.floor(canvas.clientHeight * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      flowLayer.width = canvas.width;
      flowLayer.height = canvas.height;
      flowContext.setTransform(ratio, 0, 0, ratio, 0, 0);
      flowContext.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    };

    resize();
    window.addEventListener('resize', resize);

    const handleClick = (event: MouseEvent) => {
      if (!flow || !viewportBounds) {
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const clickPoint = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
      const worldPoint = canvasToWorld(clickPoint, canvas.clientWidth, canvas.clientHeight, viewportBounds);

      if (
        worldPoint.x < viewportBounds.min_x + collisionRadius
        || worldPoint.x > viewportBounds.max_x - collisionRadius
        || worldPoint.y < viewportBounds.min_y + collisionRadius
        || worldPoint.y > viewportBounds.max_y - collisionRadius
      ) {
        return;
      }

      const blocked = isFlight2DPositionBlocked(worldPoint, obstacles, collisionRadius);
      if (blocked) {
        return;
      }

      worldRef.current.position = worldPoint;
      worldRef.current.velocity = { x: 0, y: 0 };
      physicsAccumulatorRef.current = 0;
      trailRef.current = [worldPoint];
    };

    canvas.addEventListener('click', handleClick);

    const frame = (time: number) => {
      animationRef.current = window.requestAnimationFrame(frame);
      const frameDelta = (time - lastTimeRef.current) / 1000 || 1 / 60;
      const renderDelta = Math.min(0.05, Math.max(0, frameDelta));
      lastTimeRef.current = time;

      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      context.clearRect(0, 0, width, height);

      const gradient = context.createLinearGradient(0, 0, width, height);
      gradient.addColorStop(0, '#1c2024');
      gradient.addColorStop(1, '#0f1114');
      context.fillStyle = gradient;
      context.fillRect(0, 0, width, height);

      if (!flow || !bounds || !viewportBounds) {
        context.fillStyle = '#f3f5f7';
        context.font = '600 22px ui-sans-serif, system-ui';
        context.fillText('Select a location to load geometry and wind.', 32, 48);
        return;
      }

      const world = worldRef.current;
      const input = {
        x: (keysRef.current.has('KeyD') ? 1 : 0) - (keysRef.current.has('KeyA') ? 1 : 0),
        y: (keysRef.current.has('KeyW') ? 1 : 0) - (keysRef.current.has('KeyS') ? 1 : 0),
      };
      let wind = scaledWind(sampleField(flow, obstacles, world.position), windScale);
      let energyRate = computeEnergyRate(world.velocity, wind);
      const fixedStepResult = consumeFlight2DFixedSteps(
        physicsAccumulatorRef.current,
        frameDelta,
        (fixedDelta) => {
          wind = scaledWind(sampleField(flow, obstacles, world.position), windScale);
          const nextMotion = stepFlight2DMotion(
            world,
            input,
            wind,
            obstacles,
            viewportBounds,
            fixedDelta,
            { collisionRadius },
          );
          world.position = nextMotion.position;
          world.velocity = nextMotion.velocity;
          world.heading = nextMotion.heading;
          energyRate = computeEnergyRate(world.velocity, wind);
          world.energyUsed += energyRate * fixedDelta;
        },
      );
      physicsAccumulatorRef.current = fixedStepResult.accumulatorSeconds;
      wind = scaledWind(sampleField(flow, obstacles, world.position), windScale);
      energyRate = computeEnergyRate(world.velocity, wind);

      if (showLidar) {
        lidarClockRef.current += fixedStepResult.simulatedSeconds;
        if (lidarClockRef.current >= LIDAR_INTERVAL_S) {
          lidarClockRef.current %= LIDAR_INTERVAL_S;
          const scan = scanLidar2D(
            { position: world.position, heading: world.heading },
            buildings,
            DEFAULT_LIDAR_2D_CONFIG,
            lidarDirections,
            viewportBounds,
          );
          lidarScanRef.current = scan;
          lidarTelemetryRef.current = summarizeLidar2DScan(scan, {
            x: world.position.x,
            y: 0,
            z: world.position.y,
            yaw: -world.heading,
            pitch: 0,
            roll: 0,
          });
        }
      } else {
        if (lidarScanRef.current || lidarTelemetryRef.current) {
          lidarScanRef.current = null;
          lidarTelemetryRef.current = null;
        }
        lidarClockRef.current = LIDAR_INTERVAL_S;
      }

      const trail = trailRef.current;
      const lastTrailPoint = trail.at(-1);
      if (!lastTrailPoint || Math.hypot(world.position.x - lastTrailPoint.x, world.position.y - lastTrailPoint.y) > 1.8) {
        trail.push({ ...world.position });
        if (trail.length > 180) trail.shift();
      }

      const worldWidth = viewportBounds.max_x - viewportBounds.min_x;
      const worldHeight = viewportBounds.max_y - viewportBounds.min_y;
      const scale = Math.min(width / worldWidth, height / worldHeight);
      const offsetX = (width - worldWidth * scale) * 0.5;
      const offsetY = (height - worldHeight * scale) * 0.5;

      context.fillStyle = 'rgba(255, 255, 255, 0.055)';
      context.fillRect(offsetX, offsetY, worldWidth * scale, worldHeight * scale);

      if (showFlowAnimation && speedColormap && (flowVisualization === 'colormap' || flowVisualization === 'both')) {
        context.save();
        context.globalAlpha = 0.86;
        context.globalCompositeOperation = 'screen';
        context.imageSmoothingEnabled = true;
        context.drawImage(speedColormap, offsetX, offsetY, worldWidth * scale, worldHeight * scale);
        context.restore();
      }

      if (showFlowAnimation && flowVisualization === 'arrows') {
        flowContext.save();
        flowContext.globalCompositeOperation = 'source-over';
        flowContext.fillStyle = 'rgba(15, 17, 20, 0.11)';
        flowContext.fillRect(0, 0, width, height);
        flowContext.globalCompositeOperation = 'lighter';
        for (const particle of particlesRef.current) {
          const local = scaledWind(sampleField(flow, obstacles, particle.position), windScale);
          const mag = length(local);
          particle.position.x += local.x * renderDelta * 8.5;
          particle.position.y += local.y * renderDelta * 8.5;
          particle.age += renderDelta;

          const outOfBounds = (
            particle.position.x < viewportBounds.min_x || particle.position.x > viewportBounds.max_x
            || particle.position.y < viewportBounds.min_y || particle.position.y > viewportBounds.max_y
          );
          const blocked = obstacles.some((obstacle) => pointInPolygon(particle.position, obstacle.footprint));
          if (outOfBounds || blocked || particle.age > particle.life) {
            const next = createParticle(viewportBounds, flow);
            particle.position = next.position;
            particle.age = next.age;
            particle.life = next.life;
            particle.hue = next.hue;
            particle.width = next.width;
            continue;
          }

          const start = worldToCanvas(particle.position, width, height, viewportBounds);
          const prev = worldToCanvas(
            { x: particle.position.x - local.x * (1.4 + mag * 0.08), y: particle.position.y - local.y * (1.4 + mag * 0.08) },
            width,
            height,
            viewportBounds,
          );
          const alpha = Math.min(0.35, 0.08 + mag / 18);
          const hue = mag > 1.2 ? 200 : 28;
          const light = Math.min(68, 44 + mag * 5);
          flowContext.strokeStyle = `hsla(${hue}, 100%, ${light}%, ${Math.min(0.42, alpha + 0.08)})`;
          flowContext.lineWidth = particle.width + Math.min(1.8, mag * 0.08);
          flowContext.beginPath();
          flowContext.moveTo(prev.x, prev.y);
          flowContext.lineTo(start.x, start.y);
          flowContext.stroke();

          if (mag > 2.5) {
            flowContext.fillStyle = `hsla(200, 100%, 72%, ${Math.min(0.24, alpha * 0.9)})`;
            flowContext.beginPath();
            flowContext.arc(start.x, start.y, 1.2 + Math.min(1.8, mag * 0.06), 0, Math.PI * 2);
            flowContext.fill();
          }
        }
        flowContext.restore();
        context.drawImage(flowLayer, 0, 0, width, height);
      } else {
        flowContext.clearRect(0, 0, width, height);
      }

      if (flowVisualization === 'arrows') {
        const windRibbonCount = 16;
        for (let i = 0; i < windRibbonCount; i += 1) {
          const ribbonY = offsetY + ((i + 0.5) / windRibbonCount) * worldHeight * scale;
          const shimmer = ((time * 0.04) + i * 17) % (worldWidth * scale + 140);
          context.strokeStyle = 'rgba(255, 138, 42, 0.13)';
          context.lineWidth = 1;
          context.beginPath();
          context.moveTo(offsetX + shimmer - 90, ribbonY);
          context.lineTo(offsetX + shimmer, ribbonY);
          context.stroke();
        }
      }

      context.strokeStyle = 'rgba(255, 255, 255, 0.09)';
      context.lineWidth = 1;
      for (let i = 1; i < 8; i += 1) {
        const x = offsetX + ((worldWidth * scale) / 8) * i;
        const y = offsetY + ((worldHeight * scale) / 8) * i;
        context.beginPath();
        context.moveTo(x, offsetY);
        context.lineTo(x, offsetY + worldHeight * scale);
        context.stroke();
        context.beginPath();
        context.moveTo(offsetX, y);
        context.lineTo(offsetX + worldWidth * scale, y);
        context.stroke();
      }

      if (showFlowAnimation && (flowVisualization === 'streamlines' || flowVisualization === 'both')) {
        context.save();
        context.globalCompositeOperation = 'lighter';
        context.lineCap = 'round';
        context.lineJoin = 'round';
        context.setLineDash([16, 13]);
        context.lineDashOffset = -time * 0.035;
        const maxSpeed = Math.max(0.1, flow.field.stats.max_speed_mps);

        streamlines.forEach((line) => {
          const t = clamp(line.speed / maxSpeed, 0, 1);
          const hue = 202 - t * 174;
          const alpha = 0.28 + t * 0.34;
          context.strokeStyle = `hsla(${hue}, 100%, ${58 + t * 12}%, ${alpha})`;
          context.lineWidth = 0.9 + t * 2.3;
          context.beginPath();
          line.points.forEach((point, index) => {
            const canvasPoint = worldToCanvas(point, width, height, viewportBounds);
            if (index === 0) {
              context.moveTo(canvasPoint.x, canvasPoint.y);
            } else {
              context.lineTo(canvasPoint.x, canvasPoint.y);
            }
          });
          context.stroke();
        });
        context.restore();
      }

      fieldPreview.forEach((arrow) => {
        const origin = worldToCanvas({ x: arrow.x, y: arrow.y }, width, height, viewportBounds);
        const mag = Math.hypot(arrow.vx, arrow.vy);
        if (mag < 0.25) return;
        const arrowScale = Math.min(18, 6 + mag * 1.8);
        const dir = normalize({ x: arrow.vx, y: arrow.vy });
        const tip = { x: origin.x + dir.x * arrowScale, y: origin.y - dir.y * arrowScale };
        const arrowHue = mag > 1.4 ? 200 : 28;
        context.strokeStyle = `hsla(${arrowHue}, 100%, 62%, 0.84)`;
        context.lineWidth = 1.35;
        context.beginPath();
        context.moveTo(origin.x, origin.y);
        context.lineTo(tip.x, tip.y);
        context.stroke();
        context.beginPath();
        context.moveTo(tip.x, tip.y);
        context.lineTo(tip.x - dir.x * 4.5 - dir.y * 3, tip.y + dir.y * 4.5 - dir.x * 3);
        context.lineTo(tip.x - dir.x * 4.5 + dir.y * 3, tip.y + dir.y * 4.5 + dir.x * 3);
        context.closePath();
        context.fillStyle = context.strokeStyle;
        context.fill();
      });

      if (trailRef.current.length > 1) {
        context.save();
        context.strokeStyle = 'rgba(34, 184, 255, 0.72)';
        context.lineWidth = 2;
        context.beginPath();
        trailRef.current.forEach((point, index) => {
          const canvasPoint = worldToCanvas(point, width, height, viewportBounds);
          if (index === 0) context.moveTo(canvasPoint.x, canvasPoint.y);
          else context.lineTo(canvasPoint.x, canvasPoint.y);
        });
        context.stroke();
        context.restore();
      }

      buildings.forEach((building: BuildingData) => {
        if (!building.footprint.length) return;
        context.beginPath();
        building.footprint.forEach(([x, y], index) => {
          const point = worldToCanvas({ x, y }, width, height, viewportBounds);
          if (index === 0) context.moveTo(point.x, point.y);
          else context.lineTo(point.x, point.y);
        });
        context.closePath();
        context.fillStyle = '#e7e8e4';
        context.strokeStyle = '#1f252a';
        context.lineWidth = 1.2;
        context.fill();
        context.stroke();
      });

      if (showLidar && lidarScanRef.current) {
        context.save();
        context.globalCompositeOperation = 'lighter';
        lidarScanRef.current.samples.forEach((sample) => {
          if (!sample.hit || !sample.point) return;
          const point = worldToCanvas(sample.point, width, height, viewportBounds);
          context.fillStyle = getLidarJetCss(sample.normalizedDistance);
          context.globalAlpha = 0.92;
          context.beginPath();
          context.arc(point.x, point.y, 2.2, 0, Math.PI * 2);
          context.fill();
        });
        context.restore();
      }

      const droneCanvasPoint = worldToCanvas(world.position, width, height, viewportBounds);
      drawDrone(context, droneCanvasPoint, world.heading);

      const northOrigin = { x: offsetX + 28, y: offsetY + 54 };
      context.strokeStyle = '#ffffff';
      context.lineWidth = 2;
      context.beginPath();
      context.moveTo(northOrigin.x, northOrigin.y);
      context.lineTo(northOrigin.x, northOrigin.y - 28);
      context.stroke();
      context.beginPath();
      context.moveTo(northOrigin.x, northOrigin.y - 28);
      context.lineTo(northOrigin.x - 5, northOrigin.y - 20);
      context.lineTo(northOrigin.x + 5, northOrigin.y - 20);
      context.closePath();
      context.fillStyle = '#ffffff';
      context.fill();
      context.fillStyle = 'rgba(255,255,255,0.85)';
      context.font = '600 13px ui-sans-serif, system-ui';
      context.fillText('N', northOrigin.x - 5, northOrigin.y - 36);

      onTelemetry({
        droneSpeed: length(world.velocity),
        localWindSpeed: length(wind),
        localWindDirDeg: eastNorthToCompassBearingDeg(wind.x, wind.y),
        energyRate,
        energyUsed: world.energyUsed,
        headingDeg: mapAngleToCompassBearingDeg(world.heading),
        position: { ...world.position },
        displayPose: {
          x: world.position.x,
          y: 0,
          z: world.position.y,
          yaw: -world.heading,
          pitch: 0,
          roll: 0,
        },
        lidar: showLidar ? lidarTelemetryRef.current ?? undefined : undefined,
      });
    };

    animationRef.current = window.requestAnimationFrame(frame);

    return () => {
      window.removeEventListener('resize', resize);
      canvas.removeEventListener('click', handleClick);
      flowLayerRef.current = null;
      if (animationRef.current !== null) {
        window.cancelAnimationFrame(animationRef.current);
      }
    };
  }, [
    bounds,
    buildings,
    collisionRadius,
    fieldPreview,
    flow,
    flowVisualization,
    lidarDirections,
    obstacles,
    onTelemetry,
    showFlowAnimation,
    showLidar,
    speedColormap,
    streamlines,
    viewportBounds,
    windScale,
  ]);

  return <canvas ref={canvasRef} className="game-canvas" />;
}
