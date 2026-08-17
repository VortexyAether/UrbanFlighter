import type {
    UrbanFlowInspectorFrame,
    UrbanFlowInspectorSessionResponse,
    UrbanFlowInspectorWorld,
} from '../api';

export interface InspectorPoint {
    x: number;
    y: number;
}

export interface InspectorViewport {
    width: number;
    height: number;
    padding: number;
    scale: number;
    offsetX: number;
    offsetY: number;
}

export interface EpisodeInspectorRenderModel {
    viewport: InspectorViewport;
    buildings: Array<{
        buildingId: string;
        heightM: number;
        points: InspectorPoint[];
    }>;
    start: InspectorPoint;
    goal: InspectorPoint;
    drone: InspectorPoint;
    trajectory: InspectorPoint[];
    lidarRays: Array<{
        start: InspectorPoint;
        end: InspectorPoint;
        hit: boolean;
    }>;
    headingEnd: InspectorPoint;
    actionEnd: InspectorPoint;
}

type InspectorBounds = UrbanFlowInspectorWorld['bounds'];

export function createInspectorViewport(
    bounds: InspectorBounds,
    width: number,
    height: number,
    padding: number = 26,
): InspectorViewport {
    const worldWidth = bounds.max_x_m - bounds.min_x_m;
    const worldHeight = bounds.max_y_m - bounds.min_y_m;
    if (
        !Number.isFinite(worldWidth)
        || !Number.isFinite(worldHeight)
        || worldWidth <= 0
        || worldHeight <= 0
        || !Number.isFinite(width)
        || !Number.isFinite(height)
        || width <= padding * 2
        || height <= padding * 2
    ) {
        throw new Error('Inspector viewport requires finite positive world and display bounds.');
    }
    const scale = Math.min(
        (width - padding * 2) / worldWidth,
        (height - padding * 2) / worldHeight,
    );
    return {
        width,
        height,
        padding,
        scale,
        offsetX: (width - worldWidth * scale) / 2,
        offsetY: (height - worldHeight * scale) / 2,
    };
}

export function worldToInspectorPoint(
    point: readonly number[],
    bounds: InspectorBounds,
    viewport: InspectorViewport,
): InspectorPoint {
    if (point.length !== 2 || point.some((value) => !Number.isFinite(value))) {
        throw new Error('Inspector world point must be a finite [x, y] vector.');
    }
    return {
        x: viewport.offsetX + (point[0] - bounds.min_x_m) * viewport.scale,
        // North-up means increasing local north goes toward the top of the SVG.
        y: viewport.height - viewport.offsetY - (point[1] - bounds.min_y_m) * viewport.scale,
    };
}

export function localActionToWorldVector(
    action: readonly number[],
    headingRad: number,
): [number, number] {
    if (
        action.length !== 2
        || action.some((value) => !Number.isFinite(value))
        || !Number.isFinite(headingRad)
    ) {
        throw new Error('Inspector action transform requires a finite local action and heading.');
    }
    const cosine = Math.cos(headingRad);
    const sine = Math.sin(headingRad);
    return [
        action[0] * cosine - action[1] * sine,
        action[0] * sine + action[1] * cosine,
    ];
}

export function buildEpisodeInspectorRenderModel(
    world: UrbanFlowInspectorWorld,
    frame: UrbanFlowInspectorFrame,
    width: number = 1_000,
    height: number = 620,
): EpisodeInspectorRenderModel {
    if (world.scenario_id !== frame.scenario_id) {
        throw new Error('Inspector world and frame scenario identities do not match.');
    }
    const viewport = createInspectorViewport(world.bounds, width, height);
    const toScreen = (point: readonly number[]) => worldToInspectorPoint(point, world.bounds, viewport);
    const drone = toScreen(frame.drone.position_xy_m);
    const domainDiagonal = Math.hypot(
        world.bounds.max_x_m - world.bounds.min_x_m,
        world.bounds.max_y_m - world.bounds.min_y_m,
    );
    const headingLengthM = Math.min(24, domainDiagonal * 0.045);
    const headingWorldEnd: [number, number] = [
        frame.drone.position_xy_m[0] + Math.cos(frame.drone.heading_rad) * headingLengthM,
        frame.drone.position_xy_m[1] + Math.sin(frame.drone.heading_rad) * headingLengthM,
    ];
    const actionWorld = localActionToWorldVector(
        frame.local_guidance_action.vector,
        frame.drone.heading_rad,
    );
    const actionLengthM = Math.min(34, domainDiagonal * 0.065);
    const actionWorldEnd: [number, number] = [
        frame.drone.position_xy_m[0] + actionWorld[0] * actionLengthM,
        frame.drone.position_xy_m[1] + actionWorld[1] * actionLengthM,
    ];

    return {
        viewport,
        buildings: world.buildings.map((building) => ({
            buildingId: building.building_id,
            heightM: building.height_m,
            points: building.footprint_xy_m.map(toScreen),
        })),
        start: toScreen(frame.start_xy_m),
        goal: toScreen(frame.goal_xy_m),
        drone,
        trajectory: frame.trajectory_xy_m.map(toScreen),
        lidarRays: frame.actor_lidar.rays.map((ray) => ({
            start: drone,
            end: toScreen(ray.endpoint_xy_m),
            hit: ray.hit,
        })),
        headingEnd: toScreen(headingWorldEnd),
        actionEnd: toScreen(actionWorldEnd),
    };
}

export function inspectorResponseMatchesSelection(
    result: UrbanFlowInspectorSessionResponse,
    expectedSelectionIdentity: string,
    currentSelectionIdentity: string,
    expectedScenarioId: string,
    expectedSessionId: string | null = null,
) {
    return currentSelectionIdentity === expectedSelectionIdentity
        && result.scenario_id === expectedScenarioId
        && result.frame.scenario_id === expectedScenarioId
        && (result.world === undefined || result.world.scenario_id === expectedScenarioId)
        && (expectedSessionId === null || result.session_id === expectedSessionId);
}

export function inspectorSessionIsStale(
    sessionScenarioId: string,
    selectedScenarioId: string | null,
    expectedSelectionIdentity: string,
    currentSelectionIdentity: string,
) {
    return selectedScenarioId === null
        || sessionScenarioId !== selectedScenarioId
        || expectedSelectionIdentity !== currentSelectionIdentity;
}
