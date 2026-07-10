export const API_URL = 'http://localhost:8000';

export interface BackendHealth {
    status: string;
    service?: string;
}

type JsonMetricValue = string | number | boolean | null | number[] | Record<string, number>;

export interface PolicyFrame {
    step: number;
    t_s?: number;
    drone_id?: string;
    position: number[];
    velocity?: number[];
    action?: number[];
    policy_observation: number[];
    reward?: number;
    reward_terms?: Record<string, number>;
    terminated?: boolean;
    truncated?: boolean;
    policy_had_privileged_flow_access: boolean;
}

export interface RLEnvironmentSpec {
    status: string;
    policy_status: string;
    environment: {
        id: string;
        gymnasium_installed: boolean;
        reward_terms: string[];
        cost_metrics: string[];
        policy_observation_contract: {
            fields: string[];
            source: string;
            privileged_flow_access: boolean;
            hidden_dynamics: string;
            forbidden: string[];
        };
        observation_space: {
            shape: number[];
            provider: string;
        };
        action_space: {
            shape: number[];
            provider: string;
        };
    };
    policy_observation_summary: string;
    policy_had_privileged_flow_access: boolean;
    data_sources: {
        world: string;
        wind: {
            kind: string;
            cfd_claim: string;
        };
        hidden_training_dynamics: string;
        real_cfd_eval_hook: string;
    };
}

export interface RLBaselineResponse {
    status: string;
    policy_label: string;
    environment_id: string;
    seed: number;
    max_steps: number;
    n_drones: number;
    randomize_missions?: boolean;
    policy_had_privileged_flow_access: boolean;
    metrics: {
        success: boolean;
        success_count?: number;
        steps: number;
        return: number;
        path_length_m: number;
        energy_relative_airspeed_l2: number;
        collisions: number;
        separation_violations?: number;
        min_building_clearance_m: number;
        min_pairwise_separation_m?: number;
        final_distance_m: number;
        reward_terms_total: Record<string, number>;
        controller: string;
        policy_status: string;
        waypoint_count: number;
    };
    cost_metrics: {
        collisions: number;
        separation_violations?: number;
        min_building_clearance_m: number;
        min_pairwise_separation_m?: number;
        energy_relative_airspeed_l2: number;
        path_length_m?: number;
        boundary_violations?: number;
        swept_building_hits?: number;
        final_distance_m: number;
    };
    missions: Array<{
        drone_id: string;
        start: number[];
        goal: number[];
    }>;
    drones?: Array<{
        drone_id: string;
        metrics: Record<string, JsonMetricValue>;
        trajectory: PolicyFrame[];
        waypoints: number[][];
    }>;
    trajectory: PolicyFrame[];
    data_sources: {
        world: {
            building_count: number;
        };
        wind: {
            kind: string;
            cfd_claim: string;
        };
        hidden_training_dynamics: string;
    };
}

export interface BuildingData {
    height: number;
    footprint: number[][]; // [[x, y], [x, y], ...]
}

export interface MapData {
    features: BuildingData[];
    count: number;
    message?: string;
}

export interface FlowFieldGrid {
    nx: number;
    ny: number;
    cell_size_m: number;
    bounds: {
        min_x: number;
        max_x: number;
        min_y: number;
        max_y: number;
    };
    ux: number[];
    uy: number[];
    mask: number[];
    stats: {
        mean_speed_mps: number;
        max_speed_mps: number;
        blocked_fraction: number;
    };
}

export interface FlowField2DResponse {
    buildings: BuildingData[];
    weather: {
        wind_speed: number;
        wind_deg: number;
        description: string;
    };
    inlet: {
        ux: number;
        uy: number;
        speed_mps: number;
    };
    domain: {
        geometry_radius_m: number;
        solve_radius_m: number;
    };
    field: FlowFieldGrid;
    source?: {
        kind: string;
        area?: string;
        snapshot_t?: number;
        is_latest?: boolean;
        stride?: number;
        raw_grid?: number[];
    };
}

export const fetchMapData = async (lat: number, lon: number, radius: number = 300): Promise<MapData> => {
    try {
        const response = await fetch(`${API_URL}/map?lat=${lat}&lon=${lon}&radius=${radius}`);
        if (!response.ok) {
            throw new Error('Failed to fetch map data');
        }
        return await response.json();
    } catch (error) {
        console.error("API Fetch Error:", error);
        return { features: [], count: 0 };
    }
};

export const fetchWeather = async (lat: number, lon: number) => {
    const response = await fetch(`${API_URL}/weather?lat=${lat}&lon=${lon}`);
    return await response.json();
};

export const fetchFlowField2D = async (
    lat: number,
    lon: number,
    geometry_radius_m: number = 400,
    solve_radius_m: number = 400,
    grid_size_m: number = 20
): Promise<FlowField2DResponse> => {
    const response = await fetch(`${API_URL}/flow-fields/2d`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            lat,
            lon,
            geometry_radius_m,
            solve_radius_m,
            grid_size_m,
            use_real_weather: true,
        }),
    });

    if (!response.ok) {
        throw new Error('Failed to fetch 2D flow field');
    }

    return await response.json();
};

export const fetchBackendHealth = async (): Promise<BackendHealth> => {
    const response = await fetch(`${API_URL}/health`);

    if (!response.ok) {
        throw new Error(`Backend health check failed: ${response.status}`);
    }

    return await response.json();
};

export const fetchRLEnvironmentSpec = async (): Promise<RLEnvironmentSpec> => {
    const response = await fetch(`${API_URL}/api/rl/spec`);

    if (!response.ok) {
        throw new Error(`RL environment spec failed: ${response.status}`);
    }

    return await response.json();
};

export const fetchRLBaseline = async (seed: number = 7, maxSteps: number = 300, nDrones: number = 4): Promise<RLBaselineResponse> => {
    const params = new URLSearchParams({ seed: String(seed), max_steps: String(maxSteps), n_drones: String(nDrones) });
    const response = await fetch(`${API_URL}/api/rl/baseline?${params.toString()}`);

    if (!response.ok) {
        throw new Error(`RL baseline rollout failed: ${response.status}`);
    }

    return await response.json();
};

export const fetchAeroJaxDemoFlow = async (stride: number = 8, snapshot_t?: number): Promise<FlowField2DResponse> => {
    const params = new URLSearchParams({ stride: String(stride) });
    if (snapshot_t !== undefined) {
        params.set('snapshot_t', String(snapshot_t));
    }
    const response = await fetch(`${API_URL}/flow-fields/aerojax-demo?${params.toString()}`);

    if (!response.ok) {
        throw new Error('Failed to fetch AeroJAX demo flow field');
    }

    return await response.json();
};
