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

export interface UrbanFlowBaselineAggregate {
    episodes: number;
    success_count: number;
    success_rate: number;
    collision_count: number;
    collision_episode_rate: number;
    mean_path_length_m: number;
    mean_relative_air_speed_energy: number;
    mean_time_s: number;
    min_clearance_m: number;
    mean_score: number;
}

export interface UrbanFlowBaselineSummary {
    baseline_id: string;
    label: string;
    uses_hidden_flow: false;
    allowed_inputs: string[];
    aggregate: UrbanFlowBaselineAggregate;
}

export interface UrbanFlowEvaluationSummary {
    artifact_schema_id: 'urbanflow.baseline_evaluation.v1';
    contract_version: string;
    status: 'ok';
    environment_id: 'UrbanFlowGym-v1';
    scenario_kind: 'live_osm_current_inlet' | 'synthetic_fixture';
    scenario_id: string | null;
    scenario_identity: {
        scenario_id?: string;
        content_hash_sha256?: string;
        start_xy_m?: number[];
        goal_xy_m?: number[];
        fixture_schema?: string;
        seeds?: number[];
    };
    live_scenario: UrbanFlowLiveScenarioSummary | null;
    evaluation_id: string;
    artifact_path: string | null;
    policy_status: 'not_trained';
    real_cfd_validation_status: string;
    real_cfd_validation_run: false;
    synthetic_hidden_flow: boolean;
    policy_had_privileged_flow_access: false;
    policy_full_flow_access: false;
    evaluation_config: {
        seeds: number[];
        split: string;
        max_steps: number;
        dt_s: number;
        baselines: string[];
    };
    dynamics_source: {
        kind: string;
        purpose: string;
        navier_stokes_cfd: false;
        offline_3d_dataset_run: boolean;
        real_cfd_validation_status: string;
        claim: string;
    };
    baselines: Record<string, UrbanFlowBaselineSummary>;
    metrics: Record<string, UrbanFlowBaselineAggregate>;
}

export interface UrbanFlowLiveScenarioSummary {
    schema_id: 'urbanflow.live_scenario.v1';
    schema_version: 1;
    scenario_id: string;
    content_hash_sha256: string;
    is_current?: boolean;
    registry_size?: number;
    location: {
        selected_lat_deg: number;
        selected_lon_deg: number;
        geometry_radius_m: number;
        solve_radius_m: number;
    };
    coordinate_frame: {
        name: string;
        origin: { lat_deg: number; lon_deg: number };
        horizontal_units: 'm';
        vertical_units: 'm';
        x_axis: 'east';
        y_axis: 'north';
        z_axis: 'up';
        handedness: 'right_handed';
        projected_crs: string;
        browser_3d_mapping: string;
    };
    bounds: {
        min_x_m: number;
        max_x_m: number;
        min_y_m: number;
        max_y_m: number;
    };
    structure_count: number;
    inlet: {
        velocity_xy_mps: [number, number];
        speed_mps: number;
        direction_from_north_deg: number;
        timestamp: string | null;
        timestamp_status: string;
        source: Record<string, unknown>;
        fallback: Record<string, unknown>;
    };
    provenance: {
        geometry: Record<string, unknown>;
        weather: Record<string, unknown>;
        flow_response: Record<string, unknown>;
        registration_source: string;
    };
    hidden_flow: {
        kind: string;
        model: string;
        grid_digest_sha256: string;
        grid_shape: number[];
        navier_stokes_cfd: false;
        synthetic_hidden_flow: true;
        real_cfd_validation: false;
        full_grid_hidden_from_actor: true;
    };
    collision_lidar_semantics: {
        dimension: string;
        building_model: string;
        collision_model: string;
        agent_radius_m: number;
        lidar_intersection_model: string;
        gym_actor_lidar: {
            ray_count: number;
            max_range_m: number;
            ordering: string;
        };
        browser_display_lidar: {
            ray_count: number;
            max_range_m: number;
            intersection_model_shared_with_gym: true;
        };
    };
    policy_boundaries: {
        full_flow_access: false;
        synthetic_hidden_flow: true;
        trained_policy_available: false;
        real_cfd_validation_run: false;
    };
}

export type UrbanFlowInspectorBaseline = 'direct_goal' | 'shortest_path' | 'wind_aware_inlet';

export interface UrbanFlowInspectorWorld {
    schema_id: 'urbanflow.episode_inspector_world.v1';
    scenario_id: string;
    content_hash_sha256: string;
    coordinate_frame: {
        horizontal_units: 'm';
        x_axis: 'east';
        y_axis: 'north';
        display_orientation: 'north_up';
    };
    bounds: {
        min_x_m: number;
        max_x_m: number;
        min_y_m: number;
        max_y_m: number;
    };
    start_goal_source: string;
    structure_count: number;
    buildings: Array<{
        building_id: string;
        height_m: number;
        height_source: string;
        footprint_xy_m: number[][];
    }>;
    known_inlet: {
        velocity_xy_mps: [number, number];
        speed_mps: number;
        direction_from_north_deg: number;
        timestamp: string | null;
        source: Record<string, unknown>;
        fallback: Record<string, unknown>;
    };
    source: 'exact_registered_live_osm_scenario';
    synthetic_fixture: false;
}

export interface UrbanFlowInspectorFrame {
    schema_id: 'urbanflow.episode_inspector_frame.v1';
    scenario_id: string;
    seed: number;
    baseline: {
        baseline_id: UrbanFlowInspectorBaseline;
        label: string;
        uses_full_flow: false;
    };
    world_bounds: UrbanFlowInspectorWorld['bounds'];
    drone: {
        position_xy_m: [number, number];
        heading_rad: number;
        ground_velocity_xy_mps: [number, number];
    };
    start_xy_m: [number, number];
    goal_xy_m: [number, number];
    trajectory_xy_m: Array<[number, number]>;
    actor_lidar: {
        ray_count: number;
        max_range_m: number;
        frame: string;
        rays: Array<{
            local_angle_rad: number;
            distance_m: number;
            endpoint_xy_m: [number, number];
            hit: boolean;
        }>;
    };
    local_guidance_action: {
        schema_id: string;
        frame: 'vehicle_local_forward_left';
        vector: [number, number];
        forward: number;
        left: number;
        phase: 'preview_next' | 'executed';
        source: 'deterministic_baseline' | 'validated_actor_override';
    };
    actor_observation: {
        schema_id: string;
        vector: number[];
        fields: Array<{
            name: string;
            values: number[];
            units: string;
            source: string;
        }>;
    };
    air_relative_velocity_xy_mps: [number, number];
    reward: {
        schema_id: string;
        components: Record<string, number>;
        step_total: number;
        episode_total: number;
    };
    clearance_m: number;
    collision: boolean;
    terminated: boolean;
    truncated: boolean;
    status: 'ready' | 'running' | 'success' | 'collision' | 'time_limit';
    termination_reason: string | null;
    step_index: number;
    max_steps: number;
    dt_s: number;
    simulated_elapsed_s: number;
    simulated_max_s: number;
    distance_to_goal_m: number;
    estimated_minimum_steps: number;
    estimated_minimum_time_s: number;
    flags: {
        policy_status: 'not_trained';
        policy_had_privileged_flow_access: false;
        full_flow_access: false;
        training_executed: false;
        browser_motor_training: false;
        navier_stokes_cfd: false;
        real_cfd_validation_run: false;
        real_cfd_adapter_status: 'interface_only_not_executed';
        synthetic_fixture: false;
    };
}

export interface UrbanFlowInspectorSessionResponse {
    session_id: string;
    session_active: boolean;
    scenario_id: string;
    seed: number;
    baseline_id: UrbanFlowInspectorBaseline;
    limits: {
        max_steps: number;
        max_sessions: number;
        ttl_s: number;
        reset_count: number;
        dt_s: number;
        simulated_max_s: number;
        max_batch_steps: number;
    };
    world?: UrbanFlowInspectorWorld;
    frame: UrbanFlowInspectorFrame;
    requested_steps: number;
    executed_steps: number;
    batch_reward: number;
    cleanup?: 'terminal_session_deleted';
}

export interface BuildingData {
    building_id?: string;
    height: number;
    height_source?: string;
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
        units?: {
            wind_speed: 'm/s';
            wind_deg: 'degrees_from_north';
        };
        source?: {
            provider: 'open-meteo' | 'urban-flighter';
            kind: 'forecast_model_current_conditions' | 'deterministic_fallback' | 'configured_baseline';
            endpoint?: string;
            variable_height_m?: number;
            observation_time?: string;
            upstream_provider?: string;
        };
        fallback?: {
            used: boolean;
            reason: string | null;
        };
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
    live_scenario?: UrbanFlowLiveScenarioSummary | null;
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
    grid_size_m: number = 20,
    options: { signal?: AbortSignal } = {},
): Promise<FlowField2DResponse> => {
    const response = await fetch(`${API_URL}/flow-fields/2d`, {
        method: 'POST',
        signal: options.signal,
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
        const status = response.statusText ? `${response.status} ${response.statusText}` : String(response.status);
        throw new Error(`2D flow request failed (${status})`);
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

export const evaluateUrbanFlowBaselines = async (
    seeds: number[] = [10007, 10009, 10037],
    maxSteps: number = 360,
    options: { signal?: AbortSignal } = {},
): Promise<UrbanFlowEvaluationSummary> => {
    const response = await fetch(`${API_URL}/urbanflow-gym/evaluate`, {
        method: 'POST',
        signal: options.signal,
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            seeds,
            max_steps: maxSteps,
            save_artifact: false,
        }),
    });

    if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
            const errorPayload = await response.json() as { detail?: unknown };
            if (typeof errorPayload.detail === 'string') detail = errorPayload.detail;
        } catch {
            // Keep the bounded status fallback when the backend did not return JSON.
        }
        throw new Error(`UrbanFlow baseline evaluation failed: ${detail}`);
    }

    return await response.json();
};

export const fetchUrbanFlowLiveScenario = async (
    scenarioId?: string,
    options: { signal?: AbortSignal } = {},
): Promise<UrbanFlowLiveScenarioSummary> => {
    const path = scenarioId
        ? `/urbanflow-gym/live-scenarios/${encodeURIComponent(scenarioId)}/summary`
        : '/urbanflow-gym/live-scenarios/current';
    const response = await fetch(`${API_URL}${path}`, { signal: options.signal });
    if (!response.ok) {
        throw new Error(`Live UrbanFlow scenario lookup failed: ${await responseErrorDetail(response)}`);
    }
    return await response.json();
};

export const activateUrbanFlowLiveScenario = async (
    scenarioId: string,
    options: { signal?: AbortSignal } = {},
): Promise<UrbanFlowLiveScenarioSummary> => {
    const response = await fetch(
        `${API_URL}/urbanflow-gym/live-scenarios/${encodeURIComponent(scenarioId)}/activate`,
        { method: 'POST', signal: options.signal },
    );
    if (!response.ok) {
        throw new Error(`Live UrbanFlow scenario activation failed: ${await responseErrorDetail(response)}`);
    }
    return await response.json();
};

export const evaluateUrbanFlowLiveBaselines = async (
    scenarioId: string,
    seeds: number[] = [10007, 10009, 10037],
    maxSteps: number = 360,
    options: { signal?: AbortSignal; startXY?: [number, number]; goalXY?: [number, number] } = {},
): Promise<UrbanFlowEvaluationSummary> => {
    const response = await fetch(`${API_URL}/urbanflow-gym/live/evaluate`, {
        method: 'POST',
        signal: options.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            scenario_id: scenarioId,
            seeds,
            max_steps: maxSteps,
            save_artifact: false,
            ...(options.startXY && options.goalXY
                ? { start_xy: options.startXY, goal_xy: options.goalXY }
                : {}),
        }),
    });
    if (!response.ok) {
        throw new Error(`Live UrbanFlow baseline evaluation failed: ${await responseErrorDetail(response)}`);
    }
    return await response.json();
};

export const createUrbanFlowInspectorSession = async (
    scenarioId: string,
    seed: number,
    baseline: UrbanFlowInspectorBaseline,
    maxSteps: number = 1200,
    options: { signal?: AbortSignal } = {},
): Promise<UrbanFlowInspectorSessionResponse> => {
    const response = await fetch(`${API_URL}/urbanflow-gym/inspector/sessions`, {
        method: 'POST',
        signal: options.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            scenario_id: scenarioId,
            seed,
            baseline,
            max_steps: maxSteps,
        }),
    });
    if (!response.ok) {
        throw new Error(`Episode inspector session creation failed: ${await responseErrorDetail(response)}`);
    }
    return await response.json();
};

export const resetUrbanFlowInspectorSession = async (
    sessionId: string,
    options: { signal?: AbortSignal } = {},
): Promise<UrbanFlowInspectorSessionResponse> => {
    const response = await fetch(
        `${API_URL}/urbanflow-gym/inspector/sessions/${encodeURIComponent(sessionId)}/reset`,
        { method: 'POST', signal: options.signal },
    );
    if (!response.ok) {
        throw new Error(`Episode inspector reset failed: ${await responseErrorDetail(response)}`);
    }
    return await response.json();
};

export const stepUrbanFlowInspectorSession = async (
    sessionId: string,
    options: { signal?: AbortSignal; action?: [number, number]; repeat?: number } = {},
): Promise<UrbanFlowInspectorSessionResponse> => {
    const repeat = options.repeat ?? 1;
    if (!Number.isInteger(repeat) || repeat < 1 || repeat > 64) {
        throw new Error('Episode inspector repeat must be an integer from 1 through 64.');
    }
    const response = await fetch(
        `${API_URL}/urbanflow-gym/inspector/sessions/${encodeURIComponent(sessionId)}/step`,
        {
            method: 'POST',
            signal: options.signal,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                repeat,
                ...(options.action ? { action: options.action } : {}),
            }),
        },
    );
    if (!response.ok) {
        throw new Error(`Episode inspector step failed: ${await responseErrorDetail(response)}`);
    }
    return await response.json();
};

export const deleteUrbanFlowInspectorSession = async (
    sessionId: string,
    options: { signal?: AbortSignal } = {},
): Promise<{ status: 'deleted'; session_id: string }> => {
    const response = await fetch(
        `${API_URL}/urbanflow-gym/inspector/sessions/${encodeURIComponent(sessionId)}`,
        { method: 'DELETE', signal: options.signal },
    );
    if (!response.ok) {
        throw new Error(`Episode inspector delete failed: ${await responseErrorDetail(response)}`);
    }
    return await response.json();
};

async function responseErrorDetail(response: Response) {
    let detail = `HTTP ${response.status}`;
    try {
        const payload = await response.json() as { detail?: unknown };
        if (typeof payload.detail === 'string') detail = payload.detail;
    } catch {
        // Keep the bounded HTTP status fallback for non-JSON failures.
    }
    return detail;
}

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
