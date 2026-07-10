import type { RLBaselineResponse } from './api';

export const DEFAULT_LAT = 37.451448;
export const DEFAULT_LON = 126.6515423;
export const SWARM_LAT = 37.497942;
export const SWARM_LON = 127.027621;
export const SWARM_DATASET = 'multi_drone_gangnam_v3';
export const SWARM_REPLAY_URL = `/data/${SWARM_DATASET}/urban_flighter_multi_drone_trajectories.json`;
export const SWARM_METRICS_URL = `/data/${SWARM_DATASET}/urban_flighter_multi_drone_metrics.json`;
export const TRUE_3D_WIND_URL = '/data/true3d_potential_gangnam/true3d_streamlines.json';

export const GEOMETRY_RADIUS_M = 400;
export const SOLVE_RADIUS_M = 400;
export const DEFAULT_GRID = 2.5;

export type ViewMode = '2d' | '3d';
export type SimulationMode = ViewMode | 'true3d';
export type BackendState = 'checking' | 'connected' | 'disconnected';

export interface SwarmAggregateMetrics {
  controller: string;
  n_drones: number;
  success_count: number;
  all_success: boolean;
  total_collisions: number;
  total_boundary_violations?: number;
  swept_building_hits?: number;
  collision_validation?: string;
  total_energy_relative_airspeed_l2: number;
  total_path_length_m: number;
  max_steps_used: number;
  min_pairwise_separation_m: number;
  near_miss_count_sep_lt_10m: number;
  energy_model: string;
  note: string;
  lat: number;
  lon: number;
  radius_m: number;
  building_count?: number;
  source_elements?: number;
  satellite_tile_url?: string;
  world_source?: string;
}

export interface SwarmMetricsPayload {
  aggregate: SwarmAggregateMetrics;
}

export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

export function modeHelp(mode: SimulationMode) {
  if (mode === '2d') return '2D CFD-lite B: OSM buildings + wall damping/wake wind grid. Use WASD.';
  if (mode === '3d') return '3D Lite: the same backend wind grid in Three.js with live aircraft sampling.';
  return 'True 3D wind: u/v/w potential-flow streamlines over the Gangnam demo domain.';
}

export function rewardTerm(baseline: RLBaselineResponse | null, term: string) {
  return baseline?.metrics.reward_terms_total[term] ?? 0;
}
