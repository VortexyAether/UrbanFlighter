import type { UrbanFlowEvaluationSummary, UrbanFlowLiveScenarioSummary } from '../api';

export const LIVE_LOCATION_TOLERANCE_DEG = 1e-7;

export function liveSelectionIdentity(
  location: { lat: number; lon: number },
  scenarioId: string | null,
  loading: boolean,
) {
  return `${location.lat.toFixed(8)},${location.lon.toFixed(8)}:${scenarioId ?? 'none'}:${loading ? 'loading' : 'ready'}`;
}

export function scenarioMatchesLocation(
  scenario: UrbanFlowLiveScenarioSummary,
  selectedLocation: { lat: number; lon: number },
) {
  return Math.abs(scenario.location.selected_lat_deg - selectedLocation.lat) <= LIVE_LOCATION_TOLERANCE_DEG
    && Math.abs(scenario.location.selected_lon_deg - selectedLocation.lon) <= LIVE_LOCATION_TOLERANCE_DEG;
}

export function liveEvaluationMatchesSelection(
  result: UrbanFlowEvaluationSummary,
  expectedSelectionIdentity: string,
  currentSelectionIdentity: string,
  expectedScenarioId: string,
) {
  return currentSelectionIdentity === expectedSelectionIdentity
    && result.scenario_kind === 'live_osm_current_inlet'
    && result.scenario_id === expectedScenarioId
    && result.live_scenario?.scenario_id === expectedScenarioId;
}
