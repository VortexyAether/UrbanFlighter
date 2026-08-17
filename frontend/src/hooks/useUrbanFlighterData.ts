import { useCallback, useEffect, useRef, useState } from 'react';
import {
  API_URL,
  activateUrbanFlowLiveScenario,
  fetchBackendHealth,
  fetchFlowField2D,
  type FlowField2DResponse,
} from '../api';
import {
  DEFAULT_GRID,
  DEFAULT_LAT,
  DEFAULT_LON,
  GEOMETRY_RADIUS_M,
  SOLVE_RADIUS_M,
  SWARM_LAT,
  SWARM_LON,
  errorMessage,
  type BackendState,
  type SimulationMode,
} from '../appModel';
import type { EnergyGraphScale } from '../components/EnergyGraph';
import type { Telemetry } from '../components/TopDownGame';
import {
  FLOW_CACHE_MAX_ENTRIES,
  FLOW_CACHE_TTL_MS,
  FLOW_REQUEST_TIMEOUT_MS,
  LatestFlowRequestCoordinator,
  TimedLruCache,
  createFlowCacheKey,
} from '../data/flowLoadCoordinator';

const locationLockedModes = new Set<SimulationMode>(['true3d']);

export function useUrbanFlighterData() {
  const [location, setLocation] = useState<{ lat: number; lon: number }>({ lat: DEFAULT_LAT, lon: DEFAULT_LON });
  const [flow, setFlow] = useState<FlowField2DResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('Loading default Inha/Incheon OSM buildings + current wind...');
  const [backendState, setBackendState] = useState<BackendState>('checking');
  const [backendDetail, setBackendDetail] = useState(`Checking ${API_URL}/health`);
  const [showFlowAnimation, setShowFlowAnimation] = useState(true);
  const [showLidar, setShowLidar] = useState(true);
  const [followCamera, setFollowCamera] = useState(true);
  const [simulationMode, setSimulationMode] = useState<SimulationMode>('2d');
  const [energyGraphScale, setEnergyGraphScale] = useState<EnergyGraphScale>('focus');
  const [energyHistory, setEnergyHistory] = useState<number[]>(() => Array.from({ length: 90 }, () => 0));
  const [telemetry, setTelemetry] = useState<Telemetry>({
    droneSpeed: 0,
    localWindSpeed: 0,
    localWindDirDeg: 0,
    energyRate: 0,
    energyUsed: 0,
    headingDeg: 0,
    position: { x: 0, y: 0 },
    displayPose: { x: 0, y: 0, z: 0, yaw: 0, pitch: 0, roll: 0 },
  });
  const historyTickRef = useRef(0);
  const [flowCache] = useState(() => new TimedLruCache<string, FlowField2DResponse>({
    maxEntries: FLOW_CACHE_MAX_ENTRIES,
    ttlMs: FLOW_CACHE_TTL_MS,
  }));
  const [flowRequests] = useState(() => new LatestFlowRequestCoordinator());

  const checkBackend = useCallback(async () => {
    setBackendState('checking');
    try {
      const health = await fetchBackendHealth();
      setBackendState('connected');
      setBackendDetail(`${health.service ?? 'Backend'} ${health.status}`);
    } catch (e) {
      setBackendState('disconnected');
      setBackendDetail(`Backend unavailable at ${API_URL}: ${errorMessage(e)}`);
    }
  }, []);

  const loadFlow = useCallback(async (lat: number, lon: number, options: { forceRefresh?: boolean } = {}) => {
    const timeoutSeconds = FLOW_REQUEST_TIMEOUT_MS / 1_000;
    const request = flowRequests.start(FLOW_REQUEST_TIMEOUT_MS, () => {
      setBackendState('disconnected');
      setBackendDetail(`Flow endpoint at ${API_URL} did not respond within ${timeoutSeconds} seconds.`);
      setStatus(`Flow request timed out after ${timeoutSeconds} seconds. Previous geometry remains visible when available.`);
      setLoading(false);
    });
    const cacheKey = createFlowCacheKey(lat, lon, GEOMETRY_RADIUS_M, SOLVE_RADIUS_M, DEFAULT_GRID);
    const cachedFlow = options.forceRefresh ? undefined : flowCache.get(cacheKey);
    if (cachedFlow) {
      const cachedScenarioId = cachedFlow.live_scenario?.scenario_id;
      if (cachedScenarioId) {
        try {
          await activateUrbanFlowLiveScenario(cachedScenarioId, { signal: request.signal });
          if (!flowRequests.complete(request)) return;
          setLocation({ lat, lon });
          setFlow(cachedFlow);
          setEnergyHistory(emptyHistory);
          setLoading(false);
          setStatus(flowLoadedStatus(cachedFlow, true));
          return;
        } catch (error) {
          if (!flowRequests.canApply(request)) return;
          flowCache.delete(cacheKey);
          setStatus(`Cached world is no longer registered; reloading OSM geometry + current inlet. ${errorMessage(error)}`);
        }
      } else {
        flowCache.delete(cacheKey);
      }
    }

    setLoading(true);
    setStatus(`${options.forceRefresh ? 'Refreshing' : 'Loading'} OSM geometry + current wind from backend ${API_URL}...`);
    try {
      const nextFlow = await fetchFlowField2D(
        lat,
        lon,
        GEOMETRY_RADIUS_M,
        SOLVE_RADIUS_M,
        DEFAULT_GRID,
        { signal: request.signal },
      );
      if (!flowRequests.canApply(request)) return;
      if (nextFlow.buildings.length === 0) {
        setFlow((prevFlow) => prevFlow ?? nextFlow);
        setStatus('No buildings found for this location. Previous geometry remains visible when available.');
        return;
      }
      if (!nextFlow.live_scenario?.scenario_id) {
        throw new Error('Backend returned geometry without a registered live UrbanFlow scenario identity.');
      }

      flowCache.set(cacheKey, nextFlow);
      setFlow(nextFlow);
      setEnergyHistory(emptyHistory);
      setStatus(flowLoadedStatus(nextFlow, false));
      setBackendState('connected');
    } catch (e) {
      if (!flowRequests.isCurrent(request) || flowRequests.didTimeOut(request)) return;
      setBackendState('disconnected');
      setBackendDetail(`Backend/API error: ${errorMessage(e)}`);
      setStatus(`Failed to load geometry/wind from ${API_URL}. Previous geometry remains visible when available. ${errorMessage(e)}`);
      console.error(e);
    } finally {
      if (flowRequests.complete(request)) setLoading(false);
    }
  }, [flowCache, flowRequests]);

  useEffect(() => {
    void checkBackend();
    void loadFlow(DEFAULT_LAT, DEFAULT_LON);
    return () => flowRequests.cancel();
  }, [checkBackend, flowRequests, loadFlow]);

  useEffect(() => {
    const now = Date.now();
    if (now - historyTickRef.current < 150) return;
    setEnergyHistory((prev) => [...prev.slice(1), telemetry.energyRate]);
    historyTickRef.current = now;
  }, [telemetry.energyRate]);

  const loadModeData = useCallback((nextMode: SimulationMode, lat: number, lon: number, forceRefresh = false) => {
    if (nextMode === 'true3d') {
      void loadFlow(SWARM_LAT, SWARM_LON, { forceRefresh });
    } else {
      void loadFlow(lat, lon, { forceRefresh });
    }
  }, [loadFlow]);

  const handleLocationSelect = (newLat: number, newLon: number) => {
    if (locationLockedModes.has(simulationMode)) {
      setLocation({ lat: SWARM_LAT, lon: SWARM_LON });
      setStatus('True 3D Wind uses the fixed Gangnam demo domain; location unchanged.');
      return;
    }
    setLocation({ lat: newLat, lon: newLon });
    void loadFlow(newLat, newLon);
  };

  const handlePreset = (lat: number, lon: number) => {
    if (locationLockedModes.has(simulationMode)) {
      setLocation({ lat: SWARM_LAT, lon: SWARM_LON });
      setStatus('True 3D Wind uses the fixed Gangnam demo domain; location unchanged.');
      return;
    }
    setLocation({ lat, lon });
    void loadFlow(lat, lon);
  };

  const handleSimulationModeSelect = (nextMode: SimulationMode) => {
    setSimulationMode(nextMode);
    setShowFlowAnimation(true);
    setFollowCamera(true);
    if (nextMode === 'true3d') {
      setLocation({ lat: SWARM_LAT, lon: SWARM_LON });
      setStatus('Loading Gangnam true 3D u/v/w potential-flow streamlines...');
    }
    loadModeData(nextMode, location.lat, location.lon);
  };

  const handleReload = () => {
    void checkBackend();
    loadModeData(simulationMode, location.lat, location.lon, true);
  };

  return {
    state: { location, flow, loading, status, backendState, backendDetail, showFlowAnimation, showLidar, followCamera, simulationMode, energyGraphScale, energyHistory, telemetry },
    actions: { setShowFlowAnimation, setShowLidar, setFollowCamera, setEnergyGraphScale, setTelemetry, handleLocationSelect, handlePreset, handleSimulationModeSelect, handleReload },
  };
}

function emptyHistory() {
  return Array.from({ length: 90 }, () => 0);
}

function flowLoadedStatus(flow: FlowField2DResponse, cached: boolean) {
  const cacheLabel = cached ? ' from recent cache' : ` in ${GEOMETRY_RADIUS_M}m`;
  const scenarioLabel = flow.live_scenario?.scenario_id
    ? ` Live world ${flow.live_scenario.scenario_id.slice(-10)}.`
    : '';
  return `Loaded ${flow.buildings.length} buildings${cacheLabel}. Inlet ${flow.weather.wind_speed.toFixed(1)} m/s from ${flow.weather.wind_deg.toFixed(0)}° (${weatherSourceLabel(flow)}).${scenarioLabel}`;
}

function weatherSourceLabel(flow: FlowField2DResponse) {
  if (flow.weather.fallback?.used || flow.weather.source?.kind === 'deterministic_fallback') {
    return 'labeled deterministic fallback';
  }
  if (flow.weather.source?.kind === 'forecast_model_current_conditions') {
    return 'Open-Meteo forecast-model current conditions';
  }
  if (flow.weather.source?.kind === 'configured_baseline') {
    return 'configured baseline';
  }
  return 'backend-reported wind';
}
