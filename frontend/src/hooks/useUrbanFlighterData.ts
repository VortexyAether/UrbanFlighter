import { useCallback, useEffect, useRef, useState } from 'react';
import {
  API_URL,
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

const locationLockedModes = new Set<SimulationMode>(['true3d']);

export function useUrbanFlighterData() {
  const [location, setLocation] = useState<{ lat: number; lon: number }>({ lat: DEFAULT_LAT, lon: DEFAULT_LON });
  const [flow, setFlow] = useState<FlowField2DResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('Loading default Inha/Incheon OSM buildings + live Open-Meteo wind...');
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
  const requestIdRef = useRef(0);
  const flowCacheRef = useRef<Map<string, FlowField2DResponse>>(new Map());

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

  const loadFlow = useCallback(async (lat: number, lon: number) => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    const cacheKey = `${lat.toFixed(5)},${lon.toFixed(5)},${GEOMETRY_RADIUS_M},${SOLVE_RADIUS_M},${DEFAULT_GRID}`;
    const cachedFlow = flowCacheRef.current.get(cacheKey);
    if (cachedFlow) {
      setLocation({ lat, lon });
      setFlow(cachedFlow);
      setEnergyHistory(emptyHistory);
      setStatus(`Loaded ${cachedFlow.buildings.length} cached buildings. Inlet ${cachedFlow.weather.wind_speed.toFixed(1)} m/s from ${cachedFlow.weather.wind_deg.toFixed(0)}°.`);
      return;
    }

    setLoading(true);
    setStatus(`Loading OSM geometry + Open-Meteo wind from backend ${API_URL}...`);
    try {
      const nextFlow = await fetchFlowField2D(lat, lon, GEOMETRY_RADIUS_M, SOLVE_RADIUS_M, DEFAULT_GRID);
      if (requestId !== requestIdRef.current) return;
      if (nextFlow.buildings.length === 0) {
        setFlow((prevFlow) => prevFlow ?? nextFlow);
        setStatus('No buildings found. Keeping previous geometry visible. Data may be stale.');
        return;
      }

      flowCacheRef.current.set(cacheKey, nextFlow);
      setFlow(nextFlow);
      setEnergyHistory(emptyHistory);
      setStatus(`Loaded ${nextFlow.buildings.length} buildings in ${GEOMETRY_RADIUS_M}m. Inlet ${nextFlow.weather.wind_speed.toFixed(1)} m/s from ${nextFlow.weather.wind_deg.toFixed(0)}°.`);
      setBackendState('connected');
    } catch (e) {
      if (requestId === requestIdRef.current) {
        setBackendState('disconnected');
        setBackendDetail(`Backend/API error: ${errorMessage(e)}`);
        setStatus(`Failed to load geometry/wind from ${API_URL}. Keeping previous geometry visible. ${errorMessage(e)}`);
      }
      console.error(e);
    } finally {
      if (requestId === requestIdRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void checkBackend();
    void loadFlow(DEFAULT_LAT, DEFAULT_LON);
  }, [checkBackend, loadFlow]);

  useEffect(() => {
    const now = Date.now();
    if (now - historyTickRef.current < 150) return;
    setEnergyHistory((prev) => [...prev.slice(1), telemetry.energyRate]);
    historyTickRef.current = now;
  }, [telemetry.energyRate]);

  const loadModeData = useCallback((nextMode: SimulationMode, lat: number, lon: number) => {
    if (nextMode === 'true3d') {
      void loadFlow(SWARM_LAT, SWARM_LON);
    } else {
      void loadFlow(lat, lon);
    }
  }, [loadFlow]);

  const handleLocationSelect = (newLat: number, newLon: number) => {
    setLocation({ lat: newLat, lon: newLon });
    if (!locationLockedModes.has(simulationMode)) void loadFlow(newLat, newLon);
  };

  const handlePreset = (lat: number, lon: number) => {
    setLocation({ lat, lon });
    if (!locationLockedModes.has(simulationMode)) void loadFlow(lat, lon);
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
    loadModeData(simulationMode, location.lat, location.lon);
  };

  return {
    state: { location, flow, loading, status, backendState, backendDetail, showFlowAnimation, showLidar, followCamera, simulationMode, energyGraphScale, energyHistory, telemetry },
    actions: { setShowFlowAnimation, setShowLidar, setFollowCamera, setEnergyGraphScale, setTelemetry, handleLocationSelect, handlePreset, handleSimulationModeSelect, handleReload },
  };
}

function emptyHistory() {
  return Array.from({ length: 90 }, () => 0);
}
