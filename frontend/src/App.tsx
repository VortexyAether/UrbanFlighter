import {
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react';
import Simulation3D from './components/Simulation3D';
import TopDownGame from './components/TopDownGame';
import { type ViewMode } from './appModel';
import CommandBar from './components/CommandBar';
import HudPanel from './components/HudPanel';
import LocalReturns2D from './components/LocalReturns2D';
import LocalReturnsRadar from './components/LocalReturnsRadar';
import RegionPanel from './components/RegionPanel';
import { useUrbanFlighterData } from './hooks/useUrbanFlighterData';
import './App.css';

type CockpitWindow = 'map' | 'telemetry' | 'slam';
type WindowPosition = { x: number; y: number };
type WindowSize = { width: number; height: number };

const VIEWPORT_MARGIN = 8;
const DOCK_CLEARANCE = 56;
const SLAM_MIN_SIZE: WindowSize = { width: 264, height: 248 };
const SLAM_INITIAL_SIZE: WindowSize = { width: 300, height: 286 };

function getViewportSize() {
  return {
    width: document.documentElement.clientWidth,
    height: document.documentElement.clientHeight,
  };
}

function getInitialWindowPosition(windowName: CockpitWindow): WindowPosition {
  const viewport = getViewportSize();

  if (windowName === 'map') return { x: 12, y: 12 };
  if (windowName === 'telemetry') {
    return {
      x: Math.max(12, viewport.width - 350),
      y: viewport.width < 700
        ? Math.max(12, viewport.height - viewport.height * 0.34 - DOCK_CLEARANCE - 8)
        : 12,
    };
  }

  return {
    x: Math.max(12, (viewport.width - 300) / 2),
    y: Math.max(12, viewport.height - 360),
  };
}

function clampWindowPosition(position: WindowPosition, element: HTMLDivElement): WindowPosition {
  const viewport = getViewportSize();
  const maxX = Math.max(VIEWPORT_MARGIN, viewport.width - element.offsetWidth - VIEWPORT_MARGIN);
  const maxY = Math.max(VIEWPORT_MARGIN, viewport.height - element.offsetHeight - DOCK_CLEARANCE);

  return {
    x: Math.min(Math.max(position.x, VIEWPORT_MARGIN), maxX),
    y: Math.min(Math.max(position.y, VIEWPORT_MARGIN), maxY),
  };
}

function clampResizableWindowSize(size: WindowSize, position: WindowPosition): WindowSize {
  const viewport = getViewportSize();
  const availableWidth = Math.max(1, viewport.width - position.x - VIEWPORT_MARGIN);
  const availableHeight = Math.max(1, viewport.height - position.y - DOCK_CLEARANCE);

  return {
    width: Math.min(Math.max(size.width, Math.min(SLAM_MIN_SIZE.width, availableWidth)), availableWidth),
    height: Math.min(Math.max(size.height, Math.min(SLAM_MIN_SIZE.height, availableHeight)), availableHeight),
  };
}

interface DraggableWindowProps {
  windowName: CockpitWindow;
  visible: boolean;
  active: boolean;
  onActivate: () => void;
  children: ReactNode;
  resizable?: boolean;
}

function DraggableWindow({ windowName, visible, active, onActivate, children, resizable = false }: DraggableWindowProps) {
  const windowRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    origin: WindowPosition;
  } | null>(null);
  const resizeRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    origin: WindowSize;
  } | null>(null);
  const [position, setPosition] = useState(() => getInitialWindowPosition(windowName));
  const [size, setSize] = useState<WindowSize | null>(() => (
    resizable ? clampResizableWindowSize(SLAM_INITIAL_SIZE, getInitialWindowPosition(windowName)) : null
  ));
  const positionRef = useRef(position);
  const [dragging, setDragging] = useState(false);
  const [resizing, setResizing] = useState(false);

  useEffect(() => {
    positionRef.current = position;
  }, [position]);

  useEffect(() => {
    const element = windowRef.current;
    if (!visible || !element) return;

    const clampCurrentPosition = () => {
      const clampedPosition = clampWindowPosition(positionRef.current, element);
      positionRef.current = clampedPosition;
      setPosition(clampedPosition);
      if (resizable) {
        setSize((currentSize) => currentSize && clampResizableWindowSize(currentSize, clampedPosition));
      }
    };
    const frame = window.requestAnimationFrame(clampCurrentPosition);
    const resizeObserver = new ResizeObserver(clampCurrentPosition);
    resizeObserver.observe(element);
    window.addEventListener('resize', clampCurrentPosition);

    return () => {
      window.cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      window.removeEventListener('resize', clampCurrentPosition);
    };
  }, [resizable, visible]);

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;

    const target = event.target as HTMLElement;
    if (resizable && target.closest('.cockpit-window__resize-handle')) {
      if (!size) return;
      event.preventDefault();
      event.stopPropagation();
      onActivate();
      resizeRef.current = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        origin: size,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
      setResizing(true);
      return;
    }
    const titleBar = target.closest('.panel-window-title');
    if (!titleBar || !event.currentTarget.contains(titleBar) || target.closest('button')) return;

    event.preventDefault();
    onActivate();
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      origin: position,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const resize = resizeRef.current;
    if (resize && resize.pointerId === event.pointerId) {
      setSize(clampResizableWindowSize({
        width: resize.origin.width + event.clientX - resize.startX,
        height: resize.origin.height + event.clientY - resize.startY,
      }, positionRef.current));
      return;
    }

    const drag = dragRef.current;
    const element = windowRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !element) return;

    const nextPosition = clampWindowPosition({
      x: drag.origin.x + event.clientX - drag.startX,
      y: drag.origin.y + event.clientY - drag.startY,
    }, element);
    positionRef.current = nextPosition;
    setPosition(nextPosition);
  };

  const finishDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    const finishedDragging = dragRef.current?.pointerId === event.pointerId;
    const finishedResizing = resizeRef.current?.pointerId === event.pointerId;
    if (!finishedDragging && !finishedResizing) return;
    if (finishedDragging) {
      dragRef.current = null;
      setDragging(false);
    }
    if (finishedResizing) {
      resizeRef.current = null;
      setResizing(false);
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  return (
    <div
      ref={windowRef}
      className={`cockpit-window${dragging ? ' cockpit-window--dragging' : ''}${resizing ? ' cockpit-window--resizing' : ''}`}
      style={{
        display: visible ? undefined : 'none',
        transform: `translate3d(${position.x}px, ${position.y}px, 0)`,
        zIndex: active ? 7 : 6,
        width: size?.width,
        height: size?.height,
      }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={finishDrag}
      onPointerCancel={finishDrag}
      onLostPointerCapture={finishDrag}
    >
      {visible ? children : null}
      {visible && resizable && (
        <button
          type="button"
          className="cockpit-window__resize-handle"
          aria-label="Resize rolling sensor-map window"
          title="Drag to resize window"
        />
      )}
    </div>
  );
}

function App() {
  const [windows, setWindows] = useState<Record<CockpitWindow, boolean>>({ map: true, telemetry: true, slam: true });
  const [activeWindow, setActiveWindow] = useState<CockpitWindow>('slam');
  const { state, actions } = useUrbanFlighterData();
  const {
    location,
    flow,
    loading,
    status,
    backendState,
    backendDetail,
    showFlowAnimation,
    showLidar,
    followCamera,
    simulationMode,
    energyGraphScale,
    energyHistory,
    telemetry,
  } = state;
  const {
    setShowFlowAnimation,
    setShowLidar,
    setFollowCamera,
    setEnergyGraphScale,
    setTelemetry,
    handleLocationSelect,
    handlePreset,
    handleSimulationModeSelect,
    handleReload,
  } = actions;

  const viewMode: ViewMode = simulationMode === '2d' ? '2d' : '3d';
  const isTrue3DMode = simulationMode === 'true3d';
  const buildingCount = flow?.buildings.length ?? 0;
  const windSpeed = flow?.weather.wind_speed ?? telemetry.localWindSpeed;
  const modelStatus = loading ? 'SYNCING' : flow ? 'LIVE' : 'STANDBY';
  const backendLabel = backendState === 'connected' ? 'BACKEND OK' : backendState === 'checking' ? 'BACKEND CHECK' : 'BACKEND OFF';
  const solverLabel = isTrue3DMode ? 'TRUE 3D U/V/W' : 'CFD-LITE B GRID';
  const modeLabel = isTrue3DMode ? 'TRUE 3D WIND' : simulationMode === '3d' ? '3D LITE' : '2D';
  const setWindowVisible = (windowName: CockpitWindow, visible: boolean) => {
    setWindows((current) => ({ ...current, [windowName]: visible }));
    if (visible) setActiveWindow(windowName);
  };

  return (
    <div className="app-shell">
      {viewMode === '2d' ? (
        <TopDownGame
          flow={flow}
          showFlowAnimation={showFlowAnimation}
          flowVisualization="arrows"
          showLidar={showLidar}
          onTelemetry={setTelemetry}
        />
      ) : (
        <Simulation3D
          flow={flow}
          showFlowAnimation={showFlowAnimation}
          onTelemetry={setTelemetry}
          followCamera={followCamera}
          showLidar={showLidar}
          true3DWind={isTrue3DMode}
          onLidarTelemetry={(lidar) => setTelemetry((previous) => ({ ...previous, lidar: lidar ?? undefined }))}
        />
      )}
      <CommandBar
        modelStatus={modelStatus}
        backendState={backendState}
        backendLabel={backendLabel}
        modeLabel={modeLabel}
        solverLabel={solverLabel}
        buildingCount={buildingCount}
        windSpeed={windSpeed}
        energyRate={telemetry.energyRate}
      />

      <DraggableWindow windowName="map" visible={windows.map} active={activeWindow === 'map'} onActivate={() => setActiveWindow('map')}>
        <RegionPanel
          location={location}
          simulationMode={simulationMode}
          viewMode={viewMode}
          backendLabel={backendLabel}
          backendDetail={backendDetail}
          loading={loading}
          status={status}
          isTrue3DMode={isTrue3DMode}
          onClose={() => setWindowVisible('map', false)}
          onLocationSelect={handleLocationSelect}
          onPreset={handlePreset}
          onModeSelect={handleSimulationModeSelect}
          onReload={handleReload}
        />
      </DraggableWindow>
      <DraggableWindow windowName="telemetry" visible={windows.telemetry} active={activeWindow === 'telemetry'} onActivate={() => setActiveWindow('telemetry')}>
        <HudPanel
          viewMode={viewMode}
          flow={flow}
          telemetry={telemetry}
          energyHistory={energyHistory}
          showFlowAnimation={showFlowAnimation}
          showLidar={showLidar}
          followCamera={followCamera}
          energyGraphScale={energyGraphScale}
          onClose={() => setWindowVisible('telemetry', false)}
          onShowFlowAnimationChange={setShowFlowAnimation}
          onShowLidarChange={setShowLidar}
          onFollowCameraChange={setFollowCamera}
          onEnergyGraphScaleChange={setEnergyGraphScale}
        />
      </DraggableWindow>
      <DraggableWindow windowName="slam" visible={windows.slam} active={activeWindow === 'slam'} onActivate={() => setActiveWindow('slam')} resizable>
        <div className="slam-window">
          <div className="panel-window-title panel-window-title--dark panel-window-title--draggable" title="Drag to move window">
            <span>Rolling Sensor Map <small>⠿ Move</small></span>
            <button type="button" onPointerDown={(event) => event.stopPropagation()} onClick={() => setWindowVisible('slam', false)} aria-label="Hide rolling sensor-map window">×</button>
          </div>
          {viewMode === '3d' ? (
            <LocalReturnsRadar lidar={telemetry.lidar} currentPose={telemetry.displayPose} enabled={showLidar} compact />
          ) : (
            <LocalReturns2D lidar={telemetry.lidar} currentPose={telemetry.displayPose} enabled={showLidar} headingDeg={telemetry.headingDeg} />
          )}
        </div>
      </DraggableWindow>

      <nav className="window-dock" aria-label="Cockpit windows">
        {(['map', 'telemetry', 'slam'] as CockpitWindow[]).map((windowName) => (
          <button
            key={windowName}
            type="button"
            className={windows[windowName] ? 'active' : ''}
            aria-pressed={windows[windowName]}
            onClick={() => setWindowVisible(windowName, !windows[windowName])}
          >
            <span aria-hidden="true">{windows[windowName] ? '−' : '+'}</span>
            {windowName === 'map' ? 'Map / Mode' : windowName === 'telemetry' ? 'Telemetry / Controls' : 'Sensor Map'}
          </button>
        ))}
      </nav>
    </div>
  );
}

export default App;
