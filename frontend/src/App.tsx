import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react';
import Simulation3D from './components/Simulation3D';
import TopDownGame from './components/TopDownGame';
import { type ViewMode } from './appModel';
import type { DroneControlPreset } from './simulation/flight3dControls';
import CommandBar from './components/CommandBar';
import HudPanel from './components/HudPanel';
import LocalReturns2D from './components/LocalReturns2D';
import LocalReturnsRadar from './components/LocalReturnsRadar';
import RegionPanel from './components/RegionPanel';
import UrbanFlowEpisodeInspector from './components/UrbanFlowEpisodeInspector';
import { useUrbanFlighterData } from './hooks/useUrbanFlighterData';
import './App.css';

type CockpitWindow = 'map' | 'telemetry' | 'slam' | 'inspector';
type ShotKind = '2d' | '3d' | 'map' | 'radar' | 'cockpit';

function readShotKind(): ShotKind | null {
  if (typeof window === 'undefined') return null;
  const value = new URLSearchParams(window.location.search).get('shot');
  return value === '2d' || value === '3d' || value === 'map' || value === 'radar' || value === 'cockpit'
    ? value
    : null;
}
type WindowPosition = { x: number; y: number };
type WindowSize = { width: number; height: number };

const VIEWPORT_MARGIN = 8;
const DOCK_CLEARANCE = 56;
const COMPACT_LAYOUT_BREAKPOINT = 700;
const SLAM_MIN_SIZE: WindowSize = { width: 264, height: 248 };
const SLAM_INITIAL_SIZE: WindowSize = { width: 300, height: 286 };
const INSPECTOR_MIN_SIZE: WindowSize = { width: 620, height: 500 };
const INSPECTOR_INITIAL_SIZE: WindowSize = { width: 980, height: 740 };
const RESIZE_STEP = 8;
const RESIZE_LARGE_STEP = 24;

interface CockpitLayout {
  positions: Record<CockpitWindow, WindowPosition>;
  slamSize: WindowSize;
  inspectorSize: WindowSize;
}

function getViewportSize() {
  return {
    width: document.documentElement.clientWidth,
    height: document.documentElement.clientHeight,
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

function clampResizableWindowSize(
  size: WindowSize,
  position: WindowPosition,
  minimumSize: WindowSize = SLAM_MIN_SIZE,
  viewport = getViewportSize(),
): WindowSize {
  const availableWidth = Math.max(1, viewport.width - position.x - VIEWPORT_MARGIN);
  const availableHeight = Math.max(1, viewport.height - position.y - DOCK_CLEARANCE);

  return {
    width: Math.min(Math.max(size.width, Math.min(minimumSize.width, availableWidth)), availableWidth),
    height: Math.min(Math.max(size.height, Math.min(minimumSize.height, availableHeight)), availableHeight),
  };
}

function resizableMinimum(windowName: CockpitWindow) {
  return windowName === 'inspector' ? INSPECTOR_MIN_SIZE : SLAM_MIN_SIZE;
}

function getDefaultCockpitLayout(viewport = getViewportSize()): CockpitLayout {
  const mapPosition: WindowPosition = { x: 12, y: 12 };
  const telemetryPosition: WindowPosition = {
    x: Math.max(12, viewport.width - 350),
    y: 12,
  };
  const slamPosition: WindowPosition = {
    x: Math.max(12, (viewport.width - SLAM_INITIAL_SIZE.width) / 2),
    y: Math.max(12, viewport.height - 360),
  };
  const inspectorPosition: WindowPosition = {
    x: Math.max(VIEWPORT_MARGIN, (viewport.width - INSPECTOR_INITIAL_SIZE.width) / 2),
    y: Math.max(VIEWPORT_MARGIN, (viewport.height - INSPECTOR_INITIAL_SIZE.height - DOCK_CLEARANCE) / 2),
  };

  if (viewport.width < COMPACT_LAYOUT_BREAKPOINT) {
    const panelWidth = Math.min(320, Math.max(1, viewport.width - 24));
    const usableHeight = Math.max(1, viewport.height - DOCK_CLEARANCE - VIEWPORT_MARGIN * 2);
    const verticalStep = usableHeight / 3;

    mapPosition.x = VIEWPORT_MARGIN;
    mapPosition.y = VIEWPORT_MARGIN;
    telemetryPosition.x = Math.max(VIEWPORT_MARGIN, viewport.width - panelWidth - 12);
    telemetryPosition.y = Math.round(VIEWPORT_MARGIN + verticalStep);
    slamPosition.x = Math.max(VIEWPORT_MARGIN, (viewport.width - SLAM_INITIAL_SIZE.width) / 2);
    slamPosition.y = Math.round(VIEWPORT_MARGIN + verticalStep * 2);
    inspectorPosition.x = VIEWPORT_MARGIN;
    inspectorPosition.y = VIEWPORT_MARGIN;
  }

  return {
    positions: {
      map: mapPosition,
      telemetry: telemetryPosition,
      slam: slamPosition,
      inspector: inspectorPosition,
    },
    slamSize: clampResizableWindowSize(SLAM_INITIAL_SIZE, slamPosition, SLAM_MIN_SIZE, viewport),
    inspectorSize: clampResizableWindowSize(
      INSPECTOR_INITIAL_SIZE,
      inspectorPosition,
      INSPECTOR_MIN_SIZE,
      viewport,
    ),
  };
}

interface DraggableWindowProps {
  windowName: CockpitWindow;
  visible: boolean;
  active: boolean;
  onActivate: () => void;
  resetToken: number;
  children: ReactNode;
  resizable?: boolean;
}

function DraggableWindow({
  windowName,
  visible,
  active,
  onActivate,
  resetToken,
  children,
  resizable = false,
}: DraggableWindowProps) {
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
  const [position, setPosition] = useState(() => getDefaultCockpitLayout().positions[windowName]);
  const [size, setSize] = useState<WindowSize | null>(() => {
    if (!resizable) return null;
    const layout = getDefaultCockpitLayout();
    return windowName === 'inspector' ? layout.inspectorSize : layout.slamSize;
  });
  const positionRef = useRef(position);
  const [appliedResetToken, setAppliedResetToken] = useState(resetToken);
  const [dragging, setDragging] = useState(false);
  const [resizing, setResizing] = useState(false);

  if (appliedResetToken !== resetToken) {
    const layout = getDefaultCockpitLayout();
    setAppliedResetToken(resetToken);
    setPosition(layout.positions[windowName]);
    if (resizable) setSize(windowName === 'inspector' ? layout.inspectorSize : layout.slamSize);
  }

  useLayoutEffect(() => {
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
        setSize((currentSize) => currentSize && clampResizableWindowSize(
          currentSize,
          clampedPosition,
          resizableMinimum(windowName),
        ));
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
  }, [resizable, visible, windowName]);

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
      }, positionRef.current, resizableMinimum(windowName)));
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

  const handleResizeKeyDown = (event: ReactKeyboardEvent<HTMLButtonElement>) => {
    if (!size) return;

    const step = event.shiftKey ? RESIZE_LARGE_STEP : RESIZE_STEP;
    let widthDelta = 0;
    let heightDelta = 0;

    switch (event.key) {
      case 'ArrowLeft':
        widthDelta = -step;
        break;
      case 'ArrowRight':
        widthDelta = step;
        break;
      case 'ArrowUp':
        heightDelta = -step;
        break;
      case 'ArrowDown':
        heightDelta = step;
        break;
      default:
        return;
    }

    event.preventDefault();
    event.stopPropagation();
    onActivate();
    setSize((currentSize) => currentSize && clampResizableWindowSize({
      width: currentSize.width + widthDelta,
      height: currentSize.height + heightDelta,
    }, positionRef.current, resizableMinimum(windowName)));
  };

  const resizeHelpId = `${windowName}-window-resize-help`;

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
      onPointerDownCapture={onActivate}
      onFocusCapture={onActivate}
    >
      {visible ? children : null}
      {visible && resizable && (
        <>
          <span id={resizeHelpId} className="visually-hidden">
            Drag the corner to resize. With the handle focused, use Left and Right Arrow to change width,
            or Up and Down Arrow to change height. Hold Shift for larger steps.
          </span>
          <button
            type="button"
            className="cockpit-window__resize-handle"
            aria-label={`Resize ${windowName === 'inspector' ? 'Gym Episode Inspector' : 'rolling sensor-map'} window`}
            aria-describedby={resizeHelpId}
            aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown Shift+ArrowLeft Shift+ArrowRight Shift+ArrowUp Shift+ArrowDown"
            title="Drag to resize. Arrow keys adjust by 8 px; hold Shift for 24 px."
            onKeyDown={handleResizeKeyDown}
          />
        </>
      )}
    </div>
  );
}

function App() {
  const shot = useMemo(readShotKind, []);
  const [windows, setWindows] = useState<Record<CockpitWindow, boolean>>(() => ({
    map: shot === 'map' || shot == null,
    telemetry: shot == null,
    slam: shot === 'radar' || shot == null,
    inspector: false,
  }));
  const [activeWindow, setActiveWindow] = useState<CockpitWindow>('slam');
  const [layoutResetToken, setLayoutResetToken] = useState(0);
  const [controlPreset, setControlPreset] = useState<DroneControlPreset>('arcade');
  const [presentationMode, setPresentationMode] = useState(true);
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

  useEffect(() => {
    if (!shot) return;
    if (shot === '3d' || shot === 'radar') {
      handleSimulationModeSelect('3d');
      setPresentationMode(true);
    }
    setShowLidar(shot === 'radar' || shot === 'cockpit');
    // Intentionally once per shot URL. handleSimulationModeSelect is not stable
    // and must not retrigger /flow-fields/2d on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shot]);

  const viewMode: ViewMode = simulationMode === '2d' ? '2d' : '3d';
  const isTrue3DMode = simulationMode === 'true3d';
  const buildingCount = flow?.buildings.length ?? 0;
  const windSpeed = flow?.weather.wind_speed ?? telemetry.localWindSpeed;
  const modelStatus = loading ? 'SYNCING' : flow ? 'READY' : 'STANDBY';
  const backendLabel = backendState === 'connected' ? 'BACKEND OK' : backendState === 'checking' ? 'BACKEND CHECK' : 'BACKEND OFF';
  const solverLabel = isTrue3DMode ? 'U/V/W VISUAL OVERLAY' : 'CFD-LITE B GRID';
  const modeLabel = isTrue3DMode ? 'TRUE 3D WIND' : simulationMode === '3d' ? '3D LITE' : '2D';
  const setWindowVisible = (windowName: CockpitWindow, visible: boolean) => {
    setWindows((current) => ({ ...current, [windowName]: visible }));
    if (visible) setActiveWindow(windowName);
  };

  const hideChrome = shot === '2d' || shot === '3d' || shot === 'radar';

  return (
    <div
      className={`app-shell${shot ? ` app-shell--shot app-shell--shot-${shot}` : ''}`}
      data-shot-ready={flow && !loading ? '1' : '0'}
      data-building-count={buildingCount}
    >
      {viewMode === '2d' ? (
        <TopDownGame
          flow={flow}
          showFlowAnimation={showFlowAnimation}
          flowVisualization="both"
          showLidar={showLidar}
          onTelemetry={setTelemetry}
        />
      ) : (
        <Simulation3D
          flow={flow}
          selectedLocation={location}
          showFlowAnimation={showFlowAnimation}
          onTelemetry={setTelemetry}
          followCamera={followCamera}
          onFollowCameraChange={setFollowCamera}
          showLidar={showLidar}
          true3DWind={isTrue3DMode}
          controlPreset={controlPreset}
          onControlPresetChange={setControlPreset}
          presentationMode={presentationMode}
          onPresentationModeChange={setPresentationMode}
          hideFlightHud={shot === '3d'}
          onLidarTelemetry={(lidar) => setTelemetry((previous) => ({ ...previous, lidar: lidar ?? undefined }))}
        />
      )}
      {!hideChrome && (
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
      )}

      <DraggableWindow windowName="map" visible={windows.map} active={activeWindow === 'map'} onActivate={() => setActiveWindow('map')} resetToken={layoutResetToken}>
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
      <DraggableWindow windowName="telemetry" visible={windows.telemetry} active={activeWindow === 'telemetry'} onActivate={() => setActiveWindow('telemetry')} resetToken={layoutResetToken}>
        <HudPanel
          viewMode={viewMode}
          flow={flow}
          selectedLocation={location}
          worldLoading={loading}
          telemetry={telemetry}
          energyHistory={energyHistory}
          showFlowAnimation={showFlowAnimation}
          showLidar={showLidar}
          followCamera={followCamera}
          controlPreset={controlPreset}
          presentationMode={presentationMode}
          energyGraphScale={energyGraphScale}
          onClose={() => setWindowVisible('telemetry', false)}
          onShowFlowAnimationChange={setShowFlowAnimation}
          onShowLidarChange={setShowLidar}
          onFollowCameraChange={setFollowCamera}
          onControlPresetChange={setControlPreset}
          onPresentationModeChange={setPresentationMode}
          onEnergyGraphScaleChange={setEnergyGraphScale}
        />
      </DraggableWindow>
      <DraggableWindow windowName="slam" visible={windows.slam} active={activeWindow === 'slam'} onActivate={() => setActiveWindow('slam')} resetToken={layoutResetToken} resizable>
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
      <DraggableWindow
        windowName="inspector"
        visible={windows.inspector}
        active={activeWindow === 'inspector'}
        onActivate={() => setActiveWindow('inspector')}
        resetToken={layoutResetToken}
        resizable
      >
        <UrbanFlowEpisodeInspector
          flow={flow}
          selectedLocation={location}
          worldLoading={loading}
          onClose={() => setWindowVisible('inspector', false)}
        />
      </DraggableWindow>

      {!shot && (
      <nav className="window-dock" aria-label="Cockpit windows">
        {(['map', 'telemetry', 'slam', 'inspector'] as CockpitWindow[]).map((windowName) => (
          <button
            key={windowName}
            type="button"
            className={windows[windowName] ? 'active' : ''}
            aria-pressed={windows[windowName]}
            onClick={() => setWindowVisible(windowName, !windows[windowName])}
          >
            <span aria-hidden="true">{windows[windowName] ? '−' : '+'}</span>
            {windowName === 'map'
              ? 'Map / Mode'
              : windowName === 'telemetry'
                ? 'Telemetry / Controls'
                : windowName === 'slam'
                  ? 'Sensor Map'
                  : 'Gym Inspector'}
          </button>
        ))}
        <button
          type="button"
          className="window-dock__reset"
          aria-label="Reset cockpit window positions and resizable window sizes"
          onClick={() => setLayoutResetToken((current) => current + 1)}
        >
          <span aria-hidden="true">↺</span>
          Reset Layout
        </button>
      </nav>
      )}
    </div>
  );
}

export default App;
