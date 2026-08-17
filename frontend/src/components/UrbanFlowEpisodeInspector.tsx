import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  createUrbanFlowInspectorSession,
  deleteUrbanFlowInspectorSession,
  resetUrbanFlowInspectorSession,
  stepUrbanFlowInspectorSession,
  type FlowField2DResponse,
  type UrbanFlowInspectorBaseline,
  type UrbanFlowInspectorFrame,
  type UrbanFlowInspectorWorld,
} from '../api';
import {
  buildEpisodeInspectorRenderModel,
  inspectorResponseMatchesSelection,
  inspectorSessionIsStale,
} from '../data/episodeInspectorModel';
import {
  INSPECTOR_PLAYBACK_TICK_MS,
  INSPECTOR_PLAY_RATES,
  inspectorStepResponseIsCurrent,
  isInspectorPlayRate,
  scheduleInspectorPlaybackTick,
  type InspectorPlayRate,
} from '../data/episodeInspectorScheduler';
import {
  liveSelectionIdentity,
  scenarioMatchesLocation,
} from '../data/liveScenarioIdentity';

const INSPECTOR_MAX_STEPS = 1_200;
const SVG_WIDTH = 1_000;
const SVG_HEIGHT = 620;
const BASELINES: Array<{ value: UrbanFlowInspectorBaseline; label: string }> = [
  { value: 'direct_goal', label: 'Direct goal' },
  { value: 'shortest_path', label: 'Shortest path' },
  { value: 'wind_aware_inlet', label: 'Wind-aware inlet' },
];

interface SessionPin {
  scenarioId: string;
  selectionIdentity: string;
  seed: number;
  baseline: UrbanFlowInspectorBaseline;
}

interface UrbanFlowEpisodeInspectorProps {
  flow: FlowField2DResponse | null;
  selectedLocation: { lat: number; lon: number };
  worldLoading: boolean;
  onClose: () => void;
}

function pointList(points: Array<{ x: number; y: number }>) {
  return points.map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' ');
}

function vectorLabel(values: number[], digits: number = 3) {
  return `[${values.map((value) => value.toFixed(digits)).join(', ')}]`;
}

export default function UrbanFlowEpisodeInspector({
  flow,
  selectedLocation,
  worldLoading,
  onClose,
}: UrbanFlowEpisodeInspectorProps) {
  const inlineScenario = flow?.live_scenario ?? null;
  const selectedScenarioId = (
    inlineScenario
    && !worldLoading
    && scenarioMatchesLocation(inlineScenario, selectedLocation)
  ) ? inlineScenario.scenario_id : null;
  const selectionIdentity = liveSelectionIdentity(
    selectedLocation,
    selectedScenarioId,
    worldLoading,
  );
  const [seedInput, setSeedInput] = useState('10007');
  const [baseline, setBaseline] = useState<UrbanFlowInspectorBaseline>('shortest_path');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [world, setWorld] = useState<UrbanFlowInspectorWorld | null>(null);
  const [frame, setFrame] = useState<UrbanFlowInspectorFrame | null>(null);
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<InspectorPlayRate>(4);
  const [lastBatchExecuted, setLastBatchExecuted] = useState(0);
  const [lastBatchReward, setLastBatchReward] = useState(0);
  const [status, setStatus] = useState('Choose a seed and baseline, then reset to create a replay session.');
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);
  const sessionIdRef = useRef<string | null>(null);
  const sessionPinRef = useRef<SessionPin | null>(null);
  const selectionIdentityRef = useRef(selectionIdentity);
  const selectedScenarioIdRef = useRef(selectedScenarioId);
  const requestRef = useRef<AbortController | null>(null);
  const busyRef = useRef(false);
  const playbackCreditQuantaRef = useRef(0);
  const playingRef = useRef(false);
  const lifecycleGenerationRef = useRef(0);

  selectionIdentityRef.current = selectionIdentity;
  selectedScenarioIdRef.current = selectedScenarioId;

  const clearLocalSession = useCallback((clearVisuals: boolean = true) => {
    sessionIdRef.current = null;
    sessionPinRef.current = null;
    lifecycleGenerationRef.current += 1;
    setSessionId(null);
    playingRef.current = false;
    setPlaying(false);
    playbackCreditQuantaRef.current = 0;
    if (clearVisuals) {
      setWorld(null);
      setFrame(null);
    }
  }, []);

  const deleteBestEffort = useCallback((id: string) => {
    void deleteUrbanFlowInspectorSession(id).catch(() => {
      // TTL/LRU cleanup or terminal cleanup may have removed it first.
    });
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      requestRef.current?.abort();
      lifecycleGenerationRef.current += 1;
      playingRef.current = false;
      const id = sessionIdRef.current;
      sessionIdRef.current = null;
      sessionPinRef.current = null;
      if (id) deleteBestEffort(id);
    };
  }, [deleteBestEffort]);

  useEffect(() => {
    const pin = sessionPinRef.current;
    if (!pin || !inspectorSessionIsStale(
      pin.scenarioId,
      selectedScenarioId,
      pin.selectionIdentity,
      selectionIdentity,
    )) return;

    requestRef.current?.abort();
    requestRef.current = null;
    busyRef.current = false;
    setBusy(false);
    const staleSessionId = sessionIdRef.current;
    clearLocalSession();
    if (staleSessionId) deleteBestEffort(staleSessionId);
    setError('Selected location changed; the stale inspector session was aborted and deleted.');
    setStatus('Reset after the new live OSM scenario finishes loading.');
  }, [clearLocalSession, deleteBestEffort, selectedScenarioId, selectionIdentity]);

  const beginRequest = useCallback(() => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    return controller;
  }, []);

  const finishRequest = useCallback((controller: AbortController) => {
    if (requestRef.current !== controller) return;
    requestRef.current = null;
    busyRef.current = false;
    if (mountedRef.current) setBusy(false);
  }, []);

  const handleReset = async () => {
    const seed = Number(seedInput);
    const scenarioId = selectedScenarioIdRef.current;
    if (!Number.isInteger(seed) || seed < 0 || seed > 2_147_483_647) {
      setError('Seed must be an integer from 0 through 2147483647.');
      return;
    }
    if (!scenarioId || worldLoading) {
      setError('Wait for the selected live OSM scenario to finish registering.');
      return;
    }
    const controller = beginRequest();
    playingRef.current = false;
    setPlaying(false);
    playbackCreditQuantaRef.current = 0;
    setStatus(sessionIdRef.current ? 'Resetting pinned episode…' : 'Creating bounded episode session…');
    const expectedIdentity = selectionIdentityRef.current;
    const existingSessionId = sessionIdRef.current;
    const existingPin = sessionPinRef.current;
    try {
      const canReset = Boolean(
        existingSessionId
        && existingPin
        && existingPin.scenarioId === scenarioId
        && existingPin.seed === seed
        && existingPin.baseline === baseline,
      );
      if (existingSessionId && !canReset) {
        clearLocalSession();
        deleteBestEffort(existingSessionId);
      }
      const requestGeneration = lifecycleGenerationRef.current;
      const result = canReset && existingSessionId
        ? await resetUrbanFlowInspectorSession(existingSessionId, { signal: controller.signal })
        : await createUrbanFlowInspectorSession(
          scenarioId,
          seed,
          baseline,
          INSPECTOR_MAX_STEPS,
          { signal: controller.signal },
        );
      if (
        controller.signal.aborted
        || requestRef.current !== controller
        || lifecycleGenerationRef.current !== requestGeneration
        || !mountedRef.current
      ) return;
      if (!inspectorResponseMatchesSelection(
        result,
        expectedIdentity,
        selectionIdentityRef.current,
        scenarioId,
        canReset ? existingSessionId : null,
      )) {
        deleteBestEffort(result.session_id);
        clearLocalSession();
        setError('Selected scenario changed while the episode was being prepared; stale session deleted.');
        return;
      }
      if (!result.world) throw new Error('Inspector reset did not return registered world geometry.');
      sessionIdRef.current = result.session_id;
      sessionPinRef.current = { scenarioId, selectionIdentity: expectedIdentity, seed, baseline };
      setSessionId(result.session_id);
      setWorld(result.world);
      setFrame(result.frame);
      setLastBatchExecuted(result.executed_steps);
      setLastBatchReward(result.batch_reward);
      setStatus(`Episode ready · ${result.limits.max_steps}-step / ${result.limits.simulated_max_s.toFixed(0)} s hard limit · ${result.limits.ttl_s / 60} min TTL`);
    } catch (reason) {
      if (controller.signal.aborted || !mountedRef.current) return;
      setError(reason instanceof Error ? reason.message : 'Inspector session reset failed.');
      setStatus('No active episode replay.');
    } finally {
      finishRequest(controller);
    }
  };

  const stepBatch = useCallback(async (repeat: number) => {
    const id = sessionIdRef.current;
    const pin = sessionPinRef.current;
    if (!id || !pin || busyRef.current) return;
    if (inspectorSessionIsStale(
      pin.scenarioId,
      selectedScenarioIdRef.current,
      pin.selectionIdentity,
      selectionIdentityRef.current,
    )) {
      clearLocalSession();
      deleteBestEffort(id);
      setError('The live scenario changed; stale session deleted before stepping.');
      return;
    }
    const requestGeneration = lifecycleGenerationRef.current;
    const controller = beginRequest();
    try {
      const result = await stepUrbanFlowInspectorSession(id, {
        signal: controller.signal,
        repeat,
      });
      if (
        !inspectorStepResponseIsCurrent(
          id,
          sessionIdRef.current,
          requestGeneration,
          lifecycleGenerationRef.current,
          requestRef.current === controller,
          controller.signal.aborted,
        )
        || !mountedRef.current
      ) return;
      if (!inspectorResponseMatchesSelection(
        result,
        pin.selectionIdentity,
        selectionIdentityRef.current,
        pin.scenarioId,
        id,
      )) {
        clearLocalSession();
        deleteBestEffort(result.session_id);
        setError('A stale inspector frame was rejected and its session was deleted.');
        return;
      }
      setFrame(result.frame);
      setLastBatchExecuted(result.executed_steps);
      setLastBatchReward(result.batch_reward);
      if (!result.session_active) {
        // The backend already removed terminal sessions after returning this final frame.
        sessionIdRef.current = null;
        sessionPinRef.current = null;
        lifecycleGenerationRef.current += 1;
        setSessionId(null);
        playingRef.current = false;
        setPlaying(false);
        setStatus(`Episode ${result.frame.status} at step ${result.frame.step_index} (${result.frame.termination_reason ?? 'terminal'}); terminal session cleaned up.`);
      } else {
        setStatus(`Replaying ${result.frame.baseline.label} · advanced ${result.executed_steps} step${result.executed_steps === 1 ? '' : 's'} · step ${result.frame.step_index}/${result.frame.max_steps}`);
      }
    } catch (reason) {
      if (controller.signal.aborted || !mountedRef.current) return;
      playingRef.current = false;
      setPlaying(false);
      setError(reason instanceof Error ? reason.message : 'Inspector step failed.');
    } finally {
      finishRequest(controller);
    }
  }, [beginRequest, clearLocalSession, deleteBestEffort, finishRequest]);

  useEffect(() => {
    if (!playing || !sessionId || frame?.terminated || frame?.truncated) return;
    const timer = window.setInterval(() => {
      if (!playingRef.current || sessionIdRef.current !== sessionId) return;
      const tick = scheduleInspectorPlaybackTick(
        speed,
        playbackCreditQuantaRef.current,
        busyRef.current,
      );
      playbackCreditQuantaRef.current = tick.creditQuanta;
      if (tick.repeat > 0) void stepBatch(tick.repeat);
    }, INSPECTOR_PLAYBACK_TICK_MS);
    return () => window.clearInterval(timer);
  }, [frame?.terminated, frame?.truncated, playing, sessionId, speed, stepBatch]);

  const handleManualStep = () => {
    playingRef.current = false;
    setPlaying(false);
    playbackCreditQuantaRef.current = 0;
    void stepBatch(1);
  };

  const handlePlayPause = () => {
    playbackCreditQuantaRef.current = 0;
    playingRef.current = !playingRef.current;
    setPlaying(playingRef.current);
  };

  const handleDelete = async () => {
    requestRef.current?.abort();
    requestRef.current = null;
    busyRef.current = false;
    playbackCreditQuantaRef.current = 0;
    setBusy(false);
    const id = sessionIdRef.current;
    clearLocalSession();
    setError(null);
    if (!id) {
      setStatus('No active session to delete.');
      return;
    }
    try {
      await deleteUrbanFlowInspectorSession(id);
      setStatus('Inspector session deleted.');
    } catch (reason) {
      setStatus('Inspector session closed locally.');
      setError(reason instanceof Error ? reason.message : 'Session delete failed.');
    }
  };

  const retireForControlChange = () => {
    const id = sessionIdRef.current;
    requestRef.current?.abort();
    requestRef.current = null;
    busyRef.current = false;
    playbackCreditQuantaRef.current = 0;
    setBusy(false);
    clearLocalSession();
    if (id) deleteBestEffort(id);
    setError(null);
    setStatus('Controls changed; reset to create a newly pinned deterministic episode.');
  };

  const renderModel = useMemo(() => (
    world && frame
      ? buildEpisodeInspectorRenderModel(world, frame, SVG_WIDTH, SVG_HEIGHT)
      : null
  ), [frame, world]);
  const validSeed = Number.isInteger(Number(seedInput))
    && Number(seedInput) >= 0
    && Number(seedInput) <= 2_147_483_647;

  return (
    <section className="episode-inspector" aria-labelledby="episode-inspector-title">
      <header className="episode-inspector__title panel-window-title panel-window-title--dark panel-window-title--draggable" title="Drag to move window">
        <span id="episode-inspector-title">UrbanFlow Gym Episode Inspector <small>⠿ Move · north-up</small></span>
        <button type="button" onPointerDown={(event) => event.stopPropagation()} onClick={onClose} aria-label="Hide Gym Episode Inspector window">×</button>
      </header>

      <div className="episode-inspector__boundary-strip" aria-label="Inspector claim boundaries">
        <strong>LIVE OSM WORLD</strong>
        <span>POLICY NOT TRAINED</span>
        <span>FULL FLOW ACCESS: NO</span>
        <span>TRAINING: NO</span>
        <span>REAL CFD: NO · ADAPTER ONLY</span>
      </div>

      <div className="episode-inspector__controls">
        <label>
          <span>Seed</span>
          <input
            type="number"
            min="0"
            max="2147483647"
            step="1"
            value={seedInput}
            onChange={(event) => {
              retireForControlChange();
              setSeedInput(event.target.value);
            }}
          />
        </label>
        <label>
          <span>Deterministic baseline</span>
          <select
            value={baseline}
            onChange={(event) => {
              retireForControlChange();
              setBaseline(event.target.value as UrbanFlowInspectorBaseline);
            }}
          >
            {BASELINES.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </label>
        <button type="button" onClick={() => void handleReset()} disabled={busy || !selectedScenarioId || !validSeed}>↺ Reset</button>
        <button type="button" onClick={handleManualStep} disabled={busy || !sessionId}>Step</button>
        <button
          type="button"
          className={playing ? 'episode-inspector__pause' : ''}
          onClick={handlePlayPause}
          disabled={!sessionId || Boolean(frame?.terminated || frame?.truncated)}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <label className="episode-inspector__speed">
          <span>Steps / second</span>
          <select
            value={speed}
            onChange={(event) => {
              const nextSpeed = Number(event.target.value);
              if (!isInspectorPlayRate(nextSpeed)) return;
              playbackCreditQuantaRef.current = 0;
              setSpeed(nextSpeed);
            }}
          >
            {INSPECTOR_PLAY_RATES.map((rate) => (
              <option key={rate} value={rate}>{rate}×</option>
            ))}
          </select>
        </label>
        <button type="button" className="episode-inspector__delete" onClick={() => void handleDelete()} disabled={!sessionId && !frame}>Delete</button>
      </div>

      <div className="episode-inspector__status" aria-live="polite">
        <span className={busy ? 'busy' : ''}>{busy ? 'WORKING' : sessionId ? 'SESSION LIVE' : 'SESSION CLOSED'}</span>
        <p>{status}</p>
        {error && <strong>{error}</strong>}
      </div>
      <p className="episode-inspector__batch-note">
        Requested speed: {speed} simulated step{speed === 1 ? '' : 's'}/s · high-speed visual updates are batched at up to 4 requests/s, with no overlapping requests.
      </p>

      <div className="episode-inspector__body">
        <div className="episode-inspector__map-column">
          <div className="episode-inspector__channel-label episode-inspector__channel-label--policy">
            <span>POLICY-VISIBLE / BASELINE ROUTE CONTEXT</span>
            <small>OSM geometry · own pose/history · goal · actor LiDAR · guidance action</small>
          </div>
          <div className="episode-inspector__viewport">
            {renderModel && frame && world ? (
              <svg viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`} role="img" aria-label="North-up registered OSM episode replay with drone trajectory and actor LiDAR">
                <defs>
                  <marker id="ufi-heading-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                    <path d="M0,0 L7,3.5 L0,7 Z" className="episode-inspector__heading-marker" />
                  </marker>
                  <marker id="ufi-action-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                    <path d="M0,0 L7,3.5 L0,7 Z" className="episode-inspector__action-marker" />
                  </marker>
                </defs>
                <rect x="0" y="0" width={SVG_WIDTH} height={SVG_HEIGHT} className="episode-inspector__map-bg" />
                <g className="episode-inspector__buildings">
                  {renderModel.buildings.map((building) => (
                    <polygon key={building.buildingId} points={pointList(building.points)}>
                      <title>{building.buildingId} · {building.heightM.toFixed(1)} m</title>
                    </polygon>
                  ))}
                </g>
                <polyline className="episode-inspector__trajectory" points={pointList(renderModel.trajectory)} />
                <g className="episode-inspector__lidar">
                  {renderModel.lidarRays.map((ray, index) => (
                    <line
                      key={`${frame.step_index}-${index}`}
                      className={ray.hit ? 'hit' : 'clear'}
                      x1={ray.start.x}
                      y1={ray.start.y}
                      x2={ray.end.x}
                      y2={ray.end.y}
                    />
                  ))}
                </g>
                <g className="episode-inspector__waypoint episode-inspector__waypoint--start">
                  <circle cx={renderModel.start.x} cy={renderModel.start.y} r="8" />
                  <text x={renderModel.start.x + 11} y={renderModel.start.y - 10}>START</text>
                </g>
                <g className="episode-inspector__waypoint episode-inspector__waypoint--goal">
                  <circle cx={renderModel.goal.x} cy={renderModel.goal.y} r="10" />
                  <circle cx={renderModel.goal.x} cy={renderModel.goal.y} r="4" />
                  <text x={renderModel.goal.x + 13} y={renderModel.goal.y - 11}>GOAL</text>
                </g>
                <line
                  className="episode-inspector__heading"
                  x1={renderModel.drone.x}
                  y1={renderModel.drone.y}
                  x2={renderModel.headingEnd.x}
                  y2={renderModel.headingEnd.y}
                  markerEnd="url(#ufi-heading-arrow)"
                />
                <line
                  className="episode-inspector__action"
                  x1={renderModel.drone.x}
                  y1={renderModel.drone.y}
                  x2={renderModel.actionEnd.x}
                  y2={renderModel.actionEnd.y}
                  markerEnd="url(#ufi-action-arrow)"
                />
                <g transform={`translate(${renderModel.drone.x} ${renderModel.drone.y}) rotate(${-frame.drone.heading_rad * 180 / Math.PI})`} className="episode-inspector__drone">
                  <polygon points="11,0 -7,-7 -4,0 -7,7" />
                  <circle cx="0" cy="0" r="3" />
                </g>
                <g className="episode-inspector__north" transform={`translate(${SVG_WIDTH - 38} 42)`}>
                  <path d="M0,22 L0,-13 M0,-13 L-6,-2 M0,-13 L6,-2" />
                  <text x="0" y="-20">N</text>
                </g>
              </svg>
            ) : (
              <div className="episode-inspector__empty">
                <strong>No episode frame</strong>
                <p>Reset creates a bounded backend session in the exact selected registered scenario. No toy fallback is available.</p>
              </div>
            )}
          </div>
          <div className="episode-inspector__legend">
            <span><i className="building" /> OSM footprint</span>
            <span><i className="trajectory" /> Own trajectory</span>
            <span><i className="lidar" /> Actor LiDAR</span>
            <span><i className="action" /> Local action</span>
            <span><i className="heading" /> Heading</span>
          </div>
          {world && frame && (
            <div className="episode-inspector__world-meta">
              <span>{world.structure_count.toLocaleString()} OSM polygons</span>
              <span>Inlet {vectorLabel(world.known_inlet.velocity_xy_mps, 2)} m/s</span>
              <span>Bounds {world.bounds.min_x_m.toFixed(0)}…{world.bounds.max_x_m.toFixed(0)} E / {world.bounds.min_y_m.toFixed(0)}…{world.bounds.max_y_m.toFixed(0)} N m</span>
            </div>
          )}
        </div>

        <aside className="episode-inspector__readouts">
          {frame ? (
            <>
              <section className="episode-inspector__policy-readout">
                <div className="episode-inspector__channel-label episode-inspector__channel-label--policy"><span>POLICY-VISIBLE</span></div>
                <dl className="episode-inspector__quick-grid">
                  <div><dt>Step</dt><dd>{frame.step_index} / {frame.max_steps}</dd></div>
                  <div><dt>Status</dt><dd className={`status-${frame.status}`}>{frame.status}</dd></div>
                  <div><dt>Pose E/N</dt><dd>{vectorLabel(frame.drone.position_xy_m, 2)} m</dd></div>
                  <div><dt>Heading</dt><dd>{(frame.drone.heading_rad * 180 / Math.PI).toFixed(1)}°</dd></div>
                  <div><dt>Action F/L</dt><dd>{vectorLabel(frame.local_guidance_action.vector)}</dd></div>
                  <div><dt>Air-relative</dt><dd>{vectorLabel(frame.air_relative_velocity_xy_mps, 2)} m/s</dd></div>
                </dl>
                <div className="episode-inspector__observations">
                  <h3>Actor observation · {frame.actor_observation.vector.length} scalars</h3>
                  {frame.actor_observation.fields.map((field) => (
                    <div key={field.name}>
                      <span title={field.source}>{field.name}</span>
                      <code>{vectorLabel(field.values)}</code>
                    </div>
                  ))}
                </div>
              </section>

              <section className="episode-inspector__debug-readout">
                <div className="episode-inspector__channel-label episode-inspector__channel-label--debug">
                  <span>PLAYBACK + HORIZON · NOT ACTOR INPUT</span>
                  <small>Display batches only; every simulation step is preserved</small>
                </div>
                <dl className="episode-inspector__quick-grid">
                  <div><dt>Simulation dt</dt><dd>{frame.dt_s.toFixed(2)} s</dd></div>
                  <div><dt>Simulated time</dt><dd>{frame.simulated_elapsed_s.toFixed(2)} / {frame.simulated_max_s.toFixed(2)} s</dd></div>
                  <div><dt>Requested speed</dt><dd>{speed} step{speed === 1 ? '' : 's'}/s · {playing ? 'playing' : 'paused'}</dd></div>
                  <div><dt>Last batch</dt><dd>{lastBatchExecuted} executed</dd></div>
                  <div><dt>Distance to goal</dt><dd>{frame.distance_to_goal_m.toFixed(2)} m</dd></div>
                  <div><dt>Minimum remaining</dt><dd>{frame.estimated_minimum_steps} steps · {frame.estimated_minimum_time_s.toFixed(2)} s</dd></div>
                </dl>
              </section>

              <section className="episode-inspector__debug-readout">
                <div className="episode-inspector__channel-label episode-inspector__channel-label--debug">
                  <span>EPISODE DEBUG · NOT ACTOR INPUT</span>
                  <small>No privileged flow overlay</small>
                </div>
                <dl className="episode-inspector__quick-grid">
                  <div><dt>Clearance</dt><dd>{Number.isFinite(frame.clearance_m) ? `${frame.clearance_m.toFixed(2)} m` : '∞'}</dd></div>
                  <div><dt>Collision</dt><dd>{frame.collision ? 'YES' : 'NO'}</dd></div>
                  <div><dt>Step reward</dt><dd>{frame.reward.step_total.toFixed(4)}</dd></div>
                  <div><dt>Batch reward</dt><dd>{lastBatchReward.toFixed(4)}</dd></div>
                  <div><dt>Total reward</dt><dd>{frame.reward.episode_total.toFixed(3)}</dd></div>
                </dl>
                <div className="episode-inspector__rewards">
                  {Object.entries(frame.reward.components).map(([name, value]) => (
                    <div key={name}><span>{name}</span><code className={value > 0 ? 'positive' : value < 0 ? 'negative' : ''}>{value.toFixed(4)}</code></div>
                  ))}
                </div>
              </section>

              <section className="episode-inspector__identity">
                <span>Scenario ID</span>
                <code title={frame.scenario_id}>{frame.scenario_id}</code>
                <span>Session</span>
                <code title={sessionId ?? 'terminal session deleted'}>{sessionId ?? 'terminal / deleted'}</code>
                <p>Seed {frame.seed} · {frame.baseline.label} · baseline full-flow access NO</p>
              </section>
            </>
          ) : (
            <div className="episode-inspector__readout-empty">
              <strong>Deterministic headless replay</strong>
              <p>The baseline runs one Gym-style episode server-side. This browser does not train a policy or issue motor commands.</p>
              <p>{selectedScenarioId ? `Ready for live scenario ${selectedScenarioId}.` : 'Waiting for a registered scenario matching the selected location.'}</p>
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}
