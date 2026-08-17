import { useEffect, useRef, useState } from 'react';
import {
  evaluateUrbanFlowLiveBaselines,
  fetchUrbanFlowLiveScenario,
  type FlowField2DResponse,
  type UrbanFlowEvaluationSummary,
  type UrbanFlowLiveScenarioSummary,
} from '../api';
import {
  liveEvaluationMatchesSelection,
  liveSelectionIdentity,
  scenarioMatchesLocation,
} from '../data/liveScenarioIdentity';

const BASELINE_ORDER = ['direct_goal', 'shortest_path', 'wind_aware_inlet'] as const;
// The cockpit runs a bounded single-seed preview so a polygon-heavy live world
// stays interactive on the development Mac mini. Multi-seed evaluation remains
// available through the backend API/CLI; no policy training runs here.
const EVALUATION_SEEDS = [10007];
const MAX_STEPS = 180;

interface UrbanFlowGymPanelProps {
  flow: FlowField2DResponse | null;
  selectedLocation: { lat: number; lon: number };
  worldLoading: boolean;
}

export default function UrbanFlowGymPanel({
  flow,
  selectedLocation,
  worldLoading,
}: UrbanFlowGymPanelProps) {
  const inlineScenario = flow?.live_scenario ?? null;
  const [registeredScenario, setRegisteredScenario] = useState<UrbanFlowLiveScenarioSummary | null>(null);
  const [evaluation, setEvaluation] = useState<UrbanFlowEvaluationSummary | null>(null);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const evaluationRequestRef = useRef<AbortController | null>(null);
  const metadataRequestRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);
  const selectedScenarioId = inlineScenario?.scenario_id ?? null;
  const selectedIdentity = liveSelectionIdentity(selectedLocation, selectedScenarioId, worldLoading);
  const selectedIdentityRef = useRef(selectedIdentity);
  selectedIdentityRef.current = selectedIdentity;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      evaluationRequestRef.current?.abort();
      metadataRequestRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    const hadEvaluationRequest = evaluationRequestRef.current !== null;
    evaluationRequestRef.current?.abort();
    evaluationRequestRef.current = null;
    metadataRequestRef.current?.abort();
    setEvaluating(false);
    setEvaluation(null);
    setRegisteredScenario(null);
    setError(hadEvaluationRequest ? 'Selected location changed; stale baseline results were discarded.' : null);

    if (!inlineScenario || worldLoading || !scenarioMatchesLocation(inlineScenario, selectedLocation)) {
      return;
    }

    const controller = new AbortController();
    const expectedIdentity = selectedIdentity;
    metadataRequestRef.current = controller;
    void fetchUrbanFlowLiveScenario(inlineScenario.scenario_id, { signal: controller.signal })
      .then((scenario) => {
        if (
          !mountedRef.current
          || controller.signal.aborted
          || selectedIdentityRef.current !== expectedIdentity
          || scenario.scenario_id !== inlineScenario.scenario_id
        ) return;
        setRegisteredScenario(scenario);
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted || !mountedRef.current) return;
        setError(reason instanceof Error ? reason.message : 'Live scenario verification failed.');
      })
      .finally(() => {
        if (metadataRequestRef.current === controller) metadataRequestRef.current = null;
      });

    return () => controller.abort();
  }, [inlineScenario, selectedIdentity, selectedLocation, worldLoading]);

  const compareBaselines = async () => {
    if (!registeredScenario || !scenarioMatchesLocation(registeredScenario, selectedLocation) || worldLoading) {
      setError('Wait for the selected live OSM world to finish registering before evaluation.');
      return;
    }
    evaluationRequestRef.current?.abort();
    const controller = new AbortController();
    const expectedIdentity = selectedIdentityRef.current;
    const expectedScenarioId = registeredScenario.scenario_id;
    evaluationRequestRef.current = controller;
    setEvaluating(true);
    setError(null);

    try {
      const result = await evaluateUrbanFlowLiveBaselines(
        expectedScenarioId,
        EVALUATION_SEEDS,
        MAX_STEPS,
        { signal: controller.signal },
      );
      if (controller.signal.aborted || !mountedRef.current) return;
      if (
        !liveEvaluationMatchesSelection(
          result,
          expectedIdentity,
          selectedIdentityRef.current,
          expectedScenarioId,
        )
      ) {
        setEvaluation(null);
        setError('Selected location changed; stale baseline results were discarded.');
        return;
      }
      setEvaluation(result);
    } catch (reason) {
      if (controller.signal.aborted || !mountedRef.current) return;
      setError(reason instanceof Error ? reason.message : 'UrbanFlow live evaluation failed.');
    } finally {
      if (mountedRef.current && evaluationRequestRef.current === controller) {
        evaluationRequestRef.current = null;
        setEvaluating(false);
      }
    }
  };

  const scenario = registeredScenario ?? inlineScenario;
  const isSelectedWorld = Boolean(
    scenario
    && !worldLoading
    && scenarioMatchesLocation(scenario, selectedLocation),
  );

  return (
    <section className="urbanflow-gym" aria-labelledby="urbanflow-gym-title">
      <div className="urbanflow-gym__heading">
        <span>Registered backend world bridge</span>
        <strong id="urbanflow-gym-title">URBANFLOW GYM / LIVE OSM WORLD</strong>
      </div>
      <div className="urbanflow-gym__boundaries" aria-label="UrbanFlow Gym status boundaries">
        <div><span>WORLD</span><strong>LIVE OSM WORLD</strong></div>
        <div><span>FULL FLOW ACCESS</span><strong>NO</strong></div>
        <div><span>POLICY</span><strong>NOT TRAINED</strong></div>
        <div><span>REAL CFD VALIDATION</span><strong>NO</strong></div>
      </div>
      {scenario ? (
        <dl className="urbanflow-gym__scenario">
          <div>
            <dt>Location</dt>
            <dd>{scenario.location.selected_lat_deg.toFixed(6)}, {scenario.location.selected_lon_deg.toFixed(6)}</dd>
          </div>
          <div>
            <dt>Structures</dt>
            <dd>{scenario.structure_count.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Inlet</dt>
            <dd>
              {scenario.inlet.speed_mps.toFixed(2)} m/s · [{scenario.inlet.velocity_xy_mps.map((value) => value.toFixed(2)).join(', ')}]
            </dd>
          </div>
          <div>
            <dt>Frame</dt>
            <dd>LOCAL X EAST / Y NORTH · METRES</dd>
          </div>
          <div className="urbanflow-gym__scenario-id">
            <dt>Scenario ID</dt>
            <dd title={scenario.scenario_id}>{scenario.scenario_id}</dd>
          </div>
        </dl>
      ) : (
        <p>{worldLoading ? 'Registering the selected OSM world…' : 'No live scenario is registered for this location.'}</p>
      )}
      <p>
        The hidden CFD-lite grid comes from this geometry and inlet. Policies receive no full field;
        no training and no real Navier–Stokes CFD validation were run.
      </p>
      <button
        type="button"
        className="urbanflow-gym__compare"
        onClick={compareBaselines}
        disabled={evaluating || !registeredScenario || !isSelectedWorld}
      >
        {evaluating ? 'Running bounded live preview…' : 'Compare live-world baselines'}
      </button>
      <div className="urbanflow-gym__result" aria-live="polite">
        {error && <p className="urbanflow-gym__error">{error}</p>}
        {evaluation && (
          <>
            <div className="urbanflow-gym__result-meta">
              <span>Preview seed {evaluation.evaluation_config.seeds.join(' · ')}</span>
              <span>ID {evaluation.evaluation_id}</span>
            </div>
            <div className="urbanflow-gym__table-wrap">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Baseline</th>
                    <th scope="col">Success</th>
                    <th scope="col">Hits</th>
                    <th scope="col">Energy proxy</th>
                    <th scope="col">Path</th>
                  </tr>
                </thead>
                <tbody>
                  {BASELINE_ORDER.map((baselineId) => evaluation.baselines[baselineId]).filter(Boolean).map((baseline) => (
                    <tr key={baseline.baseline_id}>
                      <th scope="row">{baseline.label}</th>
                      <td data-label="Success">{Math.round(baseline.aggregate.success_rate * 100)}%</td>
                      <td data-label="Hits">{baseline.aggregate.collision_count}</td>
                      <td data-label="Energy">{baseline.aggregate.mean_relative_air_speed_energy.toFixed(0)}</td>
                      <td data-label="Path">{baseline.aggregate.mean_path_length_m.toFixed(1)} m</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="urbanflow-gym__result-note">
              Same immutable live OSM world · full flow access no · side-by-side metrics make no baseline superiority claim
            </p>
          </>
        )}
      </div>
    </section>
  );
}
