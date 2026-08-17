import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';

const vite = await createServer({
  root: fileURLToPath(new URL('..', import.meta.url)),
  appType: 'custom',
  server: { middlewareMode: true, hmr: false },
});

const calls = [];
const scenarioId = 'urbanflow-live-v1-1234567890abcdef12345678';
const scenario = {
  schema_id: 'urbanflow.live_scenario.v1',
  schema_version: 1,
  scenario_id: scenarioId,
  content_hash_sha256: 'a'.repeat(64),
  location: { selected_lat_deg: 37.451448, selected_lon_deg: 126.6515423 },
};
const evaluation = {
  scenario_kind: 'live_osm_current_inlet',
  scenario_id: scenarioId,
  live_scenario: scenario,
};

globalThis.fetch = async (url, options = {}) => {
  calls.push({ url: String(url), options });
  const payload = String(url).endsWith('/live/evaluate') ? evaluation : scenario;
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
};

try {
  const {
    activateUrbanFlowLiveScenario,
    evaluateUrbanFlowLiveBaselines,
    fetchUrbanFlowLiveScenario,
  } = await vite.ssrLoadModule('/src/api.ts');
  const {
    liveEvaluationMatchesSelection,
    liveSelectionIdentity,
    scenarioMatchesLocation,
  } = await vite.ssrLoadModule('/src/data/liveScenarioIdentity.ts');

  await fetchUrbanFlowLiveScenario(scenarioId);
  await activateUrbanFlowLiveScenario(scenarioId);
  const result = await evaluateUrbanFlowLiveBaselines(scenarioId, [10007], 50);
  const evaluationCall = calls.at(-1);
  const body = JSON.parse(evaluationCall.options.body);
  if (
    !evaluationCall.url.endsWith('/urbanflow-gym/live/evaluate')
    || body.scenario_id !== scenarioId
    || JSON.stringify(body.seeds) !== JSON.stringify([10007])
    || body.max_steps !== 50
  ) {
    throw new Error('Expected the cockpit API client to bind live evaluation to the selected scenario id and runtime limits.');
  }
  if (
    !calls[0].url.includes(`/live-scenarios/${scenarioId}/summary`)
    || !calls[1].url.includes(`/live-scenarios/${scenarioId}/activate`)
    || calls[1].options.method !== 'POST'
  ) {
    throw new Error('Expected bounded live scenario verification and cache reactivation routes.');
  }

  const selectedLocation = { lat: 37.451448, lon: 126.6515423 };
  const expectedIdentity = liveSelectionIdentity(selectedLocation, scenarioId, false);
  if (
    !scenarioMatchesLocation(scenario, selectedLocation)
    || !liveEvaluationMatchesSelection(result, expectedIdentity, expectedIdentity, scenarioId)
    || liveEvaluationMatchesSelection(
      result,
      expectedIdentity,
      liveSelectionIdentity({ lat: 37.5, lon: 127.0 }, scenarioId, true),
      scenarioId,
    )
    || liveEvaluationMatchesSelection(result, expectedIdentity, expectedIdentity, `${scenarioId}-stale`)
  ) {
    throw new Error('Expected changed locations and stale scenario ids to reject in-flight baseline results.');
  }

  console.log(JSON.stringify({
    status: 'ok',
    tests: 4,
    contract: 'live summary lookup, cache activation, scenario-bound evaluation, stale-result rejection',
  }));
} finally {
  await vite.close();
}
