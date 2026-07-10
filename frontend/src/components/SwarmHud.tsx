import type { RLBaselineResponse } from '../api';
import { rewardTerm, type SwarmAggregateMetrics } from '../appModel';
import Metric from './Metric';
import type { SwarmReplayStatus } from './SwarmReplay';

interface SwarmHudProps {
  swarmMetrics: SwarmAggregateMetrics | null;
  swarmReplayStatus: SwarmReplayStatus;
  rlBaseline: RLBaselineResponse | null;
  rlError: string | null;
  sweptBuildingHits: number;
  boundaryViolations: number;
}

export default function SwarmHud({ swarmMetrics, swarmReplayStatus, rlBaseline, rlError, sweptBuildingHits, boundaryViolations }: SwarmHudProps) {
  const successCount = rlBaseline?.metrics.success_count ?? swarmMetrics?.success_count ?? swarmReplayStatus.droneCount;
  const droneCount = (rlBaseline?.n_drones ?? swarmMetrics?.n_drones ?? swarmReplayStatus.droneCount) || '--';
  const minSeparation = rlBaseline?.cost_metrics.min_pairwise_separation_m !== undefined
    ? `${rlBaseline.cost_metrics.min_pairwise_separation_m.toFixed(2)} m`
    : swarmMetrics ? `${swarmMetrics.min_pairwise_separation_m.toFixed(2)} m` : '--';
  const pathLength = rlBaseline?.cost_metrics.path_length_m !== undefined
    ? `${rlBaseline.cost_metrics.path_length_m.toFixed(1)} m`
    : swarmMetrics ? `${swarmMetrics.total_path_length_m.toFixed(1)} m` : '--';

  return (
    <>
      <div className="hud-grid">
        <Metric label="Drones" value={`${successCount}/${droneCount}`} />
        <Metric label="Building Collisions" value={swarmMetrics?.total_collisions ?? rlBaseline?.cost_metrics.collisions ?? '--'} />
        <Metric label="Swept Hits" value={sweptBuildingHits} />
        <Metric label="Boundary Blocks" value={boundaryViolations} />
        <Metric label="Near Miss <10m" value={swarmMetrics?.near_miss_count_sep_lt_10m ?? '--'} />
        <Metric label="Min Separation" value={minSeparation} />
        <Metric label="Total Path Length" value={pathLength} wide />
        <Metric label="Relative Airspeed Energy Proxy" value={swarmMetrics ? swarmMetrics.total_energy_relative_airspeed_l2.toFixed(1) : rlBaseline?.cost_metrics.energy_relative_airspeed_l2.toFixed(1) ?? '--'} wide accent />
      </div>
      <div className="rl-readiness">
        <div className="section-header">
          <span>RL readiness</span>
          <strong>{rlBaseline ? 'API LIVE' : rlError ? 'API OFF' : 'LOADING'}</strong>
        </div>
        <p>{rlBaseline?.policy_label ?? 'Deterministic baseline, not trained RL.'}</p>
        <p>Observation contract: OSM/building sectors + inlet wind only; hidden flow-grid access is {rlBaseline ? String(rlBaseline.policy_had_privileged_flow_access) : 'false'}.</p>
        <div className="reward-grid">
          <Metric label="Return" value={rlBaseline ? rlBaseline.metrics.return.toFixed(2) : '--'} />
          <Metric label="Progress term" value={rlBaseline ? rewardTerm(rlBaseline, 'progress').toFixed(2) : '--'} />
          <Metric label="Energy cost" value={rlBaseline ? rewardTerm(rlBaseline, 'energy_cost').toFixed(2) : '--'} />
          <Metric label="Collision cost" value={rlBaseline ? rewardTerm(rlBaseline, 'collision_cost').toFixed(2) : '--'} />
          <Metric label="Separation cost" value={rlBaseline ? rewardTerm(rlBaseline, 'separation_cost').toFixed(2) : '--'} />
          <Metric label="Building clearance" value={rlBaseline ? `${rlBaseline.cost_metrics.min_building_clearance_m.toFixed(2)} m` : '--'} />
          <Metric label="Trajectory frames" value={rlBaseline?.trajectory.length ?? '--'} />
        </div>
        {rlError && <p className="status-bar error">{rlError}</p>}
      </div>
    </>
  );
}
