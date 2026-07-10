import type { BackendState } from '../appModel';

interface CommandBarProps {
  modelStatus: string;
  backendState: BackendState;
  backendLabel: string;
  modeLabel: string;
  solverLabel: string;
  buildingCount: number;
  windSpeed: number;
  energyRate: number;
}

export default function CommandBar({
  modelStatus,
  backendState,
  backendLabel,
  modeLabel,
  solverLabel,
  buildingCount,
  windSpeed,
  energyRate,
}: CommandBarProps) {
  const backendPillState = backendState === 'connected' ? 'live' : backendState === 'checking' ? 'syncing' : 'offline';

  return (
    <header className="command-bar">
      <div className="command-title">
        <span>Urban Flighter</span>
        <strong>Drag-Aware Drone Simulator</strong>
      </div>
      <div className="command-pills" aria-label="Simulation status">
        <span data-state={modelStatus.toLowerCase()}>{modelStatus}</span>
        <span data-state={backendPillState}>{backendLabel}</span>
        <span>{modeLabel}</span>
        <span>{solverLabel}</span>
        <span>{buildingCount} STRUCTURES</span>
        <span>{windSpeed.toFixed(1)} M/S WIND</span>
        <span>{energyRate.toFixed(1)} U/S BURN</span>
      </div>
    </header>
  );
}
