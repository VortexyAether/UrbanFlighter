import type { FlowField2DResponse } from '../api';
import type { ViewMode } from '../appModel';

interface HudLegendProps {
  viewMode: ViewMode;
  flow: FlowField2DResponse | null;
}

export default function HudLegend({ viewMode, flow }: HudLegendProps) {
  return (
    <div className="legend">
      {viewMode === '2d' ? (
        <>
          <p>Controls: WASD moves directly in screen directions; the map stays north-up.</p>
          <p>The rolling 2D sensor map contains only LiDAR returns and simulator display odometry, with no world polygons.</p>
        </>
      ) : (
        <>
          <p>Controls: W/S accelerate or reverse, A/D yaw, arrows or E/Q climb and descend.</p>
          <p>The rolling 3D sensor map uses simulator display odometry and has no loop closure or pose-graph correction.</p>
        </>
      )}
      <p>Geometry {flow?.domain.geometry_radius_m.toFixed(0) ?? '--'}m, solver {flow?.domain.solve_radius_m.toFixed(0) ?? '--'}m.</p>
      <p>Solver: {flow?.weather.wind_speed.toFixed(1) ?? '--'} m/s from {flow?.weather.wind_deg.toFixed(0) ?? '--'}°</p>
    </div>
  );
}
