import { Line } from '@react-three/drei';

interface FlightDomainBoundaryProps {
  bounds: {
    min_x: number;
    max_x: number;
    min_y: number;
    max_y: number;
  };
}

/** Visualizes the same rectangular live-field bounds enforced by 3D motion. */
export default function FlightDomainBoundary({ bounds }: FlightDomainBoundaryProps) {
  const y = 0.09;
  const points: [number, number, number][] = [
    [bounds.min_x, y, -bounds.min_y],
    [bounds.max_x, y, -bounds.min_y],
    [bounds.max_x, y, -bounds.max_y],
    [bounds.min_x, y, -bounds.max_y],
    [bounds.min_x, y, -bounds.min_y],
  ];
  return (
    <Line
      name="live-flow-domain-boundary"
      points={points}
      color="#8de5ef"
      lineWidth={1.2}
      transparent
      opacity={0.56}
      depthWrite={false}
    />
  );
}
