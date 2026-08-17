import { Line } from '@react-three/drei';
import type { FreeFlightBeacon as FreeFlightBeaconData } from '../presentation/freeFlightBeacon';

interface FreeFlightBeaconProps {
  beacon: FreeFlightBeaconData;
}

export default function FreeFlightBeacon({ beacon }: FreeFlightBeaconProps) {
  const sceneZ = -beacon.y;
  return (
    <group
      name="display-only-free-flight-beacon"
      position={[beacon.x, beacon.altitude, sceneZ]}
      rotation={[0, beacon.rotation, 0]}
      userData={{ mechanics: beacon.contract }}
    >
      <mesh renderOrder={5}>
        <torusGeometry args={[3.2, 0.13, 10, 52]} />
        <meshBasicMaterial color="#ffb55a" transparent opacity={0.86} depthWrite={false} toneMapped={false} />
      </mesh>
      <mesh scale={0.72} renderOrder={5}>
        <torusGeometry args={[3.2, 0.055, 8, 52]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.72} depthWrite={false} toneMapped={false} />
      </mesh>
      <Line
        points={[[0, -beacon.altitude + 0.2, 0], [0, -3.5, 0]]}
        color="#ffb55a"
        lineWidth={0.7}
        transparent
        opacity={0.35}
        depthWrite={false}
      />
    </group>
  );
}
