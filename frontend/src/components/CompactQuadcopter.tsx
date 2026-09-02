import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { DRONE_SCALE_CONTRACT } from '../simulation/droneScale';

interface CompactQuadcopterProps {
  safetyRadiusM: number;
  verticalSafetyClearanceM: number;
  showSafetyEnvelope?: boolean;
  visualScale?: number;
}

const MOTOR_OFFSET = 0.145;
const MOTOR_POSITIONS = [
  [-MOTOR_OFFSET, -MOTOR_OFFSET],
  [MOTOR_OFFSET, -MOTOR_OFFSET],
  [-MOTOR_OFFSET, MOTOR_OFFSET],
  [MOTOR_OFFSET, MOTOR_OFFSET],
] as const;

function Rotors() {
  const refs = useRef<Array<THREE.Group | null>>([]);
  useFrame(({ clock }) => {
    const elapsed = clock.getElapsedTime();
    refs.current.forEach((rotor, index) => {
      if (rotor) rotor.rotation.y = elapsed * (index % 2 === 0 ? 58 : -58) + index * Math.PI / 4;
    });
  });

  return MOTOR_POSITIONS.map(([x, z], index) => (
    <group key={`${x}-${z}`} position={[x, 0.052, z]}>
      <mesh castShadow>
        <cylinderGeometry args={[0.032, 0.034, 0.045, 16]} />
        <meshStandardMaterial color="#20282d" metalness={0.72} roughness={0.3} />
      </mesh>
      <group ref={(node) => { refs.current[index] = node; }} position={[0, 0.028, 0]}>
        <mesh castShadow>
          <boxGeometry args={[DRONE_SCALE_CONTRACT.visualBody.propellerDiameterM, 0.006, 0.014]} />
          <meshStandardMaterial color="#161c20" metalness={0.32} roughness={0.42} />
        </mesh>
        <mesh rotation={[0, Math.PI / 2, 0]} castShadow>
          <boxGeometry args={[DRONE_SCALE_CONTRACT.visualBody.propellerDiameterM, 0.006, 0.014]} />
          <meshStandardMaterial color="#161c20" metalness={0.32} roughness={0.42} />
        </mesh>
      </group>
      <mesh position={[0, 0.032, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.072, 0.075, 28]} />
        <meshBasicMaterial color="#dce8eb" transparent opacity={0.28} side={THREE.DoubleSide} />
      </mesh>
    </group>
  ));
}

/** Metre-scale 0.58 m visual body; flight clearance is a separate research envelope. */
export default function CompactQuadcopter({
  safetyRadiusM,
  verticalSafetyClearanceM,
  showSafetyEnvelope = false,
  visualScale = 1,
}: CompactQuadcopterProps) {
  return (
    <group name="compact-0.58m-quadcopter" scale={visualScale}>
      <mesh rotation={[0, Math.PI / 4, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.41, 0.025, 0.03]} />
        <meshStandardMaterial color="#2f3b42" metalness={0.6} roughness={0.34} />
      </mesh>
      <mesh rotation={[0, -Math.PI / 4, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.41, 0.025, 0.03]} />
        <meshStandardMaterial color="#2f3b42" metalness={0.6} roughness={0.34} />
      </mesh>
      <mesh position={[0, 0.018, -0.012]} castShadow receiveShadow>
        <boxGeometry args={[0.16, 0.075, 0.22]} />
        <meshStandardMaterial color="#dfe5e5" metalness={0.48} roughness={0.3} />
      </mesh>
      <mesh position={[0, 0.022, -0.132]} rotation={[-Math.PI / 2, 0, 0]} castShadow>
        <coneGeometry args={[0.06, 0.095, 18]} />
        <meshStandardMaterial color="#f2f5f4" metalness={0.42} roughness={0.24} />
      </mesh>
      <mesh position={[0, 0.062, -0.072]}>
        <sphereGeometry args={[0.032, 14, 10]} />
        <meshStandardMaterial
          color="#85e9ff"
          emissive="#16bce5"
          emissiveIntensity={1.5}
          metalness={0.18}
          roughness={0.22}
        />
      </mesh>
      <mesh position={[-0.075, 0.042, 0.108]}>
        <sphereGeometry args={[0.012, 10, 8]} />
        <meshBasicMaterial color="#ff625c" toneMapped={false} />
      </mesh>
      <mesh position={[0.075, 0.042, 0.108]}>
        <sphereGeometry args={[0.012, 10, 8]} />
        <meshBasicMaterial color="#79ffb5" toneMapped={false} />
      </mesh>
      <pointLight position={[0, 0.13, -0.05]} color="#75eaff" intensity={0.42 + 0.18 * Math.max(0, visualScale - 1)} distance={3.2 * visualScale} decay={2} />
      <Rotors />
      {showSafetyEnvelope && (
        <mesh
          name="research-safety-envelope"
          renderOrder={4}
          scale={[1, verticalSafetyClearanceM / safetyRadiusM, 1]}
        >
          <sphereGeometry args={[safetyRadiusM, 24, 16]} />
          <meshBasicMaterial
            color="#ffad52"
            transparent
            opacity={0.12}
            wireframe
            depthWrite={false}
            toneMapped={false}
          />
        </mesh>
      )}
    </group>
  );
}
