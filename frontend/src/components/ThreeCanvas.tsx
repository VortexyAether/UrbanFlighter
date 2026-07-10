import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import type { ReactNode } from 'react';

interface ThreeCanvasProps {
    children: ReactNode;
}

export default function ThreeCanvas({ children }: ThreeCanvasProps) {
    return (
        <Canvas
            shadows
            gl={{ antialias: true, alpha: false }}
            style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                background: '#111316'
            }}
        >
            <PerspectiveCamera makeDefault position={[0, 112, 215]} fov={50} />

            {/* Atmosphere */}
            <color attach="background" args={['#111316']} />
            <fog attach="fog" args={['#111316', 140, 1800]} />

            {/* Lighting */}
            <ambientLight intensity={0.52} />
            <pointLight position={[140, 120, 140]} intensity={0.65} castShadow />
            <directionalLight
                position={[-100, 200, 100]}
                intensity={1.2}
                castShadow
                shadow-mapSize={[2048, 2048]}
            />

            <hemisphereLight intensity={0.48} color="#f7f8f5" groundColor="#111316" />

            {children}

            <OrbitControls
                enablePan={true}
                enableZoom={true}
                enableRotate={true}
                maxPolarAngle={Math.PI / 2.1}
            />
        </Canvas>
    );
}
