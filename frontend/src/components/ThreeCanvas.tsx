import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Sky } from '@react-three/drei';
import { useEffect, useRef, type ReactNode, type RefObject } from 'react';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import type { FlightCameraMode } from '../simulation/flight3dControls';

/** Stable identity — inline [x,y,z] arrays re-apply every parent render and reset the camera. */
const INITIAL_CAMERA_POSITION: [number, number, number] = [10, 38, 18];

interface ThreeCanvasProps {
    children: ReactNode;
    cameraMode: FlightCameraMode;
    orbitTarget: THREE.Vector3;
    presentationMode: boolean;
}

/**
 * Keep orbit target on the aircraft without React-prop thrashing.
 * Preserves the camera offset so drag/zoom stay under user control while the drone moves.
 */
function OrbitTargetFollower({
    target,
    enabled,
    controlsRef,
}: {
    target: THREE.Vector3;
    enabled: boolean;
    controlsRef: RefObject<OrbitControlsImpl | null>;
}) {
    const targetRef = useRef(target);
    targetRef.current = target;

    useFrame(() => {
        const controls = controlsRef.current;
        if (!enabled || !controls) return;

        const next = targetRef.current;
        const camera = controls.object;
        const offsetX = camera.position.x - controls.target.x;
        const offsetY = camera.position.y - controls.target.y;
        const offsetZ = camera.position.z - controls.target.z;

        controls.target.copy(next);
        camera.position.set(
            next.x + offsetX,
            next.y + offsetY,
            next.z + offsetZ,
        );
    });

    return null;
}

function FlightOrbitControls({
    cameraMode,
    orbitTarget,
}: {
    cameraMode: FlightCameraMode;
    orbitTarget: THREE.Vector3;
}) {
    const controlsRef = useRef<OrbitControlsImpl>(null);
    const { gl } = useThree();
    const orbitEnabled = cameraMode === 'orbit';

    // Ensure the WebGL canvas can receive drag/scroll after UI overlays steal focus.
    useEffect(() => {
        const element = gl.domElement;
        element.style.touchAction = 'none';
        element.tabIndex = 0;
    }, [gl]);

    return (
        <>
            <OrbitControls
                ref={controlsRef}
                makeDefault
                enabled={orbitEnabled}
                enableDamping
                dampingFactor={0.08}
                enablePan
                enableZoom
                enableRotate
                minDistance={2.5}
                maxDistance={520}
                maxPolarAngle={Math.PI / 2.04}
                zoomSpeed={1.05}
                rotateSpeed={0.85}
                panSpeed={0.75}
            />
            <OrbitTargetFollower
                target={orbitTarget}
                enabled={orbitEnabled}
                controlsRef={controlsRef}
            />
        </>
    );
}

export default function ThreeCanvas({
    children,
    cameraMode,
    orbitTarget,
    presentationMode,
}: ThreeCanvasProps) {
    const skyColor = presentationMode ? '#c9c6bf' : '#121110';
    return (
        <Canvas
            shadows
            dpr={[1, 1.5]}
            gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
            onCreated={({ gl }) => {
                gl.toneMapping = THREE.ACESFilmicToneMapping;
                gl.toneMappingExposure = presentationMode ? 0.92 : 0.82;
                gl.shadowMap.type = THREE.PCFSoftShadowMap;
                gl.domElement.style.touchAction = 'none';
            }}
            style={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                background: skyColor,
                touchAction: 'none',
            }}
        >
            <PerspectiveCamera
                makeDefault
                position={INITIAL_CAMERA_POSITION}
                fov={52}
                near={0.05}
                far={5_000}
            />

            {presentationMode ? (
                <Sky
                    distance={4_500}
                    sunPosition={[-120, 220, 80]}
                    inclination={0.48}
                    azimuth={0.22}
                    turbidity={2.8}
                    rayleigh={0.85}
                    mieCoefficient={0.003}
                    mieDirectionalG={0.9}
                />
            ) : (
                <color attach="background" args={[skyColor]} />
            )}
            <fog attach="fog" args={[presentationMode ? '#b7b4ad' : '#121110', 220, 1_350]} />

            <ambientLight intensity={presentationMode ? 0.28 : 0.32} />
            <hemisphereLight
                intensity={presentationMode ? 0.62 : 0.4}
                color={presentationMode ? '#efe8dc' : '#d8d4cc'}
                groundColor={presentationMode ? '#3a3936' : '#141312'}
            />
            <directionalLight
                position={[-120, 220, 80]}
                intensity={presentationMode ? 1.85 : 1.15}
                color={presentationMode ? '#f3efe6' : '#ece8e0'}
                castShadow
                shadow-mapSize={[2_048, 2_048]}
                shadow-camera-left={-460}
                shadow-camera-right={460}
                shadow-camera-top={460}
                shadow-camera-bottom={-460}
                shadow-camera-near={20}
                shadow-camera-far={850}
                shadow-bias={-0.00012}
                shadow-normalBias={0.035}
            />

            {children}

            <FlightOrbitControls cameraMode={cameraMode} orbitTarget={orbitTarget} />
        </Canvas>
    );
}
