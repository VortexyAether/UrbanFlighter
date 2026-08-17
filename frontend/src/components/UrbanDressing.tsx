import { memo, useLayoutEffect, useRef, type RefObject } from 'react';
import * as THREE from 'three';
import type {
  RoadMarkingProp,
  FacadePanelProp,
  RooftopUnitProp,
  StreetlightProp,
  TreeProp,
  UrbanDressingLayout,
} from '../presentation/urbanDressing';

interface UrbanDressingProps {
  layout: UrbanDressingLayout;
}

interface InstanceTransform {
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];
  color?: string;
}

function useInstanceTransforms(
  ref: RefObject<THREE.InstancedMesh | null>,
  transforms: readonly InstanceTransform[],
) {
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const object = new THREE.Object3D();
    const color = new THREE.Color();
    transforms.forEach((transform, index) => {
      object.position.set(...transform.position);
      object.rotation.set(...transform.rotation);
      object.scale.set(...transform.scale);
      object.updateMatrix();
      mesh.setMatrixAt(index, object.matrix);
      if (transform.color) mesh.setColorAt(index, color.set(transform.color));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingBox();
    mesh.computeBoundingSphere();
  }, [ref, transforms]);
}

function treeTransforms(trees: readonly TreeProp[]) {
  return {
    trunks: trees.map((tree) => ({
      position: [tree.x, tree.height * 0.28, -tree.y] as [number, number, number],
      rotation: [0, tree.rotation, 0] as [number, number, number],
      scale: [0.13 + tree.canopyRadius * 0.055, tree.height * 0.56, 0.13 + tree.canopyRadius * 0.055] as [number, number, number],
      color: tree.tone > 0.55 ? '#805f43' : '#6e5037',
    })),
    lowerCanopies: trees.map((tree) => ({
      position: [tree.x, tree.height * 0.68, -tree.y] as [number, number, number],
      rotation: [0, tree.rotation, 0] as [number, number, number],
      scale: [tree.canopyRadius, tree.height * 0.24, tree.canopyRadius] as [number, number, number],
      color: tree.tone > 0.66 ? '#679068' : tree.tone > 0.32 ? '#59825e' : '#4d7655',
    })),
    upperCanopies: trees.map((tree) => ({
      position: [tree.x, tree.height * 0.87, -tree.y] as [number, number, number],
      rotation: [0, -tree.rotation * 0.7, 0] as [number, number, number],
      scale: [tree.canopyRadius * 0.76, tree.height * 0.2, tree.canopyRadius * 0.76] as [number, number, number],
      color: tree.tone > 0.5 ? '#79a273' : '#639365',
    })),
  };
}

function lightTransforms(streetlights: readonly StreetlightProp[]) {
  return {
    poles: streetlights.map((light) => ({
      position: [light.x, light.height / 2, -light.y] as [number, number, number],
      rotation: [0, -light.rotation, 0] as [number, number, number],
      scale: [0.055, light.height, 0.055] as [number, number, number],
      color: '#343c3e',
    })),
    heads: streetlights.map((light) => ({
      position: [
        light.x + Math.cos(light.rotation + Math.PI / 2) * 0.32,
        light.height,
        -(light.y + Math.sin(light.rotation + Math.PI / 2) * 0.32),
      ] as [number, number, number],
      rotation: [0, -light.rotation, 0] as [number, number, number],
      scale: [0.52, 0.09, 0.18] as [number, number, number],
      color: '#d8ded9',
    })),
  };
}

function markingTransforms(markings: readonly RoadMarkingProp[]) {
  return markings.map((marking) => ({
    position: [marking.x, 0.045, -marking.y] as [number, number, number],
    rotation: [0, -marking.rotation, 0] as [number, number, number],
    scale: [marking.length, 0.018, marking.width] as [number, number, number],
    color: '#e9e2c7',
  }));
}

function rooftopTransforms(units: readonly RooftopUnitProp[]) {
  return units.map((unit) => ({
    position: [unit.x, unit.roofHeight + unit.height / 2 + 0.025, -unit.y] as [number, number, number],
    rotation: [0, -unit.rotation, 0] as [number, number, number],
    scale: [unit.width, unit.height, unit.depth] as [number, number, number],
    color: unit.tone > 0.55 ? '#889194' : '#6f797c',
  }));
}

function facadePanelTransforms(panels: readonly FacadePanelProp[]) {
  return panels.map((panel) => ({
    position: [panel.x, panel.elevation, -panel.y] as [number, number, number],
    rotation: [0, panel.rotation, 0] as [number, number, number],
    scale: [panel.width, panel.height, 0.055] as [number, number, number],
    color: panel.tone > 0.72
      ? '#d5b780'
      : panel.tone > 0.38
        ? '#86a7b2'
        : '#506b75',
  }));
}

function TreeInstances({ trees }: { trees: readonly TreeProp[] }) {
  const trunkRef = useRef<THREE.InstancedMesh>(null);
  const lowerRef = useRef<THREE.InstancedMesh>(null);
  const upperRef = useRef<THREE.InstancedMesh>(null);
  const transforms = treeTransforms(trees);
  useInstanceTransforms(trunkRef, transforms.trunks);
  useInstanceTransforms(lowerRef, transforms.lowerCanopies);
  useInstanceTransforms(upperRef, transforms.upperCanopies);
  if (trees.length === 0) return null;

  return (
    <group name="decorative-tree-instances">
      <instancedMesh ref={trunkRef} args={[undefined, undefined, trees.length]} castShadow receiveShadow>
        <cylinderGeometry args={[0.5, 0.72, 1, 7]} />
        <meshStandardMaterial vertexColors roughness={0.94} metalness={0} />
      </instancedMesh>
      <instancedMesh ref={lowerRef} args={[undefined, undefined, trees.length]} castShadow receiveShadow>
        <icosahedronGeometry args={[1, 1]} />
        <meshStandardMaterial vertexColors roughness={0.91} metalness={0} emissive="#203b25" emissiveIntensity={0.12} />
      </instancedMesh>
      <instancedMesh ref={upperRef} args={[undefined, undefined, trees.length]} castShadow receiveShadow>
        <icosahedronGeometry args={[1, 1]} />
        <meshStandardMaterial vertexColors roughness={0.88} metalness={0} emissive="#26472b" emissiveIntensity={0.12} />
      </instancedMesh>
    </group>
  );
}

function StreetlightInstances({ streetlights }: { streetlights: readonly StreetlightProp[] }) {
  const poleRef = useRef<THREE.InstancedMesh>(null);
  const headRef = useRef<THREE.InstancedMesh>(null);
  const transforms = lightTransforms(streetlights);
  useInstanceTransforms(poleRef, transforms.poles);
  useInstanceTransforms(headRef, transforms.heads);
  if (streetlights.length === 0) return null;

  return (
    <group name="decorative-streetlight-instances">
      <instancedMesh ref={poleRef} args={[undefined, undefined, streetlights.length]} castShadow>
        <cylinderGeometry args={[1, 1, 1, 8]} />
        <meshStandardMaterial vertexColors color="#343c3e" roughness={0.55} metalness={0.72} />
      </instancedMesh>
      <instancedMesh ref={headRef} args={[undefined, undefined, streetlights.length]} castShadow>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          vertexColors
          color="#d8ded9"
          roughness={0.32}
          metalness={0.38}
          emissive="#fff2c5"
          emissiveIntensity={0.18}
        />
      </instancedMesh>
    </group>
  );
}

function RoadMarkingInstances({ markings }: { markings: readonly RoadMarkingProp[] }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const transforms = markingTransforms(markings);
  useInstanceTransforms(ref, transforms);
  if (markings.length === 0) return null;
  return (
    <instancedMesh
      ref={ref}
      name="decorative-road-marking-instances"
      args={[undefined, undefined, markings.length]}
      receiveShadow
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial
        vertexColors
        color="#e9e2c7"
        roughness={0.82}
        metalness={0.02}
        emissive="#756f5a"
        emissiveIntensity={0.16}
      />
    </instancedMesh>
  );
}

function RooftopUnitInstances({ units }: { units: readonly RooftopUnitProp[] }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const transforms = rooftopTransforms(units);
  useInstanceTransforms(ref, transforms);
  if (units.length === 0) return null;
  return (
    <instancedMesh
      ref={ref}
      name="decorative-rooftop-unit-instances"
      args={[undefined, undefined, units.length]}
      castShadow
      receiveShadow
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial vertexColors roughness={0.58} metalness={0.48} />
    </instancedMesh>
  );
}

function FacadePanelInstances({ panels }: { panels: readonly FacadePanelProp[] }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const transforms = facadePanelTransforms(panels);
  useInstanceTransforms(ref, transforms);
  if (panels.length === 0) return null;
  return (
    <instancedMesh
      ref={ref}
      name="decorative-facade-panel-instances"
      args={[undefined, undefined, panels.length]}
      castShadow
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial
        vertexColors
        roughness={0.24}
        metalness={0.42}
        emissive="#6f8990"
        emissiveIntensity={0.1}
      />
    </instancedMesh>
  );
}

/**
 * Rendering-only scene dressing. None of these instances are supplied to the
 * building collision mesh list, LiDAR scanner, flow sampler, or Gym bridge.
 */
function UrbanDressing({ layout }: UrbanDressingProps) {
  return (
    <group name="presentation-only-urban-dressing" userData={{ mechanics: layout.contract, seed: layout.seed }}>
      {layout.roadBands.map((band, index) => (
        <mesh
          key={`road-${index}`}
          name="decorative-road-surface"
          position={[band.x, 0.018, -band.y]}
          rotation={[0, -band.rotation, 0]}
          receiveShadow
        >
          <boxGeometry args={[band.length, 0.025, band.width]} />
          <meshStandardMaterial
            color={band.tone > 0.5 ? '#31383a' : '#2c3335'}
            roughness={0.98}
            metalness={0.01}
          />
        </mesh>
      ))}
      <RoadMarkingInstances markings={layout.roadMarkings} />
      <TreeInstances trees={layout.trees} />
      <StreetlightInstances streetlights={layout.streetlights} />
      <RooftopUnitInstances units={layout.rooftopUnits} />
      <FacadePanelInstances panels={layout.facadePanels} />
    </group>
  );
}

export default memo(UrbanDressing);
