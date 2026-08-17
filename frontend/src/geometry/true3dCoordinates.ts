export type True3DTriplet = readonly [x: number, y: number, z: number];

/**
 * True 3D solver data uses local XY on the ground and Z for altitude. The
 * Three.js scene is Y-up, and building extrusion maps source +Y to scene -Z.
 * Keeping this transform explicit prevents wind paths from being mirrored
 * across the scene Z axis relative to the city geometry.
 */
export function mapTrue3DPointToScene([x, y, altitude]: True3DTriplet): True3DTriplet {
  return [x, altitude, -y];
}

/** Apply the source-to-scene basis change to any direction or vector. */
export function mapTrue3DVectorToScene([x, y, z]: True3DTriplet): True3DTriplet {
  return [x, z, -y];
}

/** Map solver velocity components [u, v, w] onto scene [X, Y, Z]. */
export function mapTrue3DVelocityToScene([u, v, w]: True3DTriplet): True3DTriplet {
  return mapTrue3DVectorToScene([u, v, w]);
}

export function isFiniteTrue3DTriplet(value: unknown): value is True3DTriplet {
  return Array.isArray(value)
    && value.length === 3
    && value.every((component) => typeof component === 'number' && Number.isFinite(component));
}
