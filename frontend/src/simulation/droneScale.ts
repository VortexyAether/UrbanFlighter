export const DRONE_SCALE_CONTRACT = Object.freeze({
  visualBody: Object.freeze({
    spanM: 0.58,
    heightM: 0.16,
    propellerDiameterM: 0.15,
  }),
  researchSafetyEnvelope: Object.freeze({
    fallbackRadiusM: 1.25,
    verticalClearanceM: 2,
    semantics: 'swept 1.25m horizontal clearance plus retained 2m vertical roof clearance against OSM building prisms and live field bounds',
  }),
} as const);

export function resolveDroneSafetyRadius(reportedRadius: number | null | undefined) {
  return Number.isFinite(reportedRadius) && (reportedRadius ?? 0) > 0
    ? Number(reportedRadius)
    : DRONE_SCALE_CONTRACT.researchSafetyEnvelope.fallbackRadiusM;
}
