export const INSPECTOR_PLAY_RATES = [1, 2, 4, 8, 16, 32, 64] as const;
export type InspectorPlayRate = (typeof INSPECTOR_PLAY_RATES)[number];

export const INSPECTOR_PLAYBACK_TICK_MS = 250;
export const INSPECTOR_MAX_BATCH_STEPS = 64;

const TICKS_PER_SECOND = 1_000 / INSPECTOR_PLAYBACK_TICK_MS;
const MAX_CREDIT_QUANTA = INSPECTOR_MAX_BATCH_STEPS * TICKS_PER_SECOND;

export interface InspectorPlaybackTick {
  creditQuanta: number;
  repeat: number;
}

export function isInspectorPlayRate(value: number): value is InspectorPlayRate {
  return INSPECTOR_PLAY_RATES.some((rate) => rate === value);
}

/**
 * Accumulates fixed 250 ms quanta and reserves at most one bounded request.
 * Keeping credit in integer quarter-steps makes the 1x/2x schedules exact,
 * while the in-flight guard lets the UI account for a slow response without
 * ever issuing an overlapping request.
 */
export function scheduleInspectorPlaybackTick(
  rate: InspectorPlayRate,
  creditQuanta: number,
  requestInFlight: boolean,
): InspectorPlaybackTick {
  if (!isInspectorPlayRate(rate)) {
    throw new Error('Inspector playback rate must be one of 1, 2, 4, 8, 16, 32, or 64.');
  }
  if (!Number.isInteger(creditQuanta) || creditQuanta < 0) {
    throw new Error('Inspector playback credit must be a non-negative integer.');
  }

  const nextCreditQuanta = Math.min(creditQuanta + rate, MAX_CREDIT_QUANTA);
  if (requestInFlight) {
    return { creditQuanta: nextCreditQuanta, repeat: 0 };
  }

  const repeat = Math.min(
    Math.floor(nextCreditQuanta / TICKS_PER_SECOND),
    INSPECTOR_MAX_BATCH_STEPS,
  );
  return {
    creditQuanta: nextCreditQuanta - repeat * TICKS_PER_SECOND,
    repeat,
  };
}

export function inspectorStepResponseIsCurrent(
  requestedSessionId: string,
  currentSessionId: string | null,
  requestedGeneration: number,
  currentGeneration: number,
  requestOwnsSlot: boolean,
  requestAborted: boolean,
) {
  return !requestAborted
    && requestOwnsSlot
    && requestedGeneration === currentGeneration
    && requestedSessionId === currentSessionId;
}
