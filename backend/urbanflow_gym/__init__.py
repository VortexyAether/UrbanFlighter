from .baselines import (
    BASELINE_ORDER,
    DIRECT_GOAL,
    SHORTEST_PATH,
    WIND_AWARE,
    make_baseline,
)
from .cfd_adapter import CFDFieldDataset, CFDWindProvider2DAdapter, StructuredCFDDataset
from .contract import urbanflow_contract_payload
from .env import UrbanFlowConfig, UrbanFlowEnv, relative_air_energy_step
from .evaluation import (
    evaluation_summary,
    run_baseline_evaluation,
    run_live_baseline_evaluation,
)
from .live_scenario import (
    LIVE_SCENARIO_SCHEMA_ID,
    LiveScenarioRegistry,
    build_live_scenario_record,
    live_scenario_registry,
    make_live_scenario,
)
from .scenario import DEFAULT_HELD_OUT_SEEDS, make_seeded_scenario
from .schemas import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CONTRACT_VERSION,
    ENVIRONMENT_ID,
    PRIVILEGED_CRITIC_DIM,
)
from .spaces import BoxSpec

__all__ = [
    "ACTION_DIM",
    "ACTOR_OBSERVATION_DIM",
    "BASELINE_ORDER",
    "BoxSpec",
    "CFDFieldDataset",
    "CFDWindProvider2DAdapter",
    "CONTRACT_VERSION",
    "DEFAULT_HELD_OUT_SEEDS",
    "DIRECT_GOAL",
    "ENVIRONMENT_ID",
    "GymnasiumUrbanFlowEnv",
    "LIVE_SCENARIO_SCHEMA_ID",
    "LiveScenarioRegistry",
    "PRIVILEGED_CRITIC_DIM",
    "SHORTEST_PATH",
    "StructuredCFDDataset",
    "UrbanFlowConfig",
    "UrbanFlowEnv",
    "WIND_AWARE",
    "evaluation_summary",
    "build_live_scenario_record",
    "live_scenario_registry",
    "make_baseline",
    "make_live_scenario",
    "make_seeded_scenario",
    "relative_air_energy_step",
    "run_baseline_evaluation",
    "run_live_baseline_evaluation",
    "urbanflow_contract_payload",
]


def __getattr__(name: str):
    """Keep optional Gymnasium entirely unimported during core/API use."""

    if name == "GymnasiumUrbanFlowEnv":
        from .gym_adapter import GymnasiumUrbanFlowEnv

        return GymnasiumUrbanFlowEnv
    raise AttributeError(name)
