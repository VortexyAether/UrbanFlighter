from __future__ import annotations

import numpy as np

from .contract import TRAINING_EXTRAS_INSTALL_COMMAND
from .env import UrbanFlowConfig, UrbanFlowEnv


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


if gym is not None and spaces is not None:

    class GymnasiumUrbanFlowEnv(gym.Env):
        metadata = UrbanFlowEnv.metadata

        def __init__(
            self,
            config: UrbanFlowConfig | None = None,
            *,
            scenario_factory=None,
            fixed_scenario=None,
        ) -> None:
            super().__init__()
            kwargs = {}
            if scenario_factory is not None:
                kwargs["scenario_factory"] = scenario_factory
            if fixed_scenario is not None:
                kwargs["fixed_scenario"] = fixed_scenario
            self.core = UrbanFlowEnv(config=config, **kwargs)
            self.observation_space = spaces.Box(
                low=self.core.observation_space.low,
                high=self.core.observation_space.high,
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=self.core.action_space.low,
                high=self.core.action_space.high,
                dtype=np.float32,
            )

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            core_seed = seed
            if core_seed is None:
                core_seed = int(self.np_random.integers(0, 2_147_483_648))
            return self.core.reset(seed=core_seed, options=options)

        def step(self, action):
            return self.core.step(action)

        def close(self) -> None:
            return None

else:

    class GymnasiumUrbanFlowEnv:  # type: ignore[no-redef]
        def __init__(self, config: UrbanFlowConfig | None = None, **kwargs) -> None:
            del config, kwargs
            raise ModuleNotFoundError(
                "Gymnasium is not installed. Install UrbanFlow training extras with: "
                f"{TRAINING_EXTRAS_INSTALL_COMMAND}"
            )
