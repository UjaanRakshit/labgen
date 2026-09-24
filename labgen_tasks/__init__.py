"""Isaac Lab tasks built on labgen scenes. Imports Isaac Lab; labgen core must not.

Importing this package registers the gym ids, the way isaaclab_tasks does:

    import labgen_tasks                       # registers the ids below
    env = gym.make("Isaac-LabBench-YAM-IK-Rel-v0", cfg=parse_env_cfg(...))
"""

import gymnasium as gym

gym.register(
    id="Isaac-LabBench-YAM-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.bench_yam:BenchYamIkRelEnvCfg"},
    disable_env_checker=True,
)
