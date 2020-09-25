from gym.envs.registration import registry, register, make, spec

register(
    id='TowerArc-v0',
    entry_point='custom_gym.envs:TowerArcEnv',
    max_episode_steps=200
)

register(
    id='OSG_TowerArc-v0',
    entry_point='custom_gym.envs:OSG_TowerArcEnv',
    max_episode_steps=200
)