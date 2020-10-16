from gym.envs.registration import register

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

register(
    id='Continuous_OSG_TowerArc-v0',
    entry_point='custom_gym.envs:Continuous_OSG_TowerArcEnv',
    max_episode_steps=200
)