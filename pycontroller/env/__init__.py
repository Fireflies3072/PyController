from .rocket_lander.rocket_lander import RocketLander

from gymnasium.envs.registration import register

def register_envs():
    register(
        id='coco_rocket_lander/RocketLander-v0',
        entry_point=RocketLander,
        max_episode_steps=None,
    )
