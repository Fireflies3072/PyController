from .rocket_lander.rocket_lander import RocketLander
from .room_temperature.room_temperature import RoomTemperature

from gymnasium.envs.registration import register

def register_envs():
    register(
        id='coco_rocket_lander/RocketLander-v0',
        entry_point=RocketLander
    )
    register(
        id='fireflies3072/RoomTemperature-v0',
        entry_point=RoomTemperature
    )
