import gymnasium as gym
import numpy as np
from pycontroller.controller import PID_InvertedPendulum


env = gym.make("InvertedPendulum-v5", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, "video", episode_trigger=lambda x: True, name_prefix="ip_pid")

angle_pid_params = [0.1, 0.00, 0.01]
force_pid_params = [-4.0, 0.0, -1.0]
pid = PID_InvertedPendulum(angle_pid_params, force_pid_params, env.action_space.low, env.action_space.high)

state, _ = env.reset(seed=0)
for t in range(1000):
    action = pid.update(state[:4], target=np.zeros(4)).reshape(-1)
    next_state, _, done, _, _ = env.step(action)
    state = next_state
    if done:
        break

env.close()
