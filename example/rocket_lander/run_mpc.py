import gymnasium as gym
import numpy as np
from pycontroller.controller import MPC_RocketLander

args = {
    "initial_position": (0.5, 0.9, 0.4)
}

# --- Environment Setup ---
env = gym.make("coco_rocket_lander/RocketLander-v0", render_mode="rgb_array", args=args)
env = gym.wrappers.RecordVideo(env, 'video', episode_trigger=lambda x: True, name_prefix="rl_mpc")

# --- Configuration ---
# MPC Parameters
horizon = 10
sample_time = 0.1
Q = np.diag([3.0, 0.1, 2.0, 1.0, 120.0, 30.0])
R = np.diag([0.01, 0.01, 0.01])

# MPC controller
mpc = MPC_RocketLander(env, horizon, sample_time, Q, R)

# Define the target state for the rocket
# Target: land at the landing position with zero velocity and angle
landing_position = env.unwrapped.get_landing_position()
target = np.zeros(6, dtype=np.float64)
target[0] = landing_position[0]
target[1] = landing_position[1]

# Reset the environment
state, _ = env.reset(seed=0)

for i in range(2000):
    # Let MPC calculate the optimal action
    action = mpc.update(state[:6], target)
    # If the legs are in contact, set both main and side engine thrusts to 0
    if state[6] and state[7]:
        action[:] = 0
    
    # Apply the calculated action to the environment
    next_state, rewards, done, _, info = env.step(action)

    # Update observation
    state = next_state
    
    # Check if simulation ended
    if done:
        break

print("Control phase finished.")
env.close()
