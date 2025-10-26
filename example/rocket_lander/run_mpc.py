import gymnasium as gym
import numpy as np
from pycontroller import MPC_Controller
from pycontroller.env.rocket_lander.system_model import RocketLanderSystemModel

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
model = RocketLanderSystemModel(env)
model.calculate_linear_system_matrices()
model.discretize_system_matrices(sample_time)
A, B = model.get_discrete_linear_system_matrices()
min_output = env.action_space.low
max_output = env.action_space.high
min_state = np.array([0, 0, -np.inf, -np.inf, -env.unwrapped.cfg.theta_limit, -np.inf], dtype=np.float64)
max_state = np.array([env.unwrapped.cfg.width, env.unwrapped.cfg.height, np.inf, np.inf, env.unwrapped.cfg.theta_limit, np.inf], dtype=np.float64)
mpc = MPC_Controller(A, B, horizon, Q, R, min_output, max_output, min_state, max_state)

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
