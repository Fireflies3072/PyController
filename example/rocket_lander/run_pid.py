import gymnasium as gym
import numpy as np
from pycontroller import PID_RocketLander

args = {
}

# --- Environment Setup ---
env = gym.make("coco_rocket_lander/RocketLander-v0", render_mode="rgb_array", args=args)
env = gym.wrappers.RecordVideo(env, 'video', episode_trigger=lambda x: True, name_prefix="rl_pid")

# --- Configuration ---
# PID Parameters
engine_pid_params = [10, 0, 10]
side_engine_pid_params = [5, 0, 6]
engine_vector_pid_params = [0.085, 0.001, 10.55]

# PID controller
pid = PID_RocketLander(engine_pid_params, side_engine_pid_params, engine_vector_pid_params,
                        min_output=env.action_space.low, max_output=env.action_space.high)

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
    action = pid.update(state[:6], target)
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
