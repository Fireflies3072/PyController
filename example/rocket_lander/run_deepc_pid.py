import gymnasium as gym
import numpy as np
from pycontroller.controller import PID_RocketLander
from pycontroller import DeePC_Controller

args = {
    "initial_position": (0.5, 0.8, 0.0)
}

# --- Configuration ---
# DeePC Parameters
T_ini = 1
T_f = 10

# --- Environment Setup ---
env = gym.make("coco_rocket_lander/RocketLander-v0", render_mode="rgb_array", args=args)
env = gym.wrappers.RecordVideo(env, 'video', episode_trigger=lambda x: True, name_prefix="rl_deepc_pid")

# Get system dimensions from the environment
# u_size: dimension of action space (main engine, side engine, nozzle angle)
# y_size: dimension of observation space (x,y pos, x,y vel, angle, ang. vel)
u_size = 3
y_size = 6

# PID controller
engine_pid_params = [10, 0, 10]
side_engine_pid_params = [5, 0, 6]
engine_vector_pid_params = [0.085, 0.001, 10.55]
pid = PID_RocketLander(engine_pid_params, side_engine_pid_params, engine_vector_pid_params,
                        env.action_space.low, env.action_space.high)

# DeePC controller
deepc = DeePC_Controller(
    u_size=u_size,
    y_size=y_size,
    T_ini=T_ini,
    T_f=T_f,
    Q=[10, 10, 1, 1, 1e3, 1],
    R=[1.5, 0.01, 0.01],
    lambda_g=1e-2,
    lambda_y=1e5,
    min_output=env.action_space.low,
    max_output=env.action_space.high,
    hankel_columns=None,
    y_labels=['x', 'y', 'vx', 'vy', 'angle', 'ang_vel']
)

# ===================================================================
#  PHASE 1: DATA COLLECTION
# ===================================================================
print("--- Starting Phase 1: Data Collection ---")
state, _ = env.reset(seed=0)
hover_center = state[:2]
hover_center[1] -= 5
deepc.new_episode()

while not deepc.enough_data:
    # Generate a hovering action by PID controller to explore the system dynamics
    target = hover_center + np.random.randn(2) * 2
    action = pid.update(state[:6], target)
    if state[6] and state[7]:
        action[:] = 0
    
    # Apply action and get the next state
    next_state, _, done, _, _ = env.step(action)
    
    # Add the (action, state) pair to the DeePC buffer
    deepc.collect_data_for_hankel(action, next_state[:6])
    
    # Update state and check if the episode ended
    state = next_state
    if done or deepc.enough_data:
        break

# Build the Hankel matrix from the collected data
deepc.build_hankel_matrix()
print(f"Successfully built DeePC Hankel matrix with {deepc.U_p.shape[1]} columns.")


# ===================================================================
#  PHASE 2: PREDICTIVE CONTROL
# ===================================================================
print("\n--- Starting Phase 2: DeePC Control ---")

# Define the target state for the rocket
# Target: land at the landing position with zero velocity and angle
landing_position = env.unwrapped.get_landing_position()
target = np.zeros(y_size)
target[0] = landing_position[0]
target[1] = landing_position[1]

# Initialize the DeePC controller for a new episode
deepc.new_episode()

for i in range(2000):
    # Let DeePC calculate the optimal action
    action = deepc.update(state[:6], target)
    # If the legs are in contact, set both main and side engine thrusts to 0
    if state[6] and state[7]:
        action[:] = 0
    
    # Apply action and get the next state
    next_state, rewards, done, _, info = env.step(action)

    # Roll the history
    deepc.roll_history(action, next_state[:6])

    # Update state
    state = next_state
    if done:
        break

print("Control phase finished.")
env.close()

# Analyze the DeePC results
deepc.analyze(figure_filename='figure/rl_deepc_pid.png')
