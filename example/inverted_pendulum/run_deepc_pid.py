import gymnasium as gym
import numpy as np
from pycontroller.controller import DeePC_Controller, PID_InvertedPendulum

# --- Configuration ---
# DeePC Parameters
T_ini = 3
T_f = 5

# --- Environment Setup ---
env = gym.make("InvertedPendulum-v5", render_mode="rgb_array")

# Get system dimensions from the environment
# u_size: dimension of action space (main engine, side engine, nozzle angle)
# y_size: dimension of observation space (x,y pos, x,y vel, angle, ang. vel)
u_size = 1
y_size = 4

# PID controller
angle_pid_params = [0.1, 0.00, 0.01]
force_pid_params = [-4.0, 0.0, -1.0]
pid = PID_InvertedPendulum(angle_pid_params, force_pid_params, env.action_space.low, env.action_space.high)

# DeePC controller
deepc = DeePC_Controller(
    u_size=u_size,
    y_size=y_size,
    T_ini=T_ini,
    T_f=T_f,
    Q=[1, 100, 1, 1],
    R=[0.1],
    lambda_g=1e-2,
    lambda_y=1e5,
    min_output=env.action_space.low,
    max_output=env.action_space.high,
    hankel_columns=2000,
    y_labels=['x', 'theta', 'x_dot', 'theta_dot']
)

# ===================================================================
#  PHASE 1: DATA COLLECTION
# ===================================================================
print("--- Starting Phase 1: Data Collection ---")

while not deepc.enough_data:
    state, _ = env.reset()
    deepc.new_episode()
    while True:
        # Explore system dynamics by random action
        action = pid.update(state[:4], target=np.zeros(4)).reshape(-1)
        
        # Apply action and get the next state
        next_state, _, done, _, _ = env.step(action)
        
        # Add the (action, state) pair to the DeePC buffer
        deepc.collect_data_for_hankel(action, next_state[:4])
        
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

# Reset the environment
env = gym.make("InvertedPendulum-v5", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, 'video', episode_trigger=lambda x: True, name_prefix="ip_deepc_pid")
state, _ = env.reset(seed=0)
# Define the target state for the rocket
# Target: land at the landing position with zero velocity and angle
target = np.zeros(y_size)

# Initialize the DeePC controller for a new episode
deepc.new_episode()

for i in range(2000):
    # Let DeePC calculate the optimal action
    action = deepc.update(state, target)
    # Apply action and get the next state
    next_state, rewards, done, _, info = env.step(action)
    # Roll the history
    deepc.roll_history(action, next_state)

    # Update state
    state = next_state
    if done:
        break

print("Control phase finished.")
env.close()

# Analyze the DeePC results
deepc.analyze(figure_filename='figure/deepc_pid.png')
