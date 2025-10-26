import gymnasium as gym
import numpy as np
from pycontroller import DeePC_Controller

# --- Configuration ---
# DeePC Parameters
T_ini = 1
T_f = 10

# --- Environment Setup ---
env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")

# Get system dimensions from the environment
# u_size: dimension of action space (main engine, side engine, nozzle angle)
# y_size: dimension of observation space (x,y pos, x,y vel, angle, ang. vel)
u_size = 2
y_size = 6

# DeePC controller
deepc = DeePC_Controller(
    u_size=u_size,
    y_size=y_size,
    T_ini=T_ini,
    T_f=T_f,
    Q=[10, 1, 1, 10, 1e3, 1],
    R=[0.01, 0.01],
    lambda_g=1e-2,
    lambda_y=1e5,
    min_output=env.action_space.low,
    max_output=env.action_space.high,
    hankel_columns=1000,
    y_labels=['x', 'y', 'vx', 'vy', 'angle', 'ang_vel']
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
        action = env.action_space.sample()
        if action[0] < 0:
            action[0] = -1
        if np.abs(action[1]) < 0.5:
            action[1] = 0
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

# Reset the environment
env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, 'video', episode_trigger=lambda x: True, name_prefix="ll_deepc_random")
state, _ = env.reset(seed=0)
# Define the target state for the rocket
# Target: land at the landing position with zero velocity and angle
target = np.zeros(y_size)

# Initialize the DeePC controller for a new episode
deepc.new_episode()

for i in range(2000):
    # Let DeePC calculate the optimal action
    action = deepc.update(state[:6], target)
    if action[0] < 0:
        action[0] = -1
    if np.abs(action[1]) < 0.5:
        action[1] = 0
    if state[6] and state[7]:
        action[:] = 0
    print(state[:6], action)
    
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
deepc.analyze(figure_filename='figure/deepc_random.png')
