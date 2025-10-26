import gymnasium as gym
import numpy as np
from pycontroller import DeePC_Controller

# --- Configuration ---
# DeePC Parameters
T_ini = 5
T_f = 5

# --- Environment Setup ---
env = gym.make("fireflies3072/RoomTemperature-v0")

# Get system dimensions from the environment
# u_size: dimension of action space (heater power)
# y_size: dimension of observation space (room temperature)
u_size = 1
y_size = 1

# DeePC controller
deepc = DeePC_Controller(
    u_size=u_size,
    y_size=y_size,
    T_ini=T_ini,
    T_f=T_f,
    Q=[1.0],
    R=[0.01],
    lambda_g=1e-2,
    lambda_y=1e5,
    min_output=env.action_space.low,
    max_output=env.action_space.high,
    hankel_columns=100,
    y_labels=['T_current']
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
        
        # Apply action and get the next state
        next_state, _, done, _, _ = env.step(action)
        
        # Add the (action, state) pair to the DeePC buffer
        deepc.collect_data_for_hankel(action, next_state)
        
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
env = gym.make("fireflies3072/RoomTemperature-v0")
state, _ = env.reset()
# Define the target state for the room temperature
# Target: 22.0 degC
target = np.array([22.0])

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
deepc.analyze(figure_filename='figure/rt_deepc_random.png')
