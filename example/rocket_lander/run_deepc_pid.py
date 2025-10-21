import gymnasium as gym
import numpy as np
from pycontroller import PID_RocketLander, DeePC_Controller, DeePC_Analyzer

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
    max_output=env.action_space.high
)

# DeePC analyzer
analyzer = DeePC_Analyzer(labels=['x', 'y', 'vx', 'vy', 'angle', 'ang_vel'])

# ===================================================================
#  PHASE 1: DATA COLLECTION
# ===================================================================
print("--- Starting Phase 1: Data Collection ---")
state, _ = env.reset(seed=0)
hover_center = state[:2]
hover_center[1] -= 5

for i in range(200):
    # Generate a hovering action by PID controller to explore the system dynamics
    target = hover_center + np.random.randn(2) * 2
    action = pid.update(state[:6], target)
    
    # Apply the action and get the new observation
    next_state, _, done, _, _ = env.step(action)
    
    # Add the (action, observation) pair to the DeePC buffer
    deepc.add_sample(action, next_state[:6])
    
    # Update observation for the next step
    state = next_state
    
    # Reset if the episode ends during data collection
    if done:
        break

# Build the Hankel matrices from the collected data
deepc.build_hankel_from_buffer()
print("DeePC Hankel matrices built successfully.")
print(f"Collected {deepc.U_p.shape[1]} samples")


# ===================================================================
#  PHASE 2: PREDICTIVE CONTROL
# ===================================================================
print("\n--- Starting Phase 2: DeePC Control ---")

# Reset the environment and the controller's history for a fresh start
deepc.reset(initial_y=state[:6])
# Define the reference (target) state for the rocket
# Target: land at the landing position with zero velocity and angle
landing_position = env.unwrapped.get_landing_position()
reference = np.zeros(y_size)
reference[0] = landing_position[0]
reference[1] = landing_position[1]
hover_state = state[:6]

for i in range(2000):
    # Let DeePC calculate the optimal action
    action, y_next, g, sigma_y = deepc._update_complete(state[:6], reference)
    # If the legs are in contact, set both main and side engine thrusts to 0
    if state[6] and state[7]:
        action[:] = 0
    
    # Apply the calculated action to the environment
    next_state, rewards, done, _, info = env.step(action)

    # Add the (action, measurement, target) pair to the DeePC analyzer
    analyzer.add_sample(action, y_next, state[:6], reference, g, sigma_y)

    # Update observation
    state = next_state
    
    # Check if simulation ended
    if done:
        break

print("Control phase finished.")
env.close()

# Analyze the DeePC results
analyzer.analyze()
