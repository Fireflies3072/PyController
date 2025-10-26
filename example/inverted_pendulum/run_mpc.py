import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from pycontroller.controller import MPC_Controller
from pycontroller.env.inverted_pendulum.system_model import InvertedPendulumSystemModel

# --- Environment Setup ---
env = gym.make("InvertedPendulum-v5", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, 'video', episode_trigger=lambda x: True, name_prefix="ip_mpc")

# --- Configuration ---
# MPC Parameters
horizon = 20
default_sample_time = 0.02
Q = np.diag([100, 100, 400, 60])
R = np.diag([0.05])

# MPC controller
model = InvertedPendulumSystemModel(env)
model.calculate_linear_system_matrices()
# Use environment dt if available; fall back to a small default
dt = model.get_env_dt() or default_sample_time
model.discretize_system_matrices(dt)
A, B = model.get_discrete_linear_system_matrices()
min_output = env.action_space.low
max_output = env.action_space.high
min_state = np.array([-1.0, -0.2, -np.inf, -np.inf], dtype=np.float64)
max_state = np.array([1.0, 0.2, np.inf, np.inf], dtype=np.float64)
mpc = MPC_Controller(A, B, horizon, Q, R, min_output, max_output, min_state, max_state)

# Define the target state for the inverted pendulum
# Target: keep the pendulum upright
target = np.zeros(4, dtype=np.float64)

# Reset the environment
state, _ = env.reset(seed=0)
print("Initial state:", state)

times = []
states = []
preds = []
actions = []

t = 0.0
times.append(t)
states.append(state)

for i in range(2000):
    # Use original env observation ordering with model aligned to env
    action = -mpc.update(state, target)
    if i == 0:
        print("MPC status:", mpc.last_status, "value:", mpc.last_problem_value, "feasible:", mpc.last_feasible)
        # Small sign test: try the opposite action for the first step to diagnose sign
        pred_next_pos = (A @ state + B @ action).astype(np.float64)
        pred_next_neg = (A @ state + B @ (-action)).astype(np.float64)
        print("Pred next (u):", pred_next_pos)
        print("Pred next (-u):", pred_next_neg)

    # One-step prediction with discrete model
    pred_next = (A @ state + B @ action).astype(np.float64)
    preds.append(pred_next)
    actions.append(action)

    # Apply the calculated action to the environment
    next_state, rewards, done, _, info = env.step(action)

    # Time/state roll
    t += dt
    times.append(t)
    states.append(next_state)
    state = next_state
    
    # Check if simulation ended
    if done:
        break

print("Control phase finished.")
env.close()

# ================= Plot prediction vs actual (one-step) =================
os.makedirs('figure', exist_ok=True)
states_arr = np.array(states)
preds_arr = np.array(preds)
time_arr = np.array(times)

# Align predictions (t+dt) with actual next states
t_pred = time_arr[1:]
x_actual = states_arr[1:]

labels = ['x (m)', 'theta (rad)', 'x_dot (m/s)', 'theta_dot (rad/s)']
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
for k in range(4):
    axes[k].plot(t_pred, x_actual[:, k], label='actual', color='C0')
    axes[k].plot(t_pred, preds_arr[:, k], '--', label='A@x + B@u (1-step)', color='C1')
    axes[k].set_ylabel(labels[k])
    if k == 1:
        axes[k].axhline(0.2, color='r', lw=1, ls=':')
        axes[k].axhline(-0.2, color='r', lw=1, ls=':')

axes[-1].set_xlabel('time (s)')
axes[0].legend(loc='upper right')
plt.tight_layout()
plt.savefig('figure/ip_mpc_prediction.png', dpi=150)
plt.close(fig)
