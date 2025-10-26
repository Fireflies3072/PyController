import gymnasium as gym
import matplotlib.pyplot as plt
from pycontroller import PID_Controller

# env = RoomTemperature()
env = gym.make("fireflies3072/RoomTemperature-v0")

# PID controller
pid = PID_Controller(kp=2.0, ki=0.1, kd=0.1,
                     min_output=env.action_space.low, max_output=env.action_space.high)

T_current_history = []
T_out_history = []
T_target_history = []
action_history = []

state, info = env.reset()

for t in range(500):
    action = pid.update(state, target=info['T_target']).reshape(-1)
    next_state, _, done, _, info = env.step(action)

    T_current_history.append(next_state)
    T_out_history.append(info['T_out'])
    T_target_history.append(info['T_target'])
    action_history.append(action)

    state = next_state
    if done:
        break

env.close()

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(T_current_history, label='T_current')
plt.plot(T_out_history, '-.', label='T_out')
plt.plot(T_target_history, ':', label='T_target')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(action_history, label='action')
plt.legend()
plt.show()
