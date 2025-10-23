# PyController

Controllers (PID, DeePC, etc.) for different scenarios (RocketLander, InvertedPendulum, etc.) in Python. This library provides a framework for implementing and comparing various control algorithms on simulated environments.

## Installation

If you have cloned this repository, you can install the package and its dependencies using `pip`.

From the root of the `PyController` directory, run:
```bash
pip install -e .
```
This will install the package in editable mode and also install the required dependencies: `matplotlib`, `scipy`, `cvxpy`, and `gymnasium[box2d,mojoco]`.

This library requires Python 3.10 or higher.

## Environments

This library uses `gymnasium` for its environments. The available environments are automatically registered upon importing `pycontroller`.

### Rocket Lander

This is a Box2D Gymnasium environment which simulates a Falcon 9 ocean barge landing.

To create an instance of the environment:

```python
import gymnasium as gym
import pycontroller # Imports and registers the environment

env = gym.make("coco_rocket_lander/RocketLander-v0")
```

You can find example usage in the `example/rocket_lander/` directory.

## Controllers

This library provides several controller implementations. The base controller class defines a common interface with `reset()` and `update(state, target)` methods.

### PID Controller

A Proportional-Integral-Derivative (PID) controller is available. For the Rocket Lander environment, a specialized `PID_RocketLander` controller is provided which wraps three PID controllers for the main engine, side engines, and nozzle vectoring.

Example usage:
```python
import gymnasium as gym
import numpy as np
from pycontroller import PID_RocketLander

env = gym.make("coco_rocket_lander/RocketLander-v0")
state, _ = env.reset()

# PID Parameters
engine_pid_params = [10, 0, 10]
side_engine_pid_params = [5, 0, 6]
engine_vector_pid_params = [0.085, 0.001, 10.55]

# PID controller
pid = PID_RocketLander(engine_pid_params, side_engine_pid_params, engine_vector_pid_params,
                        min_output=env.action_space.low, max_output=env.action_space.high)

# Define the target state
landing_position = env.unwrapped.get_landing_position()
target = np.zeros(6, dtype=np.float64)
target[0] = landing_position[0]
target[1] = landing_position[1]

# Control loop
for i in range(2000):
    action = pid.update(state[:6], target)
    state, rewards, done, _, info = env.step(action)
    if done:
        break
env.close()
```

### Data-Enabled Predictive Control (DeePC)

A `DeePC_Controller` is implemented. This controller uses data collected from the system to perform predictive control without requiring an explicit system model.

The usage involves two phases:
1.  **Data Collection**: Collect input/output data from the system to build a Hankel matrix.
2.  **Control**: Use the Hankel matrix to predict future behavior and compute optimal control actions.

Check `example/rocket_lander/run_deepc_random.py` and `example/rocket_lander/run_deepc_pid.py` for detailed examples.

### Model Predictive Control (MPC)

A Model Predictive Control (MPC) implementation is available for the Rocket Lander scenario (`MPC_RocketLander`). It uses a system model to predict future states and optimize control inputs. You can find an example in `example/rocket_lander/run_mpc.py`.

## Future Updates

We are planning to expand the library with more environments and examples:

*   **Room Temperature Control**: A new environment to simulate and control the temperature in a room.
*   **Inverted Pendulum**: Add examples for the classic inverted pendulum control problem.

## Appreciation

The Rocket Lander environment is a Box2D Gymnasium environment which simulates a Falcon 9 ocean barge landing.

Modified by Dylan Vogel and Gerasimos Maltezos for the 2023 Computation Control course at ETH Zurich.
Original environment created by Reuben Ferrante (https://github.com/arex18/rocket-lander).
