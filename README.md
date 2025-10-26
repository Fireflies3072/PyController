# PyController

Controllers (PID, DeePC, etc.) for different scenarios (RocketLander, InvertedPendulum, etc.) in Python. This library provides a framework for implementing and comparing various control algorithms on simulated environments.

## Installation

You can install the package directly from GitHub using `pip`:

```bash
pip install git+https://github.com/Fireflies3072/PyController.git
```
This will install the package and its required dependencies: `matplotlib`, `scipy`, `cvxpy`, and `gymnasium[box2d,mojoco]`.

For development, you can clone the repository and install it in editable mode:
```bash
git clone https://github.com/Fireflies3072/PyController.git
cd PyController
pip install -e .
```

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

### Model Predictive Control (MPC)

A Model Predictive Control (MPC) implementation is available for the Rocket Lander scenario (`MPC_RocketLander`). It uses a system model to predict future states and optimize control inputs. You can find an example in `example/rocket_lander/run_mpc.py`.

### Data-Enabled Predictive Control (DeePC)

**Data-enabled Predictive Control (DeePC)** is a control strategy that computes an optimal control policy using only previously collected input-output data from a system. Unlike traditional Model Predictive Control (MPC), DeePC does not require an explicit, identified mathematical model (like state-space equations or a transfer function). For a detailed theoretical background, refer to the original paper: [https://arxiv.org/abs/1811.05890](https://arxiv.org/abs/1811.05890).

The core assumption is that the system is linear time-invariant (LTI) and that any future trajectory of the system can be represented by a **linear combination** of its past trajectories. The "model" is replaced by a large data matrix (a Hankel matrix) containing these past trajectories.

The goal is to find the specific linear combination (represented by a vector $g$) that:
1.  Is consistent with the system's most recent behavior (the "initial condition").
2.  Steers the system's future output ($y$) as close as possible to a reference ($y_{\text{ref}}$).
3.  Minimizes the required control effort ($u$).

---

**The Hankel Matrix**

The "data-driven model" is built from a single, sufficiently long, and persistently excited data trajectory of inputs ($u$) and outputs ($y$). This data is organized into four matrices:

*   **$U_p$ (Past Inputs):** A block Hankel matrix where each column is a sequence of $T_{ini}$ past input samples.
*   **$Y_p$ (Past Outputs):** A block Hankel matrix where each column is a sequence of $T_{ini}$ past output samples.
*   **$U_f$ (Future Inputs):** A block Hankel matrix where each column is the sequence of $T_{f}$ future input samples that *followed* the corresponding $U_p, Y_p$ data.
*   **$Y_f$ (Future Outputs):** A block Hankel matrix where each column is the sequence of $T_{f}$ future output samples that *followed* the corresponding $U_p, Y_p$ data.

These are stacked to form a full data matrix $H$:
$$
H = \begin{pmatrix} U_p \\ Y_p \\ U_f \\ Y_f \end{pmatrix}
$$
Any valid trajectory $\begin{pmatrix} u \\ y \end{pmatrix}$ is assumed to be in the image of this matrix, meaning there exists a vector $g$ such that:
$$
\begin{pmatrix} u_{\text{ini}} \\ y_{\text{ini}} \\ u \\ y \end{pmatrix} \approx \begin{pmatrix} U_p \\ Y_p \\ U_f \\ Y_f \end{pmatrix} g
$$
The vector $g$ becomes the decision variable.

---

**The Optimization Problem**

Solving DeePC involves finding the optimal $g$ by solving a convex optimization problem, specifically a **Quadratic Program (QP)**.

**Variables**

1.  **$g$**: The primary decision variable. This vector (of length $L$, the number of columns in the Hankel matrix) represents the weights used to combine past trajectories to predict the future.
2.  **$\sigma_y$**: A slack variable (a vector of size $p \times T_{ini}$) that allows for a "soft" matching of the initial output condition. This adds robustness against noise or unmodeled dynamics, preventing the problem from becoming infeasible if the exact initial condition $y_{ini}$ isn't perfectly represented in the data.

**Objective Function**

The goal is to **minimize** a cost function $J(g, \sigma_y)$ that balances four terms:

$$
\min_{g, \sigma_y} \underbrace{\| Y_f g - y_{\text{ref}} \|_Q^2}_{\text{Tracking Error}} + \underbrace{\| U_f g \|_R^2}_{\text{Control Effort}} + \underbrace{\lambda_g \|g\|_2^2}_{g \text{ Regularization}} + \underbrace{\lambda_y \|\sigma_y\|_2^2}_{\text{Slack Penalty}}
$$

*   **Tracking Error:** Penalizes the difference between the predicted future outputs ($Y_f g$) and the desired reference trajectory ($y_{\text{ref}}$). $Q$ is a weighting matrix for the outputs.
*   **Control Effort:** Penalizes the magnitude of the predicted future control inputs ($U_f g$). $R$ is a weighting matrix for the inputs.
*   **$g$ Regularization:** A Tikhonov regularization term ($\lambda_g$) that keeps the $g$ vector's norm small. This helps to find a "simple" solution, improves numerical stability, and prevents overfitting to the data.
*   **Slack Penalty:** Penalizes the use of the slack variable $\sigma_y$. A large $\lambda_y$ forces the solution to strictly match the initial output $y_{ini}$.

**Constraints**

The optimization is subject to two linear constraints that "stitch" the predicted trajectory to the system's current state:

1.  **$U_p g = u_{\text{ini}}$**
    This is a **hard constraint** forcing the linear combination of past inputs ($U_p g$) to *exactly* match the sequence of the most recent $T_{ini}$ inputs ($u_{\text{ini}}$) applied to the system.

2.  **$Y_p g = y_{\text{ini}} + \sigma_y$**
    This is a **soft constraint** that matches the linear combination of past outputs ($Y_p g$) to the sequence of the most recent $T_{ini}$ measurements ($y_{\text{ini}}$), allowing for some error/slack $\sigma_y$.

---

**Solver**

This problem is a **Quadratic Program (QP)** because the objective function is quadratic and the constraints are linear.

CVXPY then passes this problem to a backend solver. The code specifies `cp.OSQP`, which is an efficient, open-source solver capable of handling QPs and other more general convex problems.

After the solver finds the optimal $g$, the controller extracts the predicted future control sequence $u_f = U_f g$. It applies only the **first control action** from this sequence to the system. At the next time step, the entire process (updating $u_{ini}$ and $y_{ini}$, and re-solving the QP) is repeated.

Check `example/rocket_lander/run_deepc_random.py` and `example/rocket_lander/run_deepc_pid.py` for detailed examples.

## Future Updates

We are planning to expand the library with more environments and examples:

*   **Inverted Pendulum**: The current examples don't work well. We will tune parameters or try new algorithms to make them work better.

## Acknowledgement

The Rocket Lander environment is a Box2D Gymnasium environment which simulates a Falcon 9 ocean barge landing.

Modified by Dylan Vogel and Gerasimos Maltezos for the 2023 Computation Control course at ETH Zurich.
Original environment created by Reuben Ferrante (https://github.com/arex18/rocket-lander).
