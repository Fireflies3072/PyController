from typing import Optional

import numpy as np
import cvxpy as cp

from ...general.controller_base import ControllerBase
from pycontroller.env.rocket_lander.system_model import RocketLanderSystemModel

class MPC_RocketLander(ControllerBase):
    """Short linear MPC for RocketLander using cvxpy."""

    def __init__(self, env, horizon: int = 10, sample_time: float = 0.1,
                 Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None) -> None:
        self.env = env
        self.horizon = int(horizon)
        self.Q = (np.diag([3.0, 0.1, 2.0, 1.0, 120.0, 30.0]) if Q is None else np.asarray(Q, float))
        self.R = (np.diag([0.01, 0.01, 0.01]) if R is None else np.asarray(R, float))

        self.u_size = 3
        self.x_size = 6
        cfg = self.env.unwrapped.cfg

        self.u_min = self.env.action_space.low.astype(np.float64)
        self.u_max = self.env.action_space.high.astype(np.float64)
        self.x_min = np.array([0, 0, -np.inf, -np.inf, -cfg.theta_limit, -np.inf], dtype=np.float64)
        self.x_max = np.array([cfg.width, cfg.height, np.inf, np.inf, cfg.theta_limit, np.inf], dtype=np.float64)

        model = RocketLanderSystemModel(self.env)
        model.calculate_linear_system_matrices()
        model.discretize_system_matrices(sample_time)
        self.A, self.B = model.get_discrete_linear_system_matrices()

    def reset(self, initial_state: Optional[np.ndarray] = None) -> None:
        return

    def update(self, state, target) -> np.ndarray:
        # Define variables
        u = cp.Variable((self.u_size, self.horizon))
        x = cp.Variable((self.x_size, self.horizon + 1))

        # Define objective and constraints
        cost = 0
        constraints = [x[:, 0] == state]
        for t in range(self.horizon):
            cost += cp.quad_form(x[:, t] - target, self.Q) + cp.quad_form(u[:, t], self.R)
            constraints += [x[:, t + 1] == self.A @ x[:, t] + self.B @ u[:, t],
                            u[:, t] >= self.u_min, u[:, t] <= self.u_max,
                            x[:, t] >= self.x_min, x[:, t] <= self.x_max]
        constraints += [x[:, -1] >= self.x_min, x[:, -1] <= self.x_max]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, verbose=False, max_iter=20000)

        if u.value is None:
            return np.zeros(3, dtype=np.float64)
        return np.clip(u[:, 0].value, self.u_min, self.u_max)
