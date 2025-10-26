import numpy as np
import cvxpy as cp

from .controller_base import ControllerBase

class MPC_Controller(ControllerBase):
    def __init__(self, A: np.ndarray, B: np.ndarray, horizon: int = 10,
                 Q: np.ndarray=None, R: np.ndarray=None,
                 min_output: np.ndarray=None, max_output: np.ndarray=None,
                 min_state: np.ndarray=None, max_state: np.ndarray=None) -> None:

        self.A = A
        self.B = B
        self.u_size = B.shape[1]
        self.x_size = A.shape[0]
        self.horizon = horizon
        self.Q = Q if Q is not None else np.eye(A.shape[0])
        self.R = R if R is not None else np.eye(B.shape[1])
        self.min_output = min_output if min_output is not None else np.full(self.u_size, -np.inf)
        self.max_output = max_output if max_output is not None else np.full(self.u_size, np.inf)
        self.min_state = min_state if min_state is not None else np.full(self.x_size, -np.inf)
        self.max_state = max_state if max_state is not None else np.full(self.x_size, np.inf)

        # Debug/status fields
        self.last_status = None
        self.last_problem_value = None
        self.last_feasible = None

    def reset(self) -> None:
        pass

    def update(self, state: np.ndarray, target: np.ndarray, dt=None) -> np.ndarray:
        # Define variables
        u = cp.Variable((self.u_size, self.horizon))
        x = cp.Variable((self.x_size, self.horizon + 1))

        # Define objective and constraints
        cost = 0
        constraints = [x[:, 0] == state]
        for t in range(self.horizon):
            cost += cp.quad_form(x[:, t] - target, self.Q) + cp.quad_form(u[:, t], self.R)
            constraints += [x[:, t + 1] == self.A @ x[:, t] + self.B @ u[:, t],
                            u[:, t] >= self.min_output, u[:, t] <= self.max_output,
                            x[:, t] >= self.min_state, x[:, t] <= self.max_state]
        constraints += [x[:, -1] >= self.min_state, x[:, -1] <= self.max_state]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, verbose=False, max_iter=20000)

        # Store status for debugging
        self.last_status = problem.status
        self.last_problem_value = problem.value

        if u.value is None:
            self.last_feasible = False
            return np.zeros(self.u_size, dtype=np.float64)
        self.last_feasible = True
        return np.clip(u[:, 0].value, self.min_output, self.max_output)
