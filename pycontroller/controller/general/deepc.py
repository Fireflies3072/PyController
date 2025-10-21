from typing import List, Optional, Tuple, Sequence, Union
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from .controller_base import ControllerBase

def build_deepc_hankel(u: np.ndarray, y: np.ndarray, T_ini: int, T_f: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build past/future Hankel blocks U_p, Y_p, U_f, Y_f from a single long
    trajectory (u, y).

    Supports SISO and MIMO. For MIMO with m inputs and p outputs, shapes are:
        U_p: (m*T_ini, L)
        Y_p: (p*T_ini, L)
        U_f: (m*T_f, L)
        Y_f: (p*T_f, L)
    where L = T - (T_ini + T_f) + 1 and T is number of samples.
    """

    # Check if u and y have the same number of time steps
    if u.shape[0] != y.shape[0]:
        raise ValueError("u and y must have the same number of time steps")

    T = u.shape[0]
    L = T - (T_ini + T_f) + 1
    if L <= 0:
        raise ValueError("Not enough data to build Hankel matrices. Increase data length or reduce horizons.")

    def hankel_blocks_matrix(signal_2d: np.ndarray, rows: int, start: int = 0) -> np.ndarray:
        # signal_2d: (T, d). Return (d*rows, L)
        blocks = []
        for i in range(rows):
            seg = signal_2d[start + i : start + i + L, :]  # (L, d)
            blocks.append(seg.T)  # (d, L)
        return np.vstack(blocks)

    U_p = hankel_blocks_matrix(u, T_ini, start=0)
    Y_p = hankel_blocks_matrix(y, T_ini, start=0)
    U_f = hankel_blocks_matrix(u, T_f, start=T_ini)
    Y_f = hankel_blocks_matrix(y, T_f, start=T_ini)
    return U_p, Y_p, U_f, Y_f

class DeePC_Controller(ControllerBase):
    """
    Data-enabled Predictive Control (DeePC) for SISO/MIMO systems using
    a regularized least-squares formulation (no external QP dependency).

    It minimizes over g:
        ||U_p g - u_ini||^2 + ||Y_p g - y_ini||^2
        + ||sqrt(Q) (Y_f g - y_ref)||^2 + ||sqrt(R) U_f g||^2 + lambda_g ||g||^2

    Control applied is the first input block of U_f g, clipped to limits.

    New features:
      - Supports multi-dimensional inputs and outputs (MIMO)
      - Internal data buffer: call reset(), then add_sample(u, y) repeatedly;
        when finished, call build_hankel_from_buffer() to prepare matrices
        without external helpers.
    """

    def __init__(self, u_size: int, y_size: int, T_ini: int = 1, T_f: int = 1,
        Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None,
        lambda_g: float = 1e-4, lambda_y: float = 1.0,
        min_output: Tuple = None, max_output: Tuple = None,
    ) -> None:
        # Optional DeePC Hankel blocks (set later via buffer or setter)
        self.U_p: Optional[np.ndarray] = None
        self.Y_p: Optional[np.ndarray] = None
        self.U_f: Optional[np.ndarray] = None
        self.Y_f: Optional[np.ndarray] = None

        self.T_ini = int(T_ini)
        self.T_f = int(T_f)
        self.Q = np.tile(Q, T_f) if Q is not None else np.ones((y_size * T_f,))
        self.R = np.tile(R, T_f) if R is not None else np.ones((u_size * T_f,))
        self.Q_sqrt = np.sqrt(self.Q)
        self.R_sqrt = np.sqrt(self.R)
        self.lambda_g = float(lambda_g)
        self.lambda_y = float(lambda_y)
        self.min_output = np.array(min_output) if min_output is not None else np.full(u_size, -np.inf)
        self.max_output = np.array(max_output) if max_output is not None else np.full(u_size, np.inf)

        # Dimensions (inferred once data is set)
        self.u_size = u_size
        self.y_size = y_size

        # Histories for initial condition (length T_ini); stored oldest->newest
        self._u_hist = None  # shape (T_ini, m)
        self._y_hist = None  # shape (T_ini, p)

        # Internal raw data buffer for identification (store as plain Python lists)
        self._buffer_u: List[List[float]] = []  # each length m
        self._buffer_y: List[List[float]] = []  # each length p

    def reset(self, initial_y: Union[float, Sequence[float]]) -> None:
        # Clear histories and initialize when sizes known
        self._u_hist = np.zeros((self.T_ini, self.u_size), dtype=np.float64)
        self._y_hist = np.tile(np.atleast_1d(np.asarray(initial_y, dtype=np.float64)).reshape(-1), (self.T_ini, 1))
        # Clear internal buffers to allow new data collection
        self._buffer_u.clear()
        self._buffer_y.clear()

    def add_sample(self, u: Union[float, Sequence[float]], y: Union[float, Sequence[float]]) -> None:
        """Append one sample to the internal identification buffer.

        - If inputs are numpy arrays, they are converted to Python lists.
        - Validates length against controller sizes (U_size, Y_size).
        - Call reset() before starting a new collection.
        """

        u_vec = np.atleast_1d(np.asarray(u, dtype=float)).reshape(-1)
        y_vec = np.atleast_1d(np.asarray(y, dtype=float)).reshape(-1)
        if u_vec.size == 0 or y_vec.size == 0:
            raise ValueError("u and y must be non-empty")
        if u_vec.size != self.u_size:
            raise ValueError(f"u dimension {u_vec.size} does not match expected {self.u_size}")
        if y_vec.size != self.y_size:
            raise ValueError(f"y dimension {y_vec.size} does not match expected {self.y_size}")

        # Store as plain lists
        self._buffer_u.append(u_vec.tolist())
        self._buffer_y.append(y_vec.tolist())

    def build_hankel_from_buffer(self) -> None:
        """Build DeePC Hankel matrices from the internally buffered data."""
        if len(self._buffer_u) == 0 or len(self._buffer_y) == 0:
            raise RuntimeError("No buffered data. Call add_sample(u, y) before building Hankel matrices.")

        u = np.array(self._buffer_u, dtype=np.float64)
        y = np.array(self._buffer_y, dtype=np.float64)
        U_p, Y_p, U_f, Y_f = build_deepc_hankel(u, y, self.T_ini, self.T_f)
        self.U_p = np.asarray(U_p, dtype=np.float64)
        self.Y_p = np.asarray(Y_p, dtype=np.float64)
        self.U_f = np.asarray(U_f, dtype=np.float64)
        self.Y_f = np.asarray(Y_f, dtype=np.float64)

    def update(self, measurement: Union[float, Sequence[float]], target: Union[float, Sequence[float]]) -> Union[float, np.ndarray]:
        u_next, y_next, g, sigma_y = self._update_complete(measurement, target)
        return u_next

    def _update_complete(self, measurement: Union[float, Sequence[float]], target: Union[float, Sequence[float]]) -> Union[float, np.ndarray]:
        if self.U_p is None or self.Y_p is None or self.U_f is None or self.Y_f is None:
            raise RuntimeError("DeePC matrices are not initialized. Build via build_hankel_from_buffer() first.")

        # Build stacked b = [u_ini; y_ini; sqrt(Q) * y_ref; 0]
        u_ini = self._u_hist.reshape(-1)
        y_ini = self._y_hist.reshape(-1)

        target_vec = np.atleast_1d(np.asarray(target, dtype=float)).reshape(-1)
        if target_vec.size != self.y_size:
            raise ValueError(f"target dimension {target_vec.size} does not match outputs {self.y_size}")
        # Repeat target across prediction horizon to match Y_f @ g shape (p*T_f,)
        y_target = np.tile(target_vec, self.T_f)

        # Define variables
        g = cp.Variable((self.U_f.shape[1],))
        sigma_y = cp.Variable((self.y_size * self.T_ini,))
        # Define objective
        objective = cp.Minimize(
            cp.sum_squares(cp.multiply(self.Q_sqrt, (self.Y_f @ g - y_target)))
            + cp.sum_squares(cp.multiply(self.R_sqrt, self.U_f @ g))
            + self.lambda_g * cp.sum_squares(g)
            + self.lambda_y * cp.sum_squares(sigma_y)
        )
        # Define constraints
        constraints = [
            self.U_p @ g == u_ini,
            self.Y_p @ g == y_ini + sigma_y
        ]

        # Solve for g
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)
        g = g.value
        sigma_y = sigma_y.value

        # Clip to output limits (apply only the first control block)
        u_pred = self.U_f @ g
        u_pred = u_pred.reshape(self.T_f, self.u_size)
        u_next = np.clip(u_pred[0], self.min_output, self.max_output)
        y_next = self.Y_f @ g
        y_next = y_next.reshape(self.T_f, self.y_size)
        y_next = y_next[0]

        # Roll histories: include current measurement, and the control to be applied
        y_vec = np.atleast_1d(np.asarray(measurement, dtype=float)).reshape(-1)
        if y_vec.size != self.y_size:
            raise ValueError(f"measurement dimension {y_vec.size} does not match outputs {self.y_size}")

        self._y_hist = np.vstack([self._y_hist[1:, :], y_vec.reshape(1, -1)])
        self._u_hist = np.vstack([self._u_hist[1:, :], u_next.reshape(1, -1)])

        return u_next, y_next, g, sigma_y

class DeePC_Analyzer:
    def __init__(self, labels: List[str]):
        self.labels = labels + ['u norm', 'g norm', 'sigma_y norm']
        self.num_column = 4

        self.u_preds = []
        self.y_preds = []
        self.y_meas = []
        self.y_targets = []
        self.targets = []
        self.g_norms = []
        self.sigma_y_norms = []

    def reset(self):
        self.u_preds.clear()
        self.y_preds.clear()
        self.y_meas.clear()
        self.y_targets.clear()
        self.g_norms.clear()
        self.sigma_y_norms.clear()

    def add_sample(self, u_pred, y_pred, y_meas, y_target, g, sigma_y):
        self.u_preds.append(np.linalg.norm(u_pred))
        self.y_preds.append(y_pred)
        self.y_meas.append(y_meas)
        self.y_targets.append(y_target)
        self.g_norms.append(np.linalg.norm(g))
        self.sigma_y_norms.append(np.linalg.norm(sigma_y))
    
    def analyze(self):
        u_preds = np.array(self.u_preds)
        y_preds = np.array(self.y_preds)
        y_meas = np.array(self.y_meas)
        y_targets = np.array(self.y_targets)
        g_norms = np.array(self.g_norms)
        sigma_y_norms = np.array(self.sigma_y_norms)

        plt.figure()

        # Plot the predictions, measurements, and targets
        for i in range(len(self.labels) - 3):
            plt.subplot(int(np.ceil(len(self.labels) / self.num_column)), self.num_column, i + 1)
            plt.plot(y_preds[:, i], label='prediction')
            plt.plot(y_meas[:, i], '-.', label='measurement')
            plt.plot(y_targets[:, i], ':', label='target')
            plt.legend()
            plt.title(self.labels[i])

        # Plot the u predictions
        plt.subplot(int(np.ceil(len(self.labels) / self.num_column)), self.num_column, len(self.labels) - 2)
        plt.plot(u_preds)
        plt.title(self.labels[-3])

        # Plot the g norms
        plt.subplot(int(np.ceil(len(self.labels) / self.num_column)), self.num_column, len(self.labels) - 1)
        plt.plot(g_norms)
        plt.title(self.labels[-2])

        # Plot the sigma_y norms
        plt.subplot(int(np.ceil(len(self.labels) / self.num_column)), self.num_column, len(self.labels))
        plt.plot(sigma_y_norms)
        plt.title(self.labels[-1])

        plt.show()
