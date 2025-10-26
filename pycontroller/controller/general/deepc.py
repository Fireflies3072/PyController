from typing import List, Optional, Tuple, Sequence, Union
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os

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
    Data-enabled Predictive Control (DeePC).

    This controller uses previously collected input-output data from a system to
    compute an optimal control policy without requiring an explicit mathematical model.
    It is based on the assumption that the system is linear time-invariant (LTI)
    and that any future trajectory can be represented by a linear combination of
    past trajectories stored in a Hankel matrix.

    The optimization problem is formulated as a Quadratic Program (QP) with the
    following components:

    Decision Variables:
    - g: The primary decision variable, representing the weights used to combine
      past trajectories to predict the future.
    - sigma_y: A slack variable that allows for a "soft" matching of the initial
      output condition, adding robustness against noise.

    Objective Function:
    min ||Y_f g - y_ref||^2_Q + ||U_f g||^2_R + lambda_g ||g||^2 + lambda_y ||sigma_y||^2

    Constraints:
    - U_p g = u_ini  (hard constraint for initial inputs)
    - Y_p g = y_ini + sigma_y (soft constraint for initial outputs)

    This implementation includes:
    - Support for MIMO systems.
    - An internal data buffer for convenient data collection.
    - Regularization terms and slack variables for robustness.
    """

    def __init__(self, u_size: int, y_size: int, T_ini: int = 1, T_f: int = 1,
        Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None,
        lambda_g: float = 1e-4, lambda_y: float = 1.0,
        min_output: Tuple = None, max_output: Tuple = None,
        hankel_columns: int = None, y_labels: List[str] = None
    ) -> None:
        """
        Initializes the DeePC controller.

        Args:
            u_size (int): The dimension of the control input vector u.
            y_size (int): The dimension of the system output vector y.
            T_ini (int): The number of past time steps to use for the initial condition.
            T_f (int): The number of future time steps in the prediction horizon.
            Q (Optional[np.ndarray]): The weighting matrix for the output tracking error.
                If None, defaults to an identity matrix. Shape (y_size,).
            R (Optional[np.ndarray]): The weighting matrix for the control effort.
                If None, defaults to an identity matrix. Shape (u_size,).
            lambda_g (float): The regularization weight for the g vector to prevent
                overfitting and improve numerical stability.
            lambda_y (float): The regularization weight for the slack variable sigma_y,
                which allows for soft-constraint satisfaction of the initial output.
            min_output (Tuple): A tuple specifying the minimum allowable values for
                each control input.
            max_output (Tuple): A tuple specifying the maximum allowable values for
                each control input.
            hankel_columns (int): The minimum number of columns required in the Hankel
                matrix before the controller considers the data sufficient. If None,
                a default value is calculated based on system dimensions.
            y_labels (List[str]): A list of strings to be used as labels for the
                output variables in plots.
        """

        self.u_size = int(u_size)
        self.y_size = int(y_size)
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
        self.hankel_columns = hankel_columns if hankel_columns is not None else (T_ini + T_f) * (u_size + y_size + 2)
        self.y_labels = y_labels if y_labels is not None else [f'state[{i}]' for i in range(y_size)]

        # Optional DeePC Hankel blocks (set later via buffer or setter)
        self.U_p: Optional[np.ndarray] = None
        self.Y_p: Optional[np.ndarray] = None
        self.U_f: Optional[np.ndarray] = None
        self.Y_f: Optional[np.ndarray] = None

        # Internal sample buffer for building Hankel matrices
        self.enough_data = False
        self._buffer_u: List[np.ndarray] = []  # each length u_size
        self._buffer_y: List[np.ndarray] = []  # each length y_size

        # Histories for initial condition (length T_ini); stored oldest->newest
        self._u_hist = np.zeros((T_ini, u_size), dtype=np.float64)  # shape (T_ini, u_size)
        self._y_hist = np.zeros((T_ini, y_size), dtype=np.float64)  # shape (T_ini, y_size)

        # Analyzer variables
        self.num_column = 4
        self.y_preds = []
        self.y_meas = []
        self.y_targets = []
        self.u_norms = []
        self.g_norms = []
        self.sigma_y_norms = []

    def reset(self) -> None:
        """
        Resets the controller to its initial state.

        This method clears all data, including Hankel matrices, internal buffers,
        and historical data. It should be called before starting a new data
        collection phase from scratch.
        """
        # Clear Hankel matrices
        self.U_p = None
        self.Y_p = None
        self.U_f = None
        self.Y_f = None
        # Clear internal buffers to allow new data collection
        self.enough_data = False
        self._buffer_u.clear()
        self._buffer_y.clear()
        # Clear histories
        self._u_hist = np.zeros((self.T_ini, self.u_size), dtype=np.float64)
        self._y_hist = np.zeros((self.T_ini, self.y_size), dtype=np.float64)
        self._u_mask_hist = np.zeros((self.T_ini, self.u_size), dtype=np.float64)
        self._y_mask_hist = np.zeros((self.T_ini, self.y_size), dtype=np.float64)
        # Clear analyzer variables
        self.y_preds.clear()
        self.y_meas.clear()
        self.y_targets.clear()
        self.u_norms.clear()
        self.g_norms.clear()
        self.sigma_y_norms.clear()
    
    def new_episode(self) -> None:
        """
        Prepares the controller for a new episode.

        This method should be called at the beginning of each new data collection
        run. It creates new buffers for data and resets histories and analyzer variables.
        """
        # Clear internal buffers to allow new data collection
        self._buffer_u.append([])
        self._buffer_y.append([])
        # Initialize histories for a new episode
        self._u_hist = np.zeros((self.T_ini, self.u_size), dtype=np.float64)
        self._y_hist = np.zeros((self.T_ini, self.y_size), dtype=np.float64)
        self._u_mask_hist = np.zeros((self.T_ini, self.u_size), dtype=np.float64)
        self._y_mask_hist = np.zeros((self.T_ini, self.y_size), dtype=np.float64)
        # Clear analyzer variables
        self.y_preds.clear()
        self.y_meas.clear()
        self.y_targets.clear()
        self.u_norms.clear()
        self.g_norms.clear()
        self.sigma_y_norms.clear()

    def collect_data_for_hankel(self, u: Union[float, Sequence[float]], y: Union[float, Sequence[float]]) -> None:
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
        self._buffer_u[-1].append(u_vec)
        self._buffer_y[-1].append(y_vec)

        # Check if the number of columns in the Hankel matrix is enough
        num_columns = 0
        for i in range(len(self._buffer_u)):
            if len(self._buffer_u[i]) < self.T_ini + self.T_f:
                continue
            num_columns += len(self._buffer_u[i]) - self.T_ini - self.T_f + 1
        self.enough_data = num_columns >= self.hankel_columns

    def build_hankel_matrix(self) -> None:
        """Build DeePC Hankel matrices from the internally buffered data."""
        if len(self._buffer_u) == 0 or len(self._buffer_y) == 0:
            raise RuntimeError("No buffered data. Call add_sample(u, y) before building Hankel matrices.")
        if len(self._buffer_u) != len(self._buffer_y):
            raise ValueError("The number of u and y buffers must be the same")
        
        U_p, Y_p, U_f, Y_f = [], [], [], []
        for i in range(len(self._buffer_u)):
            if len(self._buffer_u[i]) < self.T_ini + self.T_f:
                continue
            u = np.array(self._buffer_u[i], dtype=np.float64)
            y = np.array(self._buffer_y[i], dtype=np.float64)
            U_p_temp, Y_p_temp, U_f_temp, Y_f_temp = build_deepc_hankel(u, y, self.T_ini, self.T_f)
            U_p.append(U_p_temp)
            Y_p.append(Y_p_temp)
            U_f.append(U_f_temp)
            Y_f.append(Y_f_temp)
        self.U_p = np.concatenate(U_p, axis=1)
        self.Y_p = np.concatenate(Y_p, axis=1)
        self.U_f = np.concatenate(U_f, axis=1)
        self.Y_f = np.concatenate(Y_f, axis=1)

    def update(self, state: Union[float, Sequence[float]], target: Union[float, Sequence[float]], dt=None) -> Union[float, np.ndarray]:
        """
        Computes the optimal control action for the current state.

        This method solves the DeePC optimization problem to find the best control
        action that steers the system towards the target. It uses the historical
        data (`u_hist`, `y_hist`) as the initial condition.

        Args:
            state (Union[float, Sequence[float]]): The current measurement of the
                system output (y_k).
            target (Union[float, Sequence[float]]): The desired reference value for
                the system output (y_ref).
            dt (float, optional): Time step. Not used in this controller.

        Returns:
            Union[float, np.ndarray]: The first control action (u_k) from the
            optimal future input sequence.
        """
        # Check if the DeePC matrices are initialized
        if self.U_p is None or self.Y_p is None or self.U_f is None or self.Y_f is None:
            raise RuntimeError("DeePC matrices are not initialized. Build via build_hankel_from_buffer() first.")
        # Check if the state and target dimensions match the output dimension
        y_vec = np.atleast_1d(np.asarray(state, dtype=float)).reshape(-1)
        target_vec = np.atleast_1d(np.asarray(target, dtype=float)).reshape(-1)
        if y_vec.size != self.y_size:
            raise ValueError(f"state dimension {y_vec.size} does not match outputs {self.y_size}")
        if target_vec.size != self.y_size:
            raise ValueError(f"target dimension {target_vec.size} does not match outputs {self.y_size}")
        
        # Set the current state and mask
        self._y_hist[-1] = y_vec
        self._y_mask_hist[-1] = np.ones((self.y_size,), dtype=np.float64)

        # Build stacked b = [u_ini; y_ini; U_f @ g; sqrt(Q) * y_target]
        u_ini = self._u_hist.reshape(-1)
        y_ini = self._y_hist.reshape(-1)
        u_mask_ini = self._u_mask_hist.reshape(-1)
        y_mask_ini = self._y_mask_hist.reshape(-1)
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
            cp.multiply(self.U_p @ g, u_mask_ini) == u_ini * u_mask_ini,
            cp.multiply(self.Y_p @ g, y_mask_ini) == y_ini * y_mask_ini + sigma_y
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
        self._u_hist = np.vstack([self._u_hist[1:, :], u_next.reshape(1, -1)])
        self._u_mask_hist = np.vstack([self._u_mask_hist[1:, :], np.ones((1, self.u_size))])

        self.y_preds.append(y_next)
        self.y_targets.append(target)
        self.g_norms.append(np.linalg.norm(g))
        self.sigma_y_norms.append(np.linalg.norm(sigma_y))

        return u_next
    
    def roll_history(self, action: Union[float, Sequence[float]], next_state: Union[float, Sequence[float]]):
        """
        Updates the history of inputs and outputs with the latest data.

        This should be called after applying an action to the system and observing
        the next state. It updates the `_u_hist` and `_y_hist` buffers, which are
        used as the initial condition in the next `update` call.

        Args:
            action (Union[float, Sequence[float]]): The control action that was applied.
            next_state (Union[float, Sequence[float]]): The resulting system output.
        """
        # Check if dimensions match
        u_vec = np.atleast_1d(np.asarray(action, dtype=float)).reshape(-1)
        y_vec = np.atleast_1d(np.asarray(next_state, dtype=float)).reshape(-1)
        if u_vec.size != self.u_size:
            raise ValueError(f"action dimension {u_vec.size} does not match expected {self.u_size}")
        if y_vec.size != self.y_size:
            raise ValueError(f"next_state dimension {y_vec.size} does not match expected {self.y_size}")

        # Roll histories: include applied action and next state
        self._u_hist = np.vstack([self._u_hist[1:, :], u_vec.reshape(1, -1)])
        self._u_mask_hist = np.vstack([self._u_mask_hist[1:, :], np.ones((1, self.u_size))])
        self._y_hist = np.vstack([self._y_hist[1:, :], y_vec.reshape(1, -1)])
        self._y_mask_hist = np.vstack([self._y_mask_hist[1:, :], np.ones((1, self.y_size))])

        # Record the data
        self.u_norms.append(np.linalg.norm(u_vec))
        self.y_meas.append(y_vec)
    
    def analyze(self, figure_filename: str = None):
        """
        Plots the results of the control episode.

        Generates and displays plots for output predictions vs. measurements,
        control input norms, g vector norms, and slack variable norms.

        Args:
            figure_filename (str, optional): If provided, the plot is saved to
                this file.
        """
        y_preds = np.array(self.y_preds)
        y_meas = np.array(self.y_meas)
        y_targets = np.array(self.y_targets)
        u_norms = np.array(self.u_norms)
        g_norms = np.array(self.g_norms)
        sigma_y_norms = np.array(self.sigma_y_norms)

        # Plot the results
        num_row = int(np.ceil((self.y_size + 3) / self.num_column))
        plt.figure(figsize=(self.num_column * 4, num_row * 4))
        plt.subplots_adjust(hspace=0.5)

        # Plot the predictions, measurements, and targets
        for i in range(self.y_size):
            plt.subplot(num_row, self.num_column, i + 1)
            plt.plot(y_preds[:, i], label='prediction')
            plt.plot(y_meas[:, i], '-.', label='measurement')
            plt.plot(y_targets[:, i], ':', label='target')
            plt.legend()
            plt.title(self.y_labels[i])

        # Plot the u norms
        plt.subplot(num_row, self.num_column, self.y_size + 1)
        plt.plot(u_norms)
        plt.title('u norm')

        # Plot the g norms
        plt.subplot(num_row, self.num_column, self.y_size + 2)
        plt.plot(g_norms)
        plt.title('g norm')

        # Plot the sigma_y norms
        plt.subplot(num_row, self.num_column, self.y_size + 3)
        plt.plot(sigma_y_norms)
        plt.title('sigma_y norm')

        # Save the figure
        if figure_filename is not None:
            os.makedirs(os.path.dirname(figure_filename), exist_ok=True)
            plt.savefig(figure_filename, dpi=300, bbox_inches='tight')
        
        plt.show()
