import numpy as np
import math
from typing import Tuple

from ...general.controller_base import ControllerBase
from ...general.pid import PID_Controller

class PID_RocketLander(ControllerBase):
    """ Tuned PID Benchmark against which all other algorithms are compared. """

    def __init__(self, Fe_PID_params, FsTheta_PID_params, psi_PID_params,
        min_output: Tuple = None, max_output: Tuple = None):

        self.Fe_PID = PID_Controller(*Fe_PID_params)
        self.Fs_theta_PID = PID_Controller(*FsTheta_PID_params)
        self.psi_PID = PID_Controller(*psi_PID_params)
        self.min_output = np.array(min_output) if min_output is not None else np.full(3, -math.inf)
        self.max_output = np.array(max_output) if max_output is not None else np.full(3, math.inf)

    def reset(self):
        self.Fe_PID.reset()
        self.Fs_theta_PID.reset()
        self.psi_PID.reset()

    def update(self, state, target):
        x, y, vel_x, vel_y, theta, omega = state
        x_target, y_target = target[0], target[1]
        dx = x - x_target
        dy = y - y_target
        # ------------------------------------------
        y_ref = -0.1  # Adjust speed
        y_error = y_ref - dy + 0.1 * dx
        y_dterror = -vel_y + 0.1 * vel_x
        Fe = self.Fe_PID.update_by_error(y_error, y_dterror) * (abs(dx) * 50 + 1)
        # ------------------------------------------
        theta_ref = 0
        theta_error = theta_ref - theta + 0.2 * dx  # theta is negative when slanted to the north east
        theta_dterror = -omega + 0.2 * vel_x
        Fs_theta = self.Fs_theta_PID.update_by_error(theta_error, theta_dterror)
        Fs = -Fs_theta  # + Fs_x
        # ------------------------------------------
        theta_ref = 0
        theta_error = -theta_ref + theta
        theta_dterror = omega
        if (abs(dx) > 0.01 and dy < 0.5):
            theta_error = theta_error - 0.06 * dx  # theta is negative when slanted to the right
            theta_dterror = theta_dterror - 0.06 * vel_x
        psi = self.psi_PID.update_by_error(theta_error, theta_dterror)
        
        # Clip to output limits
        output = np.array([Fe, Fs, psi], dtype=np.float64)
        output = np.clip(output, self.min_output, self.max_output)
        return output
