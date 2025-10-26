import numpy as np

from ...general.controller_base import ControllerBase
from ...general.pid import PID_Controller

class PID_InvertedPendulum(ControllerBase):
    """ Tuned PID Benchmark against which all other algorithms are compared. """

    def __init__(self, angle_pid_params, force_pid_params, min_output, max_output):
        self.pid_angle = PID_Controller(*angle_pid_params, -0.4, 0.4)
        self.pid_force = PID_Controller(*force_pid_params, min_output, max_output)
    
    def reset(self):
        self.pid_angle.reset()
        self.pid_force.reset()

    def update(self, state: np.ndarray, target: np.ndarray):
        theta_target = self.pid_angle.update(state[0], target[0])
        force = self.pid_force.update(state[1], theta_target)
        return force
