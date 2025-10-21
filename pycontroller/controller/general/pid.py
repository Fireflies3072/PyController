import math

from .controller_base import ControllerBase

class PID_Controller(ControllerBase):
    def __init__(self, kp: float, ki: float, kd: float,
        min_output: float = -math.inf, max_output: float = math.inf,
        anti_windup_gain: float = 1.0) -> None:

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.anti_windup_gain = anti_windup_gain

        self._integrator = 0.0
        self._prev_measurement = None

    def reset(self) -> None:
        self._integrator = 0.0
        self._prev_measurement = None

    def update(self, measurement: float, target: float, dt: float = 1.0) -> float:
        error = target - measurement
        # Proportional
        p = self.kp * error
        # Derivative on measurement to avoid derivative kick
        if self._prev_measurement is None or dt <= 0:
            d = 0.0
        else:
            d_measurement = (measurement - self._prev_measurement) / dt
            d = self.kd * d_measurement
        self._prev_measurement = measurement
        # Integral
        i = self._integrator
        #  Use anti-windup back-calculation to avoid further saturation
        unlimited_output = p + i + d
        limited_output = min(self.max_output, max(self.min_output, unlimited_output))
        windup_error = limited_output - unlimited_output
        self._integrator += (self.ki * error * dt) + (self.anti_windup_gain * windup_error * dt)

        return limited_output
    
    def update_by_error(self, error, dt_error, dt: float = 1.0):
        # Proportional
        p = self.kp * error
        # Derivative
        d = self.kd * dt_error
        # Integral
        i = self._integrator
        #  Use anti-windup back-calculation to avoid further saturation
        unlimited_output = p + i + d
        limited_output = min(self.max_output, max(self.min_output, unlimited_output))
        windup_error = limited_output - unlimited_output
        self._integrator += (self.ki * error * dt) + (self.anti_windup_gain * windup_error * dt)

        return limited_output
