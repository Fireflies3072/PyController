import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RoomTemperature(gym.Env):
    """
    Simple first-order room thermal model conforming to the Gymnasium interface.

    Dynamics: dT/dt = -k * (T - T_out) + u
      - k: cooling constant (1 / time)
      - T_out: outside temperature (degC)
      - u: heater power input (degC / time)
    """
    
    # Metadata (defined by Gymnasium interface)
    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, k_cool: float = None, T_out: float = None, T_initial: float = None,
                 T_target: float = 22.0, dt: float = 0.1,  max_episode_steps: int = 1000,
                 seed: int = None) -> None:
        super().__init__()

        # Fixed parameters
        self.T_target = T_target
        self.dt = dt
        self.max_episode_steps = max_episode_steps

        # Random number generator
        self.rng = np.random.default_rng(seed)
        # Initialize physical parameters
        self.k_cool = k_cool or self.rng.uniform(0.0001, 0.01)
        self.T_out = T_out or self.rng.uniform(-40.0, 20.0)
        self.T_initial = T_initial or self.rng.uniform(-10.0, 30.0)
        self.T_current = self.T_initial

        # Observation space: Room temperature (degC)
        self.observation_space = spaces.Box(low=np.array([-50.0]), high=np.array([50.0]), dtype=np.float64)
        # Action space: Heater power (degC / time)
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([5.0]), dtype=np.float64)

        # Internal variables
        self._current_step = 0
        self._good_steps = 0
        
    def reset(self, seed=None, options=None):
        # Set seed
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Reset initial temperature and step counter
        self.T_current = self.T_initial
        self._current_step = 0
        self._good_steps = 0
        
        # Return observation and info
        observation = np.array([self.T_current], dtype=np.float64)
        info = {
            'k_cool': self.k_cool,
            'T_out': self.T_out,
            'T_target': self.T_target
        }
        return observation, info

    def step(self, action: np.ndarray):
        """ Take an action and return the next observation, reward, terminated, truncated, and info """
        
        # Check action shape
        assert action.shape == (1,), f"Action must have shape (1,), but got {action.shape}"
        u = action[0]

        # Update temperature
        dT_dt = -self.k_cool * (self.T_current - self.T_out) + u * self.dt
        self.T_current += dT_dt
        
        # Calculate reward
        error = self.T_current - self.T_target
        reward = -error**2
        
        # Check truncation status
        self._current_step += 1
        truncated = self._current_step >= self.max_episode_steps
        # Update good steps
        if abs(error) < 0.1:
            self._good_steps += 1
        else:
            self._good_steps = 0
        # Check termination status
        terminated = self._good_steps >= 20 or truncated

        # Prepare output
        observation = np.array([self.T_current], dtype=np.float64)
        info = {
            'k_cool': self.k_cool,
            'T_out': self.T_out,
            'T_target': self.T_target
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        # Simple text output
        print(f"Time Step: {self._current_step}, T_room: {self.T_current:.2f} degC, T_out: {self.T_out:.2f} degC")
        pass
        
    def close(self):
        pass
