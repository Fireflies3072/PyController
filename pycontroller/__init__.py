from .env import register_envs

# Auto-register environments on package import (idempotent)
try:
    register_envs()
except Exception:
    # If gym isn't installed or registration fails, avoid import-time crash
    pass

# Re-export commonly used controllers for convenience
from .controller import PID_RocketLander, MPC_RocketLander, PID_Controller, DeePC_Controller, DeePC_Analyzer
