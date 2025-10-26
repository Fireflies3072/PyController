"""
System model for Gymnasium's InvertedPendulum environment
"""

from typing import Tuple

import numpy as np
import scipy


class InvertedPendulumSystemModel:
    """Linearized continuous and discrete system model for an inverted pendulum on a cart.

    The model assumes the standard small-angle linearization about the upright equilibrium
    with state vector ordered as [x, theta, x_dot, theta_dot] and a single input (cart force).
    """

    def __init__(
        self,
        env=None,
        cart_mass: float | None = None,
        pole_mass: float | None = None,
        pole_length: float | None = None,
        gravity: float | None = None,
    ) -> None:
        """Initialize the system model and optionally infer parameters from the environment.

        Args:
            env: Optional Gymnasium environment instance (e.g., InvertedPendulum-v5).
            cart_mass: Mass of the cart M [kg]. Defaults to 1.0 if not provided.
            pole_mass: Mass of the pole m [kg]. Defaults to 0.1 if not provided.
            pole_length: Length from pivot to pole center of mass l [m]. Defaults to 1.0 if not provided.
            gravity: Gravitational acceleration g [m/s^2]. Defaults to 9.81 if not provided.
        """

        self.env = env

        # Fallback defaults if parameters cannot be inferred from the environment
        self.M = 1.0 if cart_mass is None else float(cart_mass)
        self.m = 0.1 if pole_mass is None else float(pole_mass)
        self.l = 1.0 if pole_length is None else float(pole_length)
        self.g = 9.81 if gravity is None else float(gravity)

        # Attempt to infer parameters from MuJoCo environment if available
        # Safe best-effort extraction; failures simply keep defaults above
        try:
            if self.env is not None and hasattr(self.env, "unwrapped"):
                unwrapped = self.env.unwrapped
                # Heuristics for MuJoCo-based InvertedPendulum
                if hasattr(unwrapped, "model"):
                    model = unwrapped.model
                    _extracted_any = False
                    # Try to identify cart and pole bodies by name
                    cart_idx = None
                    pole_idx = None
                    try:
                        names = getattr(model, "body_names", [])
                        names = [n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n) for n in names]
                        for i, name in enumerate(names):
                            lname = name.lower()
                            if cart_idx is None and ("cart" in lname or "slider" in lname):
                                cart_idx = i
                            if pole_idx is None and ("pole" in lname or "rod" in lname):
                                pole_idx = i
                    except Exception:
                        pass

                    if hasattr(model, "body_mass") and len(model.body_mass) >= 2:
                        masses = np.array(model.body_mass)
                        if cart_idx is not None:
                            self.M = float(masses[cart_idx])
                            _extracted_any = True
                        if pole_idx is not None:
                            self.m = float(masses[pole_idx])
                            _extracted_any = True
                        # Fallback if names not found: choose two largest positive masses
                        if (cart_idx is None) or (pole_idx is None):
                            sorted_pos = np.sort(masses[masses > 0.0])
                            if sorted_pos.size >= 2:
                                self.M = float(sorted_pos[-1])
                                self.m = float(sorted_pos[-2])
                                _extracted_any = True

                    # Infer pole COM distance from geometry: for capsule, geom_size[1] is half-length
                    try:
                        g_names = getattr(model, "geom_names", [])
                        g_names = [n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n) for n in g_names]
                        g_idx = None
                        for i, name in enumerate(g_names):
                            if "pole" in name.lower() or "rod" in name.lower():
                                g_idx = i
                                break
                        sizes = np.array(model.geom_size)
                        if sizes.ndim == 2 and sizes.shape[0] > 0:
                            if g_idx is not None:
                                size = sizes[g_idx]
                            else:
                                size = sizes[0]
                            # For capsule: [radius, half-length, 0]; for box/cylinder: use max dimension heuristically
                            if size.shape[0] >= 2 and size[1] > 0:
                                self.l = float(size[1])  # half-length ≈ COM distance
                                _extracted_any = True
                            else:
                                self.l = float(np.max(size))
                                _extracted_any = True
                    except Exception:
                        pass

                    # If anything was extracted, print a concise summary including dt
                    if _extracted_any:
                        dt_val = None
                        try:
                            dt_val = self.get_env_dt()
                        except Exception:
                            dt_val = None
                        dt_str = f"{dt_val:.6g}" if isinstance(dt_val, (int, float)) and dt_val is not None else "n/a"
                        print(
                            f"InvertedPendulum parameters from env -> M: {self.M:.6g}, m: {self.m:.6g}, l: {self.l:.6g}, g: {self.g:.6g}, dt: {dt_str}"
                        )
        except Exception:
            # Keep defaults on any failure
            pass

        self.A: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.Ad: np.ndarray | None = None
        self.Bd: np.ndarray | None = None

        self.state_shape = 4
        self.action_shape = 1

    def get_env_dt(self) -> float | None:
        """Best-effort extraction of the environment integration step (seconds)."""

        try:
            if self.env is None:
                return None
            unwrapped = getattr(self.env, "unwrapped", self.env)
            if hasattr(unwrapped, "dt"):
                return float(unwrapped.dt)
            model = getattr(unwrapped, "model", None)
            if model is not None and hasattr(model, "opt"):
                timestep = float(model.opt.timestep)
                frame_skip = int(getattr(unwrapped, "frame_skip", 1))
                return timestep * frame_skip
        except Exception:
            pass
        return None

    def get_parameters(self) -> dict:
        """Return the parameters currently used by the model (M, m, l, g)."""
        return {"cart_mass": self.M, "pole_mass": self.m, "pole_length": self.l, "gravity": self.g}

    def get_discrete_linear_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the discrete-time linearized system matrices (Ad, Bd)."""

        if self.Ad is None or self.Bd is None:
            raise AttributeError(
                "Ad and Bd matrices are not initialized. Please call `discretize_system_matrices` first"
            )
        return self.Ad, self.Bd

    def get_continuous_linear_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the continuous-time linearized system matrices (A, B)."""

        if self.A is None or self.B is None:
            raise AttributeError(
                "A and B matrices are not initialized. Please call `calculate_linear_system_matrices` first"
            )
        return self.A, self.B

    def calculate_linear_system_matrices(
        self, x_eq: np.ndarray | None = None, u_eq: np.ndarray | None = None
    ) -> None:
        """Calculate the linearized continuous-time system matrices at the upright equilibrium.

        Uses the standard small-angle linearization for an inverted pendulum on a cart.

        Args:
            x_eq: State equilibrium (unused; default upright equilibrium is assumed).
            u_eq: Input equilibrium (unused; default zero force is assumed).
        """

        # Parameters
        M = self.M
        m = self.m
        l = self.l
        g = self.g

        # State order: [x, theta, x_dot, theta_dot]
        self.A = np.zeros((self.state_shape, self.state_shape))
        self.B = np.zeros((self.state_shape, self.action_shape))

        # Kinematics
        self.A[0, 2] = 1.0
        self.A[1, 3] = 1.0

        # Dynamics (frictionless linearization around theta = 0)
        # x_ddot depends on theta and force u. For upright (theta ≈ 0):
        # x_ddot ≈ - (m * g / M) * theta + (1/M) * u
        self.A[2, 1] = - (m * g) / M
        self.B[2, 0] = 1.0 / M

        # theta_ddot depends on theta and force u. For upright (theta ≈ 0):
        # theta_ddot ≈ g * (M + m) / (l * M) * theta - (1/(l*M)) * u
        self.A[3, 1] = g * (M + m) / (l * M)
        self.B[3, 0] = -1.0 / (l * M)

    def discretize_system_matrices(self, sample_time: float) -> None:
        """Exact discretization of the linearized system using the matrix exponential.

        Args:
            sample_time: Discrete sampling time in seconds.
        """

        if self.A is None or self.B is None:
            # Default to the upright linearization if not already computed
            self.calculate_linear_system_matrices()

        # Exact discretization using matrix exponential
        self.Ad = scipy.linalg.expm(self.A * sample_time)

        # Integrate matrix exponential, multiply with B
        Ad_int, _ = scipy.integrate.quad_vec(
            lambda tau: scipy.linalg.expm(self.A * tau), 0.0, sample_time
        )
        self.Bd = Ad_int @ self.B


