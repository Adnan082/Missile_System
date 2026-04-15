import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MissileEnv(gym.Env):
    """
    Phase 1 Missile Guidance Environment.

    A 2D continuous environment where a missile (controlled by a DRL agent)
    must intercept a stationary target within a bounded arena.

    Observation space (5-dim, Float32):
        [0] current_d_MT / L          — normalised distance to target
        [1] phi_MT / pi               — normalised relative bearing  (-1 to 1)
        [2] v / v_M_max               — normalised speed
        [3] theta / pi                — normalised heading           (-1 to 1)
        [4] 1.0                       — fuel (always full, Phase 1)

    Action space (2-dim, Float32, each in [-1, 1]):
        [0] mapped to thrust:  a_thrust = ((a + 1) / 2) * a_max   → [0, a_max]
        [1] mapped to turn:    a_turn   = a * omega_max            → [-omega_max, omega_max]
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------ #
    #  Simulation constants                                                #
    # ------------------------------------------------------------------ #
    L         = 10_000.0   # Arena side length (m)
    a_max     = 40.0       # Max thrust acceleration (m/s²)
    omega_max = 3.0        # Max turn rate (rad/s)
    v_M_max   = 500.0      # Max missile speed (m/s)
    dt        = 0.1        # Time-step (s)
    r_kill    = 50.0       # Intercept radius (m)
    max_steps = 1_000      # Episode length cap

    def __init__(self, render_mode=None):
        super().__init__()

        self.observation_space = spaces.Box(
            low  = np.array([ 0.0, -1.0,  0.0, -1.0, 1.0], dtype=np.float32),
            high = np.array([ 1.0,  1.0,  1.0,  1.0, 1.0], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low  = np.full(2, -1.0, dtype=np.float32),
            high = np.full(2,  1.0, dtype=np.float32),
        )

        # State variables (initialised in reset)
        self.x = self.y = self.v = self.theta = 0.0
        self.target_x = self.target_y = 0.0
        self.step_count   = 0
        self.previous_d_MT = 0.0

    # ------------------------------------------------------------------ #
    #  reset()                                                             #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # --- Spawn missile ---
        self.x     = self.np_random.uniform(0, self.L)
        self.y     = self.np_random.uniform(0, self.L)
        self.v     = 100.0
        self.theta = self.np_random.uniform(-np.pi, np.pi)

        # --- Spawn target (min 300 m away) ---
        while True:
            self.target_x = self.np_random.uniform(0, self.L)
            self.target_y = self.np_random.uniform(0, self.L)
            d = np.hypot(self.target_x - self.x, self.target_y - self.y)
            if d >= 300.0:
                break

        # --- Tracking variables ---
        self.step_count    = 0
        self.previous_d_MT = np.hypot(self.target_x - self.x, self.target_y - self.y)

        return self._get_obs(
            current_d_MT=self.previous_d_MT,
            phi_MT=self._relative_bearing(),
        ), {}

    # ------------------------------------------------------------------ #
    #  step(action)                                                        #
    # ------------------------------------------------------------------ #
    def step(self, action):
        terminated = False
        truncated  = False

        # --- 2.1 Map actions ---
        a_thrust = ((action[0] + 1.0) / 2.0) * self.a_max   # [0, a_max]
        a_turn   = action[1] * self.omega_max                 # [-omega_max, omega_max]

        # --- 2.2 Physics (Euler integration) ---
        self.v     = float(np.clip(self.v + a_thrust * self.dt, 0, self.v_M_max))
        self.theta = self._wrap(self.theta + a_turn * self.dt)
        self.x    += self.v * np.cos(self.theta) * self.dt
        self.y    += self.v * np.sin(self.theta) * self.dt
        self.step_count += 1

        # --- 2.3 Geometry ---
        dx             = self.target_x - self.x
        dy             = self.target_y - self.y
        current_d_MT   = float(np.hypot(dx, dy))
        angle_to_target = np.arctan2(dy, dx)
        phi_MT         = self._wrap(angle_to_target - self.theta)

        # --- 2.4 Reward ---
        reward = 0.0

        # Dense shaping
        reward += (self.previous_d_MT - current_d_MT) / self.L
        self.previous_d_MT = current_d_MT

        # Terminal 1: intercept
        if current_d_MT < self.r_kill:
            reward     += 100.0
            terminated  = True

        # Terminal 2: out of bounds
        if (self.x < 0 or self.x > self.L or
                self.y < 0 or self.y > self.L):
            reward    -= 50.0
            terminated = True

        # Terminal 3: timeout
        if self.step_count >= self.max_steps:
            reward   -= 30.0
            truncated = True

        # --- 2.5 Build observation and return ---
        obs = self._get_obs(current_d_MT, phi_MT)
        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _get_obs(self, current_d_MT: float, phi_MT: float) -> np.ndarray:
        return np.array([
            current_d_MT / self.L,
            phi_MT       / np.pi,
            self.v       / self.v_M_max,
            self.theta   / np.pi,
            1.0,
        ], dtype=np.float32)

    def _relative_bearing(self) -> float:
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        return self._wrap(np.arctan2(dy, dx) - self.theta)

    @staticmethod
    def _wrap(angle: float) -> float:
        """Wrap angle to (-pi, pi]."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)
