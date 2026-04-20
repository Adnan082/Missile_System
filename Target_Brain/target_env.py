import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TargetEnv(gym.Env):
    """
    Target evasion environment.
    The TARGET is the agent — it must survive as long as possible
    while a frozen missile (pure pursuit approximation) chases it.

    Observation (10-dim):
        [0]  d_MT / D_3D          distance to missile (normalised)
        [1]  phi_MT / pi          bearing to missile (ego-centric)
        [2]  elev_MT / (pi/2)     elevation to missile
        [3]  v_T / v_T_max        target speed
        [4]  theta_T / pi         target heading
        [5]  gamma_T / (pi/2)     target climb angle
        [6]  m_vx / v_M_max       missile vx
        [7]  m_vy / v_M_max       missile vy
        [8]  m_vz / v_M_max       missile vz
        [9]  d_MT / D_3D          closing rate proxy (same as [0] for now)

    Action (3-dim, each in [-1, 1]):
        [0]  yaw rate
        [1]  pitch rate
        [2]  speed change
    """

    metadata = {"render_modes": []}

    # Arena
    L    = 3_000.0
    H    = 3_000.0
    D_3D = np.sqrt(2 * 3_000.0**2 + 3_000.0**2)

    # Target — agile but slower (like a fighter aircraft)
    v_T_min   = 200.0
    v_T_max   = 300.0
    omega_max = 4.0      # more agile than missile — can out-turn it
    pitch_max = 2.0
    dt        = 0.1
    max_steps = 2_000
    r_kill    = 300.0    # matches arena kill radius (MissileEnvPhase3b.r_kill)

    # Missile — fast but less maneuverable at high speed (realistic)
    v_M         = 350.0
    v_M_max     = 500.0
    m_omega_max = 1.5    # missile turn radius = 350/1.5 = 233m (wider than target)
    m_pitch_max = 0.8

    def __init__(self, render_mode=None):
        super().__init__()

        low  = np.array([0, -1, -1, 0, -1, -1, -1, -1, -1, 0], dtype=np.float32)
        high = np.array([1,  1,  1, 1,  1,  1,  1,  1,  1, 1], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Box(
            low  = np.full(3, -1.0, dtype=np.float32),
            high = np.full(3,  1.0, dtype=np.float32),
        )

        self.x = self.y = self.z = 0.0
        self.v = self.theta = self.gamma = 0.0
        self.mx = self.my = self.mz = 0.0
        self.mvx = self.mvy = self.mvz = 0.0
        self.m_theta = self.m_gamma = 0.0
        self.step_count    = 0
        self.previous_d_MT = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Spawn target
        self.x     = self.np_random.uniform(200, self.L - 200)
        self.y     = self.np_random.uniform(200, self.L - 200)
        self.z     = self.np_random.uniform(300, 2_000)
        self.v     = self.np_random.uniform(self.v_T_min, self.v_T_max)
        self.theta = self.np_random.uniform(-np.pi, np.pi)
        self.gamma = self.np_random.uniform(-np.pi / 12, np.pi / 12)

        # Spawn missile at least 800m away
        while True:
            self.mx = self.np_random.uniform(0, self.L)
            self.my = self.np_random.uniform(0, self.L)
            self.mz = self.np_random.uniform(200, 2_000)
            d = np.sqrt((self.mx - self.x)**2 +
                        (self.my - self.y)**2 +
                        (self.mz - self.z)**2)
            if d >= 800.0:
                break

        self.mvx = self.mvy = self.mvz = 0.0
        self.m_theta = float(np.arctan2(self.y - self.my, self.x - self.mx))
        self.m_gamma = 0.0
        self.step_count    = 0
        self.previous_d_MT = d

        return self._build_obs(), {}

    def step(self, action):
        terminated = False
        truncated  = False
        info       = {}

        # Target control
        a_yaw   = float(action[0]) * self.omega_max
        a_pitch = float(action[1]) * self.pitch_max
        dv      = float(action[2]) * 20.0  # speed change ±20 m/s

        self.v     = float(np.clip(self.v + dv * self.dt, self.v_T_min, self.v_T_max))
        self.theta = self._wrap(self.theta + a_yaw * self.dt)
        self.gamma = float(np.clip(self.gamma + a_pitch * self.dt, -np.pi/2, np.pi/2))

        vx = self.v * np.cos(self.gamma) * np.cos(self.theta)
        vy = self.v * np.cos(self.gamma) * np.sin(self.theta)
        vz = self.v * np.sin(self.gamma)

        self.x += vx * self.dt
        self.y += vy * self.dt
        self.z += vz * self.dt
        self.step_count += 1

        # Bounce off walls
        if self.x <= 0 or self.x >= self.L:
            self.theta = self._wrap(np.pi - self.theta)
            self.x = float(np.clip(self.x, 0, self.L))
        if self.y <= 0 or self.y >= self.L:
            self.theta = self._wrap(-self.theta)
            self.y = float(np.clip(self.y, 0, self.L))
        if self.z <= 0 or self.z >= self.H:
            self.gamma = -self.gamma
            self.z = float(np.clip(self.z, 0, self.H))

        # Move missile — pursuit with turning rate limits (realistic)
        dx = self.x - self.mx
        dy = self.y - self.my
        dz = self.z - self.mz
        dist = float(np.sqrt(dx**2 + dy**2 + dz**2))

        if dist > 1e-6:
            desired_theta = float(np.arctan2(dy, dx))
            desired_gamma = float(np.arctan2(dz, np.hypot(dx, dy)))

            # Missile can only turn so fast — target can exploit this
            dtheta = self._wrap(desired_theta - self.m_theta)
            dgamma = float(np.clip(desired_gamma - self.m_gamma, -self.m_pitch_max*self.dt, self.m_pitch_max*self.dt))
            dtheta = float(np.clip(dtheta, -self.m_omega_max*self.dt, self.m_omega_max*self.dt))

            self.m_theta = self._wrap(self.m_theta + dtheta)
            self.m_gamma = float(np.clip(self.m_gamma + dgamma, -np.pi/2, np.pi/2))

            self.mvx = self.v_M * np.cos(self.m_gamma) * np.cos(self.m_theta)
            self.mvy = self.v_M * np.cos(self.m_gamma) * np.sin(self.m_theta)
            self.mvz = self.v_M * np.sin(self.m_gamma)

        self.mx += self.mvx * self.dt
        self.my += self.mvy * self.dt
        self.mz += self.mvz * self.dt

        # Clamp missile to arena
        self.mx = float(np.clip(self.mx, 0, self.L))
        self.my = float(np.clip(self.my, 0, self.L))
        self.mz = float(np.clip(self.mz, 0, self.H))

        current_d_MT = float(np.sqrt(
            (self.x - self.mx)**2 +
            (self.y - self.my)**2 +
            (self.z - self.mz)**2
        ))

        # Reward — survive and increase distance from missile
        reward = (current_d_MT - self.previous_d_MT) / self.D_3D
        reward += 0.01  # small survival reward per step
        self.previous_d_MT = current_d_MT

        # Terminal: missile hits target
        if current_d_MT < self.r_kill:
            reward    -= 100.0
            terminated = True
            info["outcome"] = "hit"

        # Terminal: target out of bounds
        if not terminated and (
                self.x < 0 or self.x > self.L or
                self.y < 0 or self.y > self.L or
                self.z < 0 or self.z > self.H):
            reward    -= 50.0
            terminated = True
            info["outcome"] = "oob"

        # Truncated: survived full episode
        if not terminated and self.step_count >= self.max_steps:
            reward   += 50.0
            truncated = True
            info["outcome"] = "survived"

        return self._build_obs(), reward, terminated, truncated, info

    def _build_obs(self):
        dx = self.mx - self.x
        dy = self.my - self.y
        dz = self.mz - self.z
        d_MT     = float(np.sqrt(dx**2 + dy**2 + dz**2))
        phi_MT   = self._wrap(np.arctan2(dy, dx) - self.theta)
        elev_MT  = float(np.arctan2(dz, np.hypot(dx, dy)))

        return np.array([
            d_MT            / self.D_3D,
            phi_MT          / np.pi,
            elev_MT         / (np.pi / 2),
            self.v          / self.v_T_max,
            self.theta      / np.pi,
            self.gamma      / (np.pi / 2),
            self.mvx        / self.v_M_max,
            self.mvy        / self.v_M_max,
            self.mvz        / self.v_M_max,
            d_MT            / self.D_3D,
        ], dtype=np.float32)

    @staticmethod
    def _wrap(angle):
        return float((angle + np.pi) % (2 * np.pi) - np.pi)
