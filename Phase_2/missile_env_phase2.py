import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MissileEnvPhase2(gym.Env):
    """
    Phase 2 — 3D environment with a moving target.

    The missile operates in a 3D arena. It has a horizontal heading (theta)
    and a vertical climb angle (gamma). The target moves at constant speed
    in 3D and bounces off all arena walls.

    Observation space (10-dim, Float32):
        [0] d_MT / L                  — normalised 3D distance to target
        [1] phi_MT / pi               — horizontal bearing to target
        [2] elev_angle / (pi/2)       — vertical angle to target  (-1 to 1)
        [3] v / v_M_max               — normalised missile speed
        [4] theta / pi                — horizontal heading
        [5] gamma / (pi/2)            — vertical climb angle      (-1 to 1)
        [6] 1.0                       — fuel (full, Phase 2)
        [7] target_vx / v_T_max       — target x-velocity
        [8] target_vy / v_T_max       — target y-velocity
        [9] target_vz / v_T_max       — target z-velocity

    Action space (3-dim, Float32, each in [-1, 1]):
        [0] thrust:      a_thrust = ((a+1)/2) * a_max   → [0, a_max]
        [1] yaw rate:    a_yaw    = a * omega_max        → [-omega_max, omega_max]
        [2] pitch rate:  a_pitch  = a * pitch_max        → [-pitch_max, pitch_max]
    """

    metadata = {"render_modes": []}

    # Arena
    L         = 3_000.0                            # Horizontal side length (m)
    H         = 3_000.0                            # Altitude ceiling (m)
    D_3D      = np.sqrt(2 * 3_000.0**2 + 3_000.0**2)  # Max 3D diagonal (~5196m)

    # Missile
    a_max     = 40.0        # Max thrust acceleration (m/s²)
    omega_max = 3.0         # Max yaw rate (rad/s)
    pitch_max = 1.5         # Max pitch rate (rad/s)
    v_M_max   = 500.0       # Max missile speed (m/s)
    dt        = 0.1         # Time-step (s)
    r_kill    = 300.0       # Intercept radius (m)
    max_steps = 2_000       # Episode length cap

    # Target
    v_T_min      = 200.0
    v_T_max      = 300.0

    # Fuel
    max_burn_time = 150.0  # seconds of max-thrust flight before fuel exhausted

    def __init__(self, render_mode=None):
        super().__init__()

        self.observation_space = spaces.Box(
            low  = np.array([0.0, -1.0, -1.0, 0.0, -1.0, -1.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high = np.array([1.0,  1.0,  1.0, 1.0,  1.0,  1.0, 1.0,  1.0,  1.0,  1.0], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low  = np.full(3, -1.0, dtype=np.float32),
            high = np.full(3,  1.0, dtype=np.float32),
        )

        # Missile state
        self.x = self.y = self.z = 0.0
        self.v = self.theta = self.gamma = 0.0
        self.fuel = 1.0

        # Target state
        self.target_x  = self.target_y  = self.target_z  = 0.0
        self.target_vx = self.target_vy = self.target_vz = 0.0

        self.step_count    = 0
        self.previous_d_MT = 0.0

    # ------------------------------------------------------------------ #
    #  reset()                                                             #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Spawn missile
        self.x     = self.np_random.uniform(0, self.L)
        self.y     = self.np_random.uniform(0, self.L)
        self.z     = self.np_random.uniform(200, 2_000)
        self.v     = 100.0
        self.theta = self.np_random.uniform(-np.pi, np.pi)
        self.gamma = self.np_random.uniform(-np.pi / 12, np.pi / 12)  # near-level

        # Spawn target (min 300 m away in 3D)
        while True:
            self.target_x = self.np_random.uniform(0, self.L)
            self.target_y = self.np_random.uniform(0, self.L)
            self.target_z = self.np_random.uniform(200, 2_500)
            d = np.sqrt((self.target_x - self.x)**2 +
                        (self.target_y - self.y)**2 +
                        (self.target_z - self.z)**2)
            if d >= 300.0:
                break

        # Target velocity — random 3D direction
        v_T            = self.np_random.uniform(self.v_T_min, self.v_T_max)
        theta_T        = self.np_random.uniform(-np.pi, np.pi)
        gamma_T        = self.np_random.uniform(-np.pi / 6, np.pi / 6)
        self.target_vx = v_T * np.cos(gamma_T) * np.cos(theta_T)
        self.target_vy = v_T * np.cos(gamma_T) * np.sin(theta_T)
        self.target_vz = v_T * np.sin(gamma_T)

        self.fuel          = 1.0
        self.step_count    = 0
        self.previous_d_MT = np.sqrt((self.target_x - self.x)**2 +
                                      (self.target_y - self.y)**2 +
                                      (self.target_z - self.z)**2)

        return self._build_obs(), {}

    # ------------------------------------------------------------------ #
    #  step(action)                                                        #
    # ------------------------------------------------------------------ #
    def step(self, action):
        terminated = False
        truncated  = False

        # Map actions
        a_thrust = ((action[0] + 1.0) / 2.0) * self.a_max
        a_yaw    = action[1] * self.omega_max
        a_pitch  = action[2] * self.pitch_max

        # Fuel consumption — proportional to thrust applied
        fuel_burn  = (a_thrust / self.a_max) * self.dt / self.max_burn_time
        self.fuel  = float(max(0.0, self.fuel - fuel_burn))

        # No fuel → no thrust (control surfaces still work)
        if self.fuel <= 0.0:
            a_thrust = 0.0

        # Missile physics
        self.v     = float(np.clip(self.v + a_thrust * self.dt, 0, self.v_M_max))
        self.theta = self._wrap(self.theta + a_yaw   * self.dt)
        self.gamma = float(np.clip(
            self.gamma + a_pitch * self.dt, -np.pi / 2, np.pi / 2
        ))

        # 3D velocity components
        vx = self.v * np.cos(self.gamma) * np.cos(self.theta)
        vy = self.v * np.cos(self.gamma) * np.sin(self.theta)
        vz = self.v * np.sin(self.gamma)

        self.x += vx * self.dt
        self.y += vy * self.dt
        self.z += vz * self.dt
        self.step_count += 1

        # Move target and bounce off walls
        self.target_x += self.target_vx * self.dt
        self.target_y += self.target_vy * self.dt
        self.target_z += self.target_vz * self.dt

        if self.target_x <= 0 or self.target_x >= self.L:
            self.target_vx *= -1
            self.target_x   = float(np.clip(self.target_x, 0, self.L))
        if self.target_y <= 0 or self.target_y >= self.L:
            self.target_vy *= -1
            self.target_y   = float(np.clip(self.target_y, 0, self.L))
        if self.target_z <= 0 or self.target_z >= self.H:
            self.target_vz *= -1
            self.target_z   = float(np.clip(self.target_z, 0, self.H))

        # 3D distance
        current_d_MT = float(np.sqrt(
            (self.target_x - self.x)**2 +
            (self.target_y - self.y)**2 +
            (self.target_z - self.z)**2
        ))

        # Reward
        reward = 0.0
        reward += (self.previous_d_MT - current_d_MT) / self.D_3D
        self.previous_d_MT = current_d_MT

        # Terminal 1: intercept
        if current_d_MT < self.r_kill:
            reward    += 100.0
            terminated = True

        # Terminal 2: out of bounds
        if (self.x < 0 or self.x > self.L or
                self.y < 0 or self.y > self.L or
                self.z < 0 or self.z > self.H):
            reward    -= 50.0
            terminated = True

        # Terminal 2: timeout
        if self.step_count >= self.max_steps:
            reward   -= 30.0
            truncated = True

        return self._build_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _build_obs(self) -> np.ndarray:
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dz = self.target_z - self.z

        d_MT        = float(np.sqrt(dx**2 + dy**2 + dz**2))
        phi_MT      = self._wrap(np.arctan2(dy, dx) - self.theta)
        elev_angle  = np.arctan2(dz, np.hypot(dx, dy))

        return np.array([
            d_MT               / self.D_3D,
            phi_MT             / np.pi,
            elev_angle         / (np.pi / 2),
            self.v             / self.v_M_max,
            self.theta         / np.pi,
            self.gamma         / (np.pi / 2),
            self.fuel,                           # real fuel level [0, 1]
            self.target_vx     / self.v_T_max,
            self.target_vy     / self.v_T_max,
            self.target_vz     / self.v_T_max,
        ], dtype=np.float32)

    @staticmethod
    def _wrap(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)
