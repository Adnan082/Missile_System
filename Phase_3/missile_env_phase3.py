import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MissileEnvPhase3(gym.Env):
    """
    Phase 3a — 3D environment with a moving target + one interceptor.

    The missile (agent) must hit the target while evading an interceptor
    that uses pure-pursuit guidance to chase the missile.

    Observation space (16-dim, Float32):
        [0]  d_MT / D_3D          normalised distance to target
        [1]  phi_MT / pi          horizontal bearing to target
        [2]  elev_MT / (pi/2)     elevation angle to target
        [3]  v / v_M_max          missile speed
        [4]  theta / pi           horizontal heading
        [5]  gamma / (pi/2)       climb angle
        [6]  fuel                 fuel level [0, 1]
        [7]  t_vx / v_T_max       target x-velocity
        [8]  t_vy / v_T_max       target y-velocity
        [9]  t_vz / v_T_max       target z-velocity
        [10] d_MI / D_3D          normalised distance to interceptor (1.0 if neutralised)
        [11] phi_MI / pi          horizontal bearing to interceptor
        [12] elev_MI / (pi/2)     elevation angle to interceptor
        [13] i_vx / v_I_max       interceptor x-velocity
        [14] i_vy / v_I_max       interceptor y-velocity
        [15] i_vz / v_I_max       interceptor z-velocity

    Action space (3-dim, Float32, each in [-1, 1]):
        [0] thrust:     a_thrust = ((a+1)/2) * a_max   → [0, a_max]
        [1] yaw rate:   a_yaw    = a * omega_max        → [-omega_max, omega_max]
        [2] pitch rate: a_pitch  = a * pitch_max        → [-pitch_max, pitch_max]
    """

    metadata = {"render_modes": []}

    # Arena
    L    = 3_000.0
    H    = 3_000.0
    D_3D = np.sqrt(2 * 3_000.0**2 + 3_000.0**2)

    # Missile
    a_max     = 40.0
    omega_max = 3.0
    pitch_max = 1.5
    v_M_max   = 500.0
    dt        = 0.1
    r_kill    = 300.0
    max_steps = 2_000

    # Target
    v_T_min = 200.0
    v_T_max = 300.0

    # Fuel
    max_burn_time = 150.0

    # Interceptor — overridden per curriculum stage in train.py
    v_I           = 80.0    # actual speed (m/s)
    v_I_max       = 250.0   # fixed for obs normalisation across all stages
    r_kill_I      = 150.0   # interceptor kill radius (m)
    int_spawn_min = 1500.0  # min spawn distance from missile (m)

    def __init__(self, render_mode=None):
        super().__init__()

        self.observation_space = spaces.Box(
            low  = np.array([0,-1,-1, 0,-1,-1, 0,-1,-1,-1, 0,-1,-1,-1,-1,-1], dtype=np.float32),
            high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
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

        # Interceptor state
        self.int_x  = self.int_y  = self.int_z  = 0.0
        self.int_vx = self.int_vy = self.int_vz = 0.0
        self.interceptor_active = True

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
        self.gamma = self.np_random.uniform(-np.pi / 12, np.pi / 12)

        # Spawn target (min 300m from missile)
        while True:
            self.target_x = self.np_random.uniform(0, self.L)
            self.target_y = self.np_random.uniform(0, self.L)
            self.target_z = self.np_random.uniform(200, 2_500)
            d = np.sqrt((self.target_x - self.x)**2 +
                        (self.target_y - self.y)**2 +
                        (self.target_z - self.z)**2)
            if d >= 300.0:
                break

        # Target velocity
        v_T            = self.np_random.uniform(self.v_T_min, self.v_T_max)
        theta_T        = self.np_random.uniform(-np.pi, np.pi)
        gamma_T        = self.np_random.uniform(-np.pi / 6, np.pi / 6)
        self.target_vx = v_T * np.cos(gamma_T) * np.cos(theta_T)
        self.target_vy = v_T * np.cos(gamma_T) * np.sin(theta_T)
        self.target_vz = v_T * np.sin(gamma_T)

        # Spawn interceptor (min int_spawn_min from missile)
        while True:
            self.int_x = self.np_random.uniform(0, self.L)
            self.int_y = self.np_random.uniform(0, self.L)
            self.int_z = self.np_random.uniform(200, 2_000)
            d_i = np.sqrt((self.int_x - self.x)**2 +
                          (self.int_y - self.y)**2 +
                          (self.int_z - self.z)**2)
            if d_i >= self.int_spawn_min:
                break

        self.int_vx = 0.0
        self.int_vy = 0.0
        self.int_vz = 0.0
        self.interceptor_active = True

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
        info       = {}

        # Map actions
        a_thrust = ((action[0] + 1.0) / 2.0) * self.a_max
        a_yaw    = action[1] * self.omega_max
        a_pitch  = action[2] * self.pitch_max

        # Fuel consumption
        fuel_burn = (a_thrust / self.a_max) * self.dt / self.max_burn_time
        self.fuel = float(max(0.0, self.fuel - fuel_burn))
        if self.fuel <= 0.0:
            a_thrust = 0.0

        # Missile physics
        self.v     = float(np.clip(self.v + a_thrust * self.dt, 0, self.v_M_max))
        self.theta = self._wrap(self.theta + a_yaw   * self.dt)
        self.gamma = float(np.clip(self.gamma + a_pitch * self.dt, -np.pi / 2, np.pi / 2))

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

        # Move interceptor — pure pursuit toward missile
        if self.interceptor_active:
            dx_i   = self.x - self.int_x
            dy_i   = self.y - self.int_y
            dz_i   = self.z - self.int_z
            dist_i = float(np.sqrt(dx_i**2 + dy_i**2 + dz_i**2))

            if dist_i > 1e-6:
                self.int_vx = self.v_I * dx_i / dist_i
                self.int_vy = self.v_I * dy_i / dist_i
                self.int_vz = self.v_I * dz_i / dist_i

            self.int_x += self.int_vx * self.dt
            self.int_y += self.int_vy * self.dt
            self.int_z += self.int_vz * self.dt

            # Interceptor neutralised if it leaves the arena
            if (self.int_x < 0 or self.int_x > self.L or
                    self.int_y < 0 or self.int_y > self.L or
                    self.int_z < 0 or self.int_z > self.H):
                self.interceptor_active = False

        # 3D distance to target
        current_d_MT = float(np.sqrt(
            (self.target_x - self.x)**2 +
            (self.target_y - self.y)**2 +
            (self.target_z - self.z)**2
        ))

        # Reward — closing distance to target
        reward = (self.previous_d_MT - current_d_MT) / self.D_3D
        self.previous_d_MT = current_d_MT

        # Terminal: hit target
        if current_d_MT < self.r_kill:
            reward    += 100.0
            terminated = True
            info["outcome"] = "hit"

        # Terminal: intercepted by interceptor
        if not terminated and self.interceptor_active:
            d_MI = float(np.sqrt(
                (self.int_x - self.x)**2 +
                (self.int_y - self.y)**2 +
                (self.int_z - self.z)**2
            ))
            if d_MI < self.r_kill_I:
                reward    -= 100.0
                terminated = True
                info["outcome"] = "intercepted"

        # Terminal: missile out of bounds
        if not terminated and (
                self.x < 0 or self.x > self.L or
                self.y < 0 or self.y > self.L or
                self.z < 0 or self.z > self.H):
            reward    -= 50.0
            terminated = True
            info["outcome"] = "oob"

        # Truncated: timeout
        if not terminated and self.step_count >= self.max_steps:
            reward   -= 30.0
            truncated = True
            info["outcome"] = "timeout"

        return self._build_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _build_obs(self) -> np.ndarray:
        # Target obs
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dz = self.target_z - self.z

        d_MT    = float(np.sqrt(dx**2 + dy**2 + dz**2))
        phi_MT  = self._wrap(np.arctan2(dy, dx) - self.theta)
        elev_MT = float(np.arctan2(dz, np.hypot(dx, dy)))

        # Interceptor obs
        if self.interceptor_active:
            ix      = self.int_x - self.x
            iy      = self.int_y - self.y
            iz      = self.int_z - self.z
            d_MI    = float(np.sqrt(ix**2 + iy**2 + iz**2))
            phi_MI  = self._wrap(np.arctan2(iy, ix) - self.theta)
            elev_MI = float(np.arctan2(iz, np.hypot(ix, iy)))
            i_vx    = self.int_vx / self.v_I_max
            i_vy    = self.int_vy / self.v_I_max
            i_vz    = self.int_vz / self.v_I_max
        else:
            # Interceptor neutralised — signal max distance, zero velocity
            d_MI    = self.D_3D
            phi_MI  = 0.0
            elev_MI = 0.0
            i_vx = i_vy = i_vz = 0.0

        return np.array([
            d_MT             / self.D_3D,
            phi_MT           / np.pi,
            elev_MT          / (np.pi / 2),
            self.v           / self.v_M_max,
            self.theta       / np.pi,
            self.gamma       / (np.pi / 2),
            self.fuel,
            self.target_vx   / self.v_T_max,
            self.target_vy   / self.v_T_max,
            self.target_vz   / self.v_T_max,
            d_MI             / self.D_3D,
            phi_MI           / np.pi,
            elev_MI          / (np.pi / 2),
            i_vx,
            i_vy,
            i_vz,
        ], dtype=np.float32)

    @staticmethod
    def _wrap(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)