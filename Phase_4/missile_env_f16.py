import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase_3b"))

import numpy as np
from missile_env_phase3b import MissileEnvPhase3b


class MissileEnvF16(MissileEnvPhase3b):
    """
    Phase 4 — F-16 target with realistic combat maneuvers.

    Inherits the full 22-dim observation space and missile physics
    from Phase 3b/3d so Phase 3d weights transfer directly.

    Target upgrade: constant-velocity bounce → F-16 maneuver state machine
        BEAM      : fly 90 deg perpendicular to missile LOS (default)
        JINK      : rapid direction reversal every 2s (breaks PN prediction)
        BREAK_TURN: max-G hard turn when missile inside 2km
        RUN       : full afterburner escape when missile far (>5km)

    F-16 physics (4th generation subsonic combat envelope):
        v_T_min      = 250 m/s   (clean config minimum)
        v_T_max      = 450 m/s   (transonic combat — no full supersonic yet)
        omega_max    = 1.5 rad/s (9G sustained at ~400 m/s)
        pitch_max    = 0.8 rad/s
        afterburner  = +80 m/s burst (limited duration)

    Curriculum override (set in train.py before instantiation):
        Phase 4a — JINK only,         v_T_max=350, r_kill=300
        Phase 4b — BEAM + JINK,       v_T_max=400, r_kill=250
        Phase 4c — all maneuvers,     v_T_max=450, r_kill=200
    """

    # ── F-16 speed envelope ──────────────────────────────────────────────
    v_T_min = 250.0
    v_T_max = 350.0          # overridden per phase in train.py

    # ── F-16 agility ─────────────────────────────────────────────────────
    f16_omega_max = 1.5      # rad/s yaw  (9G-class turn)
    f16_pitch_max = 0.8      # rad/s pitch
    afterburner_dv = 80.0    # m/s burst speed increase
    afterburner_dur = 30     # steps (3 real seconds at dt=0.1)

    # ── Maneuver trigger distances ────────────────────────────────────────
    BREAK_DIST = 2_000.0     # below this → break turn (highest priority)
    RUN_DIST   = 5_000.0     # above this → afterburner escape

    # ── Maneuver IDs ─────────────────────────────────────────────────────
    BEAM  = 0
    JINK  = 1
    BREAK = 2
    RUN   = 3

    # ── Which maneuvers are active (overridden per phase) ─────────────────
    maneuvers_enabled = [BEAM, JINK, BREAK, RUN]   # Phase 4c default

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

        # F-16 angular state (needed for rate-limited turning)
        self.f16_theta = 0.0
        self.f16_gamma = 0.0
        self.f16_v     = 0.0

        # Maneuver state machine
        self.maneuver_state = self.BEAM
        self.maneuver_timer = 0          # steps remaining in current jink direction
        self.jink_dir       = 1.0        # +1 or -1 lateral turn
        self.ab_timer       = 0          # afterburner steps remaining

    # ------------------------------------------------------------------ #
    #  reset — initialise F-16 state on top of parent reset               #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Derive F-16 heading/climb from the velocity set by parent reset
        speed = float(np.sqrt(
            self.target_vx**2 + self.target_vy**2 + self.target_vz**2
        ))
        self.f16_v     = speed
        self.f16_theta = float(np.arctan2(self.target_vy, self.target_vx))
        self.f16_gamma = float(np.arctan2(
            self.target_vz,
            np.hypot(self.target_vx, self.target_vy)
        ))

        self.maneuver_state = self.BEAM
        self.maneuver_timer = 0
        self.jink_dir       = float(self.np_random.choice([-1.0, 1.0]))
        self.ab_timer       = 0

        return obs, info

    # ------------------------------------------------------------------ #
    #  step — same as parent but replaces target movement with F-16 AI    #
    # ------------------------------------------------------------------ #
    def step(self, action):
        terminated = False
        truncated  = False
        info       = {}

        # ── Missile action ────────────────────────────────────────────────
        a_thrust = ((action[0] + 1.0) / 2.0) * self.a_max
        a_yaw    = action[1] * self.omega_max
        a_pitch  = action[2] * self.pitch_max

        fuel_burn  = (a_thrust / self.a_max) * self.dt / self.max_burn_time
        self.fuel  = float(max(0.0, self.fuel - fuel_burn))
        if self.fuel <= 0.0:
            a_thrust = 0.0

        self.v     = float(np.clip(self.v + a_thrust * self.dt, 0, self.v_M_max))
        self.theta = self._wrap(self.theta + a_yaw   * self.dt)
        self.gamma = float(np.clip(self.gamma + a_pitch * self.dt, -np.pi / 2, np.pi / 2))

        self.x += self.v * np.cos(self.gamma) * np.cos(self.theta) * self.dt
        self.y += self.v * np.cos(self.gamma) * np.sin(self.theta) * self.dt
        self.z += self.v * np.sin(self.gamma) * self.dt
        self.step_count += 1

        # ── F-16 maneuver decision ────────────────────────────────────────
        d_MT = float(np.sqrt(
            (self.target_x - self.x)**2 +
            (self.target_y - self.y)**2 +
            (self.target_z - self.z)**2
        ))
        self._f16_decide_maneuver(d_MT)
        self._f16_execute_maneuver(d_MT)

        # Bounce F-16 off arena walls
        if self.target_x <= 0 or self.target_x >= self.L:
            self.f16_theta = self._wrap(np.pi - self.f16_theta)
            self.target_x  = float(np.clip(self.target_x, 0, self.L))
        if self.target_y <= 0 or self.target_y >= self.L:
            self.f16_theta = self._wrap(-self.f16_theta)
            self.target_y  = float(np.clip(self.target_y, 0, self.L))
        if self.target_z <= 0 or self.target_z >= self.H:
            self.f16_gamma = -self.f16_gamma
            self.target_z  = float(np.clip(self.target_z, 0, self.H))

        # Rebuild target velocity from updated F-16 heading (used in obs)
        self.target_vx = self.f16_v * np.cos(self.f16_gamma) * np.cos(self.f16_theta)
        self.target_vy = self.f16_v * np.cos(self.f16_gamma) * np.sin(self.f16_theta)
        self.target_vz = self.f16_v * np.sin(self.f16_gamma)

        # ── Interceptors ──────────────────────────────────────────────────
        self._move_interceptor(1)
        self._move_interceptor(2)

        # ── Distance after moves ─────────────────────────────────────────
        current_d_MT = float(np.sqrt(
            (self.target_x - self.x)**2 +
            (self.target_y - self.y)**2 +
            (self.target_z - self.z)**2
        ))

        # ── Reward ───────────────────────────────────────────────────────
        reward = (self.previous_d_MT - current_d_MT) / self.D_3D
        self.previous_d_MT = current_d_MT

        # Terminal: hit F-16
        if current_d_MT < self.r_kill:
            reward    += 100.0
            terminated = True
            info["outcome"] = "hit"

        # Terminal: intercepted
        if not terminated:
            for active, ix, iy, iz in [
                (self.int1_active, self.int1_x, self.int1_y, self.int1_z),
                (self.int2_active, self.int2_x, self.int2_y, self.int2_z),
            ]:
                if active:
                    d_MI = float(np.sqrt(
                        (ix - self.x)**2 + (iy - self.y)**2 + (iz - self.z)**2
                    ))
                    if d_MI < self.r_kill_I:
                        reward    -= 100.0
                        terminated = True
                        info["outcome"] = "intercepted"
                        break

        # Terminal: out of bounds
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
    #  F-16 maneuver state machine                                         #
    # ------------------------------------------------------------------ #
    def _f16_decide_maneuver(self, d_MT: float):
        """
        Priority-based maneuver selection.
        BREAK_TURN overrides everything when missile is inside 2km.
        RUN triggers when missile is far and fuel allows.
        BEAM/JINK alternate otherwise.
        """
        if self.BREAK in self.maneuvers_enabled and d_MT < self.BREAK_DIST:
            self.maneuver_state = self.BREAK
            return

        if self.RUN in self.maneuvers_enabled and d_MT > self.RUN_DIST:
            if self.maneuver_state != self.RUN:
                self.maneuver_state = self.RUN
                self.ab_timer = self.afterburner_dur
            return

        # Alternate BEAM / JINK every 20 steps (2 seconds)
        if self.maneuver_state in (self.BREAK, self.RUN):
            self.maneuver_state = self.BEAM
            self.maneuver_timer = 20

        if self.maneuver_timer <= 0:
            if self.JINK in self.maneuvers_enabled and self.maneuver_state == self.BEAM:
                self.maneuver_state = self.JINK
                self.maneuver_timer = 20
                self.jink_dir = -self.jink_dir   # reverse jink direction
            elif self.BEAM in self.maneuvers_enabled:
                self.maneuver_state = self.BEAM
                self.maneuver_timer = 20
            else:
                self.maneuver_state = self.JINK
                self.maneuver_timer = 20

        self.maneuver_timer -= 1

    def _f16_execute_maneuver(self, d_MT: float):
        """Apply the current maneuver to update F-16 heading and position."""

        if self.maneuver_state == self.BEAM:
            self._do_beam()

        elif self.maneuver_state == self.JINK:
            self._do_jink()

        elif self.maneuver_state == self.BREAK:
            self._do_break_turn()

        elif self.maneuver_state == self.RUN:
            self._do_run()

        # Move F-16 position
        self.target_x += self.f16_v * np.cos(self.f16_gamma) * np.cos(self.f16_theta) * self.dt
        self.target_y += self.f16_v * np.cos(self.f16_gamma) * np.sin(self.f16_theta) * self.dt
        self.target_z += self.f16_v * np.sin(self.f16_gamma) * self.dt

    # ── Individual maneuvers ─────────────────────────────────────────────

    def _do_beam(self):
        """
        Beam maneuver — turn 90 deg to missile LOS.
        Maximises angular tracking rate the missile must match.
        """
        los = float(np.arctan2(self.y - self.target_y, self.x - self.target_x))
        perp_left  = self._wrap(los + np.pi / 2)
        perp_right = self._wrap(los - np.pi / 2)
        err_l = abs(self._wrap(perp_left  - self.f16_theta))
        err_r = abs(self._wrap(perp_right - self.f16_theta))
        desired = perp_left if err_l < err_r else perp_right

        self.f16_theta = self._wrap(
            self.f16_theta +
            float(np.clip(self._wrap(desired - self.f16_theta) / self.dt,
                          -self.f16_omega_max, self.f16_omega_max)) * self.dt
        )
        # Fly level during beam
        self.f16_gamma = float(np.clip(
            self.f16_gamma - float(np.clip(self.f16_gamma / self.dt,
                                           -self.f16_pitch_max, self.f16_pitch_max)) * self.dt,
            -np.pi / 2, np.pi / 2
        ))
        self.f16_v = float(np.clip(self.f16_v, self.v_T_min, self.v_T_max))

    def _do_jink(self):
        """
        Jinking — hard lateral turn in alternating directions.
        Breaks PN by making target path unpredictable.
        """
        desired = self._wrap(self.f16_theta + self.jink_dir * np.pi / 2)
        self.f16_theta = self._wrap(
            self.f16_theta +
            float(np.clip(self._wrap(desired - self.f16_theta) / self.dt,
                          -self.f16_omega_max, self.f16_omega_max)) * self.dt
        )
        self.f16_v = float(np.clip(self.f16_v, self.v_T_min, self.v_T_max))

    def _do_break_turn(self):
        """
        Break turn — maximum G perpendicular to missile approach.
        Triggered inside 2km — forces missile to overshoot.
        """
        missile_bearing = float(np.arctan2(self.y - self.target_y, self.x - self.target_x))
        # Turn hard perpendicular to incoming missile (choose best side)
        perp_left  = self._wrap(missile_bearing + np.pi / 2)
        perp_right = self._wrap(missile_bearing - np.pi / 2)
        err_l = abs(self._wrap(perp_left  - self.f16_theta))
        err_r = abs(self._wrap(perp_right - self.f16_theta))
        desired = perp_left if err_l < err_r else perp_right

        self.f16_theta = self._wrap(
            self.f16_theta +
            float(np.clip(self._wrap(desired - self.f16_theta) / self.dt,
                          -self.f16_omega_max, self.f16_omega_max)) * self.dt
        )
        # Pull up slightly to add 3D complexity
        desired_gamma = float(np.clip(self.f16_gamma + 0.3, -np.pi / 4, np.pi / 4))
        self.f16_gamma = float(np.clip(
            self.f16_gamma +
            float(np.clip((desired_gamma - self.f16_gamma) / self.dt,
                          -self.f16_pitch_max, self.f16_pitch_max)) * self.dt,
            -np.pi / 2, np.pi / 2
        ))
        self.f16_v = float(np.clip(self.f16_v, self.v_T_min, self.v_T_max))

    def _do_run(self):
        """
        Afterburner escape — max speed directly away from missile.
        Used when missile is far — try to outrun or drain its fuel.
        """
        away = self._wrap(np.arctan2(self.target_y - self.y, self.target_x - self.x))
        self.f16_theta = self._wrap(
            self.f16_theta +
            float(np.clip(self._wrap(away - self.f16_theta) / self.dt,
                          -self.f16_omega_max, self.f16_omega_max)) * self.dt
        )
        # Afterburner burst
        if self.ab_timer > 0:
            self.f16_v = float(np.clip(
                self.f16_v + self.afterburner_dv * self.dt,
                self.v_T_min, self.v_T_max
            ))
            self.ab_timer -= 1
        else:
            self.f16_v = float(np.clip(self.f16_v, self.v_T_min, self.v_T_max))
