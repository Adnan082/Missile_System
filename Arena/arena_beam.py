import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase_3b"))

import numpy as np
from stable_baselines3 import SAC
from missile_env_phase3b import MissileEnvPhase3b

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MISSILE_PATH = os.path.join(BASE_DIR, "..", "Phase_3d", "missile_sac_phase3d")

print("=" * 60)
print("  ARENA — SAC Missile  vs  Classical Beam Maneuver Target")
print("  Missile brain: FROZEN SAC (Phase 3d)")
print("  Target brain : SCRIPTED beam maneuver (no learning)")
print("=" * 60)

missile_model = SAC.load(MISSILE_PATH)

L         = MissileEnvPhase3b.L
H         = MissileEnvPhase3b.H
D_3D      = MissileEnvPhase3b.D_3D
dt        = MissileEnvPhase3b.dt
v_M_max   = MissileEnvPhase3b.v_M_max
r_kill    = MissileEnvPhase3b.r_kill
r_kill_I  = MissileEnvPhase3b.r_kill_I
max_steps = MissileEnvPhase3b.max_steps

v_T_min      = 200.0
v_T_max      = 300.0
omega_max    = 4.0
pitch_max    = 2.0
v_I          = 250.0
v_I_max      = 250.0
int_spawn_min = 800.0

episodes     = 100
missile_hits = target_survived = intercepted = oob = 0
rng = np.random.default_rng(42)


def wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def beam_maneuver(tx, ty, tz, ttheta, tgamma, tv, mx, my, mz):
    """
    Beam maneuver — fly 90 degrees perpendicular to the missile LOS.

    The missile LOS is the vector from target to missile.
    Flying perpendicular maximises the angular rate the missile must
    track, causing proportional-navigation and pursuit guidance to
    saturate and miss.

    Choose whichever perpendicular direction requires the smallest
    heading change (so the target doesn't waste time reversing).
    """
    # Horizontal LOS angle from target toward missile
    los_angle = np.arctan2(my - ty, mx - tx)

    # Two perpendicular candidates (left/right of LOS)
    perp_left  = wrap(los_angle + np.pi / 2)
    perp_right = wrap(los_angle - np.pi / 2)

    # Pick whichever requires the smaller turn
    err_left  = abs(wrap(perp_left  - ttheta))
    err_right = abs(wrap(perp_right - ttheta))
    desired_heading = perp_left if err_left < err_right else perp_right

    # Turn toward desired heading at max yaw rate
    heading_error = wrap(desired_heading - ttheta)
    a_yaw   = float(np.clip(heading_error / dt, -omega_max, omega_max))

    # Fly level — beam is a horizontal maneuver
    a_pitch = float(np.clip(-tgamma / dt, -pitch_max, pitch_max))

    # Full throttle to maximise lateral displacement
    dv = 20.0

    return a_yaw, a_pitch, dv


for ep in range(episodes):

    # --- Spawn missile ---
    mx = rng.uniform(0, L); my = rng.uniform(0, L); mz = rng.uniform(200, 2000)
    mv     = 100.0
    mtheta = rng.uniform(-np.pi, np.pi)
    mgamma = rng.uniform(-np.pi / 12, np.pi / 12)
    mfuel  = 1.0

    # --- Spawn target (min 300m from missile) ---
    while True:
        tx = rng.uniform(0, L); ty = rng.uniform(0, L); tz = rng.uniform(200, 2500)
        if np.sqrt((tx - mx)**2 + (ty - my)**2 + (tz - mz)**2) >= 300:
            break

    tv     = rng.uniform(v_T_min, v_T_max)
    ttheta = rng.uniform(-np.pi, np.pi)
    tgamma = rng.uniform(-np.pi / 12, np.pi / 12)

    # --- Spawn 2 interceptors ---
    while True:
        i1x = rng.uniform(0, L); i1y = rng.uniform(0, L); i1z = rng.uniform(200, 2000)
        if np.sqrt((i1x - mx)**2 + (i1y - my)**2 + (i1z - mz)**2) >= int_spawn_min:
            break
    while True:
        i2x = rng.uniform(0, L); i2y = rng.uniform(0, L); i2z = rng.uniform(200, 2000)
        d_m  = np.sqrt((i2x - mx)**2 + (i2y - my)**2 + (i2z - mz)**2)
        d_12 = np.sqrt((i2x - i1x)**2 + (i2y - i1y)**2 + (i2z - i1z)**2)
        if d_m >= int_spawn_min and d_12 >= 500:
            break

    i1vx = i1vy = i1vz = i2vx = i2vy = i2vz = 0.0
    i1_active = i2_active = True
    outcome = "timeout"

    for step in range(max_steps):

        # ── Build missile observation (22-dim) ──
        dx = tx - mx; dy = ty - my; dz = tz - mz
        d_MT    = float(np.sqrt(dx**2 + dy**2 + dz**2))
        phi_MT  = wrap(np.arctan2(dy, dx) - mtheta)
        elev_MT = float(np.arctan2(dz, np.hypot(dx, dy)))

        if i1_active:
            ix = i1x - mx; iy = i1y - my; iz = i1z - mz
            d_MI1    = float(np.sqrt(ix**2 + iy**2 + iz**2))
            phi_MI1  = wrap(np.arctan2(iy, ix) - mtheta)
            elev_MI1 = float(np.arctan2(iz, np.hypot(ix, iy)))
            n_i1vx = i1vx / v_I_max; n_i1vy = i1vy / v_I_max; n_i1vz = i1vz / v_I_max
        else:
            d_MI1 = D_3D; phi_MI1 = elev_MI1 = n_i1vx = n_i1vy = n_i1vz = 0.0

        if i2_active:
            ix = i2x - mx; iy = i2y - my; iz = i2z - mz
            d_MI2    = float(np.sqrt(ix**2 + iy**2 + iz**2))
            phi_MI2  = wrap(np.arctan2(iy, ix) - mtheta)
            elev_MI2 = float(np.arctan2(iz, np.hypot(ix, iy)))
            n_i2vx = i2vx / v_I_max; n_i2vy = i2vy / v_I_max; n_i2vz = i2vz / v_I_max
        else:
            d_MI2 = D_3D; phi_MI2 = elev_MI2 = n_i2vx = n_i2vy = n_i2vz = 0.0

        missile_obs = np.array([
            d_MT / D_3D,   phi_MT / np.pi,   elev_MT / (np.pi / 2),
            mv / v_M_max,  mtheta / np.pi,   mgamma / (np.pi / 2),  mfuel,
            0.0, 0.0, 0.0,
            d_MI1 / D_3D,  phi_MI1 / np.pi,  elev_MI1 / (np.pi / 2),
            n_i1vx, n_i1vy, n_i1vz,
            d_MI2 / D_3D,  phi_MI2 / np.pi,  elev_MI2 / (np.pi / 2),
            n_i2vx, n_i2vy, n_i2vz,
        ], dtype=np.float32)

        # ── SAC missile decides ──
        m_action, _ = missile_model.predict(missile_obs, deterministic=True)

        a_thrust = ((float(m_action[0]) + 1.0) / 2.0) * 40.0
        a_yaw    = float(m_action[1]) * 3.0
        a_pitch  = float(m_action[2]) * 1.5

        fuel_burn = (a_thrust / 40.0) * dt / 150.0
        mfuel = max(0.0, mfuel - fuel_burn)
        if mfuel <= 0:
            a_thrust = 0.0

        mv     = float(np.clip(mv + a_thrust * dt, 0, v_M_max))
        mtheta = wrap(mtheta + a_yaw * dt)
        mgamma = float(np.clip(mgamma + a_pitch * dt, -np.pi / 2, np.pi / 2))
        mx += mv * np.cos(mgamma) * np.cos(mtheta) * dt
        my += mv * np.cos(mgamma) * np.sin(mtheta) * dt
        mz += mv * np.sin(mgamma) * dt

        # ── Beam maneuver target ──
        a_yaw_t, a_pitch_t, dv_t = beam_maneuver(
            tx, ty, tz, ttheta, tgamma, tv, mx, my, mz
        )
        tv     = float(np.clip(tv + dv_t * dt, v_T_min, v_T_max))
        ttheta = wrap(ttheta + a_yaw_t * dt)
        tgamma = float(np.clip(tgamma + a_pitch_t * dt, -np.pi / 2, np.pi / 2))
        tx += tv * np.cos(tgamma) * np.cos(ttheta) * dt
        ty += tv * np.cos(tgamma) * np.sin(ttheta) * dt
        tz += tv * np.sin(tgamma) * dt

        # Bounce target off walls
        if tx <= 0 or tx >= L: ttheta = wrap(np.pi - ttheta); tx = float(np.clip(tx, 0, L))
        if ty <= 0 or ty >= L: ttheta = wrap(-ttheta);        ty = float(np.clip(ty, 0, L))
        if tz <= 0 or tz >= H: tgamma = -tgamma;              tz = float(np.clip(tz, 0, H))

        # ── Move interceptors (pure pursuit) ──
        for i in [1, 2]:
            active = i1_active if i == 1 else i2_active
            if not active:
                continue
            ix_ = (i1x if i == 1 else i2x)
            iy_ = (i1y if i == 1 else i2y)
            iz_ = (i1z if i == 1 else i2z)
            dx_ = mx - ix_; dy_ = my - iy_; dz_ = mz - iz_
            dist_ = float(np.sqrt(dx_**2 + dy_**2 + dz_**2))
            if dist_ > 1e-6:
                vx_ = v_I * dx_ / dist_; vy_ = v_I * dy_ / dist_; vz_ = v_I * dz_ / dist_
            else:
                vx_ = vy_ = vz_ = 0.0
            nx_ = ix_ + vx_ * dt; ny_ = iy_ + vy_ * dt; nz_ = iz_ + vz_ * dt
            if nx_ < 0 or nx_ > L or ny_ < 0 or ny_ > L or nz_ < 0 or nz_ > H:
                if i == 1: i1_active = False
                else:      i2_active = False
                continue
            if i == 1: i1x, i1y, i1z, i1vx, i1vy, i1vz = nx_, ny_, nz_, vx_, vy_, vz_
            else:      i2x, i2y, i2z, i2vx, i2vy, i2vz = nx_, ny_, nz_, vx_, vy_, vz_

        # ── Check outcomes ──
        current_d_MT = float(np.sqrt((tx - mx)**2 + (ty - my)**2 + (tz - mz)**2))

        if current_d_MT < r_kill:
            outcome = "missile_hit"; break

        for active, ix_, iy_, iz_ in [
            (i1_active, i1x, i1y, i1z),
            (i2_active, i2x, i2y, i2z)
        ]:
            if active and np.sqrt((ix_ - mx)**2 + (iy_ - my)**2 + (iz_ - mz)**2) < r_kill_I:
                outcome = "intercepted"; break

        if outcome != "timeout": break

        if mx < 0 or mx > L or my < 0 or my > L or mz < 0 or mz > H:
            outcome = "oob"; break

    if outcome == "missile_hit":   missile_hits    += 1
    elif outcome == "intercepted": intercepted     += 1
    elif outcome == "oob":         oob             += 1
    else:                          target_survived += 1

    if (ep + 1) % 10 == 0:
        print(f"  Episode {ep+1:3d}/100 — {outcome}")

print()
print("=" * 60)
print("  RESULTS — SAC Missile  vs  Classical Beam Maneuver")
print("=" * 60)
print(f"  Missile hit target   : {missile_hits}/100  ({missile_hits}%)")
print(f"  Missile intercepted  : {intercepted}/100  ({intercepted}%)")
print(f"  Missile out of bounds: {oob}/100  ({oob}%)")
print(f"  Target survived      : {target_survived}/100  ({target_survived}%)")
print("=" * 60)
print(f"  Missile win rate     : {missile_hits}%")
print(f"  Target  win rate     : {target_survived}%")
print("=" * 60)
print()
print("  Compare against arena.py (SAC vs PPO target) to see")
print("  whether RL evasion outperforms classical beam maneuver.")
print("=" * 60)
