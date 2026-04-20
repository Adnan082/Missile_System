import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase_3b"))

from stable_baselines3 import SAC
from missile_env_f16 import MissileEnvF16

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Full difficulty evaluation
MissileEnvF16.v_T_max           = 450.0
MissileEnvF16.r_kill            = 200.0
MissileEnvF16.v_I               = 250.0
MissileEnvF16.int_spawn_min     = 800.0
MissileEnvF16.maneuvers_enabled = [
    MissileEnvF16.BEAM,
    MissileEnvF16.JINK,
    MissileEnvF16.BREAK,
    MissileEnvF16.RUN,
]

model = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase4"))
env   = MissileEnvF16()

hit = intercepted = oob = timeout = 0
episodes = 100

print("=" * 55)
print("  PHASE 4 — Evaluation vs Full F-16 Maneuver Suite")
print("  (BEAM + JINK + BREAK TURN + AFTERBURNER RUN)")
print("=" * 55)

maneuver_hits = {MissileEnvF16.BEAM: 0, MissileEnvF16.JINK: 0,
                 MissileEnvF16.BREAK: 0, MissileEnvF16.RUN: 0}

for ep in range(episodes):
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            outcome = info.get("outcome", "timeout")
            if outcome == "hit":          hit          += 1
            elif outcome == "intercepted": intercepted += 1
            elif outcome == "oob":         oob         += 1
            else:                          timeout     += 1
            break

print(f"  Hit F-16           : {hit}/100  ({hit}%)")
print(f"  Intercepted        : {intercepted}/100  ({intercepted}%)")
print(f"  Out of bounds      : {oob}/100  ({oob}%)")
print(f"  Timeout (F-16 won) : {timeout}/100  ({timeout}%)")
print("=" * 55)
print(f"  Missile win rate   : {hit}%")
print(f"  F-16 survival rate : {timeout}%")
print("=" * 55)
