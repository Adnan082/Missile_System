import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase_3b"))

from stable_baselines3 import SAC
from missile_env_phase3b import MissileEnvPhase3b

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MissileEnvPhase3b.v_I           = 250.0
MissileEnvPhase3b.int_spawn_min = 800.0

model = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase3c"))
env   = MissileEnvPhase3b()

hits = intercepted = oob = timeout = 0
episodes = 100

for ep in range(episodes):
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            outcome = info.get("outcome", "timeout")
            if outcome == "hit":         hits += 1
            elif outcome == "intercepted": intercepted += 1
            elif outcome == "oob":         oob += 1
            else:                          timeout += 1
            break

print("=" * 45)
print(f"  Target hit    : {hits}/100  ({hits}%)")
print(f"  Intercepted   : {intercepted}/100  ({intercepted}%)")
print(f"  Out of bounds : {oob}/100  ({oob}%)")
print(f"  Timeout       : {timeout}/100  ({timeout}%)")
print("=" * 45)
