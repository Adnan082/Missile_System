import os
from stable_baselines3 import PPO
from target_env import TargetEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = PPO.load(os.path.join(BASE_DIR, "target_ppo"))
env   = TargetEnv()

hit = survived = oob = 0
episodes = 100

for ep in range(episodes):
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            outcome = info.get("outcome", "hit")
            if outcome == "survived": survived += 1
            elif outcome == "oob":    oob      += 1
            else:                     hit      += 1
            break

print("=" * 45)
print("  TARGET BRAIN — Evaluation vs scripted missile")
print("=" * 45)
print(f"  Survived (missile missed) : {survived}/100  ({survived}%)")
print(f"  Hit by missile            : {hit}/100  ({hit}%)")
print(f"  Out of bounds             : {oob}/100  ({oob}%)")
print("=" * 45)
