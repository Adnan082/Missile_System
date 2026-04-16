import os
from stable_baselines3 import SAC
from missile_env_phase3 import MissileEnvPhase3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MissileEnvPhase3.v_I           = 250.0
MissileEnvPhase3.int_spawn_min = 800.0

model    = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase3a"))
env      = MissileEnvPhase3()

hits        = 0
intercepted = 0
oob         = 0
timeout     = 0
episodes    = 100

for ep in range(episodes):
    obs, _ = env.reset()
    ep_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        if terminated or truncated:
            outcome = info.get("outcome", "timeout")
            if outcome == "hit":
                hits += 1
            elif outcome == "intercepted":
                intercepted += 1
            elif outcome == "oob":
                oob += 1
            else:
                timeout += 1
            break

print("=" * 45)
print(f"  Target hit    : {hits}/100  ({hits}%)")
print(f"  Intercepted   : {intercepted}/100  ({intercepted}%)")
print(f"  Out of bounds : {oob}/100  ({oob}%)")
print(f"  Timeout       : {timeout}/100  ({timeout}%)")
print("=" * 45)