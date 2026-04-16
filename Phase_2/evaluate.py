import os
from stable_baselines3 import SAC
from missile_env_phase2 import MissileEnvPhase2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model    = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase2b"))
env      = MissileEnvPhase2()

wins         = 0
total_steps  = 0
total_reward = 0
episodes     = 100

for episode in range(episodes):
    obs, _ = env.reset()
    episode_reward = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            if reward >= 100.0:
                wins += 1
            total_steps  += env.step_count
            total_reward += episode_reward
            break

print("=" * 40)
print(f"  Hit rate  : {wins}/{episodes} ({wins}%)")
print(f"  Avg steps : {total_steps  / episodes:.1f}")
print(f"  Avg reward: {total_reward / episodes:.2f}")
print("=" * 40)
