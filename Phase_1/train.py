from stable_baselines3 import SAC
from missile_env import MissileEnv

env = MissileEnv()
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
model.save("missile_sac_phase1")
print("Training complete. Model saved to missile_sac_phase1.zip")
