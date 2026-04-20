import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from target_env import TargetEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 58)
print("  TARGET BRAIN — PPO training")
print("  1,000,000 steps vs scripted missile")
print("  Goal: learn evasive maneuvers")
print("=" * 58)

# Vectorized env — PPO benefits from multiple parallel envs
env = make_vec_env(TargetEnv, n_envs=4)

model = PPO(
    "MlpPolicy",
    env,
    verbose         = 1,
    n_steps         = 2048,
    batch_size      = 256,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,   # encourages exploration
    learning_rate   = 3e-4,
    policy_kwargs   = dict(net_arch=[256, 256]),
)

ckpt = CheckpointCallback(
    save_freq   = 25_000,
    save_path   = BASE_DIR,
    name_prefix = "target_ppo_ckpt"
)

model.learn(total_timesteps=1_000_000, callback=ckpt)
model.save(os.path.join(BASE_DIR, "target_ppo"))
print("\nTarget brain training complete. Saved to target_ppo.zip")
