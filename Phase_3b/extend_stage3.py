import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from missile_env_phase3b import MissileEnvPhase3b

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 58)
print("  PHASE 3B  Stage 3 extended — 500k more steps")
print("  Loading final phase3b model...")
print("=" * 58)

MissileEnvPhase3b.v_I           = 250.0
MissileEnvPhase3b.int_spawn_min = 800.0

env   = MissileEnvPhase3b()
model = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase3b"), env=env)
model.verbose = 1

ckpt = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = BASE_DIR,
    name_prefix = "missile_sac_phase3b_ext_ckpt"
)
model.learn(total_timesteps=1_000_000, callback=ckpt, reset_num_timesteps=False)
model.save(os.path.join(BASE_DIR, "missile_sac_phase3b"))
print("\nExtended training complete. Saved to missile_sac_phase3b.zip")
