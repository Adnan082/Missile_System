import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from missile_env_phase3b import MissileEnvPhase3b

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 58)
print("  PHASE 3B  Stage 3 — 2x Full interceptors (250 m/s)")
print("  Resuming from Stage 2 weights")
print("=" * 58)

MissileEnvPhase3b.v_I           = 250.0
MissileEnvPhase3b.int_spawn_min = 800.0

env3   = MissileEnvPhase3b()
model3 = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase3b_stage2"), env=env3)
model3.verbose = 1

ckpt3 = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = BASE_DIR,
    name_prefix = "missile_sac_phase3b_s3_ckpt"
)
model3.learn(total_timesteps=500_000, callback=ckpt3, reset_num_timesteps=False)
model3.save(os.path.join(BASE_DIR, "missile_sac_phase3b"))
print("\nPhase 3b complete. Saved to missile_sac_phase3b.zip")