import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase_3b"))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from missile_env_phase3b import MissileEnvPhase3b

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH  = os.path.join(BASE_DIR, "missile_sac_phase3d_ckpt_3000000_steps")

print("=" * 58)
print("  PHASE 3D — Resuming from 3M checkpoint")
print("  500,000 steps remaining at full difficulty")
print("=" * 58)

MissileEnvPhase3b.v_I           = 250.0
MissileEnvPhase3b.int_spawn_min = 800.0

env   = MissileEnvPhase3b()
model = SAC.load(CKPT_PATH, env=env)
model.verbose = 1

print(f"Loaded checkpoint from {CKPT_PATH}.zip")

ckpt = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = BASE_DIR,
    name_prefix = "missile_sac_phase3d_ckpt"
)
model.learn(total_timesteps=500_000, callback=ckpt, reset_num_timesteps=False)
model.save(os.path.join(BASE_DIR, "missile_sac_phase3d"))
print("\nPhase 3d complete. Saved to missile_sac_phase3d.zip")
