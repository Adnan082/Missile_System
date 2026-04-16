import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from missile_env_phase2 import MissileEnvPhase2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fuel enabled for Phase 2b
MissileEnvPhase2.max_burn_time = 150.0

train_env = MissileEnvPhase2()

# Load Phase 2b checkpoint
checkpoint_path = os.path.join(BASE_DIR, "missile_sac_phase2b_ckpt_1500000_steps")
model = SAC.load(checkpoint_path, env=train_env)
print("Checkpoint loaded. Resuming Phase 2b from 500k steps...")

checkpoint = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = BASE_DIR,
    name_prefix = "missile_sac_phase2b_ckpt"
)

# Run remaining 500k steps
model.learn(
    total_timesteps     = 500_000,
    callback            = checkpoint,
    reset_num_timesteps = False
)

model.save(os.path.join(BASE_DIR, "missile_sac_phase2b"))
print("Done. Saved to missile_sac_phase2b.zip")