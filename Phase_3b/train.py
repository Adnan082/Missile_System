import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from missile_env_phase3b import MissileEnvPhase3b

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================== #
#  PHASE 3B — Stage 1: Slow interceptors (80 m/s, spawn 1500m)       #
#  Goal: missile re-establishes pursuit while handling 2 threats      #
# ================================================================== #
print("=" * 58)
print("  PHASE 3B  Stage 1 — 2x Slow interceptors (80 m/s)")
print("=" * 58)

MissileEnvPhase3b.v_I           = 80.0
MissileEnvPhase3b.int_spawn_min = 1500.0

env1 = MissileEnvPhase3b()

transferred = os.path.join(BASE_DIR, "missile_sac_phase3b_transferred")
if os.path.exists(transferred + ".zip"):
    print("Loading transferred Phase 3a weights...")
    model = SAC.load(transferred, env=env1)
else:
    print("No transferred weights found — starting fresh. Run transfer_weights.py first.")
    model = SAC("MlpPolicy", env1, verbose=1, learning_rate=3e-4)

model.verbose = 1

ckpt1 = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = BASE_DIR,
    name_prefix = "missile_sac_phase3b_s1_ckpt"
)
model.learn(total_timesteps=500_000, callback=ckpt1)
model.save(os.path.join(BASE_DIR, "missile_sac_phase3b_stage1"))
print("\nStage 1 complete. Saved to missile_sac_phase3b_stage1.zip\n")

# ================================================================== #
#  PHASE 3B — Stage 2: Medium interceptors (150 m/s, 1000m)          #
#  Goal: missile learns to prioritise the closer threat               #
# ================================================================== #
print("=" * 58)
print("  PHASE 3B  Stage 2 — 2x Medium interceptors (150 m/s)")
print("=" * 58)

MissileEnvPhase3b.v_I           = 150.0
MissileEnvPhase3b.int_spawn_min = 1000.0

env2   = MissileEnvPhase3b()
model2 = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase3b_stage1"), env=env2)

ckpt2 = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = BASE_DIR,
    name_prefix = "missile_sac_phase3b_s2_ckpt"
)
model2.learn(total_timesteps=500_000, callback=ckpt2, reset_num_timesteps=False)
model2.save(os.path.join(BASE_DIR, "missile_sac_phase3b_stage2"))
print("\nStage 2 complete. Saved to missile_sac_phase3b_stage2.zip\n")

# ================================================================== #
#  PHASE 3B — Stage 3: Full interceptors (250 m/s, 800m)             #
#  Goal: missile masters simultaneous pursuit + dual evasion          #
# ================================================================== #
print("=" * 58)
print("  PHASE 3B  Stage 3 — 2x Full interceptors (250 m/s)")
print("=" * 58)

MissileEnvPhase3b.v_I           = 250.0
MissileEnvPhase3b.int_spawn_min = 800.0

env3   = MissileEnvPhase3b()
model3 = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase3b_stage2"), env=env3)

ckpt3 = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = BASE_DIR,
    name_prefix = "missile_sac_phase3b_s3_ckpt"
)
model3.learn(total_timesteps=500_000, callback=ckpt3, reset_num_timesteps=False)
model3.save(os.path.join(BASE_DIR, "missile_sac_phase3b"))
print("\nPhase 3b complete. Saved to missile_sac_phase3b.zip")