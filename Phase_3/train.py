import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from missile_env_phase3 import MissileEnvPhase3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================== #
#  PHASE 3A — Stage 1: Slow interceptor (80 m/s, spawns 1500m away) #
#  Goal: missile learns target pursuit while interceptor is easy     #
# ================================================================== #
print("=" * 58)
print("  PHASE 3A  Stage 1 — Slow interceptor (80 m/s)")
print("=" * 58)

MissileEnvPhase3.v_I           = 80.0
MissileEnvPhase3.int_spawn_min = 1500.0

env1  = MissileEnvPhase3()

transferred = os.path.join(BASE_DIR, "missile_sac_phase3a_transferred")
if os.path.exists(transferred + ".zip"):
    print("Loading transferred Phase 2b weights...")
    model = SAC.load(transferred, env=env1)
else:
    print("No transferred weights found — starting fresh. Run transfer_weights.py first.")
    model = SAC("MlpPolicy", env1, verbose=1, learning_rate=3e-4)

model.verbose = 1

ckpt1 = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = BASE_DIR,
    name_prefix = "missile_sac_phase3a_s1_ckpt"
)
model.learn(total_timesteps=500_000, callback=ckpt1)
model.save(os.path.join(BASE_DIR, "missile_sac_phase3a_stage1"))
print("\nStage 1 complete. Saved to missile_sac_phase3a_stage1.zip\n")

# ================================================================== #
#  PHASE 3A — Stage 2: Medium interceptor (150 m/s, 1000m away)     #
#  Goal: missile begins learning active evasion                      #
# ================================================================== #
print("=" * 58)
print("  PHASE 3A  Stage 2 — Medium interceptor (150 m/s)")
print("=" * 58)

MissileEnvPhase3.v_I           = 150.0
MissileEnvPhase3.int_spawn_min = 1000.0

env2   = MissileEnvPhase3()
model2 = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase3a_stage1"), env=env2)

ckpt2 = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = BASE_DIR,
    name_prefix = "missile_sac_phase3a_s2_ckpt"
)
model2.learn(total_timesteps=500_000, callback=ckpt2, reset_num_timesteps=False)
model2.save(os.path.join(BASE_DIR, "missile_sac_phase3a_stage2"))
print("\nStage 2 complete. Saved to missile_sac_phase3a_stage2.zip\n")

# ================================================================== #
#  PHASE 3A — Stage 3: Full interceptor (250 m/s, 800m away)        #
#  Goal: missile masters simultaneous pursuit + evasion              #
# ================================================================== #
print("=" * 58)
print("  PHASE 3A  Stage 3 — Full interceptor (250 m/s)")
print("=" * 58)

MissileEnvPhase3.v_I           = 250.0
MissileEnvPhase3.int_spawn_min = 800.0

env3   = MissileEnvPhase3()
model3 = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase3a_stage2"), env=env3)

ckpt3 = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = BASE_DIR,
    name_prefix = "missile_sac_phase3a_s3_ckpt"
)
model3.learn(total_timesteps=500_000, callback=ckpt3, reset_num_timesteps=False)
model3.save(os.path.join(BASE_DIR, "missile_sac_phase3a"))
print("\nPhase 3a complete. Saved to missile_sac_phase3a.zip")