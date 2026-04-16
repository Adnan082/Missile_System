import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from missile_env_phase2 import MissileEnvPhase2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================== #
#  PHASE 2 — Train without fuel until model masters pursuit           #
# ================================================================== #
print("=" * 50)
print("  PHASE 2 — No fuel (mastering pursuit)")
print("=" * 50)

MissileEnvPhase2.max_burn_time = 999_999.0   # fuel disabled

train_env = MissileEnvPhase2()

model = SAC("MlpPolicy", train_env, verbose=1, learning_rate=3e-4)

checkpoint1 = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = BASE_DIR,
    name_prefix = "missile_sac_phase2_ckpt"
)

model.learn(total_timesteps=1_000_000, callback=checkpoint1)

model.save(os.path.join(BASE_DIR, "missile_sac_phase2"))
print("\nPhase 2 complete. Saved to missile_sac_phase2.zip\n")

# ================================================================== #
#  PHASE 2B — Fine-tune with real fuel                                #
# ================================================================== #
print("=" * 50)
print("  PHASE 2B — Fuel enabled (fine-tuning)")
print("=" * 50)

MissileEnvPhase2.max_burn_time = 150.0   # real fuel active

train_env2 = MissileEnvPhase2()

model2 = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase2"), env=train_env2)
print("Phase 2 model loaded. Fine-tuning with fuel...")

checkpoint2 = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = BASE_DIR,
    name_prefix = "missile_sac_phase2b_ckpt"
)

model2.learn(
    total_timesteps     = 1_000_000,
    callback            = checkpoint2,
    reset_num_timesteps = False
)

model2.save(os.path.join(BASE_DIR, "missile_sac_phase2b"))
print("\nPhase 2b complete. Saved to missile_sac_phase2b.zip")
