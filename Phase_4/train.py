import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase_3b"))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from missile_env_f16 import MissileEnvF16

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PHASE3D    = os.path.join(BASE_DIR, "..", "Phase_3d", "missile_sac_phase3d")

print("=" * 60)
print("  PHASE 4 — F-16 Specialisation")
print("  Loads Phase 3d weights (3.5M steps)")
print("  Continues training against F-16 maneuvers")
print("  Same 22-dim obs — full weight transfer")
print("=" * 60)

# ── Stage 1 — Jinking only, slower F-16 (500k steps) ────────────────────
print("\n[Stage 1]  F-16 JINKS only  |  v_T_max=350  |  r_kill=300")
MissileEnvF16.v_T_max           = 350.0
MissileEnvF16.r_kill            = 300.0
MissileEnvF16.v_I               = 250.0
MissileEnvF16.int_spawn_min     = 800.0
MissileEnvF16.maneuvers_enabled = [MissileEnvF16.JINK]

env   = MissileEnvF16()
model = SAC.load(PHASE3D, env=env)
model.verbose = 1

ckpt = CheckpointCallback(
    save_freq   = 50_000,
    save_path   = BASE_DIR,
    name_prefix = "missile_sac_phase4_ckpt"
)
model.learn(total_timesteps=500_000, callback=ckpt, reset_num_timesteps=False)
model.save(os.path.join(BASE_DIR, "missile_sac_phase4_s1"))
print("  Stage 1 complete — saved missile_sac_phase4_s1")

# ── Stage 2 — Beam + Jink, faster F-16 (500k steps) ────────────────────
print("\n[Stage 2]  BEAM + JINK  |  v_T_max=400  |  r_kill=250")
MissileEnvF16.v_T_max           = 400.0
MissileEnvF16.r_kill            = 250.0
MissileEnvF16.maneuvers_enabled = [MissileEnvF16.BEAM, MissileEnvF16.JINK]

env   = MissileEnvF16()
model = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase4_s1"), env=env)
model.verbose = 1

model.learn(total_timesteps=500_000, callback=ckpt, reset_num_timesteps=False)
model.save(os.path.join(BASE_DIR, "missile_sac_phase4_s2"))
print("  Stage 2 complete — saved missile_sac_phase4_s2")

# ── Stage 3 — All maneuvers, full F-16 (1M steps) ───────────────────────
print("\n[Stage 3]  ALL MANEUVERS (BEAM+JINK+BREAK+RUN)  |  v_T_max=450  |  r_kill=200")
MissileEnvF16.v_T_max           = 450.0
MissileEnvF16.r_kill            = 200.0
MissileEnvF16.maneuvers_enabled = [
    MissileEnvF16.BEAM,
    MissileEnvF16.JINK,
    MissileEnvF16.BREAK,
    MissileEnvF16.RUN,
]

env   = MissileEnvF16()
model = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase4_s2"), env=env)
model.verbose = 1

model.learn(total_timesteps=1_000_000, callback=ckpt, reset_num_timesteps=False)
model.save(os.path.join(BASE_DIR, "missile_sac_phase4"))
print("\n  Phase 4 complete — saved missile_sac_phase4.zip")
print("  Total training: ~5.5M steps (3.5M Phase3d + 2M Phase4)")
