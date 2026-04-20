"""
Weight surgery: transfer Phase 3a brain into Phase 3b network.

Phase 3a obs = 16-dim  →  Phase 3b obs = 22-dim (6 new interceptor-2 inputs)
Hidden layers (64→64) and output layer are identical size in both models,
so we copy them directly. For the first layer we copy the first 16 input
weights and zero-initialise the 6 new interceptor-2 inputs.

Result: Phase 3b starts knowing how to fly, evade one interceptor, and
hit the target. The second interceptor inputs start at zero influence
and are learned from experience — true curriculum learning.
"""

import os
import torch
from stable_baselines3 import SAC
from missile_env_phase3b import MissileEnvPhase3b

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PHASE3A_PATH = os.path.join(BASE_DIR, "..", "Phase_3", "missile_sac_phase3a")
OUT_PATH     = os.path.join(BASE_DIR, "missile_sac_phase3b_transferred")

# ------------------------------------------------------------------ #
#  Load Phase 3a model (16-dim obs)                                   #
# ------------------------------------------------------------------ #
print("Loading Phase 3a model...")
phase3a = SAC.load(PHASE3A_PATH)

# ------------------------------------------------------------------ #
#  Create fresh Phase 3b model (22-dim obs)                           #
# ------------------------------------------------------------------ #
print("Creating Phase 3b model...")
MissileEnvPhase3b.v_I           = 80.0
MissileEnvPhase3b.int_spawn_min = 1500.0

env3b   = MissileEnvPhase3b()
phase3b = SAC("MlpPolicy", env3b, verbose=0, learning_rate=3e-4)

# ------------------------------------------------------------------ #
#  Copy actor weights                                                  #
# ------------------------------------------------------------------ #
print("Transferring actor weights...")

src_actor = phase3a.policy.actor.state_dict()
dst_actor = phase3b.policy.actor.state_dict()

for key in dst_actor:
    if key not in src_actor:
        print(f"  [skip] {key} — not in source")
        continue

    src_w = src_actor[key]
    dst_w = dst_actor[key]

    if src_w.shape == dst_w.shape:
        dst_actor[key] = src_w.clone()
        print(f"  [copy] {key}  {tuple(src_w.shape)}")

    elif dst_w.dim() == 2 and dst_w.shape[1] > src_w.shape[1]:
        # First weight matrix: copy first 16 cols, zero the new 6
        new_w = torch.zeros_like(dst_w)
        new_w[:, :src_w.shape[1]] = src_w.clone()
        dst_actor[key] = new_w
        print(f"  [pad]  {key}  {tuple(src_w.shape)} → {tuple(dst_w.shape)}")

    else:
        print(f"  [skip] {key}  shape mismatch {tuple(src_w.shape)} vs {tuple(dst_w.shape)}")

phase3b.policy.actor.load_state_dict(dst_actor)

# ------------------------------------------------------------------ #
#  Copy critic weights (both Q-networks)                              #
# ------------------------------------------------------------------ #
print("Transferring critic weights...")

for critic_attr in ["critic", "critic_target"]:
    src_critic = getattr(phase3a.policy, critic_attr).state_dict()
    dst_critic = getattr(phase3b.policy, critic_attr).state_dict()

    for key in dst_critic:
        if key not in src_critic:
            print(f"  [skip] {critic_attr}.{key} — not in source")
            continue

        src_w = src_critic[key]
        dst_w = dst_critic[key]

        if src_w.shape == dst_w.shape:
            dst_critic[key] = src_w.clone()
            print(f"  [copy] {critic_attr}.{key}  {tuple(src_w.shape)}")

        elif dst_w.dim() == 2 and dst_w.shape[1] > src_w.shape[1]:
            # Critic first layer: obs+action input
            # 16+3=19 (phase3a) → 22+3=25 (phase3b)
            new_w = torch.zeros_like(dst_w)
            new_w[:, :src_w.shape[1]] = src_w.clone()
            dst_critic[key] = new_w
            print(f"  [pad]  {critic_attr}.{key}  {tuple(src_w.shape)} → {tuple(dst_w.shape)}")

        else:
            print(f"  [skip] {critic_attr}.{key}  shape mismatch")

    getattr(phase3b.policy, critic_attr).load_state_dict(dst_critic)

# ------------------------------------------------------------------ #
#  Save transferred model                                             #
# ------------------------------------------------------------------ #
phase3b.save(OUT_PATH)
print(f"\nTransfer complete. Saved to {OUT_PATH}.zip")
print("Now run train.py — it will load this as Stage 1 starting point.")