"""
Weight surgery: transfer Phase 2b brain into Phase 3a network.

Phase 2b obs = 10-dim  →  Phase 3a obs = 16-dim (6 new interceptor inputs added)
Hidden layers (64→64) and output layer are identical size in both models,
so we copy them directly. For the first layer we copy the first 10 input
weights and zero-initialise the 6 new interceptor inputs.

Result: Phase 3a starts with Phase 2b's full pursuit + fuel knowledge.
The new interceptor inputs start at zero influence and are learned from
experience as training progresses — true curriculum learning.
"""

import os
import torch
import numpy as np
from stable_baselines3 import SAC
from missile_env_phase3 import MissileEnvPhase3

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PHASE2B_PATH = os.path.join(BASE_DIR, "..", "Phase_2", "missile_sac_phase2b")
OUT_PATH     = os.path.join(BASE_DIR, "missile_sac_phase3a_transferred")

# ------------------------------------------------------------------ #
#  Load Phase 2b model (10-dim obs)                                   #
# ------------------------------------------------------------------ #
print("Loading Phase 2b model...")
phase2b = SAC.load(PHASE2B_PATH)

# ------------------------------------------------------------------ #
#  Create fresh Phase 3a model (16-dim obs)                           #
# ------------------------------------------------------------------ #
print("Creating Phase 3a model...")
MissileEnvPhase3.v_I           = 80.0
MissileEnvPhase3.int_spawn_min = 1500.0

env3    = MissileEnvPhase3()
phase3a = SAC("MlpPolicy", env3, verbose=0, learning_rate=3e-4)

# ------------------------------------------------------------------ #
#  Copy actor weights                                                  #
# ------------------------------------------------------------------ #
print("Transferring actor weights...")

src_actor  = phase2b.policy.actor.state_dict()
dst_actor  = phase3a.policy.actor.state_dict()

for key in dst_actor:
    if key not in src_actor:
        print(f"  [skip] {key} — not in source")
        continue

    src_w = src_actor[key]
    dst_w = dst_actor[key]

    if src_w.shape == dst_w.shape:
        # Hidden layers and output — copy exactly
        dst_actor[key] = src_w.clone()
        print(f"  [copy] {key}  {tuple(src_w.shape)}")

    elif dst_w.dim() == 2 and dst_w.shape[1] > src_w.shape[1]:
        # First weight matrix: (out_features, in_features)
        # dst is wider (16 inputs) — copy first 10 columns, zero rest
        new_w = torch.zeros_like(dst_w)
        new_w[:, :src_w.shape[1]] = src_w.clone()
        dst_actor[key] = new_w
        print(f"  [pad]  {key}  {tuple(src_w.shape)} → {tuple(dst_w.shape)}")

    else:
        print(f"  [skip] {key}  shape mismatch {tuple(src_w.shape)} vs {tuple(dst_w.shape)}")

phase3a.policy.actor.load_state_dict(dst_actor)

# ------------------------------------------------------------------ #
#  Copy critic weights (both Q-networks)                              #
# ------------------------------------------------------------------ #
print("Transferring critic weights...")

for critic_attr in ["critic", "critic_target"]:
    src_critic = getattr(phase2b.policy, critic_attr).state_dict()
    dst_critic = getattr(phase3a.policy, critic_attr).state_dict()

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
            # Critic first layer takes obs + action concatenated
            # obs expands from 10→16, action stays at 3
            # so first layer input goes from 13→19
            new_w = torch.zeros_like(dst_w)
            new_w[:, :src_w.shape[1]] = src_w.clone()
            dst_critic[key] = new_w
            print(f"  [pad]  {critic_attr}.{key}  {tuple(src_w.shape)} → {tuple(dst_w.shape)}")

        else:
            print(f"  [skip] {critic_attr}.{key}  shape mismatch")

    getattr(phase3a.policy, critic_attr).load_state_dict(dst_critic)

# ------------------------------------------------------------------ #
#  Save transferred model                                             #
# ------------------------------------------------------------------ #
phase3a.save(OUT_PATH)
print(f"\nTransfer complete. Saved to {OUT_PATH}.zip")
print("Now run train.py — it will load this as Stage 1 starting point.")
