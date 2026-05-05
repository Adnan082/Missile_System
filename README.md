# RL Missile Guidance System

A reinforcement learning project that trains a missile to intercept an evasive target in a 3D arena — with no hard-coded guidance laws. The missile discovers interception geometry, fuel management, and counter-maneuver strategies entirely through trial and error.

---

## What This Project Does

Traditional missiles use fixed mathematical guidance laws (Proportional Navigation, TVM, etc.) that are hand-tuned by engineers. This project asks a different question:

> **Can a neural network learn to guide a missile better than a formula — just by practicing?**

The answer, after 5.5 million training steps, is **yes — 87% hit rate against a target executing real F-16 combat maneuvers.**

---

## Architecture

```
3D Arena (3km × 3km × 3km)
│
├── Missile Agent (SAC — learns to intercept)
│     22-dim observation: target bearing, range, elevation,
│                         own speed/heading/fuel,
│                         2× interceptor positions/velocities
│     3-dim action: thrust, yaw rate, pitch rate
│
├── Target (Phase 1–3: scripted bounce / Phase 4: F-16 maneuver AI)
│
└── 2 Interceptors (pure pursuit — chase the missile)
```

**Algorithm:** Soft Actor-Critic (SAC) — off-policy, continuous action space, sample efficient

---

## Training Phases

### Phase 1 — Basic Pursuit
- Simple 3D arena, stationary or slow-moving target
- Missile learns to chase and close distance
- **500k steps**

### Phase 2 — Physics Constraints
- Fuel system added (limited burn time)
- Speed envelope enforced
- Missile learns fuel-efficient interception
- **500k steps**

### Phase 3a — Interceptors Introduced
- 2 defending interceptors chase the missile
- Missile must reach target while evading
- **1.5M steps**

### Phase 3b — Faster Target + Curriculum
- Target speed increases across training
- Multi-stage curriculum: slow → medium → fast
- Observation space locked at **22 dimensions**
- **500k steps**

### Phase 3c/3d — Full Curriculum
- 2 interceptors at varying spawn distances
- Interceptor speed curriculum (80 → 250 m/s)
- Best general-purpose missile model saved
- **1M steps** (cumulative Phase 3 total: ~3.5M)

### Phase 4 — F-16 Specialisation
Loads Phase 3d weights. Continues training against a target that executes real fighter jet combat maneuvers.

| Stage | Maneuvers Active | Target Speed | Kill Radius | Steps |
|-------|-----------------|-------------|-------------|-------|
| 4-1 | JINK only | 350 m/s | 300m | 500k |
| 4-2 | BEAM + JINK | 400 m/s | 250m | 500k |
| 4-3 | BEAM + JINK + BREAK + RUN | 450 m/s | 200m | 1M |

**Total Phase 4: 2M steps**

**Grand total: ~5.5M training steps**

---

## F-16 Maneuver System

The Phase 4 target is a state machine that selects maneuvers by priority:

| Maneuver | Trigger | Tactic |
|----------|---------|--------|
| **BREAK TURN** | Missile inside 2km | Max-G turn perpendicular to missile — forces overshoot |
| **AFTERBURNER RUN** | Missile beyond 5km | Full speed directly away — drains missile fuel |
| **BEAM** | Default | Fly 90° to missile LOS — maximizes tracking demand |
| **JINK** | Alternates with BEAM | Hard direction reversal every 2s — breaks PN prediction |

F-16 physics: `v_max = 450 m/s`, `omega_max = 1.5 rad/s (9G)`, `pitch_max = 0.8 rad/s`

---

## Results

| Opponent | Missile Win | F-16 Survived |
|----------|------------|---------------|
| Straight-line bouncing target | ~95% | ~5% |
| Classical beam maneuver (scripted) | 78% | 0% |
| RL-trained evader (PPO) | 79% | 21% |
| **Full F-16 maneuver suite (Phase 4)** | **87%** | **0%** |

**Key finding:** The RL evader (PPO-trained, 21% survival) outperformed the classical beam maneuver (scripted, 0% survival) against the same SAC missile — RL discovered non-obvious evasion strategies that hand-coded tactics missed.

---

## Project Structure

```
Missile_System/
│
├── Phase_1/                    # Basic pursuit
│   ├── missile_env.py
│   ├── train.py
│   ├── evaluate.py
│   └── simulate.py
│
├── Phase_3b/                   # Full curriculum (shared base env)
│   ├── missile_env_phase3b.py  # 22-dim obs, interceptors, fuel
│   ├── train.py
│   ├── evaluate.py
│   └── simulate.py
│
├── Phase_4/                    # F-16 specialisation
│   ├── missile_env_f16.py      # Extends Phase3b with F-16 maneuver AI
│   ├── train.py                # 3-stage curriculum (2M steps)
│   └── evaluate.py
│
├── Arena/
│   ├── arena_beam.py           # SAC missile vs scripted beam maneuver
│   └── arena.py                # SAC missile vs PPO target
│
└── Target_Brain/
    ├── target_env.py           # RL target evasion environment
    └── train.py                # PPO target training
```

---

## How It Differs From Real Missiles

| Feature | Real Missiles (AMRAAM, PL-15) | This System |
|---------|-------------------------------|-------------|
| Guidance law | Hard-coded PN formula | Learned neural policy |
| Adaptation | Fixed — can't change after manufacture | Retrain against new maneuvers |
| Multi-threat awareness | Ignores interceptors | Tracks 2 interceptors simultaneously |
| Development | Years + hardware | Days on a laptop |
| Verification | Mathematically provable | Empirically tested |

---

## Why Defense Industry Hasn't Done This Yet

1. **Certification** — RL policies are black boxes; PN guidance has mathematical proofs
2. **Sim-to-real gap** — real radar is noisy, susceptible to ECM; simulation assumes perfect state
3. **Hardware constraints** — neural network inference at 100Hz on radiation-hardened embedded chips
4. **Accountability** — laws of war require human accountability for lethal decisions
5. **Adversarial brittleness** — an enemy who knows your training distribution can find exploits

DARPA's ACE program (2020) already demonstrated RL defeating human F-16 pilots 5-0 in simulation. Deployment is a verification and trust problem, not a capability problem.

---

## Future Directions

- [ ] **Phase 5** — Train against drone swarms (multiple targets)
- [ ] **10m kill radius curriculum** — closer-to-real proximity fuze accuracy
- [ ] **Sensor noise** — add radar noise and ECM to observation to test robustness
- [ ] **Proportional Navigation baseline** — compare SAC vs classical PN to quantify RL advantage
- [ ] **Beam reward shaping** — train RL target to learn beam maneuver rather than execute it rigidly
- [ ] **Expand arena** — scale to 30km × 30km to model BVR engagement ranges

---

## Setup

```bash
# Create virtual environment
python -m venv missile_venv
missile_venv\Scripts\activate

# Install dependencies
pip install stable-baselines3 gymnasium numpy

# Train Phase 4 (loads Phase 3d weights)
cd Phase_4
python train.py

# Evaluate
python evaluate.py
```

---

## Dependencies

- Python 3.10+
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) — SAC/PPO implementations
- gymnasium — RL environment standard
- numpy — physics simulation

---

## Results Comparison

```
Phase 3d (general)  →  79% vs RL evader,  78% vs beam maneuver
Phase 4  (F-16)     →  87% vs full F-16 suite,  0% F-16 survivals
```

Transfer learning from Phase 3d retained 3.5M steps of learned behavior while the 2M Phase 4 steps added F-16-specific counter-maneuver knowledge.
