import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from stable_baselines3 import SAC
from missile_env import MissileEnv

# ------------------------------------------------------------------ #
#  Load model and environment                                          #
# ------------------------------------------------------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase1"))
env   = MissileEnv()

obs, _ = env.reset()

# Storage for trails
missile_trail_x = [env.x]
missile_trail_y = [env.y]

done = False
frames_data = []

# Pre-simulate full episode and store frames
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    missile_trail_x.append(env.x)
    missile_trail_y.append(env.y)
    frames_data.append({
        "x"        : env.x,
        "y"        : env.y,
        "v"        : env.v,
        "theta"    : env.theta,
        "target_x" : env.target_x,
        "target_y" : env.target_y,
        "d_MT"     : env.previous_d_MT,
        "step"     : env.step_count,
        "trail_x"  : list(missile_trail_x),
        "trail_y"  : list(missile_trail_y),
    })
    done = terminated or truncated

# ------------------------------------------------------------------ #
#  Setup figure                                                        #
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("#0d0d1a")
ax.set_facecolor("#0d0d1a")
ax.set_xlim(0, env.L)
ax.set_ylim(0, env.L)
ax.set_aspect("equal")
ax.set_title("Missile Guidance — Phase 1", color="white", fontsize=14, pad=12)
ax.tick_params(colors="gray")
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")

# Grid
ax.grid(color="#1a1a2e", linewidth=0.5, linestyle="--")

# Static: kill radius circle (drawn once)
first = frames_data[0]
kill_circle = plt.Circle(
    (first["target_x"], first["target_y"]),
    env.r_kill, color="#ff4444", alpha=0.25, zorder=2
)
ax.add_patch(kill_circle)

# Target marker
target_plot, = ax.plot(
    first["target_x"], first["target_y"],
    marker="*", color="#ff4444", markersize=16, zorder=5, label="Target"
)

# Missile trail
trail_line, = ax.plot([], [], color="#00aaff", linewidth=1.2,
                      alpha=0.5, zorder=3, label="Trail")

# Missile marker (arrow)
missile_arrow = ax.annotate(
    "", xy=(first["x"], first["y"]),
    xytext=(first["x"] - 200, first["y"]),
    arrowprops=dict(arrowstyle="-|>", color="#00ffcc",
                    lw=2.5, mutation_scale=20),
    zorder=6
)

# Stats text box
stats_text = ax.text(
    0.02, 0.97, "", transform=ax.transAxes,
    color="white", fontsize=10, verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#111133",
              edgecolor="#334466", alpha=0.85)
)

# Step counter
step_text = ax.text(
    0.98, 0.97, "", transform=ax.transAxes,
    color="#aaaacc", fontsize=9, verticalalignment="top",
    horizontalalignment="right", fontfamily="monospace"
)

ax.legend(loc="lower right", facecolor="#111133",
          edgecolor="#334466", labelcolor="white", fontsize=9)

# ------------------------------------------------------------------ #
#  Animation update                                                    #
# ------------------------------------------------------------------ #
def update(frame_idx):
    f = frames_data[frame_idx]

    # Trail
    trail_line.set_data(f["trail_x"], f["trail_y"])

    # Missile arrow — tip at current position, tail 300m behind heading
    tip_x  = f["x"]
    tip_y  = f["y"]
    tail_x = f["x"] - 300 * np.cos(f["theta"])
    tail_y = f["y"] - 300 * np.sin(f["theta"])
    missile_arrow.set_position((tip_x, tip_y))
    missile_arrow.xy = (tip_x, tip_y)
    missile_arrow.xyann = (tail_x, tail_y)
    missile_arrow.xytext = (tail_x, tail_y)

    # Stats
    stats_text.set_text(
        f"Speed   : {f['v']:6.1f} m/s\n"
        f"Distance: {f['d_MT']:6.1f} m\n"
        f"Heading : {np.degrees(f['theta']):6.1f}°"
    )

    step_text.set_text(f"Step {f['step']}")

    return trail_line, missile_arrow, stats_text, step_text


anim = FuncAnimation(
    fig, update,
    frames=len(frames_data),
    interval=30,       # ms between frames (~33 fps)
    blit=False,
    repeat=False
)

plt.tight_layout()
plt.show()
