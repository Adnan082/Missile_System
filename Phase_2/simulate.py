import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from stable_baselines3 import SAC
from missile_env_phase2 import MissileEnvPhase2

# ------------------------------------------------------------------ #
#  Load model and run one episode                                      #
# ------------------------------------------------------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model    = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase2b"))
env      = MissileEnvPhase2()

obs, _ = env.reset()

missile_trail = {"x": [env.x], "y": [env.y], "z": [env.z]}
target_trail  = {"x": [env.target_x], "y": [env.target_y], "z": [env.target_z]}

frames_data = []
done = False
hit = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    missile_trail["x"].append(env.x)
    missile_trail["y"].append(env.y)
    missile_trail["z"].append(env.z)
    target_trail["x"].append(env.target_x)
    target_trail["y"].append(env.target_y)
    target_trail["z"].append(env.target_z)

    frames_data.append({
        "mx": env.x,  "my": env.y,  "mz": env.z,
        "tx": env.target_x, "ty": env.target_y, "tz": env.target_z,
        "v" : env.v,
        "theta": env.theta,
        "gamma": env.gamma,
        "d_MT" : env.previous_d_MT,
        "fuel" : env.fuel,
        "step" : env.step_count,
        "m_trail_x": list(missile_trail["x"]),
        "m_trail_y": list(missile_trail["y"]),
        "m_trail_z": list(missile_trail["z"]),
        "t_trail_x": list(target_trail["x"]),
        "t_trail_y": list(target_trail["y"]),
        "t_trail_z": list(target_trail["z"]),
    })
    done = terminated or truncated
    if terminated and reward >= 100.0:
        hit = True

# ------------------------------------------------------------------ #
#  Setup 3D figure                                                     #
# ------------------------------------------------------------------ #
fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor("#0d0d1a")

ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#0d0d1a")
ax.set_xlim(0, env.L)
ax.set_ylim(0, env.L)
ax.set_zlim(0, env.H)
ax.set_xlabel("X (m)", color="gray")
ax.set_ylabel("Y (m)", color="gray")
ax.set_zlabel("Z / Alt (m)", color="gray")
ax.set_title("Missile Guidance — Phase 2b (3D Moving Target + Fuel)",
             color="white", fontsize=13, pad=12)
ax.tick_params(colors="gray")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("#1a1a2e")
ax.yaxis.pane.set_edgecolor("#1a1a2e")
ax.zaxis.pane.set_edgecolor("#1a1a2e")

# Trails
missile_trail_line, = ax.plot([], [], [], color="#00aaff", lw=1.2,
                               alpha=0.6, label="Missile trail")
target_trail_line,  = ax.plot([], [], [], color="#ff8800", lw=1.2,
                               alpha=0.6, label="Target trail")

# Markers
missile_dot, = ax.plot([], [], [], marker="o", color="#00ffcc",
                        markersize=8, ls="none", label="Missile")
target_dot,  = ax.plot([], [], [], marker="*", color="#ff4444",
                        markersize=12, ls="none", label="Target")

# Stats text (2D overlay)
stats_text = fig.text(
    0.02, 0.92, "", color="white", fontsize=9,
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#111133",
              edgecolor="#334466", alpha=0.85)
)

# Outcome text (hidden until final frame)
outcome_text = fig.text(
    0.5, 0.5, "", color="white", fontsize=28,
    fontweight="bold", ha="center", va="center",
    fontfamily="monospace", alpha=0.0,
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#000000", alpha=0.0)
)

ax.legend(loc="upper right", facecolor="#111133",
          edgecolor="#334466", labelcolor="white", fontsize=8)

# ------------------------------------------------------------------ #
#  Animation                                                           #
# ------------------------------------------------------------------ #
def update(i):
    f = frames_data[i]

    missile_trail_line.set_data(f["m_trail_x"], f["m_trail_y"])
    missile_trail_line.set_3d_properties(f["m_trail_z"])

    target_trail_line.set_data(f["t_trail_x"], f["t_trail_y"])
    target_trail_line.set_3d_properties(f["t_trail_z"])

    missile_dot.set_data([f["mx"]], [f["my"]])
    missile_dot.set_3d_properties([f["mz"]])

    target_dot.set_data([f["tx"]], [f["ty"]])
    target_dot.set_3d_properties([f["tz"]])

    fuel_pct = f['fuel'] * 100
    stats_text.set_text(
        f"Speed   : {f['v']:6.1f} m/s\n"
        f"Distance: {f['d_MT']:6.1f} m\n"
        f"Heading : {np.degrees(f['theta']):6.1f}°\n"
        f"Climb   : {np.degrees(f['gamma']):6.1f}°\n"
        f"Fuel    : {fuel_pct:5.1f} %\n"
        f"Step    : {f['step']}"
    )

    # Final frame — show hit/miss outcome
    if i == len(frames_data) - 1:
        if hit:
            outcome_text.set_text("TARGET HIT")
            outcome_text.set_color("#00ff88")
            outcome_text.set_alpha(1.0)
            outcome_text.get_bbox_patch().set_facecolor("#003322")
            outcome_text.get_bbox_patch().set_alpha(0.85)
            # Flash target red → green
            target_dot.set_color("#00ff88")
            target_dot.set_markersize(20)
            missile_dot.set_data([], [])
            missile_dot.set_3d_properties([])
        else:
            outcome_text.set_text("MISS")
            outcome_text.set_color("#ff4444")
            outcome_text.set_alpha(1.0)
            outcome_text.get_bbox_patch().set_facecolor("#330000")
            outcome_text.get_bbox_patch().set_alpha(0.85)

    return missile_trail_line, target_trail_line, missile_dot, target_dot, stats_text, outcome_text


anim = FuncAnimation(
    fig, update,
    frames=len(frames_data),
    interval=30,
    blit=False,
    repeat=False
)

plt.tight_layout()
plt.show()
