import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from stable_baselines3 import SAC
from missile_env_phase3 import MissileEnvPhase3

NUM_EPISODES = 20
PAUSE_FRAMES = 30
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ #
#  Load model — full difficulty (Stage 3 settings)                    #
# ------------------------------------------------------------------ #
MissileEnvPhase3.v_I           = 250.0
MissileEnvPhase3.int_spawn_min = 800.0

model = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase3a"))
env   = MissileEnvPhase3()

# ------------------------------------------------------------------ #
#  Pre-run all episodes                                                #
# ------------------------------------------------------------------ #
print(f"Pre-running {NUM_EPISODES} episodes...")
episodes_data = []

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    frames  = []
    m_trail = {"x": [env.x],        "y": [env.y],        "z": [env.z]}
    t_trail = {"x": [env.target_x], "y": [env.target_y], "z": [env.target_z]}
    i_trail = {"x": [env.int_x],    "y": [env.int_y],    "z": [env.int_z]}
    outcome      = "timeout"
    total_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        m_trail["x"].append(env.x)
        m_trail["y"].append(env.y)
        m_trail["z"].append(env.z)
        t_trail["x"].append(env.target_x)
        t_trail["y"].append(env.target_y)
        t_trail["z"].append(env.target_z)

        if env.interceptor_active:
            i_trail["x"].append(env.int_x)
            i_trail["y"].append(env.int_y)
            i_trail["z"].append(env.int_z)
        else:
            i_trail["x"].append(i_trail["x"][-1])
            i_trail["y"].append(i_trail["y"][-1])
            i_trail["z"].append(i_trail["z"][-1])

        t_spd = float(np.sqrt(env.target_vx**2 + env.target_vy**2 + env.target_vz**2))
        t_hdg = float(np.degrees(np.arctan2(env.target_vy, env.target_vx)))
        t_clb = float(np.degrees(np.arctan2(env.target_vz,
                                            np.hypot(env.target_vx, env.target_vy))))

        i_spd = float(np.sqrt(env.int_vx**2 + env.int_vy**2 + env.int_vz**2))

        # Distance missile → interceptor
        d_MI = float(np.sqrt(
            (env.int_x - env.x)**2 +
            (env.int_y - env.y)**2 +
            (env.int_z - env.z)**2
        )) if env.interceptor_active else 9999.0

        frames.append({
            # Missile
            "mx": env.x, "my": env.y, "mz": env.z,
            "v":  env.v,
            "heading_deg": float(np.degrees(env.theta)),
            "climb_deg":   float(np.degrees(env.gamma)),
            "fuel":        env.fuel,
            "d_MT":        env.previous_d_MT,
            "d_MI":        d_MI,
            "step":        env.step_count,
            # Target
            "tx": env.target_x, "ty": env.target_y, "tz": env.target_z,
            "target_vx": env.target_vx,
            "target_vy": env.target_vy,
            "target_vz": env.target_vz,
            "target_speed":   t_spd,
            "target_heading": t_hdg,
            "target_climb":   t_clb,
            # Interceptor
            "ix": env.int_x, "iy": env.int_y, "iz": env.int_z,
            "int_vx": env.int_vx, "int_vy": env.int_vy, "int_vz": env.int_vz,
            "int_speed":  i_spd,
            "int_active": env.interceptor_active,
            # Trails
            "mx_trail": list(m_trail["x"]),
            "my_trail": list(m_trail["y"]),
            "mz_trail": list(m_trail["z"]),
            "tx_trail": list(t_trail["x"]),
            "ty_trail": list(t_trail["y"]),
            "tz_trail": list(t_trail["z"]),
            "ix_trail": list(i_trail["x"]),
            "iy_trail": list(i_trail["y"]),
            "iz_trail": list(i_trail["z"]),
            "closing_speed": 0.0,
        })

        if terminated or truncated:
            outcome = info.get("outcome", "timeout")
            break

    # Compute closing speed per frame
    for i in range(1, len(frames)):
        frames[i]["closing_speed"] = (
            (frames[i-1]["d_MT"] - frames[i]["d_MT"]) / env.dt
        )

    episodes_data.append({
        "frames":         frames,
        "outcome":        outcome,
        "reward":         total_reward,
        "steps":          env.step_count,
        "fuel_remaining": env.fuel,
    })
    tag = outcome.upper().ljust(12)
    print(f"  Ep {ep+1:2d}: {tag}  steps={env.step_count:4d}  "
          f"reward={total_reward:6.1f}  fuel={env.fuel:.2f}")

n_hits = sum(1 for e in episodes_data if e["outcome"] == "hit")
n_int  = sum(1 for e in episodes_data if e["outcome"] == "intercepted")
print(f"\nHit: {n_hits}/{NUM_EPISODES}   Intercepted: {n_int}/{NUM_EPISODES}\n")

# ------------------------------------------------------------------ #
#  Build global frame sequence                                         #
# ------------------------------------------------------------------ #
sequence = []
for ep_idx, ep in enumerate(episodes_data):
    for fr_idx in range(len(ep["frames"])):
        sequence.append(("play",  ep_idx, fr_idx))
    for p in range(PAUSE_FRAMES):
        sequence.append(("pause", ep_idx, p))

all_rewards = [e["reward"] for e in episodes_data]
bar_ylim    = max(abs(min(all_rewards)), abs(max(all_rewards))) * 1.2

# ------------------------------------------------------------------ #
#  Figure layout                                                       #
# ------------------------------------------------------------------ #
fig = plt.figure(figsize=(22, 11))
fig.patch.set_facecolor("#060614")

gs = gridspec.GridSpec(
    5, 2, figure=fig,
    width_ratios=[2.2, 1],
    height_ratios=[1.2, 1.2, 0.9, 0.9, 0.9],
    hspace=0.60, wspace=0.28
)

ax3d   = fig.add_subplot(gs[:, 0], projection="3d")
ax_mis = fig.add_subplot(gs[0, 1])
ax_tgt = fig.add_subplot(gs[1, 1])
ax_int = fig.add_subplot(gs[2, 1])
ax_bar = fig.add_subplot(gs[3, 1])
ax_rt  = fig.add_subplot(gs[4, 1])

# ── Style telemetry axes ─────────────────────────────────────────────
for ax, title, col in [
    (ax_mis, "MISSILE  TELEMETRY", "#00ffcc"),
    (ax_tgt, "TARGET   TELEMETRY", "#ff9900"),
    (ax_int, "INTERCEPTOR  STATUS", "#ff4466"),
]:
    ax.set_facecolor("#050518")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, color=col, fontsize=8, pad=4,
                 fontfamily="monospace", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_edgecolor("#223344")

# ── Style chart axes ─────────────────────────────────────────────────
for ax in [ax_bar, ax_rt]:
    ax.set_facecolor("#050518")
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#334466")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── 3D axes ──────────────────────────────────────────────────────────
ax3d.set_facecolor("#060614")
ax3d.set_xlim(0, env.L); ax3d.set_ylim(0, env.L); ax3d.set_zlim(0, env.H)
ax3d.set_xlabel("X (m)", color="gray", fontsize=8)
ax3d.set_ylabel("Y (m)", color="gray", fontsize=8)
ax3d.set_zlabel("Alt (m)", color="gray", fontsize=8)
ax3d.tick_params(colors="gray", labelsize=7)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor("#0d0d2a")
ax3d.yaxis.pane.set_edgecolor("#0d0d2a")
ax3d.zaxis.pane.set_edgecolor("#0d0d2a")

# Ground grid
for xi in np.linspace(0, env.L, 7):
    ax3d.plot([xi, xi], [0, env.L], [0, 0], color="#1a2244", lw=0.5, alpha=0.6)
for yi in np.linspace(0, env.L, 7):
    ax3d.plot([0, env.L], [yi, yi], [0, 0], color="#1a2244", lw=0.5, alpha=0.6)

# Missile trail (blue)
old_tm, = ax3d.plot([], [], [], color="#003355", lw=0.8, alpha=0.2)
mid_tm, = ax3d.plot([], [], [], color="#0077aa", lw=1.4, alpha=0.5)
new_tm, = ax3d.plot([], [], [], color="#00ccff", lw=2.2, alpha=1.0)

# Target trail (orange)
old_tt, = ax3d.plot([], [], [], color="#552200", lw=0.8, alpha=0.2)
mid_tt, = ax3d.plot([], [], [], color="#aa5500", lw=1.4, alpha=0.5)
new_tt, = ax3d.plot([], [], [], color="#ff9900", lw=2.2, alpha=1.0)

# Interceptor trail (red/pink)
old_ti, = ax3d.plot([], [], [], color="#440011", lw=0.8, alpha=0.2)
mid_ti, = ax3d.plot([], [], [], color="#990033", lw=1.4, alpha=0.5)
new_ti, = ax3d.plot([], [], [], color="#ff2255", lw=2.2, alpha=1.0)

# Markers
missile_dot,     = ax3d.plot([], [], [], "o",  color="#00ffcc", ms=9,  zorder=5)
target_dot,      = ax3d.plot([], [], [], "*",  color="#ff9900", ms=14, zorder=5)
interceptor_dot, = ax3d.plot([], [], [], "^",  color="#ff2255", ms=10, zorder=5)

explosion = ax3d.scatter([env.L/2], [env.L/2], [env.H/2],
                          c="#ffff00", s=60, alpha=0.0, zorder=10)
int_kill  = ax3d.scatter([env.L/2], [env.L/2], [env.H/2],
                          c="#ff2255", s=60, alpha=0.0, zorder=10)

# Engagement overlay
engage_text = fig.text(
    0.02, 0.02, "", color="white", fontsize=8, va="bottom",
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a0a22",
              edgecolor="#334466", alpha=0.88)
)

# Outcome overlay
outcome_text = fig.text(
    0.32, 0.50, "", color="white", fontsize=26,
    fontweight="bold", ha="center", va="center",
    fontfamily="monospace", alpha=0.0,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#000000", alpha=0.0)
)

# Telemetry text objects
mis_text = ax_mis.text(0.04, 0.96, "initializing...",
    transform=ax_mis.transAxes, color="#00ffcc",
    fontsize=7.5, va="top", fontfamily="monospace")

tgt_text = ax_tgt.text(0.04, 0.96, "initializing...",
    transform=ax_tgt.transAxes, color="#ff9900",
    fontsize=7.5, va="top", fontfamily="monospace")

int_text = ax_int.text(0.04, 0.96, "initializing...",
    transform=ax_int.transAxes, color="#ff4466",
    fontsize=7.5, va="top", fontfamily="monospace")

# ── Bar chart ────────────────────────────────────────────────────────
ax_bar.set_title("Reward / Episode", color="white", fontsize=8, pad=4)
ax_bar.set_xlim(0.3, NUM_EPISODES + 0.7)
ax_bar.set_ylim(-bar_ylim, bar_ylim)
ax_bar.set_ylabel("Reward", color="gray", fontsize=7)
ax_bar.axhline(0, color="#334466", lw=0.8)

bar_rects = [
    ax_bar.bar(i + 1, 0, color="gray", width=0.65, alpha=0.25)[0]
    for i in range(NUM_EPISODES)
]

# ── Hit rate chart ───────────────────────────────────────────────────
ax_rt.set_title("Running Hit Rate", color="white", fontsize=8, pad=4)
ax_rt.set_xlim(0, NUM_EPISODES + 1)
ax_rt.set_ylim(-5, 108)
ax_rt.set_ylabel("Hit %", color="gray", fontsize=7)
ax_rt.axhline(70, color="#335533", lw=0.8, ls="--", alpha=0.7)
ax_rt.text(NUM_EPISODES - 0.5, 73, "70%", color="#446644", fontsize=6)

hit_rate_line, = ax_rt.plot([], [], color="#00ff88", lw=2,
                             marker="o", ms=3, zorder=3)

# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #
completed_eps = []

def _set_trail(old_l, mid_l, new_l, xs, ys, zs):
    n  = len(xs)
    e1 = max(0, n - 60)
    s2, e2 = max(0, n - 60), max(0, n - 20)
    old_l.set_data(xs[:e1], ys[:e1]);       old_l.set_3d_properties(zs[:e1])
    mid_l.set_data(xs[s2:e2], ys[s2:e2]);   mid_l.set_3d_properties(zs[s2:e2])
    new_l.set_data(xs[max(0,n-20):], ys[max(0,n-20):])
    new_l.set_3d_properties(zs[max(0,n-20):])


def _update_charts(ep_idx):
    ep = episodes_data[ep_idx]
    if ep["outcome"] == "hit":
        color = "#00cc66"
    elif ep["outcome"] == "intercepted":
        color = "#ff2255"
    else:
        color = "#cc3333"
    bar_rects[ep_idx].set_height(ep["reward"])
    bar_rects[ep_idx].set_color(color)
    bar_rects[ep_idx].set_alpha(0.85)

    done = completed_eps
    hr_x = [i + 1 for i in done]
    hr_y = [
        sum(1 for j in done[:k+1] if episodes_data[j]["outcome"] == "hit") / (k + 1) * 100
        for k in range(len(done))
    ]
    hit_rate_line.set_data(hr_x, hr_y)


def _missile_hud(f):
    closing = f["closing_speed"]
    cdir    = "+" if closing >= 0 else "-"
    return (
        f"  SPD  {f['v']:6.1f} m/s\n"
        f"  ALT  {f['mz']:6.0f} m\n"
        f"  HDG  {f['heading_deg']:6.1f} °\n"
        f"  CLB  {f['climb_deg']:6.1f} °\n"
        f"  FUEL {f['fuel']*100:5.1f} %\n"
        f"  POS  {f['mx']:5.0f} / {f['my']:5.0f} m\n"
        f"  DIST {f['d_MT']:6.0f} m (target)\n"
        f"  DIST {f['d_MI']:6.0f} m (interceptor)\n"
        f"  CLOS {cdir}{abs(closing):5.1f} m/s"
    )


def _target_hud(f):
    return (
        f"  SPD  {f['target_speed']:6.1f} m/s\n"
        f"  ALT  {f['tz']:6.0f} m\n"
        f"  HDG  {f['target_heading']:6.1f} °\n"
        f"  CLB  {f['target_climb']:6.1f} °\n"
        f"  VX   {f['target_vx']:+6.1f} m/s\n"
        f"  VY   {f['target_vy']:+6.1f} m/s\n"
        f"  VZ   {f['target_vz']:+6.1f} m/s\n"
        f"  POS  {f['tx']:5.0f} / {f['ty']:5.0f} m"
    )


def _int_hud(f):
    status = "ACTIVE  " if f["int_active"] else "NEUTRALISED"
    threat  = "HIGH" if f["d_MI"] < 500 else ("MED" if f["d_MI"] < 1000 else "LOW")
    return (
        f"  STATUS  {status}\n"
        f"  SPD     {f['int_speed']:6.1f} m/s\n"
        f"  ALT     {f['iz']:6.0f} m\n"
        f"  VX      {f['int_vx']:+6.1f} m/s\n"
        f"  VY      {f['int_vy']:+6.1f} m/s\n"
        f"  VZ      {f['int_vz']:+6.1f} m/s\n"
        f"  RANGE   {f['d_MI']:6.0f} m\n"
        f"  THREAT  {threat}"
    )


def _engage_hud(f, ep_idx):
    return (
        f" EP {ep_idx+1:2d}/{NUM_EPISODES}   "
        f"STEP {f['step']:4d}   "
        f"TIME {f['step']*env.dt:5.1f}s"
    )


OUTCOME_STYLES = {
    "hit":         ("TARGET HIT",    "#00ff88", "#003322"),
    "intercepted": ("INTERCEPTED",   "#ff2255", "#330011"),
    "oob":         ("OUT OF BOUNDS", "#ffaa00", "#332200"),
    "timeout":     ("TIMEOUT",       "#aaaaaa", "#222222"),
}

# ------------------------------------------------------------------ #
#  Animation update                                                    #
# ------------------------------------------------------------------ #
def update(gi):
    kind, ep_idx, sub_idx = sequence[gi]
    ep = episodes_data[ep_idx]

    if kind == "pause" and sub_idx == 0 and ep_idx not in completed_eps:
        completed_eps.append(ep_idx)
        _update_charts(ep_idx)

    if kind == "play":
        f = ep["frames"][sub_idx]

        # Missile trail
        _set_trail(old_tm, mid_tm, new_tm,
                   f["mx_trail"], f["my_trail"], f["mz_trail"])
        # Target trail
        _set_trail(old_tt, mid_tt, new_tt,
                   f["tx_trail"], f["ty_trail"], f["tz_trail"])
        # Interceptor trail
        if f["int_active"]:
            _set_trail(old_ti, mid_ti, new_ti,
                       f["ix_trail"], f["iy_trail"], f["iz_trail"])
        else:
            for ln in [old_ti, mid_ti, new_ti]:
                ln.set_data([], []); ln.set_3d_properties([])

        # Markers
        missile_dot.set_data([f["mx"]], [f["my"]])
        missile_dot.set_3d_properties([f["mz"]])
        missile_dot.set_color("#00ffcc"); missile_dot.set_markersize(9)

        target_dot.set_data([f["tx"]], [f["ty"]])
        target_dot.set_3d_properties([f["tz"]])
        target_dot.set_color("#ff9900"); target_dot.set_markersize(14)

        if f["int_active"]:
            interceptor_dot.set_data([f["ix"]], [f["iy"]])
            interceptor_dot.set_3d_properties([f["iz"]])
            interceptor_dot.set_color("#ff2255"); interceptor_dot.set_markersize(10)
        else:
            interceptor_dot.set_data([], []); interceptor_dot.set_3d_properties([])

        explosion._offsets3d  = (np.array([env.L/2]), np.array([env.L/2]), np.array([env.H/2]))
        explosion.set_alpha(0.0)
        int_kill._offsets3d   = (np.array([env.L/2]), np.array([env.L/2]), np.array([env.H/2]))
        int_kill.set_alpha(0.0)
        outcome_text.set_alpha(0.0)
        outcome_text.get_bbox_patch().set_alpha(0.0)

        mis_text.set_text(_missile_hud(f))
        tgt_text.set_text(_target_hud(f))
        int_text.set_text(_int_hud(f))
        engage_text.set_text(_engage_hud(f, ep_idx))

        ax3d.set_title(
            f"Phase 3a  —  Episode {ep_idx+1} / {NUM_EPISODES}",
            color="white", fontsize=11, pad=8
        )

    elif kind == "pause":
        f    = ep["frames"][-1]
        fade = max(0.0, 1.0 - sub_idx / PAUSE_FRAMES)
        outcome = ep["outcome"]

        _set_trail(old_tm, mid_tm, new_tm,
                   f["mx_trail"], f["my_trail"], f["mz_trail"])
        _set_trail(old_tt, mid_tt, new_tt,
                   f["tx_trail"], f["ty_trail"], f["tz_trail"])

        label, color, bg = OUTCOME_STYLES.get(outcome, ("MISS", "#ff4444", "#330000"))

        if outcome == "hit":
            np.random.seed(ep_idx * 7)
            angles = np.random.uniform(0, 2 * np.pi, 30)
            radii  = np.random.uniform(40, 220, 30)
            ex = np.clip(f["tx"] + radii * np.cos(angles), 0, env.L)
            ey = np.clip(f["ty"] + radii * np.sin(angles), 0, env.L)
            ez = np.clip(f["tz"] + np.random.uniform(-130, 130, 30), 0, env.H)
            explosion._offsets3d = (ex, ey, ez)
            explosion.set_alpha(fade * 0.9)
            missile_dot.set_data([], [])
            missile_dot.set_3d_properties([])
            target_dot.set_color("#00ff88"); target_dot.set_markersize(22)

        elif outcome == "intercepted":
            np.random.seed(ep_idx * 13)
            angles = np.random.uniform(0, 2 * np.pi, 20)
            radii  = np.random.uniform(30, 150, 20)
            ex = np.clip(f["mx"] + radii * np.cos(angles), 0, env.L)
            ey = np.clip(f["my"] + radii * np.sin(angles), 0, env.L)
            ez = np.clip(f["mz"] + np.random.uniform(-80, 80, 20), 0, env.H)
            int_kill._offsets3d = (ex, ey, ez)
            int_kill.set_alpha(fade * 0.9)
            missile_dot.set_data([], [])
            missile_dot.set_3d_properties([])

        else:
            explosion._offsets3d = (np.array([env.L/2]), np.array([env.L/2]), np.array([env.H/2]))
            explosion.set_alpha(0.0)

        outcome_text.set_text(label)
        outcome_text.set_color(color)
        outcome_text.get_bbox_patch().set_facecolor(bg)
        outcome_text.set_alpha(fade)
        outcome_text.get_bbox_patch().set_alpha(0.85 * fade)

    return (old_tm, mid_tm, new_tm,
            old_tt, mid_tt, new_tt,
            old_ti, mid_ti, new_ti,
            missile_dot, target_dot, interceptor_dot,
            explosion, int_kill,
            mis_text, tgt_text, int_text,
            engage_text, outcome_text,
            hit_rate_line, *bar_rects)


anim = FuncAnimation(
    fig, update,
    frames=len(sequence),
    interval=60,
    blit=False,
    repeat=False
)

plt.tight_layout()
plt.show()