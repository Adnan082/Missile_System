import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase_3b"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from stable_baselines3 import SAC
from missile_env_phase3b import MissileEnvPhase3b

NUM_EPISODES = 20
PAUSE_FRAMES = 30
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))

MissileEnvPhase3b.v_I           = 250.0
MissileEnvPhase3b.int_spawn_min = 800.0

model = SAC.load(os.path.join(BASE_DIR, "missile_sac_phase3c"))
env   = MissileEnvPhase3b()

print(f"Pre-running {NUM_EPISODES} episodes...")
episodes_data = []

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    frames  = []
    m_trail  = {"x": [env.x],        "y": [env.y],        "z": [env.z]}
    t_trail  = {"x": [env.target_x], "y": [env.target_y], "z": [env.target_z]}
    i1_trail = {"x": [env.int1_x],   "y": [env.int1_y],   "z": [env.int1_z]}
    i2_trail = {"x": [env.int2_x],   "y": [env.int2_y],   "z": [env.int2_z]}
    outcome      = "timeout"
    total_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        m_trail["x"].append(env.x);        m_trail["y"].append(env.y);        m_trail["z"].append(env.z)
        t_trail["x"].append(env.target_x); t_trail["y"].append(env.target_y); t_trail["z"].append(env.target_z)

        for trail, ax, ay, az, active in [
            (i1_trail, env.int1_x, env.int1_y, env.int1_z, env.int1_active),
            (i2_trail, env.int2_x, env.int2_y, env.int2_z, env.int2_active),
        ]:
            if active:
                trail["x"].append(ax); trail["y"].append(ay); trail["z"].append(az)
            else:
                trail["x"].append(trail["x"][-1]); trail["y"].append(trail["y"][-1]); trail["z"].append(trail["z"][-1])

        t_spd = float(np.sqrt(env.target_vx**2 + env.target_vy**2 + env.target_vz**2))
        t_hdg = float(np.degrees(np.arctan2(env.target_vy, env.target_vx)))
        t_clb = float(np.degrees(np.arctan2(env.target_vz, np.hypot(env.target_vx, env.target_vy))))
        i1_spd = float(np.sqrt(env.int1_vx**2 + env.int1_vy**2 + env.int1_vz**2))
        i2_spd = float(np.sqrt(env.int2_vx**2 + env.int2_vy**2 + env.int2_vz**2))
        d_MI1 = float(np.sqrt((env.int1_x-env.x)**2+(env.int1_y-env.y)**2+(env.int1_z-env.z)**2)) if env.int1_active else 9999.0
        d_MI2 = float(np.sqrt((env.int2_x-env.x)**2+(env.int2_y-env.y)**2+(env.int2_z-env.z)**2)) if env.int2_active else 9999.0

        frames.append({
            "mx": env.x, "my": env.y, "mz": env.z,
            "v": env.v, "heading_deg": float(np.degrees(env.theta)),
            "climb_deg": float(np.degrees(env.gamma)), "fuel": env.fuel,
            "d_MT": env.previous_d_MT, "d_MI1": d_MI1, "d_MI2": d_MI2, "step": env.step_count,
            "tx": env.target_x, "ty": env.target_y, "tz": env.target_z,
            "target_vx": env.target_vx, "target_vy": env.target_vy, "target_vz": env.target_vz,
            "target_speed": t_spd, "target_heading": t_hdg, "target_climb": t_clb,
            "i1x": env.int1_x, "i1y": env.int1_y, "i1z": env.int1_z,
            "int1_vx": env.int1_vx, "int1_vy": env.int1_vy, "int1_vz": env.int1_vz,
            "int1_speed": i1_spd, "int1_active": env.int1_active,
            "i2x": env.int2_x, "i2y": env.int2_y, "i2z": env.int2_z,
            "int2_vx": env.int2_vx, "int2_vy": env.int2_vy, "int2_vz": env.int2_vz,
            "int2_speed": i2_spd, "int2_active": env.int2_active,
            "mx_trail": list(m_trail["x"]),  "my_trail": list(m_trail["y"]),  "mz_trail": list(m_trail["z"]),
            "tx_trail": list(t_trail["x"]),  "ty_trail": list(t_trail["y"]),  "tz_trail": list(t_trail["z"]),
            "i1x_trail": list(i1_trail["x"]),"i1y_trail": list(i1_trail["y"]),"i1z_trail": list(i1_trail["z"]),
            "i2x_trail": list(i2_trail["x"]),"i2y_trail": list(i2_trail["y"]),"i2z_trail": list(i2_trail["z"]),
            "closing_speed": 0.0,
        })

        if terminated or truncated:
            outcome = info.get("outcome", "timeout")
            break

    for i in range(1, len(frames)):
        frames[i]["closing_speed"] = (frames[i-1]["d_MT"] - frames[i]["d_MT"]) / env.dt

    episodes_data.append({"frames": frames, "outcome": outcome,
                           "reward": total_reward, "steps": env.step_count, "fuel_remaining": env.fuel})
    print(f"  Ep {ep+1:2d}: {outcome.upper().ljust(12)}  steps={env.step_count:4d}  "
          f"reward={total_reward:6.1f}  fuel={env.fuel:.2f}")

n_hits = sum(1 for e in episodes_data if e["outcome"] == "hit")
n_int  = sum(1 for e in episodes_data if e["outcome"] == "intercepted")
print(f"\nHit: {n_hits}/{NUM_EPISODES}   Intercepted: {n_int}/{NUM_EPISODES}\n")

sequence = []
for ep_idx, ep in enumerate(episodes_data):
    for fr_idx in range(len(ep["frames"])): sequence.append(("play",  ep_idx, fr_idx))
    for p in range(PAUSE_FRAMES):           sequence.append(("pause", ep_idx, p))

all_rewards = [e["reward"] for e in episodes_data]
bar_ylim    = max(abs(min(all_rewards)), abs(max(all_rewards))) * 1.2

fig = plt.figure(figsize=(24, 12))
fig.patch.set_facecolor("#060614")
gs = gridspec.GridSpec(6, 2, figure=fig, width_ratios=[2.2, 1],
    height_ratios=[1.1, 1.1, 1.1, 1.1, 0.9, 0.9], hspace=0.65, wspace=0.28)

ax3d  = fig.add_subplot(gs[:, 0], projection="3d")
ax_mis = fig.add_subplot(gs[0, 1])
ax_tgt = fig.add_subplot(gs[1, 1])
ax_i1  = fig.add_subplot(gs[2, 1])
ax_i2  = fig.add_subplot(gs[3, 1])
ax_bar = fig.add_subplot(gs[4, 1])
ax_rt  = fig.add_subplot(gs[5, 1])

for ax, title, col in [
    (ax_mis, "MISSILE     TELEMETRY", "#00ffcc"),
    (ax_tgt, "TARGET      TELEMETRY", "#ff9900"),
    (ax_i1,  "INTERCEPTOR 1  STATUS", "#ff2255"),
    (ax_i2,  "INTERCEPTOR 2  STATUS", "#ff66aa"),
]:
    ax.set_facecolor("#050518"); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.set_title(title, color=col, fontsize=8, pad=4, fontfamily="monospace", fontweight="bold")

for ax in [ax_bar, ax_rt]:
    ax.set_facecolor("#050518"); ax.tick_params(colors="gray", labelsize=7)
    for spine in ["bottom","left"]: ax.spines[spine].set_color("#334466")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax3d.set_facecolor("#060614")
ax3d.set_xlim(0, env.L); ax3d.set_ylim(0, env.L); ax3d.set_zlim(0, env.H)
ax3d.set_xlabel("X (m)", color="gray", fontsize=8)
ax3d.set_ylabel("Y (m)", color="gray", fontsize=8)
ax3d.set_zlabel("Alt (m)", color="gray", fontsize=8)
ax3d.tick_params(colors="gray", labelsize=7)
ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor("#0d0d2a")
ax3d.yaxis.pane.set_edgecolor("#0d0d2a")
ax3d.zaxis.pane.set_edgecolor("#0d0d2a")

for xi in np.linspace(0, env.L, 7): ax3d.plot([xi,xi],[0,env.L],[0,0], color="#1a2244", lw=0.5, alpha=0.6)
for yi in np.linspace(0, env.L, 7): ax3d.plot([0,env.L],[yi,yi],[0,0], color="#1a2244", lw=0.5, alpha=0.6)

old_tm, = ax3d.plot([],[],[], color="#003355", lw=0.8, alpha=0.2)
mid_tm, = ax3d.plot([],[],[], color="#0077aa", lw=1.4, alpha=0.5)
new_tm, = ax3d.plot([],[],[], color="#00ccff", lw=2.2, alpha=1.0)
old_tt, = ax3d.plot([],[],[], color="#552200", lw=0.8, alpha=0.2)
mid_tt, = ax3d.plot([],[],[], color="#aa5500", lw=1.4, alpha=0.5)
new_tt, = ax3d.plot([],[],[], color="#ff9900", lw=2.2, alpha=1.0)
old_ti1,= ax3d.plot([],[],[], color="#440011", lw=0.8, alpha=0.2)
mid_ti1,= ax3d.plot([],[],[], color="#990033", lw=1.4, alpha=0.5)
new_ti1,= ax3d.plot([],[],[], color="#ff2255", lw=2.2, alpha=1.0)
old_ti2,= ax3d.plot([],[],[], color="#330022", lw=0.8, alpha=0.2)
mid_ti2,= ax3d.plot([],[],[], color="#882255", lw=1.4, alpha=0.5)
new_ti2,= ax3d.plot([],[],[], color="#ff66aa", lw=2.2, alpha=1.0)

missile_dot, = ax3d.plot([],[],[], "o", color="#00ffcc", ms=9,  zorder=5)
target_dot,  = ax3d.plot([],[],[], "*", color="#ff9900", ms=14, zorder=5)
int1_dot,    = ax3d.plot([],[],[], "^", color="#ff2255", ms=10, zorder=5)
int2_dot,    = ax3d.plot([],[],[], "v", color="#ff66aa", ms=10, zorder=5)
explosion = ax3d.scatter([env.L/2],[env.L/2],[env.H/2], c="#ffff00", s=60, alpha=0.0, zorder=10)
int_kill  = ax3d.scatter([env.L/2],[env.L/2],[env.H/2], c="#ff2255", s=60, alpha=0.0, zorder=10)

engage_text = fig.text(0.02, 0.02, "", color="white", fontsize=8, va="bottom", fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a0a22", edgecolor="#334466", alpha=0.88))
outcome_text = fig.text(0.32, 0.50, "", color="white", fontsize=26, fontweight="bold",
    ha="center", va="center", fontfamily="monospace", alpha=0.0,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#000000", alpha=0.0))

mis_text = ax_mis.text(0.04, 0.96, "init", transform=ax_mis.transAxes, color="#00ffcc", fontsize=7.5, va="top", fontfamily="monospace")
tgt_text = ax_tgt.text(0.04, 0.96, "init", transform=ax_tgt.transAxes, color="#ff9900", fontsize=7.5, va="top", fontfamily="monospace")
i1_text  = ax_i1.text( 0.04, 0.96, "init", transform=ax_i1.transAxes,  color="#ff2255", fontsize=7.5, va="top", fontfamily="monospace")
i2_text  = ax_i2.text( 0.04, 0.96, "init", transform=ax_i2.transAxes,  color="#ff66aa", fontsize=7.5, va="top", fontfamily="monospace")

ax_bar.set_title("Reward / Episode", color="white", fontsize=8, pad=4)
ax_bar.set_xlim(0.3, NUM_EPISODES+0.7); ax_bar.set_ylim(-bar_ylim, bar_ylim)
ax_bar.set_ylabel("Reward", color="gray", fontsize=7); ax_bar.axhline(0, color="#334466", lw=0.8)
bar_rects = [ax_bar.bar(i+1, 0, color="gray", width=0.65, alpha=0.25)[0] for i in range(NUM_EPISODES)]

ax_rt.set_title("Running Hit Rate", color="white", fontsize=8, pad=4)
ax_rt.set_xlim(0, NUM_EPISODES+1); ax_rt.set_ylim(-5, 108)
ax_rt.set_ylabel("Hit %", color="gray", fontsize=7)
ax_rt.axhline(60, color="#335533", lw=0.8, ls="--", alpha=0.7)
ax_rt.text(NUM_EPISODES-0.5, 63, "60%", color="#446644", fontsize=6)
hit_rate_line, = ax_rt.plot([], [], color="#00ff88", lw=2, marker="o", ms=3, zorder=3)

completed_eps = []

def _set_trail(old_l, mid_l, new_l, xs, ys, zs):
    n = len(xs); e1 = max(0, n-60); s2, e2 = max(0, n-60), max(0, n-20)
    old_l.set_data(xs[:e1], ys[:e1]);     old_l.set_3d_properties(zs[:e1])
    mid_l.set_data(xs[s2:e2], ys[s2:e2]); mid_l.set_3d_properties(zs[s2:e2])
    new_l.set_data(xs[max(0,n-20):], ys[max(0,n-20):]); new_l.set_3d_properties(zs[max(0,n-20):])

def _update_charts(ep_idx):
    ep = episodes_data[ep_idx]
    color = {"hit": "#00cc66", "intercepted": "#ff2255"}.get(ep["outcome"], "#cc3333")
    bar_rects[ep_idx].set_height(ep["reward"]); bar_rects[ep_idx].set_color(color); bar_rects[ep_idx].set_alpha(0.85)
    done = completed_eps
    hr_x = [i+1 for i in done]
    hr_y = [sum(1 for j in done[:k+1] if episodes_data[j]["outcome"]=="hit")/(k+1)*100 for k in range(len(done))]
    hit_rate_line.set_data(hr_x, hr_y)

def _missile_hud(f):
    c = "+" if f["closing_speed"] >= 0 else "-"
    return (f"  SPD   {f['v']:6.1f} m/s\n  ALT   {f['mz']:6.0f} m\n"
            f"  HDG   {f['heading_deg']:6.1f} °\n  CLB   {f['climb_deg']:6.1f} °\n"
            f"  FUEL  {f['fuel']*100:5.1f} %\n  POS   {f['mx']:5.0f} / {f['my']:5.0f} m\n"
            f"  TGT   {f['d_MT']:6.0f} m\n  INT1  {f['d_MI1']:6.0f} m\n"
            f"  INT2  {f['d_MI2']:6.0f} m\n  CLOS  {c}{abs(f['closing_speed']):5.1f} m/s")

def _target_hud(f):
    return (f"  SPD  {f['target_speed']:6.1f} m/s\n  ALT  {f['tz']:6.0f} m\n"
            f"  HDG  {f['target_heading']:6.1f} °\n  CLB  {f['target_climb']:6.1f} °\n"
            f"  VX   {f['target_vx']:+6.1f} m/s\n  VY   {f['target_vy']:+6.1f} m/s\n"
            f"  VZ   {f['target_vz']:+6.1f} m/s\n  POS  {f['tx']:5.0f} / {f['ty']:5.0f} m")

def _int_hud(f, n):
    active = f[f"int{n}_active"]; spd = f[f"int{n}_speed"]; iz = f[f"i{n}z"]
    vx = f[f"int{n}_vx"]; vy = f[f"int{n}_vy"]; vz = f[f"int{n}_vz"]; rng = f[f"d_MI{n}"]
    status = "ACTIVE  " if active else "NEUTRALISED"
    threat = "HIGH" if rng < 500 else ("MED" if rng < 1000 else "LOW")
    return (f"  STATUS  {status}\n  SPD     {spd:6.1f} m/s\n  ALT     {iz:6.0f} m\n"
            f"  VX      {vx:+6.1f} m/s\n  VY      {vy:+6.1f} m/s\n  VZ      {vz:+6.1f} m/s\n"
            f"  RANGE   {rng:6.0f} m\n  THREAT  {threat}")

def _engage_hud(f, ep_idx):
    return f" EP {ep_idx+1:2d}/{NUM_EPISODES}   STEP {f['step']:4d}   TIME {f['step']*env.dt:5.1f}s"

OUTCOME_STYLES = {
    "hit":         ("TARGET HIT",    "#00ff88", "#003322"),
    "intercepted": ("INTERCEPTED",   "#ff2255", "#330011"),
    "oob":         ("OUT OF BOUNDS", "#ffaa00", "#332200"),
    "timeout":     ("TIMEOUT",       "#aaaaaa", "#222222"),
}

def update(gi):
    kind, ep_idx, sub_idx = sequence[gi]
    ep = episodes_data[ep_idx]

    if kind == "pause" and sub_idx == 0 and ep_idx not in completed_eps:
        completed_eps.append(ep_idx); _update_charts(ep_idx)

    if kind == "play":
        f = ep["frames"][sub_idx]
        _set_trail(old_tm, mid_tm, new_tm, f["mx_trail"], f["my_trail"], f["mz_trail"])
        _set_trail(old_tt, mid_tt, new_tt, f["tx_trail"], f["ty_trail"], f["tz_trail"])
        if f["int1_active"]: _set_trail(old_ti1, mid_ti1, new_ti1, f["i1x_trail"], f["i1y_trail"], f["i1z_trail"])
        else:
            for ln in [old_ti1, mid_ti1, new_ti1]: ln.set_data([],[]); ln.set_3d_properties([])
        if f["int2_active"]: _set_trail(old_ti2, mid_ti2, new_ti2, f["i2x_trail"], f["i2y_trail"], f["i2z_trail"])
        else:
            for ln in [old_ti2, mid_ti2, new_ti2]: ln.set_data([],[]); ln.set_3d_properties([])

        missile_dot.set_data([f["mx"]],[f["my"]]); missile_dot.set_3d_properties([f["mz"]]); missile_dot.set_color("#00ffcc"); missile_dot.set_markersize(9)
        target_dot.set_data([f["tx"]],[f["ty"]]);  target_dot.set_3d_properties([f["tz"]]);  target_dot.set_color("#ff9900"); target_dot.set_markersize(14)
        if f["int1_active"]: int1_dot.set_data([f["i1x"]],[f["i1y"]]); int1_dot.set_3d_properties([f["i1z"]])
        else: int1_dot.set_data([],[]); int1_dot.set_3d_properties([])
        if f["int2_active"]: int2_dot.set_data([f["i2x"]],[f["i2y"]]); int2_dot.set_3d_properties([f["i2z"]])
        else: int2_dot.set_data([],[]); int2_dot.set_3d_properties([])

        explosion._offsets3d = int_kill._offsets3d = (np.array([env.L/2]), np.array([env.L/2]), np.array([env.H/2]))
        explosion.set_alpha(0.0); int_kill.set_alpha(0.0)
        outcome_text.set_alpha(0.0); outcome_text.get_bbox_patch().set_alpha(0.0)
        mis_text.set_text(_missile_hud(f)); tgt_text.set_text(_target_hud(f))
        i1_text.set_text(_int_hud(f,1));   i2_text.set_text(_int_hud(f,2))
        engage_text.set_text(_engage_hud(f, ep_idx))
        ax3d.set_title(f"Phase 3c  —  Episode {ep_idx+1} / {NUM_EPISODES}", color="white", fontsize=11, pad=8)

    elif kind == "pause":
        f = ep["frames"][-1]; fade = max(0.0, 1.0 - sub_idx / PAUSE_FRAMES); outcome = ep["outcome"]
        _set_trail(old_tm, mid_tm, new_tm, f["mx_trail"], f["my_trail"], f["mz_trail"])
        _set_trail(old_tt, mid_tt, new_tt, f["tx_trail"], f["ty_trail"], f["tz_trail"])
        label, color, bg = OUTCOME_STYLES.get(outcome, ("MISS", "#ff4444", "#330000"))

        if outcome == "hit":
            np.random.seed(ep_idx*7); angles = np.random.uniform(0,2*np.pi,30); radii = np.random.uniform(40,220,30)
            explosion._offsets3d = (np.clip(f["tx"]+radii*np.cos(angles),0,env.L),
                                    np.clip(f["ty"]+radii*np.sin(angles),0,env.L),
                                    np.clip(f["tz"]+np.random.uniform(-130,130,30),0,env.H))
            explosion.set_alpha(fade*0.9); missile_dot.set_data([],[]); missile_dot.set_3d_properties([])
            target_dot.set_color("#00ff88"); target_dot.set_markersize(22)
        elif outcome == "intercepted":
            np.random.seed(ep_idx*13); angles = np.random.uniform(0,2*np.pi,20); radii = np.random.uniform(30,150,20)
            int_kill._offsets3d = (np.clip(f["mx"]+radii*np.cos(angles),0,env.L),
                                   np.clip(f["my"]+radii*np.sin(angles),0,env.L),
                                   np.clip(f["mz"]+np.random.uniform(-80,80,20),0,env.H))
            int_kill.set_alpha(fade*0.9); missile_dot.set_data([],[]); missile_dot.set_3d_properties([])
        else:
            explosion._offsets3d = (np.array([env.L/2]), np.array([env.L/2]), np.array([env.H/2])); explosion.set_alpha(0.0)

        outcome_text.set_text(label); outcome_text.set_color(color)
        outcome_text.get_bbox_patch().set_facecolor(bg)
        outcome_text.set_alpha(fade); outcome_text.get_bbox_patch().set_alpha(0.85*fade)

    return (old_tm, mid_tm, new_tm, old_tt, mid_tt, new_tt,
            old_ti1, mid_ti1, new_ti1, old_ti2, mid_ti2, new_ti2,
            missile_dot, target_dot, int1_dot, int2_dot, explosion, int_kill,
            mis_text, tgt_text, i1_text, i2_text, engage_text, outcome_text,
            hit_rate_line, *bar_rects)

anim = FuncAnimation(fig, update, frames=len(sequence), interval=60, blit=False, repeat=False)
plt.tight_layout()
plt.show()
