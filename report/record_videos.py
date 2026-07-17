"""
Record the live demo of every training experiment to MP4 for the progress
report. Renders off-screen (rgb_array) with caption bars; no windows open.

Output: report/videos/*.mp4  (30 fps, H.264)

    python report/record_videos.py            # all videos
    python report/record_videos.py --only 01  # one video by prefix
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tmsd.metrics import w2_1d_exact
from tmsd.trainer import TMSDTrainer, TMSDConfig
from tmsd.wrappers import SkillDiscoveryEnv

OUT = ROOT / "report" / "videos"
OUT.mkdir(parents=True, exist_ok=True)
FPS = 30
SEED = 123

try:
    FONT = ImageFont.truetype("segoeui.ttf", 26)
    FONT_SM = ImageFont.truetype("segoeui.ttf", 20)
except OSError:
    FONT = FONT_SM = ImageFont.load_default()


def caption(frame: np.ndarray, title: str, sub: str = "") -> np.ndarray:
    """Add a dark caption bar above the frame."""
    h = 78
    img = Image.new("RGB", (frame.shape[1], frame.shape[0] + h), (16, 16, 20))
    img.paste(Image.fromarray(frame), (0, h))
    d = ImageDraw.Draw(img)
    d.text((14, 8), title, fill=(235, 235, 240), font=FONT)
    if sub:
        d.text((14, 44), sub, fill=(160, 200, 255), font=FONT_SM)
    return np.asarray(img)


def load(run_name):
    ckpt = ROOT / "runs" / run_name / "ckpt_latest.pt"
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg: TMSDConfig = saved["cfg"]
    measure = getattr(cfg, "measure", "1d")
    env = SkillDiscoveryEnv(measure=measure,
                            hist_bins=cfg.hist_bins if measure == "1d" else 64,
                            max_episode_steps=10**9,
                            randomize_terrain=True, render_mode="rgb_array")
    tr = TMSDTrainer(cfg, env.grid)
    tr.load(str(ckpt))
    return env, tr


def protocol_skills(dim, n=8):
    rng = np.random.default_rng(SEED)
    v = rng.normal(size=(n, dim))
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def rollout_frames(env, tr, z, steps, title, sub_fn=None, every=1):
    frames = []
    obs, hist, _ = env.reset(seed=SEED)
    h0 = env.eval_hist_1d()
    for t in range(steps):
        obs, hist, term, trunc, _ = env.step(tr.act(obs, z, deterministic=True))
        if t % every == 0:
            sub = sub_fn(env, h0) if sub_fn else ""
            frames.append(caption(env.render(), title, sub))
        if term or trunc:
            break
    return frames


def moved_sub(env, h0):
    m = w2_1d_exact(h0, env.eval_hist_1d(), env.grid_eval)
    return f"soil transported so far: {m:.2f} m (W2)   |   trained with ZERO rewards"


def write(name, frames):
    path = OUT / name
    imageio.mimwrite(path, frames, fps=FPS, codec="libx264", quality=8,
                     macro_block_size=None)
    print(f"{name}: {len(frames)} frames, {len(frames)/FPS:.0f}s")


# â”€â”€ 01: our skills â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def video_01():
    """The centerpiece: every sampled skill, back to back, same terrain."""
    env, tr = load("tmsd_w2_rt_d4")
    zs = protocol_skills(tr.cfg.skill_dim, 8)
    frames = []
    for k, z in enumerate(zs):
        zstr = ", ".join(f"{v:+.2f}" for v in z)
        frames += rollout_frames(
            env, tr, z, 200,
            f"Skill {k + 1}/8   z = ({zstr})   -- emergent behavior, zero rewards",
            moved_sub)
    env.close()
    write("01_ours_emergent_skills.mp4", frames)


# â”€â”€ 02: vanilla METRA arm dance, side by side with ours â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def video_02():
    env_a, tr_a = load("tmsd_w2_rt_d4")
    env_b, tr_b = load("abl_fullstate_temporal")
    za = protocol_skills(tr_a.cfg.skill_dim, 8)[0]
    zb = protocol_skills(tr_b.cfg.skill_dim, 8)[0]
    obs_a, _, _ = env_a.reset(seed=SEED)
    obs_b, _, _ = env_b.reset(seed=SEED)
    h0a = env_a.eval_hist_1d()
    h0b = env_b.eval_hist_1d()
    frames = []
    for t in range(260):
        obs_a, *_ = env_a.step(tr_a.act(obs_a, za, deterministic=True))
        obs_b, *_ = env_b.step(tr_b.act(obs_b, zb, deterministic=True))
        if True:
            fa, fb = env_a.render(), env_b.render()
            half = np.concatenate([fa, fb], axis=1)
            half = np.asarray(Image.fromarray(half).resize(
                (fa.shape[1], fa.shape[0] // 2)))
            ma = w2_1d_exact(h0a, env_a.eval_hist_1d(), env_a.grid_eval)
            mb = w2_1d_exact(h0b, env_b.eval_hist_1d(), env_b.grid_eval)
            frames.append(caption(
                half, "SAME training, one difference: what the objective can see",
                f"LEFT ours (sees soil only): {ma:.2f} m moved   |   "
                f"RIGHT vanilla METRA (sees own body): {mb:.2f} m -- it dances"))
    env_a.close(); env_b.close()
    write("02_metra_armdance_sidebyside.mp4", frames)


# â”€â”€ 03/04: DIAYN and DADS on soil (the failures, honestly) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _failure_video(run, label, fname):
    env, tr = load(run)
    zs = protocol_skills(tr.cfg.skill_dim, 8)
    frames = []
    for k in [0, 4]:
        frames += rollout_frames(
            env, tr, zs[k], 150,
            f"{label} -- skill {k}: the soil barely changes",
            moved_sub)
    env.close()
    write(fname, frames)


def video_03():
    _failure_video("mi_soil_s0",
                   "DIAYN-style (MI objective, soil-conditioned)",
                   "03_diayn_saturation.mp4")


def video_04():
    _failure_video("dads_soil_s0",
                   "DADS-style (dynamics objective, soil-conditioned)",
                   "04_dads_starvation.mp4")


# â”€â”€ 05: zero-shot shaping with goal overlay â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def video_05():
    env, tr = load("tmsd_w2_rt_d4")
    obs, hist, _ = env.reset(seed=SEED)
    h0 = env.eval_hist_1d()
    grid = env.grid_eval
    t = h0.astype(np.float64).copy()
    cut = (grid > 7.2) & (grid < 8.8)
    dst = (grid > 5.2) & (grid < 6.8)
    moved = t[cut].sum() * 0.85
    t[cut] *= 0.15
    t[dst] += moved / dst.sum()
    target = (np.clip(t, 0, None) / t.sum()).astype(np.float32)

    env.render()
    r = env.env._renderer
    r.overlay_polyline = list(zip(env.grid, env.hist_to_heights(target)))
    r.overlay_label = "COMMANDED PROFILE (red)"

    tgt_t = torch.as_tensor(target, device=tr.device).unsqueeze(0)
    frames = []
    z = None
    for step in range(420):
        if step % 10 == 0:
            with torch.no_grad():
                h = torch.as_tensor(env._hist(), device=tr.device).unsqueeze(0)
                dz = (tr.phi(tgt_t) - tr.phi(h)).squeeze(0).cpu().numpy()
            n = np.linalg.norm(dz)
            z = (dz / n if n > 1e-8 else tr.sample_skill()).astype(np.float32)
        obs, hist, *_ = env.step(tr.act(obs, z, deterministic=True))
        if True:
            w2 = w2_1d_exact(env.eval_hist_1d(), target, grid)
            frames.append(caption(
                env.render(),
                "ZERO-SHOT: 'dig a trench at x=7-9, pile spoil at x=5-7'",
                f"distance to commanded profile: {w2:.3f} m (W2) -- no goal training, no planner"))
    env.close()
    write("05_zeroshot_trench.mp4", frames)


# â”€â”€ 06: excavate & dump zone order â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def video_06():
    env, tr = load("tmsd_w2_rt_d4")
    dig, dump = (7.0, 9.0), (4.5, 6.5)
    obs, hist, _ = env.reset(seed=SEED)
    h0 = env.eval_hist_1d()
    grid = env.grid_eval
    t = h0.astype(np.float64).copy()
    src = (grid >= dig[0]) & (grid <= dig[1])
    dst = (grid >= dump[0]) & (grid <= dump[1])
    mv = t[src].sum() * 0.8
    t[src] *= 0.2
    t[dst] += mv / dst.sum()
    target = (np.clip(t, 0, None) / t.sum()).astype(np.float32)
    dig0 = float(h0[src].sum())

    env.render()
    r = env.env._renderer
    r.overlay_polyline = list(zip(env.grid, env.hist_to_heights(target)))
    r.overlay_label = "target profile"
    r.overlay_zones = [(dig[0], dig[1], (235, 90, 70), "DIG"),
                       (dump[0], dump[1], (90, 200, 110), "DUMP")]

    tgt_t = torch.as_tensor(target, device=tr.device).unsqueeze(0)
    frames = []
    z = None
    for step in range(500):
        if step % 10 == 0:
            with torch.no_grad():
                h = torch.as_tensor(env._hist(), device=tr.device).unsqueeze(0)
                dz = (tr.phi(tgt_t) - tr.phi(h)).squeeze(0).cpu().numpy()
            n = np.linalg.norm(dz)
            z = (dz / n if n > 1e-8 else tr.sample_skill()).astype(np.float32)
        obs, hist, *_ = env.step(tr.act(obs, z, deterministic=True))
        if True:
            cur = env.eval_hist_1d()
            cleared = max(0.0, 1.0 - float(cur[src].sum()) / max(dig0, 1e-9))
            frames.append(caption(
                env.render(),
                "ORDER: excavate the red zone, deposit in the green zone",
                f"dig zone cleared: {cleared:.0%}"))
    env.close()
    write("06_excavate_and_dump.mp4", frames)


# -- 07: every training run of every experiment, one reel each -----------
# (file names carry the experiment number so the folder sorts by story)
RUN_REELS = [
    # Experiment 1: pilot
    ("tmsd_w2_s0", "exp1_pilot_w2_500k", "EXP 1 pilot: W2, fixed terrain, dim 2"),
    # Experiment 2: metric ablation (working family, 3 metrics x 3 seeds)
    ("tmsd_w2_rt_d4", "exp2_w2_s0", "EXP 2: W2 metric, seed 0"),
    ("tmsd_w2_rt_d4_s1", "exp2_w2_s1", "EXP 2: W2 metric, seed 1"),
    ("tmsd_w2_rt_d4_s2", "exp2_w2_s2", "EXP 2: W2 metric, seed 2"),
    ("abl_euclid_rt_d4", "exp2_euclid_s0", "EXP 2: Euclidean metric, seed 0"),
    ("abl_euclid_rt_d4_s1", "exp2_euclid_s1", "EXP 2: Euclidean metric, seed 1"),
    ("abl_euclid_rt_d4_s2", "exp2_euclid_s2", "EXP 2: Euclidean metric, seed 2"),
    ("abl_temporal_rt_d4", "exp2_temporal_s0", "EXP 2: temporal metric, seed 0"),
    ("abl_temporal_rt_d4_s1", "exp2_temporal_s1", "EXP 2: temporal metric, seed 1"),
    ("abl_temporal_rt_d4_s2", "exp2_temporal_s2", "EXP 2: temporal metric, seed 2"),
    # Experiment 3: full-state baseline (= vanilla METRA), 3 seeds
    ("abl_fullstate_temporal", "exp3_metra_vanilla_s0", "EXP 3: vanilla METRA (full-state phi), seed 0"),
    ("abl_fullstate_temporal_s1", "exp3_metra_vanilla_s1", "EXP 3: vanilla METRA (full-state phi), seed 1"),
    ("abl_fullstate_temporal_s2", "exp3_metra_vanilla_s2", "EXP 3: vanilla METRA (full-state phi), seed 2"),
    # Experiment 4: 2D-measure trio
    ("tmsd_sw2_2d_s0", "exp4_slicedw2_2d", "EXP 4: sliced-W2 on 2D measure (starved)"),
    ("abl_euclid_2d_s0", "exp4_euclid_2d", "EXP 4: Euclidean on 2D measure"),
    ("abl_temporal_2d_s0", "exp4_temporal_2d", "EXP 4: temporal on 2D measure"),
    # Experiment 6: MI (DIAYN-style) 2x2
    ("mi_soil_s0", "exp6_diayn_soil_s0", "EXP 6: DIAYN-style + soil, seed 0"),
    ("mi_soil_s1", "exp6_diayn_soil_s1", "EXP 6: DIAYN-style + soil, seed 1"),
    ("mi_soil_s2", "exp6_diayn_soil_s2", "EXP 6: DIAYN-style + soil, seed 2"),
    ("mi_full_s0", "exp6_diayn_vanilla_s0", "EXP 6: DIAYN-style full-state, seed 0"),
    ("mi_full_s1", "exp6_diayn_vanilla_s1", "EXP 6: DIAYN-style full-state, seed 1"),
    ("mi_full_s2", "exp6_diayn_vanilla_s2", "EXP 6: DIAYN-style full-state, seed 2"),
    # Experiment 7: DADS 2x2
    ("dads_soil_s0", "exp7_dads_soil_s0", "EXP 7: DADS-style + soil, seed 0"),
    ("dads_soil_s1", "exp7_dads_soil_s1", "EXP 7: DADS-style + soil, seed 1"),
    ("dads_full_s0", "exp7_dads_vanilla_s0", "EXP 7: DADS-style full-state, seed 0"),
    ("dads_full_s1", "exp7_dads_vanilla_s1", "EXP 7: DADS-style full-state, seed 1"),
    # Experiment 8/9: stage-2 retrains
    ("stage2_1d_s0", "exp8_stage2_1d", "EXP 8: stage-2, user-mess-randomized resets"),
    ("stage2_comp_s0", "exp8_stage2_composite_s0", "EXP 8: stage-2 composite measure, seed 0"),
    ("stage2_comp_s1", "exp8_stage2_composite_s1", "EXP 8: stage-2 composite measure, seed 1"),
]


def video_07():
    """One reel per training run, all experiments: 4 sampled skills each."""
    for run, fname, desc in RUN_REELS:
        if not (ROOT / "runs" / run / "ckpt_latest.pt").exists():
            print(f"skip {run} (no checkpoint)")
            continue
        env, tr = load(run)
        zs = protocol_skills(tr.cfg.skill_dim, 8)
        frames = []
        for k in [0, 2, 5, 7]:
            zstr = ", ".join(f"{v:+.2f}" for v in zs[k])
            frames += rollout_frames(
                env, tr, zs[k], 150,
                f"{desc}   |   skill z = ({zstr})",
                moved_sub)
        env.close()
        write(f"{fname}.mp4", frames)


def video_08():
    """Experiment 9 control: goal-conditioned SAC trained from scratch,
    commanded toward a zone target -- shows it failed to learn."""
    from tmsd.gc_trainer import GCTrainer
    ckpt = ROOT / "runs" / "gc_s0" / "ckpt_latest.pt"
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    env = SkillDiscoveryEnv(hist_bins=64, max_episode_steps=10**9,
                            randomize_terrain=True, render_mode="rgb_array")
    tr = GCTrainer(saved["cfg"], env.grid_eval)
    tr.load(str(ckpt))
    obs, hist, _ = env.reset(seed=SEED)
    h0 = env.eval_hist_1d()
    grid = env.grid_eval
    t = h0.astype(np.float64).copy()
    src = (grid >= 7.0) & (grid <= 9.0)
    dst = (grid >= 4.5) & (grid <= 6.5)
    mv = t[src].sum() * 0.8
    t[src] *= 0.2
    t[dst] += mv / dst.sum()
    target = (np.clip(t, 0, None) / t.sum()).astype(np.float32)
    frames = []
    for step in range(300):
        obs, hist, *_ = env.step(tr.act(obs, target, deterministic=True))
        w2 = w2_1d_exact(env.eval_hist_1d(), target, grid)
        frames.append(caption(
            env.render(),
            "EXP 9 control: goal-conditioned SAC from scratch (no skill pretraining)",
            f"distance to goal: {w2:.3f} m -- 800k training steps, never learned to dig"))
    env.close()
    write("exp9_gc_from_scratch.mp4", frames)


def video_09():
    """The method, transparent: sim on the left; on the right the soil
    measure mu and the latent map phi drawing itself, skill by skill.
    Deterministic policy => a cheap no-render pre-pass fixes the phi axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env, tr = load("tmsd_w2_rt_d4")
    zs = protocol_skills(tr.cfg.skill_dim, 8)
    colors = ["#2a78d6", "#e34948", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4"]

    def phi_of(h):
        with torch.no_grad():
            return tr.phi(torch.as_tensor(h, dtype=torch.float32,
                                          device=tr.device).unsqueeze(0)
                          ).cpu().numpy()[0]

    # pre-pass over ALL protocol skills: phi trajectories + physical movement
    pre_all = {}
    moved_all = {}
    for k in range(len(zs)):
        obs, hist, _ = env.reset(seed=SEED)
        h0 = env.eval_hist_1d()
        p0 = phi_of(hist)
        traj = [p0 - p0]
        for _ in range(200):
            obs, hist, term, trunc, _ = env.step(
                tr.act(obs, zs[k], deterministic=True))
            traj.append(phi_of(hist) - p0)
            if term or trunc:
                break
        pre_all[k] = np.stack(traj)
        moved_all[k] = w2_1d_exact(h0, env.eval_hist_1d(), env.grid_eval)
    # show the 5 strongest movers, then 1 idle direction (captioned as such)
    ranked = sorted(moved_all, key=moved_all.get, reverse=True)
    picks = ranked[:5] + [ranked[-1]]
    idle_k = ranked[-1]
    pre = [pre_all[k] for k in picks]
    allp = np.concatenate(pre)
    xlim = (allp[:, 0].min() - 0.08, allp[:, 0].max() + 0.08)
    ylim = (allp[:, 1].min() - 0.08, allp[:, 1].max() + 0.08)

    fig, (ax_mu, ax_phi) = plt.subplots(2, 1, figsize=(6.4, 8.0), dpi=100)
    fig.patch.set_facecolor("#101014")
    frames = []
    done_trajs = []

    def panel(hist0, hist, traj, col, label):
        for ax in (ax_mu, ax_phi):
            ax.clear()
            ax.set_facecolor("#16161a")
            for s in ax.spines.values():
                s.set_color("#444")
            ax.tick_params(colors="#999", labelsize=8)
        w = env.grid[1] - env.grid[0]
        ax_mu.bar(env.grid, hist0, width=w, color="#555", label="initial")
        ax_mu.bar(env.grid, hist, width=w, color=col, alpha=0.75, label="now")
        ax_mu.set_title("$\\mu$: soil mass distribution (all the objective sees)",
                        color="#ddd", fontsize=10)
        ax_mu.legend(fontsize=7, loc="upper right", facecolor="#222",
                     labelcolor="#ccc", edgecolor="#444")
        for P, c in done_trajs:
            ax_phi.plot(P[:, 0], P[:, 1], color=c, lw=1.2, alpha=0.45)
            ax_phi.scatter(P[-1, 0], P[-1, 1], color=c, marker="*", s=60,
                           alpha=0.6)
        P = np.stack(traj)
        ax_phi.plot(P[:, 0], P[:, 1], color=col, lw=2.4)
        ax_phi.scatter(P[-1, 0], P[-1, 1], color=col, marker="*", s=150,
                       zorder=5, edgecolors="white", linewidths=0.6)
        ax_phi.scatter(0, 0, color="white", s=25, zorder=5)
        ax_phi.set_xlim(*xlim)
        ax_phi.set_ylim(*ylim)
        ax_phi.set_title(f"$\\phi(\\mu)$: the learned transport map -- {label}",
                         color="#ddd", fontsize=10)
        ax_phi.set_xlabel("$\\phi_1$", color="#999")
        ax_phi.set_ylabel("$\\phi_2$", color="#999")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        return buf

    for i, (k, col) in enumerate(zip(picks, colors)):
        obs, hist, _ = env.reset(seed=SEED)
        hist0 = hist.copy()
        p0 = phi_of(hist)
        traj = [p0 - p0]
        for t in range(200):
            obs, hist, term, trunc, _ = env.step(
                tr.act(obs, zs[k], deterministic=True))
            traj.append(phi_of(hist) - p0)
            if t % 2 == 0:
                sim = env.render()
                is_idle = (k == idle_k)
                lbl = (f"skill {i + 1}/{len(picks)}"
                       + (" (an infeasible direction: it honestly idles)"
                          if is_idle else ""))
                side = panel(hist0, hist, traj, col, lbl)
                side = np.asarray(Image.fromarray(side).resize(
                    (int(side.shape[1] * sim.shape[0] / side.shape[0]),
                     sim.shape[0])))
                combo = np.concatenate([sim, side], axis=1)
                sub = ("this z commands physically infeasible transport -- "
                       "zero reward available, so doing NOTHING is optimal. "
                       "(DIAYN would fake a 'skill' here; ours cannot.)"
                       if is_idle else
                       "left: simulation | top right: what the objective sees"
                       " | bottom right: the map drawing itself, skill by skill")
                frames.append(caption(
                    combo,
                    f"Our method, transparent: skill {i + 1}/{len(picks)} "
                    "-- the world, its measure, and the latent map",
                    sub))
            if term or trunc:
                break
        done_trajs.append((np.stack(traj), col))
    plt.close(fig)
    env.close()
    write("09_method_with_latent.mp4", frames)


def video_10():
    """Video-09-style transparency for the BASELINES: sim + mu + each
    method's own latent map, with its failure mechanism captioned live."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baselines = [
        ("abl_fullstate_temporal", "10_metra_vanilla_latent",
         "VANILLA METRA (full-state phi)",
         "watch its map RACE while the soil never changes -- "
         "latent 'diversity' decoupled from the world"),
        ("mi_soil_s0", "10_diayn_latent",
         "DIAYN-style (MI + soil phi)",
         "the map barely moves: the discriminator saturated long ago -- "
         "identifiable without doing anything"),
        ("dads_soil_s0", "10_dads_latent",
         "DADS-style (dynamics + soil phi)",
         "the map is frozen: slow soil made every skill equally "
         "predictable -- reward starved to exactly zero"),
    ]
    colors = ["#2a78d6", "#e34948", "#1baf7a", "#eda100"]
    picks = [0, 2, 5, 7]

    for run, fname, title_tag, mech in baselines:
        env, tr = load(run)
        phi_on_obs = getattr(tr.cfg, "phi_input", "hist") == "obs"
        zs = protocol_skills(tr.cfg.skill_dim, 8)

        def phi_of(x):
            with torch.no_grad():
                return tr.phi(torch.as_tensor(x, dtype=torch.float32,
                                              device=tr.device).unsqueeze(0)
                              ).cpu().numpy()[0]

        # pre-pass: fix the latent axes to this model's own scale
        pre = []
        for k in picks:
            obs, hist, _ = env.reset(seed=SEED)
            x = obs if phi_on_obs else hist
            p0 = phi_of(x)
            traj = [p0 - p0]
            for _ in range(150):
                obs, hist, term, trunc, _ = env.step(
                    tr.act(obs, zs[k], deterministic=True))
                x = obs if phi_on_obs else hist
                traj.append(phi_of(x) - p0)
                if term or trunc:
                    break
            pre.append(np.stack(traj))
        allp = np.concatenate(pre)
        span = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]), 1e-6)
        pad = 0.08 * span
        xlim = (allp[:, 0].min() - pad, allp[:, 0].max() + pad)
        ylim = (allp[:, 1].min() - pad, allp[:, 1].max() + pad)

        fig, (ax_mu, ax_phi) = plt.subplots(2, 1, figsize=(6.4, 8.0), dpi=100)
        fig.patch.set_facecolor("#101014")
        frames = []
        done_trajs = []

        for i, (k, col) in enumerate(zip(picks, colors)):
            obs, hist, _ = env.reset(seed=SEED)
            hist0 = hist.copy()
            h0 = env.eval_hist_1d()
            x = obs if phi_on_obs else hist
            p0 = phi_of(x)
            traj = [p0 - p0]
            for t in range(150):
                obs, hist, term, trunc, _ = env.step(
                    tr.act(obs, zs[k], deterministic=True))
                x = obs if phi_on_obs else hist
                traj.append(phi_of(x) - p0)
                if t % 2 == 0:
                    sim = env.render()
                    for ax in (ax_mu, ax_phi):
                        ax.clear()
                        ax.set_facecolor("#16161a")
                        for s in ax.spines.values():
                            s.set_color("#444")
                        ax.tick_params(colors="#999", labelsize=8)
                    w = env.grid[1] - env.grid[0] if not phi_on_obs else 0.2
                    g = env.grid if not phi_on_obs else env.grid_eval
                    cur = hist if not phi_on_obs else env.eval_hist_1d()
                    ax_mu.bar(env.grid_eval, h0, width=0.2, color="#555",
                              label="initial")
                    ax_mu.bar(env.grid_eval, env.eval_hist_1d(), width=0.2,
                              color=col, alpha=0.75, label="now")
                    ax_mu.set_title("the soil (has it changed at all?)",
                                    color="#ddd", fontsize=10)
                    ax_mu.legend(fontsize=7, loc="upper right",
                                 facecolor="#222", labelcolor="#ccc",
                                 edgecolor="#444")
                    for P, c in done_trajs:
                        ax_phi.plot(P[:, 0], P[:, 1], color=c, lw=1.2,
                                    alpha=0.45)
                    P = np.stack(traj)
                    ax_phi.plot(P[:, 0], P[:, 1], color=col, lw=2.4)
                    ax_phi.scatter(P[-1, 0], P[-1, 1], color=col, marker="*",
                                   s=150, zorder=5, edgecolors="white",
                                   linewidths=0.6)
                    ax_phi.scatter(0, 0, color="white", s=25, zorder=5)
                    ax_phi.set_xlim(*xlim)
                    ax_phi.set_ylim(*ylim)
                    ax_phi.set_title(
                        f"its OWN latent map (axis span: {span:.2g})",
                        color="#ddd", fontsize=10)
                    fig.tight_layout()
                    fig.canvas.draw()
                    side = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
                    side = np.asarray(Image.fromarray(side).resize(
                        (int(side.shape[1] * sim.shape[0] / side.shape[0]),
                         sim.shape[0])))
                    combo = np.concatenate([sim, side], axis=1)
                    frames.append(caption(
                        combo, f"{title_tag} -- skill {i + 1}/{len(picks)}",
                        mech))
                if term or trunc:
                    break
            done_trajs.append((np.stack(traj), col))
        plt.close(fig)
        env.close()
        write(f"{fname}.mp4", frames)


VIDEOS = {"01": video_01, "02": video_02, "03": video_03,
          "04": video_04, "05": video_05, "06": video_06,
          "07": video_07, "08": video_08, "09": video_09,
          "10": video_10}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", type=str, default=None)
    args = p.parse_args()
    todo = ([args.only] if args.only else sorted(VIDEOS))
    for key in todo:
        print(f"--- recording {key} ---")
        VIDEOS[key]()
    print(f"\nvideos -> {OUT}")


if __name__ == "__main__":
    main()

