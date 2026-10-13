"""Closed-loop simulation evaluation of a DexGen controller (paper Sec. IV: DexGen as a low-level controller).

The env is the one the training data came from, under the collection conditions (dexdata/logs/collect_v1.sh):
  --task nut_goal   dexgen NutGoalEnv, free nut, size 0.7-1.3, full DR      (policy: dexgen D, rsl-rl)
  --task nutbolt    dexscrew XHandHoraNutBolt, trinut, size 0.7-1.25         (policy: dexscrew student)

Conditions (run one after another, all envs reset in between):
  policy           the RL policy alone (reference)
  noisy            the RL policy with degraded actions  a + noise * N(0, 1)          (weak high-level controller)
  dexgen           DexGen alone, unguided (samples a motion from p(dx | state))
  policy+dexgen    DexGen guided by the RL policy's command
  noisy+dexgen     DexGen guided by the degraded policy's command               (paper: DexGen as a safety layer)

Guidance: a policy action a gives the joint-target change dq = action_scale * a. Assuming the command is held
for the horizon, the commanded keypoint motion is dx_i = i * J(q) dq (i = 1..T, J = keypoint Jacobian in the wrist
frame). DexGen output: joint-target change da -> env action da / action_scale (clipped to [-1, 1]).

Metrics (per condition, over all envs and --steps control steps):
  drops/min        episodes ending by drop or contact loss per env-minute
  ep_len_s         mean length of finished episodes
  end reasons      share of finished episodes (timeout / drop / low_contact / stagnation / other)
  nut_goal: goals/min (goal_reached_event), |w_z| mean object twist rate about the wrist z axis [rad/s]
  nutbolt : nut_rad_s  mean nut joint velocity [rad/s] (positive = tightening direction of the task reward)
  motion_mm        mean realized 1-step keypoint motion [mm] (scale for the errors below)
  jacobian_mm      |realized 1-step motion - J(q) dq| (accuracy of the linear guide model)
  DexGen  : track_mm   |realized 1-step keypoint motion - DexGen sampled dx_1|, mean over keypoints [mm]
            guide_mm   |realized 1-step keypoint motion - commanded dx_1| (guided conditions)

  ~/env_isaaclab6/bin/python scripts/eval_sim.py --task nut_goal \
      --ckpt runs/2026-10-11_12-22-15_dexgen_src/tips_mid_last.pt --num_envs 512 --steps 1200
  ~/env_isaaclab6/bin/python scripts/eval_sim.py --task nutbolt \
      --ckpt runs/2026-10-11_12-22-15_dexscrew_src/tips_mid_last.pt --num_envs 512 --steps 1200
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEXGEN_ROOT = os.path.expanduser("~/IsaacLab/projects/dexgen")
DEXSCREW_ROOT = os.path.expanduser("~/IsaacLab/projects/dexscrew")
POLICIES = {
    "nut_goal": f"{DEXGEN_ROOT}/outputs/xhand_nut_goal/2026-10-10_23-26-04_goal_scale0713_ft_initq/model_14250.pt",
    "nutbolt": f"{DEXSCREW_ROOT}/outputs/XHandHoraNutBolt_student_padapt/lab_scale07125_t1413/stage2_nn/model_best.ckpt",
}
ALL_CONDITIONS = ["policy", "noisy", "dexgen", "policy+dexgen", "noisy+dexgen"]

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=list(POLICIES))
p.add_argument("--ckpt", required=True, help="DexGen checkpoint (train.py output, *_last.pt)")
p.add_argument("--policy_ckpt", default=None, help="RL policy (default: the one used for data collection)")
p.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS, choices=ALL_CONDITIONS)
p.add_argument("--noise", type=float, default=0.5, help="action noise std of the degraded policy (actions in [-1, 1])")
p.add_argument("--guide_scale", type=float, default=0.3)
p.add_argument("--ddim_steps", type=int, default=10)
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--steps", type=int, default=1200, help="control steps per condition (20 Hz)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--policy_warmup", type=int, default=0,
               help="DexGen conditions: the clean RL policy acts for the first N steps of every episode "
                    "(data collected with skip_first_steps=N never covers the reset transient)")
p.add_argument("--out", default=None, help="json result path (default: next to the DexGen checkpoint)")
p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
args, overrides = p.parse_known_args()
app = AppLauncher(headless=args.headless).app

import torch  # noqa: E402

sys.path.insert(0, ROOT)
from dexgen_model.controller import DexGenController, KeypointJacobian  # noqa: E402
from dexgen_model.data import DEXDATA_ROOT  # noqa: E402

sys.path.insert(0, DEXDATA_ROOT)
from dexdata import schema  # noqa: E402
from dexdata.adapters import DexgenAdapter, DexscrewAdapter  # noqa: E402

dev = "cuda:0"
torch.manual_seed(args.seed)
policy_ckpt = os.path.expanduser(args.policy_ckpt or POLICIES[args.task])


# ------------------------------------------------------------------------------------------------ env + policy
def make_nut_goal():
    import importlib.metadata

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    sys.path.insert(0, DEXGEN_ROOT)
    from dexgen.nut_goal_env import NutGoalEnv, NutGoalEnvCfg
    from dexgen.ppo_cfg import NutSpinPpoCfg

    cfg = NutGoalEnvCfg()
    cfg.seed, cfg.scene.num_envs, cfg.sim.device = args.seed, args.num_envs, dev
    cfg.scale_range, cfg.dr_ramp_steps = (0.7, 1.3), 0            # collection conditions (collect_v1.sh)
    agent_cfg = NutSpinPpoCfg()
    agent_cfg.seed, agent_cfg.device = args.seed, dev
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    raw = NutGoalEnv(cfg)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=dev)
    runner.load(policy_ckpt, map_location=dev)
    pol = runner.get_inference_policy(device=dev)
    adapter = DexgenAdapter(raw, env)

    def reset_all():
        raw.reset()
        return env.get_observations()

    return dict(adapter=adapter, hand=raw.scene["hand"], action_scale=cfg.action_scale, reset_all=reset_all,
                first_obs=env.get_observations,
                policy=lambda obs: pol(obs).clamp(-1.0, 1.0), step_dt=raw.step_dt, action_mask=None)


def make_nutbolt():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    sys.path.insert(0, DEXSCREW_ROOT)
    from dexscrew.algo.ppo.padapt import ProprioAdapt
    from dexscrew.tasks import isaaclab_task_map
    from dexscrew.utils.reformat import omegaconf_to_dict

    for name, fn in (("eq", lambda x, y: x.lower() == y.lower()), ("contains", lambda x, y: x.lower() in y.lower()),
                     ("if", lambda pred, a, b: a if pred else b),
                     ("resolve_default", lambda default, arg: default if arg == "" else arg)):
        OmegaConf.register_new_resolver(name, fn)
    # student preset + size range of the collection run (dexdata/scripts/collect_dexscrew.py, collect_v1.sh)
    env_overrides = ["task.env.object.type=screw_trinut", "task.env.reset_dist_threshold=0.12",
                     "task.env.randomization.obs_noise_t_scale=0.01", "task.env.randomization.obs_noise_e_scale=0.02",
                     "task.env.randomization.randomizeScaleList=[0.7,0.8,0.9,1.0,1.1,1.2,1.25]",
                     "+task.env.keepNutHeight=True"] + overrides
    with initialize_config_dir(config_dir=os.path.join(DEXSCREW_ROOT, "configs"), version_base=None):
        config = compose("config", overrides=["task=XHandHoraNutBolt", f"task.env.numEnvs={args.num_envs}", "test=True",
                                              f"seed={args.seed}", "train.algo=ProprioAdapt",
                                              "train.ppo.proprio_adapt=True"] + env_overrides)
    task_cfg = omegaconf_to_dict(config.task)
    task_cfg["seed"] = args.seed
    env = isaaclab_task_map["XHandHoraNutBolt"](task_cfg, config.sim_device, 0, True)
    agent = ProprioAdapt(env, os.path.join(ROOT, "runs", ".agent_scratch"), full_config=config)
    agent.restore_test(policy_ckpt)
    agent.set_eval()

    def pol(obs):
        pc = obs["point_cloud_info"]
        if agent.normalize_point_cloud:
            pc = agent.point_cloud_mean_std(pc.reshape(-1, 3)).reshape((pc.shape[0], -1, 3))
        inp = {"obs": agent.running_mean_std(obs["obs"]), "proprio_hist": agent.sa_mean_std(obs["proprio_hist"]),
               "point_cloud_info": pc}
        return agent.model.act_inference(inp)[0].clamp(-1.0, 1.0)

    def reset_all():
        env.reset_buf[:] = 1
        return env.reset()

    mask = torch.ones(schema.N_J, device=dev)
    mask[5:9] = 0.0  # the env zeroes pinky/ring actions for the nut-bolt init pose (XHandHoraLab.step)
    return dict(adapter=DexscrewAdapter(env), hand=env.hand, action_scale=env.action_scale, reset_all=reset_all,
                first_obs=env.reset,
                policy=pol, step_dt=env.control_freq_inv * env.dt, action_mask=mask)


E = make_nut_goal() if args.task == "nut_goal" else make_nutbolt()
adapter, scale, dt = E["adapter"], E["action_scale"], E["step_dt"]
ctrl = DexGenController(os.path.expanduser(args.ckpt), args.num_envs, dev, args.ddim_steps, args.guide_scale, args.seed)
jac = KeypointJacobian(E["hand"], [schema.KEYPOINT_BODIES[i] for i in schema.KEYPOINT_SETS[ctrl.version]])
steps_t = torch.arange(1, ctrl.horizon + 1, device=dev, dtype=torch.float32)[None, :, None, None]
R = schema.END_REASONS
REASON_NAMES = {v: k for k, v in R.items()}


@torch.no_grad()
def run(cond: str, first: bool) -> dict:
    obs = E["first_obs"]() if first else E["reset_all"]()
    ctrl.fresh[:] = True
    n = args.num_envs
    ends = torch.zeros(len(R), device=dev)
    ep_steps = torch.zeros(n, device=dev)
    finished_len, goals, wz, nut_v = [], 0.0, 0.0, 0.0
    track, guide_err, n_track = 0.0, 0.0, 0
    jac_err, motion, n_motion = 0.0, 0.0, 0
    prev_kp = prev_q = prev_pred = prev_guide = None
    prev_warm = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(args.steps):
        state = adapter.read_state()
        kp = ctrl.keypoints(state)
        if prev_kp is not None:  # 1-step tracking of the previous command (envs not reset in between)
            ok = ~just_reset
            ok_t = ok & ~prev_warm                                                            # DexGen acted
            real = (kp - prev_kp)[ok]
            lin = jac.displacement(state["joint_pos"] - prev_q)[ok]                          # first-order J dq
            jac_err += (real - lin).norm(dim=-1).mean(-1).sum().item()
            motion += real.norm(dim=-1).mean(-1).sum().item()
            n_motion += int(ok.sum())
            if prev_pred is not None:
                real_t = (kp - prev_kp)[ok_t]
                track += (real_t - prev_pred[ok_t]).norm(dim=-1).mean(-1).sum().item()
                n_track += int(ok_t.sum())
                if prev_guide is not None:
                    guide_err += (real_t - prev_guide[ok_t]).norm(dim=-1).mean(-1).sum().item()
        prev_pred = prev_guide = None
        if cond in ("policy", "noisy") or "+" in cond:
            a = E["policy"](obs)
            if cond.startswith("noisy"):
                a = (a + args.noise * torch.randn_like(a)).clamp(-1.0, 1.0)
            if E["action_mask"] is not None:
                a = a * E["action_mask"]
        if "dexgen" in cond:
            ctrl.observe(state)
            guide = None
            if "+" in cond:
                guide = steps_t * jac.displacement(scale * a)[:, None]                     # (N,T,K,3)
                prev_guide = guide[:, 0]
            da, dx = ctrl.act(guide)
            prev_pred = dx[:, 0]
            a = (da / scale).clamp(-1.0, 1.0)
            prev_warm = ep_steps < args.policy_warmup
            if prev_warm.any():
                a_pol = E["policy"](obs)
                if E["action_mask"] is not None:
                    a_pol = a_pol * E["action_mask"]
                a = torch.where(prev_warm[:, None], a_pol, a)
        obs, post, done, reason = adapter.step(a)
        # task progress
        if args.task == "nut_goal":
            goals += post["goal_reached_event"].float().sum().item()
            wz += state["object_angvel_wrist"][:, 2].abs().sum().item()
        else:
            nut_v += state["object_dof_vel"][:, 0].sum().item()
        ep_steps += 1
        d = done.bool()
        if d.any():
            ends += torch.bincount(reason[d].long(), minlength=len(R)).float()
            finished_len.append(ep_steps[d].clone())
            ep_steps[d] = 0
        ctrl.mark_reset(d)
        just_reset = d
        prev_kp, prev_q = kp, state["joint_pos"]
    env_min = n * args.steps * dt / 60.0
    lens = torch.cat(finished_len) if finished_len else torch.zeros(0, device=dev)
    n_end = max(1.0, ends.sum().item())
    res = {"condition": cond, "episodes_finished": int(ends.sum().item()),
           "drops_per_min": float((ends[R["drop"]] + ends[R["low_contact"]]).item() / env_min),
           "ep_len_s": float(lens.mean().item() * dt) if len(lens) else float("nan"),
           "end_reasons": {REASON_NAMES[i]: round(ends[i].item() / n_end, 3) for i in range(len(R)) if ends[i] > 0}}
    if args.task == "nut_goal":
        res["goals_per_min"] = goals / env_min
        res["twist_rate_rad_s"] = wz / (n * args.steps)
    else:
        res["nut_rad_s"] = nut_v / (n * args.steps)
    res["motion_mm"] = motion / max(1, n_motion) * 1e3
    res["jacobian_mm"] = jac_err / max(1, n_motion) * 1e3
    if n_track:
        res["track_mm"] = track / n_track * 1e3
        if "+" in cond:
            res["guide_mm"] = guide_err / n_track * 1e3
    return res


results = {"task": args.task, "ckpt": args.ckpt, "policy_ckpt": policy_ckpt, "version": ctrl.version,
           "noise": args.noise, "guide_scale": args.guide_scale, "ddim_steps": args.ddim_steps,
           "num_envs": args.num_envs, "steps": args.steps, "policy_warmup": args.policy_warmup, "overrides": overrides, "conditions": []}
for ci, cond in enumerate(args.conditions):
    t0 = time.time()
    r = run(cond, ci == 0)
    r["wall_s"] = round(time.time() - t0, 1)
    results["conditions"].append(r)
    print(f"[eval] {json.dumps(r)}", flush=True)

out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                               f"eval_{args.task}_{ctrl.version}_n{args.noise}_g{args.guide_scale}"
                               + (f"_w{args.policy_warmup}" if args.policy_warmup else "") + ".json")
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print(f"[eval] saved {out}", flush=True)
os._exit(0)
