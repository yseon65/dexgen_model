"""Train DexGen controller models (keypoint diffusion + inverse dynamics) on one data source.

One process trains four networks on the same batches:
  diffusion / inverse dynamics  x  keypoint version {tips (K=5), tips_mid (K=10)}

  ~/env_isaaclab6/bin/python train.py --name dexgen_src --policies nut_goal_normal nut_goal_slow15 \
      --max_steps 200000 --max_hours 6
  ~/env_isaaclab6/bin/python train.py --name dexscrew_src --policies nutbolt_student_scale ...

The packed cache (``cache/<name>``) is built on the first run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import queue
import threading
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dexgen_model.data import KP_VERSIONS, PackedWindows, pack_dataset, split_window
from dexgen_model.models import DDPMSchedule, InverseDynamics, KeypointDiffusion

ROOT = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--data_root", default=os.path.expanduser("~/IsaacLab/projects/dexdata/data/v1"))
p.add_argument("--policies", nargs="+", required=True, help="dexdata policy names of this data source")
p.add_argument("--versions", nargs="+", default=["tips", "tips_mid"], choices=list(KP_VERSIONS))
p.add_argument("--history", type=int, default=4, help="proprioception history steps (paper: 4)")
p.add_argument("--horizon", type=int, default=4, help="future keypoint steps; 4 x 0.05 s = 0.2 s (paper: T=2 at 10 Hz)")
p.add_argument("--batch", type=int, default=4096)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--lr_id", type=float, default=3e-4)
p.add_argument("--weight_decay", type=float, default=1e-4)
p.add_argument("--warmup", type=int, default=2000)
p.add_argument("--max_steps", type=int, default=200000)
p.add_argument("--max_hours", type=float, default=0.0, help="stop early after this wall time (0 = off)")
p.add_argument("--ema", type=float, default=0.999)
p.add_argument("--diffusion_steps", type=int, default=100)
p.add_argument("--ddim_steps", type=int, default=10)
p.add_argument("--val_every", type=int, default=2000)
p.add_argument("--save_every", type=int, default=10000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--no_in_memory", action="store_true", help="memory-map the packed rows instead of loading to RAM")
args = p.parse_args()

dev = "cuda:0"
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
cache = os.path.join(ROOT, "cache", args.name)
if not os.path.exists(os.path.join(cache, "meta.json")):
    pack_dataset(args.data_root, args.policies, cache)
with open(os.path.join(cache, "meta.json")) as f:
    meta = json.load(f)
assert meta["policies"] == args.policies, f"cache {cache} was built for {meta['policies']}"

t0 = time.time()
train = PackedWindows(cache, "train", args.history, args.horizon, in_memory=not args.no_in_memory)
val = PackedWindows(cache, "val", args.history, args.horizon, in_memory=False)
print(f"[data] train windows {len(train):,}  val windows {len(val):,}  ({time.time() - t0:.0f}s)", flush=True)
rng = np.random.default_rng(args.seed)

# ---- normalization statistics per keypoint version (from 500k training windows)
stat_w = train.sample(500_000, rng)
norm = {}
for v in args.versions:
    s, m, a = split_window(stat_w, args.history, v)
    norm[v] = {k: (torch.tensor(x.mean(0), device=dev), torch.tensor(x.std(0) + 1e-6, device=dev))
               for k, x in (("state", s), ("motion", m), ("action", a))}
del stat_w
val_w = torch.tensor(val.sample(8192, np.random.default_rng(123)), device=dev)


def normalize(v, w):
    s, m, a = split_window(w, args.history, v)
    n = norm[v]
    return ((s - n["state"][0]) / n["state"][1], (m - n["motion"][0]) / n["motion"][1],
            (a - n["action"][0]) / n["action"][1])


# ---- models
sched = DDPMSchedule(args.diffusion_steps, device=dev)
nets = {}
for v in args.versions:
    s, m, a = (x.shape[1] for x in split_window(val_w[:2], args.history, v))
    diff = KeypointDiffusion(m, s).to(dev)
    idm = InverseDynamics(s, m, a).to(dev)
    nets[v] = {"diff": diff, "ema": copy.deepcopy(diff).requires_grad_(False), "id": idm,
               "opt_d": torch.optim.AdamW(diff.parameters(), lr=args.lr, weight_decay=args.weight_decay),
               "opt_i": torch.optim.AdamW(idm.parameters(), lr=args.lr_id, weight_decay=args.weight_decay),
               "dims": {"state": s, "motion": m, "action": a}}
    print(f"[model] {v}: state {s} motion {m} action {a} | diffusion {sum(p.numel() for p in diff.parameters())/1e6:.1f}M "
          f"inverse dynamics {sum(p.numel() for p in idm.parameters())/1e6:.1f}M params", flush=True)


def lr_factor(step):
    if step < args.warmup:
        return (step + 1) / args.warmup
    prog = min(1.0, (step - args.warmup) / max(1, args.max_steps - args.warmup))
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))


# ---- background batch producer
batches: queue.Queue = queue.Queue(maxsize=8)


def producer():
    r = np.random.default_rng(args.seed + 1)
    while True:
        batches.put(torch.from_numpy(train.sample(args.batch, r)).pin_memory())


for _ in range(3):
    threading.Thread(target=producer, daemon=True).start()

run_dir = os.path.join(ROOT, "runs", f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.name}")
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(vars(args) | {"cache_meta": meta, "train_windows": len(train), "val_windows": len(val)}, f, indent=1)
writer = SummaryWriter(run_dir)


@torch.no_grad()
def validate(step):
    g = torch.Generator(device=dev).manual_seed(0)
    for v, n in nets.items():
        s, m, a = normalize(v, val_w)
        mode = torch.zeros(len(s), dtype=torch.long, device=dev)
        torch.manual_seed(1)
        vloss = sched.loss(n["ema"], m, s, mode).item()
        mu, _ = n["id"](s, m)
        sa = norm[v]["action"][1]
        id_err = ((mu - a) * sa).abs().mean().item()                       # rad, ground-truth motion
        sample = sched.ddim_sample(n["ema"], s, mode, args.ddim_steps, generator=g)
        sm = norm[v]["motion"]
        dx_pred = (sample * sm[1] + sm[0]).view(len(s), args.horizon, -1, 3)
        dx_gt = (m * sm[1] + sm[0]).view(len(s), args.horizon, -1, 3)
        end_err = (dx_pred[:, -1] - dx_gt[:, -1]).norm(dim=-1).mean().item() * 1e3   # mm at t+T
        end_zero = dx_gt[:, -1].norm(dim=-1).mean().item() * 1e3                      # "no motion" baseline
        mu_s, _ = n["id"](s, sample)
        id_sample_err = ((mu_s - a) * sa).abs().mean().item()             # rad, sampled motion
        writer.add_scalar(f"{v}/val_diffusion_loss", vloss, step)
        writer.add_scalar(f"{v}/val_id_mae_rad_gt_motion", id_err, step)
        writer.add_scalar(f"{v}/val_endpoint_err_mm", end_err, step)
        writer.add_scalar(f"{v}/val_endpoint_motion_mm", end_zero, step)
        writer.add_scalar(f"{v}/val_id_mae_rad_sampled_motion", id_sample_err, step)
        print(f"[val {step}] {v}: diff {vloss:.4f} | ID MAE {id_err:.4f} rad (gt dx) {id_sample_err:.4f} rad (sampled dx) "
              f"| endpoint err {end_err:.2f} mm (motion {end_zero:.2f} mm)", flush=True)


def save(tag):
    for v, n in nets.items():
        torch.save({"version": v, "args": vars(args), "dims": n["dims"],
                    "norm": {k: [x.cpu() for x in val_] for k, val_ in norm[v].items()},
                    "diffusion": n["diff"].state_dict(), "diffusion_ema": n["ema"].state_dict(),
                    "inverse_dynamics": n["id"].state_dict()}, os.path.join(run_dir, f"{v}_{tag}.pt"))


t_start, last_log = time.time(), time.time()
for step in range(args.max_steps):
    w = batches.get().to(dev, non_blocking=True)
    f = lr_factor(step)
    logs = {}
    for v, n in nets.items():
        s, m, a = normalize(v, w)
        mode = torch.zeros(len(s), dtype=torch.long, device=dev)
        for o, base in ((n["opt_d"], args.lr), (n["opt_i"], args.lr_id)):
            for gp in o.param_groups:
                gp["lr"] = base * f
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ld = sched.loss(n["diff"], m, s, mode)
        n["opt_d"].zero_grad(set_to_none=True)
        ld.backward()
        torch.nn.utils.clip_grad_norm_(n["diff"].parameters(), 1.0)
        n["opt_d"].step()
        with torch.no_grad():
            for pe, pm in zip(n["ema"].parameters(), n["diff"].parameters()):
                pe.lerp_(pm, 1 - args.ema)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, log_std = n["id"](s, m)
        li, mse = InverseDynamics.loss(mu.float(), log_std.float(), a)
        n["opt_i"].zero_grad(set_to_none=True)
        li.backward()
        torch.nn.utils.clip_grad_norm_(n["id"].parameters(), 1.0)
        n["opt_i"].step()
        logs[v] = (ld.item() if step % 100 == 0 else None, mse.item() if step % 100 == 0 else None)
    if step % 100 == 0:
        for v, (ld, mse) in logs.items():
            writer.add_scalar(f"{v}/train_diffusion_loss", ld, step)
            writer.add_scalar(f"{v}/train_id_mse_norm", mse, step)
        writer.add_scalar("lr_factor", f, step)
    if time.time() - last_log > 60:
        el = time.time() - t_start
        print(f"[train] step {step} | {step / el:.1f} it/s | {step * args.batch / el / 1e3:.0f}k samples/s | "
              f"epoch {step * args.batch / len(train):.2f} | " +
              " | ".join(f"{v}: diff {l[0]:.4f} id {l[1]:.4f}" for v, l in logs.items() if l[0] is not None), flush=True)
        last_log = time.time()
    if step % args.val_every == 0:
        validate(step)
    if step and step % args.save_every == 0:
        save(f"step{step}")
        save("last")
    if args.max_hours and time.time() - t_start > args.max_hours * 3600:
        print(f"[train] time budget reached at step {step}", flush=True)
        break
validate(step)
save("last")
print("[train] done", flush=True)
os._exit(0)
