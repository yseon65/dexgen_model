"""Sample-quality check of a trained checkpoint on validation windows.

For each validation state, draw N DDIM samples and report the endpoint error (keypoint offset at t+T, mm)
of the first sample, the mean over samples, and the best of N. A large gap between single and best-of-N
errors means the data has several modes (e.g. both rotation directions) that the model represents.

  ~/env_isaaclab6/bin/python scripts/eval_samples.py runs/<run>/tips_mid_step20000.pt --n 16
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dexgen_model.data import PackedWindows, split_window  # noqa: E402
from dexgen_model.models import DDPMSchedule, InverseDynamics, KeypointDiffusion  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("--n", type=int, default=16)
p.add_argument("--states", type=int, default=2048)
p.add_argument("--ddim_steps", type=int, default=10)
args = p.parse_args()
dev = "cuda:0"

ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
a, v, dims = ck["args"], ck["version"], ck["dims"]
norm = {k: [x.to(dev) for x in val] for k, val in ck["norm"].items()}
diff = KeypointDiffusion(dims["motion"], dims["state"]).to(dev).eval()
diff.load_state_dict(ck["diffusion_ema"])
idm = InverseDynamics(dims["state"], dims["motion"], dims["action"]).to(dev).eval()
idm.load_state_dict(ck["inverse_dynamics"])
sched = DDPMSchedule(a["diffusion_steps"], device=dev)

val = PackedWindows(os.path.join(ROOT, "cache", a["name"]), "val", a["history"], a["horizon"], in_memory=False)
w = torch.tensor(val.sample(args.states, np.random.default_rng(7)), device=dev)
s, m, act = split_window(w, a["history"], v)
s = (s - norm["state"][0]) / norm["state"][1]
gt = m.view(len(m), a["horizon"], -1, 3)
gt_act = act
mode = torch.zeros(len(s), dtype=torch.long, device=dev)

ends, act_err = [], []
g = torch.Generator(device=dev).manual_seed(0)
with torch.no_grad():
    for _ in range(args.n):
        x = sched.ddim_sample(diff, s, mode, args.ddim_steps, generator=g)
        mu, _ = idm(s, x)
        act_err.append(((mu * norm["action"][1] + norm["action"][0]) - gt_act).abs().mean(-1))
        dx = (x * norm["motion"][1] + norm["motion"][0]).view_as(gt)
        ends.append((dx[:, -1] - gt[:, -1]).norm(dim=-1).mean(-1) * 1e3)            # (B,) mm
ends, act_err = torch.stack(ends), torch.stack(act_err)                              # (N, B)
res = {"ckpt": args.ckpt, "version": v, "n": args.n,
       "motion_mm": float(gt[:, -1].norm(dim=-1).mean() * 1e3),
       "endpoint_single_mm": float(ends[0].mean()), "endpoint_mean_mm": float(ends.mean()),
       "endpoint_best_of_n_mm": float(ends.min(0).values.mean()),
       "id_mae_single_rad": float(act_err[0].mean()),
       "id_mae_best_of_n_rad": float(act_err.min(0).values.mean())}
print(json.dumps(res), flush=True)
os._exit(0)
