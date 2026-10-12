"""Closed-loop DexGen controller: proprioception history -> keypoint motion (diffusion) -> joint-target change (ID).

Used by scripts/eval_sim.py. Inputs follow the dexdata schema (dexdata.adapters ``read_state()`` dict):
keypoint_pos_wrist (N,17,3), joint_pos, joint_target, control_error (N,12).

At each control step
    dx  ~ p(dx | state)                       (DDIM; optionally guided toward a commanded motion, paper Eq. 3)
    da  = E[a | state, dx]                    (inverse dynamics mean; a = applied_target - joint_target)
and the env applies joint_target + da. History at an episode start is filled with the first state
(training windows never contain padding; see dexgen_model/data.py).
"""

from __future__ import annotations

import sys

import torch

from .data import DEXDATA_ROOT
from .models import DDPMSchedule, InverseDynamics, KeypointDiffusion

sys.path.insert(0, DEXDATA_ROOT)
from dexdata import schema  # noqa: E402


class DexGenController:
    def __init__(self, ckpt_path: str, num_envs: int, device: str = "cuda:0", ddim_steps: int = 10,
                 guide_scale: float = 0.0, seed: int = 0):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        a, dims = ck["args"], ck["dims"]
        self.version, self.history, self.horizon = ck["version"], a["history"], a["horizon"]
        self.kp_idx = torch.as_tensor(schema.KEYPOINT_SETS[self.version], device=device)
        self.k = len(self.kp_idx)
        self.norm = {k: [x.to(device) for x in v] for k, v in ck["norm"].items()}
        self.diff = KeypointDiffusion(dims["motion"], dims["state"]).to(device).eval()
        self.diff.load_state_dict(ck["diffusion_ema"])
        self.idm = InverseDynamics(dims["state"], dims["motion"], dims["action"]).to(device).eval()
        self.idm.load_state_dict(ck["inverse_dynamics"])
        self.sched = DDPMSchedule(a["diffusion_steps"], device=device)
        self.n, self.device, self.ddim_steps, self.guide_scale = num_envs, device, ddim_steps, guide_scale
        self.step_dim = 3 * self.k + 3 * schema.N_J
        assert self.history * self.step_dim == dims["state"], (self.history, self.step_dim, dims["state"])
        self.hist = torch.zeros((num_envs, self.history, self.step_dim), device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.mode = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def keypoints(self, state: dict) -> torch.Tensor:
        """(N, K, 3) keypoints of this model's version, wrist frame."""
        return state["keypoint_pos_wrist"][:, self.kp_idx]

    def observe(self, state: dict) -> None:
        """Append the current state; envs marked fresh (episode start) get their history filled with it."""
        f = torch.cat([self.keypoints(state).reshape(self.n, -1), state["joint_pos"], state["joint_target"],
                       state["control_error"]], dim=-1).float()
        self.hist = torch.cat([self.hist[:, 1:], f[:, None]], dim=1)
        self.hist[self.fresh] = f[self.fresh, None].expand(-1, self.history, -1)
        self.fresh[:] = False

    def mark_reset(self, done: torch.Tensor) -> None:
        self.fresh |= done.bool()

    @torch.no_grad()
    def act(self, guide_dx: torch.Tensor | None = None):
        """Returns (joint-target change (N,12), sampled motion dx (N,T,K,3) in metres).

        guide_dx: optional commanded motion (N,T,K,3) in the wrist frame, metres.
        """
        s = (self.hist.reshape(self.n, -1) - self.norm["state"][0]) / self.norm["state"][1]
        mm, ms = self.norm["motion"]
        guide = None
        if guide_dx is not None and self.guide_scale > 0:
            guide = (guide_dx.reshape(self.n, -1) - mm) / ms
        x = self.sched.ddim_sample(self.diff, s, self.mode, self.ddim_steps, guide=guide, guide_scale=self.guide_scale,
                                   generator=self.gen)
        mu, _ = self.idm(s, x)
        da = mu * self.norm["action"][1] + self.norm["action"][0]
        return da, (x * ms + mm).view(self.n, self.horizon, self.k, 3)


class KeypointJacobian:
    """Linear keypoint velocity Jacobians of a fixed-base Isaac Lab articulation, expressed in the wrist frame.

    Maps a joint-target change dq (N,12, dexdata JOINT_NAMES order) to keypoint displacements (N,K,3); used to turn
    a joint-space policy command into a keypoint-motion guide.
    """

    def __init__(self, hand, kp_body_names: list[str]):
        assert hand.is_fixed_base
        self.hand = hand
        self.body = torch.as_tensor(hand.find_bodies(kp_body_names, preserve_order=True)[0], device=hand.device) - 1
        self.cols = torch.as_tensor(hand.find_joints(list(schema.JOINT_NAMES), preserve_order=True)[0], device=hand.device)
        self.wrist = hand.find_bodies([schema.WRIST_BODY])[0][0]

    def displacement(self, dq: torch.Tensor) -> torch.Tensor:
        jac = self.hand.root_physx_view.get_jacobians()[:, self.body, :3][..., self.cols]      # (N,K,3,12) world
        dx_w = torch.einsum("nkij,nj->nki", jac, dq)
        q = self.hand.data.body_link_quat_w[:, self.wrist]                                      # wxyz
        # rotate world -> wrist: v' = R^T v
        w, xyz = q[:, None, :1], -q[:, None, 1:]
        t = 2.0 * torch.cross(xyz.expand_as(dx_w), dx_w, dim=-1)
        return dx_w + w * t + torch.cross(xyz.expand_as(dx_w), t, dim=-1)
