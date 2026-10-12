"""Packed in-memory training windows built from a dexdata dataset.

Every row of the source dataset is packed into one float16 vector (``PACK_LAYOUT``):

    keypoints (tips_mid, K=10, wrist frame) | joint_pos | joint_target | control_error | action

where action = applied_target - joint_target is computed in float32 before packing (float16 precision).

A training sample at row ``t`` uses rows ``t-H+1 .. t+T`` of the same episode:

- state  : H history steps of [keypoints (K,3), joint_pos, joint_target, control_error]   (paper App. F)
- motion : future keypoint offsets  dx_i = x_{t+i} - x_t,  i = 1..T                       (paper III-C)
- action : applied_target[t] - joint_target[t]   (joint-target change commanded at t; the
           absolute target is joint_target[t] + action)

The tips set (K=5) is a subset of tips_mid, so one packed array serves both keypoint versions.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

DEXDATA_ROOT = os.path.expanduser("~/IsaacLab/projects/dexdata")
sys.path.insert(0, DEXDATA_ROOT)
from dexdata import schema  # noqa: E402
from dexdata.dataset import TrajectoryDataset  # noqa: E402

PACK_KP = schema.KEYPOINT_SETS["tips_mid"]          # indices into KEYPOINT_BODIES, packed order
N_J = len(schema.JOINT_NAMES)
K_PACK = len(PACK_KP)
# column slices of a packed row
SL_KP = slice(0, 3 * K_PACK)
SL_Q = slice(SL_KP.stop, SL_KP.stop + N_J)
SL_QT = slice(SL_Q.stop, SL_Q.stop + N_J)
SL_ERR = slice(SL_QT.stop, SL_QT.stop + N_J)
SL_ACT = slice(SL_ERR.stop, SL_ERR.stop + N_J)
PACK_DIM = SL_ACT.stop
# keypoint version -> positions inside the packed keypoint block
KP_VERSIONS = {name: [PACK_KP.index(i) for i in schema.KEYPOINT_SETS[name]] for name in ("tips", "tips_mid")}


def pack_dataset(root: str, policies: list[str], cache: str, val_every: int = 50) -> None:
    """Pack the selected policies of a dexdata dataset into ``cache`` (rows.npy, starts_*.npy, meta.json)."""
    ds = TrajectoryDataset(root, policies=policies)
    total = sum(int(sh["episodes"]["length"].sum()) for sh in ds.shards)
    os.makedirs(cache, exist_ok=True)
    rows = np.lib.format.open_memmap(os.path.join(cache, "rows.npy"), mode="w+", dtype=np.float16,
                                     shape=(total, PACK_DIM))
    starts_train, starts_val, offset, t0 = [], [], 0, time.time()
    for si, sh in enumerate(ds.shards):
        eps = sh["episodes"]
        kp = ds.field(si, "keypoint_pos_wrist")
        cols = [ds.field(si, f) for f in ("joint_pos", "joint_target", "control_error")]
        tgt = ds.field(si, "applied_target")
        for s, n, eid in zip(eps["start"], eps["length"], eps["episode_id"]):
            s, n = int(s), int(n)
            q_t = np.asarray(cols[1][s:s + n])
            block = np.concatenate([np.asarray(kp[s:s + n, PACK_KP]).reshape(n, -1)]
                                   + [np.asarray(c[s:s + n]) for c in cols]
                                   + [np.asarray(tgt[s:s + n]) - q_t], axis=1)
            rows[offset:offset + n] = block.astype(np.float16)
            # store (episode start, length) in packed coordinates; windows are expanded at load time
            (starts_val if int(eid) % val_every == 0 else starts_train).append((offset, n))
            offset += n
        print(f"[pack] shard {si + 1}/{len(ds.shards)} rows {offset:,}/{total:,} ({time.time() - t0:.0f}s)", flush=True)
    rows.flush()
    np.save(os.path.join(cache, "episodes_train.npy"), np.asarray(starts_train, dtype=np.int64))
    np.save(os.path.join(cache, "episodes_val.npy"), np.asarray(starts_val, dtype=np.int64))
    with open(os.path.join(cache, "meta.json"), "w") as f:
        json.dump({"action": "applied_target - joint_target", "root": os.path.abspath(root), "policies": policies, "rows": total, "pack_dim": PACK_DIM,
                   "pack_keypoints": [schema.KEYPOINT_BODIES[i] for i in PACK_KP], "val_every": val_every}, f, indent=1)


class PackedWindows:
    """Random windows (history H, horizon T) from a packed cache, held in RAM."""

    def __init__(self, cache: str, split: str, history: int, horizon: int, in_memory: bool = True):
        self.h, self.t = history, horizon
        rows = np.load(os.path.join(cache, "rows.npy"), mmap_mode="r")
        self.rows = np.ascontiguousarray(rows) if in_memory else rows
        eps = np.load(os.path.join(cache, f"episodes_{split}.npy"))
        # valid centre rows t: t-H+1 >= start and t+T <= start+n-1
        idx = [np.arange(s + history - 1, s + n - horizon) for s, n in eps if n >= history + horizon]
        self.index = np.concatenate(idx).astype(np.int64)
        self.offsets = np.arange(-history + 1, horizon + 1)

    def __len__(self):
        return len(self.index)

    def sample(self, batch: int, rng: np.random.Generator) -> np.ndarray:
        """(B, H+T, PACK_DIM) float32 windows."""
        t = self.index[rng.integers(0, len(self.index), size=batch)]
        return self.rows[t[:, None] + self.offsets[None]].astype(np.float32)


def split_window(w, history: int, kp_version: str):
    """Window (B, H+T, PACK_DIM) -> state (B, H*(3K+36)), motion (B, T*K*3), action (B, 12). Works for numpy/torch."""
    sel = KP_VERSIONS[kp_version]
    b = w.shape[0]
    kp = w[..., SL_KP].reshape(b, w.shape[1], K_PACK, 3)[:, :, sel]            # (B, H+T, K, 3)
    hist = w[:, :history]
    k = len(sel)
    state = [kp[:, :history].reshape(b, history, 3 * k), hist[..., SL_Q], hist[..., SL_QT], hist[..., SL_ERR]]
    cat = np.concatenate if isinstance(w, np.ndarray) else __import__("torch").cat
    state = cat(state, -1).reshape(b, -1)
    motion = (kp[:, history:] - kp[:, history - 1:history]).reshape(b, -1)      # dx_i = x_{t+i} - x_t
    action = w[:, history - 1, SL_ACT]
    return state, motion, action
