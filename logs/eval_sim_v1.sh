#!/bin/bash
# Closed-loop sim evaluation of the 2026-10-11 DexGen checkpoints (docs/sim_eval.md)
cd ~/IsaacLab/projects/dexgen_model
PY=~/env_isaaclab6/bin/python
G=runs/2026-10-11_12-22-15_dexgen_src
S=runs/2026-10-11_12-22-15_dexscrew_src
for v in tips_mid tips; do
  $PY -u scripts/eval_sim.py --task nut_goal --ckpt $G/${v}_last.pt --num_envs 512 --steps 1200 > logs/eval_nut_goal_${v}.log 2>&1
  $PY -u scripts/eval_sim.py --task nutbolt --ckpt $S/${v}_last.pt --num_envs 512 --steps 1200 > logs/eval_nutbolt_${v}.log 2>&1
done
for g in 0.1 1.0; do
  $PY -u scripts/eval_sim.py --task nut_goal --ckpt $G/tips_mid_last.pt --num_envs 512 --steps 1200 --guide_scale $g \
      --conditions policy+dexgen noisy+dexgen > logs/eval_nut_goal_tips_mid_g$g.log 2>&1
  $PY -u scripts/eval_sim.py --task nutbolt --ckpt $S/tips_mid_last.pt --num_envs 512 --steps 1200 --guide_scale $g \
      --conditions policy+dexgen noisy+dexgen > logs/eval_nutbolt_tips_mid_g$g.log 2>&1
done
echo done
