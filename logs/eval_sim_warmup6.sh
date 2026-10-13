#!/bin/bash
# DexGen conditions with the RL policy acting for the first 6 steps of every episode (v1 vs v2)
cd ~/IsaacLab/projects/dexgen_model
PY=~/env_isaaclab6/bin/python
C="dexgen policy+dexgen noisy+dexgen"
for tag in v1:2026-10-11_12-22-15 v2:2026-10-12_19-14-44,2026-10-12_19-24-45; do
  d=${tag%%:*}; r=${tag#*:}
  if [ $d = v1 ]; then G=runs/${r}_dexgen_src; S=runs/${r}_dexscrew_src
  else G=runs/${r%,*}_dexgen_src_v2; S=runs/${r#*,}_dexscrew_src_v2; fi
  for v in tips_mid tips; do
    $PY -u scripts/eval_sim.py --task nut_goal --ckpt $G/${v}_last.pt --num_envs 512 --steps 1200 --policy_warmup 6 \
        --conditions $C > logs/eval_w6_${d}_nut_goal_${v}.log 2>&1
    $PY -u scripts/eval_sim.py --task nutbolt --ckpt $S/${v}_last.pt --num_envs 512 --steps 1200 --policy_warmup 6 \
        --conditions $C > logs/eval_w6_${d}_nutbolt_${v}.log 2>&1
  done
done
echo done
