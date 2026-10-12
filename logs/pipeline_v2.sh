#!/bin/bash
# v2 (2026-10-12): recollect dexdata with --skip_first_steps 6 (same policies/commands as v1),
# then train both DexGen sources with the v1 settings and run the closed-loop sim eval.
set -e
PY=~/env_isaaclab6/bin/python
DG=~/IsaacLab/projects/dexgen/outputs/xhand_nut_goal
DS=~/IsaacLab/projects/dexscrew/outputs
OUT=~/IsaacLab/projects/dexdata/data/v2
M=~/IsaacLab/projects/dexgen_model
L=$M/logs

cd ~/IsaacLab/projects/dexdata
echo "[pipeline] collect $(date)"
$PY scripts/collect_dexgen.py --out $OUT --run_name nut_goal_normal_s0 --seed 0 --num_envs 1024 \
    --target_transitions 4e7 --stochastic --policy_name nut_goal_normal \
    --checkpoint $DG/2026-10-10_23-26-04_goal_scale0713_ft_initq/model_14250.pt \
    --cfg "scale_range=(0.7,1.3)" dr_ramp_steps=0 > $L/collect_v2_nut_goal_normal.log 2>&1
$PY scripts/collect_dexgen.py --out $OUT --run_name nut_goal_slow_s0 --seed 1 --num_envs 1024 \
    --target_transitions 2e7 --stochastic --policy_name nut_goal_slow15 \
    --checkpoint $DG/2026-10-11_03-53-58_goal_scale0713_ft_slowstyle15/model_13998.pt \
    --cfg "scale_range=(0.7,1.3)" dr_ramp_steps=0 > $L/collect_v2_nut_goal_slow.log 2>&1
$PY scripts/collect_dexscrew.py --out $OUT --run_name nutbolt_student_s0 --seed 2 --num_envs 1024 \
    --target_transitions 4e7 --policy_name nutbolt_student_scale \
    --ckpt $DS/XHandHoraNutBolt_student_padapt/lab_scale07125_t1413/stage2_nn/model_best.ckpt \
    "task.env.randomization.randomizeScaleList=[0.7,0.8,0.9,1.0,1.1,1.2,1.25]" +task.env.keepNutHeight=True \
    > $L/collect_v2_nutbolt_student.log 2>&1
$PY scripts/validate_dataset.py $OUT > $L/validate_v2.log 2>&1 || echo "[pipeline] validate reported problems (see validate_v2.log)"

cd $M
echo "[pipeline] train $(date)"
$PY -u train.py --name dexgen_src_v2 --data_root $OUT --policies nut_goal_normal nut_goal_slow15 \
    --max_steps 210000 --max_hours 6 --val_every 5000 --save_every 20000 > $L/train_dexgen_src_v2.log 2>&1 &
sleep 600   # stagger the cache packing of the two sources
$PY -u train.py --name dexscrew_src_v2 --data_root $OUT --policies nutbolt_student_scale \
    --max_steps 210000 --max_hours 6 --val_every 5000 --save_every 20000 > $L/train_dexscrew_src_v2.log 2>&1 &
wait

echo "[pipeline] eval $(date)"
G=$(ls -d runs/*_dexgen_src_v2 | tail -1)
S=$(ls -d runs/*_dexscrew_src_v2 | tail -1)
for v in tips_mid tips; do
  $PY -u scripts/eval_sim.py --task nut_goal --ckpt $G/${v}_last.pt --num_envs 512 --steps 1200 > $L/eval_v2_nut_goal_${v}.log 2>&1
  $PY -u scripts/eval_sim.py --task nutbolt --ckpt $S/${v}_last.pt --num_envs 512 --steps 1200 > $L/eval_v2_nutbolt_${v}.log 2>&1
done
echo "[pipeline] done $(date)"
