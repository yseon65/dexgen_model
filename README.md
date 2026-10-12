# dexgen_model: DexGen 제어기 모델 (keypoint diffusion + inverse dynamics)

DexterityGen (arXiv 2502.04307) III-C절과 부록 F의 DexGen 제어기를 XHand 데이터로 학습한다. 데이터는 `../dexdata/data/v1`(schema v1.0)을 쓴다. 수집 방법은 `../dexdata/docs/collection_guide.md`에 있다.

## 모델

| 구성 | 사양 | 논문 |
|---|---|---|
| 입력 상태 | 4 step history × [keypoint 위치(K,3), joint_pos, joint_target, control_error], 손목 좌표계 | 4 step history, fingertip, joint, target, control error |
| 출력 motion | 미래 keypoint offset `dx_i = x_{t+i} − x_t`, i = 1..4 (20 Hz, 0.2 s) | T=2 (10 Hz, 0.2 s) |
| keypoint | `tips` (K=5)와 `tips_mid` (K=10, distal과 middle link)를 둘 다 학습 | K=8 (PIP+tip, Allegro 4손가락) |
| diffusion | state encoder: 6층 MLP 1024 + GroupNorm(8) | 같음 |
| | UNet: 1D conv 대신 FC, hidden 768, encoder와 decoder 각 3 block, skip, FiLM | 같음 (논문은 FiLM을 중간 층에, 여기서는 모든 block에) |
| | mode embedding: 1개(default) | one-hot mode |
| | DDPM 100 step (cosine), ε 예측, DDIM 10 step, EMA 0.999 | DDIM 8–12 step |
| inverse dynamics | residual MLP (1024, 4 block), Gaussian 출력 | residual MLP, normal 분포 |
| | 학습: 평균은 MSE, std는 평균을 detach한 NLL | MSE |
| | 목표: `applied_target − joint_target` (joint target 변화량) | target joint position |
| 학습 | AdamW (diffusion lr 1e-4, ID 3e-4), warmup 후 cosine, batch 4096, bf16 | AdamW, 15 epoch |

- keypoint 버전별로 diffusion과 ID를 따로 학습한다.
- 한 process는 한 데이터 원천의 4개 network를 같은 batch로 학습한다: {diffusion, ID} × {tips, tips_mid}.
- guided sampling(논문 식 3)은 `DDPMSchedule.ddim_sample(guide=..., guide_scale=...)`에 있다. guide는 정규화된 motion 공간에서 준다.

## 데이터 원천별 학습 (2026-10-11)

| 이름 | dexdata policy | train / val window |
|---|---|---|
| `dexgen_src` | `nut_goal_normal`, `nut_goal_slow15` (자유 너트, 크기 0.7–1.3) | 57.0M / 1.17M |
| `dexscrew_src` | `nutbolt_student_scale` (nut-bolt, 크기 0.7–1.25) | 38.2M / 0.78M |

- **validation:** `episode_id % 50 == 0`인 episode를 따로 뗐다. 검증 지표는 다음과 같다.
  - diffusion val loss (EMA)
  - ID MAE (rad): 실제 motion을 넣었을 때와 DDIM으로 sample한 motion을 넣었을 때
  - 0.2 s 뒤 keypoint endpoint 오차 (mm). 비교 기준은 실제 이동량이다.

```bash
cd ~/IsaacLab/projects/dexgen_model
~/env_isaaclab6/bin/python -u train.py --name dexgen_src --policies nut_goal_normal nut_goal_slow15 \
    --max_steps 210000 --max_hours 6 --val_every 5000 --save_every 20000
~/env_isaaclab6/bin/python -u train.py --name dexscrew_src --policies nutbolt_student_scale \
    --max_steps 210000 --max_hours 6 --val_every 5000 --save_every 20000
```

- **처음 실행할 때:** `cache/<name>/rows.npy`에 학습 행을 float16으로 압축한다. 한 행은 keypoint(tips_mid) 30 + joint 36 + action 12다. dexgen은 9.4 GB, dexscrew는 6.2 GB다.
- **결과:** `runs/<시각>_<name>/`에 TensorBoard 기록, `{tips,tips_mid}_{stepN,last}.pt`, `config.json`이 남는다.
- **checkpoint 내용:** 정규화 통계, diffusion, EMA diffusion, inverse dynamics 가중치가 함께 들어 있다.
- **속도 (RTX 5080, 두 process 동시):** 각각 약 10 it/s, 4.2만 sample/s다.

## 구조

```
dexgen_model/data.py     dexdata → 압축 cache, window 샘플링, state/motion/action 분리
dexgen_model/models.py   KeypointDiffusion, InverseDynamics, DDPMSchedule (DDIM + guidance)
train.py                 한 데이터 원천에 대해 4개 network 학습
```
