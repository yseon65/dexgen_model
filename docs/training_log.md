# DexGen 모델 학습 점검 기록

## 2026-10-11 12:22 시작

- `dexgen_src`, `dexscrew_src`를 동시에 학습한다. 둘 다 `--max_steps 210000 --max_hours 6`.
- smoke test(600 step): 두 process가 각각 10.2 it/s, GPU 7 GB를 썼다. diffusion loss는 1.0에서 0.27로, ID MAE는 0.026에서 0.007 rad로 줄었다.
- DDIM에 x0 clip(±6, 정규화 단위)을 추가했다. 학습하지 않은 모델에서는 마지막 step에서 x0가 발산했기 때문이다.
- smoke run 폴더 `runs/2026-10-11_12-18-*`는 남아 있다. 지워도 된다.

## 12:54 — step 20000 (dexgen 1.44 epoch, dexscrew 2.13 epoch)

| 원천 | kp | diff val | ID MAE gt dx (rad) | ID MAE sampled dx (rad) | endpoint err / 실제 이동 (mm) |
|---|---|---|---|---|---|
| dexgen | tips | 0.141 | 0.0042 | 0.0180 | 7.22 / 6.21 |
| dexgen | tips_mid | 0.096 | 0.0041 | 0.0195 | 6.90 / 5.23 |
| dexscrew | tips | 0.111 | 0.0026 | 0.0032 | 2.80 / 5.68 |
| dexscrew | tips_mid | 0.091 | 0.0023 | 0.0033 | 2.99 / 4.75 |

- 둘 다 10.3–10.4 it/s이고 오류는 없다.
- **dexgen의 endpoint 오차가 실제 이동보다 크다.**
  - 자유 너트 데이터는 양방향 목표다. 그래서 같은 상태에서도 미래 움직임이 여러 갈래(multimodal)로 나뉜다.
  - 이 경우 sample 1개를 실제 궤적 1개와 비교하면 방향이 반대인 sample에서 오차가 실제 이동의 최대 2배까지 나온다.
  - 따라서 이 지표만으로 학습 실패라고 판단할 수 없다.
  - dexscrew는 회전 방향이 하나라서 오차가 2.8–3.0 mm다. 실제 이동의 약 50%다.
  - diffusion val loss는 두 원천 모두 순조롭게 내려가고 있다.
- 개입하지 않는다. 학습이 끝난 뒤 best-of-N 오차로 multimodal 여부를 확인한다.

## 13:24 — step 35000 점검과 multimodal 진단

- 지표가 계속 내려가고 있고 오류는 없다.
  - dexgen: diff 0.135 (tips) / 0.090 (tips_mid), endpoint 7.0 / 6.5 mm
  - dexscrew: diff 0.107 / 0.086, endpoint 2.4 / 2.4 mm, ID MAE 0.0018–0.0020 rad
- dexgen의 endpoint 오차가 2회 연속 실제 이동보다 크다. 그래서 `scripts/eval_samples.py`로 진단했다.
  - 조건: step20000 checkpoint, val 상태 2048개, 상태마다 DDIM sample 16개

| 원천 | kp | 실제 이동 (mm) | sample 1개 (mm) | best-of-16 (mm) | ID MAE sample 1개 / best (rad) |
|---|---|---|---|---|---|
| dexgen | tips | 6.29 | 6.94 | **2.73** | 0.0176 / 0.0086 |
| dexgen | tips_mid | 5.31 | 6.86 | **2.63** | 0.0197 / 0.0096 |
| dexscrew | tips | 5.62 | 2.83 | 1.12 | 0.0033 / 0.0022 |
| dexscrew | tips_mid | 4.70 | 3.05 | 1.12 | 0.0035 / 0.0021 |

- **결론: 학습 실패가 아니라 데이터의 multimodality 때문이다.**
  - 16개 sample 중 가장 가까운 것은 실제 궤적에서 2.6–2.7 mm 안에 있다.
  - sample 1개의 오차는 실제 이동보다 크다. 같은 상태에서 정회전과 역회전(양방향 목표), 일반과 slow 스타일이 모두 가능하기 때문이다.
  - DexGen에서는 이 분포에서 sample한 뒤 외부 명령 guidance로 mode를 고른다. 따라서 예상한 동작이다.
- 개입하지 않는다.

## 13:54 — step 57000 (dexgen 4.1 epoch, dexscrew 6.1 epoch)

- **dexgen:** diff 0.131 / 0.086, ID MAE(gt dx) 0.0038 / 0.0035 rad
- **dexscrew:** diff 0.105 / 0.082, endpoint 2.28 / 2.20 mm, ID MAE 0.0017 / 0.0019 rad
- 모두 천천히 내려가고 있고 오류는 없다. 개입하지 않는다.

## 14:24 — step 75000, dexscrew tips ID spike 진단

- **현상:** dexscrew `tips`의 ID val MAE가 65k까지 0.0016–0.0017 rad였다가, 70k와 75k에서 0.0059 / 0.0056 rad가 됐다. 2회 연속이다.
- **진단**
  - 학습 MSE(정규화)는 보통 ~0.01이다. 그런데 가끔 0.03–0.13으로 튄다. 20k step 이후 100 step 단위 기록 562개 중 13개였다.
  - 튄 뒤 다음 기록에서는 정상으로 돌아온다. 70k와 75k 기록이 마침 이 spike와 겹쳤다.
  - 다른 3개 network(dexscrew tips_mid, dexgen 두 개)에는 이런 현상이 없다.
  - dexscrew action에는 이상값이 없다. 정규화 단위로 |z| 최대 1.5다.
  - 추정 원인: dexscrew action은 범위가 정해져 있어 ID가 매우 정확해지고, 그만큼 `log_std`가 작아진다. 그러면 NLL 항이 작은 오차에도 큰 gradient를 내고, 공유 trunk를 순간적으로 흔든다.
- **조치**
  - 학습은 계속한다. 손상이 일시적이고 diffusion에는 영향이 없다.
  - 학습이 끝나면 마지막 checkpoint 몇 개(step180000, 200000, last)의 val ID를 비교해서 가장 좋은 것을 쓴다.
  - 다음 학습부터 바꿀 것: `log_std` 하한을 −7에서 −5로 올리거나, ID에도 EMA를 쓴다.
- **나머지:** dexgen diff 0.129 / 0.084, dexscrew tips_mid ID 0.0015 rad로 순조롭다.

## 14:54 — step 94000 (dexgen 6.8 epoch, dexscrew 10.1 epoch)

- **dexgen:** diff 0.128 / 0.083, ID MAE(gt dx) 0.0036 / 0.0034 rad
- **dexscrew:** diff 0.102 / 0.079, ID MAE 0.0016 / 0.0015 rad, endpoint 2.14 / 2.05 mm
- dexscrew tips ID는 0.0016 rad로 돌아왔다. 일시적 spike였다는 판단과 맞다.
- 개입하지 않는다.

## 15:24 — step 113000 (dexgen 8.2 epoch, dexscrew 12.1 epoch)

- **dexgen:** diff 0.127 / 0.082, ID MAE 0.0035 / 0.0033 rad
- **dexscrew:** diff 0.101 / 0.078, ID MAE 0.0017 / 0.0015 rad, endpoint 2.08 / 1.96 mm
- 천천히 개선되고 있다. 개입하지 않는다.

## 15:54 — step 132000 (dexgen 9.5 epoch, dexscrew 14.1 epoch)

- **dexgen:** diff 0.126 / 0.082, ID MAE 0.0034 / 0.0032 rad
- **dexscrew:** diff 0.100 / 0.077, ID MAE 0.0015 / 0.0015 rad, endpoint 2.02 / 1.90 mm
- 개선 폭이 줄고 있지만 계속 내려간다. 개입하지 않는다.

## 16:24 — step 150000 (dexgen 10.8 epoch, dexscrew 16.1 epoch)

- **dexgen:** diff 0.125 / 0.081, ID MAE 0.0034 / 0.0032 rad
- **dexscrew:** diff 0.100 / 0.077, ID MAE 0.0018 / 0.0014 rad, endpoint 1.98 / 1.87 mm
  - tips ID가 0.0018 rad인 것은 일시적 spike의 영향일 수 있다. 끝나면 checkpoint별로 비교한다.
- 개입하지 않는다.

## 16:54 — step 169000 (dexgen 12.2 epoch, dexscrew 18.1 epoch)

- **dexgen:** diff 0.125 / 0.081, ID MAE 0.0033 / 0.0031 rad
- **dexscrew:** diff 0.0998 / 0.0765, ID MAE 0.0014 / 0.0014 rad, endpoint 1.97 / 1.85 mm
- dexscrew tips ID가 0.0014 rad로 회복했다. 개입하지 않는다.

## 17:24 — step 188000 (dexgen 13.5 epoch, dexscrew 20.1 epoch)

- **dexgen:** diff 0.1248 / 0.0808, ID MAE 0.0033 / 0.0031 rad
- **dexscrew:** diff 0.0995 / 0.0763, ID MAE 0.0014 / 0.0013 rad, endpoint 1.95 / 1.81 mm
- 210000 step에 ~18:00경 도달할 예상이다. 시간 예산보다 먼저 끝난다. 개입하지 않는다.

## 18:00 — 학습 종료, 최종 결과

두 process 모두 210000 step에서 끝났다. 소요 시간은 5시간 38분이다. dexgen은 15.1 epoch, dexscrew는 22.5 epoch를 돌았다.

### 최종 validation (step 209999, EMA diffusion)

| 원천 | kp | diff val loss | ID MAE, 실제 dx (rad) | ID MAE, sample dx (rad) | endpoint err sample 1개 / best-of-16 (mm) | 실제 이동 (mm) |
|---|---|---|---|---|---|---|
| dexgen | tips (K=5) | 0.1246 | 0.0032 | 0.0166 | 6.43 / **2.24** | 6.29 |
| dexgen | tips_mid (K=10) | 0.0807 | 0.0031 | 0.0173 | 5.62 / **2.14** | 5.31 |
| dexscrew | tips (K=5) | 0.0994 | 0.0014 | 0.0021 | 1.98 / **0.80** | 5.62 |
| dexscrew | tips_mid (K=10) | 0.0762 | 0.0013 | 0.0022 | 1.83 / **0.77** | 4.70 |

(best-of-16 열은 `scripts/eval_samples.py`로 쟀다. 조건은 val 상태 2048개, DDIM 10 step.)

- **dexscrew:** sample 1개로도 0.2 s endpoint 오차가 실제 이동의 33–39%다. ID 오차는 0.0013–0.0014 rad다.
- **dexgen:** sample 1개 오차는 실제 이동과 비슷하다. best-of-16은 2.1–2.2 mm다.
  - 양방향 목표와 두 스타일 때문에 분포가 multimodal이다. 실제로 쓸 때는 외부 명령 guidance로 mode를 고른다.
- **tips와 tips_mid 비교**
  - ID 오차(실제 dx)는 비슷하다. tips_mid가 근소하게 더 좋다.
  - endpoint 오차는 tips_mid가 조금 작다. 다만 keypoint 평균 기준이라 직접 비교하기는 어렵다.
  - 논문은 K=4(tips)일 때 ID loss가 컸다고 보고했다. 여기서는 XHand 12 DOF가 tips 5개(15차원)로 거의 결정되기 때문에 차이가 작은 것으로 보인다.
- **dexscrew tips ID:** 중간의 일시적 spike(70k/75k)에서 회복했다. 최종값 0.0014 rad라서 `last`를 그대로 쓴다.

### checkpoint

- `runs/2026-10-11_12-22-15_dexgen_src/{tips,tips_mid}_last.pt`
- `runs/2026-10-11_12-22-15_dexscrew_src/{tips,tips_mid}_last.pt`
- 중간 checkpoint는 `*_step{20000..200000}.pt`다.
- 각 파일에는 정규화 통계, diffusion, diffusion_ema, inverse_dynamics 가중치가 들어 있다.

### 다음 단계 후보

1. 시뮬레이션 closed-loop 평가(논문 IV절): 열화된 정책에 DexGen guidance를 붙여 유지 시간, 회전량, 낙하율을 잰다. 이를 위해 env에 DexGen 제어기를 연결하는 코드가 필요하다.
2. ID의 NLL 안정화: `log_std` 하한을 −5로 올리거나 ID에도 EMA를 쓴다.
3. 두 원천을 합친 혼합 데이터로 학습하고 mode label을 붙인다.
