# DexGen 제어기 closed-loop sim 평가 (2026-10-12)

`scripts/eval_sim.py`는 학습된 DexGen 제어기를 데이터를 모은 env에서 직접 돌린다. 실행 묶음은 `logs/eval_sim_v1.sh`이고, 결과 json은 checkpoint 옆 `runs/<run>/eval_*.json`에 남는다.

## 방법

- **env:** 수집 조건(`dexdata/logs/collect_v1.sh`)을 그대로 쓴다.
  - `nut_goal`: dexgen NutGoalEnv, 자유 너트, 크기 0.7–1.3, DR 전체 적용
  - `nutbolt`: dexscrew trinut, 크기 0.7–1.25, student preset
- **제어 루프 (20 Hz):**
  1. 4 step history로 미래 keypoint 이동 `dx`를 DDIM 10 step으로 sample한다.
  2. ID 평균으로 joint target 변화량 `da`를 구한다.
  3. env action은 `da / action_scale`이고 [-1, 1]로 자른다. 두 env 모두 `target ← target + action_scale·a`다.
  4. episode가 시작될 때는 history를 첫 상태로 채운다.
- **조건:**

  | 조건 | 내용 |
  |---|---|
  | `policy` | 수집에 쓴 RL 정책 (deterministic) |
  | `noisy` | 성능을 떨어뜨린 정책: `a + 0.5·N(0,1)` |
  | `dexgen` | DexGen만, guidance 없음 |
  | `policy+dexgen` | RL 정책의 명령을 guidance로 준 DexGen |
  | `noisy+dexgen` | 떨어뜨린 정책의 명령을 guidance로 준 DexGen (논문 IV절의 safety layer 설정) |

- **guidance:** 정책 action은 `dq = action_scale·a`다. 이 명령을 horizon 동안 유지한다고 보고, `dx_i = i·J(q)·dq`로 keypoint 이동으로 바꾼다. `J`는 PhysX keypoint Jacobian을 손목 좌표계로 옮긴 것이다. 이 1차 근사의 오차(`jacobian_mm`)는 실제 1 step 이동의 4–20%다.
  - 정규화된 motion 공간에서 논문 식 3의 guidance를 쓴다: `x0 ← x0 − s(1−ᾱ)·2(x0 − guide)`. 기본값은 s = 0.3이다.
- **규모:** envs 512개 × 1200 step(60초)를 조건마다 돌린다. 조건 사이에 모든 env를 reset한다.

## 결과

### nut_goal (dexgen_src checkpoint)

| 조건 | 낙하/분 ↓ | 끝난 episode 평균 길이 (s) | 목표 도달/분 ↑ | 1 step 추종 오차 (mm) | guide 오차 (mm) |
|---|---|---|---|---|---|
| policy | **9.7** | 4.5 | **554** | – | – |
| noisy (σ 0.5) | 15.5 | 3.2 | 324 | – | – |
| dexgen, tips_mid | 60.4 | 1.0 | 5 | 1.08 | – |
| dexgen, tips | 46.1 | 1.2 | 6 | 1.16 | – |
| policy+dexgen, tips_mid | 12.6 | 3.7 | 237 | 1.03 | 2.43 |
| policy+dexgen, tips | 11.4 | 4.1 | 204 | 1.15 | 2.86 |
| noisy+dexgen, tips_mid | 12.7 | 3.7 | 178 | 0.98 | 2.36 |
| noisy+dexgen, tips | 11.8 | 4.0 | 143 | 1.10 | 2.77 |

### nutbolt (dexscrew_src checkpoint)

모든 조건에서 낙하가 0이다. episode는 timeout 아니면 회전 멈춤(stagnation)으로 끝난다.

| 조건 | 너트 회전 (rad/s) ↑ | 끝난 episode 평균 길이 (s) | timeout 비율 | 1 step 추종 오차 (mm) | guide 오차 (mm) |
|---|---|---|---|---|---|
| policy | **0.95** | 11.5 | 15% | – | – |
| noisy (σ 0.5) | 0.66 | 9.3 | 8% | – | – |
| dexgen, tips_mid | 0.28 | 5.6 | 0% | 0.98 | – |
| dexgen, tips | 0.42 | 6.8 | 1.4% | 1.00 | – |
| policy+dexgen, tips_mid | 0.57 | 7.9 | 4.5% | 0.86 | 1.48 |
| policy+dexgen, tips | 0.57 | 8.2 | 4.3% | 0.89 | 1.65 |
| noisy+dexgen, tips_mid | 0.48 | 7.3 | 2.2% | 0.87 | 1.53 |
| noisy+dexgen, tips | 0.53 | 7.8 | 3.3% | 0.90 | 1.77 |

### guidance 세기 (tips_mid, noisy+dexgen)

| s | nut_goal 낙하/분 | nut_goal 목표/분 | nutbolt rad/s |
|---|---|---|---|
| 0.1 | 33.5 | 12 | 0.36 |
| **0.3** | **12.7** | **178** | 0.48 |
| 1.0 | 18.8 | 94 | **0.53** |

## 해석

- **DexGen은 자기가 고른 motion을 잘 따라간다.** 1 step 추종 오차는 0.9–1.2 mm이고, 실제 1 step 이동량(1.2–3.3 mm)보다 작다. ID와 diffusion 사이의 연결은 sim에서도 맞는다.
- **guidance 없는 DexGen만으로는 잡고 있지 못한다.**
  - nut_goal에서는 낙하가 46–60/분으로 정책의 5–6배이고, 거의 모든 episode가 1초 안에 끝난다.
  - nutbolt에서는 낙하가 없지만 회전이 정책의 30–45%다.
  - 논문은 DexGen만 돌려도 안전한 동작이 나온다고 했다. 여기서 그렇지 않은 이유로 보이는 것:
    1. **낙하 직전 데이터.** 수집 정책이 원래 자주 떨어뜨린다(nut_goal episode의 61%가 낙하). 낙하 전 마지막 2초를 버렸지만, 낙하로 가는 상태 분포가 여전히 데이터에 섞여 있다.
    2. **step마다 다른 갈래.** 정회전과 역회전, 일반 동작과 느린 동작이 섞인 데이터라서, step마다 다른 갈래가 sample되면 동작이 흔들린다.
- **safety layer 효과 (noisy → noisy+dexgen):**
  - **nut_goal:** 낙하가 15.5에서 11.8–12.7/분으로 18–24% 줄었지만, 목표 도달도 324에서 143–178로 줄었다.
  - **nutbolt:** 회전이 0.66에서 0.48–0.53 rad/s로 줄었다. 낙하가 원래 없는 task라 이득이 없다.
  - **깨끗한 정책에 붙이면:** 두 task 모두 정책 단독보다 나빠진다.
- **guidance 세기:**
  - s = 0.1이면 guidance가 너무 약해 DexGen 단독에 가깝다.
  - s = 1.0이면 명령을 더 따라간다. guide 오차는 줄지만 추종 오차가 2 mm로 커진다.
  - 이번 설정에서 nut_goal은 s = 0.3이 가장 낫다.
- **tips와 tips_mid:** 큰 차이가 없다. 단독 DexGen과 nutbolt에서는 tips가 조금 낫다.

## 추가 (2026-10-13): reset 직후 warmup과 v2 데이터

- **reset 직후 0.3 s의 큰 가속도.** reset pose가 정지 상태가 아니라서, 손가락이 너트 접촉에 밀려 자리를 잡는 동안 관절 가속도가 정상의 약 3.7배다 (`dexscrew/scripts/measure_finger_jitter.py`의 `by_age`).
- **v2 데이터:** 같은 정책과 명령으로 다시 수집하되 episode 첫 6 step을 버렸다 (`dexdata/data/v2`, `--skip_first_steps 6`). 모델은 v1과 같은 설정으로 학습했다 (`logs/pipeline_v2.sh`).
- **warmup 평가:** DexGen 조건에서 각 episode의 첫 6 step은 RL 정책이 움직인다 (`--policy_warmup 6`, 이제 기본값). 그동안 DexGen history가 채워진다. 이 6 step은 추종 오차에서 뺀다. 위의 결과 표는 warmup 0이다.

### nut_goal (낙하/분, 목표 도달/분)

| 조건 | v1 warmup 0 | v1 warmup 6 | v2 warmup 6 |
|---|---|---|---|
| tips_mid, dexgen 단독 | 60.4 / 5 | **34.8** / 16 | 34.6 / 14 |
| tips_mid, policy+dexgen | 12.6 / 237 | **10.5 / 256** | 10.8 / 203 |
| tips_mid, noisy+dexgen | 12.7 / 178 | **10.8 / 193** | 11.2 / 160 |
| tips, dexgen 단독 | 46.1 / 6 | **30.1** / 15 | 33.6 / 14 |
| tips, policy+dexgen | 11.4 / 204 | **10.0 / 216** | 10.8 / 202 |
| tips, noisy+dexgen | 11.8 / 143 | 11.0 / 153 | 11.1 / 135 |

정책 단독은 9.65 / 554, noisy는 15.5 / 324다.

### nutbolt (너트 회전 rad/s, 낙하 없음)

| 조건 | v1 warmup 6 | v2 warmup 6 |
|---|---|---|
| tips_mid: dexgen / policy+dexgen / noisy+dexgen | 0.32 / 0.58 / 0.50 | 0.37 / 0.60 / 0.51 |
| tips: dexgen / policy+dexgen / noisy+dexgen | 0.42 / 0.57 / 0.53 | 0.39 / 0.49 / 0.45 |

- **warmup 효과가 크다.** DexGen 단독의 낙하가 60에서 35/분으로 줄었다. guidance를 주면 낙하가 정책 단독 수준(10.5 vs 9.65)이 된다. noisy+dexgen(10.8)이 noisy 정책 단독(15.5)보다 낙하가 적어, safety layer 효과가 분명해졌다.
- **v2는 v1과 같거나 조금 못하다.** 낙하는 비슷하고, nut_goal 목표 도달은 v2가 약 20% 적다. nutbolt는 섞여 있다. 학습과 평가 모두 seed 하나라 작은 차이는 noise일 수 있다.
- **정리:** reset 과도 구간은 데이터에서 빼는 것보다 제어할 때 정책이 시작을 맡는 쪽이 낫다. v1 모델 + warmup 6을 기본으로 쓰고, v2 모델과 데이터도 남겨 둔다.
- **checkpoint:**
  - v1: `runs/2026-10-11_12-22-15_{dexgen_src,dexscrew_src}`
  - v2: `runs/2026-10-12_19-14-44_dexgen_src_v2`, `runs/2026-10-12_19-24-45_dexscrew_src_v2`
  - warmup 결과 json은 각 checkpoint 옆의 `eval_*_w6.json`이다. 다시 돌리려면 `logs/eval_sim_warmup6.sh`를 쓴다.

## 다음 후보

1. **mode 조건 학습:** 회전 방향과 style을 mode 라벨로 붙여 학습하고, 평가 때 mode를 고정한다. step마다 갈래가 바뀌는 문제를 줄이는 것이 목적이다.
2. **데이터 필터:** 낙하 episode를 더 강하게 거른다. 예를 들어 낙하 episode 전체를 빼거나 버리는 꼬리를 4초로 늘린다.
3. **guidance:** motion 공간 대신 keypoint 공간 오차로 guidance를 주고 horizon 전체에 가중치를 둔다. 정책 명령을 keypoint로 바꿀 때 Jacobian 대신 FK를 쓴다.
4. **guide 입력:** 정책 action 대신 teleop 같은 keypoint 명령을 넣는 평가를 추가한다.
