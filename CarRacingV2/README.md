
# 강화학습 과제 2 (Gym CarRacing V2)

## Action

**Discrete (5 actions)**

- Do nothing
- Steer left
- Steer right
- Gas
- Brake

## Observation

- Shape: (96, 96, 3)
- RGB 값, Range: 0 ~ 255

## Termination 조건

- 모든 타일을 방문했을 경우
- 차가 밖으로 나갈 경우 **(-100 reward)**

## Reward

- 매 frame마다 -0.1
- 트랙 타일 하나 방문 시 **+1000 / N**
- 모든 타일 방문 시 총 보상 = `1000 - 0.1 * frame`
- 트랙 밖으로 나가면 **-100 reward**

---

## 실험 내용

### 2.1 기본 DQN 실행 및 결과 분석 (10^5)

**설정한 파라미터**

```
lr: 0.00025
epsilon: 1.0
epsilon_min: 0.1
gamma: 0.99
batch_size: 32
warmup_steps: 5000
buffer_size: 20000
target_update_interval: 10000
epsilon_decay: 0.000009
```

**결과 (Episode 20)**

| Mean | Std | Max | Min |
|------|-----|-----|-----|
| -29.13 | 51.33 | 88.10 | -115.92 |

---

### 3.2 파라미터 변화 실험

#### 3.2.1 더 복잡한 CNN + 파라미터 변경

**변경한 설정**

```
-> batch_size: 64
-> target_update_interval: 5000
```

**CNN 구조**

```
Conv2D (16 filters, 3x3, stride=2, padding=1) + BatchNorm + ReLU -> [N, 16, 42, 42]
AvgPool (2x2, stride=2)

Conv2D (32 filters, 3x3, stride=1, padding=1) + BatchNorm + ReLU -> [N, 32, 21, 21]
AvgPool (2x2, stride=2)

Conv2D (64 filters, 3x3, stride=1, padding=1) + BatchNorm + ReLU -> [N, 64, 10, 10]
AvgPool (2x2, stride=2)
```

**결과 (Episode 20)**

| Mean | Std | Max | Min |
|------|-----|-----|-----|
| 182.93 | 288.38 | 738.33 | -95.00 |

> 5000 스텝 주기로 target update → loss 급증 가능성 있음

---

#### 3.2.2 Reward 방식 변경 실험

- 도로, 잔디를 구분하여 reward 부여
- 자동차 좌표 고정 → 주변 지형 기반 보상 계산
- 평가 시 reward 변경 X (학습 시만 변경)

**결과 (Episode 20)**

| Mean | Std | Max | Min |
|------|-----|-----|-----|
| 494.50 | 287.80 | 860.56 | -95.00 |

---

### 3.2.3 DDQN + Reward 수정 실험

- Double DQN 방식으로 target error 계산

**결과 (Episode 20)**

| Mean | Std | Max | Min |
|------|-----|-----|-----|
| -22.00 | 234.77 | 901.40 | -95.00 |

> 성능 급락: replay buffer 내 과거 transition 영향 → 새로운 경로 탐색 데이터 유입 → loss 급증

---

## 비교

| Algorithm | Mean | Std | Max | Min |
|-----------|------|-----|-----|-----|
| advDDQN | -22.00 | 234.77 | 901.40 | -95.00 |
| advDQN (Reward Change) | 494.50 | 287.80 | 860.56 | -95.00 |
| advDQN | 182.93 | 288.38 | 738.33 | -95.00 |
| DQN (Normal) | -29.13 | 51.33 | 88.10 | -115.92 |

- reward 수정 → 학습 안정성 증가 (loss 감소)
- 일반 DQN → Max return 높음
- DDQN → return은 높지만 불안정성 존재

---

## 결론

- 충분한 학습 스텝 부족 -> 수렴 부족
- 평가 효율성 문제 (멀티스레드 부재)
- 초기 transition에 의한 DDQN 성능 저하
 -> Prioritized Experience Replay기법 도입 필요


