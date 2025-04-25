# RL
# TASK: MiniGrid 6x6 empty

<img src= width="400"/>

---

## 환경 설명

- 6x6 격자, padding=1 (벽)
- 실제 움직일 수 있는 칸은 총 15칸 (종료 칸 제외)

---

## Action (총 3개)
1. turn left
2. turn right
3. forward

---

## Termination
1. `max_step` 초과 → Timeout
2. 목표 위치(초록색 칸) 도달 → 성공

---

## Reward
- 성공 시 보상 = `1 - 0.9 * (step_count / max_steps)`
- **step 수가 적을수록 더 많은 보상**

---

## State 구조
- (4x4 구조 격자 수 - 1) x 현재 방향(4) + 도달 가능 방향(2)
- 총 상태 수 = **62개**

---

## Optimal Path 예시
→ → → →  
↘ turn right → ↓ forward  
→ 가장 적은 step으로 도달 가능

---

## site
https://minigrid.farama.org/environments/minigrid/EmptyEnv/
