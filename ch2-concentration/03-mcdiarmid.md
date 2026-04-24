# 03. McDiarmid 부등식 (Bounded Differences)

## 🎯 핵심 질문

- **Bounded differences 조건**: 함수 $f: \mathcal{X}^n \to \mathbb{R}$이 "한 좌표를 바꿔도 $c_i$ 이상 변하지 않으면" 무엇이 성립하는가?
- **McDiarmid 부등식**: $\mathbb{P}(|f(X) - \mathbb{E}[f(X)]| \geq t) \leq 2\exp(-2t^2/\sum c_i^2)$는 왜 **Markov를 Hoeffding 형태로 승격**하는가?
- **Doob martingale**: 이 증명의 핵심 도구로, "조건부 기대값의 수열"이 어떻게 bounded difference를 만드는가?
- **Azuma-Hoeffding lemma**: martingale 차분의 집중을 bound하는 이 기법은 왜 **Ch5-02의 Rademacher 기반 bound**와 연결되는가?
- **왜 이것이 "Rademacher 복잡도 집중의 핵심 도구"**인가?

---

## 🔍 왜 이 이론이 현대 ML에서 중요한가

McDiarmid는 Hoeffding을 **함수 버전**으로 일반화한다. Hoeffding은 "표본 평균"에 대해서만 작동하지만, McDiarmid는 **어떤 함수든** — 최댓값, 경험적 Rademacher, bootstrap statistic — 을 다룬다. 

실제로, Rademacher 복잡도 $\hat{\mathcal{R}}_S = \sup_f \frac{1}{n}\sum \sigma_i f(x_i)$는 한 샘플 $x_i$에 대한 bounded difference를 만족한다. 따라서 McDiarmid를 적용하면 Rademacher 자체의 집중을 bound할 수 있고, 이것이 **Ch5-02에서 "Rademacher 복잡도로 일반화 경계"를 유도하는 열쇠**가 된다.

또한, bootstrap의 수렴성 증명, cross-validation error의 집중, online learning의 worst-case regret bound에 모두 McDiarmid가 쓰인다. **함수의 일반적 안정성을 보장하는 유일한 거리-자유(distance-free) 방법**이다.

---

## 📐 수학적 선행 조건

- Ch2-01, Ch2-02: Markov, Hoeffding (개념)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Martingale, 조건부 기대값, 독립성
- 기초: 함수합성, 기대값의 선형성

---

## 📖 직관적 이해

### "한 데이터를 바꿔도 함수가 크게 안 변한다"

데이터 $X = (x_1, \ldots, x_n)$에 대한 통계량 $f(X)$ (예: 최댓값, 중앙값, 표본 분산)을 생각하자. 한 데이터 포인트 $x_i$를 다른 것으로 바꾸면 $f(X)$는 변한다. 하지만 그 변화가 **최대 $c_i$**라면?

- 최댓값: $f(X) = \max_j X_j$. $x_i$를 바꿔도 다른 최댓값이 있으면 불변. 하나만 차이 나도 최대 1 (범위가 bounded이면). $c_i = \sup_X |\max(X) - \max(X_{\setminus i})|$.
- 중앙값: 한 포인트 변화는 중앙값을 최대 $\Delta$ 정도 바꾼다. $c_i = \Delta$.

이 조건이 있으면, **표본 평균 형태의 Hoeffding처럼** 지수적 집중을 얻을 수 있다는 것이 McDiarmid의 주장.

### Doob martingale: "정보 공개의 과정"

$n$개 샘플을 하나씩 공개하면서:
- 처음: 정보 없음, 기대값 $\mathbb{E}[f(X)]$
- $i$번째: $X_1, \ldots, X_i$ 알고 있음, 조건부 기대값 $\mathbb{E}[f(X) | X_1, \ldots, X_i]$
- 끝: 모든 샘플 알고 있음, 그냥 $f(X)$

이 수열 $V_i = \mathbb{E}[f(X) | X_1, \ldots, X_i]$는 **martingale**이다. 차분 $D_i = V_i - V_{i-1}$은 한 좌표 $X_i$에 대한 변화만 포착한다. **Bounded difference 조건이 이 $D_i$들을 bounded로 만든다**.

---

## ✏️ 엄밀한 정의

### 정의 2.5 (Bounded Differences 조건)

함수 $f: \mathcal{X}^n \to \mathbb{R}$이 **bounded differences with constants $c_1, \ldots, c_n$**을 만족 ⟺ 모든 $i \in [n]$과 모든 $x, x' \in \mathcal{X}^n$에 대해 (단, $x, x'$는 $i$-번째 좌표에서만 다름):
$$|f(x) - f(x')| \leq c_i.$$

이것을 **$c$-Lipschitz with respect to the $\ell_\infty$ norm**이라고도 부른다.

### 정의 2.6 (Doob Martingale)

표본 $X = (X_1, \ldots, X_n) \sim \mathcal{D}^n$ iid에 대해 **Doob martingale**:
$$V_0 := \mathbb{E}[f(X)], \quad V_i := \mathbb{E}[f(X) | X_1, \ldots, X_i] \text{ for } i = 1, \ldots, n.$$

마지막 항: $V_n = f(X)$.

---

## 🔬 정리와 증명

### 정리 2.7 (McDiarmid 부등식 — Bounded Differences)

$f: \mathcal{X}^n \to \mathbb{R}$이 bounded differences $c_1, \ldots, c_n$을 만족하고, $X_1, \ldots, X_n$ iid일 때
$$\mathbb{P}(|f(X) - \mathbb{E}[f(X)]| \geq t) \leq 2\exp\left(-\frac{2t^2}{\sum_{i=1}^n c_i^2}\right).$$

**증명 (Azuma-Hoeffding via Doob)**. 

1. **Martingale 차분**: Doob martingale $V_i$의 차분을 $D_i = V_i - V_{i-1}$이라 하자:
$$f(X) - \mathbb{E}[f(X)] = \sum_{i=1}^n D_i = \sum_{i=1}^n [V_i - V_{i-1}].$$

2. **Bounded differences → Bounded martingale differences**: Bounded differences 조건에 의해, $D_i = \mathbb{E}[f(X) | X_1, \ldots, X_i] - \mathbb{E}[f(X) | X_1, \ldots, X_{i-1}]$는 $X_i$가 $[0, 1]$ 범위에서 변할 때 최대 $c_i$만큼 변한다. 따라서
$$|D_i| \leq c_i \quad \text{a.s.}$$

(이 부분이 bounded difference 조건이 martingale 차분을 bounded로 만드는 핵심.)

3. **Azuma-Hoeffding Lemma**: $S_n = \sum D_i$가 martingale이고 $|D_i| \leq c_i$이면, 모든 $t > 0$에 대해
$$\mathbb{P}(S_n \geq t) \leq \exp\left(-\frac{2t^2}{\sum c_i^2}\right).$$

**Azuma 증명**: Chernoff 방법을 각 $D_i$에 적용. $\lambda > 0$:
$$\mathbb{E}[e^{\lambda D_i} | X_1, \ldots, X_{i-1}] \leq e^{\lambda^2 c_i^2 / 8}$$
(이것이 bounded $D_i$에 대한 Hoeffding-style MGF bound).

따라서 $\mathbb{E}[\prod e^{\lambda D_i}] \leq \exp(\lambda^2 \sum c_i^2 / 8)$. Markov + Chernoff 최적화로 $\mathbb{P}(S_n \geq t) \leq \exp(-2t^2/\sum c_i^2)$.

4. **대칭성**: $-S_n$도 같은 bound를 만족하므로
$$\mathbb{P}(|f(X) - \mathbb{E}[f(X)]| \geq t) = \mathbb{P}(|S_n| \geq t) \leq 2\exp\left(-\frac{2t^2}{\sum c_i^2}\right). \qquad \square$$

### 정리 2.8 (특수 경우: 같은 상수)

모든 $c_i = c$이면
$$\mathbb{P}(|f(X) - \mathbb{E}[f(X)]| \geq t) \leq 2\exp\left(-\frac{2t^2}{nc^2}\right).$$

---

## 💻 NumPy 구현 검증

### 실험 1: 최댓값의 집중 (bounded differences 예제)

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# f(X) = max(X), X_i ~ Uniform[0, 1]
# Bounded difference: c_i = 1 for all i
# E[max(X)] = n/(n+1) for n uniforms

n = 50
t_vals = np.linspace(0.01, 0.2, 30)
n_trials = 5000

empirical = []
mcdiarmid = []

for t in t_vals:
    max_vals = []
    for _ in range(n_trials):
        X = rng.uniform(0, 1, n)
        max_vals.append(X.max())
    
    emp_mean = np.mean(max_vals)
    emp_prob = np.mean(np.abs(np.array(max_vals) - emp_mean) >= t)
    empirical.append(emp_prob)
    
    # McDiarmid: sum c_i^2 = n
    bound = 2 * np.exp(-2 * t**2 / n)
    mcdiarmid.append(bound)

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogy(t_vals, empirical, 'ko-', linewidth=2, markersize=6, label='Empirical')
ax.semilogy(t_vals, mcdiarmid, 'bs--', linewidth=2, label='McDiarmid')
ax.set_xlabel('t')
ax.set_ylabel('P(|max(X) - E[max(X)]| ≥ t)')
ax.set_title(f'Max of {n} Uniforms: McDiarmid bound')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# → Empirical이 bound 아래에 있음 (때로 loose)
```

### 실험 2: 경험적 Rademacher 복잡도의 집중

```python
# f(X) = Rademacher({x_i}), 한 샘플 변화에 대한 bounded diff
# 이것이 Ch5-02의 핵심: Rademacher 자체의 집중

def empirical_rademacher(X, n_trials=1000):
    """경험적 Rademacher complexity를 Monte Carlo로 계산"""
    n = len(X)
    rads = []
    for _ in range(n_trials):
        sigma = rng.choice([-1, 1], n)
        rad = np.mean(np.abs(X * sigma))  # 단순 예제: |x_i sigma_i|
        rads.append(rad)
    return np.mean(rads), np.std(rads)

# 데이터: X ~ N(0, 1), n 포인트
X_true = rng.standard_normal(100)
mean_X = np.abs(X_true).mean()

# McDiarmid 적용: 한 포인트 $x_i$ 바꾸면 Rademacher는 최대 1/n 변한다
# (각 샘플이 전체 합에 1/n 기여)
c = 1 / len(X_true)

# n개 샘플로 여러 번 Rademacher 추정
n_exps = 100
rads_empirical = []
for _ in range(n_exps):
    X_sample = rng.standard_normal(len(X_true))
    rad, _ = empirical_rademacher(X_sample, n_trials=500)
    rads_empirical.append(rad)

rads_empirical = np.array(rads_empirical)
t = 0.05

# McDiarmid bound: sum c_i^2 = n * (1/n)^2 = 1/n
bound_mcdiarmid = 2 * np.exp(-2 * t**2 / (1 / len(X_true)))
emp_prob = np.mean(np.abs(rads_empirical - rads_empirical.mean()) >= t)

print(f'McDiarmid bound for Rademacher concentration: {bound_mcdiarmid:.4e}')
print(f'Empirical P(|Rad - E[Rad]| ≥ {t}): {emp_prob:.4f}')

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(rads_empirical, bins=30, alpha=0.7, density=True, edgecolor='k')
ax.axvline(rads_empirical.mean(), color='r', linestyle='--', linewidth=2, label=f'E[Rad]')
ax.axvline(rads_empirical.mean() + t, color='orange', linestyle='--', linewidth=2, label=f'E[Rad] ± {t}')
ax.axvline(rads_empirical.mean() - t, color='orange', linestyle='--', linewidth=2)
ax.set_xlabel('Empirical Rademacher complexity')
ax.set_title('Rademacher 복잡도의 집중 (McDiarmid)')
ax.legend()
plt.tight_layout()
plt.show()

# → Rademacher 자체가 몇몇 표본에 대해 집중됨
```

### 실험 3: Bounded differences 계수의 효과

```python
# 함수 1: sum of |x_i| — c_i = 1 each
# 함수 2: max{|x_i|} — c_i도 최대 1, but smaller sum
# 함수 3: median(|x_i|) — slower change

def sum_abs(X): return np.sum(np.abs(X))
def max_abs(X): return np.max(np.abs(X))
def median_abs(X): return np.median(np.abs(X))

X_sample = rng.standard_normal(30)

# Bounded diff 계수
c_sum = 30 * 1  # sum c_i^2 = 30
c_max = 30 * 1  # worst case c_i^2 = 1 each
c_med = 30 * 0.1  # median changes slower

t = 0.1
bound_sum = 2 * np.exp(-2 * t**2 / c_sum)
bound_max = 2 * np.exp(-2 * t**2 / c_max)
bound_med = 2 * np.exp(-2 * t**2 / c_med)

print(f'Bound for sum:    {bound_sum:.4e} (worst)')
print(f'Bound for max:    {bound_max:.4e}')
print(f'Bound for median: {bound_med:.4e} (best, slower changes)')

# 실제로 몇 번 샘플
n_trials = 1000
for trial in range(3):
    X = rng.standard_normal((n_trials, 30))
    sums = np.array([sum_abs(X[i]) for i in range(n_trials)])
    maxs = np.array([max_abs(X[i]) for i in range(n_trials)])
    meds = np.array([median_abs(X[i]) for i in range(n_trials)])
    
    if trial == 0:
        print(f'\nEmpirical P(|f(X) - E[f(X)]| ≥ {t}):')
    print(f'  sum:    {np.mean(np.abs(sums - sums.mean()) >= t):.4f}')
    print(f'  max:    {np.mean(np.abs(maxs - maxs.mean()) >= t):.4f}')
    print(f'  median: {np.mean(np.abs(meds - meds.mean()) >= t):.4f}')
```

---

## 🔗 ML 알고리즘 연결

| 적용 | McDiarmid 형태 | 좌표 변화 $c_i$ |
|-----|---|---|
| **Cross-validation error** | 한 폴드 데이터 변화 | $1/K$ (폴드 수) |
| **Bootstrap statistic** | 한 부트스트랩 샘플 재샘플링 | $O(1/n)$ |
| **경험적 Rademacher** | 한 점 $x_i$ 변화 | $1/n$ (정규화 후) |
| **트리 가지치기 결과** | 한 리프 노드 값 변화 | 트리 깊이 의존 |
| **Voting classifier** | 한 기저 분류기의 변경 | $1/m$ (분류기 수) |

**Ch5-02에서 Rademacher 일반화 경계를 유도할 때 McDiarmid가 핵심 도구**이다.

---

## ⚖️ 가정과 한계

1. **Bounded differences 가정**: 실제 함수가 이 조건을 만족하는가? 예: 최댓값은 만족, 정확한 선형분류기 마진은 만족 (거리 한계 있을 때).
2. **$c_i$ 추정 어려움**: 최악의 경우 상한을 찾아야 하는데, 함수 구조에 의존. 느슨할 수 있다.
3. **분포 자유**: 분포를 모르는 것이 강점이지만, 실제 구조를 활용하면 tighter bound 가능.
4. **독립성**: iid 가정이 필수.
5. **함수 의존성**: 함수가 충분히 stable하지 않으면 $c_i$들이 커져 bound가 약해진다.

---

## 📌 핵심 정리

- **Bounded differences**: 함수 $f$가 한 좌표 변화에 $c_i$ 이상 변하지 않음.
- **McDiarmid 부등식**: $\mathbb{P}(|f(X)-\mathbb{E}[f(X)]| \geq t) \leq 2\exp(-2t^2/\sum c_i^2)$.
- **Doob martingale**: 조건부 기대값의 수열로 bounded difference를 구현.
- **Azuma-Hoeffding**: Martingale 차분의 지수 집중.
- **Ch5 연결**: Rademacher 복잡도 자체의 집중을 bound하는 열쇠.
- **부등식 위계**: Hoeffding (표본 평균) ⊂ McDiarmid (일반 함수) ⊂ Rademacher (데이터 의존).

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> $f(X) = \sum_{i=1}^n X_i$ (합)이 bounded differences $c_i = 1$ (각 $X_i \in [0,1]$)을 만족함을 보여라. 이것이 Hoeffding의 특수 경우임을 설명하라.</summary>

<br/>

**해설**. $X, X'$가 $i$-번째 좌표에서만 다르면 ($X_i \neq X'_i$, 나머지는 같음):
$$|f(X) - f(X')| = |X_i - X'_i| \leq |1 - 0| = 1$$
(왜냐하면 $X_i, X'_i \in [0,1]$이므로). 따라서 $c_i = 1$ 모두.

McDiarmid: $\mathbb{P}(|\sum X_i - \mathbb{E}[\sum X_i]| \geq t) \leq 2\exp(-2t^2/n)$. 이것은 정확히 Hoeffding (표본 합 버전)이다. 즉, Hoeffding은 McDiarmid의 선형 함수 특수 경우. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> $f(X) = \text{median}(X)$의 bounded difference를 구하라. 한 $x_i$를 바꾸면 중앙값이 최대 얼마만큼 변하는가?</summary>

<br/>

**해설**. 중앙값은 순서 통계량 중 중앙. 한 점을 바꿔도 중앙값은 매우 천천히 변한다. 예: $n = 5$, 정렬된 값이 $[1, 2, 3, 4, 5]$일 때 중앙값은 3. $x_5 = 5$를 $x_5 = 1000$으로 바꿔도 중앙값은 여전히 3. 

하지만 최악: $[1, 2, 3, 4, 5]$에서 $x_3 = 3$을 $x_3 = \infty$로 바꾸면 중앙값이 변할 수 있다. 정확히는, bounded $X_i \in [a, b]$일 때 $c_i = (b-a)$. (한 점이 극단으로 가면 중앙값이 한 "뻣금" 움직인다.)

따라서 $\sum c_i^2 = n(b-a)^2$, McDiarmid bound는 평균과 비슷한 속도. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> 5-fold cross-validation 오차 $\text{CV}_5$가 bounded difference를 만족하는가? 각 폴드의 test error를 바꿔도 CV 오차가 얼마나 변하는가?</summary>

<br/>

**해설**. $\text{CV}_5 = \frac{1}{5}\sum_{i=1}^5 L_i$ (5개 폴드의 test error 평균). 한 폴드 $L_i$를 다른 fold로 바꾸면 (다른 데이터로 재학습 후 재평가), CV 오차는 최대 $1/5$ 정도 변한다 (loss 범위가 $[0, 1]$이면).

따라서 $c_i = 1/5$ (정규화된 형태). McDiarmid: $\mathbb{P}(|\text{CV}_5 - \mathbb{E}[\text{CV}_5]| \geq t) \leq 2\exp(-2t^2/(5 \cdot (1/5)^2)) = 2\exp(-50t^2)$.

이것이 **nested cross-validation과 confidence interval 유도의 수학적 기반**이다 (Ch7-03). $\square$

</details>

---

<div align="center">

◀ [이전: 02. Hoeffding 부등식](./02-hoeffding.md) | [📚 README](../README.md) | [다음: 04. Bernstein 부등식 ▶](./04-bernstein.md)

</div>
