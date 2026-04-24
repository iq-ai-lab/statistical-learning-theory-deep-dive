# 01. Markov·Chebyshev 부등식

## 🎯 핵심 질문

- **Markov 부등식** $\mathbb{P}(X \geq t) \leq \mathbb{E}[X]/t$는 어떻게 증명되는가? 비음수 확률변수만 쓸 수 있는 이유는?
- **Chebyshev 부등식** $\mathbb{P}(|X - \mu| \geq t) \leq \sigma^2/t^2$은 분산이 작을수록 확률변수가 평균 근처에 집중된다는 직관을 어떻게 정량화하는가?
- 왜 Markov·Chebyshev의 $O(1/t^2)$ 꼬리 경계는 **일반화 오차 bound에 부족**한가? $O(e^{-nt^2})$ 지수적 경계가 왜 필요한가?
- **Paley-Zygmund 부등식**은 Chebyshev의 어떤 "반대 방향"을 보여주는가?
- Markov의 indicator trick은 Ch5-04(Massart)부터 다시 나타나는 **기본 기법**이다.

---

## 🔍 왜 이 이론이 현대 ML에서 중요한가

집중부등식은 **"몇 개의 샘플로 모집단 평균을 얼마나 정확히 추정할 수 있는가"**에 답하는 이론의 기초다. Markov·Chebyshev는 이 이론의 **첫 번째 계단**이다. Hoeffding이 $e^{-2nt^2}$을 얻기 전에, 이 둘은 "충분히 약한 경계지만 증명 가능한 가장 단순한 형태"를 제공한다.

실무적으로는 이 경계들이 자주 **이론적으로는 충분하지만 계산상으로는 느슨(loose)**하다. 예를 들어, 의료 시험에서 10명 샘플의 성공률을 봤을 때, Chebyshev는 "95% 확률로 진짜 성공률이 관찰값 ±30% 안"이라고만 보장한다. 반면 Hoeffding은 같은 신뢰도에서 "±30% 이상은 지수적으로 희물음"이라고 더 강하게 말한다. Ch2 전체의 **"왜 다음 부등식이 필요한가"를 이해하려면, 이전 부등식의 한계를 먼저 보아야 한다.**

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 확률공간, 확률변수, 기대값, 지시함수
- 기초: 비음수 확률변수, 조건부 기대값(Chebyshev 유도에서)
- Ch1-01: 손실함수와 위험의 정의

---

## 📖 직관적 이해

### Markov: "평균이 크지 않으면 오른쪽 꼬리는 작다"

고양이 무게의 평균이 5kg이라고 하자. "20kg 이상 무거운 고양이가 있을 확률은 최대 얼마인가?" 직관: 평균이 5kg인데 평균의 4배까지 가려면 엄청 희귀해야 한다. Markov는 이를 정량화한다:

$$\mathbb{P}(X \geq 20) \leq \frac{\mathbb{E}[X]}{20} = \frac{5}{20} = 0.25.$$

놀랍게도 **X의 분포를 전혀 모르고도** 비음수성만으로 이 바운드가 나온다.

### Chebyshev: "분산이 작으면 평균 근처에 집중"

표준편차 $\sigma = 1$이고 평균 $\mu = 10$인 확률변수에서, "평균에서 3 표준편차 이상 벗어날 확률"은:

$$\mathbb{P}(|X - 10| \geq 3) \leq \frac{\sigma^2}{3^2} = \frac{1}{9} \approx 0.111.$$

**분포를 몰라도** "표준편차 관점에서의 꼬리"를 bound할 수 있다 — 이것이 Chebyshev의 마력이고, 정규분포가 아닌 거의 모든 분포에 적용 가능한 까닭이다.

### 왜 지수적 경계가 필요한가?

$n$ 개의 Bernoulli(1/2) 표본의 합 $S_n = \sum X_i$를 생각하자. 평균은 $\mu = n/2$, 분산은 $\sigma^2 = n/4$.

Chebyshev로 "표본 평균이 진짜 평균에서 0.1 이상 벗어날 확률":
$$\mathbb{P}(|\bar{X} - 1/2| \geq 0.1) \leq \frac{n/4}{(0.1n)^2} = \frac{n/4}{0.01n^2} = \frac{25}{n} \approx \frac{25}{100} = 0.25 \quad (n=100).$$

하지만 정규근사·정확한 이항분포로는 이 확률은 약 $10^{-5}$ 정도로 훨씬 작다. **Chebyshev는 최악의 경우만 본다** — "분포가 얼마나 이상한가"를 상한하는 것이지, 실제 분포의 꼬리를 본 것이 아니다.

ML에서: $n = 10000$명의 시험 데이터로 모델의 오차를 추정할 때, Chebyshev는 여전히 "오차가 0.01 이상 벗어날 확률"을 0.25로 upper bound한다 — 거의 쓸모가 없다. Hoeffding·Bernstein은 $e^{-2n(0.01)^2} = e^{-2}$로 약 0.14 수준까지 떨어뜨린다.

---

## ✏️ 엄밀한 정의

### 정의 2.1 (비음수 확률변수와 좌우 꼬리)

$X \geq 0$ a.s.인 확률변수에 대해 $t > 0$:
- **오른쪽 꼬리(right tail)**: $\mathbb{P}(X \geq t)$
- **왼쪽 꼬리(left tail)**: $\mathbb{P}(X \leq -t)$ (음수 값 가능할 때)

확률변수 $X$에 대해 중심화된(centered) 버전: $Y = X - \mathbb{E}[X]$로 $\mathbb{E}[Y] = 0$.

### 정의 2.2 (Chebyshev를 위한 집중 용어)

확률변수 $X$가 기대값 $\mu = \mathbb{E}[X]$ 주변에 **집중(concentrate)**된다 ⟺ 대부분의 확률이 $\mu$ 근처의 작은 구간 내에 집중.

이를 정량화하는 지표:
- **분산(variance)**: $\sigma^2 = \text{Var}(X) = \mathbb{E}[(X - \mu)^2]$
- **표준편차**: $\sigma = \sqrt{\sigma^2}$

---

## 🔬 정리와 증명

### 정리 2.1 (Markov 부등식)

$X$가 **비음수** 확률변수이고 $t > 0$이면
$$\mathbb{P}(X \geq t) \leq \frac{\mathbb{E}[X]}{t}.$$

**증명**. 지시함수 indicator trick을 사용한다:
$$\mathbb{E}[X] = \mathbb{E}[X \cdot \mathbb{1}[X \geq t] + X \cdot \mathbb{1}[X < t]] \geq \mathbb{E}[X \cdot \mathbb{1}[X \geq t]] \geq \mathbb{E}[t \cdot \mathbb{1}[X \geq t]] = t \cdot \mathbb{P}(X \geq t).$$

첫 번째 부등식은 $X \geq 0$이므로 $X \cdot \mathbb{1}[X < t] \geq 0$ 항을 버린 것. 두 번째 부등식은 $X \geq t$일 때 $X \cdot \mathbb{1}[X \geq t] = X \geq t$를 사용. 양변을 $t$로 나누면 $\mathbb{P}(X \geq t) \leq \mathbb{E}[X]/t$. $\square$

### 정리 2.2 (Chebyshev 부등식)

모든 확률변수 $X$와 $t > 0$에 대해
$$\mathbb{P}(|X - \mathbb{E}[X]| \geq t) \leq \frac{\text{Var}(X)}{t^2}.$$

**증명**. $Y = (X - \mathbb{E}[X])^2$라 두면 $Y \geq 0$. Markov를 $Y$에 적용:
$$\mathbb{P}(Y \geq t^2) \leq \frac{\mathbb{E}[Y]}{t^2} = \frac{\mathbb{E}[(X - \mathbb{E}[X])^2]}{t^2} = \frac{\text{Var}(X)}{t^2}.$$

한편 $Y \geq t^2 \iff (X - \mathbb{E}[X])^2 \geq t^2 \iff |X - \mathbb{E}[X]| \geq t$. 따라서
$$\mathbb{P}(|X - \mathbb{E}[X]| \geq t) = \mathbb{P}(Y \geq t^2) \leq \frac{\text{Var}(X)}{t^2}. \qquad \square$$

> **적용**: 표본 평균 $\bar{X}_n = \frac{1}{n}\sum X_i$에서 $\text{Var}(\bar{X}_n) = \sigma^2/n$이므로
> $$\mathbb{P}(|\bar{X}_n - \mu| \geq t) \leq \frac{\sigma^2}{nt^2}.$$
> $n$ 증가에 따라 $1/n$ 속도로 떨어진다 — 하지만 **$1/t^2$ 때문에 $t$ 한 자리 바꿀 때마다 $100$배씩 손해**.

### 정리 2.3 (Paley-Zygmund 부등식 — Chebyshev의 반대)

비음수 확률변수 $X$와 $0 < \lambda < 1$에 대해
$$\mathbb{P}(X \geq \lambda \mathbb{E}[X]) \geq (1 - \lambda)^2 \frac{(\mathbb{E}[X])^2}{\mathbb{E}[X^2]}.$$

**증명 스케치**. Chebyshev를 $X^2$에 적용하면 $\mathbb{P}(|X - \mu| \leq \mu(1-\lambda))$를 바운드할 수 있고, 이로부터 왼쪽 꼬리를 얻는다. 상세는 Boucheron et al. (2013) 참조. $\square$

**의미**: 분산이 작으면 (즉, $\mathbb{E}[X^2]$가 $(\mathbb{E}[X])^2$에 가까우면) $X$가 $\mathbb{E}[X]$ 근처에 있을 확률이 크다.

---

## 💻 NumPy 구현 검증

### 실험 1: Markov vs Chebyshev vs 실제 꼬리

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# 분포 1: Exponential(1) — Markov가 의도한 "heavy tail" 분포
X_exp = rng.exponential(1, 100000)
mu_exp = X_exp.mean()
sig_exp = X_exp.std()

t_vals = np.linspace(0.5, 5, 20)
empirical_P = []
markov_bound = []
chebyshev_bound = []

for t in t_vals:
    emp = np.mean(X_exp >= t)
    markov = mu_exp / t
    chebyshev = sig_exp**2 / t**2
    empirical_P.append(emp)
    markov_bound.append(markov)
    chebyshev_bound.append(chebyshev)

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(t_vals, empirical_P, 'ko-', linewidth=2, markersize=6, label='Empirical P(X≥t)')
ax.semilogy(t_vals, markov_bound, 'bs--', alpha=0.6, label='Markov bound')
ax.semilogy(t_vals, chebyshev_bound, 'r^--', alpha=0.6, label='Chebyshev bound')
ax.set_xlabel('threshold t')
ax.set_ylabel('Probability (log scale)')
ax.set_title('Markov·Chebyshev bounds vs 실제 꼬리 (Exponential 분포)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# → Markov는 실제보다 loose, Chebyshev는 더 loose
```

### 실험 2: 표본 평균의 Chebyshev 수렴

```python
# X_i ~ Bernoulli(0.3), 표본 평균 \bar{X}_n의 n에 따른 편차
p = 0.3
n_vals = [10, 50, 100, 500, 2000]
t = 0.05  # 편차 threshold

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 좌: Chebyshev bound가 n과 함께 어떻게 감소
sigma2 = p * (1 - p)
bounds = [sigma2 / (n * t**2) for n in n_vals]
axes[0].loglog(n_vals, bounds, 'o-', linewidth=2, markersize=8, label='Chebyshev P(|X̄ₙ-p|≥t)')
axes[0].loglog(n_vals, [1/np.sqrt(n) for n in n_vals], '--', label='~1/√n 비교')
axes[0].set_xlabel('n')
axes[0].set_ylabel('Probability')
axes[0].set_title(f'표본 평균의 Chebyshev bound (t={t}, p={p})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 우: 실제 경험적 편차
empirical_probs = []
for n in n_vals:
    means = []
    for _ in range(5000):
        X = rng.binomial(1, p, n)
        means.append(X.mean())
    empirical_probs.append(np.mean(np.abs(np.array(means) - p) >= t))

axes[1].loglog(n_vals, bounds, 'o-', label='Chebyshev bound', linewidth=2)
axes[1].loglog(n_vals, empirical_probs, 's-', label='Empirical', linewidth=2)
axes[1].set_xlabel('n')
axes[1].set_ylabel('P(|X̄ₙ-p|≥0.05)')
axes[1].set_title('Chebyshev bound vs 실제')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# → bound가 real보다 크고, 1/n 속도는 두 경우 모두 맞음
```

### 실험 3: Paley-Zygmund 하한 검증

```python
# 비음수 X: Beta(2, 5) (왼쪽 치우친 분포)
a, b = 2, 5
X_beta = rng.beta(a, b, 100000)
mu = X_beta.mean()
ex2 = np.mean(X_beta ** 2)

lambda_vals = np.linspace(0.1, 0.9, 20)
pz_lower = []
empirical_lower = []

for lam in lambda_vals:
    # Paley-Zygmund lower
    pz = (1 - lam)**2 * mu**2 / ex2
    pz_lower.append(max(pz, 0))
    # Empirical
    emp = np.mean(X_beta >= lam * mu)
    empirical_lower.append(emp)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(lambda_vals, pz_lower, 'bs--', alpha=0.6, label='Paley-Zygmund lower')
ax.plot(lambda_vals, empirical_lower, 'ko-', linewidth=2, label='Empirical P(X ≥ λμ)')
ax.set_xlabel('λ (fraction of mean)')
ax.set_ylabel('Probability')
ax.set_title('Paley-Zygmund: 낮은 분산 ⟹ 평균 근처 높은 집중')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# → PZ lower bound가 실제 확률의 대략적 진실을 포착
```

---

## 🔗 ML 알고리즘 연결

| 적용 | Markov/Chebyshev 사용 | 한계 |
|-----|----------------------|------|
| **Cross-validation error** | Chebyshev로 test error 집중 | $O(1/\sqrt{n})$ bound는 too weak |
| **SGD 수렴 증명** | Markov로 gradient norm의 꼬리 | distributed setting에서 복잡 |
| **매개변수 추정** | 불편 추정량의 분산 bound | distribution-free이지만 loose |
| **Dimensionality reduction** | 가우시안 random projection의 거리 보존 | concentration 필요 |

**앞으로**: Ch2-02부터는 Markov·Chebyshev를 "**출발점**"으로 삼아, Hoeffding의 **$e^{-nt^2}$ 지수적 경계**를 얻는 경로를 따라간다.

---

## ⚖️ 가정과 한계

1. **Markov는 비음수만**: 음수 값을 가질 수 있는 분포(정규분포, 라플라스)에는 직접 쓸 수 없다. Chebyshev로 우회.
2. **분포 자유**: 분포 가정이 없어서 "최악의 경우"에만 유효. 실제 분포가 깔끔하면 bound는 loose.
3. **$t$에 대한 거듭제곱 의존**: Chebyshev의 $1/t^2$ 때문에 편차가 커질수록 매우 빠르게 타이트하지 않다.
4. **이변량 이상 구조 놓침**: 독립성 외에 변수 간 의존성이나 구조를 이용하지 않는다 (Ch2-02의 Hoeffding과 달리).
5. **편향·불편성**: Chebyshev는 중심화 조건(중심 $\mu$ 알려짐)을 가정하는데, 추정 상황에서는 bias term이 추가.

---

## 📌 핵심 정리

- **Markov**: 비음수 $X$에서 $\mathbb{P}(X \geq t) \leq \mathbb{E}[X]/t$. **분포 무관**, but very loose.
- **Chebyshev**: 임의의 $X$에서 $\mathbb{P}(|X-\mu| \geq t) \leq \sigma^2/t^2$. 분산으로 집중 정량화.
- **표본 평균**: $\bar{X}_n$에서 Chebyshev는 $O(1/(nt^2))$. **$n$ 증가에도 $t$ 차이로 인한 손해**.
- **Paley-Zygmund**: 역방향 — 저분산 ⟹ 높은 집중.
- **다음 필요성**: Hoeffding부터는 **MGF·Chernoff 방법**으로 $e^{-nt^2}$ 지수 속도 달성.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> Markov 부등식에서 indicator trick을 사용하는 이유를 설명하라. 즉, 왜 "분포를 모르고도" 지시함수만으로 bound가 나오는가?</summary>

<br/>

**해설**. Markov는 **기대값 하나 정보만 사용**한다: $\mathbb{E}[X] = $ 평균 무게. $X \geq t$ 사건에서 $X \geq t$이므로, 그 사건의 기여분이 최소 $t \cdot \mathbb{P}(X \geq t)$임을 표현하는 것이 indicator의 역할이다. Chebyshev는 한 단계 더: $X$를 $(X-\mu)^2$로 변환해서 $\mathbb{E}[(X-\mu)^2]$(분산)을 활용한다. 이것이 "**더 강한 조건(분산 정보) → 더 타이트한 bound**"의 원리. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> $n$ 독립 Bernoulli(1/2) 시행의 합 $S_n = \sum_{i=1}^n X_i$에서 $S_n = k$일 확률이 정규근사로 $\Phi$로 주어질 때, Chebyshev bound $\frac{n/4}{(cn)^2} = \frac{1}{4c^2 n}$이 왜 "실제 이항 꼬리 $e^{-2(cn)^2}$"보다 훨씬 느리게 감소하는가?</summary>

<br/>

**해설**. Chebyshev는 **최악의 분포를 가정**한다. 정규분포는 꼬리가 빠르게 떨어지지만(가우시안), Chebyshev는 "만약 분산만 있고 분포는 알 수 없다면?"을 본다. 따라서 $1/n$ 속도만 보장할 수 있다. Hoeffding(Ch2-02)은 **bounded variable의 MGF를 구체적으로 사용**해서 $e^{-2nt^2}$ 같은 지수 속도를 얻는다 — "분포는 모르지만, 범위는 알 수 있다"는 정보를 활용. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> 의료 진단 시스템: 1000명 테스트에서 양성 판정률 0.45, 분산 추정 0.25. Chebyshev로 진짜 양성률이 0.35 이상 0.55 이하일 확률은? 이 bound가 실용적으로는 왜 "너무 약한가"?</summary>

<br/>

**해설**. $\bar{X}_n = 0.45$, $\sigma = \sqrt{0.25} = 0.5$, $t = 0.1$. Chebyshev: $\mathbb{P}(|\bar{X}_n - \mu| \geq 0.1) \leq \frac{0.25}{1000 \cdot 0.01} = 0.25$. 즉, **최대 75% 신뢰도**만 보장 — 의료 진단에서는 95% 이상 신뢰도가 필요하다. Hoeffding이면 $2\exp(-2 \cdot 1000 \cdot 0.01) \approx 2e^{-20} \approx 5 \times 10^{-9}$ — **사실상 확실**. 이것이 "표본 크기가 증가하면 경험적 추정이 신뢰할 수 있다"는 SLT의 근거. $\square$

</details>

---

<div align="center">

◀ [이전: Ch1-05 IID 가정](../ch1-learning-formulation/05-iid-assumption.md) | [📚 README](../README.md) | [다음: 02. Hoeffding 부등식 ▶](./02-hoeffding.md)

</div>
