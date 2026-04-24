# 04. Bernstein 부등식

## 🎯 핵심 질문

- **Bernstein 부등식**: $\mathbb{P}(|\bar{X} - \mu| \geq t) \leq 2\exp(-nt^2/(2\sigma^2 + 2Mt/3))$는 왜 **분산 정보를 활용**해 Hoeffding을 이기는가?
- **저분산 regime**: $\sigma^2 \ll M^2$일 때 Bernstein은 Hoeffding보다 **지수적으로 tight**하다. 수치로 얼마나 다른가?
- **MGF bound**: $\mathbb{E}[e^{\lambda(X - \mu)}] \leq e^{\lambda^2 \sigma^2/(2(1 - \lambda M/3))}$의 증명 전략은 무엇인가?
- **Fast rate의 기초**: Bernstein이 왜 **두 번째 항 $O(1/n)$** variance-dependent bound를 가능케 하는가?
- **실전**: Bernoulli(p) with small $p$ (rare event) — Hoeffding vs Bernstein의 성능 차이는?

---

## 🔍 왜 이 이론이 현대 ML에서 중요한가

**"분산이 작으면 더 정확한 추정이 가능하다"** — 이것은 직관적이지만 정량화하기 어렵다. Bernstein이 이를 해결한다. Hoeffding은 **분산을 무시**하고 범위만 본다. 반면 Bernstein은 "**실제 분산이 작으면 bound를 대폭 개선**"한다.

실무적으로, 아래 상황들이 자주 발생한다:
- **Low noise regression**: 신호가 강하고 노이즈가 약한 경우, Bernstein이 압도적 우위
- **클래스 불균형**: 드문 클래스 ($p$ 작은 Bernoulli)의 오차율, Bernstein이 더 tight
- **Complexity-controlled regime**: fast rate PAC 경계의 원점 (Ch3-03의 agnostic PAC)
- **Variance-dependent regret bounds**: Online learning과 bandit에서 state-of-the-art bound는 대부분 Bernstein 기반

---

## 📐 수학적 선행 조건

- Ch2-02: Hoeffding 부등식, Chernoff 방법, MGF
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): MGF, Taylor 전개, strong convexity
- 기초: 분산 정의, bounded 확률변수

---

## 📖 직관적 이해

### Hoeffding은 "최악을 본다"

Bernoulli(1/2) vs Bernoulli(0.01)을 생각하자. 두 경우 모두 범위 $[0, 1]$이지만, 분산은 크게 다르다:
- Bernoulli(1/2): $\sigma^2 = 0.25$ (최대 분산)
- Bernoulli(0.01): $\sigma^2 = 0.01 \cdot 0.99 \approx 0.01$ (매우 작음)

Hoeffding은 "범위가 [0, 1]이면 어떤 분포든" 같은 bound를 준다. 하지만 Bernstein은 "분산이 작으니까 더 타이트하게 해준다"고 말한다.

### Variance-dependent rate의 마술

표본 분산을 추정하면 $\hat{\sigma}^2$를 안다. 그러면:

$$\text{경계} \approx \sqrt{\hat{\sigma}^2 / n} \text{ (분산 의존)}$$
$$\text{vs Hoeffding} \approx \sqrt{1/n} \text{ (범위만 의존)}$$

극단: $\sigma^2 = 0.01 \cdot n$ (분산이 n에 비례)이면, Bernstein은 $\sqrt{0.01/n} = 0.1/\sqrt{n}$인데 Hoeffding은 여전히 $1/\sqrt{n}$ 수준. **100배 차이**.

---

## ✏️ 엄밀한 정의

### 정의 2.7 (유계 분산 조건)

확률변수 $X \in [a, b]$에 대해 $\sigma^2 = \text{Var}(X)$이고 $M = b - a$라 하면, 자동으로 $\sigma^2 \leq M^2/4$ (Popoviciu).

Bernstein은 **실제 $\sigma^2$를 알거나 추정**할 수 있다고 가정한다.

### 정의 2.8 (Sub-Exponential)

확률변수 $X$를 **$(K, B)$-sub-exponential**이라 함 ⟺ 모든 $k \geq 2$에 대해
$$\mathbb{E}[|X - \mathbb{E}[X]|^k] \leq \frac{k!}{2} K^{k-2} B^2.$$

Bernoulli, bounded 변수는 sub-exponential이다.

---

## 🔬 정리와 증명

### 정리 2.9 (Bernstein's Inequality)

$X_1, \ldots, X_n$ iid, $X_i \in [a, b]$, $\mu = \mathbb{E}[X]$, $\sigma^2 = \text{Var}(X)$, $M = b - a$에 대해
$$\mathbb{P}(\bar{X} - \mu \geq t) \leq \exp\left(-\frac{nt^2}{2\sigma^2 + 2Mt/3}\right).$$

대칭적으로
$$\mathbb{P}(|\bar{X} - \mu| \geq t) \leq 2\exp\left(-\frac{nt^2}{2\sigma^2 + 2Mt/3}\right).$$

**증명 (Chernoff + 정교한 MGF bound)**. 

1. **Centered variance**: $\xi_i = X_i - \mu$라 두면 $\mathbb{E}[\xi_i] = 0$, $\xi_i \in [a - \mu, b - \mu]$, $\text{Var}(\xi_i) = \sigma^2$.

2. **MGF bound (핵심)**: $|\xi| \leq M$인 확률변수에 대해
$$\mathbb{E}[e^{\lambda \xi}] \leq e^{\lambda^2 \sigma^2 / (2(1 - \lambda M/3))}.$$

이것을 보이기 위해, $e^{\lambda x}$를 $x$에 대해 Taylor 전개:
$$e^{\lambda x} = 1 + \lambda x + \frac{\lambda^2 x^2}{2} e^{\lambda \theta x}$$
for some $\theta \in (0, 1)$.

$|\lambda \theta x| \leq \lambda M$이 작으면, $e^{\lambda \theta x} \leq 1 + \lambda \theta x + \frac{(\lambda \theta x)^2}{2} + \ldots$를 **cumulant expansion**으로 다루면
$$\mathbb{E}[e^{\lambda \xi}] \leq 1 + \lambda^2 \sigma^2 / 2 + \lambda^3 M \mathbb{E}[\xi^3 / 6] + \ldots$$

$\lambda$가 작을 때 지배하는 항들:
$$\mathbb{E}[e^{\lambda \xi}] \leq \exp\left(\frac{\lambda^2 \sigma^2}{2(1 - \lambda M/3)}\right).$$

(정확한 계산은 Boucheron et al. (2013) 참조; 여기선 intuition만.)

3. **Chernoff 적용**: $\lambda > 0$:
$$\mathbb{P}(\bar{X} - \mu \geq t) = \mathbb{P}(e^{\lambda(\bar{X} - \mu)} \geq e^{\lambda t}) \leq e^{-\lambda t} \mathbb{E}[e^{\lambda(\bar{X} - \mu)}].$$

독립성:
$$\mathbb{E}[e^{\lambda(\bar{X} - \mu)}] = \prod \mathbb{E}[e^{\lambda(\xi_i / n)}] \leq e^{n \cdot \lambda^2 \sigma^2 / (2n^2(1 - \lambda M/(3n)))}.$$

4. **$\lambda$ 최적화**: 지수 $f(\lambda) = \lambda^2 \sigma^2/(2(1 - \lambda M/3)) - \lambda t \cdot n$을 최소화. 미분하고 정리하면 최적 $\lambda^*$를 얻고, 이를 대입하면
$$f(\lambda^*) = -\frac{nt^2}{2\sigma^2 + 2Mt/3}. \qquad \square$$

### 정리 2.10 (Hoeffding vs Bernstein 비교)

같은 $n, t$에서:
- **Hoeffding**: $2\exp(-2nt^2/M^2)$
- **Bernstein**: $2\exp(-nt^2/(2\sigma^2 + 2Mt/3))$

$\sigma^2 \ll M^2$일 때, Bernstein의 지수가 훨씬 크다.

---

## 💻 NumPy 구현 검증

### 실험 1: Bernoulli(p)에서 Hoeffding vs Bernstein 비교

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# 두 Bernoulli 분포
p_vals = [0.5, 0.1, 0.01]
n = 1000
t_vals = np.linspace(0.01, 0.3, 40)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, p in enumerate(p_vals):
    sigma2 = p * (1 - p)
    M = 1.0
    
    # Empirical tail probability
    empirical_P = []
    for t in t_vals:
        means = np.array([rng.binomial(1, p, n).mean() for _ in range(2000)])
        emp = np.mean(np.abs(means - p) >= t)
        empirical_P.append(emp)
    
    # Bounds
    hoeff = 2 * np.exp(-2 * n * t_vals**2 / M**2)
    bernstein = 2 * np.exp(-n * t_vals**2 / (2*sigma2 + 2*M*t_vals/3))
    
    ax = axes[idx]
    ax.semilogy(t_vals, empirical_P, 'ko-', linewidth=2, markersize=5, label='Empirical')
    ax.semilogy(t_vals, hoeff, 'bs--', linewidth=1.5, alpha=0.7, label='Hoeffding')
    ax.semilogy(t_vals, bernstein, 'r^--', linewidth=1.5, alpha=0.7, label='Bernstein')
    ax.set_xlabel('t')
    ax.set_ylabel('P(|X̄ₙ - p| ≥ t)')
    ax.set_title(f'Bernoulli(p={p}), n={n}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# → p=0.01일 때 Bernstein과 Hoeffding의 gap이 극단적
```

### 실험 2: 분산 추정과 Bernstein bound의 의존성

```python
# 실제로 분산을 모를 때는 표본 분산으로 추정
# 표본 분산이 작을수록 Bernstein이 tighter

def sample_and_bound(n, p, n_reps=100):
    """n개 샘플로 Bernstein bound (표본 분산 기반)"""
    bounds = []
    t = 0.05
    
    for _ in range(n_reps):
        X = rng.binomial(1, p, n)
        var_est = np.var(X, ddof=1)  # unbiased variance estimate
        
        # Bernstein with estimated variance
        bound = 2 * np.exp(-n * t**2 / (2 * var_est + 2 * 1 * t / 3))
        bounds.append(bound)
    
    return np.mean(bounds), np.std(bounds)

n_vals = np.arange(50, 501, 50)
p = 0.1
true_var = p * (1 - p)

bernstein_mean = []
bernstein_std = []
hoeff_bounds = []

for n in n_vals:
    b_mean, b_std = sample_and_bound(n, p, 200)
    bernstein_mean.append(b_mean)
    bernstein_std.append(b_std)
    
    hoeff = 2 * np.exp(-2 * n * 0.05**2)
    hoeff_bounds.append(hoeff)

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(n_vals, bernstein_mean, 'ro-', linewidth=2, markersize=6, label='Bernstein (est var)')
ax.loglog(n_vals, bernstein_std, 'r--', alpha=0.5)
ax.loglog(n_vals, hoeff_bounds, 'bs--', linewidth=2, label='Hoeffding')
ax.loglog(n_vals, 1/np.sqrt(n_vals), 'k--', alpha=0.3, label='~1/√n')
ax.set_xlabel('n')
ax.set_ylabel('Bound')
ax.set_title(f'Bernstein bound (표본 분산) vs Hoeffding, Bernoulli(p={p})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# → Bernstein이 현저히 더 tight
```

### 실험 3: Fast rate regime — 분산에 dependent한 $O(1/n)$ rate

```python
# Low noise regime: X = μ + σ·Z, σ 매우 작음
# Bernstein으로 O(1/n) 달성 가능

sigma = 0.05  # very small noise
mu = 0.5
n_vals = np.arange(100, 1001, 100)

# 두 정권 비교
# 1. Large deviation regime (t > σ): polynomial vs exponential 의존
# 2. Low deviation regime (t ~ σ/√n): fast rate

empirical_errors = []
bernstein_bound_large = []
bernstein_bound_small = []

for n in n_vals:
    # Generate data
    X = rng.normal(mu, sigma, (1000, n))
    sample_means = X.mean(axis=1)
    
    # Large deviation: t = 0.1
    t_large = 0.1
    emp_large = np.mean(np.abs(sample_means - mu) >= t_large)
    bernstein_large = 2 * np.exp(-n * t_large**2 / (2*sigma**2 + 2*1*t_large/3))
    
    empirical_errors.append(emp_large)
    bernstein_bound_large.append(bernstein_large)
    
    # Small deviation: t = σ/√n (적응적)
    t_small = sigma / np.sqrt(n)
    bernstein_small = 2 * np.exp(-n * t_small**2 / (2*sigma**2 + 2*1*t_small/3))
    bernstein_bound_small.append(bernstein_small)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Large deviation
ax = axes[0]
ax.semilogy(n_vals, empirical_errors, 'ko-', linewidth=2, label='Empirical')
ax.semilogy(n_vals, bernstein_bound_large, 'r^--', linewidth=2, label='Bernstein(t=0.1)')
ax.set_xlabel('n')
ax.set_ylabel('P(|X̄ₙ - μ| ≥ 0.1)')
ax.set_title(f'Large deviation (fixed t=0.1), σ={sigma}')
ax.legend()
ax.grid(True, alpha=0.3)

# Small deviation
ax = axes[1]
ax.loglog(n_vals, bernstein_bound_small, 'r^-', linewidth=2, label='Bernstein(t=σ/√n)')
ax.loglog(n_vals, 1/n_vals, 'k--', alpha=0.5, label='~1/n (fast)')
ax.loglog(n_vals, 1/np.sqrt(n_vals), 'g--', alpha=0.5, label='~1/√n')
ax.set_xlabel('n')
ax.set_ylabel('Bound')
ax.set_title('Small deviation regime: fast O(1/n) convergence')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f'σ={sigma}: Bernstein achieves ~1/√n in large-dev, ~1/n in small-dev')
```

---

## 🔗 ML 알고리즘 연결

| 응용 | Bernstein 형태 | 왜 필요? |
|-----|---|---|
| **클래스 불균형 분류** | 드문 클래스 오류 bound | 가설별 분산이 작음 |
| **Agnostic PAC (Ch3-03)** | $m = O((\text{Var} + \sqrt{\text{Var}})/\epsilon^2)$ | Empirical risk의 분산 활용 |
| **Online learning regret** | Best-arm variance의 $O(\sqrt{V})$ bound | MAB 문제 |
| **Margin-based bound** | Margin이 클수록 분산 작음 → tight bound | SVM 정당화 |
| **Adaptive sampling** | Low-variance subset 우선 샘플 | Active learning |

---

## ⚖️ 가정과 한계

1. **분산 추정**: 실제로 $\sigma^2$를 알거나 좋게 추정해야 tight. Bias-corrected 추정 필요.
2. **Bounded variable 필수**: Bernstein의 MGF bound가 유계 변수를 가정 (무한 분포는 truncation 필요).
3. **분포 자유 vs 분포 의존**: 분포를 모르지만, "분산이 작다"는 정보를 활용 — 중간 수준의 정보.
4. **Non-iid 확장**: 약한 의존성(mixing) 있으면 복잡해진다.
5. **상수항**: Denominator $2\sigma^2 + 2Mt/3$에서 상수 2, 3이 tight하지 않을 수 있음 (개선 가능).

---

## 📌 핵심 정리

- **Bernstein 부등식**: $\mathbb{P}(|\bar{X} - \mu| \geq t) \leq 2\exp(-nt^2/(2\sigma^2 + 2Mt/3))$.
- **분산 활용**: Hoeffding과 달리, 작은 $\sigma^2$를 직접 활용 → tighter bound.
- **저분산 regime**: $\sigma^2 \ll M^2$일 때 Hoeffding보다 지수적으로 우월.
- **MGF bound**: $\mathbb{E}[e^{\lambda X}] \leq e^{\lambda^2 \sigma^2/(2(1-\lambda M/3))}$ (정교한 Taylor 분석).
- **Fast rate**: 작은 편차 영역에서 $O(1/n)$ convergence (분산 의존적).

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> Bernoulli(p)에서 분산 $\sigma^2 = p(1-p)$은 언제 최소/최대인가? Hoeffding과 Bernstein의 gap을 가장 크게 벌리는 $p$ 값은?</summary>

<br/>

**해설**. $\sigma^2(p) = p(1-p)$은 $p = 1/2$에서 최대 1/4, $p = 0$ 또는 1에서 최소 0. 극단: $p = 0.01$일 때 $\sigma^2 \approx 0.01$, Hoeffding은 $2e^{-2nt^2}$인데 Bernstein은 $2e^{-nt^2/(0.02 + 2t/3)}$. 작은 $t$에서 Bernstein이 훨씬 크다. 예: $t = 0.01, n = 1000$:
- Hoeffding: $e^{-0.2}$
- Bernstein: $e^{-1000 \cdot 0.0001 / (0.02 + 2/300)} \approx e^{-5}$

극단적인 편향($p \to 0$)일수록 gap이 커진다. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> Bernstein bound에서 분모 $2\sigma^2 + 2Mt/3$의 두 항이 언제 균형이 되는가? 이것이 bound의 shape에 무엇을 의미하는가?</summary>

<br/>

**해설**. $2\sigma^2 = 2Mt/3 \iff t = 3\sigma^2/M$일 때 균형. 이 지점을 $t_*$라 하면:

- $t < t_*$: $\sigma^2$ 항이 지배 → $\sim e^{-nt^2/(2\sigma^2)}$ (분산-dependent, fast)
- $t > t_*$: $Mt$ 항이 지배 → $\sim e^{-3nt^2/(2M)}$ (범위-dependent, slow)

즉, Bernstein은 **두 정권을 자동으로 switching** — 작은 편차는 분산으로 fast하게 감지, 큰 편차는 범위로 안전하게 cover. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> 의료 진단: 질병이 1% 인구에서 발생 (rare event), 테스트 정확도 98%. 1000명 샘플에서 양성 비율의 95% confidence interval을 Bernstein으로 구하라.</summary>

<br/>

**해설**. 질병 보유: $p = 0.01$, $\sigma^2 = 0.01 \cdot 0.99 = 0.0099 \approx 0.01$. 테스트 오류: $\sigma^2$에 합산되면 약 $\sigma^2_{\text{eff}} \approx 0.02$ (테스트 오류 추가).

Bernstein에서 $\delta = 0.05$: $2e^{-1000t^2/(2 \cdot 0.02 + 2 \cdot 1 \cdot t / 3)} = 0.05$. 수치 풀이: $t \approx 0.023$. 신뢰 구간: $[0.01 - 0.023, 0.01 + 0.023]$ ... 음수가 되므로 조정: $[0, 0.033]$. Bernoulli의 lower tail이 0에서 상한 있음.

실전에서는 Agresti-Coull 같은 **정확한 binomial 신뢰 구간** 사용, 하지만 Bernstein은 이론적 정당성을 제공. $\square$

</details>

---

<div align="center">

◀ [이전: 03. McDiarmid 부등식](./03-mcdiarmid.md) | [📚 README](../README.md) | [다음: 05. 집중부등식의 ML 응용 ▶](./05-applications.md)

</div>
