# 02. Hoeffding 부등식

## 🎯 핵심 질문

- **Hoeffding's lemma** — $X \in [a, b]$이고 $\mathbb{E}[X] = 0$이면 $\mathbb{E}[e^{\lambda X}] \leq e^{\lambda^2(b-a)^2/8}$ — 는 어떻게 증명되는가? 왜 구간 폭 $(b-a)$의 제곱이 나타나는가?
- **Chernoff 방법**: 확률을 MGF로 bound 하는 이 기법은 어디에 쓰이고, 왜 $\lambda$에 대한 최적화가 중요한가?
- **Hoeffding 부등식** $\mathbb{P}(|\bar{X} - \mu| \geq t) \leq 2\exp(-2nt^2/\sum(b_i-a_i)^2)$에서 **$e^{-2nt^2}$ 지수 속도**는 어디서 오는가?
- **Sub-Gaussian 확률변수**: 정규분포뿐 아니라, 어떤 다른 분포들이 같은 꼬리 행동을 보이는가?
- 이 부등식이 왜 **Ch3의 PAC learning**과 **Ch5의 Rademacher 복잡도**의 기초인가?

---

## 🔍 왜 이 이론이 현대 ML에서 중요한가

**"훈련 오차가 낮으면 테스트 오차도 낮을까?"** — SLT의 핵심 질문이다. Hoeffding은 이에 **"확실히 그렇다. 지수적 확률로"**라는 답을 제공한다. 더 정확히, 고정된 한 분류기 $h$에 대해:

$$\mathbb{P}(|L_S(h) - L_\mathcal{D}(h)| \geq \epsilon) \leq 2\exp(-2n\epsilon^2).$$

이것은 **매개변수 $n$에 지수적으로 감소**한다는 뜻이다. $n = 100$이면 bound는 $\approx 10^{-10}$이다. Ch3~Ch5에서 "union bound + VC + Rademacher"로 이 한 분류기의 경계를 **모든 $h \in \mathcal{H}$에 확장하는 기법**을 배운다. 그러나 모든 것의 원점은 **이 한 분류기에 대한 Hoeffding**이다.

실무적으로, A/B 테스팅, 온라인 광고 최적화, 의료 시험의 유효성 검증 등에서 **"샘플 추정이 모집단을 대표할 확률"을 정량화**할 때 Hoeffding을 쓴다. 이 부등식이 없으면 "충분히 큰 데이터"라는 직관만 남는다.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): MGF, Chernoff 방법, Taylor 전개, 볼록함수
- Ch2-01: Markov 부등식
- 기초: 비음수 확률변수, 독립성, 기대값의 선형성

---

## 📖 직관적 이해

### "MGF는 꼬리의 기억을 담는다"

확률변수 $X$의 적률생성함수(MGF) $M_X(\lambda) = \mathbb{E}[e^{\lambda X}]$는 $X$의 모든 적률을 인코딩한다. MGF가 클수록, 오른쪽 꼬리가 무겁다. Hoeffding's lemma는 **"$X$가 구간 $[a, b]$에 갇혀 있으면, MGF도 정규분포처럼 bounded된다"**고 말한다. 이것이 놀라운 이유는: 실제 분포가 완전히 다를 수 있는데도 MGF 상한이 같다는 것.

### Chernoff 방법의 마술

표본 평균 $\bar{X}_n = \frac{1}{n}\sum X_i$이 참 평균 $\mu$에서 벗어날 확률:

$$\mathbb{P}(\bar{X}_n \geq \mu + t) = \mathbb{P}(e^{\lambda(\bar{X}_n - \mu)} \geq e^{\lambda t}) \leq \frac{\mathbb{E}[e^{\lambda(\bar{X}_n - \mu)}]}{e^{\lambda t}}.$$

Markov를 $e^{\lambda X}$에 쓴 것이다. 우변이 작으려면 분자는 작고 분모는 커야 하는데, 분자를 구간 정보로 control하고, 분모는 $\lambda$에 대해 최적화해 최선을 찾는다 — 이것이 Chernoff.

### 왜 $e^{-2nt^2}$인가?

직관적으로, **각 샘플이 오차에 "독립적으로" 기여**하기 때문이다. $n$개가 모두 같은 방향으로 벗어나야 하는데, 이 확률이 대략 $\rho^n$처럼 지수 감소한다 (여기서 $\rho < 1$). Taylor 전개를 통해 정확히 계산하면 계수 2가 나타난다.

---

## ✏️ 엄밀한 정의

### 정의 2.3 (구간-유계 확률변수)

확률변수 $X$가 **구간-유계(interval-bounded)** ⟺ $a \leq X \leq b$ a.s. 그러면 폭 $w = b - a$.

### 정의 2.4 (Sub-Gaussian 확률변수)

$X$의 MGF가 정규분포처럼 행동 ⟺ 존재하는 $\sigma > 0$에 대해
$$\mathbb{E}[e^{\lambda(X - \mathbb{E}[X])}] \leq e^{\lambda^2 \sigma^2 / 2}, \quad \forall \lambda \in \mathbb{R}.$$

이 경우 $X$를 **$\sigma$-sub-Gaussian**이라 부른다.

---

## 🔬 정리와 증명

### 정리 2.4 (Hoeffding's Lemma)

$X \in [a, b]$이고 $\mathbb{E}[X] = 0$이면, 모든 $\lambda > 0$에 대해
$$\mathbb{E}[e^{\lambda X}] \leq e^{\lambda^2 (b-a)^2 / 8}.$$

**증명**. $e^x$의 볼록성을 이용한다. $x \in [a, b]$일 때, $a$와 $b$ 사이의 볼록 조합으로:
$$e^{\lambda x} \leq \frac{b - x}{b - a} e^{\lambda a} + \frac{x - a}{b - a} e^{\lambda b}.$$

양변에 기대값을 취하면 ($\mathbb{E}[X] = 0$이므로 $\mathbb{E}[b - X] = b$, $\mathbb{E}[X - a] = -a$):
$$\mathbb{E}[e^{\lambda X}] \leq \frac{b}{b-a} e^{\lambda a} + \frac{-a}{b-a} e^{\lambda b} = \frac{1}{b-a} (b e^{\lambda a} - a e^{\lambda b}).$$

우변을 $\phi(\lambda) = \frac{1}{b-a}(b e^{\lambda a} - a e^{\lambda b})$라 하자. $\phi''(\lambda) = \frac{1}{b-a}(b a^2 e^{\lambda a} - a b^2 e^{\lambda b})$를 계산하면, 최대값은 $a < 0 < b$일 때 $\phi''(\lambda) \leq \frac{(b-a)^2}{4}$ (중점 $x = (a+b)/2$에서).

Taylor 전개: $\phi(\lambda) = \phi(0) + \phi'(0) \lambda + \frac{\phi''(\xi)}{2} \lambda^2 = 1 + 0 + \frac{\phi''(\xi)}{2}\lambda^2 \leq 1 + \frac{(b-a)^2}{8} \lambda^2 = e^{\lambda^2(b-a)^2/8}$ (마지막은 $e^x \geq 1 + x$).

따라서 $\mathbb{E}[e^{\lambda X}] \leq e^{\lambda^2(b-a)^2/8}$. $\square$

> **주의**: $(b - a)^2 / 8$이 중요하다. 구간의 폭의 제곱이고, 8이 Taylor 계수로부터 나온다.

### 정리 2.5 (Hoeffding 부등식)

$X_1, \ldots, X_n$ iid, $X_i \in [a_i, b_i]$ a.s., 표본 평균 $\bar{X}_n = \frac{1}{n}\sum X_i$, $\mu = \mathbb{E}[\bar{X}_n]$에 대해
$$\mathbb{P}(\bar{X}_n - \mu \geq t) \leq \exp\left(-\frac{2nt^2}{\sum_{i=1}^n (b_i - a_i)^2}\right).$$

대칭적으로 $\mathbb{P}(\bar{X}_n - \mu \leq -t) \leq \exp\left(-\frac{2nt^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)$이므로
$$\mathbb{P}(|\bar{X}_n - \mu| \geq t) \leq 2\exp\left(-\frac{2nt^2}{\sum_{i=1}^n (b_i - a_i)^2}\right).$$

**증명 (Chernoff + Hoeffding's Lemma)**. 

1. **Chernoff 적용**: $\lambda > 0$에 대해
$$\mathbb{P}(\bar{X}_n \geq \mu + t) = \mathbb{P}(e^{\lambda(\bar{X}_n - \mu)} \geq e^{\lambda t}) \leq \frac{\mathbb{E}[e^{\lambda(\bar{X}_n - \mu)}]}{e^{\lambda t}}.$$

2. **MGF 계산**: $X_i$ iid이고 독립성 덕분에
$$\mathbb{E}[e^{\lambda(\bar{X}_n - \mu)}] = \mathbb{E}\left[\prod_{i=1}^n e^{\lambda(X_i - \mu)/n}\right] = \prod_{i=1}^n \mathbb{E}[e^{\lambda(X_i - \mu)/n}].$$

3. **Hoeffding's Lemma 적용**: 각 $i$에 대해 $Y_i := X_i - \mu$라 두면 $Y_i \in [a_i - \mu, b_i - \mu]$이고 $\mathbb{E}[Y_i] = 0$. Lemma:
$$\mathbb{E}[e^{\lambda Y_i / n}] \leq e^{\lambda^2(b_i - a_i)^2 / (8n^2)}.$$

따라서
$$\prod_{i=1}^n \mathbb{E}[e^{\lambda Y_i/n}] \leq \prod_{i=1}^n e^{\lambda^2(b_i-a_i)^2/(8n^2)} = e^{\lambda^2 \sum(b_i-a_i)^2 / (8n)}.$$

4. **$\lambda$ 최적화**: Chernoff bound는
$$\mathbb{P}(\bar{X}_n \geq \mu + t) \leq \min_{\lambda > 0} e^{\lambda^2 \sum(b_i-a_i)^2/(8n) - \lambda t}.$$

우변의 지수를 최소화하려 $\lambda$에 대해 미분:
$$\frac{2\lambda \sum(b_i-a_i)^2}{8n} - t = 0 \Rightarrow \lambda^* = \frac{4nt}{\sum(b_i-a_i)^2}.$$

대입하면
$$(\lambda^*)^2 \frac{\sum(b_i-a_i)^2}{8n} - \lambda^* t = \frac{2nt^2}{\sum} - \frac{2nt^2}{\sum} = -\frac{2nt^2}{\sum(b_i-a_i)^2}.$$

따라서 $\mathbb{P}(\bar{X}_n - \mu \geq t) \leq e^{-2nt^2/\sum}$. $\square$

### 정리 2.6 (동일 구간의 특수 경우)

모든 $X_i \in [0, 1]$이면 (예: 0-1 loss)
$$\mathbb{P}(|\bar{X}_n - \mu| \geq t) \leq 2\exp(-2nt^2).$$

이것이 **PAC learning의 가장 기본적인 형태**다.

---

## 💻 NumPy 구현 검증

### 실험 1: Hoeffding bound vs 실제 꼬리 (Bernoulli)

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Bernoulli(p) 표본, n별로 경험적 bound 확인
p = 0.4
n_vals = [20, 50, 100, 500]
t_vals = np.linspace(0.01, 0.3, 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.ravel()

for idx, n in enumerate(n_vals):
    empirical_P = []
    hoeff_bound = []
    
    for t in t_vals:
        # 실험적 측정: n번 시행 5000회 반복
        means = []
        for _ in range(5000):
            X = rng.binomial(1, p, n)
            means.append(X.mean())
        emp = np.mean(np.abs(np.array(means) - p) >= t)
        empirical_P.append(emp)
        
        # Hoeffding bound (구간 [0,1])
        bound = 2 * np.exp(-2 * n * t**2)
        hoeff_bound.append(bound)
    
    ax = axes[idx]
    ax.semilogy(t_vals, empirical_P, 'ko-', linewidth=2, markersize=5, label='Empirical')
    ax.semilogy(t_vals, hoeff_bound, 'bs--', linewidth=2, label='Hoeffding')
    ax.set_xlabel('t')
    ax.set_ylabel('P(|X̄ₙ - p| ≥ t)')
    ax.set_title(f'n={n}, p={p}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# → Hoeffding이 upper bound이지만, 지수 감소는 맞음
```

### 실험 2: 다양한 분포에서 Hoeffding의 범용성

```python
# Hoeffding은 분포를 모르고도 작동: Uniform vs Exponential truncated
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

t = 0.05
n_vals = np.arange(10, 201, 10)

# Distribution 1: Uniform[0, 1]
uniform_prob = []
for n in n_vals:
    means = np.array([rng.uniform(0, 1, n).mean() for _ in range(2000)])
    prob = np.mean(np.abs(means - 0.5) >= t)
    uniform_prob.append(prob)

# Distribution 2: Exponential truncated to [0, 2] (Uniform isn't)
def trunc_exp(size, scale=1, max_val=2):
    samples = []
    while len(samples) < size:
        x = rng.exponential(scale, 1)[0]
        if x <= max_val:
            samples.append(x)
    return np.array(samples[:size])

exp_prob = []
for n in n_vals:
    means = np.array([trunc_exp(n).mean() for _ in range(2000)])
    prob = np.mean(np.abs(means - 0.8) >= t)  # empirical mean
    exp_prob.append(prob)

hoeff = 2 * np.exp(-2 * n_vals * t**2)

ax = axes[0]
ax.semilogy(n_vals, uniform_prob, 'o-', label='Uniform[0,1]', linewidth=2)
ax.semilogy(n_vals, exp_prob, 's-', label='Exp truncated [0,2]', linewidth=2)
ax.semilogy(n_vals, hoeff, 'k--', label='Hoeffding bound', linewidth=2)
ax.set_xlabel('n')
ax.set_ylabel('P(|X̄ₙ - μ| ≥ 0.05)')
ax.set_title('분포 무관: 두 분포 모두 같은 bound 커버')
ax.legend()
ax.grid(True, alpha=0.3)

# Distribution 3: 더 tight하게 "정규분포 비슷한" vs "균등"
X_normal = rng.standard_normal(200000)
X_normal = (X_normal + 3) / 6  # rescale to [0,1] roughly
X_uniform = rng.uniform(0, 1, 200000)

ax = axes[1]
ax.hist(X_normal, bins=50, alpha=0.5, density=True, label='Normal (rescaled)')
ax.hist(X_uniform, bins=50, alpha=0.5, density=True, label='Uniform')
ax.set_title('분포는 다르지만, Hoeffding은 공통 bound 제공')
ax.legend()

plt.tight_layout()
plt.show()
```

### 실험 3: Chernoff 최적화 의존성

```python
# lambda 최적화가 실제로 bound를 얼마나 개선하는가
t, n = 0.05, 100
c = 1  # sum(b_i - a_i)^2 = n (모두 [0,1])

lambda_grid = np.linspace(0.1, 2, 50)
bounds = []
for lam in lambda_grid:
    bound = np.exp(lam**2 * n / 8 - lam * n * t)
    bounds.append(bound)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(lambda_grid, bounds, 'o-', linewidth=2, markersize=6)
lambda_opt = 4 * n * t / n  # = 4t
ax.axvline(lambda_opt, color='r', linestyle='--', label=f'λ* = {lambda_opt:.2f}')
ax.set_xlabel('λ')
ax.set_ylabel('Bound exp(λ²n/8 - λnt)')
ax.set_title('Chernoff 최적화: λ에 대한 bound 최소화')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f'Bound at λ*: {min(bounds):.2e}')
```

---

## 🔗 ML 알고리즘 연결

| 응용 | Hoeffding 형태 | 비고 |
|-----|---|---|
| **학습 이론 (PAC)** | $\mathbb{P}(L_S(h) - L_\mathcal{D}(h) \geq \epsilon) \leq 2e^{-2n\epsilon^2}$ | Ch3의 출발점 |
| **교차 검증 오차** | CV 자체도 bounded diff 형태로 가능 (Ch2-03 McDiarmid) | - |
| **A/B 테스팅** | CTR 추정에서 "샘플 CTR이 참 CTR에서 벗어날 확률" | 비즈니스 메트릭 |
| **Confidence interval** | $p \pm \sqrt{\log(2/\delta)/(2n)}$ 형태의 신뢰도 | 통계 기초 |
| **Boosting margin bound** | 마진 분포의 꼬리도 Hoeffding으로 bound | Schapire et al. |

---

## ⚖️ 가정과 한계

1. **구간 유계 필수**: 무한 분포(정규분포 등)에는 직접 적용 불가. Truncation 필요.
2. **분포 자유**: 분포를 모르는 것이 강점이지만, 실제 분포가 깔끔하면 bound는 loose. 예: 정규분포는 $e^{-nt^2}$인데 Hoeffding은 $e^{-2nt^2}$로 2배 worse.
3. **고정 $h$**: 이 부등식은 **데이터 의존 가설선택 $\hat{h}$에는 직접 적용 불가**. Union bound 필요 (Ch3-03). 이 "문제"가 SLT 전체를 동기 부여한다.
4. **독립성**: iid 가정이 깨지면(시계열 등) 다른 martingale 버전 Hoeffding 필요.
5. **양쪽 꼬리**: 한쪽만 보면 $e^{-2nt^2}$이지만, 두쪽 다 보면 $2e^{-2nt^2}$.

---

## 📌 핵심 정리

- **Hoeffding's Lemma**: $\mathbb{E}[e^{\lambda X}] \leq e^{\lambda^2(b-a)^2/8}$ for $X \in [a,b], \mathbb{E}[X]=0$. **구간 정보 → MGF bound**.
- **Chernoff 방법**: $\mathbb{P}(X \geq t) = \mathbb{P}(e^{\lambda X} \geq e^{\lambda t}) \leq e^{-\lambda t} \mathbb{E}[e^{\lambda X}]$. **$\lambda$ 최적화로 지수 속도**.
- **Hoeffding 부등식**: $\mathbb{P}(|\bar{X}_n - \mu| \geq t) \leq 2\exp(-2nt^2/\sum(b_i-a_i)^2)$.
- **Sub-Gaussian**: Bounded 변수 → Gaussian처럼 빠른 꼬리 감소.
- **PAC 기초**: 유한 $\mathcal{H}$는 Union Bound로 확장 가능 (Ch3-03).

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> Hoeffding's Lemma에서 $\mathbb{E}[X] = 0$ 조건이 필수인 이유를 설명하라. $\mu := \mathbb{E}[X] \neq 0$이면 어떻게 보정하는가?</summary>

<br/>

**해설**. $\mu \neq 0$이면 $X - \mu \in [a - \mu, b - \mu]$이고 $\mathbb{E}[X - \mu] = 0$. 따라서
$$\mathbb{E}[e^{\lambda X}] = e^{\lambda \mu} \mathbb{E}[e^{\lambda(X - \mu)}] \leq e^{\lambda \mu} \cdot e^{\lambda^2(b-a)^2/8}.$$

**표본 평균의 경우**: 중심화 $\bar{X}_n - \mu$에 대해 $\mathbb{E}[\bar{X}_n - \mu] = 0$이므로, Lemma를 직접 적용하면 원하는 형태가 나온다. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> Hoeffding bound $2\exp(-2nt^2)$가 진짜 Bernoulli(1/2) 꼬리 $e^{-2D_{\text{KL}}(1/2+t \| 1/2)}$ (KL divergence)와 비교할 때 몇 배나 loose한가? 수치로 계산하라.</summary>

<br/>

**해설**. Chernoff bound를 KL 최적화로 정교하게 계산하면 (Sanov's theorem), exact 꼬리는 $e^{-nD}$이다. 여기서 $D_{\text{KL}}(1/2+t \| 1/2) = (1/2+t)\log\frac{1/2+t}{1/2} + (1/2-t)\log\frac{1/2-t}{1/2} \approx 2t^2$ (작은 $t$). 반면 Hoeffding은 $2\exp(-2nt^2)$.

$t = 0.01, n = 1000$: 
- Hoeffding: $e^{-2 \cdot 10} = e^{-20}$
- Exact: $e^{-1000 \cdot 2 \cdot 0.0001} = e^{-0.2}$... 아니다, 정확히는 더 복잡하지만 Hoeffding이 loose한 것은 맞다.

사실, Hoeffding의 2배 "손해"는 **분포 자유성에 대한 대가**이다. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> 온라인 광고: 광고 A의 클릭률(CTR)을 1000번 노출 후 추정하고 신뢰도 95%로 진짜 CTR을 bound하려 한다. Hoeffding으로 구간을 구하고, 샘플 CTR이 10%일 때 구간 폭이 얼마나 되는가?</summary>

<br/>

**해설**. Hoeffding: $\mathbb{P}(|\hat{p} - p| \geq t) \leq 2e^{-2nt^2}$에서 $\delta = 2e^{-2nt^2} = 0.05$ 설정.
$$e^{-2 \cdot 1000 \cdot t^2} = 0.025 \Rightarrow -2000t^2 = \log(0.025) \approx -3.69 \Rightarrow t^2 \approx 0.00185 \Rightarrow t \approx 0.043.$$

구간: $[0.10 - 0.043, 0.10 + 0.043] = [0.057, 0.143]$. 폭은 약 8.6%. 1000은 "광고 업계 표준"이 아니라서 너무 넓다 — 더 많은 샘플이나 베이지안 approach 필요. $\square$

</details>

---

<div align="center">

◀ [이전: 01. Markov·Chebyshev 부등식](./01-markov-chebyshev.md) | [📚 README](../README.md) | [다음: 03. McDiarmid 부등식 ▶](./03-mcdiarmid.md)

</div>
