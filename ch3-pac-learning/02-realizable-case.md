# 02. Realizable Case의 학습 가능성

## 🎯 핵심 질문

- **Realizable 가정** ($\exists h^* \in \mathcal{H}, L_\mathcal{D}(h^*) = 0$) 하에서 ERM이 왜 **무조건 PAC-learnable**인가?
- **정리**: 유한 $|\mathcal{H}| < \infty$에서 ERM은 샘플 $m(\epsilon, \delta) = \lceil \log(|\mathcal{H}|/\delta) / \epsilon \rceil$개로 PAC learn한다 — **왜 $1/\epsilon$ (not $1/\epsilon^2$)**?
- 핵심 증명 기술은 무엇인가? **"Bad event"** $B_h = \{L_\mathcal{D}(h) > \epsilon\} \cap \{L_S(h) = 0\}$의 확률을 **Union bound**로 묶는 것.
- 이 fast rate ($1/\epsilon$)는 **어디서 오는가**? Realizable case에서는 $L_S(h) = 0$이면 충분히 좋은 이유.
- **Agnostic case** (Ch3-03)로 넘어갈 때 왜 $\epsilon \to \epsilon^2$로 악화되는가?

---

## 🔍 왜 이 정리가 현대 ML에서 중요한가

"우리의 모델이 정확히 맞을 수 있다면 (realizable), ERM은 **확률 보장을 준다**"는 이 정리는 간단하지만 강력하다. 왜냐하면:

1. **모든 유한 가설공간에 적용**: $h$가 선형 분류기, 깊이 제약 결정 트리, 또는 다른 무엇이든 간에, 크기만 유한하면 **같은 형태의 bound**가 성립.

2. **Fast rate ($1/\epsilon$)**: Agnostic case의 $1/\epsilon^2$보다 **4배 빠른** convergence. 실전에서 "우리 모델이 완벽할 수 있다"고 믿는다면 훨씬 적은 데이터로 충분.

3. **증명의 명확성**: 이 장의 증명은 **Hoeffding + Union bound**만으로 끝난다. 후속 단계들(VC, Rademacher)의 기초가 되는 "prototype" 증명.

---

## 📐 수학적 선행 조건

- Ch1-01, Ch1-02: 위험 $L_\mathcal{D}, L_S$ 정의
- Ch2-02: **Hoeffding 부등식** $\mathbb{P}(|\bar{X} - \mu| \geq t) \leq 2e^{-2nt^2}$
- Ch3-01: PAC learnability 정의, sample complexity
- 기초: Union bound (유한 개 사건), 지수 확률의 성질

---

## 📖 직관적 이해

### Realizable이 왜 쉬운가?

Realizable 가정 ($\exists h^* \in \mathcal{H}, L_\mathcal{D}(h^*) = 0$)은 "세상의 진짜 데이터 생성 과정이 $\mathcal{H}$의 **어떤 가설과 정확히 일치한다**"는 뜻이다. 따라서:

- 나쁜 가설 $h$ (즉, $L_\mathcal{D}(h) > \epsilon$)가 우연히 샘플에서 완벽하게 맞을 확률은 **매우 낮다**: $(1-\epsilon)^n \approx e^{-n\epsilon}$.
- 우리는 $h^*$를 찾아야 하는데, ERM($\min_h L_S(h)$)은 **$L_S = 0$을 달성하는 가설을 고르므로**, 높은 확률로 $h^*$나 $h^*$와 비슷한 좋은 가설을 고른다.

### "Bad event"는 무엇인가?

**정의**: 가설 $h$가 "나쁜 사건"을 일으킨다 = $L_\mathcal{D}(h) > \epsilon$인데도 $L_S(h) = 0$이다. 즉:
- 진짜로는 오류율이 높음
- 하지만 우리 샘플에서는 운 좋게 완벽 — **misleading**

$\log(h/\delta) / \epsilon$개 샘플이 있으면, **이런 misleading이 일어날 전체 확률이 $\delta$ 이하**가 된다.

---

## ✏️ 엄밀한 정의

### 정의 3.2.1 (Realizable PAC Learning의 구체화)

Realizable 가정 하에서 ERM이 PAC learn한다는 것:

$$\mathbb{P}_{S \sim \mathcal{D}^n} \left[ L_\mathcal{D}\left(\arg\min_{h \in \mathcal{H}} L_S(h)\right) \leq \epsilon \right] \geq 1 - \delta.$$

즉, 높은 확률로 **ERM이 고른 가설의 진짜 위험이 $\epsilon$ 이하**.

---

## 🔬 정리와 증명

### 정리 3.2.1 (Realizable Case의 PAC Learnability)

**가정**: 
- $|\mathcal{H}| < \infty$
- Realizable: $\exists h^* \in \mathcal{H}$ s.t. $L_\mathcal{D}(h^*) = 0$
- iid 샘플 $S \sim \mathcal{D}^n$
- $\ell \in [0, 1]$ (0-1 loss 등)

**결론**: ERM 알고리즘은 PAC learn하며, sample complexity는

$$m(\epsilon, \delta) = \left\lceil \frac{\log(|\mathcal{H}|/\delta)}{\epsilon} \right\rceil.$$

**완전 증명**:

**Step 1. "Bad event" 정의**

각 가설 $h \in \mathcal{H}$에 대해, 다음 "나쁜 사건"을 정의하자:

$$B_h := \left\{L_\mathcal{D}(h) > \epsilon \text{ and } L_S(h) = 0\right\}.$$

이것은 "$h$가 진짜로는 오류율 $\epsilon$을 초과하지만, 샘플에서는 완벽하게 맞춘다"는 뜻이다.

**Step 2. Bad event의 확률 bound (하나의 $h$)**

고정된 $h \in \mathcal{H}$에 대해, realizable 가정이 있을 때:

$$\mathbb{P}[L_S(h) = 0 | L_\mathcal{D}(h) > \epsilon] \leq (1-\epsilon)^n.$$

왜냐하면 $h \neq h^*$이고 ($L_\mathcal{D}(h) > 0$이므로), $L_S(h) = 0$이려면 **$n$개 샘플 모두가 우연히 $h$를 선호**해야 한다. 각 샘플에서 $h$가 오분류할 확률이 $\geq \epsilon$ (왜냐하면 $L_\mathcal{D}(h) > \epsilon$)이므로, **모두 맞을 확률**은 $\leq (1-\epsilon)^n$.

더 정확히는 Hoeffding (Ch2-02):

$$\mathbb{P}[L_S(h) = 0] = \mathbb{P}\left[L_S(h) \leq 0\right] = \mathbb{P}\left[L_\mathcal{D}(h) - (L_\mathcal{D}(h) - L_S(h)) \leq 0\right].$$

$L_\mathcal{D}(h) > \epsilon$이면

$$\mathbb{P}\left[L_S(h) = 0\right] \leq \mathbb{P}\left[L_\mathcal{D}(h) - L_S(h) > \epsilon\right] \leq e^{-2n\epsilon^2}$$

by Hoeffding. 하지만 더 직접적으로는, 각 샘플 $i$마다 $\ell(h(x_i), y_i) \sim \text{Bernoulli}(p)$ where $p = L_\mathcal{D}(h) > \epsilon$이고, $L_S(h) = (1/n) \sum \ell_i$이다. $L_S(h) = 0$이려면 모든 $\ell_i = 0$, 즉 확률 $(1-p)^n \leq (1-\epsilon)^n$.

표준 형태로:

$$\mathbb{P}[B_h] = \mathbb{P}[L_S(h) = 0 \text{ and } L_\mathcal{D}(h) > \epsilon] \leq (1-\epsilon)^n \leq e^{-n\epsilon}.$$

(마지막 부등식은 $1-x \leq e^{-x}$에서 $x = \epsilon$.)

**Step 3. Union bound over all hypotheses**

$$\mathbb{P}\left[\exists h \in \mathcal{H}: B_h\right] \leq \sum_{h \in \mathcal{H}} \mathbb{P}[B_h] \leq |\mathcal{H}| \cdot e^{-n\epsilon}.$$

**Step 4. Sample complexity 유도**

$\delta$를 실패 확률의 상한으로 두자:

$$|\mathcal{H}| \cdot e^{-n\epsilon} \leq \delta.$$

양변에 $\log$:

$$\log|\mathcal{H}| - n\epsilon \leq \log \delta \Rightarrow n\epsilon \geq \log|\mathcal{H}| - \log \delta = \log(|\mathcal{H}|/\delta).$$

따라서

$$n \geq \frac{\log(|\mathcal{H}|/\delta)}{\epsilon} \Rightarrow m(\epsilon, \delta) = \left\lceil \frac{\log(|\mathcal{H}|/\delta)}{\epsilon} \right\rceil. \qquad \square$$

**Step 5. ERM의 성공 보장**

이 $n$일 때, 확률 $\geq 1-\delta$로 **bad event가 일어나지 않는다**. 즉, $L_\mathcal{D}(h) > \epsilon$인 가설 중 $L_S(h) = 0$인 것이 없다.

따라서 ERM이 선택한 가설 $\hat{h} = \arg\min_h L_S(h)$는:
- $L_S(\hat{h}) = 0$ (realizable이므로 달성 가능)
- Bad event 미발생 ⟹ $L_\mathcal{D}(\hat{h}) \leq \epsilon$ (정의상).

### 정리 3.2.2 (Sample complexity의 의존성)

Sample complexity $m(\epsilon, \delta) = \lceil \log(h/\delta) / \epsilon \rceil$는:

- **$1/\epsilon$ 의존** (linear, not quadratic): Realizable의 특징. $\epsilon$ 2배 정밀도를 원하면 샘플 2배.
- **$\log h$ 의존** (logarithmic in hypothesis space size): Union bound의 대가. $h$가 100배 커도 log(100) ≈ 5배만 증가.
- **$\log(1/\delta)$ 의존** (logarithmic in confidence): 실패 확률 10배 감소하려면 log(10) ≈ 3배만 증가.

---

## 💻 NumPy 구현 검증

### 실험 1: Realizable PAC bound와 실제 sample complexity의 비교

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# 진짜 함수: f*(x) = (x_1 > 0) ∧ (x_2 > 0) — 2D 축정렬 사각형의 한 사분면
def ground_truth(X):
    """Returns 1 if both coords positive, 0 otherwise"""
    return ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)

def sample_D(n):
    X = rng.uniform(-1, 1, (n, 2))
    Y = ground_truth(X)
    return X, Y

# 가설공간: 축정렬 직사각형들
# h_{(a1, a2, b1, b2)}(x) = (a1 < x_1 < b1) ∧ (a2 < x_2 < b2)
# 이산화: 각 축마다 K 개 경계값, 총 K^4개 가설
K = 20
ax_vals = np.linspace(-1, 1, K)
h_size = K ** 4
print(f'Hypothesis space size: |H| = {h_size}')

def generate_hypotheses():
    """Generate all K^4 hypotheses"""
    rects = []
    for a1 in ax_vals:
        for b1 in ax_vals[ax_vals > a1]:  # b1 > a1
            for a2 in ax_vals:
                for b2 in ax_vals[ax_vals > a2]:
                    rects.append((a1, b1, a2, b2))
    return rects[:h_size]  # truncate if needed

def classify_rect(rect, X):
    a1, b1, a2, b2 = rect
    return ((X[:, 0] >= a1) & (X[:, 0] <= b1) & 
            (X[:, 1] >= a2) & (X[:, 1] <= b2)).astype(int)

def emp_loss(rect, X, Y):
    pred = classify_rect(rect, X)
    return (pred != Y).sum() / len(Y)

def erm(X, Y, hypotheses):
    """Find h minimizing L_S"""
    losses = [emp_loss(h, X, Y) for h in hypotheses]
    return hypotheses[np.argmin(losses)]

# Realizable PAC: 정리 3.2.1
epsilon_vals = [0.1, 0.15, 0.2, 0.25, 0.3]
delta = 0.05

hypotheses = generate_hypotheses()

fig, ax = plt.subplots(figsize=(10, 5))

theory_ns = [np.ceil(np.log(h_size/delta) / eps) for eps in epsilon_vals]

for eps in epsilon_vals:
    success_rates = []
    test_ns = np.linspace(10, 200, 12).astype(int)
    
    for n in test_ns:
        n_trials = 100
        successes = 0
        
        for _ in range(n_trials):
            X_train, Y_train = sample_D(n)
            # ERM
            h_hat = erm(X_train, Y_train, hypotheses)
            
            # Test on large set
            X_test, Y_test = sample_D(5000)
            test_loss = emp_loss(h_hat, X_test, Y_test)
            
            # Success = test loss ≤ ε
            if test_loss <= eps:
                successes += 1
        
        success_rates.append(successes / n_trials)
    
    ax.plot(test_ns, success_rates, 'o-', label=f'ε={eps}, theory m={np.ceil(np.log(h_size/delta) / eps):.0f}', linewidth=2)

ax.axhline(1-delta, color='k', linestyle=':', linewidth=1, label=f'1-δ={1-delta}')
ax.set_xlabel('Sample size n'); ax.set_ylabel('Success rate')
ax.set_title(f'Realizable PAC (K={K}, |H|={h_size}, δ={delta}): ERM성공률 vs 이론')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# → 각 epsilon에서 이론적 m 근처부터 성공률이 1-δ에 도달함을 확인.
```

### 실험 2: 부등식의 각 항이 어떻게 작용하는가?

```python
# m = log(|H|/δ) / ε의 각 성분
h_size = 10000
epsilon_vals = [0.05, 0.1, 0.2, 0.3]
delta_vals = [0.001, 0.01, 0.05, 0.1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ε에 대한 의존성
theory_ms = [np.ceil(np.log(h_size/0.05) / eps) for eps in epsilon_vals]
axes[0].semilogy(epsilon_vals, theory_ms, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('ε'); axes[0].set_ylabel('m(ε)')
axes[0].set_title(f'Sample complexity vs ε (|H|={h_size}, δ=0.05)\nm = log(|H|/δ)/ε')
axes[0].grid(alpha=0.3)

# δ에 대한 의존성 (log scale)
theory_ms_delta = [np.ceil(np.log(h_size/d) / 0.1) for d in delta_vals]
axes[1].semilogy(delta_vals, theory_ms_delta, 's-', linewidth=2, markersize=8)
axes[1].set_xlabel('δ'); axes[1].set_ylabel('m(δ)')
axes[1].set_title(f'Sample complexity vs δ (|H|={h_size}, ε=0.1)\nm = log(|H|/δ)/ε')
axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()

print(f'Examples:')
for eps in epsilon_vals:
    m = np.log(h_size/0.05) / eps
    print(f'  ε={eps}: m≈{m:.0f}')
```

---

## 🔗 ML 알고리즘 연결

Realizable case는 다음 실제 시나리오에 대응:

| 상황 | 모델 | Realizable인가? |
|------|------|----------------|
| **Perfect labeled data** | Linear separable 분류 | ✓ (충분히 작은 margin이면) |
| **Synthetic data** | Rule-based generation | ✓ (정확한 규칙이 있으면) |
| **Noise-free setting** | 결정 트리 (충분히 깊으면) | ✓ (shatter 가능) |
| **Real-world noisy data** | 대부분의 실전 ML | ✗ (Agnostic으로 넘어가야 함) |

---

## ⚖️ 가정과 한계

1. **Realizable의 비현실성**: 현실 데이터는 **항상 노이즈·모순·모델링 오류**를 포함. 완벽한 가설은 드물다.

2. **개별 가설의 확률이 독립이 아님**: Union bound는 최악의 경우를 본다. 실제로는 많은 가설이 similar하면 (data-dependent redundancy), 개별 bound의 합보다 훨씬 타이트.

3. **느슨한 bound**: Hoeffding은 **분포 자유(distribution-free)** bound이므로 실제 데이터보다 보수적. 특정 분포(예: low margin)는 훨씬 작은 sample로 충분할 수 있다.

4. **$\log h$의 지수적 대가**: $h$가 차원 $d$에 지수적이면 (예: 신경망), 이론적 bound는 쓸모없어진다. 무한 $\mathcal{H}$ 이론(Ch4-05의 VC bound)이 필요.

5. **Computational complexity**: PAC learning의 정보 이론적 보장이 **다항시간 알고리즘**을 함의하지 않는다. ERM 자체가 NP-hard일 수 있다.

---

## 📌 핵심 정리

- **Realizable PAC Learning**: $\exists h^* \in \mathcal{H}, L_\mathcal{D}(h^*) = 0$ 가정 하에 ERM은 **무조건 PAC learnable**.
- **Sample complexity** (유한 $\mathcal{H}$): $m = \lceil \log(h/\delta) / \epsilon \rceil$ — **$1/\epsilon$ fast rate**.
- **증명의 핵심**: Bad event $B_h = \{L_\mathcal{D}(h) > \epsilon, L_S(h) = 0\}$의 확률을 $(1-\epsilon)^n \leq e^{-n\epsilon}$로 bound, Union bound로 $h \cdot e^{-n\epsilon} \leq \delta$.
- **왜 $1/\epsilon$**: Realizable에서 $L_S(h) = 0$은 **충분히 강한 신호** — 나머지는 Hoeffding 하나로 끝남.
- **다음**: Agnostic case (Ch3-03)에서 이것이 왜 $1/\epsilon^2$로 악화되는지 이해.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> 정리 3.2.1의 증명에서 "$(1-\epsilon)^n \leq e^{-n\epsilon}$" 부등식이 왜 성립하는가? 이것은 어떤 기하학적/확률론적 의미를 가지는가?</summary>

<br/>

**해설**. $1-x \leq e^{-x}$ for all $x \in [0, 1]$이기 때문이다 (Taylor 전개: $e^{-x} = 1 - x + x^2/2 - \ldots \geq 1-x$).

기하학적으로는: $y = 1-x$는 선형이고, $y = e^{-x}$는 위로 볼록한 곡선. 두 함수가 $(0, 1)$과 $(1, e^{-1})$에서 만나고, 전체 $[0, 1]$에서 직선이 곡선 아래.

확률론적으로는: iid Bernoulli 시행 $n$번에서 모두 성공할 확률이 $(1-p)^n$인데, 이것을 **Chernoff bound** $(1-p)^n \leq e^{-np}$로 exponentially 감소시킨다는 뜻. 정확한 확률보다 bound가 깔끔하고 이 bound가 **매우 타이트**하다. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> Realizable case의 $m = O(\log(h/\delta)/\epsilon)$과 agnostic case의 $m = O([\log h + \log(1/\delta)]/\epsilon^2)$를 비교할 때, "언제 realizable이 유리한가"?</summary>

<br/>

**해설**. 두 식을 비교: realizable이 나으려면

$$\frac{\log(h/\delta)}{\epsilon} < \frac{\log h + \log(1/\delta)}{\epsilon^2}$$

정리하면

$$\epsilon \log(h/\delta) < \log h + \log(1/\delta).$$

$h, \delta$를 고정하고 $\epsilon$에 대해 보면:
- **$\epsilon$ 크면** (비정확 허용, e.g. $\epsilon = 0.3$): 왼쪽이 작아지므로 realizable이 우월.
- **$\epsilon$ 작으면** (높은 정확도, e.g. $\epsilon = 0.01$): 오른쪽이 우월 (분자가 절대값이므로).

실전 의미: "근처 답"으로 충분하면 realizable의 빠른 수렴이 huge 이득. 하지만 "매우 정확한 답"을 원하면 agnostic의 추가 요구사항(노이즈 정규화)이 피할 수 없다. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> 신경망이 훈련 데이터를 "완벽하게 기억"(memorization, $L_S = 0$)할 수 있다면, realizable 가정이 성립하는가? 이 경우 정리 3.2.1을 신경망에 적용할 수 있을까?</summary>

<br/>

**해설**. 신경망이 $L_S = 0$을 달성하려면 충분히 깊거나 넓어야 한다 (Ch1-03의 "문제 3" — Zhang et al. 2017). 이 경우 **정보 이론적으로**는 realizable PAC를 적용할 수 있다.

하지만 문제는:
1. **$|\mathcal{H}|$가 거대** (weight space, $W$ parameters): $|H| \approx 2^{poly(W)}$이므로 $\log(h/\delta) \approx poly(W)$. Sample complexity $m \approx poly(W) / \epsilon$ — 실전 $n$ 샘플보다 훨씬 클 수 있음.
2. **Computational complexity**: ERM (정확한 $L_S$ 최소화)은 non-convex, SGD는 local optimum에 빠질 수 있음.
3. **실전과 맞지 않음**: 신경망의 일반화는 weight magnitude(Rademacher, Ch5), margin(SVM bound), implicit regularization(stability, Ch6) 등 **다른 메커니즘**으로 설명되는 게 더 정확.

결론: realizable PAC는 원리상 적용 가능하지만, 신경망의 **실제 행동**을 설명하려면 Ch5(Rademacher)나 Ch6(Stability)의 도구가 필요. $\square$

</details>

---

<div align="center">

◀ [이전: 01. PAC Learnability의 정의](./01-pac-definition.md) | [📚 README](../README.md) | [다음: 03. Agnostic PAC Learning ▶](./03-agnostic-pac.md)

</div>
