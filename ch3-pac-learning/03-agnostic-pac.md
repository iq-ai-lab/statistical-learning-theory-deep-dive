# 03. Agnostic PAC Learning

## 🎯 핵심 질문

- **Agnostic PAC**는 realizable 가정을 버렸을 때 무엇이 달라지는가? 최적 가설이 여전히 오차를 가질 수 있으면?
- **정리**: 유한 $|\mathcal{H}|$에서 agnostic PAC의 sample complexity는 $m(\epsilon, \delta) = \lceil 2\log(2|\mathcal{H}|/\delta) / \epsilon^2 \rceil$ — **왜 $\epsilon^2$로 제곱되는가?**
- **양쪽 꼬리(two-sided tail)** vs **한쪽 꼬리(one-sided tail)**의 차이는? Realizable에서는 "크면 안 좋다" 한 방향만 봤다면, agnostic에서는?
- **Union bound의 대가**: Agnostic case에서도 여전히 $|\mathcal{H}|$에 대한 의존이 남는가?
- **실전 의미**: 대부분의 현실 문제가 agnostic인데, 이 $1/\epsilon^2$ rate는 어디서 개선 가능한가?

---

## 🔍 왜 Agnostic이 "현실적"인가

Realizable case는 "우리 모델이 **정확히** 맞을 수 있다"고 가정했다. 하지만 현실은:
- 데이터에 **노이즈**가 있다
- 우리 모델이 **incomplete**하다
- 라벨 오류·측정 오류가 있다

따라서 $\inf_{h \in \mathcal{H}} L_\mathcal{D}(h) =: L_\mathcal{H}^* > 0$이 일반적. Agnostic PAC는 **이 기본적 제약을 받아들이고**, "$L_\mathcal{D}(\hat{h})$이 $L_\mathcal{H}^* + \epsilon$을 이하로 남는다"를 보장한다. 이것이 **현실적 학습의 첫 정식화**.

---

## 📐 수학적 선행 조건

- Ch1-01, Ch1-02: 위험 $L_\mathcal{D}, L_S$, Bayes error $L^*$
- Ch2-02: **Hoeffding 부등식** $\mathbb{P}(|\bar{X} - \mu| \geq t) \leq 2e^{-2nt^2}$
- Ch3-01, Ch3-02: PAC learnability 정의, realizable case
- 기초: 양쪽 꼬리 확률, 절댓값 부등식, Union bound

---

## 📖 직관적 이해

### Realizable → Agnostic: 무엇이 달라지는가?

| 측면 | Realizable | Agnostic |
|------|-----------|---------|
| 최적 가설의 오차 | $L_\mathcal{D}(h^*) = 0$ | $L_\mathcal{D}(h^*) = L_\mathcal{H}^* \geq 0$ |
| ERM 해석 | "$L_S = 0$이면 좋음" | "$L_S$가 작으면 좋음 (정확하게 얼마나?)" |
| 꼬리 경계 | $(1-\epsilon)^n \leq e^{-n\epsilon}$ (한쪽) | $e^{-2n\epsilon^2}$ (양쪽) |
| Sample complexity | $O(\log(h/\delta) / \epsilon)$ | $O([\log(2h/\delta)] / \epsilon^2)$ |
| 율(rate) | Fast rate ($1/\epsilon$) | Slower ($1/\epsilon^2$) |

### 양쪽 꼬리가 왜 필요한가?

Realizable에서는 "나쁜 가설 $h$가 우연히 $L_S(h) = 0$이 되는" **한 가지 나쁜 일**만 피하면 됐다.

Agnostic에서는:
- **과대 평가**: $L_\mathcal{D}(h)$가 실제로는 크지만, $L_S(h)$는 작게 뽑혀서 ERM이 나쁜 $h$를 고를 수 있다.
- **과소 평가**: $L_\mathcal{D}(h)$가 실제로는 작지만, $L_S(h)$는 크게 뽑혀서 좋은 $h$를 버릴 수 있다.

즉, **두 방향 모두** 일반화 gap $|L_\mathcal{D}(h) - L_S(h)|$을 통제해야 한다. 이것이 **양쪽 꼬리 Hoeffding**을 만든다.

### 제곱이 어디서 오는가?

Realizable: $(1-\epsilon)^n$ bound는 **확률 자체** (한 방향). 지수는 $n\epsilon$이므로 $\log 1/P \sim n\epsilon$.

Agnostic: $e^{-2n\epsilon^2}$ bound는 **양쪽 꼬리의 합** (Chernoff). 지수는 $n\epsilon^2$이므로 $\log 1/P \sim n\epsilon^2$. 

역으로 풀면:
- Realizable: $P \leq \delta$ ⟹ $n \geq \log(1/\delta) / \epsilon$
- Agnostic: $P \leq \delta$ ⟹ $n \geq \log(1/\delta) / \epsilon^2$

---

## ✏️ 엄밀한 정의

### 정의 3.3.1 (Agnostic PAC Learning)

학습자 $A$가 $\mathcal{H}$를 **agnostic PAC learn**한다:

$$\forall \epsilon, \delta \in (0,1), \quad n \geq m(\epsilon, \delta) \Rightarrow \mathbb{P}_{S \sim \mathcal{D}^n}\left[L_\mathcal{D}(A(S)) \leq \inf_{h \in \mathcal{H}} L_\mathcal{D}(h) + \epsilon\right] \geq 1-\delta.$$

즉, $\mathcal{H}$ 내 최선의 가설의 오차 + $\epsilon$ 이상을 넘지 않는다는 보장.

---

## 🔬 정리와 증명

### 정리 3.3.1 (Agnostic PAC Learning — Finite $\mathcal{H}$)

**가정**:
- $|\mathcal{H}| < \infty$
- iid 샘플 $S \sim \mathcal{D}^n$
- $\ell \in [0, 1]$ (0-1 loss 등)
- Agnostic (no realizability assumption)

**결론**: ERM (혹은 정규화된 ERM)은 agnostic PAC learnable하며

$$m(\epsilon, \delta) = \left\lceil \frac{2\log(2|\mathcal{H}|/\delta)}{\epsilon^2} \right\rceil.$$

**완전 증명**:

**Step 1. Generalization gap의 정의와 목표**

고정된 $h \in \mathcal{H}$에 대해 **generalization gap**:

$$\text{gap}_h := |L_\mathcal{D}(h) - L_S(h)|.$$

우리의 목표: $\sup_h |\text{gap}_h| \leq \epsilon/2$를 높은 확률로 보이기. (factor 1/2는 나중에 보상)

**Step 2. 개별 가설에 대한 Hoeffding (양쪽 꼬리)**

각 고정된 $h$에 대해, Hoeffding 부등식 (Ch2-02)의 양쪽 꼬리:

$$\mathbb{P}[|L_S(h) - L_\mathcal{D}(h)| \geq t] \leq 2e^{-2nt^2}.$$

$t = \epsilon/2$로 놓으면

$$\mathbb{P}\left[|L_S(h) - L_\mathcal{D}(h)| \geq \frac{\epsilon}{2}\right] \leq 2e^{-2n(\epsilon/2)^2} = 2e^{-n\epsilon^2/2}.$$

**Step 3. Union bound over $\mathcal{H}$**

모든 가설에 대한 event의 합:

$$\mathbb{P}\left[\exists h \in \mathcal{H}: |L_S(h) - L_\mathcal{D}(h)| \geq \frac{\epsilon}{2}\right] \leq \sum_{h \in \mathcal{H}} 2e^{-n\epsilon^2/2} = 2|\mathcal{H}| e^{-n\epsilon^2/2}.$$

**Step 4. Sample complexity 유도**

성공 확률을 $1-\delta$ 이상으로 하려면

$$2|\mathcal{H}| e^{-n\epsilon^2/2} \leq \delta.$$

양변에 log:

$$\log 2 + \log |\mathcal{H}| - \frac{n\epsilon^2}{2} \leq \log \delta.$$

정리하면

$$\frac{n\epsilon^2}{2} \geq \log 2 + \log |\mathcal{H}| - \log \delta = \log(2|\mathcal{H}|/\delta).$$

따라서

$$n \geq \frac{2\log(2|\mathcal{H}|/\delta)}{\epsilon^2}.$$

**Step 5. ERM의 성공 보장**

이 $n$일 때, 확률 $\geq 1-\delta$로 **모든** $h$에 대해 $|L_S(h) - L_\mathcal{D}(h)| \leq \epsilon/2$.

ERM이 선택한 가설 $\hat{h} = \arg\min_h L_S(h)$에 대해:

$$L_\mathcal{D}(\hat{h}) \leq L_S(\hat{h}) + \frac{\epsilon}{2} \leq \min_h L_S(h) + \frac{\epsilon}{2}.$$

한편, $h^*_\mathcal{H} := \arg\min_h L_\mathcal{D}(h)$에 대해

$$\min_h L_S(h) \leq L_S(h^*_\mathcal{H}) \leq L_\mathcal{D}(h^*_\mathcal{H}) + \frac{\epsilon}{2}.$$

결합하면

$$L_\mathcal{D}(\hat{h}) \leq L_\mathcal{D}(h^*_\mathcal{H}) + \frac{\epsilon}{2} + \frac{\epsilon}{2} = \inf_h L_\mathcal{D}(h) + \epsilon. \qquad \square$$

### 정리 3.3.2 (Realizable과의 비교)

| 조건 | Realizable ($L_\mathcal{H}^* = 0$) | Agnostic ($L_\mathcal{H}^* \geq 0$) |
|------|-----------|----------|
| $m(\epsilon, \delta)$ | $\log(h/\delta) / \epsilon$ | $2\log(2h/\delta) / \epsilon^2$ |
| Rate | $1/\epsilon$ | $1/\epsilon^2$ |
| 주된 한계 | $\log h$ term | $1/\epsilon^2$ term |

$\epsilon^2$의 출현으로 **정확도를 두 배 높이려면 4배의 샘플**이 필요하다.

---

## 💻 NumPy 구현 검증

### 실험 1: Agnostic PAC bound와 실제 sample complexity 비교

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Agnostic 분포: X ~ U[-1, 1], Y = sign(X) with 20% noise
noise_rate = 0.20

def sample_D(n):
    X = rng.uniform(-1, 1, n)
    Y = np.sign(X).astype(int)
    # Add label noise
    flip = rng.random(n) < noise_rate
    Y[flip] *= -1
    return X, Y

# Hypothesis space: threshold classifiers
K = 200
theta_grid = np.linspace(-1, 1, K)
h_size = K
print(f'|H| = {h_size}, noise rate = {noise_rate}')

def classifier(theta):
    return lambda x: np.sign(x - theta).astype(int)

def emp_loss_01(theta, X, Y):
    pred = classifier(theta)(X)
    return (pred != Y).sum() / len(Y)

def true_loss_01(theta):
    """True misclassification rate for sign(X-theta) with noise"""
    # Assuming perfect signal + noise
    # P(Y ≠ sign(X-theta)) = P(noise) + P(no noise, wrong) = noise_rate (approx)
    return noise_rate

def erm(X, Y):
    """ERM: find theta minimizing L_S"""
    losses = [emp_loss_01(t, X, Y) for t in theta_grid]
    return theta_grid[np.argmin(losses)]

# Experiment
epsilon_vals = [0.05, 0.1, 0.15, 0.2]
delta = 0.05

fig, ax = plt.subplots(figsize=(10, 5))

for eps in epsilon_vals:
    success_rates = []
    test_ns = np.linspace(50, 500, 15).astype(int)
    
    # Theoretical sample complexity
    theory_n = np.ceil(2 * np.log(2*h_size/delta) / eps**2)
    print(f'ε={eps}: theoretical m ≥ {theory_n:.0f}')
    
    for n in test_ns:
        n_trials = 100
        successes = 0
        
        for _ in range(n_trials):
            X_train, Y_train = sample_D(n)
            theta_hat = erm(X_train, Y_train)
            
            # Test
            X_test, Y_test = sample_D(5000)
            test_loss = emp_loss_01(theta_hat, X_test, Y_test)
            
            # Success: test loss ≤ L_H* + ε (where L_H* ≈ noise_rate)
            if test_loss <= noise_rate + eps:
                successes += 1
        
        success_rates.append(successes / n_trials)
    
    ax.plot(test_ns, success_rates, 'o-', 
            label=f'ε={eps}, m≥{theory_n:.0f}', linewidth=2)

ax.axhline(1-delta, color='k', linestyle=':', linewidth=1, label=f'1-δ={1-delta}')
ax.set_xlabel('Sample size n'); ax.set_ylabel('Success rate')
ax.set_title(f'Agnostic PAC (|H|={h_size}, noise={noise_rate}, δ={delta})')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# → Realizable과 달리 noise 때문에 성공하려면 더 큰 n 필요, 그리고 ε² 의존 확인
```

### 실험 2: Realizable vs Agnostic의 Sample Complexity 비교

```python
h_size = 1000
epsilon_vals = [0.05, 0.1, 0.15, 0.2, 0.25]
delta = 0.05

# Sample complexity 계산
realizable_m = [np.log(h_size/delta) / eps for eps in epsilon_vals]
agnostic_m   = [2*np.log(2*h_size/delta) / (eps**2) for eps in epsilon_vals]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 절대값 비교
ax1.semilogy(epsilon_vals, realizable_m, 'o-', label='Realizable: m ∝ 1/ε', linewidth=2)
ax1.semilogy(epsilon_vals, agnostic_m, 's-', label='Agnostic: m ∝ 1/ε²', linewidth=2)
ax1.set_xlabel('ε'); ax1.set_ylabel('m(ε)')
ax1.set_title(f'Sample Complexity Comparison (|H|={h_size}, δ={delta})')
ax1.legend(); ax1.grid(alpha=0.3)

# 비율 비교: agnostic / realizable
ratio = np.array(agnostic_m) / np.array(realizable_m)
ax2.semilogy(epsilon_vals, ratio, 'go-', linewidth=2)
ax2.set_xlabel('ε'); ax2.set_ylabel('m_agnostic / m_realizable')
ax2.set_title('Sample Complexity Overhead of Agnosticism')
ax2.grid(alpha=0.3)

plt.tight_layout(); plt.show()

print(f'\nExamples:')
for eps in epsilon_vals:
    m_real = np.log(h_size/delta) / eps
    m_agn  = 2*np.log(2*h_size/delta) / (eps**2)
    print(f'  ε={eps}: realizable={m_real:.0f}, agnostic={m_agn:.0f}, ratio={m_agn/m_real:.1f}x')
```

---

## 🔗 ML 알고리즘 연결

| 알고리즘 | Assumption | 적용 형태 |
|---------|-----------|---------|
| **Logistic Regression** | Agnostic | 노이즈 허용, threshold로 분류 |
| **SVM (soft-margin)** | Agnostic | Hinge loss + regularization |
| **Random Forest** | Agnostic | Bagging + ensemble |
| **Neural Networks** | Agnostic | Surrogate loss (cross-entropy) |

---

## ⚖️ 가정과 한계

1. **$1/\epsilon^2$의 느린 수렴**: 높은 정확도를 원하면 샘플이 제곱으로 필요. 현대 ML에서는 이보다 빠른 수렴(Rademacher, Ch5)이 관찰된다.

2. **Loose union bound**: 여전히 $h$에 대한 union bound를 쓰므로, $h$가 크면 bound가 vacuous. 무한 $\mathcal{H}$에서는 더 정교한 도구(VC, Rademacher) 필요.

3. **Distribution-free의 대가**: 모든 분포에서 성립하는 bound를 원하므로, 특정 "쉬운" 분포(low noise, large margin)의 정보를 활용하지 못한다.

4. **Uniform bound의 과도한 보수성**: Union bound는 "최악의 경우"를 본다. 실제로는 많은 가설이 redundant하면 훨씬 타이트.

5. **Computational complexity**: 정보 이론적으로 agnostic PAC learnable해도, ERM 최적화가 어려울 수 있다 (surrogate loss의 필요성).

---

## 📌 핵심 정리

- **Agnostic PAC Learning**: 최적 가설이 오차를 가질 수 있는 현실적 설정. $\inf_h L_\mathcal{D}(h) =: L_\mathcal{H}^* \geq 0$.
- **Sample complexity** (유한 $\mathcal{H}$): $m = O([log h + \log(1/\delta)] / \epsilon^2)$ — **느린 $1/\epsilon^2$ rate**.
- **양쪽 꼬리의 필요성**: $|L_\mathcal{D} - L_S|$의 **절댓값**을 bound해야 하므로, Hoeffding의 양쪽을 모두 쓴다.
- **제곱의 출현**: Realizable의 $(1-\epsilon)^n \sim e^{-n\epsilon}$가 agnostic에서 $e^{-2n\epsilon^2}$로 지수 강화 → $1/\epsilon$ → $1/\epsilon^2$ 의존.
- **현실성**: Agnostic이 현실 문제에 더 가까우나, 샘플 복잡도가 높다. Rademacher/Stability로 개선 가능.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> 정리 3.3.1의 증명 Step 5에서 "factor 2"가 두 번 나타나는 부분을 자세히 설명하라: 왜 $\epsilon$ 대신 $\epsilon/2$를 Hoeffding에 쓰는가?</summary>

<br/>

**해설**. Hoeffding으로 $\sup_h |L_\mathcal{D} - L_S| \leq \epsilon/2$를 확률 $1-\delta$로 보장했을 때, ERM의 gap을 계산하면:

$$L_\mathcal{D}(\hat{h}) \leq L_S(\hat{h}) + \epsilon/2 \quad \text{(gap 정의)}$$
$$\leq \min_h L_S(h) + \epsilon/2 \quad \text{(ERM 정의)}$$
$$\leq L_S(h^*_\mathcal{H}) + \epsilon/2$$
$$\leq L_\mathcal{D}(h^*_\mathcal{H}) + \epsilon/2 + \epsilon/2 = L_\mathcal{H}^* + \epsilon.$$

두 번째 $\epsilon/2$가 $L_S(h^*_\mathcal{H}) \leq L_\mathcal{D}(h^*_\mathcal{H}) + \epsilon/2$ (gap bound)에서 온다. 따라서 원하는 최종 bound $\epsilon$을 얻기 위해 Hoeffding에 $\epsilon/2$를 input으로 준다. **총 gap이 $\epsilon/2 + \epsilon/2 = \epsilon$**. 효율적인 상수 factor handling. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> 정리 3.3.1에서 $2|\mathcal{H}|$ 대신 $|\mathcal{H}|$를 union bound에 쓸 수 없는 이유는 무엇인가? (Hint: Hoeffding의 "2"에서 옴)</summary>

<br/>

**해설**. Hoeffding의 양쪽 꼬리:
$$\mathbb{P}[|L_S - L_\mathcal{D}| \geq t] = \mathbb{P}[L_S - L_\mathcal{D} \geq t] + \mathbb{P}[L_S - L_\mathcal{D} \leq -t] \leq 2e^{-2nt^2}.$$

각 방향에 $e^{-2nt^2}$인데, 합치면 "2" 계수가 붙는다 (union of two tail events). 따라서 union bound when there are $h$ hypotheses:

$$2|\mathcal{H}| e^{-2nt^2}.$$

이것이 step 3의 "$2|\mathcal{H}|$"의 원인. 정리할 때 이것이 "$2\log(2|\mathcal{H}|/\delta)$" 형태가 되는 것. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> 실전에서 noise rate가 정확히 모를 때, agnostic bound를 어떻게 적용할 수 있을까? "$\epsilon$을 **선택**할 수 없다면" — 예를 들어 cross-validation으로 진짜 noise를 추정해야 하는가?</summary>

<br/>

**해설**. Agnostic PAC는 **$\inf_h L_\mathcal{D}(h)$를 알고 있다고 가정**했다. 하지만 실전에서는 모른다. 해결책:

1. **Cross-validation**: 독립 test set에서 $\hat{L}_\mathcal{H}^* := \min_h \frac{1}{|S_{\text{val}}|}\sum_{(x,y) \in S_{\text{val}}} \ell(h(x), y)$로 추정.

2. **Empirical bound**: $L_\mathcal{D}(\hat{h}) \leq \hat{L}_\mathcal{H}^* + \epsilon$ 대신 $L_\mathcal{D}(\hat{h}) \leq \hat{L}_{S_{\text{train}}}(\hat{h}) + \epsilon + \text{gap}_{\text{train}}$.

3. **SRM (Ch7-01)**: Agnostic + Model Selection. 여러 $\mathcal{H}$ 크기를 시도, 각각의 bound 계산, 최선 선택.

기술적으로는 **$L_\mathcal{H}^*$의 추정 오차**가 추가되지만, 큰 test set이 있으면 무시 가능. $\square$

</details>

---

<div align="center">

◀ [이전: 02. Realizable Case의 학습 가능성](./02-realizable-case.md) | [📚 README](../README.md) | [다음: 04. Fundamental Theorem of Statistical Learning ▶](./04-fundamental-theorem.md)

</div>
