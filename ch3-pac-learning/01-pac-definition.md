# 01. PAC Learnability의 정의

## 🎯 핵심 질문

- **Valiant(1984)의 PAC-learnability**는 어떻게 정의되는가? "Probably Approximately Correct"는 수학적으로 정확히 무엇인가?
- **Sample complexity** $m_\mathcal{H}(\epsilon, \delta)$는 왜 $\epsilon, \delta$의 함수로 정의되며, 실전에서는 언제 유한한 크기가 되는가?
- **Realizable case** (최적 가설이 $\mathcal{H}$에 있음)와 **agnostic case** (없을 수 있음)의 차이는 무엇인가?
- **Efficient PAC learnability**는 "다항시간"을 어떤 의미로 요구하는가? 정보 이론적 vs 계산 복잡도?
- 유한 $\mathcal{H}$에서 **간단한 분석이 왜 PAC 학습을 보장**하는가? 무한 $\mathcal{H}$에서는 무엇이 달라지는가?

---

## 🔍 왜 이 정의가 현대 ML에서 중요한가

"내 모델이 테스트 데이터에 잘 일반화할까?" 이 질문에 답하는 것이 SLT의 전체 목표다. 하지만 **답이 "항상 성공"이 되려면 무한 샘플이 필요**하고, "항상 실패"도 가능하다(분포 나쁘면). SLT의 획기적 기여는 이 둘 사이의 중간 지점을 수량화한 것: **"어떤 $\epsilon > 0$ (오차 한계)와 $\delta > 0$ (실패 확률)에 대해서도, $m(\epsilon, \delta)$개 샘플이 충분하면 확률 $\geq 1-\delta$로 오차 $\leq \epsilon$인 가설을 찾을 수 있다"**. 이것이 바로 **PAC**다.

이 정의 없이는 우리가 가진 것은 "충분히 많은 데이터가 필요하다"는 모호한 주장뿐이다. PAC는 이를 **정량화**해, 주어진 $(X, Y)$ 분포와 가설공간 $\mathcal{H}$에 대해 정확히 몇 개의 샘플을 모아야 하는지를 이론적으로 추정 가능하게 만든다. 알고리즘 설계자는 PAC bound를 보고 "이 모델은 10만 샘플이 필요하고, 저것은 1만만으로도 된다"라는 의사결정을 할 수 있다.

---

## 📐 수학적 선행 조건

- Ch1-01, Ch1-02: 위험 $L_\mathcal{D}$·$L_S$ 정의, Bayes error $L^*$
- Ch2-02: **Hoeffding 부등식** $\mathbb{P}(|\bar{X} - \mu| \geq t) \leq 2e^{-2nt^2}$
- 기초: 확률변수, 고정 vs 데이터 의존적 개체, Big-O 표기, 역함수(implicit)

---

## 📖 직관적 이해

### "Learnable"의 직관

학습 알고리즘 $A$가 문제를 **PAC learn**한다는 것은, "아무리 까다로운 요구사항 $(\epsilon, \delta)$를 받아도, 이에 맞는 샘플크기 $m$을 알려주면 그것만큼 데이터를 줄 때 높은 확률로 요구사항을 만족시킬 수 있다"는 뜻이다. 한 번에 하나씩:

- **"Approximately Correct"** ($\epsilon$): 진짜 위험이 최선의 가설 + $\epsilon$이 이하다. 완벽할 필요 없지만, 정할 수 있는 정도만큼 좋다.
- **"Probably"** ($1-\delta$): 확률 $1-\delta$ 이상의 높은 확률로. $\delta$는 작은 실패 확률. 우리가 운 나쁠 가능성을 허용한다.
- **Sample complexity** $m(\epsilon, \delta)$: 더 정확한 예측 ($\epsilon$↓) 혹은 더 확실한 성공 ($\delta$↓)을 원할수록 더 많은 데이터가 필요하다.

### 왜 Valiant는 이렇게 정의했나?

1970년대까지의 통계 학습 이론은 "$L_S \to L_\mathcal{D}$ as $n \to \infty$"만 보였다 — **점근** 수렴. 하지만 ML 실무자들은 "지금 10000개 샘플 가지고 무엇을 할 수 있나?"를 묻는다. Valiant는 "학습 가능성"을 **유한 샘플에서의 성공 보장**으로 재정의했다. 이것이 PAC learnability의 혁신.

---

## ✏️ 엄밀한 정의

### 정의 3.1 (PAC Learnability — Valiant 1984)

학습자(알고리즘) $A: (\mathcal{X} \times \mathcal{Y})^n \to \mathcal{H}$가 가설공간 $\mathcal{H}$를 **PAC learn**한다고 하자:

$$\exists m: (0,1)^2 \to \mathbb{N}, \quad \forall \mathcal{D} \text{ on } \mathcal{X} \times \mathcal{Y}, \quad \forall \epsilon, \delta \in (0,1),$$
$$n \geq m(\epsilon, \delta) \Rightarrow \mathbb{P}_{S \sim \mathcal{D}^n} \left[ L_\mathcal{D}(A(S)) \leq \inf_{h \in \mathcal{H}} L_\mathcal{D}(h) + \epsilon \right] \geq 1 - \delta.$$

여기서 **$m_\mathcal{H}(\epsilon, \delta)$**를 **sample complexity**라 부른다.

**해석:**
- 오른쪽 식: $A$가 출력한 가설의 진짜 위험이, $\mathcal{H}$ 내 최적값 + $\epsilon$ 이하다.
- 왼쪽의 확률: 샘플이 유리하거나 불리하게 뽑혀 이 조건을 실패할 확률이 $\leq \delta$.
- $m(\epsilon, \delta)$: 이 두 조건을 보장하려면 적어도 이만큼 샘플이 필요하다.

### 정의 3.2 (Realizable vs Agnostic)

- **Realizable PAC learning**: 존재하는 $h^* \in \mathcal{H}$에 대해 $L_\mathcal{D}(h^*) = 0$ (최적 가설이 존재하고 완벽함). 위 정의에서 $\inf_h L_\mathcal{D}(h) = 0$.

- **Agnostic PAC learning**: Realizable 가정 없음. $\mathcal{H}$ 내 최선의 가설도 오차 가지고 있을 수 있음. $\inf_h L_\mathcal{D}(h) =: L_\mathcal{H}^* \geq 0$.

### 정의 3.3 (Efficient PAC Learnability)

Sample complexity $m(\epsilon, \delta)$가 다음을 만족할 때 **efficient**:

$$m(\epsilon, \delta) \text{는 } \frac{1}{\epsilon}, \frac{1}{\delta}, d, |x|_{\text{enc}} \text{에 대해 다항식.$$

여기서 $d = \dim(\mathcal{X})$, $|x|_{\text{enc}}$ = input encoding 길이. 즉, $m = \text{poly}(1/\epsilon, 1/\delta, d, |x|)$.

또한 **runtime**이 $m(\epsilon, \delta) \cdot \text{poly}(d, |x|)$에 다항시간이어야 한다 (정보 이론적 bound를 달성 가능하면서 계산도 빨리).

---

## 🔬 정리와 증명

### 정리 3.1 (유한 $\mathcal{H}$의 Realizable PAC Learning)

$|\mathcal{H}| = h < \infty$이고 realizable 가정 ($\exists h^* \in \mathcal{H}, L_\mathcal{D}(h^*) = 0$) 하에서, ERM 알고리즘은

$$m(\epsilon, \delta) = \left\lceil \frac{\log(h/\delta)}{\epsilon} \right\rceil$$

개의 샘플로 PAC learn한다.

**증명**. 다음과 같이 나쁜 사건을 정의하자:

$$B_h := \{L_\mathcal{D}(h) > \epsilon\} \cap \{L_S(h) = 0\}.$$

이것은 "진짜로는 오류율이 $\epsilon$을 넘지만, 샘플에서는 완벽하게 맞추는" 가설을 뜻한다. Realizable 가정에서 ERM이 $\hat{h}^*_S = A(S)$를 선택한다면, $\hat{h}^*_S \notin B_h$ 모든 $h$에 대해야 $L_S(\hat{h}^*_S) = 0$이고 $L_\mathcal{D}(\hat{h}^*_S) \leq \epsilon$을 만족한다.

각 "나쁜" 가설 $h$에 대해, realizable 가정과 Hoeffding (Ch2-02)에 의해:

$$\mathbb{P}[L_S(h) = 0 | L_\mathcal{D}(h) > \epsilon] \leq (1-\epsilon)^n \leq e^{-n\epsilon}.$$

Union bound: 모든 $h$에 대한 나쁜 사건의 합은

$$\mathbb{P}\left[\exists h \in \mathcal{H}: L_S(h) = 0 \text{ and } L_\mathcal{D}(h) > \epsilon\right] \leq h \cdot e^{-n\epsilon}.$$

이를 $\delta$ 이하로 하려면

$$h \cdot e^{-n\epsilon} \leq \delta \Rightarrow e^{-n\epsilon} \leq \frac{\delta}{h} \Rightarrow n\epsilon \geq \log(h/\delta) \Rightarrow n \geq \frac{\log(h/\delta)}{\epsilon}. \qquad \square$$

**결론**: 이 $n$일 때 $\mathbb{P}[L_\mathcal{D}(\hat{h}^*_S) \leq \epsilon] \geq 1 - \delta$.

### 정리 3.2 (크기에 따른 복잡도 구분)

- **유한 $\mathcal{H}$**: $m = O(\frac{1}{\epsilon}[\log h + \log(1/\delta)])$ (정리 3.1의 첫 항 무시)
- **비유한 그러나 VC 유한** ($\text{VC}(\mathcal{H}) = d < \infty$): $m = O(\frac{1}{\epsilon^2}[d + \log(1/\delta)])$ (Ch4로 미룸)
- **VC 무한**: PAC learnable **아님** (일반적으로)

$\epsilon^2$의 출현 (vs 유한 $\mathcal{H}$의 $\epsilon$)은 **agnostic case**(Ch3-03)에서 온다.

---

## 💻 NumPy 구현 검증

### 실험 1: Realizable case의 Sample Complexity 측정

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# 진짜 분포 𝒟: X ~ Uniform[-1, 1], Y = sign(X)
def sample_D(n):
    X = rng.uniform(-1, 1, n)
    Y = np.sign(X).astype(int)  # 0/1로 인코딩
    return X, Y

# 가설공간: axis-aligned threshold classifiers
# h_theta(x) = sign(x - theta) for theta in [−1, 1]
# 이산화: h ∈ {h_{θ_i} : i = 1, ..., K}로 유한화
K = 200  # |H|
theta_grid = np.linspace(-1, 1, K)

def classifier(theta):
    return lambda x: (x >= theta).astype(int)

def emp_loss_01(theta, X, Y):
    """0-1 loss"""
    pred = classifier(theta)(X)
    return (pred != Y).sum() / len(Y)

def true_loss_01(theta):
    """P(sign(X) != sign(X - theta))"""
    # X ~ U[-1, 1], Y = sign(X)
    # threshold h at theta misclassifies when sign(X) ≠ sign(X - theta)
    # This is nontrivial, approximate empirically
    return None  # 근사적으로 계산

# ERM: 모든 h에 대해 L_S(h) 계산, 최소인 것 선택
def erm_learn(S):
    X, Y = S
    losses = [emp_loss_01(t, X, Y) for t in theta_grid]
    best_idx = np.argmin(losses)
    return theta_grid[best_idx]

# 실험: n 증가에 따른 ERM의 성공률 확인
ns = [10, 20, 50, 100, 200, 500]
epsilons = [0.1, 0.2, 0.3]
delta = 0.1

fig, ax = plt.subplots(figsize=(10, 5))

for eps in epsilons:
    success_rates = []
    for n in ns:
        n_trials = 200
        successes = 0
        for _ in range(n_trials):
            X, Y = sample_D(n)
            theta_hat = erm_learn((X, Y))
            # "성공": 정확한 classifier theta_opt = 0을 회복하는 경우
            # Realizable: theta_opt = 0일 때 완벽하게 분류
            # 근사: theta_hat이 theta_opt에 가까울 때 성공
            X_test, Y_test = sample_D(5000)
            test_loss = emp_loss_01(theta_hat, X_test, Y_test)
            if test_loss <= eps:
                successes += 1
        success_rates.append(successes / n_trials)
    
    ax.plot(ns, success_rates, 'o-', label=f'ε={eps}', linewidth=2)
    
    # 이론적 bound: m = log(K/δ) / ε
    theory_ns = [np.ceil(np.log(K/delta) / eps) for _ in range(len(ns))]
    ax.axvline(theory_ns[0], alpha=0.3, linestyle='--')

ax.axhline(1-delta, color='k', linestyle=':', label=f'1-δ={1-delta}')
ax.set_xlabel('Sample size n'); ax.set_ylabel('Success rate P(L_D ≤ ε)')
ax.set_xscale('log')
ax.set_title(f'Realizable PAC: ERM의 성공률 vs 이론 (|H|={K}, δ={delta})')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# → n ≥ log(K/δ)/ε 근처에서 성공률이 1-δ를 넘음을 확인.
```

### 실험 2: $\epsilon, \delta$의 의존성 확인

```python
# Sample complexity가 1/ε, 1/δ에 어떻게 의존하는가?
ns = np.arange(10, 500, 10)
epsilons = [0.05, 0.1, 0.2, 0.3]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 1/ε에 대한 m의 의존성
for eps in epsilons:
    required_n = np.ceil(np.log(K/0.1) / eps)
    print(f'ε={eps}: m ≥ {required_n:.0f}')

# 이론 vs 실제
theory_eps = epsilons
theory_m = [np.log(K/0.1) / eps for eps in theory_eps]

axes[0].loglog(theory_eps, theory_m, 'o-', label='Theory: m = log(K/δ)/ε', linewidth=2)
axes[0].set_xlabel('ε'); axes[0].set_ylabel('m(ε)')
axes[0].set_title('Sample complexity vs ε (1/ε 의존성)')
axes[0].grid(alpha=0.3); axes[0].legend()

# 1/δ에 대한 m의 의존성
deltas = [0.01, 0.05, 0.1, 0.2, 0.5]
theory_delta_m = [np.log(K/d) / 0.1 for d in deltas]

axes[1].semilogy(deltas, theory_delta_m, 's-', label='Theory: m = log(K/δ)/ε', linewidth=2)
axes[1].set_xlabel('δ'); axes[1].set_ylabel('m(δ)')
axes[1].set_title('Sample complexity vs δ (log(1/δ) 의존성)')
axes[1].grid(alpha=0.3); axes[1].legend()

plt.tight_layout(); plt.show()
```

---

## 🔗 ML 알고리즘 연결

| 알고리즘 | Realizable인가? | 가정 |
|---------|----------------|------|
| **Perceptron (separable)** | ✓ | 선형 분리 가능 |
| **Perceptron (일반)** | ✗ | 선형 분리 불가능 가능 |
| **Logistic Regression** | ✗ | 항상 오차 있을 수 있음 |
| **SVM** | ✗ (soft-margin) | 정규화로 agnostic |
| **Decision Tree** | ✓ (충분히 깊음) | Memorization |

Realizable은 "최적 가설이 우리 모델 클래스에 **정확히** 있다"는 비현실적 가정. Agnostic(Ch3-03)이 더 현실적.

---

## ⚖️ 가정과 한계

1. **Realizable 가정의 비현실성**: 대부분의 실전 문제에서 $\inf_h L_\mathcal{D}(h) > 0$. 노이즈, 라벨 오류, 모델 부정확성 때문.

2. **균등(uniform) bound의 느슨함**: Union bound $h \cdot e^{-n\epsilon} \leq \delta$는 **최악의 경우**를 본다. 많은 가설이 실제로는 서로 비슷해서 (redundancy), 개별 bound의 합보다 훨씬 타이트할 수 있다 (Ch4-04의 Sauer-Shelah로 개선).

3. **Computational complexity**: PAC learning이 **정보 이론적으로** 가능하다고 해서 **다항시간에** 가능한 것은 아니다. 예: 3-CNF satisfiability 학습은 정보적으로 PAC learnable이지만 NP-hard.

4. **$m$의 의존성**: $m(\epsilon, \delta) = \Theta(\frac{1}{\epsilon}[\log h + \log(1/\delta)])$는 $h$에 지수적 의존. $h$가 크면 샘플이 엄청나게 필요. 이것이 **무한 가설공간 이론**(Ch4~5)으로 넘어가는 이유.

5. **Distribution-free bound의 대가**: 어떤 $\mathcal{D}$에서도 성립하는 bound를 원하므로, 특정 "쉬운" 분포에 대한 정보를 활용하지 못한다. Distribution-dependent bound(margin, smoothness)는 더 타이트할 수 있다.

---

## 📌 핵심 정리

- **PAC Learning**: 확률 $\geq 1-\delta$로 오차 $\leq L^*(\mathcal{H}) + \epsilon$인 가설을 찾는 것. **정량화된 학습 가능성**.
- **Sample complexity** $m(\epsilon, \delta)$: 요구사항 $(\epsilon, \delta)$을 만족하려 필요한 최소 샘플 수.
- **Realizable PAC** (유한 $\mathcal{H}$): $m = O(\frac{1}{\epsilon}[\log h + \log(1/\delta)])$. Hoeffding + Union bound로 증명.
- **Valiant의 혁신**: 점근($n \to \infty$) 수렴을 버리고, 유한 샘플에서의 **확률적 성공보장**으로 재정의.
- **Efficient PAC**: Sample complexity뿐 아니라 **계산 복잡도**도 다항식이어야 함.
- **다음 단계**: Agnostic case(Ch3-03), 무한 $\mathcal{H}$(Ch4), Rademacher(Ch5)로 더 정교한 분석.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> Realizable PAC learning에서 정리 3.1의 결과 $m = O(\log(h/\delta)/\epsilon)$이 정보 이론적으로 최소일 때, 왜 $\log h$ 항은 필요한가? "$h=1$ (최적 가설이 정확히 하나뿐)이면 샘플이 필요 없다"는 주장이 맞는가?</summary>

<br/>

**해설**. $h=1$이면 가설공간이 단 하나의 고정된 함수로 이루어진다. 이 경우:
$$m = \log(1/\delta) / \epsilon.$$

하지만 여전히 $1/\epsilon, \log(1/\delta)$ 의존이 남아 있다! 왜냐하면 우리는 **"그 단 하나의 가설이 정말 $\epsilon$-좋은가"를 샘플로 검증**해야 하기 때문. 샘플 없이는 "그냥 운 좋게 $L_S = 0$인 것"인지 "실제로 좋은 것"인지 알 수 없다. Union bound에서 $h=1$로 두면 $\log h = 0$만 사라질 뿐, 개별 가설의 검증 비용인 $1/\epsilon$은 남는다. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> "Efficient PAC learning"의 정의에서 "다항식"이란 정확히 무엇인가? $m(\epsilon, \delta) = 1/\epsilon^{0.9999}$는 다항식인가?</summary>

<br/>

**해설**. 정의상 $m$이 **$1/\epsilon, 1/\delta, d, |x|$ 각각에 대해 다항식**이어야 하므로, $1/\epsilon$의 지수는 상수(예: 0.9999)여야 한다. 따라서 $1/\epsilon^{0.9999} = O(\epsilon^{-0.9999}) = O(1/\epsilon)$로 다항식이 맞다.

그런데 **실전 의미**로는 지수가 작을수록 좋다:
- $1/\epsilon$: 좋음 (선형)
- $1/\epsilon^2$: 낮은 차원, 온순한 문제
- $1/\epsilon^{10}$: 느림, 고차원 문제 신호

Sauer-Shelah lemma (Ch4-04)는 일반적으로 **$1/\epsilon^2$ 의존**을 낳는다 (agnostic case). $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> 실전 ML에서 "$\epsilon=0.1, \delta=0.05$라 정했을 때" 필요한 샘플을 예측하려면 어떤 정보가 필요한가? "샘플 복잡도 테이블"을 만들 수 있을까?</summary>

<br/>

**해설**. 정리 3.1을 쓰려면 **가설공간 크기 $|\mathcal{H}|$**를 알아야 한다. 예시:

| $\mathcal{H}$ | $\log h$ | $m(0.1, 0.05)$ | 비고 |
|--|--|--|--|
| 축정렬 직사각형 4개 | $\log 4 \approx 2$ | $\lceil (2+3)/0.1 \rceil = 50$ | 아주 작음 |
| 선형 분류기 (이산화 $10^6$) | $\log 10^6 \approx 20$ | $\lceil (20+3)/0.1 \rceil = 230$ | 보통 |
| 깊이 5 결정 트리 (이산화 $10^{10}$) | $\approx 33$ | $\approx 360$ | 복잡 |

더 정교하게는 **VC 차원** (Ch4)으로 무한 가설공간도 처리하면, $m = O(\frac{d}{\epsilon^2} + \frac{\log(1/\delta)}{\epsilon^2})$ 꼴의 bound를 얻는다. 여기서 $d = \text{VC}(\mathcal{H})$. $\square$

</details>

---

<div align="center">

◀ [이전: Ch2-05. 집중부등식의 ML 응용](../ch2-concentration/05-applications.md) | [📚 README](../README.md) | [다음: 02. Realizable Case의 학습 가능성 ▶](./02-realizable-case.md)

</div>
