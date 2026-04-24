# 04. Fundamental Theorem of Statistical Learning

## 🎯 핵심 질문

- **Fundamental Theorem**은 무엇을 말하는가? PAC learnability와 VC 차원, uniform convergence, ERM의 성공이 모두 **동치**라는 주장의 의미는?
- 다음 네 개념이 정말 **동치**인가?
  - (a) $\mathcal{H}$가 agnostic PAC learnable
  - (b) $\mathcal{H}$가 uniform convergence property 만족
  - (c) ERM이 $\mathcal{H}$를 agnostic PAC learn
  - (d) $\text{VC}(\mathcal{H}) < \infty$
- 각 방향의 증명이 어떻게 연결되는가? 특히 (d) ⇒ (b)는?
- **Sample complexity의 특성화**: $m_\mathcal{H}(\epsilon, \delta)$의 정확한 형태는 $\text{VC}(\mathcal{H})$에만 의존하는가?
- 이 정리의 역사적 의미는? Vapnik-Chervonenkis(1971)부터 Blumer-Ehrenfeucht-Haussler-Warmuth(1989)까지.

---

## 🔍 왜 이 정리가 "기초정리"인가?

이 정리는 **SLT의 심장**이다. 왜냐하면:

1. **통일된 언어**: "PAC learnable"·"VC 유한"·"uniform convergence"는 **다르게 들리지만 정말로는 같은 현상의 세 측면**이라는 것을 보인다. 이것이 SLT를 통일된 학문으로 만든다.

2. **무한 가설공간의 처리**: Ch3-03까지는 유한 $\mathcal{H}$만 다뤘다. 이 정리는 **무한 $\mathcal{H}$도 해결 가능한 조건**을 제시한다 — VC가 유한하면 된다.

3. **데이터 복잡도의 특성화**: 가설공간의 어떤 기하학적 성질(VC 차원)이 정보 이론적 비용(sample complexity)을 결정한다는 명확한 인과관계.

4. **실전 적용**: "우리 모델의 VC 차원을 계산하면, 필요한 샘플 크기를 이론적으로 예측할 수 있다" (though vacuous in practice for DNNs).

---

## 📐 수학적 선행 조건

- Ch1-01, 1-02: 위험의 정의
- Ch2-02: Hoeffding 부등식
- Ch3-01~03: PAC learnability, realizable/agnostic
- **Ch4 (앞부분)**: VC 차원의 정의, shattering (이 문서에서도 소개하지만, Ch4와 병렬 학습 권장)
- 기초: Uniform convergence ($\sup_h |L_\mathcal{D} - L_S|$), growth function

---

## 📖 직관적 이해

### "동치"라는 주장의 의미

네 개념의 관계를 시각화하면:

```
         (a) PAC learnable
              ⇅
         (c) ERM 성공
              ⇅
      (b) Uniform convergence ←→ (d) VC < ∞
```

- (a) ⇒ (b): "학습자가 $\epsilon, \delta$로 성공할 수 있다" ⟹ "$\sup_h |L_\mathcal{D} - L_S|$이 작아진다" (정의상)
- (b) ⇔ (d): "모든 $n$에서 uniform convergence" ⟺ "VC가 유한" (Sauer-Shelah + 고전 정리)
- (d) ⇒ (a): "VC가 유한" ⟹ "growth function이 다항" ⟹ "PAC가능" (이 문서의 핵심)
- (c) ⟺ (a): ERM이 성공한다는 것의 정의가 곧 PAC learnability

### VC 차원이 왜 핵심인가?

VC 차원은 **$\mathcal{H}$의 크기를 가장 정보 이론적으로 측정하는 방법**이다:
- 개수로 센다면? $|\mathcal{H}| = \infty$이면 답이 없다.
- VC 차원으로 센다면? 기하학적 구조를 반영해 **유한성을 보장**한다.

---

## ✏️ 엄밀한 정의

### 정의 3.4.1 (Uniform Convergence Property)

가설공간 $\mathcal{H}$가 **uniform convergence property**를 만족한다:

$$\forall \epsilon, \delta \in (0,1), \quad \exists n(\epsilon, \delta) \text{ s.t. } \forall \mathcal{D},$$
$$n \geq n(\epsilon, \delta) \Rightarrow \mathbb{P}_{S \sim \mathcal{D}^n}\left[\sup_{h \in \mathcal{H}} |L_\mathcal{D}(h) - L_S(h)| \leq \epsilon\right] \geq 1 - \delta.$$

즉, **모든 분포**와 **모든 가설**에 대해 일정히 수렴한다.

### 정의 3.4.2 (Shattering과 VC 차원 — 간단 소개)

$\mathcal{H}$가 점집합 $C = \{x_1, \ldots, x_k\}$를 **shatter**한다 ⇔ $\mathcal{H}$가 $C$의 모든 $2^k$개 부분집합을 분류할 수 있다:

$$\forall S \subseteq C, \quad \exists h \in \mathcal{H} \text{ s.t. } h|_C = \text{indicator of } S.$$

**VC dimension**:

$$\text{VC}(\mathcal{H}) := \max\{k : \exists C, |C|=k, \mathcal{H} \text{ shatters } C\}.$$

---

## 🔬 정리와 증명

### 정리 3.4.1 (Fundamental Theorem of Statistical Learning — Simplified)

**Binary classification with 0-1 loss** 위에서, 다음은 모두 동치이다:

1. $\mathcal{H}$가 **agnostic PAC learnable**
2. $\mathcal{H}$가 **uniform convergence property** 만족
3. **ERM**이 $\mathcal{H}$를 agnostic PAC learn
4. $\text{VC}(\mathcal{H}) < \infty$

**증명 스케치** (엄밀하지 않은 개요):

**방향 1 ⇒ 2**: (a) agnostic PAC learnable ⇒ (b) uniform convergence

PAC learnability의 정의:

$$\mathbb{P}\left[L_\mathcal{D}(A(S)) \leq \inf_h L_\mathcal{D}(h) + \epsilon\right] \geq 1-\delta.$$

ERM의 경우 (Ch3-03의 정리 3.3.2에서):

$$\mathbb{P}\left[\sup_h |L_\mathcal{D}(h) - L_S(h)| \leq \epsilon/2\right] \geq 1-\delta.$$

따라서 uniform convergence 성립. $\square$

**방향 2 ⇒ 4**: (b) uniform convergence ⇒ (d) VC finite

**대우**로 증명: VC가 무한이면 uniform convergence가 실패함을 보인다.

VC가 무한이면, 임의의 $k$에 대해 $k$개 점을 shatter하는 집합이 존재한다. 이는 **"임의로 복잡한 패턴이 $\mathcal{H}$에 있다"**는 의미.

특정 상황(adversarial distribution)에서, 이렇게 복잡한 $\mathcal{H}$는 **small $n$**에서 overfitting을 피할 수 없고, uniform convergence가 성립하지 않는다.

**엄밀한 증명**: Vapnik-Chervonenkis inequality (symmetric ruin), empirical process theory. 여기서는 스케치만.

**방향 4 ⇒ 2**: (d) VC finite ⇒ (b) uniform convergence

이것이 **가장 중요한 방향**이고 핵심은 **Sauer-Shelah Lemma** (Ch4-04):

$$\text{VC}(\mathcal{H}) = d < \infty \Rightarrow \text{growth function } \Pi_\mathcal{H}(n) \leq \sum_{i=0}^d \binom{n}{i} \leq \left(\frac{en}{d}\right)^d.$$

VC bound (Ch4-05):

$$\mathbb{P}\left[\sup_h |L_\mathcal{D}(h) - L_S(h)| \geq \epsilon\right] \leq 4 \Pi_\mathcal{H}(2n) e^{-n\epsilon^2/8} \leq 4 \left(\frac{2en}{d}\right)^d e^{-n\epsilon^2/8}.$$

$n$이 충분히 크면 지수항 $e^{-n\epsilon^2/8}$이 다항식 $\left(\frac{2en}{d}\right)^d$를 압도하므로, 오른쪽이 0으로 간다. 따라서 uniform convergence. $\square$

**방향 2 ⇒ 3**: (b) uniform convergence ⇒ (c) ERM 성공

Uniform convergence ⟹ 모든 $h$에 대해 $|L_\mathcal{D}(h) - L_S(h)| \approx 0$. 따라서

$$L_\mathcal{D}(\hat{h}) \approx L_S(\hat{h}) = \min_h L_S(h) \approx \min_h L_\mathcal{D}(h) = L_\mathcal{H}^*.$$

ERM이 성공적. $\square$

**방향 3 ⇒ 1**: (c) ERM 성공 ⇒ (a) PAC learnable

정의에서 직접 따름. $\square$

### 정리 3.4.2 (Sample Complexity의 특성화)

VC 유한 ($d := \text{VC}(\mathcal{H}) < \infty$) 일 때, agnostic PAC learning의 sample complexity는

$$m_\mathcal{H}(\epsilon, \delta) = \Theta\left(\frac{1}{\epsilon^2}\left[d + \log\frac{1}{\delta}\right]\right).$$

즉, 하한과 상한이 모두 이 형태.

**의미**:
- VC 차원 $d$에 **선형 의존** (not exponential) — 무한 $\mathcal{H}$ 처리 가능
- $\epsilon^2, \log(1/\delta)$ 의존은 agnostic case의 전형
- **최적성(optimality)**: 이것이 정보 이론적으로 최선임이 증명됨 (Vapnik 1999)

---

## 💻 NumPy 구현 검증

### 실험 1: VC 차원 계산 → Sample complexity 예측

```python
import numpy as np
import matplotlib.pyplot as plt

# 1D threshold classifiers: h_θ(x) = sign(x - θ)
# VC(1D threshold) = 1 (can shatter at most 1 point)

# 2D axis-aligned rectangles: h_{(a1,b1,a2,b2)}(x) = (a1<x1<b1)∧(a2<x2<b2)
# VC(2D rectangles) = 4

# 3D axis-aligned boxes: 
# VC(3D boxes) = 6

# General pattern: VC(d-dim axis-aligned) = 2d

vc_dims = {
    '1D threshold': 1,
    '2D rectangle': 4,
    '3D box': 6,
    'd-dim (linear)': None,  # d+1
}

epsilon_vals = [0.01, 0.05, 0.1, 0.2, 0.3]
delta = 0.05

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 고정 VC로 여러 ε 보기
for name, d in [('1D threshold', 1), ('2D rectangle', 4), ('d=10', 10)]:
    if d is None:
        continue
    m_vals = [2 * (d + np.log(1/delta)) / (eps**2) for eps in epsilon_vals]
    axes[0].semilogy(epsilon_vals, m_vals, 'o-', label=f'{name} (d={d})', linewidth=2)

axes[0].set_xlabel('ε'); axes[0].set_ylabel('m(ε) = 2(d + log(1/δ))/ε²')
axes[0].set_title(f'Sample complexity vs ε (δ={delta}, various VC dims)')
axes[0].legend(); axes[0].grid(alpha=0.3)

# 고정 ε로 여러 d 보기
eps_fixed = 0.1
d_vals = [1, 2, 5, 10, 20, 50, 100]
m_vals = [2 * (d + np.log(1/delta)) / (eps_fixed**2) for d in d_vals]

axes[1].semilogy(d_vals, m_vals, 's-', linewidth=2)
axes[1].set_xlabel('VC dimension d'); axes[1].set_ylabel(f'm(d) with ε={eps_fixed}')
axes[1].set_title(f'Sample complexity vs VC dim (δ={delta})')
axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.show()

print('Sample complexity examples:')
for d in [1, 4, 6, 10]:
    for eps in [0.01, 0.1]:
        m = 2 * (d + np.log(1/delta)) / (eps**2)
        print(f'  d={d:2d}, ε={eps}: m ≈ {m:.0f}')
```

### 실험 2: Uniform convergence 실제 관찰

```python
# 가설공간: axis-aligned 2D rectangles (VC=4)
rng = np.random.default_rng(42)

def sample_D_separable(n):
    """Generate data separable by axis-aligned rectangle"""
    X = rng.uniform(-1, 1, (n, 2))
    # True classifier: (|x_1| < 0.5) & (|x_2| < 0.3)
    Y = ((np.abs(X[:, 0]) < 0.5) & (np.abs(X[:, 1]) < 0.3)).astype(int)
    return X, Y

def gen_hypotheses_2d_rect(n_grid=10):
    """Generate hypotheses: axis-aligned rectangles"""
    a_vals = np.linspace(-1, 1, n_grid)
    rects = []
    for a1 in a_vals:
        for b1 in a_vals[a_vals > a1]:
            for a2 in a_vals:
                for b2 in a_vals[a_vals > a2]:
                    rects.append((a1, b1, a2, b2))
    return rects

def classify_rect(rect, X):
    a1, b1, a2, b2 = rect
    return ((X[:, 0] >= a1) & (X[:, 0] <= b1) &
            (X[:, 1] >= a2) & (X[:, 1] <= b2)).astype(int)

def emp_loss(rect, X, Y):
    pred = classify_rect(rect, X)
    return (pred != Y).sum() / len(Y)

# Experiment: sup_h |L_D - L_S| → 0 as n increases (uniform convergence)
hypotheses = gen_hypotheses_2d_rect(n_grid=8)
print(f'Number of hypotheses: {len(hypotheses)}')

sup_gaps = []
ns = np.linspace(20, 200, 10).astype(int)

for n in ns:
    gaps = []
    for _ in range(50):  # multiple trials
        X_train, Y_train = sample_D_separable(n)
        X_test, Y_test = sample_D_separable(5000)
        
        max_gap = 0
        for rect in hypotheses:
            L_S = emp_loss(rect, X_train, Y_train)
            L_D = emp_loss(rect, X_test, Y_test)
            gap = abs(L_S - L_D)
            max_gap = max(max_gap, gap)
        gaps.append(max_gap)
    
    sup_gaps.append(np.mean(gaps))

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(ns, sup_gaps, 'o-', label='Empirical sup_h |L_D - L_S|', linewidth=2)

# Theoretical VC bound (rough approximation)
vc_d = 4
theory_bound = [4 * ((2*n/vc_d)**vc_d) * np.exp(-n*0.01/8) for n in ns]
ax.semilogy(ns, theory_bound, 's--', label='VC bound (rough)', linewidth=2, alpha=0.7)

ax.set_xlabel('Sample size n'); ax.set_ylabel('sup_h |L_D - L_S|')
ax.set_title('Uniform Convergence: sup_h gap → 0 as n increases')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 🔗 ML 알고리즘 연결

| 개념 | ML 관점 | SLT 해석 |
|------|--------|---------|
| **Model selection** | 어떤 $\mathcal{H}$ 고를지? | VC가 작은 모델 선호 |
| **Regularization** | 복잡도 페널티 추가 | Effective VC 감소 |
| **Early stopping** | 훈련 초기 멈추기 | Implicit regularization (Ch6) |
| **Ensemble** | 여러 모델 결합 | Voting으로 effective VC 감소 |

---

## ⚖️ 가정과 한계

1. **Binary classification with 0-1 loss에만**: 다중 분류, 회귀, 일반 loss는 다른 형태의 정리. 그러나 원리는 유사.

2. **Equivalence는 정성적**: (a)-(d)가 모두 동치지만, **정량적 상수**는 다를 수 있다. Sample complexity의 상수가 loose할 수 있다.

3. **VC 유한의 의미**: VC < ∞는 PAC 가능을 보장하지만, **computational hardness**는 해결하지 않는다. ERM이 NP-hard일 수 있다.

4. **Vacuous bound in practice**: VC bound는 정보 이론적으로 최적이지만, 상수가 크고 고차항이 있어서 **실전 $n$에서는 vacuous** (bound > 1). 신경망 같은 큰 VC에서는 특히.

5. **Distribution-free의 한계**: 모든 분포에서 성립하는 bound를 원하므로, 특정 분포(margin, smoothness)의 구조를 활용하지 못한다.

---

## 📌 핵심 정리

- **Fundamental Theorem**: 다음 네 개념이 **동치**:
  - PAC learnability
  - Uniform convergence
  - ERM의 성공
  - VC dimension < ∞
  
- **핵심 방향**: (d) VC finite ⟹ (b) uniform convergence는 **Sauer-Shelah** (다항 growth) ⟹ **VC bound** (양쪽 꼬리 exponential).

- **Sample complexity 특성화**: $m = \Theta((d/\epsilon^2 + \log(1/\delta)/\epsilon^2))$. VC에만 의존, 분포 자유.

- **무한 가설공간의 처리**: VC 유한이면 무한 $\mathcal{H}$도 PAC 가능. "크기"의 new 정의.

- **현대적 관점**: 고전 VC bound는 이론적으로 완벽하지만, 신경망의 실제 행동은 Rademacher/margin/stability (Ch5-6)로 더 잘 설명됨.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> "VC 차원이 유한이면 PAC learnable"이라는 주장에서, 구체적으로 어느 부분이 무한 VC에서 깨지는가?</summary>

<br/>

**해설**. 무한 VC가 되면:

1. Growth function이 지수적: $\Pi_\mathcal{H}(n) = 2^n$ (가능).
2. Union bound: $2 \Pi_\mathcal{H}(n) e^{-2n\epsilon^2/4} = 2 \cdot 2^n \cdot e^{-n\epsilon^2/2}$.
3. 지수항 vs 다항항: $2^n$은 $e^{-n\epsilon^2/2}$보다 빠르게 커진다 ($n$ 충분히 크면).
4. 따라서 bound가 0으로 안 간다 — **uniform convergence 실패**.

구체 예: $\mathcal{H} = $ 모든 0-1 함수면 $|\mathcal{H}| = 2^{\infty}$이고, 실제로 분포가 "나쁜" (adversarial)이면 overfitting 피할 수 없다. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> Fundamental Theorem이 "binary classification + 0-1 loss"에만 성립한다고 했는데, 회귀 ($\mathcal{Y} = \mathbb{R}$, squared loss)에서는 무엇이 달라지는가?</summary>

<br/>

**해설**. 회귀의 경우:

1. **Shattering이 없다**: $\mathcal{Y}$가 연속이면 유한 점 집합의 "모든 부분집합"을 실현할 수 없다. 따라서 VC 차원의 정의가 자연스럽지 않다.

2. **Fat-shattering dimension** 도입: 연속 예측을 다루는 일반화된 VC. 정의가 더 복잡하지만, 원리는 유사.

3. **Rademacher complexity** 선호: 회귀나 일반 loss에서는 (Ch5로) Rademacher 복잡도가 더 자연스러운 도구.

결론: Fundamental Theorem의 **정신**은 일반화되지만, **정확한 형태**는 분류와 회귀가 다르다. 후속 단계(Ch5)로 넘어가면 더 일반적인 프레임워크 (empirical process theory)가 등장. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> CNN의 parameter space를 생각해보자. Weight matrix에서의 무한 정밀도는 VC를 무한으로 만들지는 않을까? 아니면 "바운드된 weights" 가정이 VC를 유한하게 유지하는가?</summary>

<br/>

**해설**. CNN의 VC를 분석하려면:

1. **Parameter space**: $\mathbb{R}^W$ (W parameters). 실수 공간은 연속이므로, 무한 정밀도에서 VC = ∞ (가능).

2. **하지만 "바운드된 weights"**: $\|w\| \leq B$ 가정 하에, VC는 有限이지만 **매우 크다** — 예: VC $\approx O(W^2 \log W)$ (Bartlett-Maiorov).

3. **그래도 vacuous**: 신경망은 $W = 10^7$ 정도이므로, $\log(VC(\mathcal{H})/\delta) / \epsilon^2 \approx 10^{14}$ 같은 천문학적 bound.

4. **실전 일반화**: 이론과 달리 신경망은 잘 일반화한다. 이것이 "generalization puzzle"이고, Ch5(Rademacher norm bound)와 Ch6(Stability)의 동기. $\square$

</details>

---

<div align="center">

◀ [이전: 03. Agnostic PAC Learning](./03-agnostic-pac.md) | [📚 README](../README.md) | [다음: 05. Occam's Razor와 MDL 원리 ▶](./05-occam-mdl.md)

</div>
