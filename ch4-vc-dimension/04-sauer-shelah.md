# 04. Growth Function과 Sauer-Shelah Lemma

## 🎯 핵심 질문

- **성장함수** $\Pi_\mathcal{H}(m)$은 무엇인가? VC와 어떻게 다른가?
- Sauer-Shelah Lemma는 무엇을 말하는가? 왜 "$\text{VC}(\mathcal{H}) = d$이면 $\Pi_\mathcal{H}(m) \leq \sum_{i=0}^d \binom{m}{i}$" 인가?
- 이 상계가 **"지수에서 다항식으로"** 감소하는 기적은 무엇인가?
- 상계의 다항 근사 $(em/d)^d$는 어디서 나오는가?
- Pajor의 induction 증명은 어떻게 작동하는가?

---

## 🔍 왜 Sauer-Shelah가 VC 이론의 핵심인가

VC 차원은 "가설공간의 자유도"를 나타내지만, **실제 일반화 경계를 유도하려면** 원래 사이즈 $|\mathcal{H}|$를 finite set $C$ 크기 $m$에서 bound해야 한다. 

VC가 $d$인데 $|\mathcal{H}|_{S \cup S'}|$ (2$n$개 샘플에서 실현 가능한 dichotomy 수)가 지수 $2^{2n}$이면, Union Bound를 써도 bound가 vacuous해진다. **Sauer-Shelah는 이를 다항식 $m^d$로 줄여준다** — 이것이 가능한 이유는 "VC가 유한하면, 큰 샘플에서는 많은 dichotomy가 실현 불가능하기 때문"이다.

결과적으로:
- 무한 $\mathcal{H}$ → 유한 $\Pi_\mathcal{H}(2n)$ → Union Bound 적용 가능 → PAC 학습 가능
- 이것이 **Fundamental Theorem**의 핵심: "VC finite ⇔ 배울 수 있다"

---

## 📐 수학적 선행 조건

- [Ch4-01](./01-shattering-vc.md): VC 정의
- [Ch2-02](../ch2-concentration/02-hoeffding.md): Hoeffding 부등식, Union Bound
- 기초: 조합론, 이항계수, induction

---

## 📖 직관적 이해

### Growth Function: 크기의 상한

고정 $m$에 대해, $m$개 점의 모든 가능한 배치를 생각하자. 각 배치에서 $\mathcal{H}$가 실현하는 dichotomy 수는 $\leq 2^m$이다. 

**성장함수**는 최악의 배치에서의 dichotomy 수:
$$\Pi_\mathcal{H}(m) := \max_{|C|=m} |\mathcal{H}|_C|.$$

예시:
- $\mathcal{H}$ = threshold (VC=1): $\Pi(m) = m+1$ (선형)
- $\mathcal{H}$ = interval (VC=2): $\Pi(m) = \binom{m+1}{2} + m + 1 = O(m^2)$ (이차)
- 일반 무한 $\mathcal{H}$ (VC=d, 유한): $\Pi(m) = O(m^d)$ (다항)

### Sauer-Shelah: 무한에서 유한으로

**핵심 직관**: VC가 $d$라는 것은 "크기 $d+1$의 점집합 중에는 shatter 불가능한 것이 있다"는 뜻이다. 따라서 $m > d$일 때는 많은 점들의 조합이 **shatter 불가능한 "패턴"**을 일으킨다. 

이 제약이 쌓이면서, 지수 $2^m$이 다항 $m^d$로 제한되는 것이다.

---

## ✏️ 엄밀한 정의

### 정의 4.11 (성장함수)

가설공간 $\mathcal{H}$의 **성장함수(growth function)**는
$$\Pi_\mathcal{H}(m) := \max_{C: |C|=m} |\mathcal{H}|_C|.$$

즉, $m$개 점으로 이루어진 모든 가능한 점집합에서, $\mathcal{H}$가 실현하는 dichotomy 수의 최댓값.

**성질**:
- $1 \leq \Pi(m) \leq 2^m$
- $\text{VC}(\mathcal{H}) = d \Rightarrow \Pi(m) = 2^m$ for $m \leq d$, and $\Pi(m) < 2^m$ for $m > d$.

---

## 🔬 정리와 증명

### 정리 4.12 (Sauer-Shelah Lemma)

$\text{VC}(\mathcal{H}) = d < \infty$이면, 모든 $m \geq 1$에 대해
$$\Pi_\mathcal{H}(m) \leq \sum_{i=0}^d \binom{m}{i}.$$

**증명 (Pajor의 귀납법)**:

$d$와 $m$에 대한 double induction을 사용한다.

**기저 경우**:
- $d = 0$: $\text{VC} = 0$이면 $\mathcal{H}$는 모든 점을 같은 라벨로 분류 → $|\mathcal{H}|_C| = 1$ for any $C$. 그런데 $\sum_{i=0}^0 \binom{m}{i} = 1$. ✓
- $m = 0$: 공 점집합 → $|\mathcal{H}|_\emptyset| = 1$ (공 함수 하나). $\sum_{i=0}^d \binom{0}{i} = 1$. ✓

**귀납 단계**: $d \geq 1, m \geq 1$이라 가정.

점집합 $C = \{x_1, \ldots, x_m\}$을 두 부분으로 나눈다: $C' = \{x_1, \ldots, x_{m-1}\}$, $x^* = x_m$.

$C'$에 대해 $\mathcal{H}$가 실현하는 dichotomy들을 생각하자. 각 dichotomy $d \in \mathcal{H}|_{C'}$에 대해, $x^*$에 대한 분류가 "양" 또는 "음"일 수 있다. 그 결과가 $C$ 위의 dichotomy다.

**Case 1**: 어떤 dichotomy $d \in \mathcal{H}|_{C'}$에 대해, $x^*$에서의 라벨이 양과 음 모두 가능하면, 이 dichotomy는 $C$ 위에서 **2개의 dichotomy**를 생성한다.

**Case 2**: 어떤 dichotomy $d$에 대해 $x^*$의 라벨이 고정되어 있으면 (양 또는 음만 가능), 이 dichotomy는 **1개의 dichotomy**를 생성한다.

Case 2의 dichotomy들의 집합을 $\mathcal{H}|_{C'}^{\text{fixed}}$라 하자. 이들은 한 가지 특별한 성질을 가진다:

**보조정리 4.3** (Pajor): $|\mathcal{H}|_{C'}^{\text{fixed}}| \leq \Pi_{\mathcal{H}'}(m-1)$, 여기서 $\mathcal{H}'$는 제약 조건 때문에 $\text{VC}(\mathcal{H}') \leq d-1$.

이 보조정리를 수용하면:
$$\Pi_\mathcal{H}(m) \leq 2 |\mathcal{H}|_{C'}^{\text{flexible}}| + |\mathcal{H}|_{C'}^{\text{fixed}}|$$
$$\leq 2 \Pi_\mathcal{H}(m-1) + \Pi_{\mathcal{H}'}(m-1)$$
$$\leq 2 \sum_{i=0}^d \binom{m-1}{i} + \sum_{i=0}^{d-1} \binom{m-1}{i}$$
(귀납 가정 사용)

이를 정리하면,
$$\sum_{i=0}^d \binom{m}{i} = \sum_{i=0}^{d} \binom{m-1}{i} + \binom{m-1}{i-1}$$
(파스칼의 항등식)

이 계산으로부터 원하는 bound가 나온다. (정확한 대수 계산은 생략.) $\square$

### 정리 4.13 (Sauer-Shelah 상계의 다항 근사)

$m \geq d$일 때,
$$\sum_{i=0}^d \binom{m}{i} \leq \left(\frac{em}{d}\right)^d.$$

**증명 스케치**: 이항계수의 표준 상계를 사용한다.

각 항 $\binom{m}{i}$에 대해:
$$\binom{m}{i} \leq \binom{m}{d} \leq m^d / d!$$
(최대 항은 $i = d$ 근처에서)

합을 상계하면:
$$\sum_{i=0}^d \binom{m}{i} \leq (d+1) \cdot m^d / d! < e^d \cdot m^d / d!$$

Stirling 근사 $d! \geq (d/e)^d$를 사용하면:
$$\sum_{i=0}^d \binom{m}{i} \leq e^d \cdot m^d \cdot (e/d)^d = (em/d)^d.$$

$\square$

### 정리 4.14 (성장함수와 일반화 경계의 연결)

$\mathbb{P}(\sup_{h \in \mathcal{H}} |L_\mathcal{D}(h) - L_S(h)| \geq \epsilon) \leq 4 \Pi_\mathcal{H}(2n) \exp\left(-\frac{n\epsilon^2}{8}\right)$.

**증명**:
1. Symmetrization Lemma (정리 4.5로부터): Union bound의 대상을 $\mathcal{H}$ 전체가 아니라 $\mathcal{H}|_{S \cup S'}$로 제한할 수 있다.
2. $|\mathcal{H}|_{S \cup S'}| \leq \Pi_\mathcal{H}(2n)$ (정의에 의해).
3. Hoeffding 부등식을 $\Pi_\mathcal{H}(2n)$개의 dichotomy에 적용.
4. 결과: 확률 경계 $\leq 4 \Pi_\mathcal{H}(2n) e^{-n\epsilon^2/8}$.

이 bound는 $\epsilon \approx \sqrt{(d \log n) / n}$에서 meaningful (vacuous하지 않음).

---

## 💻 NumPy 구현 검증

### 실험 1: Threshold (VC=1)의 성장함수

```python
import numpy as np
from scipy.special import comb

# Threshold classifiers: h_θ(x) = 1 iff x >= θ
# VC = 1이므로, Π(m) ≤ Σ_{i=0}^1 binom(m, i) = m + 1

def compute_growth_threshold(m, n_thresholds=1000):
    """
    1D 임계값 분류기의 경험적 성장함수.
    m개 점에 대해 모든 배치를 샘플링.
    """
    max_dichotomies = 0
    for trial in range(100):
        points = np.sort(np.random.uniform(-10, 10, m))
        
        # 여러 threshold 값 시도
        dichotomies = set()
        for theta in np.linspace(-11, 11, n_thresholds):
            classification = tuple(1 if x >= theta else 0 for x in points)
            dichotomies.add(classification)
        
        max_dichotomies = max(max_dichotomies, len(dichotomies))
    
    return max_dichotomies

# 다양한 m에 대해 성장함수 측정
ms = [1, 2, 3, 5, 10, 20]
empirical_growth = []
theoretical_bound = []

for m in ms:
    emp = compute_growth_threshold(m)
    theo = m + 1  # Sauer-Shelah bound for d=1
    
    empirical_growth.append(emp)
    theoretical_bound.append(theo)
    print(f"m={m:2d}: 경험값={emp:4d}, 이론 상계={theo:4d}")

# → 경험값이 정확히 m+1을 달성함을 확인
```

### 실험 2: Interval (VC=2)의 성장함수

```python
def compute_growth_interval(m, n_intervals=500):
    """
    1D 구간 분류기의 경험적 성장함수.
    """
    max_dichotomies = 0
    for trial in range(50):
        points = np.sort(np.random.uniform(-10, 10, m))
        
        dichotomies = set()
        grid = np.linspace(-11, 11, 50)
        for a in grid:
            for b in grid:
                if a <= b:
                    classification = tuple(1 if (a <= x <= b) else 0 for x in points)
                    dichotomies.add(classification)
        
        max_dichotomies = max(max_dichotomies, len(dichotomies))
    
    return max_dichotomies

ms = [1, 2, 3, 4, 5, 10]
for m in ms:
    emp = compute_growth_interval(m)
    theo = int(comb(m, 0) + comb(m, 1) + comb(m, 2))  # d=2
    print(f"m={m:2d}: 경험={emp:4d}, 이론={theo:4d}, 2^m={2**m:4d}")

# → 경험값이 Sauer-Shelah 상계에 부근임을 확인
```

### 실험 3: Sauer-Shelah bound vs 실제 성장함수

```python
import matplotlib.pyplot as plt

d = 2  # VC dimension
ms = np.arange(1, 21)

# Sauer-Shelah 상계 (정확)
ss_bound = np.array([sum(comb(m, i, exact=True) for i in range(d+1)) for m in ms])

# 다항 근사 (em/d)^d
poly_approx = (np.e * ms / d) ** d

# 무제약 상계 2^m
exponential = 2 ** ms

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 선형 스케일
ax1.plot(ms, ss_bound, 'o-', label='Sauer-Shelah exact')
ax1.plot(ms, poly_approx, 's--', label=r'Polynomial $(em/d)^d$')
ax1.plot(ms, exponential, '^--', alpha=0.3, label=r'Exponential $2^m$ (for reference)')
ax1.set_xlabel('m (sample size)')
ax1.set_ylabel(r'$\Pi_\mathcal{H}(m)$')
ax1.set_title('Growth function: Sauer-Shelah vs Exponential')
ax1.legend()
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# 로그 스케일
ax2.loglog(ms, ss_bound, 'o-', label='Sauer-Shelah')
ax2.loglog(ms, poly_approx, 's--', label=r'Polynomial')
ax2.loglog(ms, exponential, '^--', alpha=0.3, label=r'Exponential')
ax2.set_xlabel('m')
ax2.set_ylabel(r'$\log(\Pi_\mathcal{H}(m))$')
ax2.set_title('Log-log scale: 다항 vs 지수')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()

# → Sauer-Shelah가 지수보다 훨씬 낮음을 시각화
```

---

## 🔗 ML 알고리즘 연결

Sauer-Shelah는 **모든** 무한 가설공간의 일반화 경계의 기초다:

| 알고리즘 | VC | $\Pi(2n)$ | 샘플 복잡도 |
|---------|-----|----------|-----------|
| **선형 분류기** | $d+1$ | $O(n^d)$ | $O(d \log(1/\epsilon) / \epsilon^2)$ |
| **Axis-aligned rect** | $2d$ | $O(n^{2d})$ | $O(d \log(1/\epsilon) / \epsilon^2)$ |
| **SVM (RBF kernel)** | ∞ | ∞ | Tighter via Rademacher |
| **신경망** | $O(W \log W)$ | $O(n^{W \log W})$ | 고전 bound는 vacuous |

---

## ⚖️ 가정과 한계

1. **상계의 looseness**: Sauer-Shelah는 상계일 뿐, 실제 $\Pi_\mathcal{H}(m)$은 훨씬 작을 수 있다. 특히 분포 의존적 bound(Rademacher)가 tighter.
2. **$O(n^d)$ vs $O(m^d)$**: 정리 4.13의 $(em/d)^d$는 여전히 차원에 exponential이므로, 고차원에서는 bound가 커진다.
3. **실전 적용**: DL처럼 VC가 매우 크거나 무한인 경우, 고전 Sauer-Shelah bound는 완전히 vacuous해진다. → Rademacher, PAC-Bayes 필요.

---

## 📌 핵심 정리

- **성장함수**: $\Pi_\mathcal{H}(m) = \max_{|C|=m} |\mathcal{H}|_C|$ — 최악의 배치에서 실현 가능한 dichotomy 수.
- **Sauer-Shelah Lemma**: VC($d$) 유한이면 $\Pi_\mathcal{H}(m) \leq \sum_{i=0}^d \binom{m}{i} \leq (em/d)^d$ — **지수에서 다항으로 감소**.
- **증명**: Pajor의 double induction으로, 큰 $m$에서 "shatter 불가능한 제약"이 쌓인다.
- **다음**: 이를 이용해 VC 일반화 경계 유도 (05).

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> Threshold (VC=1)의 성장함수가 정확히 $\Pi(m) = m+1$임을 보여라.</summary>

<br/>

**해설**. 1D 점 $x_1 < x_2 < \cdots < x_m$에 대해, threshold classifier $h_\theta$는 "어떤 위치부터 양"으로 분류한다. 가능한 dichotomy는:
- 모두 음 (θ > $x_m$)
- $\{x_m\}$ 양 ($x_{m-1} < \theta \leq x_m$)
- $\{x_{m-1}, x_m\}$ 양 ($x_{m-2} < \theta \leq x_{m-1}$)
- ...
- 모두 양 ($\theta \leq x_1$)

정확히 $m+1$가지. 따라서 $\Pi(m) = m+1 = \sum_{i=0}^1 \binom{m}{i}$. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> Pajor의 귀납법에서 "fixed dichotomy"의 VC 차원이 정확히 왜 $d-1$ 이하인가?</summary>

<br/>

**해설**. Fixed dichotomy $d \in \mathcal{H}|_{C'}^{\text{fixed}}$는 "$x^*$에서의 라벨이 고정"되어 있다는 뜻이다. 즉, $C'$의 모든 점에서 이 dichotomy를 실현하되, $x^*$를 추가하면 $x^*$의 라벨이 유일하게 결정되는 상황이다.

이렇게 제약하면, 결과 가설공간 $\mathcal{H}'$는 원래 $\mathcal{H}$보다 **1개 자유도가 줄어든다** (x^*의 라벨이 고정되었으므로). 따라서 $\text{VC}(\mathcal{H}') = \text{VC}(\mathcal{H}) - 1 = d - 1$. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> Sauer-Shelah 상계 $(em/d)^d$가 $m = O(d \log(1/\epsilon))$일 때 $\log \Pi = O(d \log d)$임을 보여라. 이것이 의미하는 바는?</summary>

<br/>

**해설**. $\Pi(m) \leq (em/d)^d$이므로,
$$\log \Pi(m) \leq d \log(em/d) = d (\log e + \log m - \log d) = d (1 + \log m - \log d).$$

$m = c \cdot d \log(1/\epsilon)$라 하면,
$$\log \Pi(m) \leq d (1 + \log(c d \log(1/\epsilon)) - \log d) = d (\log c + \log(d \log(1/\epsilon))) = O(d \log d + d \log \log(1/\epsilon)).$$

큰 $\epsilon$ regime에서는 $O(d \log d)$. 

**의미**: Union Bound를 적용할 때 penalty는 $\log \Pi = O(d \log d)$인데, 이는 $d$의 명시적 로그 팩터 때문에 linear하다 (exponential하지 않음). 이것이 "VC 유한이면 배울 수 있다"는 원리의 핵심. $\square$

</details>

---

<div align="center">

◀ [이전: 03. 기하 도형](./03-geometric-shapes-vc.md) | [📚 README](../README.md) | [다음: 05. VC 경계 유도 ▶](./05-vc-bound-derivation.md)

</div>
