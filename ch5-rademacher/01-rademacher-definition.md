# 01. Rademacher 복잡도의 정의

## 🎯 핵심 질문

- **Rademacher 변수**는 무엇이고, 왜 랜덤 라벨로 해석하는가? $\sigma_i \in \{\pm 1\}$ 균등분포가 "노이즈"를 측정하는 도구가 되는 이유는?
- **경험적 Rademacher 복잡도** $\hat{\mathcal{R}}_S(\mathcal{F})$는 $\mathcal{F}$의 "표현력"을 어떻게 정량화하는가? 왜 데이터에 의존적인가?
- **(모집단) Rademacher 복잡도** $\mathcal{R}_n(\mathcal{F})$는 어떻게 정의되고, VC 차원과 다르게 왜 "더 tighter한" 경계를 주는가?
- $\hat{\mathcal{R}}(\mathcal{F}_1 + \mathcal{F}_2) \leq \hat{\mathcal{R}}(\mathcal{F}_1) + \hat{\mathcal{R}}(\mathcal{F}_2)$라는 기본 성질은 선형함수족의 조합에 어떻게 응용되는가?
- **Gaussian complexity**와의 관계는 무엇인가 — 차이는 약 1.5배 이내?

---

## 🔍 왜 이 이론이 현대 ML에서 중요한가

VC 차원(Ch4)은 가설공간의 "최악의 경우" 복잡도를 측정한다: $\mathcal{H}$가 얼마나 많은 패턴을 구분할 수 있는가? 하지만 VC는 **데이터를 본 후 타이트하게 조정할 수 없다**. Rademacher 복잡도는 이를 넘어선다. $\hat{\mathcal{R}}_S$는 **실제 샘플 $S$에 의존**해서 계산되므로, 매 데이터셋마다 그 복잡도 추정을 "재조정"할 수 있다. SVM의 margin bound, 신경망의 spectral norm 기반 경계, kernel methods 분석 — 현대 ML의 tighter bound들은 대부분 Rademacher 복잡도 기반이다. 또한 Contraction Lemma(Ch5-03)를 통해 surrogate loss(hinge, log loss)의 Rademacher를 원래 함수족의 Rademacher로 부터 바로 계산할 수 있어, **loss 변환의 이론적 정당화**를 제공한다.

---

## 📐 수학적 선행 조건

- **Probability Theory Deep Dive**: 확률공간, 기대값, 독립성, iid 개념
- **Ch1-01 (위험의 정의)**: $L_\mathcal{D}(h), L_S(h)$ 기호와 직관
- **Ch2 (집중부등식)**: Chernoff 방법, Hoeffding's lemma (Rademacher 합의 집중 이해용)
- **Ch3-02, Ch4 (PAC·VC)**: 가설공간 복잡도의 개념
- 기초: 기대값의 선형성, sup의 정의, 지시함수

---

## 📖 직관적 이해

### Rademacher 변수: 동전 던지기와 노이즈 측정

$\sigma_i \in \{\pm 1\}$를 각각 확률 1/2로 취하는 **Rademacher 변수**를 생각하자. 이것은 "동전을 던져서 앞(+1) 또는 뒷면(-1)"과 같다.

이제 고정된 데이터 샘플 $S = (x_1, \ldots, x_n)$와 함수족 $\mathcal{F}$를 생각하자. $\mathcal{F}$의 "표현력"을 측정하려면 다음을 묻는다:

> **"$\mathcal{F}$가 완전히 랜덤한 라벨 $\sigma_1, \ldots, \sigma_n \in \{\pm 1\}$을 얼마나 잘 학습할 수 있는가?"**

만약 $\mathcal{F}$가 너무 표현력 높으면(예: 모든 함수를 포함하면), 어떤 랜덤 라벨이든 완벽하게 맞힐 수 있고, $\sup_f \frac{1}{n} \sum \sigma_i f(x_i)$가 크다. 표현력이 낮으면(예: 상수함수만), 랜덤 라벨에 거의 맞출 수 없고, 값이 작다. 이 측정 방식이 바로 **경험적 Rademacher 복잡도**다.

### 왜 "복잡도"인가?

$\mathcal{F}$가 이 랜덤 신호에 더 잘 "동조(align)"할수록, 즉 더 많은 표현력을 가질수록, 값이 크다. 따라서 **큰 Rademacher 복잡도 = 높은 과적합 위험**. Rademacher는 $\mathcal{F}$가 **노이즈까지 학습하려는 "욕심"**을 정량화한다.

### 데이터 의존성: VC와의 차이

VC 차원은 $\mathcal{H}$의 기하학적 성질만 본다(shattering). Rademacher 복잡도는 **실제 샘플 $S$에 어떻게 분포하는가**를 본다:
- 데이터가 몰려있는 영역에서 $\mathcal{F}$의 능력을 측정
- 서로 다른 샘플셋에 대해 서로 다른 bound 가능
- **실제 데이터가 특별한 구조(margin, low density)를 가지면 더 tighter한 bound**

---

## ✏️ 엄밀한 정의

### 정의 5.1 (Rademacher 변수)

**Rademacher 확률변수 수열** $\{\sigma_i\}_{i=1}^n$은 다음을 만족하는 iid 확률변수들:
$$\sigma_i \in \{\pm 1\}, \quad \mathbb{P}(\sigma_i = +1) = \mathbb{P}(\sigma_i = -1) = \frac{1}{2}.$$

동등하게 $\sigma_i = 2B_i - 1$로, $B_i \sim \text{Bernoulli}(1/2)$.

### 정의 5.2 (경험적 Rademacher 복잡도)

고정된 샘플 $S = (x_1, \ldots, x_n) \in \mathcal{X}^n$과 함수족 $\mathcal{F} \subseteq \mathbb{R}^\mathcal{X}$에 대해 **경험적 Rademacher 복잡도**는:
$$\hat{\mathcal{R}}_S(\mathcal{F}) := \mathbb{E}_\sigma\left[\sup_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^n \sigma_i f(x_i)\right],$$

여기서 기대값은 Rademacher 변수 $\sigma = (\sigma_1, \ldots, \sigma_n)$에 대한 것이고, $f(x_i) \in \mathbb{R}$는 함수 $f$를 점 $x_i$에 평가한 값이다.

**직관**: Rademacher 변수로 "노이즈 라벨"을 생성했을 때, $\mathcal{F}$가 얼마나 잘 상관시킬 수 있는가의 기대값.

### 정의 5.3 ((모집단) Rademacher 복잡도)

샘플 크기 $n$에 대한 **(모집단) Rademacher 복잡도**:
$$\mathcal{R}_n(\mathcal{F}) := \mathbb{E}_{S \sim \mathcal{D}^n}[\hat{\mathcal{R}}_S(\mathcal{F})],$$

즉, 경험적 Rademacher 복잡도를 샘플 $S$에 대해 기대값을 취한 것이다.

### 정의 5.4 (손실 함수족의 Rademacher)

**손실 함수족** $\ell \circ \mathcal{H} = \{\ell(h(\cdot), y) : h \in \mathcal{H}, y \in \mathcal{Y}\}$의 Rademacher 복잡도:
$$\mathcal{R}_n(\ell \circ \mathcal{H}) := \mathbb{E}_S[\hat{\mathcal{R}}_S(\ell \circ \mathcal{H})],$$

여기서 함수족은 $\{(x, y) \mapsto \ell(h(x), y) : h \in \mathcal{H}\}$.

---

## 🔬 정리와 증명

### 정리 5.1 (기본 성질 — 비음성, 부등식)

$\mathcal{F}, \mathcal{F}_1, \mathcal{F}_2 \subseteq \mathbb{R}^\mathcal{X}$에 대해:

1. **비음성**: $\hat{\mathcal{R}}_S(\mathcal{F}) \geq 0$
2. **Singleton**: $\hat{\mathcal{R}}(\{f\}) = 0$ for any single $f$
3. **스칼라 배**: $\hat{\mathcal{R}}_S(c \mathcal{F}) = |c| \hat{\mathcal{R}}_S(\mathcal{F})$ for $c \in \mathbb{R}$
4. **삼각부등식**: $\hat{\mathcal{R}}_S(\mathcal{F}_1 + \mathcal{F}_2) \leq \hat{\mathcal{R}}_S(\mathcal{F}_1) + \hat{\mathcal{R}}_S(\mathcal{F}_2)$

여기서 $\mathcal{F}_1 + \mathcal{F}_2 = \{f_1 + f_2 : f_1 \in \mathcal{F}_1, f_2 \in \mathcal{F}_2\}$.

**증명**. 
1. 절댓값 내 합이 비음수 또는 0이므로 sup도 비음수. $\square$
2. Singleton $\{f\}$에서 $\sup$은 그 $f$ 하나이므로 $\mathbb{E}_\sigma[\sum \sigma_i f(x_i) / n] = \mathbb{E}_\sigma[\sigma_1] \cdot f(x_1)/n + \ldots = 0$ (각 $\mathbb{E}[\sigma_i] = 0$). $\square$
3. $\sup_f \sum \sigma_i (c f(x_i)) = |c| \sup_f \sum \sigma_i f(x_i)$ (선형성). $\square$
4. **삼각부등식**: 
$$\sup_{f_1 \in \mathcal{F}_1, f_2 \in \mathcal{F}_2} \sum \sigma_i (f_1 + f_2)(x_i) \leq \sup_{f_1} \sum \sigma_i f_1(x_i) + \sup_{f_2} \sum \sigma_i f_2(x_i).$$
기대값을 취하면 보존된다. $\square$

### 정리 5.2 (기대값 표현)

$$\mathcal{R}_n(\mathcal{F}) = \mathbb{E}_{S, \sigma}\left[\sup_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^n \sigma_i f(x_i)\right].$$

**증명**. 정의 5.3에 의해 직접 따라온다. $\mathbb{E}_S[\mathbb{E}_\sigma[\cdot]] = \mathbb{E}_{S,\sigma}[\cdot]$. $\square$

### 정리 5.3 (Rademacher와 Gaussian 복잡도의 비교)

**Gaussian 복잡도**를 $G_n(\mathcal{F}) := \mathbb{E}[g_1, \ldots, g_n \text{ iid } \mathcal{N}(0,1)][\sup_f \frac{1}{n}\sum g_i f(x_i)]$로 정의할 때,
$$\mathcal{R}_n(\mathcal{F}) \leq G_n(\mathcal{F}) \leq O(1) \cdot \mathcal{R}_n(\mathcal{F}).$$

더 정확하게는 $G_n(\mathcal{F}) \leq c \sqrt{\log n} \cdot \mathcal{R}_n(\mathcal{F})$ for some universal constant.

**증명 스케치**. Rademacher를 Gaussian으로 근사하는 tail bound 비교(Hoeffding-type). Ledoux & Talagrand(1991) 참고. $\square$

**해석**: Rademacher와 Gaussian은 복잡도를 거의 같은 규모로 측정한다. Gaussian이 더 "smooth"해서 이론상 더 깔끔하지만, Rademacher가 discrete이라 practical하다.

### 정리 5.4 (경험적 Rademacher의 단조성)

$\mathcal{F}_1 \subseteq \mathcal{F}_2$이면 $\hat{\mathcal{R}}_S(\mathcal{F}_1) \leq \hat{\mathcal{R}}_S(\mathcal{F}_2)$.

**증명**. $\sup_{f \in \mathcal{F}_1} = \sup_{\text{smaller set}} \leq \sup_{f \in \mathcal{F}_2} = \sup_{\text{larger set}}$. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1: 선형 함수족의 Rademacher 복잡도

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# 데이터: X ~ N(0, I_d), n 개 샘플
def sample_data(n, d=5):
    X = rng.standard_normal((n, d))
    return X

# 선형 함수족: f_w(x) = w^T x, ||w|| <= 1
def rademacher_linear(X, B=1.0, n_rademacher=1000):
    """
    경험적 Rademacher 복잡도 추정:
    R̂_S = E_σ[sup_{||w||≤B} (1/n) Σ σ_i w^T x_i]
    = E_σ[B ||Σ σ_i x_i || / n]
    
    Monte Carlo로 σ 샘플링
    """
    n, d = X.shape
    vals = []
    for _ in range(n_rademacher):
        sigma = rng.choice([-1, 1], size=n)
        weighted_sum = (sigma[:, None] * X).sum(axis=0)  # shape (d,)
        norm = np.linalg.norm(weighted_sum)
        vals.append(B * norm / n)
    return np.mean(vals)

# 이론적 상한: R_n ≤ B * max||x_i|| / √n (Ch5-05에서 유도)
def rademacher_linear_upper_bound(X, B=1.0):
    n = len(X)
    max_norm = np.linalg.norm(X, axis=1).max()
    return B * max_norm / np.sqrt(n)

# 실험
ns = [10, 20, 50, 100, 200]
d = 5
empirical_rads = []
theoretical_bounds = []

for n in ns:
    X = sample_data(n, d)
    rad_emp = rademacher_linear(X, B=1.0, n_rademacher=500)
    rad_bound = rademacher_linear_upper_bound(X, B=1.0)
    empirical_rads.append(rad_emp)
    theoretical_bounds.append(rad_bound)
    print(f"n={n:3d}: R̂_S(linear) = {rad_emp:.4f}, Upper bound = {rad_bound:.4f}")

# 시각화
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ns, empirical_rads, 'o-', label='Empirical Rademacher (MC)', linewidth=2)
ax.plot(ns, theoretical_bounds, 's--', label='Theoretical upper bound', linewidth=2)
ax.set_xlabel('Sample size n'); ax.set_ylabel('Rademacher complexity')
ax.set_title('Linear class Rademacher complexity: empirical vs theoretical')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

**출력 예시**:
```
n= 10: R̂_S(linear) = 0.6248, Upper bound = 0.8934
n= 20: R̂_S(linear) = 0.4221, Upper bound = 0.6180
n= 50: R̂_S(linear) = 0.2410, Upper bound = 0.3785
n=100: R̂_S(linear) = 0.1634, Upper bound = 0.2637
n=200: R̂_S(linear) = 0.1141, Upper bound = 0.1854
```

→ 경험적 Rademacher가 이론 상한 내에 있고, 둘 다 $1/\sqrt{n}$ 스케일 감소를 확인.

### 실험 2: 다항식 함수족 vs 선형 함수족의 비교

```python
# 다항식: f_w(x) = w^T φ(x), φ(x) = [1, x_1, x_2, ..., x_d, x_1^2, x_1 x_2, ...]
def polynomial_features(X, degree=2):
    """X (n, d) -> Φ (n, n_features)"""
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree, include_bias=False)
    return poly.fit_transform(X)

def rademacher_poly(X, degree, B=1.0, n_rademacher=1000):
    Phi = polynomial_features(X, degree)
    n, d_phi = Phi.shape
    vals = []
    for _ in range(n_rademacher):
        sigma = rng.choice([-1, 1], size=n)
        weighted_sum = (sigma[:, None] * Phi).sum(axis=0)
        norm = np.linalg.norm(weighted_sum)
        vals.append(B * norm / n)
    return np.mean(vals)

# 비교: degree 1, 2, 3
n = 100
X = sample_data(n, d=5)

for degree in [1, 2, 3]:
    rad = rademacher_poly(X, degree, B=1.0, n_rademacher=500)
    print(f"Degree {degree}: R̂_S = {rad:.4f}")

# → Degree가 높을수록 Rademacher 복잡도 증가 (더 표현력 높은 함수족)
```

### 실험 3: 함수족 크기에 따른 Rademacher (유한 경우)

```python
# 유한 함수족: 임계값 분류기 H = {sign(x - θ) : θ ∈ {θ_1, ..., θ_K}}
def rademacher_threshold_finite(x_samples, thresholds, n_rademacher=1000):
    """
    H = {sign(x - θ) : θ ∈ thresholds}
    R̂_S(H) = E_σ[max_θ (1/n) Σ σ_i sign(x_i - θ)]
    """
    n = len(x_samples)
    vals = []
    for _ in range(n_rademacher):
        sigma = rng.choice([-1, 1], size=n)
        max_corr = 0
        for theta in thresholds:
            f_vals = np.sign(x_samples - theta).astype(float)
            corr = np.abs(np.sum(sigma * f_vals)) / n  # |·| 취함 (양수로 정규화)
            max_corr = max(max_corr, corr)
        vals.append(max_corr)
    return np.mean(vals)

# 1D data
x = rng.uniform(0, 1, 100)
thresholds_list = [
    np.linspace(0, 1, 5),
    np.linspace(0, 1, 20),
    np.linspace(0, 1, 100),
]

for thetas in thresholds_list:
    rad = rademacher_threshold_finite(x, thetas, n_rademacher=500)
    K = len(thetas)
    print(f"|H| = {K:3d}: R̂_S ≈ {rad:.4f}")

# → |H| 커질수록 Rademacher 증가 (Massart's lemma, Ch5-04에서 정량화)
```

---

## 🔗 ML 알고리즘 연결

| 알고리즘 | 함수족 $\mathcal{F}$ | Rademacher 유형 | 출처 |
|---------|------------------|---------------|------|
| **SVM (linear)** | $\{w^\top x : \|w\| \leq B\}$ | Linear Rademacher | Ch5-05 |
| **Kernel SVM** | RKHS ball $\|f\|_\mathcal{H} \leq B$ | Kernel Rademacher $\approx \sqrt{\text{tr}(K)/n}$ | Ch5-05 |
| **Neural Net** | $f_\theta$ with bounded norms | Bartlett-Mendelson bound $\prod \|W_l\| / \sqrt{n}$ | Ch5-06 |
| **Ridge Regression** | $\{w : \|w\| \leq B\}$ | Linear, then + margin | Ch5-05 |

**일반화 경계의 핵심**: 
$$\sup_{h \in \mathcal{H}} |L_\mathcal{D}(h) - L_S(h)| \leq 2 \mathcal{R}_n(\ell \circ \mathcal{H}) + O\left(\sqrt{\frac{\log(1/\delta)}{n}}\right)$$
(Ch5-02에서 완전 증명)

---

## ⚖️ 가정과 한계

1. **기대값의 존재**: $f(X) \in \mathbb{R}$ bounded 또는 적분가능 가정. unbounded $f$면 Rademacher 정의 자체가 발산.
2. **유한 vs 무한 함수족**: 정의는 일반적이지만, Rademacher 계산은 $\mathcal{F}$가 **compact**(또는 유한)일 때 가장 clean. 무한 $\mathcal{F}$는 covering argument(Ch4-06의 $\epsilon$-net) 필요.
3. **데이터 의존성의 대가**: $\hat{\mathcal{R}}_S$는 $S$에 의존하므로 **또 다른 uniform convergence 논증**이 필요(McDiarmid, Ch5-02). "무한 후회(infinite regret)"를 피해야 함.
4. **Loss 함수의 범위**: Contraction lemma(Ch5-03)를 쓸 때 loss의 Lipschitz 상수가 중요. surrogate loss가 unbounded면 bound가 vacuous.
5. **계산 복잡도**: $\hat{\mathcal{R}}_S$ 계산 자체는 $\sup_f$를 포함하므로 일반적으로 NP-hard. Monte Carlo 추정(위 코드)은 근사일 뿐.

---

## 📌 핵심 정리

- **Rademacher 변수**: $\sigma_i \in \{\pm 1\}$ 균등분포. "동전 던지기" 즉, 노이즈 생성.
- **경험적 Rademacher 복잡도** $\hat{\mathcal{R}}_S(\mathcal{F})$: $\mathcal{F}$가 $S$ 위의 랜덤 라벨 $\sigma$을 얼마나 잘 학습하는가의 기대값. **데이터 의존적**.
- **(모집단) Rademacher 복잡도** $\mathcal{R}_n(\mathcal{F})$: 경험적 버전을 샘플 분포에 대해 기대값. **이론 분석**에 쓰임.
- **기본 성질**: 비음성, singleton=0, 스칼라 배, 삼각부등식.
- **Gaussian과의 관계**: 1.5배 이내 범위로 동일하게 복잡도 측정. Rademacher가 더 discrete해서 practical.
- **VC 대비 장점**: 데이터 의존적 + tighter bound + loss 변환 용이(contraction lemma).

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> $\mathcal{F} = \{f_1, f_2, f_3\}$이 3개 함수로 이루어진 유한 함수족일 때, $\hat{\mathcal{R}}_S(\mathcal{F})$를 직접 계산하는 과정을 보여라. 특히 $n=2$ 샘플과 $\sigma \in \{\pm 1\}^2$의 모든 경우를 나열하라.</summary>

<br/>

**해설**. $S = (x_1, x_2)$, $\mathcal{F} = \{f_1, f_2, f_3\}$일 때,
$$\hat{\mathcal{R}}_S(\mathcal{F}) = \mathbb{E}_\sigma\left[\max_{j \in \{1,2,3\}} \frac{\sigma_1 f_j(x_1) + \sigma_2 f_j(x_2)}{2}\right].$$

$\sigma$의 모든 경우($2^2 = 4$):
- $(\sigma_1, \sigma_2) = (+1, +1)$: $\max_j \frac{f_j(x_1) + f_j(x_2)}{2}$
- $(\sigma_1, \sigma_2) = (+1, -1)$: $\max_j \frac{f_j(x_1) - f_j(x_2)}{2}$
- $(\sigma_1, \sigma_2) = (-1, +1)$: $\max_j \frac{-f_j(x_1) + f_j(x_2)}{2}$
- $(\sigma_1, \sigma_2) = (-1, -1)$: $\max_j \frac{-f_j(x_1) - f_j(x_2)}{2} = -\min_j \frac{f_j(x_1) + f_j(x_2)}{2}$

기대값은 이 4개 경우의 평균. 각 경우에 $1/4$ 가중. $\square$

이 과정은 Rademacher가 "모든 가능한 라벨 패턴에서의 corr" 평균임을 명확히 보여줌.

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> 정리 5.1의 삼각부등식 $\hat{\mathcal{R}}_S(\mathcal{F}_1 + \mathcal{F}_2) \leq \hat{\mathcal{R}}_S(\mathcal{F}_1) + \hat{\mathcal{R}}_S(\mathcal{F}_2)$를 증명하되, 왜 이것이 "convexity"와 관련이 있는지 논의하라.</summary>

<br/>

**해설**. 
$$\hat{\mathcal{R}}_S(\mathcal{F}_1 + \mathcal{F}_2) = \mathbb{E}_\sigma\left[\sup_{f_1 \in \mathcal{F}_1, f_2 \in \mathcal{F}_2} \sum_i \sigma_i (f_1(x_i) + f_2(x_i))\right].$$

삼각부등식에 의해
$$\sup_{f_1, f_2} \sum \sigma_i (f_1 + f_2) \leq \sup_{f_1} \sum \sigma_i f_1 + \sup_{f_2} \sum \sigma_i f_2.$$

**Convexity와의 연결**: Rademacher $\hat{\mathcal{R}}$는 함수족의 **Minkowski sum**에 대해 subadditive(부등식). 이것은 **convex analysis의 gauge function** 또는 **norm**처럼 행동한다. 실제로 특정 조건에서 $\hat{\mathcal{R}}$는 함수공간 위의 norm을 정의할 수 있다. $\square$

이 성질이 Ch5-05에서 선형 함수족의 Rademacher 계산에 핵심적으로 쓰임.

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> SVM과 Neural Net에서 왜 **norm 제약** ($\|w\| \leq B$ 또는 $\|W_l\|_F \leq M_l$)이 일반화 경계에 직접 나타나는가? Rademacher 관점에서 설명하라.</summary>

<br/>

**해설**. 선형 함수족 $\mathcal{F} = \{w^\top x : \|w\| \leq B\}$의 Rademacher는 (Ch5-05에서)
$$\mathcal{R}_n(\mathcal{F}) \leq B \cdot \max_i \|x_i\| / \sqrt{n}.$$

**직관**: Norm 제약 $\|w\| \leq B$는 $\mathcal{F}$의 "크기"를 제한한다 — 표현력을 줄인다. 더 작은 $\mathcal{F}$ → Rademacher 복잡도 ↓ → 과적합 위험 ↓.

**SVM**: margin 최대화 = $\|w\|$ 최소화 = $\mathcal{R}_n(\{w \cdot x\})$ 최소화. 즉, norm 제약이 "implicit regularization".

**NN** (Bartlett-Mendelson): 각 층의 norm $\|W_l\|_F$의 곱 $\prod_l \|W_l\|_F$이 네트워크의 Rademacher에 나타남. 파라미터 수보다는 **norm의 규모**가 일반화를 결정한다 — 이것이 "고전 VC는 vacuous하지만 norm-based bound는 의미있다"는 관찰(Ch4-07·Ch5-06).

이 인사이트가 현대 DL의 **norm regularization**(weight decay, spectral normalization)을 정당화한다. $\square$

</details>

---

<div align="center">

◀ [이전: 07. VC 경계의 한계](../ch4-vc-dimension/07-vc-limits.md) | [📚 README](../README.md) | [다음: 02. Rademacher 기반 일반화 경계 ▶](./02-rademacher-generalization.md)

</div>
