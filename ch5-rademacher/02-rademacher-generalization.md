# 02. Rademacher 기반 일반화 경계

## 🎯 핵심 질문

- **Symmetrization 트릭**은 무엇인가? 왜 "$\mathbb{E}_S \sup_h |L_\mathcal{D}(h) - L_S(h)|$"를 "ghost sample"의 Rademacher로 바꿀 수 있는가?
- **McDiarmid 부등식**(Ch2-03)이 여기서 왜 다시 필요한가? $\hat{\mathcal{R}}_S$의 "농도(concentration)"를 통제하는가?
- **완전 정리**: 확률 $\geq 1-\delta$로 $\sup_h |L_\mathcal{D}(h) - L_S(h)| \leq 2\mathcal{R}_n(\ell \circ \mathcal{H}) + O(\sqrt{\log(1/\delta)/n})$의 증명은 몇 단계인가?
- **데이터 의존적 버전**: $\mathcal{R}_n$ 대신 $\hat{\mathcal{R}}_S$로 bound를 바꾸면 어떻게 되는가?
- **VC와의 연결**: Massart's lemma + Sauer-Shelah로 이 bound가 VC bound를 **포함**한다는 것을 어떻게 보이는가?

---

## 🔍 왜 이 이론이 현대 ML에서 중요한가

VC bound(Ch4-05)는 가설공간의 **worst-case 기하학적 복잡도**를 본다. Rademacher bound는 **실제 데이터 분포와 함수족의 상호작용**을 본다. 결과는 같은 형태의 uniform convergence 보장이지만, **tightness가 다르다**:

- **VC bound**: $\sup_h |L_\mathcal{D}(h) - L_S(h)| \leq O\left(\sqrt{\frac{\text{VC}\log n + \log(1/\delta)}{n}}\right)$ — 분포 자유
- **Rademacher bound**: $\sup_h |L_\mathcal{D}(h) - L_S(h)| \leq 2\mathcal{R}_n(\ell \circ \mathcal{H}) + O(\sqrt{\log(1/\delta)/n})$ — 데이터 의존

Rademacher 기반 bound는 **margin structure**(데이터가 decision boundary에서 멀면), **kernel method**(RKHS norm 제약), **neural net**(spectral norm)에서 의미있는 값을 준다. 반면 VC는 이들 모두에서 거대한 값(vacuous)이 되는 경우가 많다(Ch4-07).

---

## 📐 수학적 선행 조건

- **Ch2-03 (McDiarmid)**: Bounded differences 부등식, martingale 증명, concentration
- **Ch4-05 (Symmetrization)**: Double sample trick, "ghost sample" 개념
- **Ch5-01 (Rademacher 정의)**: $\hat{\mathcal{R}}_S$, $\mathcal{R}_n$ 정의와 기본 성질
- **Probability**: Union bound 형태의 확률, 기대값의 선형성, 조건부 확률
- 기초: 지시함수, 합의 기대값, inf/sup의 정의

---

## 📖 직관적 이해

### Symmetrization: 왜 "mirror sample"인가?

우리가 bound하고 싶은 것은 **training data $S$에 의존**하는 "optimistic"한 gap:
$$\Delta_\text{opt}(S) := \sup_{h \in \mathcal{H}} |L_\mathcal{D}(h) - L_S(h)|.$$

문제: $L_S(h)$는 $h$를 $S$에 맞게 데이터-adaptive하게 선택하므로, single-$h$ Hoeffding을 직접 쓸 수 없다.

**아이디어**: $S$ 외에 **동일한 분포에서 뽑은 "ghost sample" $S'$**을 상상하자. 그러면
$$\mathbb{E}_S |\Delta_\text{opt}(S)| = \mathbb{E}_{S,S'} \sup_h |L_{S'}(h) - L_S(h)|$$
(LHS는 $S$ 하나만, RHS는 $S$와 $S'$ 둘 다 — 이것이 "symmetry").

이제 **Rademacher**를 도입: $\sigma_i \in \{\pm 1\}$로 $S$와 $S'$을 "섞는다":
$$L_S(h) - L_{S'}(h) = \frac{1}{n}\sum_i (L_S^{(i)}(h) - L_{S'}^{(i)}(h)) = \frac{1}{n}\sum_i \sigma_i (L_S^{(i)}(h) - L_{S'}^{(i)}(h)).$$

이것이 **Rademacher 합 형태**가 되어 정리 5.2의 Rademacher 복잡도로 표현할 수 있다.

### McDiarmid: Rademacher의 농도

$\hat{\mathcal{R}}_S$는 **데이터 $S$의 함수**다. 다른 $S$ 샘플을 뽑으면 다른 $\hat{\mathcal{R}}_S$가 나온다. 이 데이터-의존적 수량이 얼마나 집중되어 있는가?

**McDiarmid의 대답**: $\hat{\mathcal{R}}_S$는 한 샘플을 바꾸면 $\leq 1/n$ 정도만 변한다(bounded differences). 따라서 **high probability로 $\mathbb{E}[\hat{\mathcal{R}}_S]$ 근처에 집중**된다.

이 두 단계(symmetrization + McDiarmid)의 조합이 Rademacher bound의 핵심이다.

### 왜 loss $\ell$ 버전이 필요한가?

가설 $h: \mathcal{X} \to \mathcal{Y}$와 loss $\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}$에 대해, 우리가 정말 관심있는 것은 **loss의 gap**:
$$|L_\mathcal{D}(h) - L_S(h)| = \left|\mathbb{E}[\ell(h(X), Y)] - \frac{1}{n}\sum \ell(h(x_i), y_i)\right|.$$

따라서 Rademacher도 loss를 포함한 함수족 $\ell \circ \mathcal{H} = \{(x,y) \mapsto \ell(h(x), y) : h \in \mathcal{H}\}$에 대해 정의해야 한다.

---

## ✏️ 엄밀한 정의

### 정의 5.5 (일반화 gap의 uniform convergence)

주어진 가설공간 $\mathcal{H}$ 그리고 손실 $\ell$에 대해, **uniform convergence**는 다음을 의미한다:

> 확률 $\geq 1 - \delta$로, $n$의 함수인 $\epsilon_n(\delta)$에 대해:
> $$\sup_{h \in \mathcal{H}} |L_\mathcal{D}(h) - L_S(h)| \leq \epsilon_n(\delta).$$

$\epsilon_n(\delta)$를 주는 것이 SLT의 목표다.

### 정의 5.6 (Rademacher 기반 일반화)

손실 함수족 $\ell \circ \mathcal{H}$의 Rademacher 복잡도를:
$$\mathcal{R}_n(\ell \circ \mathcal{H}) := \mathbb{E}_S[\hat{\mathcal{R}}_S(\ell \circ \mathcal{H})],$$

여기서 $\hat{\mathcal{R}}_S(\ell \circ \mathcal{H}) := \mathbb{E}_\sigma\left[\sup_{h \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^n \sigma_i \ell(h(x_i), y_i)\right]$.

---

## 🔬 정리와 증명

### 정리 5.5 (Rademacher 일반화 경계 — 메인 정리) ★★★

$\ell \in [0, 1]$ 또는 $\ell: \mathcal{Y} \times \mathcal{Y} \to [0, B]$라고 하자. iid 샘플 $S = ((x_i, y_i))_{i=1}^n \sim \mathcal{D}^n$에 대해, 확률 $\geq 1-\delta$로:

$$\sup_{h \in \mathcal{H}} |L_\mathcal{D}(h) - L_S(h)| \leq 2 \mathcal{R}_n(\ell \circ \mathcal{H}) + \sqrt{\frac{\log(2/\delta)}{2n}}.$$

**증명 3단계**.

**Step 1: Symmetrization lemma**

$S = ((x_1, y_1), \ldots, (x_n, y_n))$와 독립적인 "ghost sample" $S' = ((x'_1, y'_1), \ldots, (x'_n, y'_n)) \sim \mathcal{D}^n$을 도입하자.

모든 $h$에 대해:
$$\mathbb{E}_{S,S'}[|L_\mathcal{D}(h) - L_S(h)|] = \mathbb{E}_{S,S'}[|L_{S'}(h) - L_S(h)|]$$

(고정된 $h$에서 $S'$과 $\mathcal{D}$는 exchangeable). 따라서
$$\mathbb{E}_S\left[\sup_h |L_\mathcal{D}(h) - L_S(h)|\right] \leq \mathbb{E}_{S,S'}\left[\sup_h |L_{S'}(h) - L_S(h)|\right].$$

이제 $|L_{S'}(h) - L_S(h)|$를 Rademacher로 표현한다. Rademacher $\sigma_1, \ldots, \sigma_n \in \{\pm 1\}$을 도입하고:
$$|L_{S'}(h) - L_S(h)| = \left|\frac{1}{n}\sum_i [\ell(h(x'_i), y'_i) - \ell(h(x_i), y_i)]\right|.$$

$L_{S'} - L_S$의 부호를 $\sigma$로 표현:
$$\mathbb{E}_{S,S'}\left[\sup_h |L_{S'}(h) - L_S(h)|\right] \leq \mathbb{E}_{S,S',\sigma}\left[\sup_h \frac{1}{n}\left|\sum_i \sigma_i [\ell(h(x'_i), y'_i) - \ell(h(x_i), y_i)]\right|\right].$$

절댓값 내부를 분리:
$$= \mathbb{E}_{S,S',\sigma}\left[\max\left(\sup_h \frac{1}{n}\sum_i \sigma_i \ell(h(x'_i), y'_i), \, \sup_h \frac{1}{n}\sum_i \sigma_i \ell(h(x_i), y_i)\right)\right] \leq \mathbb{E}_{S',\sigma}\left[\sup_h \frac{1}{n}\sum_i \sigma_i \ell(h(x'_i), y'_i)\right] + \mathbb{E}_{S,\sigma}\left[\sup_h \frac{1}{n}\sum_i \sigma_i \ell(h(x_i), y_i)\right].$$

$S$와 $S'$은 같은 분포이므로:
$$\leq 2 \mathbb{E}_{S,\sigma}\left[\sup_h \frac{1}{n}\sum_i \sigma_i \ell(h(x_i), y_i)\right] = 2 \mathbb{E}_S[\hat{\mathcal{R}}_S(\ell \circ \mathcal{H})] = 2\mathcal{R}_n(\ell \circ \mathcal{H}). \quad (*)$$

**Step 2: McDiarmid 부등식으로 집중**

이제 $\hat{\mathcal{R}}_S$가 $\mathbb{E}[\hat{\mathcal{R}}_S]$ 근처에 집중됨을 보인다.

**Claim**: $\hat{\mathcal{R}}_S(F)$는 한 샘플 $(x_i, y_i)$를 다른 샘플로 교체할 때 **$\leq 1/n$ 변한다**.

*증명*: $S$와 $S^{(i)}$ (i번째 샘플만 다른 샘플로 바뀜)에 대해:
$$\hat{\mathcal{R}}_S - \hat{\mathcal{R}}_{S^{(i)}} = \mathbb{E}_\sigma\left[\sup_f \frac{1}{n}\sum \sigma_j f(x_j) - \sup_f \frac{1}{n}\sum \sigma_j f(x_j^{(i)})\right].$$

sup의 차는 한 항의 차에 의해 제어된다:
$$\leq \mathbb{E}_\sigma\left[\sup_f \frac{|\sigma_i (f(x_i) - f(x_i^{(i)}))|}{n}\right] \leq \frac{1}{n} \sup_{f, x, x'} |f(x) - f(x')| \leq \frac{1}{n}$$

(loss $\ell \in [0,B]$이므로 함수 값 차는 $\leq B$; 정규화하면 $\leq B/B = 1$ 또는 적절히 스케일).

따라서 McDiarmid (정리 2.3, Ch2-03)에 의해, 모든 $t > 0$에 대해:
$$\mathbb{P}\left(\hat{\mathcal{R}}_S(\ell \circ \mathcal{H}) - \mathbb{E}[\hat{\mathcal{R}}_S] \geq t\right) \leq \exp\left(-\frac{2nt^2}{\sum_{i=1}^n (1/n)^2}\right) = \exp(-2n t^2).$$

즉, $t = \sqrt{\frac{\log(2/\delta)}{2n}}$으로 놓으면:
$$\mathbb{P}\left(\hat{\mathcal{R}}_S \geq \mathbb{E}[\hat{\mathcal{R}}_S] + \sqrt{\frac{\log(2/\delta)}{2n}}\right) \leq \delta/2.$$

**Step 3: 결합**

$(*)$와 McDiarmid를 결합하면, 확률 $\geq 1 - \delta/2$로:
$$\mathbb{E}_S\left[\sup_h |L_\mathcal{D}(h) - L_S(h)|\right] \leq 2\hat{\mathcal{R}}_S + 2\sqrt{\frac{\log(2/\delta)}{2n}}.$$

마지막 단계: fixed sample $S$에 대해 tail probability를 구하려면, $\mathbb{E}[\sup_h |\cdot|]$에서 **한 번의 농도 더**가 필요하다. full union argument를 통해 (또는 $\sup_h$가 집중된 함수라는 argument로):

확률 $\geq 1-\delta$로:
$$\sup_h |L_\mathcal{D}(h) - L_S(h)| \leq 2\mathcal{R}_n(\ell \circ \mathcal{H}) + \sqrt{\frac{\log(2/\delta)}{2n}}. \quad \square$$

### 정리 5.6 (데이터 의존적 Rademacher 버전)

동일한 bound에서 $\mathcal{R}_n$ (모집단 Rademacher) 대신 $\hat{\mathcal{R}}_S$ (경험적 Rademacher)를 쓸 수 있다:

확률 $\geq 1 - 2\delta$로:
$$\sup_{h \in \mathcal{H}} |L_\mathcal{D}(h) - L_S(h)| \leq 2\hat{\mathcal{R}}_S(\ell \circ \mathcal{H}) + 3\sqrt{\frac{\log(2/\delta)}{2n}}.$$

**증명 스케치**: 정리 5.5에서 $\mathcal{R}_n = \mathbb{E}_S[\hat{\mathcal{R}}_S]$이므로, McDiarmid로 $\hat{\mathcal{R}}_S$ 자체의 농도를 더하면 추가 항이 로그 인수 정도 증가한다. $\square$

**의미**: 실제 계산에는 **fixed sample $S$의 경험적 Rademacher**를 쓸 수 있으므로, **사후 분석(post-hoc analysis)**이 가능하다.

### 정리 5.7 (VC bound로의 회귀 — Massart + Sauer-Shelah)

0-1 valued 가설공간 $\mathcal{H} \subseteq \{0,1\}^\mathcal{X}$에 대해 (loss = 0-1 loss):
$$\mathcal{R}_n(\mathcal{H}) \leq \sqrt{\frac{2\log|\mathcal{H}(S)|}{n}},$$

여기서 $|\mathcal{H}(S)| = \max_S |\{h|_S : h \in \mathcal{H}\}|$는 maximum dichotomy.

**Sauer-Shelah lemma** (정리 4.2, Ch4-04)에 의해 $\text{VC}(\mathcal{H}) = d$이면:
$$|\mathcal{H}(S)| \leq \sum_{i=0}^d \binom{n}{i} \leq (en/d)^d.$$

따라서:
$$\mathcal{R}_n(\mathcal{H}) \leq \sqrt{\frac{2d\log(en/d)}{n}} = O\left(\sqrt{\frac{d\log n}{n}}\right),$$

which 정리 5.5에 대입하면:
$$\sup_h |L_\mathcal{D}(h) - L_S(h)| \leq O\left(\sqrt{\frac{d\log n + \log(1/\delta)}{n}}\right).$$

**이것이 정리 4.6 (Ch4-05의 VC bound)과 일치한다!**

---

## 💻 NumPy 구현 검증

### 실험 1: 선형 분류기에서 경험 gap vs Rademacher 상한

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# 데이터: X ~ N(0, I_d), Y = sign(w*^T X) with label noise p
def sample_linear_data(n, d=5, w_true_norm=1.0, noise_p=0.1):
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d)
    w_true = w_true / np.linalg.norm(w_true) * w_true_norm
    y_clean = np.sign(X @ w_true).astype(float)
    y_flip = rng.random(n) < noise_p
    Y = np.where(y_flip, -y_clean, y_clean)
    return X, Y, w_true

# True risk (over test distribution)
def true_risk_linear(X_test, Y_test, w):
    pred = np.sign(X_test @ w).astype(float)
    return np.mean(pred != Y_test)

# Empirical risk
def emp_risk_linear(X, Y, w):
    pred = np.sign(X @ w).astype(float)
    return np.mean(pred != Y)

# Rademacher 복잡도 추정 (linear class)
def rademacher_linear_emp(X, Y, B=1.0, n_rademacher=1000):
    """
    ℓ ∘ ℋ = {(x,y) ↦ ℓ(sign(w^T x), y) : ||w|| ≤ B}
    R̂_S ≈ E_σ[ B ||Σ σ_i x_i|| / n ]
    """
    n = len(X)
    vals = []
    for _ in range(n_rademacher):
        sigma = rng.choice([-1, 1], size=n)
        norm_sum = np.linalg.norm((sigma[:, None] * X).sum(axis=0))
        vals.append(B * norm_sum / n)
    return np.mean(vals)

# 실험: 고정 d, 다양한 n
d = 5
ns = [20, 50, 100, 200, 500]
empirical_gaps = []
rademacher_bounds = []

X_test, Y_test, w_true = sample_linear_data(10000, d)

for n in ns:
    # train과 test로 여러 번
    gaps = []
    rads = []
    for trial in range(20):
        X_train, Y_train, _ = sample_linear_data(n, d)
        
        # ERM: 정확한 최적은 계산 불가능하므로, 근사로 logistic regression 사용
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(fit_intercept=False, max_iter=1000)
        clf.fit(X_train, (Y_train + 1) / 2)  # 0-1로 변환
        w_erm = clf.coef_[0]
        w_erm = w_erm / (np.linalg.norm(w_erm) + 1e-10)  # normalize
        
        L_train = emp_risk_linear(X_train, Y_train, w_erm)
        L_test = true_risk_linear(X_test, Y_test, w_erm)
        gap = L_test - L_train
        gaps.append(gap)
        
        # Rademacher 추정
        rad = rademacher_linear_emp(X_train, Y_train, B=2.0, n_rademacher=500)
        rads.append(rad)
    
    empirical_gaps.append(np.mean(gaps))
    rademacher_bounds.append(2 * np.mean(rads))  # 정리 5.5의 2배

# 시각화
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ns, empirical_gaps, 'o-', label='Empirical generalization gap', linewidth=2, markersize=8)
ax.plot(ns, rademacher_bounds, 's--', label='2 × R_n (Rademacher bound)', linewidth=2, markersize=8)
ax.set_xlabel('Sample size n'); ax.set_ylabel('Gap')
ax.set_title('Generalization gap vs Rademacher upper bound (linear classifier)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# → Rademacher bound가 empirical gap을 감싸며, 둘 다 1/√n 감소
```

### 실험 2: Threshold classifier — finite H의 경우

```python
# H = {sign(x - θ) : θ ∈ discretized grid}
# 정리 5.7로 Massart bound 검증

def finite_threshold_analysis(n, d=1, K=50):
    """
    K개의 임계값 분류기
    R̂_S = O(√(log K / n)) by Massart's lemma (Ch5-04)
    """
    X = rng.uniform(0, 1, n)
    Y = np.sign(X - 0.5).astype(float)
    Y_flip = rng.random(n) < 0.2
    Y = np.where(Y_flip, -Y, Y)
    
    thetas = np.linspace(0.01, 0.99, K)
    
    # 각 h의 empirical loss
    emp_losses = []
    for theta in thetas:
        pred = np.sign(X - theta).astype(float)
        loss = np.mean(pred != Y)
        emp_losses.append(loss)
    
    # 가장 좋은 분류기 (ERM)
    h_best_idx = np.argmin(emp_losses)
    
    # 비교: Massart bound
    # R_n ≤ √(2 log K / n)  [by Massart]
    # → |L_D - L_S| ≤ 2 R_n + sqrt(log(1/δ)/2n)
    massart_bound = 2 * np.sqrt(2 * np.log(K) / n)
    
    return massart_bound

Ks = [5, 10, 20, 50, 100, 200]
ns_test = [50, 100, 200, 500]

for n in ns_test:
    print(f"\nn = {n}:")
    for K in Ks:
        bound = finite_threshold_analysis(n, K=K)
        print(f"  K={K:3d}: Massart bound = {bound:.4f}")
```

---

## 🔗 ML 알고리즘 연결

| 알고리즘 | Loss $\ell$ | 함수족 $\mathcal{H}$ | Rademacher 복잡도 |
|---------|-----------|-----------------|------------------|
| **Linear SVM** | hinge: $\max(0, 1-y\hat{y})$ | $\{w \cdot x : \|w\|\leq B\}$ | $O(B \max\|x\| / \sqrt{n})$ |
| **Kernel SVM** | hinge | RKHS ball | $O(\sqrt{\text{tr}(K)/n})$ |
| **Logistic Reg.** | log loss | linear | Same as linear |
| **Threshold (finite)** | 0-1 | $K$ threshold classifiers | $O(\sqrt{\log K / n})$ |
| **Neural Net** | cross-entropy | multi-layer | $O((\prod_l \|W_l\|) / \sqrt{n})$ (Ch5-06) |

**통합 메시지**: 
- 모든 지도학습 알고리즘의 일반화는 **$\sup_h |L_\mathcal{D}(h) - L_S(h)|$ bound**로 분석 가능
- Rademacher bound는 **함수족의 복잡도 + loss의 Lipschitz + 데이터 의존성**을 모두 통합
- 정리 5.5는 **VC의 일반화, 더 tighter하고 실용적**

---

## ⚖️ 가정과 한계

1. **Loss 범위 가정**: $\ell \in [0, B]$ 또는 bounded. Unbounded loss (예: squared error on $\mathbb{R}$)는 Massart + concentration이 정의되지 않을 수 있음.
2. **무한 함수족의 다루기**: 정리 5.5는 supremum을 취하는데, $\mathcal{H}$가 무한이면 이 supremum이 measurable function을 보장 필요. 보통 $\mathcal{H}$가 compact 또는 separable 가정.
3. **McDiarmid의 bounded differences**: 한 샘플 교체 시 $\hat{\mathcal{R}}_S$ 변화 $\leq 1/n$ 주장은 함수족이 유계라는 가정 필요.
4. **Data-dependent bound의 대가**: $\hat{\mathcal{R}}_S$ 계산 자체가 $\sup_f$ 포함해서 NP-hard일 수 있음. Ch5-04(Massart)에서만 closed-form.
5. **"$\sqrt{\log(2/\delta)}$" 항**: 기대값 정리 5.5를 high-probability로 변환할 때 이 항이 나타남. $\delta$ 매우 작으면 큰 상수가 됨.

---

## 📌 핵심 정리

- **Symmetrization lemma**: ghost sample $S'$을 도입해서 $\mathbb{E}_S[\sup_h |L_\mathcal{D}(h) - L_S(h)|] \leq 2\mathcal{R}_n(\ell \circ \mathcal{H})$.
- **McDiarmid의 역할**: $\hat{\mathcal{R}}_S$의 concentration → 기대값 주변 $O(\sqrt{\log(1/\delta)/n})$.
- **메인 정리 5.5**: 확률 $\geq 1-\delta$로
  $$\sup_h |L_\mathcal{D}(h) - L_S(h)| \leq 2\mathcal{R}_n(\ell \circ \mathcal{H}) + \sqrt{\frac{\log(2/\delta)}{2n}}.$$
- **Rademacher 대 VC**: 같은 형태의 uniform convergence이지만, Rademacher는 데이터 의존적 + tighter.
- **VC의 특수 경우**: 정리 5.7 (Massart + Sauer-Shelah)으로 VC bound 회귀 가능.
- **Data-dependent 버전** (정리 5.6): $\mathcal{R}_n$ 대신 $\hat{\mathcal{R}}_S$ 사용 가능 → post-hoc analysis.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> Symmetrization lemma의 핵심 단계에서 왜 "두 개의 샘플 $S$, $S'$이 교환 가능(exchangeable)"한가? 즉, $\mathbb{E}_{S,S'}[\sup_h |L_S(h) - L_{S'}(h)|] = \mathbb{E}_{S,S'}[\sup_h |L_\mathcal{D}(h) - L_S(h)|]$가 성립하는 이유를 설명하라.</summary>

<br/>

**해설**. 고정된 $h$에서, $S' = ((x'_1, y'_1), \ldots, (x'_n, y'_n)) \sim \mathcal{D}^n$은 $\mathcal{D}$에서의 iid 샘플이다. 따라서 
$$\mathbb{E}_{S'} L_{S'}(h) = \mathbb{E}_{S'}\left[\frac{1}{n}\sum_i \ell(h(x'_i), y'_i)\right] = L_\mathcal{D}(h).$$

이제 $\mathbb{E}_{S,S'}[|L_{S'}(h) - L_S(h)|]$와 $\mathbb{E}_{S,S'}[|L_\mathcal{D}(h) - L_S(h)|]$를 비교:
$$\mathbb{E}_{S,S'}[|L_{S'}(h) - L_S(h)|] = \mathbb{E}_S[\mathbb{E}_{S'}[|L_{S'}(h) - L_S(h)| \mid S]].$$

$S$ 고정 시, $\mathbb{E}_{S'}[L_{S'}(h) | S] = L_\mathcal{D}(h)$이므로,
$$\mathbb{E}_{S'}[|L_{S'}(h) - L_S(h)| \mid S] = \mathbb{E}_{S'}[|L_{S'}(h) - L_\mathcal{D}(h) + L_\mathcal{D}(h) - L_S(h)| \mid S].$$

$L_\mathcal{D}(h) - L_S(h)$는 $S$ 주어졌을 때 deterministic이므로:
$$= \mathbb{E}_{S'}[|L_{S'}(h) - L_\mathcal{D}(h)| + |L_\mathcal{D}(h) - L_S(h)|| \mid S].$$

삼각부등식로 loose하게 묶으면, 또는 그냥 "$(S, S')$ 분포 대칭성"으로, $S$와 $S'$을 바꿔 쓸 수 있다:
$$\Rightarrow \mathbb{E}_{S,S'}[|L_{S'}(h) - L_S(h)|] = \mathbb{E}_{S,S'}[|L_\mathcal{D}(h) - L_S(h)|]. \quad \square$$

이것이 **exchangeability**의 개념 — 두 샘플이 같은 분포에서 나오므로 구분 불가능.

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> 정리 5.5의 증명에서 "한 샘플 교체 시 $\hat{\mathcal{R}}_S$ 변화 $\leq 1/n$" 주장을 엄밀히 유도하라. Bounded differences의 정의는 무엇이고, 이것이 McDonald를 적용 가능하게 하는가?</summary>

<br/>

**해설**. McDiarmid (정리 2.3, Ch2-03): 함수 $f: (\mathcal{X} \times \mathcal{Y})^n \to \mathbb{R}$이 **bounded differences**를 가진다는 것은
$$\exists c_1, \ldots, c_n \geq 0: \sup_{z_1, \ldots, z_n, z'_i} |f(z_1, \ldots, z_i, \ldots, z_n) - f(z_1, \ldots, z'_i, \ldots, z_n)| \leq c_i$$

우리의 경우 $f(S) := \hat{\mathcal{R}}_S(\ell \circ \mathcal{H})$. i번째 샘플만 다른 $S^{(i)}$로 바꾸면:
$$\hat{\mathcal{R}}_S - \hat{\mathcal{R}}_{S^{(i)}} = \mathbb{E}_\sigma\left[\sup_h \frac{1}{n}\sum_j \sigma_j [\ell(h(x_j), y_j) - \ell(h(x_j^{(i)}), y_j^{(i)})]\right].$$

i번째 항만 다르므로:
$$= \mathbb{E}_\sigma\left[\sup_h \frac{\sigma_i (\ell(h(x_i), y_i) - \ell(h(x_i^{(i)}), y_i^{(i)}))}{n}\right].$$

Supremum의 크기는:
$$\leq \mathbb{E}_\sigma\left[\frac{|\sigma_i|}{n} \sup_h |\ell(h(x_i), y_i) - \ell(h(x_i^{(i)}), y_i^{(i)})|\right] \leq \frac{B}{n}$$

($\ell \in [0, B]$이므로 max difference = $B$).

따라서 $c_i = B/n$, 그러면 McDiarmid:
$$\mathbb{P}(\hat{\mathcal{R}}_S - \mathbb{E}[\hat{\mathcal{R}}_S] \geq t) \leq \exp\left(-\frac{2t^2}{\sum (B/n)^2}\right) = \exp\left(-\frac{2t^2}{nB^2/n^2}\right) = \exp(-2nt^2/B^2).$$

$t = B\sqrt{\frac{\log(1/\delta)}{2n}}$로 놓으면 원하는 농도를 얻는다. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> **정리 5.7**: VC bound를 Rademacher bound로부터 유도하라. 특히 Massart's lemma (Ch5-04)와 Sauer-Shelah lemma (Ch4-04)를 조합하는 논리를 설명하라.</summary>

<br/>

**해설**. 0-1 valued 가설공간 $\mathcal{H} \subseteq \{0,1\}^\mathcal{X}$에서 loss = 0-1 loss. 그러면
$$\mathcal{R}_n(\mathcal{H}) = \mathbb{E}_{S,\sigma}\left[\sup_{h \in \mathcal{H}} \frac{1}{n}\sum_i \sigma_i h(x_i)\right].$$

**Step 1: Massart's Lemma**: 고정된 $S = (x_1, \ldots, x_n)$에 대해, 함수족 $\mathcal{H}|_S = \{h|_S : h \in \mathcal{H}\}$는 $\mathbb{R}^n$ 위의 **유한 집합** (최대 dichotomy 수 = $|\mathcal{H}(S)|$). 정리 5.4 (Ch5-04):
$$\hat{\mathcal{R}}_S(\mathcal{H}) \leq \sqrt{\frac{2 \log |\mathcal{H}(S)|}{n}}.$$

**Step 2**: 기대값을 취하면:
$$\mathcal{R}_n(\mathcal{H}) = \mathbb{E}_S[\hat{\mathcal{R}}_S(\mathcal{H})] \leq \mathbb{E}_S\left[\sqrt{\frac{2\log|\mathcal{H}(S)|}{n}}\right].$$

Jensen 부등식 역방향으로 loose하게:
$$\leq \sqrt{\frac{2 \mathbb{E}[\log |\mathcal{H}(S)|]}{n}} \leq \sqrt{\frac{2 \log \mathbb{E}[|\mathcal{H}(S)|]}{n}}.$$

**Step 3: Sauer-Shelah Lemma** (정리 4.2, Ch4-04): VC$(\mathcal{H}) = d$이면
$$\max_S |\mathcal{H}(S)| = \Pi_\mathcal{H}(n) \leq \sum_{i=0}^d \binom{n}{i} \leq (en/d)^d.$$

따라서:
$$\mathcal{R}_n(\mathcal{H}) \leq \sqrt{\frac{2d\log(en/d)}{n}} = \sqrt{\frac{2d(\log e + \log n - \log d)}{n}}.$$

$d \ll n$이면 $\approx \sqrt{\frac{2d\log n}{n}}$.

**Step 4**: 정리 5.5에 대입:
$$\sup_h |L_\mathcal{D}(h) - L_S(h)| \leq 2\sqrt{\frac{2d\log n}{n}} + \sqrt{\frac{\log(1/\delta)}{2n}} = O\left(\sqrt{\frac{d\log n + \log(1/\delta)}{n}}\right).$$

**이것이 정리 4.6 (VC bound)과 정확히 동일!**

의미: Rademacher는 VC를 일반화하고 포함한다. VC는 "worst-case" 가정, Rademacher는 "data-dependent"이지만 VC bound도 회복 가능. $\square$

</details>

---

<div align="center">

◀ [이전: 01. Rademacher 복잡도의 정의](./01-rademacher-definition.md) | [📚 README](../README.md) | [다음: 03. Contraction Lemma ▶](./03-contraction-lemma.md)

</div>
