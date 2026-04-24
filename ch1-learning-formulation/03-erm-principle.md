# 03. Empirical Risk Minimization (ERM)

## 🎯 핵심 질문

- **Empirical Risk Minimization(ERM)** $\hat{h} = \arg\min_{h \in \mathcal{H}} L_S(h)$은 왜 "자연스러운" 학습 원리인가? 단순하지만 왜 **SLT 이론의 중심**인가?
- ERM이 잘 동작하려면 **어떤 조건**이 필요한가 — $|\mathcal{H}|$가 유한하거나 VC가 유한해야? 왜?
- Excess risk $L_\mathcal{D}(\hat{h}) - L^*$를 **근사·추정·최적화** 세 원천으로 분해하는 것이 왜 **모델 설계**·**데이터 수집**·**알고리즘 튜닝** 각각에 대응하는가?
- ERM은 왜 **NP-hard**일 수 있고, 그래서 왜 hinge/log 같은 **surrogate loss**로 대체하는가?
- $\mathcal{H} = \mathcal{Y}^\mathcal{X}$ (모든 가측함수)에서 ERM을 하면 왜 **과적합**이 심한가 — memorization 현상의 ERM 해석?

---

## 🔍 왜 ERM이 SLT의 중심인가

"훈련 데이터에서 오차를 최소화하자" — ERM은 **너무 당연해서** 원리라고 부르는 것이 이상할 정도다. 하지만 바로 이 당연함이 SLT 전체의 **분석 중심**이다. 고전 통계는 "추정량을 제안 → 성질(일치성·불편성) 증명"의 순서로 간다. SLT의 혁신은 그 순서를 뒤집어 **"거의 모든 ML 알고리즘이 암묵적으로 ERM"이라는 관찰에서 출발**해, $L_S$의 최소값이 $L_\mathcal{D}$의 최소값을 얼마나 잘 근사하는지를 **분포 자유(distribution-free)**로 분석한다.

이 분석은 두 단계다. 먼저 **근사·추정·최적화**의 3분해로 excess risk를 쪼개고, 각 항을 독립적으로 다룬다. approximation은 $\mathcal{H}$ 설계의 문제, estimation은 uniform convergence(Ch3~5)의 문제, optimization은 convex analysis·SGD 이론(Ch6-04 연결)의 문제다. 둘째, ERM의 **실패 양상**을 분류한다 — overfitting은 estimation의 실패이고, underfitting은 approximation의 실패이며, 수렴 실패는 optimization의 실패다. 이 프레임워크가 없으면 "왜 더 큰 모델이 도움이 되는가", "왜 더 많은 데이터가 도움이 되는가", "왜 때때로 덜 훈련하는 것이 더 나은가"(Ch6-04 SGD stability)가 하나로 통합 설명되지 않는다.

---

## 📐 수학적 선행 조건

- Ch1-01 (위험의 정의), Ch1-02 (Bayes 최적 예측기)
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): 볼록 최적화, strong convexity, KKT
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): M-estimator 일관성
- 기초: 최소화 문제의 존재, argmin의 표기

---

## 📖 직관적 이해

### ERM = "알고리즘이 갖는 유일한 신"

우리는 $L_\mathcal{D}$를 직접 볼 수 없다. 볼 수 있는 것은 **샘플 $S$**를 통해 재구성한 $L_S$뿐. **ERM은 "가장 정직한 전략"** — 볼 수 있는 지형에서 가장 낮은 지점을 고른다. 다른 선택은 모두 "부분 정보에 기반한 선호"를 추가로 주입하는 것이다:

- **Regularized ERM**: $\min L_S(h) + \lambda \Omega(h)$ — "덜 복잡한 것을 선호"
- **MAP estimation**: $\min [-\log p(S|h)] + [-\log p(h)]$ — "사전 분포에 가까운 것을 선호"
- **SRM** (Ch7-01): $\min \{L_S(h) + \text{complexity penalty}\}$ — "VC 작은 것을 선호"

이들은 모두 **ERM + 추가 항**이다.

### 3분해의 실용적 의미

$$\underbrace{L_\mathcal{D}(\hat{h}) - L^*}_{\text{excess risk}} = \underbrace{L_\mathcal{D}(h^*_\mathcal{H}) - L^*}_{\substack{\text{approximation} \\ \mathcal{H} \text{ 디자인 문제}}} + \underbrace{L_\mathcal{D}(\hat{h}^*) - L_\mathcal{D}(h^*_\mathcal{H})}_{\substack{\text{estimation} \\ \text{데이터·uniform conv 문제}}} + \underbrace{L_\mathcal{D}(\hat{h}) - L_\mathcal{D}(\hat{h}^*)}_{\substack{\text{optimization} \\ \text{수렴·계산 문제}}}$$

여기서
- $h^*_\mathcal{H} = \arg\min_{h \in \mathcal{H}} L_\mathcal{D}(h)$: $\mathcal{H}$ 내의 **진짜 최적** (도달 불가능)
- $\hat{h}^* = \arg\min_{h \in \mathcal{H}} L_S(h)$: **진짜 ERM 해** (정확한 최적화 가정)
- $\hat{h}$: 실제 알고리즘이 출력하는 **근사 해**

이 셋은 각각 **무엇을 바꿔서 줄이는가**가 다르다:
- approximation ↓: 더 큰/표현력 높은 $\mathcal{H}$
- estimation ↓: 더 많은 $n$, 혹은 더 작은 $\mathcal{H}$ (trade-off!)
- optimization ↓: 더 오래/똑똑한 최적화

---

## ✏️ 엄밀한 정의

### 정의 3.1 (ERM 원리)

가설공간 $\mathcal{H}$와 샘플 $S = ((x_i, y_i))_{i=1}^n$에 대해 **ERM 규칙**은 $S$에서 경험 위험을 최소화하는 가설 중 **하나를 고르는** 학습기:
$$\text{ERM}_\mathcal{H}(S) \in \arg\min_{h \in \mathcal{H}} L_S(h).$$

(argmin이 여러 개일 수 있으므로 tie-breaking 규칙이 필요하나 이론 분석에는 영향 없음.)

### 정의 3.2 (3 가지 가설)

- $h^* \in \arg\min_{h \text{ 가측}} L_\mathcal{D}(h)$: **Bayes 예측기**
- $h^*_\mathcal{H} \in \arg\min_{h \in \mathcal{H}} L_\mathcal{D}(h)$: **$\mathcal{H}$ 내 모집단 최적**
- $\hat{h}^*_S \in \arg\min_{h \in \mathcal{H}} L_S(h)$: **$\mathcal{H}$ 내 표본 최적 (ERM)**

### 정의 3.3 (Approximation·Estimation·Optimization 오차)

실제 알고리즘이 출력하는 가설 $\hat{h}$에 대해:
$$\text{err}_{\text{approx}} := L_\mathcal{D}(h^*_\mathcal{H}) - L^*, \quad \text{err}_{\text{est}} := L_\mathcal{D}(\hat{h}^*_S) - L_\mathcal{D}(h^*_\mathcal{H}), \quad \text{err}_{\text{opt}} := L_\mathcal{D}(\hat{h}) - L_\mathcal{D}(\hat{h}^*_S).$$

---

## 🔬 정리와 증명

### 정리 3.1 (Excess risk의 3분해)

임의의 $\hat{h} \in \mathcal{H}$에 대해
$$L_\mathcal{D}(\hat{h}) - L^* = \text{err}_{\text{approx}} + \text{err}_{\text{est}} + \text{err}_{\text{opt}}.$$

**증명**. 기호 조작:
$$L_\mathcal{D}(\hat{h}) - L^* = [L_\mathcal{D}(h^*_\mathcal{H}) - L^*] + [L_\mathcal{D}(\hat{h}^*_S) - L_\mathcal{D}(h^*_\mathcal{H})] + [L_\mathcal{D}(\hat{h}) - L_\mathcal{D}(\hat{h}^*_S)]. \qquad \square$$

이 분해는 **확률변수 등식**이다(각 항이 $S$·알고리즘의 랜덤성에 의존). 확률/기대값을 취할 때 분해가 유지된다.

### 정리 3.2 (Estimation error의 표준 바운드)

$\hat{h}^*_S \in \arg\min_{h \in \mathcal{H}} L_S(h)$에 대해
$$L_\mathcal{D}(\hat{h}^*_S) - L_\mathcal{D}(h^*_\mathcal{H}) \leq 2 \sup_{h \in \mathcal{H}} |L_\mathcal{D}(h) - L_S(h)|.$$

**증명**. 다음 연쇄 부등식을 보인다:
$$\begin{aligned}
L_\mathcal{D}(\hat{h}^*_S) - L_\mathcal{D}(h^*_\mathcal{H}) &= \underbrace{[L_\mathcal{D}(\hat{h}^*_S) - L_S(\hat{h}^*_S)]}_{\leq \sup_h |L_\mathcal{D}(h) - L_S(h)|} + \underbrace{[L_S(\hat{h}^*_S) - L_S(h^*_\mathcal{H})]}_{\leq 0 \text{ (ERM 정의)}} + \underbrace{[L_S(h^*_\mathcal{H}) - L_\mathcal{D}(h^*_\mathcal{H})]}_{\leq \sup_h |L_\mathcal{D}(h) - L_S(h)|}.
\end{aligned}$$
첫째와 셋째 항은 각각 $\sup_h |L_S - L_\mathcal{D}|$로 bound되고, 둘째 항은 $\hat{h}^*_S$가 $L_S$의 최소화자이므로 $\leq 0$. 합치면 $2 \sup$. $\square$

> 이것이 **uniform convergence가 estimation error를 통제**한다는 SLT의 핵심 관찰이다. Ch3~5 전체가 $\sup_h |L_\mathcal{D}(h) - L_S(h)|$를 bound하는 일이다.

### 정리 3.3 (ERM + 집중부등식 → 유한 $\mathcal{H}$의 첫 PAC 바운드)

$|\mathcal{H}| < \infty$, $\ell \in [0, 1]$, iid 샘플 $|S| = n$일 때 확률 $\geq 1 - \delta$로
$$L_\mathcal{D}(\hat{h}^*_S) - L_\mathcal{D}(h^*_\mathcal{H}) \leq 2 \sqrt{\frac{\log(2|\mathcal{H}|/\delta)}{2n}}.$$

**증명 스케치**. Hoeffding(Ch2-02): 각 $h$에 대해 $\mathbb{P}(|L_S(h) - L_\mathcal{D}(h)| \geq \epsilon) \leq 2 e^{-2n\epsilon^2}$. Union bound:
$$\mathbb{P}\!\left(\sup_{h \in \mathcal{H}} |L_S(h) - L_\mathcal{D}(h)| \geq \epsilon\right) \leq 2|\mathcal{H}| e^{-2n\epsilon^2}.$$
$\delta$와 같게 놓고 $\epsilon$에 대해 풀면 $\epsilon = \sqrt{\log(2|\mathcal{H}|/\delta) / (2n)}$. 정리 3.2와 결합. $\square$

> 이 주장은 Ch3-03에서 **Agnostic PAC learnability**의 공식으로 확립된다. 여기서는 ERM 분석의 "첫 성공" 사례로 관찰.

### 정리 3.4 (ERM의 실패 — No Free Lunch 예고)

$\mathcal{H} = \mathcal{Y}^\mathcal{X}$ (모든 가측함수)일 때, ERM은 본질적으로 "**훈련 라벨 기억(memorization)**"이 되어 $L_\mathcal{D}(\hat{h})$는 Bayes risk로 수렴하지 않을 수 있다.

**증명 스케치**. $\hat{h}(x_i) = y_i$로 점별 정의된 후 다른 $x$에서 임의 라벨을 주는 $\hat{h}$는 $L_S(\hat{h}) = 0$을 달성한다 (0-1 loss). 따라서 ERM은 이런 $\hat{h}$를 허용할 수 있다. 하지만 $\mathcal{D}$가 연속이면 test point에서 훈련 포인트와 거의 절대로 일치하지 않고, $\hat{h}$의 **훈련 밖 값은 임의**라 $L_\mathcal{D}(\hat{h})$이 $L^*$에 가까울 보장이 없다. 상세한 formal 형태는 Ch1-04의 No Free Lunch. $\square$

이것이 **$\mathcal{H}$를 제약해야 학습이 가능**하다는 SLT의 근본적 주장이다.

---

## 💻 NumPy 구현 검증

### 실험 1: 3분해를 수치로 재구성 (1D 회귀)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(7)

# 진짜 함수: f*(x) = sin(2πx), noise N(0, 0.1^2)
def sample(n):
    X = rng.uniform(0, 1, n)
    Y = np.sin(2 * np.pi * X) + 0.1 * rng.standard_normal(n)
    return X.reshape(-1, 1), Y

x_grid = np.linspace(0, 1, 300).reshape(-1, 1)
f_star = np.sin(2 * np.pi * x_grid).ravel()
sigma2 = 0.01                                  # noise variance
L_star = sigma2                                # Bayes risk (정리 2.1 기반)

# 고정 샘플 크기 n, degree d에서 ERM (polynomial regression)
n, n_test = 50, 10000
X_train, Y_train = sample(n)
X_test,  Y_test  = sample(n_test)

def erm_poly(d, X_tr, Y_tr):
    poly = PolynomialFeatures(d, include_bias=False)
    Xp = poly.fit_transform(X_tr)
    reg = LinearRegression().fit(Xp, Y_tr)
    def h(x):
        return reg.predict(poly.transform(x))
    return h

# h_H*: 매우 큰 n으로 근사한 𝒟-최적 다항식 (단, degree d 제약)
X_big, Y_big = sample(200000)

degrees = [1, 2, 3, 5, 7, 10]
approx, estim, excess = [], [], []
for d in degrees:
    h_H_star = erm_poly(d, X_big, Y_big)     # population 최적 근사
    h_hat    = erm_poly(d, X_train, Y_train) # 경험 ERM

    # approximation error: L_D(h_H*) - L*
    L_h_H_star = np.mean((h_H_star(X_test) - Y_test) ** 2)
    err_a = L_h_H_star - L_star

    # estimation error: L_D(h_hat) - L_D(h_H*)
    L_h_hat = np.mean((h_hat(X_test) - Y_test) ** 2)
    err_e = L_h_hat - L_h_H_star
    err_ex = L_h_hat - L_star    # excess risk 전체

    approx.append(max(err_a, 0))
    estim.append(max(err_e, 0))
    excess.append(max(err_ex, 0))

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(degrees, approx,  'o-', label='Approx err (↓ as d↑)')
ax.plot(degrees, estim,   's-', label='Estim err (↑ as d↑, 과적합)')
ax.plot(degrees, excess,  'd-', label='Excess = Approx + Estim')
ax.set_xlabel('polynomial degree d'); ax.set_ylabel('risk gap')
ax.set_title('ERM의 3분해 — 고전적 U-shape (bias-variance SLT 버전)')
ax.legend(); plt.tight_layout(); plt.show()

# → degree가 커지면 approx는 감소, estim은 증가, 둘의 합인 excess는 U-모양.
#   이것이 "모델 복잡도 선택"의 수학적 본질.
```

### 실험 2: $|\mathcal{H}|$ 유한한 경우의 PAC 바운드 검증

```python
# 정리 3.3 검증: 유한 H에서 Hoeffding+Union
# H: {sign(x - θ) : θ ∈ {θ_1, ..., θ_K}} for equally spaced θ_k
# 데이터: X ~ U[0,1], Y = sign(X - 0.5) flipped w.p. p
p_flip = 0.1
def sample_bin(n):
    X = rng.uniform(0, 1, n)
    y_clean = np.sign(X - 0.5)
    flip = rng.random(n) < p_flip
    return X, np.where(flip, -y_clean, y_clean)

K = 100                                        # |H|
theta_grid = np.linspace(0.01, 0.99, K)
def classifier(theta):
    return lambda x: np.sign(x - theta)

# L_S(h_theta)
def emp_loss(theta, X, Y):
    return np.mean(classifier(theta)(X) != Y)

ns = [50, 200, 500, 2000]
delta = 0.05
for n in ns:
    sup_gap = []
    for _ in range(200):
        X, Y = sample_bin(n)
        gaps = []
        for theta in theta_grid:
            LS = emp_loss(theta, X, Y)
            # true risk 계산 (|X - 0.5| 이하 영역에서 noise)
            true_error = p_flip + (1-2*p_flip)*abs(theta - 0.5)
            gaps.append(abs(LS - true_error))
        sup_gap.append(max(gaps))
    sup_gap = np.array(sup_gap)
    bound = np.sqrt(np.log(2*K/delta) / (2*n))
    frac_below = np.mean(sup_gap <= bound)
    print(f'n={n:5d}: P(sup gap ≤ Hoeffding-Union bound) ≈ {frac_below:.2f} (target ≥ {1-delta:.2f})')
# → 모든 n에서 실험 확률이 1-δ=0.95 이상임을 확인.
```

---

## 🔗 ML 알고리즘 연결

| 알고리즘 | ERM 형태 | $\mathcal{H}$ | $\ell$ |
|---------|---------|--------------|-------|
| **OLS** | $\min_w \sum (y_i - w^\top x_i)^2$ | 선형함수 | squared |
| **Logistic Regression** | $\min_w \sum \log(1 + e^{-y_i w^\top x_i})$ | 선형 | log loss |
| **SVM (soft-margin)** | $\min_w \frac{1}{n}\sum \max(0, 1 - y_i w^\top x_i) + \lambda \|w\|^2$ | 선형 | hinge + $\ell_2$ 정규화 |
| **Neural network (SGD)** | $\min_\theta \sum \ell(f_\theta(x_i), y_i)$ | NN 매개변수화 | cross-entropy/MSE |
| **Decision tree (CART)** | $\min_\mathcal{T} \sum \ell$ over tree structure | 트리 | impurity-based |

**관찰**: 거의 모든 지도학습 알고리즘이 ERM이다. 차이점은 (1) $\mathcal{H}$의 선택, (2) $\ell$(surrogate vs 0-1), (3) 정규화 항의 포함, (4) 실제 최적화 방법이다. SLT의 추상화는 이 네 축을 분리해서 분석하게 해준다.

### Surrogate loss의 필요성

0-1 loss로 ERM을 직접 푸는 것은 **NP-hard**(Feldman, Guruswami, Raghavendra, Wu 2009). 볼록이 아니고 미분 불가능. 실용 알고리즘은 **surrogate**를 쓴다:

- **Hinge**: $\max(0, 1 - y f(x))$ → SVM
- **Log loss**: $\log(1 + e^{-y f(x)})$ → logistic
- **Exponential**: $e^{-y f(x)}$ → AdaBoost
- **Squared**: $(1 - y f(x))^2$ → least-squares classifier

이들이 **calibration**되어 있으면 surrogate ERM의 최소화자가 0-1 loss의 Bayes 최적에 수렴(Bartlett, Jordan, McAuliffe 2006, Ch5-03의 contraction lemma와 연결).

---

## ⚖️ 가정과 한계

1. **ERM 최소화자의 존재**: $\mathcal{H}$ 위의 $L_S$가 최소를 달성하지 않을 수 있다(무한 $\mathcal{H}$에서 inf이 달성 안 됨). $\epsilon$-approximate minimizer로 충분한 경우가 많다.
2. **Tie-breaking**: $\arg\min$이 여러 개일 때 어느 것을 고르는가? 이론적으로는 측도 0 차이지만, implicit regularization 관점에선 **선택 자체가 선호를 주입**한다(예: SGD의 minimum-norm solution).
3. **계산 복잡도**: 많은 $\mathcal{H}$에서 ERM은 계산적으로 어렵거나 불가능. 선형 모델 + 볼록 loss에서는 convex optimization으로 풀리지만, neural network ERM은 non-convex.
4. **분포 자유 vs 분포 의존**: ERM 이론은 **어떤 $\mathcal{D}$에서도**(최악의 경우) 성립하는 바운드를 찾는다. 분포 의존 바운드(데이터 평활성·margin)로 넘어가면 tighter한 결과가 가능(Ch5의 Rademacher).
5. **일반화 vs 최적화의 얽힘**: DL에서는 `실제 실행 알고리즘(SGD)이 찾는 $\hat{h}`가 `엄밀한 ERM $\hat{h}^*_S$`와 다를 수 있고, 오히려 이것이 **implicit regularization**의 원천(Ch6-04).

---

## 📌 핵심 정리

- **ERM**: $\hat{h} \in \arg\min_{h \in \mathcal{H}} L_S(h)$. 거의 모든 지도학습 알고리즘의 원리.
- **3분해**: $L_\mathcal{D}(\hat{h}) - L^* = \text{approx} + \text{est} + \text{opt}$. **$\mathcal{H}$ 디자인·$n$·옵티마이저** 각각 대응.
- **핵심 보조정리**: $\text{err}_{\text{est}} \leq 2 \sup_h |L_\mathcal{D}(h) - L_S(h)|$. **Uniform convergence**가 estimation 통제.
- **유한 $\mathcal{H}$**: Hoeffding + Union Bound → $\text{est err} \leq \sqrt{\log(2|\mathcal{H}|/\delta)/(2n)}$.
- **ERM 실패**: $\mathcal{H}$가 무제약이면 **memorization**. $\mathcal{H}$ 제약이 학습의 전제(Ch1-04의 NFL).
- **Surrogate loss**: 0-1은 NP-hard, hinge/log는 convex. calibration(Ch5-03)으로 surrogate ERM → Bayes.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> 정리 3.2의 부등식에서 "2" 상수는 왜 필요한가? 다음 시나리오를 생각해보라: $L_S(\hat{h}^*_S)$가 $L_\mathcal{D}(\hat{h}^*_S)$보다 작고, $L_S(h^*_\mathcal{H})$는 $L_\mathcal{D}(h^*_\mathcal{H})$보다 클 때.</summary>

<br/>

**해설**. ERM은 $L_S$에서 이기는 것을 고르므로 $L_S(\hat{h}^*_S) \leq L_S(h^*_\mathcal{H})$. 이것은 $L_\mathcal{D}$에서도 $\hat{h}^*_S$가 이긴다는 것을 의미하지 **않는다**. 최악의 경우, $\hat{h}^*_S$가 $L_S$에서 "운 좋게" 낮고 ($L_S - L_\mathcal{D}$가 $-\epsilon$) 실제로는 나쁘며, $h^*_\mathcal{H}$가 "운 나쁘게" 높다($L_S - L_\mathcal{D}$가 $+\epsilon$)면, 두 $\epsilon$이 더해져 $L_\mathcal{D}(\hat{h}^*_S) - L_\mathcal{D}(h^*_\mathcal{H}) \leq 2\epsilon$. 한 방향에서 들어오는 $\sup$가 아니라 **두 방향에서의 overlap**이 2 계수를 만든다. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> $\mathcal{H}$를 **확대**하면 approximation error는 (약하게) 감소하지만 estimation error는 (일반적으로) 증가한다. 이 trade-off가 **동일한 데이터**에서 일어날 때, 최적 $\mathcal{H}$ 크기가 $n$에 따라 어떻게 변하는가?</summary>

<br/>

**해설**. 간단 모델: $\text{err}_{\text{approx}}(\mathcal{H}) \approx A/|\mathcal{H}|^\alpha$ (더 큰 $\mathcal{H}$로 Bayes 접근, $\alpha > 0$), $\text{err}_{\text{est}}(\mathcal{H}, n) \approx B \sqrt{\log|\mathcal{H}|/n}$ (Hoeffding+Union). 합을 최소화하려 $|\mathcal{H}|$에 대해 미분 $= 0$:
$$-\frac{\alpha A}{|\mathcal{H}|^{\alpha+1}} + \frac{B}{2|\mathcal{H}| \sqrt{n \log|\mathcal{H}|}} = 0.$$
근사적으로 $|\mathcal{H}|^\alpha \sim \sqrt{n}$, 즉 $\log|\mathcal{H}| \sim \frac{1}{2\alpha} \log n$. **$n$이 커짐에 따라 최적 $|\mathcal{H}|$도 커져야 한다** — 더 많은 데이터가 주어지면 더 풍부한 모델을 쓸 수 있다. 이 관찰이 **SRM(Ch7-01)**의 수학적 기반이고, "데이터가 적으면 단순 모델, 많으면 복잡 모델"이라는 실무 규칙의 이론적 근거.

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> **"Zhang et al. 2017"**(Rethinking Generalization)의 실험: ResNet은 ImageNet 라벨을 **완전 랜덤화**해도 훈련 오차 0을 달성한다. 이 관찰이 ERM 3분해에서 무엇을 말하는가? 왜 "고전 VC bound는 vacuous"한가?</summary>

<br/>

**해설**. 랜덤 라벨에서 $L_S(\hat{h}) = 0$ 달성은 **$\mathcal{H}$가 임의의 라벨링을 realize할 수 있음**(shatter)을 의미한다 — Ch4의 용어로 **VC 차원이 매우 크다**는 것. 실험적 VC 차원 하한 $\approx n$. 그러면 정리 3.3 스타일 Hoeffding+Union bound $\sqrt{\log|\mathcal{H}|/n}$이 $\log|\mathcal{H}| \geq n$이므로 $\geq 1$ — **전혀 쓸모없는(vacuous) bound**.

그럼에도 **진짜(참) 라벨에서는 일반화가 잘 된다**는 사실은 SLT의 고전 프레임이 DL의 **estimation error 통제**를 놓치고 있음을 뜻한다. 돌파구로 제안된 것들:
- **Norm-based Rademacher**(Ch5-06): $\mathcal{H}$ 크기가 아니라 $\prod \|W_l\|$ 같은 크기 의존 복잡도
- **Algorithmic stability**(Ch6): SGD가 implicit regularization
- **PAC-Bayes, Margin bound**: posterior 혹은 margin 분포 의존
- **NTK·Double descent**(Layer 2 Generalization Theory)

이 관찰이 "고전 SLT는 NN을 설명 못한다"는 DL 이론의 출발점이다. 이 레포의 **Ch4-07**과 **Ch5-06**에서 이 파라독스를 다시 다룬다.

</details>

---

<div align="center">

◀ [이전: 02. Bayes 최적 예측기](./02-bayes-optimal.md) | [📚 README](../README.md) | [다음: 04. 일반화 오차와 과적합 ▶](./04-generalization-overfitting.md)

</div>
