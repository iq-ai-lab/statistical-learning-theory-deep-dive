# 01. Structural Risk Minimization (SRM)

## 🎯 핵심 질문

- **Vapnik의 Structural Risk Minimization (SRM)**이란 무엇인가? 단순히 $L_S + \text{penalty}$인가, 아니면 그보다 깊은 원리가 있는가?
- 중첩된 가설공간 $\mathcal{H}_1 \subset \mathcal{H}_2 \subset \ldots$에서 **어떤 $k$를 선택하는가**? VC bound에서 유도되는 penalty는 무엇인가?
- 왜 **oracle** $\min_k(\text{approx}_k + \text{complexity}_k)$를 알 수 없어도 SRM이 oracle의 log-factor 이내에서 최적인가?
- **Regularization path** — ridge의 $\lambda$ 다양화, LASSO의 경로 — 는 SRM의 어떤 실체화인가?
- SRM은 현대 DL의 **early stopping**·**weight decay**를 어떻게 정당화하는가?

---

## 🔍 왜 SRM이 현대 ML에서 중요한가

Ch3~Ch6에서 우리는 **고정된 $\mathcal{H}$에서의 일반화**만 다뤘다. 하지만 실전은 다르다 — "어떤 모델 복잡도를 써야 하는가?"가 **가장 중요한 질문**이다. 더 큰 신경망을 써야 하는가? 더 높은 차수 다항식? 얼마나 많은 선택 알고리즘 반복? SRM은 **모델 복잡도 선택을 이론적으로 정식화**한 첫 원리적 접근이다.

고전적 접근("CV로 고르자")도 있지만, **왜 CV가 일반화 오차를 잘 추정하는가**는 따로 증명이 필요하다. SRM은 **복잡도-경험위험 트레이드오프를 명시적으로 수식화**하고, "oracle을 모를 때도 최적에 가깝게 갈 수 있다"는 이론적 보장을 제공한다. 이것이 modern practice의 **regularization path**들(Ridge, LASSO, boosting margin)의 수학적 정당화다.

---

## 📐 수학적 선행 조건

- Ch1-03 (ERM과 3분해): approximation·estimation·optimization 개념
- Ch3~Ch4 (PAC·VC theory): $\sup_h |L_\mathcal{D}(h) - L_S(h)|$ bounded family를 위한 uniform convergence
- Ch2 (집중부등식): Hoeffding, Union Bound의 $\log$ 계수 이해
- [Calculus & Optimization](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Regularization, strong convexity

---

## 📖 직관적 이해

### 모델 복잡도 선택의 딜레마

$n=100$ 샘플이 있고, 다항식 회귀를 하려 한다:
- degree 1: 훈련 오차 높음, 테스트 오차 높음 (underfitting)
- degree 5: 훈련 오차 중간, 테스트 오차 중간 (good fit)
- degree 50: 훈련 오차 거의 0, 테스트 오차 높음 (overfitting)

**하지만 우리는 테스트 데이터를 모른다.** 오직 훈련 데이터만 본다. 훈련 오차 $L_S$는 degree가 높을수록 항상 감소한다. 그렇다면 "어떻게 5를 선택하는가?"

### SRM의 아이디어: "모델 복잡도도 하나의 비용"

$$\text{SRM criterion} = L_S(h_k) + \text{Complexity Penalty}(k)$$

여기서:
- $L_S(h_k)$: $\mathcal{H}_k$에서 최적인 가설의 훈련 오차 (낮을수록 좋음)
- $\text{Complexity Penalty}(k)$: $\mathcal{H}_k$가 얼마나 큰가 (높은 차수 = 큼 = 나쁨)

이 둘의 합을 최소화하는 $k^*$를 고르면 **oracle**이 하는 선택 $\arg\min_k (\text{approx}_k + \text{est}_k)$에 가깝다.

---

## ✏️ 엄밀한 정의

### 정의 7.1 (중첩된 가설공간)

**구조화된 가설공간**은 다음을 만족하는 수열이다:
$$\mathcal{H}_1 \subset \mathcal{H}_2 \subset \mathcal{H}_3 \subset \ldots,$$
여기서 각 $\mathcal{H}_k$는 **VC 차원 $d_k < \infty$**를 갖는 유한한 표현(parametric)이다. 예:

- **Polynomial regression**: $\mathcal{H}_k = $ {degree $\leq k$ polynomial}
- **Neural networks**: $\mathcal{H}_k = $ {width $\leq k$ neural net}
- **Linear model with $\ell_2$ reg**: $\mathcal{H}_\lambda = \{w: \|w\|_2 \leq 1/\sqrt{\lambda}\}$ as $\lambda$ decreases

### 정의 7.2 (SRM 규칙)

주어진 표본 $S$, 신뢰도 $\delta \in (0, 1)$, 각 $k$에 대한 VC 경계를 사용하여:

$$\hat{h} := \hat{h}_{\hat{k}}, \quad \hat{k} = \arg\min_{k} \left\{ L_S(\hat{h}_k) + \Omega_k(n, \delta) \right\},$$

여기서 $\Omega_k(n, \delta)$는 **VC bound에서 유도된 복잡도 penalty**:

$$\Omega_k(n, \delta) := C \sqrt{\frac{d_k \log(n/d_k) + \log(2^k/\delta)}{n}}.$$

(상수 $C$는 기술적 세부사항, Union Bound에서 $\sum_k \delta_k = \delta$로 $\delta_k = \delta / 2^k$ 선택.)

### 정의 7.3 (Regularization path)

파라미터 $\lambda$ (regularization strength)를 다양화하여 얻는 중첩 가설공간:

$$\mathcal{H}_\lambda = \arg\min_{h} \left\{ L_S(h) + \lambda \Omega(h) \right\},$$

여기서 $\Omega(h)$ = $\|h\|_2$ (Ridge), $\|h\|_1$ (LASSO) 등. $\lambda$를 작게(과적합)에서 크게(과소적합)로 변화시키면 regularization path가 생성.

---

## 🔬 정리와 증명

### 정리 7.1 (Vapnik의 SRM 정리)

$\mathcal{H}_1 \subset \mathcal{H}_2 \subset \ldots$이 중첩되고 각 $\text{VC}(\mathcal{H}_k) = d_k$이면, SRM 규칙으로 선택한 $\hat{h} = \hat{h}_{\hat{k}}$의 excess risk는 확률 $\geq 1 - \delta$로 다음을 만족한다:

$$L_\mathcal{D}(\hat{h}) - L^* \leq 2 \min_{k} \left\{ L_\mathcal{D}(h^*_k) - L^* + \Omega_k(n, \delta) \right\} + o(1/\sqrt{n}),$$

여기서 $h^*_k = \arg\min_{h \in \mathcal{H}_k} L_\mathcal{D}(h)$이고, 우변의 $\min_k(\cdot)$는 **oracle optimal choice**를 나타낸다.

**증명 아이디어**. 
1. VC bound에 의해 각 $k$에 대해 확률 $\geq 1 - \delta_k$로:
$$L_\mathcal{D}(h_k) - L_S(h_k) \leq \Omega_k(n, \delta_k), \quad \delta_k = \delta/2^k.$$

2. Union bound로 확률 $\geq 1 - \sum_k \delta_k = 1 - \delta$로 **모든 $k$ 동시에** 성립.

3. SRM이 선택한 $\hat{k}$에 대해:
$$L_S(\hat{h}_{\hat{k}}) + \Omega_{\hat{k}} \leq L_S(h^*_k) + \Omega_k, \quad \forall k.$$

4. 이를 $L_\mathcal{D}$로 변환:
$$L_\mathcal{D}(\hat{h}_{\hat{k}}) = L_S(\hat{h}_{\hat{k}}) + (L_\mathcal{D} - L_S)(\hat{h}_{\hat{k}})$$
$$\leq L_S(h^*_k) + \Omega_k + \Omega_{\hat{k}}$$
$$\leq L_\mathcal{D}(h^*_k) - (L_\mathcal{D} - L_S)(h^*_k) + \Omega_k + \Omega_{\hat{k}}.$$

5. 각각의 $(L_\mathcal{D} - L_S)$ 항이 $\leq \Omega$ 바운드 가능 → 결합하면 $2\min_k(\cdot)$ 형태.

**결론**: SRM의 차선택이 oracle을 **로그 팩터와 상수 배수(2)** 이내에서 따라간다. $\square$

### 정리 7.2 (SRM의 수렴 속도)

$L_\mathcal{D}(h^*_k) - L^* = O(k^{-\alpha})$ (근사율)이고 $d_k = \Theta(k^\beta)$ (VC 증가)이면, 최적 $k^*$를 고른 SRM의 excess risk는:

$$\mathbb{E}[L_\mathcal{D}(\hat{h})] - L^* = O\left( n^{-\frac{2\alpha}{2\alpha + 1 + \beta}} \right) + o(1/\sqrt{n}).$$

**증명 스케치**. 최적 balance $\text{approx} \approx \text{est}$에서 $k^{-\alpha} \approx \sqrt{k^\beta/n}$를 풀면 $k^* \sim n^{1/(2\alpha + \beta)}$를 얻고, excess risk는 $\text{approx}(k^*) = k^{*-\alpha} = n^{-2\alpha/(2\alpha+1+\beta)}$. $\square$

### 정리 7.3 (Regularization path와 SRM의 동치성)

Ridge regression $\min_w L_S(w) + \lambda \|w\|_2^2$에서 $\lambda$를 다양화하는 것은 **implicit 중첩 가설공간**을 정의한다:
$$\mathcal{H}_\lambda = \{w: \|w\|_2 \leq 1/\sqrt{\lambda}\}.$$

이 경로에서 **GCV (Generalized Cross-Validation) 규칙** 또는 **AIC**로 $\lambda$를 고르는 것은 SRM과 점근적으로 동치이다.

**직관**: $\lambda$가 작을수록(약한 정규화) $\|w\|$ 상한이 커져 더 풍부한 모델 가능. $\lambda$를 스캔하며 CV error를 추적하는 것이 "최적 복잡도 찾기"와 같다. $\square$

---

## 💻 NumPy 구현 검증

### 실험: 다항식 회귀에서 SRM의 degree 선택

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)

# 진짜 함수: f(x) = sin(2πx)
def sample(n):
    X = rng.uniform(0, 1, n)
    Y = np.sin(2 * np.pi * X) + 0.1 * rng.standard_normal(n)
    return X.reshape(-1, 1), Y

n_train, n_test = 50, 5000
X_train, Y_train = sample(n_train)
X_test,  Y_test  = sample(n_test)

# 각 degree에서 ERM 수행 및 SRM penalty 계산
degrees = np.arange(1, 21)
L_S = []
L_D = []
penalties = []
srm_criterion = []

for d in degrees:
    # Polynomial regression
    poly = PolynomialFeatures(d, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    reg = LinearRegression().fit(X_train_poly, Y_train)
    
    # L_S (training loss)
    L_S_d = np.mean((reg.predict(X_train_poly) - Y_train) ** 2)
    L_S.append(L_S_d)
    
    # L_D (test loss, oracle proxy)
    L_D_d = np.mean((reg.predict(X_test_poly) - Y_test) ** 2)
    L_D.append(L_D_d)
    
    # VC bound penalty: Ω_d ≈ C√(d/n)
    # d = polynomial degree, VC ≤ d+1
    C = 0.5  # scaling constant
    penalty = C * np.sqrt((d + 1) / n_train)
    penalties.append(penalty)
    
    # SRM criterion
    srm_criterion.append(L_S_d + penalty)

# 최적 degree (oracle vs SRM)
deg_optimal_oracle = degrees[np.argmin(L_D)]      # 진짜 최선 (테스트로 안다)
deg_optimal_srm = degrees[np.argmin(srm_criterion)]  # SRM이 고르는 것

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot 1: Training loss, penalty, SRM criterion
axes[0].plot(degrees, L_S, 'o-', label='$L_S$ (training loss)', linewidth=2)
axes[0].plot(degrees, penalties, 's-', label='Penalty $\Omega_d$', linewidth=2)
axes[0].plot(degrees, srm_criterion, 'd-', label='SRM criterion', linewidth=2, color='purple')
axes[0].axvline(deg_optimal_oracle, color='green', linestyle='--', 
                label=f'Oracle: d={deg_optimal_oracle}')
axes[0].axvline(deg_optimal_srm, color='red', linestyle='--', 
                label=f'SRM: d={deg_optimal_srm}')
axes[0].set_xlabel('Polynomial degree $d$')
axes[0].set_ylabel('Loss / Penalty')
axes[0].set_title('SRM Criterion 구성: $L_S(h_d) + \Omega_d(n,\\delta)$')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Training vs test loss
axes[1].plot(degrees, L_S, 'o-', label='Training loss', linewidth=2)
axes[1].plot(degrees, L_D, 's-', label='Test loss (oracle)', linewidth=2)
axes[1].axvline(deg_optimal_oracle, color='green', linestyle='--', alpha=0.7)
axes[1].axvline(deg_optimal_srm, color='red', linestyle='--', alpha=0.7)
axes[1].set_xlabel('Polynomial degree $d$')
axes[1].set_ylabel('Loss (MSE)')
axes[1].set_title('Training vs Test Loss — Underfitting vs Overfitting')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f'Oracle optimal degree: {deg_optimal_oracle}')
print(f'SRM optimal degree: {deg_optimal_srm}')
print(f'Oracle test loss: {L_D[deg_optimal_oracle - 1]:.4f}')
print(f'SRM test loss: {L_D[deg_optimal_srm - 1]:.4f}')
# → SRM이 oracle 근처의 degree를 선택함을 관찰.
```

### 실험 2: Ridge regularization path

```python
# Ridge regression의 λ (strength) 다양화
from sklearn.linear_model import Ridge

# 간단한 선형 데이터
X = rng.standard_normal((50, 10))
w_true = rng.standard_normal(10)
Y = X @ w_true + rng.standard_normal(50)

lambdas = np.logspace(-3, 3, 50)
train_errors = []
test_X = rng.standard_normal((1000, 10))
test_Y = test_X @ w_true + rng.standard_normal(1000)
test_errors = []

for lam in lambdas:
    ridge = Ridge(alpha=lam, fit_intercept=False).fit(X, Y)
    train_errors.append(np.mean((ridge.predict(X) - Y) ** 2))
    test_errors.append(np.mean((ridge.predict(test_X) - test_Y) ** 2))

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(lambdas, train_errors, 'o-', label='Training error', linewidth=2)
ax.semilogx(lambdas, test_errors, 's-', label='Test error', linewidth=2)
ax.set_xlabel('Regularization strength $\lambda$')
ax.set_ylabel('MSE')
ax.set_title('Ridge Regularization Path — 최적 $\lambda$는 어디인가?')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# → 작은 λ (약한 정규화): 훈련 오차 낮음, 테스트 오차 높음
#   큰 λ (강한 정규화): 둘 다 높음
#   최적 λ는 중간에 있음 — SRM이 찾는 지점
```

---

## 🔗 ML 알고리즘 연결

| 알고리즘 | 중첩 $\mathcal{H}_k$ | 복잡도 measure | SRM 관점 |
|---------|------------------|--------------|----------|
| **Polynomial regression** | degree $1, 2, \ldots$ | VC$(d)$ | degree 선택 |
| **Ridge regression** | $\lambda$ path: $\\|w\\| \leq 1/\sqrt{\lambda}$ | norm bound | AIC/GCV로 $\lambda$ 선택 |
| **LASSO** | $\lambda$ path: $\|w\|_1 \leq \lambda$ 제약 | $\ell_1$ norm | $\lambda$ path tracing |
| **SVM** | margin: $\|w\|^2 \leq C$ 제약 | $\|w\|$ norm | $C$ 선택 by CV |
| **Boosting** | round $t = 1, 2, \ldots$ | margin 분포 | early stopping |
| **Neural network** | layer/width/depth 증가 | $\prod\|W_l\|$ 또는 early stopping | weight decay, early stop |

**관찰**: 거의 모든 **regularization parameter 선택**이 implicit SRM이다. CV나 GCV가 이를 실체화한다.

---

## ⚖️ 가정과 한계

1. **중첩 구조 가정**: 현실에서 $\mathcal{H}_k$들이 깔끔하게 중첩되지 않을 수 있다 (예: 신경망은 depth와 width 동시 증가). 다중 구조로 확장 가능하나 복잡해짐.

2. **VC 차원의 명확한 증가**: VC가 명확하게 계산되어야 $\Omega_k$ 정의 가능. 많은 실전 모델(CNN, Transformer)은 VC가 명확하지 않음.

3. **타이트성**: SRM bound에서 2배의 상수, Union bound의 $\log(1/\delta)$ 등 보수적 계수들이 있음. 실전에서는 더 타이트한 data-driven penalty (AIC/BIC)가 더 효과적.

4. **계산 비용**: 모든 $k$에서 ERM을 수행해야 함 → 많은 models를 학습해야 함. Lasso path, Ridge path 같은 **computational tricks**이 필요.

5. **고차원 한계**: $d_k \gg n$인 경우(DL) SRM bound가 의미 없음 → data-dependent bound (Rademacher, stability)로 전환 필요.

---

## 📌 핵심 정리

- **SRM**: 중첩 가설공간 시퀀스에서 $\hat{k} = \arg\min_k (L_S(\hat{h}_k) + \Omega_k)$로 모델 복잡도 선택.
- **VC 기반 penalty**: $\Omega_k(n, \delta) \propto \sqrt{(\text{VC}(\mathcal{H}_k) \log n + \log(1/\delta))/n}$.
- **Oracle comparison**: SRM이 선택한 $\hat{h}$의 excess risk는 oracle $\min_k(\text{approx}_k + \text{est}_k)$의 **2배 내지 log factor**.
- **실전화**: Ridge/LASSO의 regularization path, AIC/BIC/CV가 모두 implicit SRM.
- **한계**: 고전 VC bound는 보수적 → data-driven penalty (Ch7-02: AIC/BIC) 또는 stability 기반 bound (Ch6) 선호.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> 다항식 회귀에서 degree $d = 5$의 VC 차원이 6일 때, $n = 100$, $\delta = 0.05$이면 SRM penalty $\Omega_5$는 대략 얼마인가?</summary>

<br/>

**해설**. 공식 $\Omega_k = C\sqrt{(d_k \log(n/d_k) + \log(2^k/\delta))/n}$ (단, 여기서 $k=5$는 degree index):

$$\Omega_5 \approx C\sqrt{\frac{6 \log(100/6) + \log(2^5/0.05)}{100}} \approx C\sqrt{\frac{6 \times 2.8 + 7.7}{100}} \approx C\sqrt{\frac{24.5}{100}} \approx 0.495 C.$$

$C \approx 1$(기술적 상수)로 두면 $\Omega_5 \approx 0.5$. 이것을 $L_S(\hat{h}_5)$에 더해서 SRM criterion을 구성한다. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> SRM에서 Union bound를 사용하여 $\delta_k = \delta/2^k$로 설정했을 때, 무한 시퀀스 $k = 1, 2, 3, \ldots$에서도 $\sum_k \delta_k = \delta$가 성립함을 보여라. 이것이 왜 중요한가?</summary>

<br/>

**해설**. 기하급수: $\sum_{k=1}^\infty \delta/2^k = \delta \sum_{k=1}^\infty (1/2)^k = \delta \cdot \frac{1/2}{1-1/2} = \delta$. 

따라서 union bound로 **모든 $k$ 동시에** 확률 $\geq 1 - \delta$로 각 bound가 성립함을 보장할 수 있다. 이는 $k$가 미리 정해지지 않고 데이터에서 **선택**되더라도(adversarial) 수렴성을 유지한다는 뜻 — SRM의 강점. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> Deep neural network에서 "early stopping"(전체 epoch 완료 전 멈추기)는 왜 regularization처럼 작용하는가? SGD의 스텝 수 $T$가 implicit 복잡도 measure가 된다고 주장할 때, 이것을 SRM 프레임에서 어떻게 해석할 것인가?</summary>

<br/>

**해설**. Ch6-04 (SGD stability)에서: 유한한 step SGD는 $\beta \leq O(\eta T)$ stable — 즉, $T$가 작을수록(적게 훈련) 더 stable하고 일반화 gap이 작다. 이를 SRM으로 보면:

$$\mathcal{H}_T = \{\text{$T$ steps SGD에서 도달 가능한 가설}\},$$

이는 implicit 중첩 $\mathcal{H}_1 \subset \mathcal{H}_2 \subset \ldots$ 구조를 이룬다(더 많은 스텝 = 더 풍부한 class). SRM은 $\arg\min_T (L_S(\hat{h}_T) + \text{complexity}(T))$로 최적 $T^*$를 선택 → **이것이 early stopping의 수학적 정당화**. 현대 DL에서는 validation loss로 $T^*$를 찾는 것이 CV 기반 SRM 실체화다. $\square$

</details>

---

<div align="center">

◀ [이전: Ch6-04. SGD의 Stability](../ch6-stability/04-sgd-stability.md) | [📚 README](../README.md) | [다음: 02. AIC, BIC, MDL ▶](./02-aic-bic-mdl.md)

</div>
