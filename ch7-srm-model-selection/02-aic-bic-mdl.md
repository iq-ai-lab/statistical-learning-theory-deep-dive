# 02. AIC, BIC, 그리고 MDL

## 🎯 핵심 질문

- **AIC (Akaike Information Criterion)** $\text{AIC}(k) = -2\log L + 2k$는 무엇을 최소화하려는 것인가? 왜 정확히 이 형태인가?
- AIC의 유도에서 **KL divergence**와 **bias correction**이 어떤 역할을 하는가?
- **BIC (Bayesian Information Criterion)** $\text{BIC}(k) = -2\log L + k \log n$는 어떻게 유도되는가? Laplace approximation 없이 증명할 수 있는가?
- **MDL (Minimum Description Length)** — 데이터를 가장 짧게 설명하는 모델 — 은 **확률론적으로** 어떤 의미인가? BIC와의 관계?
- **AIC는 예측 능력이 최적**, **BIC는 참 모델 선택(consistency)**인데, 이 둘을 어떻게 구분 해석하는가?

---

## 🔍 왜 AIC/BIC/MDL이 현대 ML에서 중요한가

Ch7-01의 SRM은 **VC 기반 이론적 penalty**를 제공했다. 하지만 실전에서 대안이 많다 — AIC, BIC, CV. 이들은 **원칙적인 통계 이론**에서 나온 것들이고, 각각 다른 목표를 최적화한다:

- **AIC**: 미래 예측 정확도 최대화 (Kullback-Leibler loss)
- **BIC**: 진짜(참) 모델의 사후확률 최대화 (Bayesian)
- **MDL**: 데이터와 모델을 합쳐 설명하는 길이 최소화

SRM은 "복잡도 + 경험위험"의 **원칙적 공식화**지만, AIC/BIC는 **통계적 해석**을 더한다. 현대 practice에선 CV(Ch7-03)가 가장 널리 쓰이지만, 계산 비용이 많으므로 AIC/BIC를 빠른 근사로 쓴다. 또한 **DL 시대에도** "모델 크기" 대신 "early stopping 시점"을 정하는 데 AIC/BIC 스타일의 정보이론 직관이 도움된다.

---

## 📐 수학적 선행 조건

- Ch1-01, Ch1-02 (위험, Bayes 최적)
- Ch2-02 (Hoeffding 부등식)
- [Probability Theory](https://github.com/iq-ai-lab/probability-theory-deep-dive): KL divergence, relative entropy
- [Mathematical Statistics](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): MLE, Fisher Information
- [Calculus & Optimization](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Taylor 전개

---

## 📖 직관적 이해

### "Best model for prediction"와 "Best model for truth" 구분

우리의 목표가 **무엇**인지가 중요하다:

1. **예측 정확도 최대화**: "미래 데이터를 최대한 정확히 맞추고 싶다"
   - KL divergence $D_\text{KL}(p^* \| p_\theta)$ 최소화 (참과 예측 분포의 차이)
   - 이것을 최소화하는 모델을 고르기 → **AIC**

2. **참 모델 복원**: "세상을 생성하는 진짜 모델이 무엇인가?"
   - 만약 참 모델이 $\mathcal{M}_k$ 중 하나라면, 사후확률 $p(M_k | S)$ 최대화
   - 샘플이 충분하면 BIC가 참 모델에 확률 1로 수렴 (consistency)
   - → **BIC**

이 둘은 **다른 목표**다! AIC는 "나쁜 참 모델의 예측이 좋은 과포장 모델보다 나을 수 있다"는 가능성을 인정한다. BIC는 "충분한 데이터가 있으면 참 모델을 찾는다"고 보장한다 (표본 크기가 커질 때).

### 정보 이론적 직관

- **AIC**: $-2\ell + 2k$ = "**모델이 데이터를 얼마나 잘 설명하는가**" (음의 우도) + "**모델이 얼마나 복잡한가**" (parameter 수) × 2배
- **BIC**: $-2\ell + k\log n$ = 같은 구조이지만 penalty가 $k\log n$ (샘플 크기에 비례) — **큰 샘플에선 BIC가 AIC보다 더 단순한 모델 선호**
- **MDL**: "데이터를 전송하는 총 비트 수" = "모델을 설명하는 비트" + "모델 주어졌을 때 데이터를 설명하는 비트"

---

## ✏️ 엄밀한 정의

### 정의 7.4 (AIC)

주어진 샘플 $S$와 $k$개 매개변수를 갖는 모델 $M_k$에 대해:

$$\text{AIC}(M_k) = -2 \log \widehat{L}_k + 2k,$$

여기서 $\widehat{L}_k = \max_{\theta \in \Theta_k} L_S(\theta)$ = 샘플 $S$에 대한 **최대우도 추정값**.

모델 선택 규칙: $\hat{M} = \arg\min_{k} \text{AIC}(M_k)$.

### 정의 7.5 (BIC)

$$\text{BIC}(M_k) = -2 \log \widehat{L}_k + k \log n,$$

유사하게 $\widehat{L}_k$는 MLE이고, 유일한 차이는 penalty가 $2k$ 대신 $k \log n$.

### 정의 7.6 (MDL)

**설명 길이(Description Length)**:

$$L(S, M_k, \theta) = L(M_k) + L(S | M_k, \theta),$$

여기서:
- $L(M_k)$: 모델 $M_k$를 설명하는 **코드 길이** (비트)
- $L(S | M_k, \theta)$: 모델과 매개변수 $\theta$가 주어졌을 때 데이터 $S$의 **코드 길이**

**MDL 규칙**: $\hat{M}, \hat{\theta} = \arg\min_{k, \theta} L(S, M_k, \theta)$.

---

## 🔬 정리와 증명

### 정리 7.4 (AIC 유도 — KL divergence 해석)

$p^*$를 참 데이터 생성 분포, $p_\theta$를 모델 분포라 하자. **예측 성능의 척도**:

$$\text{Expected KL divergence} = \mathbb{E}_{S \sim (p^*)^n} \left[ D_\text{KL}(p^* \| p_{\widehat{\theta}}) \right],$$

이를 unbiased하게 추정하면 $\text{AIC}/2$이다.

**증명 스케치**.

1. KL divergence 분해:
$$D_\text{KL}(p^* \| p_\theta) = -\mathbb{E}_{p^*}[\log p_\theta(x)] + \mathbb{E}_{p^*}[\log p^*(x)].$$

2. 두 번째 항은 $\theta$에 무관하므로, KL 최소화 $\iff$ $-\mathbb{E}_{p^*}[\log p_\theta(x)]$ 최소화.

3. 경험적 버전: 샘플 $S$에서,
$$-\frac{1}{n}\sum_{i=1}^n \log p_{\hat{\theta}}(x_i) = -\frac{1}{n}\log L_S(\hat{\theta}).$$

4. **Bias correction** (Takeuchi, Shibata): MLE $\hat{\theta}$의 위험도는 기대값이:
$$\mathbb{E}[-\log p_{\hat{\theta}}(X)] \approx -\frac{1}{n}\log L_S(\hat{\theta}) + \frac{k}{n} + o(1/n),$$

여기서 $k$는 매개변수 수, 계수는 **Fisher Information 관련 기술적 사항**.

5. 양변에 $-2n$을 곱하고 정리하면:
$$\text{Expected out-of-sample KL} \approx \frac{-2\log L_S(\hat{\theta}) + 2k}{n}.$$

6. 따라서 AIC $= -2\log L_S(\hat{\theta}) + 2k$는 **out-of-sample KL loss의 비편향 추정량**(Akaike, 1973). $\square$

### 정리 7.5 (BIC 유도 — Laplace Approximation)

모델 $M_k$의 **마지널 우도(marginal likelihood)**:

$$p(S | M_k) = \int L_S(\theta) p(\theta | M_k) \, d\theta.$$

Laplace approximation을 적용하면:

$$\log p(S | M_k) \approx \log L_S(\hat{\theta}) - \frac{k}{2} \log \frac{2\pi}{n} - \frac{1}{2} \log |I(\hat{\theta})|,$$

여기서 $I(\hat{\theta})$는 Fisher Information matrix.

따라서:
$$-2 \log p(S | M_k) \approx -2\log L_S(\hat{\theta}) + k \log n + O(1).$$

**Bayes model selection** (균등 사전 $p(M_k)$ 가정):

$$\text{Choose } M_k = \arg\max_k p(M_k | S) \propto p(S | M_k),$$

$$\Leftrightarrow \arg\min_k [-2 \log p(S | M_k)] = \arg\min_k \text{BIC}(M_k). \quad \square$$

### 정리 7.6 (MDL과 BIC의 확률적 동치)

Universal code 이론 (Rissanen, 1978)에서, 데이터 길이를 **나트(nats)로** 표현하면:

$$\text{Shortest description length} \approx \text{BIC}/2 = -\log L_S(\hat{\theta}) + \frac{k}{2} \log n + O(1).$$

따라서 **BIC를 최소화하기 = 정보 이론적 description length 최소화하기**.

**증명**: Universal code의 prefix code 비용이 parameter space에 대해 $\frac{k}{2} \log n$임을 information theory에서 보일 수 있고, 이것이 정확히 BIC의 penalty 부분. 상세한 증명은 Rissanen, Barron 참고. $\square$

### 정리 7.7 (BIC의 Consistency — 참 모델 복원)

$\mathcal{M}_1, \mathcal{M}_2$ 중 참 모델을 생성하는 분포가 $\mathcal{M}^*$ (finite parameter)라 하자. 그러면 $n \to \infty$일 때:

$$\mathbb{P}(\text{BIC selects } M^*) \to 1.$$

한편 AIC는 이 성질을 갖지 않는다 — "더 좋은 예측"을 위해 참보다 복잡한 모델을 선택할 수 있다.

**증명 스케치**: Nested 모델 $\mathcal{M}_1 \subset \mathcal{M}_2$인 경우, $M^* = M_1$이면
$$\text{BIC}(M_1) - \text{BIC}(M_2) = -2(\log L_1 - \log L_2) + (k_1 - k_2) \log n,$$

여기서 $\log L_1 - \log L_2 = O_p(1)$ (bounded), 하지만 $(k_2 - k_1) \log n \to \infty$. 따라서 $\mathbb{P}(\text{BIC selects } M_1) \to 1$. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1: 선형 회귀에서 AIC vs BIC

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

rng = np.random.default_rng(42)

# 진짜 모델: y = 1 + 2x + 3x^2 + noise
def sample(n):
    x = rng.uniform(-1, 1, n)
    y_true = 1 + 2*x + 3*x**2
    y = y_true + 0.2 * rng.standard_normal(n)
    return x.reshape(-1, 1), y

n_train = 50
x_train, y_train = sample(n_train)

# 다양한 polynomial degree로 모델 적합
degrees = np.arange(1, 8)
n_test = 1000
x_test, y_test = sample(n_test)

aic_scores = []
bic_scores = []
test_errors = []

for d in degrees:
    # Polynomial features
    X_train = np.column_stack([x_train**i for i in range(1, d+1)])
    X_test = np.column_stack([x_test**i for i in range(1, d+1)])
    
    # Fit
    model = LinearRegression().fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Residual sum of squares
    rss = np.sum((y_train - y_pred_train)**2)
    
    # MLE 우도 (정규분포 가정)
    sigma2_mle = rss / n_train
    log_likelihood = -n_train/2 * np.log(2*np.pi*sigma2_mle) - rss/(2*sigma2_mle)
    
    # AIC, BIC
    aic = -2 * log_likelihood + 2 * d
    bic = -2 * log_likelihood + d * np.log(n_train)
    
    aic_scores.append(aic)
    bic_scores.append(bic)
    
    # Test error (oracle)
    test_error = np.mean((y_pred_test - y_test)**2)
    test_errors.append(test_error)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot 1: AIC vs BIC
axes[0].plot(degrees, aic_scores, 'o-', label='AIC', linewidth=2)
axes[0].plot(degrees, bic_scores, 's-', label='BIC', linewidth=2)
axes[0].axvline(degrees[np.argmin(aic_scores)], color='blue', linestyle='--', 
                alpha=0.5, label=f'AIC min: d={degrees[np.argmin(aic_scores)]}')
axes[0].axvline(degrees[np.argmin(bic_scores)], color='orange', linestyle='--', 
                alpha=0.5, label=f'BIC min: d={degrees[np.argmin(bic_scores)]}')
axes[0].set_xlabel('Polynomial degree $d$')
axes[0].set_ylabel('Information Criterion')
axes[0].set_title('AIC vs BIC — BIC가 더 단순 모델 선호 (큰 $n$에서)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: 선택된 모델의 test error
d_aic = degrees[np.argmin(aic_scores)]
d_bic = degrees[np.argmin(bic_scores)]
axes[1].plot(degrees, test_errors, 'o-', linewidth=2, label='Test error')
axes[1].axvline(d_aic, color='blue', linestyle='--', alpha=0.5)
axes[1].axvline(d_bic, color='orange', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Polynomial degree $d$')
axes[1].set_ylabel('Test MSE')
axes[1].set_title('AIC vs BIC의 선택과 실제 test error')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f'True model: degree 2')
print(f'AIC selects: degree {d_aic}')
print(f'BIC selects: degree {d_bic}')
print(f'Test error at AIC choice: {test_errors[np.argmin(aic_scores)]:.4f}')
print(f'Test error at BIC choice: {test_errors[np.argmin(bic_scores)]:.4f}')
```

### 실험 2: 샘플 크기에 따른 BIC의 consistency

```python
# 샘플 크기 n을 키우면서 BIC가 참 모델을 선택하는 비율 추적
ns = [30, 50, 100, 200, 500]
n_trials = 100
true_degree = 2

# 각 n에서 여러 샘플로 BIC 테스트
bic_selects_truth = []

for n in ns:
    count = 0
    for trial in range(n_trials):
        x, y = sample(n)
        
        bic_values = []
        for d in degrees:
            X = np.column_stack([x**i for i in range(1, d+1)])
            model = LinearRegression().fit(X, y)
            rss = np.sum((y - model.predict(X))**2)
            sigma2 = rss / n
            ll = -n/2 * np.log(2*np.pi*sigma2) - rss/(2*sigma2)
            bic = -2*ll + d*np.log(n)
            bic_values.append(bic)
        
        if np.argmin(bic_values) + 1 == true_degree:  # degree는 1부터 시작
            count += 1
    
    prob_correct = count / n_trials
    bic_selects_truth.append(prob_correct)

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(ns, bic_selects_truth, 'o-', linewidth=2, markersize=8)
ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Target: 100%')
ax.set_xlabel('Sample size $n$')
ax.set_ylabel('P(BIC selects true model)')
ax.set_title('BIC의 Consistency — $n$ 증가에 따른 참 모델 선택 확률')
ax.set_ylim([0, 1.1])
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# → BIC가 샘플 크기가 커질수록 참 모델(degree 2)을 더 높은 확률로 선택함을 관찰.
```

---

## 🔗 ML 알고리즘 연결

| 선택 방법 | 핵심 목표 | 공식 | 특징 |
|---------|---------|------|------|
| **AIC** | 예측 KL 최소화 | $-2\log L + 2k$ | 모든 모델, 크기 $n$ 무관 |
| **BIC** | 참 모델 복원 (Bayesian) | $-2\log L + k\log n$ | 큰 $n$: 단순 모델 선호 |
| **MDL** | 설명 길이 최소화 | $\log L + k\log n/2$ | 정보이론, BIC와 동치 |
| **Cross-Validation (Ch7-03)** | 일반화 오차 직접 추정 | $\frac{1}{K}\sum \text{test error}$ | 더 정확, 계산 비용 높음 |

**관찰**: $n$이 충분하면 BIC가 더 단순한 모델을 선호하고, 실제로 참 모델을 복원한다. 하지만 "참 모델이 없거나" "모든 모델이 부정확"한 실전에선 AIC나 CV가 예측성능 면에서 더 낫다.

---

## ⚖️ 가정과 한계

1. **MLE 가정**: AIC/BIC는 최대우도 추정에 기반 → 비볼록 손실(NN)에서 global optimum 보장 없음.

2. **정규성 가정**: 정규분포/지수족 외의 분포에선 정확한 유도 불가능, 하지만 경험적으로 많은 경우 작동.

3. **아래첨자(nested model) 제약**: AIC/BIC는 "모델 A와 B 중 선택"에 자연스러움. "A, B, C 모두 고려" 시 여러 비교 필요 → multiple testing 보정 고려.

4. **고차원 한계**: $k \gg n$ 인 경우(DL) 이론이 깨짐. Regularization 또는 stability 기반 접근 필요.

5. **모델 구조 이외의 복잡도**: parameter 수만 세는 것이 모델 capacity를 완전히 반영하지 못함 (implicit bias, overparameterization).

---

## 📌 핵심 정리

- **AIC $= -2\log L + 2k$**: out-of-sample KL divergence의 **비편향 추정량**. 예측 성능 최적화.
- **BIC $= -2\log L + k\log n$**: 마지널 우도의 Laplace 근사. 참 모델 **사후확률** 최대화. 큰 $n$에서 consistency.
- **MDL**: 정보이론적 관점. "모델 + 데이터의 설명 길이" 최소화. BIC와 점근적 동치.
- **AIC vs BIC**: AIC는 예측 성능이 최적, BIC는 진짜 모델 선택이 optimal. 목표에 따라 선택.
- **실전 선택**:
  - $n$이 크고 "참 모델이 class에 있다" 믿으면 → BIC
  - 예측 성능만 중요하면 → AIC
  - 계산 여유 있으면 → CV (Ch7-03)

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> AIC와 BIC의 penalty 항을 비교하라: $2k$ vs $k\log n$. $n=50$, $n=1000$일 때 각각 어떤 penalty가 더 큰가? ($k=5$일 때)</summary>

<br/>

**해설**. 
- $k=5$, $n=50$: AIC penalty $= 2 \times 5 = 10$, BIC penalty $= 5 \log 50 \approx 5 \times 3.91 = 19.55$. **BIC가 더 크다**.
- $k=5$, $n=1000$: AIC penalty $= 10$, BIC penalty $= 5 \log 1000 \approx 5 \times 6.91 = 34.55$. **BIC가 훨씬 더 크다**.

따라서 **$n$이 커질수록 BIC가 AIC보다 더 강하게 단순 모델을 선호**한다. 이것이 "BIC는 consistency, AIC는 예측 성능"이라는 성질의 수학적 근거다. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> 정규분포 가정 아래 선형 회귀 $y = X\beta + \epsilon$, $\epsilon \sim N(0, \sigma^2)$에서 잔차제곱합(RSS)로부터 MLE $\hat{\sigma}^2 = \text{RSS}/n$을 유도하고, 로그우도를 명시적으로 써라.</summary>

<br/>

**해설**. 우도함수:
$$L(\beta, \sigma^2 | X, y) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - x_i^\top\beta)^2}{2\sigma^2}\right) = (2\pi\sigma^2)^{-n/2} \exp\left(-\frac{\text{RSS}}{2\sigma^2}\right).$$

로그우도:
$$\log L = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{\text{RSS}}{2\sigma^2}.$$

$\hat{\sigma}^2$에 대해 미분 $= 0$:
$$\frac{\partial}{\partial \sigma^2} \left[-\frac{n}{2}\log\sigma^2 - \frac{\text{RSS}}{2\sigma^2}\right] = -\frac{n}{2\sigma^2} + \frac{\text{RSS}}{2\sigma^4} = 0 \Rightarrow \hat{\sigma}^2 = \frac{\text{RSS}}{n}.$$

따라서:
$$\log L_{\max} = -\frac{n}{2}\log(2\pi \text{RSS}/n) - \frac{n}{2} = -\frac{n}{2}[\log \text{RSS} - \log n + \log(2\pi) + 1].$$

AIC $= -2\log L_{\max} + 2k = n[\log \text{RSS} - \log n + \log(2\pi) + 1] + 2k$. 상수 제외 $\propto \log \text{RSS} + 2k/n$. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> "Rethinking Generalization" (Zhang et al. 2017) 실험: ResNet이 ImageNet 데이터에서 임의 라벨(random labels)을 완벽히 암기한다는 관찰이, AIC/BIC 입장에서 무엇을 의미하는가? 왜 고전적 정보기준이 DL에서 실패하는가?</summary>

<br/>

**해설**. 라벨을 random하게 섞어도 ResNet은 $L_S(\text{DNN}) = 0$ 달성 가능 (overfitting). 이 경우:

1. **AIC/BIC 입장**: parameter 수 $k \approx 10^7$ (ResNet), $n = 10^6$ (ImageNet). 따라서:
   - AIC penalty $= 2k \approx 2 \times 10^7$ (거대)
   - BIC penalty $= k \log n \approx 10^7 \times 14 \approx 1.4 \times 10^8$ (더 거대)
   - $\log L_S \approx 0$ (perfect fit). 최악 경우 둘 다 "최악의 선택"이라 표시.

2. **그런데 실제로는?**: 진짜 라벨에선 일반화 잘 됨. 이는:
   - Parameter 수가 모든 복잡도를 반영하지 않음 (implicit bias, 구조)
   - MLE 가정 깨짐 (non-convex, SGD의 암묵적 정규화)
   - 정보기준이 "capacity" 중심, "실제 알고리즘 복잡도" 외면

3. **해결 방향**:
   - Effective parameter (Rademacher, stability)
   - Margin 기반 복잡도 (Ch5-05)
   - PAC-Bayes, NTK (Layer 2 Generalization Theory)

결론: 고전 AIC/BIC는 **정규 표본 크기 $n \gg k$ 영역**을 가정. DL의 "큰 모델, 작은 데이터도 일반화" 현상은 고전 이론 밖 → 현대 이론 필요. $\square$

</details>

---

<div align="center">

◀ [이전: 01. SRM](./01-srm.md) | [📚 README](../README.md) | [다음: 03. Cross-Validation ▶](./03-cross-validation.md)

</div>
