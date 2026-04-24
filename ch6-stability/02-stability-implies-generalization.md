# 02. Stability가 Generalization을 함의

## 🎯 핵심 질문

- **$\beta$-uniformly stable 알고리즘이라면, 일반화 gap이 왜 $\beta$에 비례해야 하는가?**
- "Rename trick" — 동일분포 $z_i$와 $z_i'$를 교환해 기대값의 특성을 어떻게 이용하는가?
- 고확률 경계(high-probability)를 얻으려면 왜 **McDiarmid 부등식**이 필수인가?
- Generalization gap의 두 가지 형태: **(평균적) 경계** vs **(고확률) 경계**는 무엇이 다른가?

---

## 🔍 왜 이 정리가 중요한가

Ch2에서 배운 집중부등식들(Hoeffding·McDiarmid)은 모두 **고정된 함수** $f(S)$의 편차를 bound한다. 하지만 학습에서는:
$$f(S) := L_\mathcal{D}(A(S)) - L_S(A(S))$$
이라는 특수한 형태를 갖는다. 이 함수는:
1. **Unbounded**: 손실이 $[0, M]$에 있어도, 차이는 원칙적으로 크다
2. **Data-dependent**: 알고리즘 $A$가 $S$에 의존하므로 복잡하다

**정리 6.2**는 이 특수한 구조를 이용해, 만약 알고리즘이 **stable**하다면 (즉, 작은 데이터 변화에 sensitive하지 않다면), McDiarmid를 직접 적용하는 것보다 훨씬 **깔끔한 경계**를 얻을 수 있음을 보인다. 이것이 현대 ML에서 stability 분석이 강력한 이유다.

---

## 📐 수학적 선행 조건

- Ch1-01~03: 위험 $L_\mathcal{D}$, $L_S$, 3분해
- Ch2-03: McDiarmid 부등식 (bounded differences)
- Ch6-01: Uniform stability의 정의
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 기대값의 선형성, 확률변수의 독립성

---

## 📖 직관적 이해

### "샘플 하나를 바꿔도 기대값은 비슷하다"

두 가지 샘플 모음을 생각하자:
- **$S$**: 데이터 분포 $\mathcal{D}$에서 iid로 뽑은 $n$개 샘플
- **$S'$**: $S$에서 **$i$-번째 샘플만 새로 뽑은** 샘플 (즉 $S' = S^{(i)}$)

$z_i$와 $z_i'$는 모두 **같은 분포 $\mathcal{D}$에서 독립적으로** 뽑혔으므로, **분포적으로 동일**하다. (exchangeability)

**Rename trick의 아이디어**:
$$\mathbb{E}_S[\ell(A(S), z_i)] = \mathbb{E}_{S^{(i)}}[\ell(A(S^{(i)}), z_i')]$$

왜? $z_i$와 $z_i'$ 모두 $\mathcal{D}$에서 같은 방식으로 뽑혔으므로. 따라서:
$$\mathbb{E}_S[\ell(A(S), z_i)] - \mathbb{E}_{S^{(i)}}[\ell(A(S), z_i)] \leq \mathbb{E}[\text{stability 항}] \approx \beta$$

이것이 일반화 경계의 핵심 계산이다.

---

## ✏️ 엄밀한 정의

### 정의 6.2.1 (Generalization Gap)

학습 알고리즘 $A$, iid 샘플 $S \sim \mathcal{D}^n$, 그리고 손실 $\ell \in [0, M]$에 대해:

**Generalization gap**은
$$\text{Gap}_S := L_\mathcal{D}(A(S)) - L_S(A(S))$$

이것은 **$S$에 의존하는 확률변수**다. 우리는 이 gap의 기대값 혹은 고확률 bound를 구한다.

---

## 🔬 정리와 증명

### 정리 6.2 (Uniform Stability ⇒ 평균적 일반화)

**가정**: 
- 알고리즘 $A$가 $\beta$-uniformly stable
- 손실 함수 $\ell: \mathcal{Y} \times \mathcal{Y} \to [0, M]$ (유계)
- 샘플 $S \sim \mathcal{D}^n$ iid

**결론**: 
$$\mathbb{E}_S[L_\mathcal{D}(A(S)) - L_S(A(S))] \leq \beta.$$

**증명**:

기대값의 선형성으로 경험 위험을 분해하면:
$$\mathbb{E}_S[L_S(A(S))] = \mathbb{E}_S\left[\frac{1}{n}\sum_{i=1}^n \ell(A(S), z_i)\right] = \frac{1}{n}\sum_{i=1}^n \mathbb{E}_S[\ell(A(S), z_i)].$$

각 항에 대해 **rename trick**을 적용한다. $z_i' \sim \mathcal{D}$는 $z_i$와 독립적이고 같은 분포를 따르므로:
$$\mathbb{E}_S[\ell(A(S), z_i)] = \mathbb{E}_{S^{(i)}, z_i'}[\ell(A(S^{(i)}), z_i')]$$

따라서:
\begin{align}
\mathbb{E}_S[\ell(A(S), z_i)] &= \mathbb{E}_{S^{(i)}, z_i'}[\ell(A(S^{(i)}), z_i')] \\
&= \mathbb{E}_{S^{(i)}}[\mathbb{E}_{z_i'}[\ell(A(S^{(i)}), z_i')]] \\
&= \mathbb{E}_{S^{(i)}}[L_\mathcal{D}(A(S^{(i)}))]
\end{align}

여기서 두 번째 줄은 $S^{(i)}$를 고정했을 때 $z_i'$에 대한 기대값.

다시 정렬하면:
\begin{align}
\mathbb{E}_S[\ell(A(S), z_i)] - \mathbb{E}_{S^{(i)}}[\ell(A(S), z_i')]  
&= \mathbb{E}[\ell(A(S), z_i)] - \mathbb{E}[\ell(A(S^{(i)}), z_i')]
\end{align}

이제 이 차이를 두 부분으로 나눈다:
\begin{align}
&\mathbb{E}_S[\ell(A(S), z_i)] - \mathbb{E}_{S^{(i)}}[L_\mathcal{D}(A(S^{(i)}))] \\
&= \mathbb{E}_S[\ell(A(S), z_i)] - \mathbb{E}_S[\ell(A(S), z_i')] + \mathbb{E}[\ell(A(S), z_i')] - \mathbb{E}[\ell(A(S^{(i)}), z_i')] \\
&\leq \mathbb{E}_{S, z_i'}[|\ell(A(S), z_i') - \ell(A(S^{(i)}), z_i')|] \quad \text{(by exchangeability)} \\
&\leq \beta \quad \text{(by uniform stability).}
\end{align}

따라서 $i$-번째 항에 대해:
$$\mathbb{E}_S[\ell(A(S), z_i)] \leq \mathbb{E}_S[L_\mathcal{D}(A(S))] + \beta.$$

모든 $i$를 합치면:
$$\frac{1}{n}\sum_{i=1}^n \mathbb{E}_S[\ell(A(S), z_i)] \leq \mathbb{E}_S[L_\mathcal{D}(A(S))] + \beta.$$

즉:
$$\mathbb{E}_S[L_S(A(S))] \leq \mathbb{E}_S[L_\mathcal{D}(A(S))] + \beta.$$

정렬하면:
$$\mathbb{E}_S[L_\mathcal{D}(A(S)) - L_S(A(S))] \leq \beta. \qquad \square$$

### 정리 6.3 (고확률 일반화 경계)

**가정**: 
- 알고리즘 $A$가 $\beta$-uniformly stable
- 손실 $\ell \in [0, M]$
- 샘플 $S \sim \mathcal{D}^n$ iid

**결론**: 확률 $\geq 1 - \delta$로
$$L_\mathcal{D}(A(S)) - L_S(A(S)) \leq \beta + O\left(\sqrt{\frac{M \log(1/\delta)}{n}}\right).$$

더 구체적으로,
$$\mathbb{P}\left(L_\mathcal{D}(A(S)) - L_S(A(S)) \geq \beta + \epsilon\right) \leq 2\exp\left(-2n\epsilon^2 / (n\beta + M)^2\right).$$

**증명 스케치**:

함수 $f(S) := L_\mathcal{D}(A(S)) - L_S(A(S))$에 McDiarmid 부등식을 적용한다. 

핵심은 한 샘플 교체 $z_i \to z_i'$에 대한 Lipschitz 상수를 구하는 것이다:
$$|f(S) - f(S^{(i)})| = |L_\mathcal{D}(A(S)) - L_S(A(S)) - L_\mathcal{D}(A(S^{(i)})) + L_S(A(S^{(i)}))|$$

각 항을 bound하면:
\begin{align}
|L_\mathcal{D}(A(S)) - L_\mathcal{D}(A(S^{(i)}))| &\leq \beta \quad \text{(stability)} \\
|L_S(A(S)) - L_S(A(S^{(i)}))| &\leq |L_S(A(S)) - L_S(A(S^{(i)}))|
\end{align}

두 번째 항에서, $S$와 $S^{(i)}$는 하나의 좌표만 다르므로, 평균값 $\frac{1}{n}\sum$의 변화는:
$$|L_S(A(S)) - L_S(A(S^{(i)})| \leq \frac{1}{n}[\ell(A(S), z_i) - \ell(A(S^{(i)}), z_i)] + \frac{1}{n}\ell(A(S^{(i)}), z_i')|$$

$\ell \in [0, M]$이므로:
$$|L_S(A(S)) - L_S(A(S^{(i)}))| \leq \frac{2M}{n}.$$

결합하면:
$$|f(S) - f(S^{(i)})| \leq \beta + \frac{2M}{n}.$$

McDiarmid (Ch2-03)를 적용하면 $c_i = \beta + 2M/n$이고:
$$\mathbb{P}(|f - \mathbb{E}f| \geq \epsilon) \leq 2\exp\left(-\frac{2n\epsilon^2}{n(\beta + 2M/n)^2}\right) = 2\exp\left(-\frac{2n\epsilon^2}{(n\beta + 2M)^2}\right).$$

정리 6.2에서 $\mathbb{E}[f] \leq \beta$이므로:
$$\mathbb{P}(f \geq \beta + \epsilon) \leq 2\exp\left(-\frac{2n\epsilon^2}{(n\beta + 2M)^2}\right). \qquad \square$$

---

## 💻 NumPy 구현 검증

### 실험: Leave-One-Out (LOO) Cross-Validation과 Stability

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 설정: 1D 회귀
# y = sin(2πx) + noise
# ─────────────────────────────────────────────

def generate_data(n, noise_std=0.1):
    X = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * X) + noise_std * rng.standard_normal(n)
    return X.reshape(-1, 1), y

# Ridge regression의 안정성 확인
lambdas = [0.001, 0.01, 0.1, 1.0, 10.0]
n_samples = 50
n_trials = 200
results = {lam: [] for lam in lambdas}

for lam in lambdas:
    gaps = []
    
    for trial in range(n_trials):
        X, y = generate_data(n_samples)
        
        # 알고리즘 A: Ridge(lambda)
        ridge = Ridge(alpha=lam, fit_intercept=True)
        ridge.fit(X, y)
        
        # L_S(A(S)): 훈련 오차
        train_pred = ridge.predict(X)
        train_loss = np.mean((train_pred - y) ** 2)
        
        # L_D(A(S)): 테스트 오차 (많은 테스트 샘플로 근사)
        X_test, y_test = generate_data(5000, noise_std=0.1)
        test_pred = ridge.predict(X_test)
        test_loss = np.mean((test_pred - y_test) ** 2)
        
        gap = test_loss - train_loss
        gaps.append(gap)
    
    results[lam] = np.array(gaps)

# 결과 플롯
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (1) Lambda와 Gap의 관계
mean_gaps = [np.mean(results[lam]) for lam in lambdas]
std_gaps = [np.std(results[lam]) for lam in lambdas]

axes[0].errorbar(np.log10(lambdas), mean_gaps, yerr=std_gaps, marker='o', capsize=5)
axes[0].set_xlabel('log10(lambda)')
axes[0].set_ylabel('Mean Generalization Gap')
axes[0].set_title('Ridge Regression: Lambda vs Generalization Gap')
axes[0].grid(True)

# (2) 여러 lambda에서의 gap 분포
for lam in [0.001, 0.1, 10.0]:
    axes[1].hist(results[lam], bins=20, alpha=0.5, label=f'λ={lam}', edgecolor='black')

axes[1].set_xlabel('Generalization Gap')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Gaps Across Lambda')
axes[1].legend()

plt.tight_layout()
plt.savefig('/tmp/stability_generalization.png', dpi=100)
print("✓ Plot saved")

# ─────────────────────────────────────────────
# 정리 6.3 검증: McDiarmid bound vs 경험적 gap
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("정리 6.3 검증: McDiarmid 경계 vs 경험적 Gap")
print("="*60)

n_samples_range = [20, 50, 100, 200]
M = 4.0  # loss bound (MSE는 대략 [0, 4])
beta = 0.1  # approximate stability (ridge)
delta = 0.1

empirical_gaps = []
mcdarmid_bounds = []

for n in n_samples_range:
    gaps = []
    for _ in range(500):
        X, y = generate_data(n, noise_std=0.1)
        ridge = Ridge(alpha=0.1)
        ridge.fit(X, y)
        
        train_loss = np.mean((ridge.predict(X) - y)**2)
        X_test, y_test = generate_data(1000)
        test_loss = np.mean((ridge.predict(X_test) - y_test)**2)
        gaps.append(test_loss - train_loss)
    
    emp_gap = np.mean(gaps)
    empirical_gaps.append(emp_gap)
    
    # McDiarmid bound from 정리 6.3
    epsilon = np.sqrt((2 * np.log(1/delta)) / (2*n))
    mcd_bound = beta + (n*beta + M) * epsilon / n
    mcdarmid_bounds.append(mcd_bound)

plt.figure(figsize=(8, 5))
plt.plot(n_samples_range, empirical_gaps, 'o-', label='Empirical Mean Gap', markersize=8)
plt.plot(n_samples_range, mcdarmid_bounds, 's--', label='McDiarmid Bound (δ=0.1)', markersize=8)
plt.xlabel('Sample Size n')
plt.ylabel('Generalization Gap')
plt.title('정리 6.3: McDiarmid Bound vs Empirical Gap')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/tmp/mcdarmid_bound.png', dpi=100)
print("✓ McDiarmid bound plot saved")

print(f"\nExperimental verification:")
print(f"  n        | Empirical Gap | McDiarmid Bound")
print(f"  {'-'*45}")
for n, emp, mcd in zip(n_samples_range, empirical_gaps, mcdarmid_bounds):
    print(f"  {n:4d}     | {emp:13.6f} | {mcd:13.6f}")
```

**결과 해석**:
1. Lambda가 크면 (strong regularization) gap이 작다 → 안정성 개선
2. 표본 크기 $n$이 크면 bound가 $1/\sqrt{n}$로 감소 (이론과 일치)
3. McDiarmid bound는 실제 gap보다 여유 있게(loose) 나온다 — 이것이 이론적 bound의 특성

---

## 🔗 ML 알고리즘 연결

### Ridge Regression과 Generalization
정리 6.3을 $\beta = O(1/(\lambda n))$ (정리 6.3에서)과 결합하면:
$$\text{Gap} \leq \frac{C}{\lambda n} + O\left(\sqrt{\frac{\log(1/\delta)}{n}}\right)$$

따라서 $\lambda$ 증가 → gap 감소. 이것이 "정규화 강도가 높을수록 일반화가 좋다"의 수학적 근거.

### Early Stopping
정리 6.4 (SGD stability)에서 $\beta = O(\eta T / n)$이므로, 훈련 단계 $T$를 줄이면 gap이 감소한다.

---

## ⚖️ 가정과 한계

### 한계 1: 손실의 경계성 필수
정리 6.3의 고확률 bound는 $\ell \in [0, M]$을 가정한다. Unbounded loss (e.g., squared loss on $\mathbb{R}$)는 다르게 취급해야 한다.

### 한계 2: Loose Bound
정리 6.3의 bound는 보통 **실험적 gap보다 훨씬 크다** (실험 참조). 이는 이론적 worst-case bound의 특성.

### 한계 3: Stability 추정의 어려움
실제로 $\beta$를 추정하기는 어렵다. 정리 6.3은 $\beta$를 알 때의 bound이지, $\beta$를 찾는 방법이 아니다.

---

## 📌 핵심 정리

1. **정리 6.2**: $\beta$-uniform stability ⇒ $\mathbb{E}[\text{Gap}] \leq \beta$
   - 증명: rename trick + 기대값의 선형성

2. **정리 6.3**: 고확률 bound는 $\text{Gap} \leq \beta + O(\sqrt{\log(1/\delta)/n})$
   - 증명: McDiarmid + Lipschitz 계산

3. **핵심 메시지**: Stability가 작으면 ($\beta$ 작음) 일반화가 좋다 (gap 작음)

4. **Regularization의 정당화**: Strong convex 정규화나 early stopping은 stability를 개선해 gap을 줄인다.

---

## 🤔 생각해볼 문제

### 문제 6.2.1 (기초)
**문제**: 정리 6.2의 증명에서 "rename trick"을 사용했다. $z_i$와 $z_i'$를 교환할 수 있는 수학적 근거는 무엇인가?

<details>
<summary><b>해설</b></summary>

두 확률변수 $z_i$와 $z_i'$가:
1. **같은 분포에서** 뽑히고 ($\mathcal{D}$에서 iid)
2. **독립적으로** 뽑혔다

이를 **exchangeability** 혹은 "동치성"이라 한다. 따라서 $z_i$와 $z_i'$를 교환해도 joint distribution이 바뀌지 않으므로:
$$\mathbb{E}[\ell(A(S), z_i)] = \mathbb{E}[\ell(A(S^{(i)}), z_i')].$$

(단, $A(S)$는 $S$의 다른 샘플들에 의존하지만, $z_i'$는 $S$와 독립이므로 이 등식이 성립.)

</details>

### 문제 6.2.2 (심화)
**문제**: 정리 6.3에서 샘플 하나의 교체가 경험 위험 $L_S$를 최대 $2M/n$만큼 바꾼다고 했다. 왜 정확히 $2M/n$인가?

<details>
<summary><b>해설</b></summary>

$S$와 $S^{(i)}$는 $i$-번째 항만 다르다:
$$L_S(A(S)) = \frac{1}{n}[\ell(A(S), z_i) + \sum_{j \neq i} \ell(A(S), z_j)]$$
$$L_S(A(S^{(i)})) = \frac{1}{n}[\ell(A(S^{(i)}), z_i') + \sum_{j \neq i} \ell(A(S), z_j)]$$

따라서 차이는 (approximation) $A$가 약간 달라지므로:
$$|L_S(A(S)) - L_S(A(S^{(i)})| \approx \frac{1}{n}|\ell(A(S), z_i) - \ell(A(S^{(i)}), z_i')|$$

$\ell$의 값이 $[0, M]$ 범위이므로 최악의 경우:
$$\frac{1}{n} \cdot M + \frac{1}{n} \cdot M = \frac{2M}{n}.$$

</details>

### 문제 6.2.3 (ML 연결)
**문제**: "Stability는 VC·Rademacher와 달리 $\mathcal{H}$-독립적"이라는 주장을 정리 6.2·6.3과 연결해 설명하시오.

<details>
<summary><b>해설</b></summary>

정리 6.2·6.3의 bound:
$$\text{Gap} \leq \beta + O(\sqrt{\log(1/\delta)/n})$$

에서 $\beta$는:
- **알고리즘** $A$와
- **손실** $\ell$, **샘플 크기** $n$에만 의존

하지만 **가설공간 $\mathcal{H}$에는 의존하지 않는다!**

반면 VC나 Rademacher bound는:
$$\text{Gap} \leq O\left(\sqrt{\frac{\text{complexity}(\mathcal{H})}{n}}\right)$$

로 $\mathcal{H}$의 크기/VC 차원에 의존한다.

이것이 stability의 강점: **같은 $\mathcal{H}$라도 다른 알고리즘을 쓰면 다른 bound를 얻는다**. 예를 들어:
- Unregularized ERM: 큰 $\beta$ (불안정)
- Ridge ERM: 작은 $\beta \propto 1/\lambda$ (안정)

둘 다 같은 $\mathcal{H}$를 쓰지만 알고리즘이 다르니까 gap이 다르다.

</details>

---

<div align="center">

◀ [이전: Uniform Stability의 정의](./01-uniform-stability.md) | [📚 README](../README.md) | [다음: Ridge Regression의 Stability ▶](./03-ridge-stability.md)

</div>
