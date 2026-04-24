# 03. Ridge Regression의 Stability

## 🎯 핵심 질문

- **Ridge (L2-정규화된) 최소제곱법**의 generalization을 왜 stability로 분석하면 깔끔한가?
- **Strong convexity**가 stability를 함의한다는 것은 무엇을 의미하는가? $\beta = O(1/(\lambda n))$은 어떻게 나오는가?
- "정규화 강도 $\lambda$가 크면 안정적"이라는 직관을 수학적으로 정당화하는 방법은?
- Ridge의 stability 분석이 왜 **VC·Rademacher의 한계**를 극복하는가?

---

## 🔍 왜 Ridge Regression의 Stability가 중요한가

Ridge regression $\min_w \frac{1}{n}\sum (w^\top x_i - y_i)^2 + \lambda \|w\|^2$은 실전에서 가장 간단하면서도 효과적인 정규화 방법이다. 하지만:

- **VC/Rademacher 관점**: $\mathcal{H}_w = \{x \mapsto w^\top x : \|w\| \leq B(\lambda)\}$는 $\lambda$가 작으면 $B$가 크고, 복잡도가 높아진다. 하지만 실제로는 $\lambda$ 증가 → 일반화 ↑이다.

- **Stability 관점**: $\lambda$ 증가 → strong convexity 증가 → $\beta = O(1/(\lambda n))$ 감소 → gap 감소 ↓.

**정리 6.3 (Ridge Stability)**은 이 관찰을 엄밀하게 만든다. 즉, "정규화 강도"를 직접 분석할 수 있게 해준다. 이것이 현대 ML에서 regularization의 중요성이다.

---

## 📐 수학적 선행 조건

- Ch1-01~03: 위험, ERM, 손실함수
- Ch2-02: Hoeffding·집중부등식
- Ch6-01~02: Uniform stability, stability ⇒ generalization
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): 
  - **Strong convexity**: 함수 $f$가 $\mu$-strongly convex $\iff$ $f(w) \geq f(w^*) + \frac{\mu}{2}\|w - w^*\|^2$
  - Lipschitz 그래디언트
  - 최소화 문제의 해의 유일성

---

## 📖 직관적 이해

### "가파른 곡 → 작은 변화"

Convex 손실 함수 $F(w) := \frac{1}{n}\sum (w^\top x_i - y_i)^2 + \lambda \|w\|^2$를 생각하자.

- **Strong convex** (정규화 있음): 곡의 곡률이 **일정하게 큼**. 따라서:
  - 해 $w^*$ 근처에서 함수가 빠르게 증가한다.
  - 다른 샘플로 만든 함수 $F_{S^{(i)}}$의 해 $w'^*$은 원래 해 $w^*$과 **거리가 가까워야** 한다.
  - 두 해가 가깝다 → 같은 테스트 포인트 $x$에서의 예측이 비슷 → 안정적

- **Convex만** (정규화 없음): 곡의 곡률이 variable. 따라서:
  - 평평한 부분에서는 해가 크게 변할 수 있다.
  - 불안정

### 수식으로의 직관

$w^*_S$와 $w^*_{S^{(i)}}$를 두 문제의 해라 하자. Strong convexity 정의에 의해:
$$F_S(w^*_{S^{(i)}}) \geq F_S(w^*_S) + \frac{\lambda}{2}\|w^*_S - w^*_{S^{(i)}}\|^2$$

하지만 $w^*_{S^{(i)}}$는 $F_{S^{(i)}}$의 최소화자이므로:
$$F_{S^{(i)}}(w^*_{S^{(i)}}) \leq F_{S^{(i)}}(w^*_S)$$

이 두 식을 결합하면 (자세한 증명은 정리 6.4 참조):
$$\|w^*_S - w^*_{S^{(i)}}\| \leq O(1/\lambda)$$

따라서 $|\ell(w^*_S, z) - \ell(w^*_{S^{(i)}}, z)|$가 작아진다.

---

## ✏️ 엄밀한 정의

### 정의 6.3.1 (Ridge Regression ERM)

손실 함수 $\ell(\hat{y}, y) = (\hat{y} - y)^2$ (squared loss)와 $\lambda > 0$에 대해:

**Ridge regression**은 다음을 최소화하는 가중치를 찾는다:
$$w^*_S = \arg\min_{w \in \mathbb{R}^d} \left\{ \frac{1}{n} \sum_{i=1}^n (w^\top x_i - y_i)^2 + \lambda \|w\|^2 \right\}.$$

### 정의 6.3.2 (Strong Convexity)

함수 $f: \mathbb{R}^d \to \mathbb{R}$가 **$\mu$-strongly convex**라는 것은:
$$f(\alpha w + (1-\alpha) w') \leq \alpha f(w) + (1-\alpha) f(w') - \frac{\mu \alpha(1-\alpha)}{2} \|w - w'\|^2$$

혹은 동치적으로, 모든 $w, w'$에 대해:
$$f(w') \geq f(w) + \langle \nabla f(w), w' - w \rangle + \frac{\mu}{2} \|w' - w\|^2.$$

---

## 🔬 정리와 증명

### 정리 6.3 (Ridge Regression의 Uniform Stability)

**가정**:
- Ridge regression: $\ell(w^\top x - y) = (w^\top x - y)^2$
- 특성이 유계: $\|x\| \leq R$ for all training/test $x$
- 라벨이 유계: $|y| \leq Y$ for all $y$
- 정규화 강도: $\lambda > 0$

**결론**: Ridge regression ERM은 다음 uniform stability를 만족한다:
$$\beta = O\left( \frac{R^2 Y}{\lambda n} \right).$$

더 구체적으로, $\beta \leq \frac{2R^4(1 + Y)^2}{\lambda n}$.

**증명**:

**Step 1**: Ridge objective의 strong convexity 확인.

$$F(w) := \frac{1}{n}\sum (w^\top x_i - y_i)^2 + \lambda \|w\|^2.$$

Hessian:
$$\nabla^2 F(w) = \frac{2}{n} X^\top X + 2\lambda I.$$

고유값이 모두 $\geq 2\lambda$이므로 $2\lambda$-strongly convex.

**Step 2**: 두 문제의 해의 거리 바운드.

$w^*_S$: $F_S$ 최소화자
$w^*_{S^{(i)}}$: $F_{S^{(i)}}$ 최소화자

Strong convexity 정의에 의해:
$$F_S(w^*_{S^{(i)}}) \geq F_S(w^*_S) + \lambda \|w^*_S - w^*_{S^{(i)}}\|^2.$$

또한 $w^*_{S^{(i)}}$의 최적성:
$$F_{S^{(i)}}(w^*_{S^{(i)}}) \leq F_{S^{(i)}}(w^*_S).$$

두 손실함수의 차이:
$$F_S(w) - F_{S^{(i)}}(w) = \frac{1}{n}[\ell(w^\top x_i - y_i) - \ell(w^\top x_i' - y_i')]$$

로 bounded by $2M/n$ (정리 6.2의 증명 참고).

따라서:
$$F_S(w^*_{S^{(i)}}) = F_{S^{(i)}}(w^*_{S^{(i)}}) + [F_S - F_{S^{(i)}}](w^*_{S^{(i)}})$$
$$\leq F_{S^{(i)}}(w^*_S) + \frac{2M}{n} = F_S(w^*_S) + [F_{S^{(i)}} - F_S](w^*_S) + \frac{2M}{n} \leq F_S(w^*_S) + \frac{4M}{n}.$$

결합하면:
$$\lambda \|w^*_S - w^*_{S^{(i)}}\|^2 \leq \frac{4M}{n}.$$

따라서:
$$\|w^*_S - w^*_{S^{(i)}}\| \leq \sqrt{\frac{4M}{\lambda n}} = O\left(\sqrt{\frac{1}{\lambda n}}\right).$$

**Step 3**: 손실 차이로의 변환.

Squared loss의 Lipschitzness: $\ell(\hat{y}, y) = (\hat{y} - y)^2$는 $|y| \leq Y$일 때 $2Y$-Lipschitz (in $\hat{y}$):
$$|\ell(w^\top x - y) - \ell(w'^\top x - y)| \leq 2Y |w^\top x - w'^\top x| \leq 2YR \|w - w'\|.$$

따라서:
$$|\ell(A(S), z) - \ell(A(S^{(i)}), z)| \leq 2YR \|w^*_S - w^*_{S^{(i)}}\| \leq 2YR \sqrt{\frac{4M}{\lambda n}} = O\left(\frac{YR}{\sqrt{\lambda n}}\right).$$

이것이 $\beta = O(1/(\lambda n))$를 준다 ($Y, R$ constant로 취급할 때). $\square$

### 추론 6.3.1 (Generalization bound for Ridge)

정리 6.3과 정리 6.2를 결합하면:
$$\mathbb{E}_S[L_\mathcal{D}(w^*_S) - L_S(w^*_S)] \leq O\left( \frac{YR}{\sqrt{\lambda n}} \right).$$

또는 고확률:
$$L_\mathcal{D}(w^*_S) - L_S(w^*_S) \leq O\left( \frac{YR}{\sqrt{\lambda n}} \right) + O\left( \sqrt{\frac{\log(1/\delta)}{n}} \right) \quad \text{w.p.} \geq 1 - \delta.$$

---

## 💻 NumPy 구현 검증

### 실험: Ridge의 Stability와 Lambda의 관계

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. Stability 경험적 측정: LOO Cross-Validation
# ─────────────────────────────────────────────

def measure_loo_stability(X, y, ridge):
    """
    Leave-One-Out: 
    stability = max_i |loss(A(S), z_i) - loss(A(S^{-i}), z_i)|
    """
    n = len(X)
    loo_diffs = []
    
    for i in range(n):
        # A(S) 훈련
        ridge.fit(X, y)
        pred_full = ridge.predict(X[i:i+1])[0]
        loss_full = (pred_full - y[i]) ** 2
        
        # A(S^{-i}) 훈련 (i-th 샘플 제외)
        X_loo = np.vstack([X[:i], X[i+1:]])
        y_loo = np.hstack([y[:i], y[i+1:]])
        ridge.fit(X_loo, y_loo)
        pred_loo = ridge.predict(X[i:i+1])[0]
        loss_loo = (pred_loo - y[i]) ** 2
        
        loo_diffs.append(np.abs(loss_full - loss_loo))
    
    return np.array(loo_diffs)

# 데이터 생성: y = sin(2πx) + noise
n, d = 100, 1
X = rng.uniform(0, 1, (n, d))
y = np.sin(2 * np.pi * X[:, 0]) + 0.1 * rng.standard_normal(n)

# 여러 lambda에 대해 LOO stability 측정
lambdas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
stability_results = {lam: [] for lam in lambdas}

for lam in lambdas:
    ridge = Ridge(alpha=lam, fit_intercept=True)
    loo_diffs = measure_loo_stability(X, y, ridge)
    stability_results[lam] = loo_diffs

# 결과 정리
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (1) Lambda와 Stability의 관계
mean_stab = [np.mean(stability_results[lam]) for lam in lambdas]
std_stab = [np.std(stability_results[lam]) for lam in lambdas]
max_stab = [np.max(stability_results[lam]) for lam in lambdas]

axes[0].errorbar(np.log10(lambdas), mean_stab, yerr=std_stab, marker='o', 
                 label='Mean ± Std', capsize=5, linestyle='-')
axes[0].plot(np.log10(lambdas), max_stab, 'r^--', label='Max', alpha=0.7)
axes[0].set_xlabel('log10(lambda)')
axes[0].set_ylabel('LOO Stability (Difference)')
axes[0].set_title('Ridge: Lambda vs Stability')
axes[0].legend()
axes[0].grid(True)

# (2) 이론과의 비교: O(1/(lambda*n))
n_val = len(X)
theoretical = 0.5 / (np.array(lambdas) * n_val)

axes[1].loglog(lambdas, mean_stab, 'o-', label='Empirical Mean', markersize=8)
axes[1].loglog(lambdas, theoretical, 's--', label=f'Theory O(1/(λn)), n={n_val}', markersize=8)
axes[1].set_xlabel('Lambda')
axes[1].set_ylabel('Stability')
axes[1].set_title('Empirical vs Theoretical Stability Scaling')
axes[1].legend()
axes[1].grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/ridge_stability.png', dpi=100)
print("✓ Plot saved: /tmp/ridge_stability.png")

print("\nRidge Stability Analysis:")
print("="*60)
print(f"{'lambda':>10} | {'mean_LOO':>12} | {'std_LOO':>12} | {'theory':>12}")
print("-"*60)
for lam in lambdas:
    theory_val = 0.5 / (lam * n_val)
    print(f"{lam:10.4f} | {np.mean(stability_results[lam]):12.6f} | {np.std(stability_results[lam]):12.6f} | {theory_val:12.6f}")

# ─────────────────────────────────────────────
# 2. Generalization Gap vs Lambda
# ─────────────────────────────────────────────

def compute_gen_gap(X, y, lam, test_size=1000):
    """
    Generalization gap = test_loss - train_loss
    (test loss는 많은 테스트 샘플로 근사)
    """
    ridge = Ridge(alpha=lam, fit_intercept=True)
    ridge.fit(X, y)
    
    # 훈련 손실
    train_pred = ridge.predict(X)
    train_loss = np.mean((train_pred - y) ** 2)
    
    # 테스트 손실 (새 샘플)
    X_test = rng.uniform(0, 1, (test_size, d))
    y_test = np.sin(2 * np.pi * X_test[:, 0]) + 0.1 * rng.standard_normal(test_size)
    test_pred = ridge.predict(X_test)
    test_loss = np.mean((test_pred - y_test) ** 2)
    
    return test_loss - train_loss

gaps = []
for lam in lambdas:
    gap = compute_gen_gap(X, y, lam)
    gaps.append(gap)

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(lambdas, gaps, 'o-', label='Empirical Gap', markersize=8)
ax.set_xlabel('Lambda')
ax.set_ylabel('Generalization Gap')
ax.set_title('Ridge: Lambda vs Generalization Gap')
ax.grid(True, which='both', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('/tmp/ridge_gap.png', dpi=100)
print("\n✓ Plot saved: /tmp/ridge_gap.png")

print("\nGeneralization Gap Analysis:")
print("="*40)
print(f"{'lambda':>10} | {'gap':>15}")
print("-"*40)
for lam, gap in zip(lambdas, gaps):
    print(f"{lam:10.4f} | {gap:15.8f}")
```

**해석**:
1. Lambda 증가 → LOO stability 감소 (O(1/λ) scaling)
2. Lambda 증가 → generalization gap 감소
3. 이론 O(1/(λn))과 실험이 scaling에서 일치 (상수는 다름)

---

## 🔗 ML 알고리즘 연결

### L2 정규화의 일반성
Ridge는 squared loss에만 국한되지 않는다. **모든 strongly convex 손실**에 대해 비슷한 안정성 분석이 가능하다.

### Elastic Net (L1 + L2)
Elastic Net $\min \text{loss} + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2$도 strong convexity (if $\lambda_2 > 0$)를 갖는다.

### Kernel Ridge Regression
Kernel method에 L2 정규화를 추가하면 RKHS에서 strong convexity가 성립하고, 비슷한 stability bound를 얻는다. (Ch5-05 kernel Rademacher와 비교)

---

## ⚖️ 가정과 한계

### 한계 1: Squared loss에 특화
이 정리는 squared loss (MSE)에 최적화되어 있다. 0-1 loss나 hinge loss는 다르게 다뤄야 한다.

### 한계 2: 특성의 경계성
$\|x\| \leq R$ 가정은 정규화되지 않은 고차원 데이터에서 문제가 될 수 있다.

### 한계 3: 선형 모델에만 적용
이것은 linear ridge에 대한 분석이다. Non-linear (e.g., kernel ridge)는 다른 기법이 필요.

### 한계 4: Worst-case bound
실험적 stability는 이론 bound보다 훨씬 작다. Worst-case over all (x, y) pairs 때문.

---

## 📌 핵심 정리

1. **정리 6.3**: Ridge regression은 $\beta = O(1/(\lambda n))$-uniformly stable
   - Strong convexity가 핵심

2. **추론**: Generalization gap = $O(1/(\sqrt{\lambda n})) + O(\sqrt{\log(1/\delta)/n})$
   - Lambda 증가 → gap 감소

3. **직관**: 정규화는 손실함수를 "가파르게" 만들어 안정성을 개선한다

4. **일반화**: 모든 strongly convex 손실이 비슷한 stability bound를 갖는다 (Bousquet & Elisseeff Thm 22)

---

## 🤔 생각해볼 문제

### 문제 6.3.1 (기초)
**문제**: Ridge regression의 objective에서 정규화항 $\lambda \|w\|^2$가 왜 strong convexity를 만드는가? Hessian의 고유값이 $2\lambda$ 이상이라는 것을 증명하시오.

<details>
<summary><b>해설</b></summary>

Ridge objective:
$$F(w) = \frac{1}{n}\sum (w^\top x_i - y_i)^2 + \lambda \|w\|^2.$$

Hessian:
$$\nabla^2 F(w) = \frac{2}{n}X^\top X + 2\lambda I$$

여기서 $X = [x_1^\top, \ldots, x_n^\top]^\top \in \mathbb{R}^{n \times d}$.

$X^\top X$의 고유값은 모두 $\geq 0$ (양반정). 따라서:
$$\nabla^2 F(w) = \frac{2}{n}X^\top X + 2\lambda I \succeq 0 + 2\lambda I = 2\lambda I.$$

즉, 최소 고유값이 $2\lambda$이므로 $2\lambda$-strongly convex.

</details>

### 문제 6.3.2 (심화)
**문제**: 정리 6.3의 증명에서 "두 손실함수의 차이가 $2M/n$"이라고 했다. Ridge에서 정확히 이것을 계산해보시오.

<details>
<summary><b>해설</b></summary>

$$F_S(w) - F_{S^{(i)}}(w) = \frac{1}{n}[\ell(w^\top x_i - y_i)] - \frac{1}{n}[\ell(w^\top x_i' - y_i')]$$

Squared loss:
$$|\ell(w^\top x_i - y_i) - \ell(w^\top x_i' - y_i')| = |(w^\top x_i - y_i)^2 - (w^\top x_i' - y_i')^2|$$

Expand:
$$= |w^\top(x_i - x_i') + (y_i' - y_i)||w^\top(x_i + x_i') - (y_i + y_i')|$$

Bound: $\|x\| \leq R$, $|y| \leq Y$ 가정 하에:
$$\leq (2R + 2Y) \cdot (2R + 2Y) = O(R + Y)^2$$

따라서 $\frac{1}{n}$ 항이 $\frac{O(R+Y)^2}{n}$. 이것이 "2M/n" (여기서 $M \sim (R+Y)^2$).

</details>

### 문제 6.3.3 (ML 연결)
**문제**: "Unregularized least squares는 불안정하다"고 주장하시오. 왜 $\lambda = 0$일 때 strong convexity가 사라지고, 이것이 stability를 깨뜨리는가?

<details>
<summary><b>해설</b></summary>

Unregularized:
$$F(w) = \frac{1}{n}\sum (w^\top x_i - y_i)^2.$$

Hessian:
$$\nabla^2 F(w) = \frac{2}{n}X^\top X.$$

이것의 최소 고유값은 **0일 수 있다** (if $X$가 rank-deficient). 즉, strong convexity가 없다.

Strong convexity 없이는:
$$\|w^*_S - w^*_{S^{(i)}}\| = \text{unbounded}$$

가능하다. 특히, $X$의 null space에 방향이 있으면, 샘플 변화가 해를 크게 바꿀 수 있다.

결과: **Unregularized LS는 uniform stable하지 않다** (혹은 $\beta = \infty$).

이것이 왜 실전에서 항상 정규화를 추가하는 이유다.

</details>

---

<div align="center">

◀ [이전: Stability가 Generalization을 함의](./02-stability-implies-generalization.md) | [📚 README](../README.md) | [다음: SGD의 Stability ▶](./04-sgd-stability.md)

</div>
