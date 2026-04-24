# 01. Uniform Stability의 정의

## 🎯 핵심 질문

- **Stability**는 "가설공간 복잡도"(Ch3~5)와 달리 왜 "**알고리즘의 복잡도**"를 측정하는가?
- 하나의 샘플을 바꿔도 알고리즘의 출력이 안정적이라는 것이 왜 **일반화**를 의미하는가?
- **$\beta$-uniform stability** $\sup_S \sup_{z_i'} \sup_z |\ell(A(S), z) - \ell(A(S^{(i)}), z)| \leq \beta$의 정의는 무엇이고, 왜 이런 형태인가?
- Hypothesis·Pointwise·Error stability 등 여러 variant들은 어떻게 다르며, 언제 어느 것을 쓰는가?
- Stability는 $\mathcal{H}$-독립적이라는 것이 무엇을 의미하고, VC·Rademacher와 근본적으로 어떻게 다른가?

---

## 🔍 왜 이 이론이 현대 ML에서 중요한가

Ch3~5에서 우리는 **"가설공간이 작을수록 일반화가 쉽다"**는 원리(VC·Rademacher)를 배웠다. 하지만 현대 ML의 역설이 있다: **신경망은 VC 차원이 엄청 크지만 일반화가 잘 된다**(Ch4-07, Zhang et al. 2017). 왜인가?

**Stability** 관점은 이 역설을 풀기 위한 **새로운 렌즈**다. 가설공간의 크기를 묻는 대신, **"알고리즘 자체가 얼마나 robust한가"**를 묻는다. 즉:
- VC는 묻는다: "$\mathcal{H}$이 복잡한가?"
- Stability는 묻는다: "$A$가 작은 데이터 변화에 sensitive한가?"

이 관점의 위력은 **규제(regularization)의 수학적 정당화**에서 드러난다:
- **Ridge regression**: $\lambda$-정규화 → $\beta = O(1/(\lambda n))$ (Ch6-03)
- **SGD**: 유한 단계 훈련 → $\beta = O(\eta T / n)$ (Ch6-04) = "**적게 훈련 = implicit regularization**"

Deep learning의 관점에서, "왜 SGD가 generalize하는가"의 부분적 답이 바로 이 stability 프레임워크다.

---

## 📐 수학적 선행 조건

- Ch1-01~03: 위험 $L_\mathcal{D}$·$L_S$, ERM, 3분해
- Ch2-02~03: Hoeffding·McDiarmid 부등식 (집중부등식 기초)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Expectation, concentration
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): 강볼록성(strong convexity) — Ch6-03에서 필수

---

## 📖 직관적 이해

### "샘플을 바꿔도 방향은 같다"

샘플 집합 $S = \{z_1, \ldots, z_n\}$에서 하나의 샘플 $z_i$를 새로운 샘플 $z_i' \sim \mathcal{D}$로 **교체**한 new 샘플 $S^{(i)} = S \setminus \{z_i\} \cup \{z_i'\}$를 생각하자. 알고리즘 $A$는 이 두 샘플을 입력받아 **두 개의 모델** $A(S)$와 $A(S^{(i)})$를 출력한다.

**안정적인 알고리즘**: 이 두 모델이 어떤 test point $z$에서 비슷한 손실을 낸다. 즉 $|\ell(A(S), z) - \ell(A(S^{(i)}), z)| \leq \beta$ 정도.

**불안정한 알고리즘**: 하나의 샘플 변화가 커다란 손실 변화를 초래한다.

왜 안정성이 일반화를 함의하는가? 만약 하나의 샘플 변화가 평균적으로 손실을 $\beta$ 이상 바꾸지 않는다면, 훈련 데이터의 분포 변화가 알고리즘의 행동을 크게 바꾸지 않는다는 뜻이다. 따라서:
- 훈련 데이터에서의 오차 $L_S(A(S))$
- 새로운 테스트 데이터에서의 오차 $L_\mathcal{D}(A(S))$

이 둘의 차이가 크지 않을 **확률**이 높다. 이것이 **Stability ⇒ Generalization**(정리 6.2)의 핵심이다.

### 왜 $\mathcal{H}$-독립적인가?

**VC·Rademacher**: 경계가 $\mathcal{H}$의 크기·복잡도에 의존한다.
$$\text{Gap} \lesssim \sqrt{\text{complexity}(\mathcal{H}) / n}$$

**Stability**: 경계가 알고리즘 자체에만 의존한다.
$$\text{Gap} \lesssim \beta$$

따라서 같은 $\mathcal{H}$라도, **알고리즘이 다르면** stability가 다를 수 있다. 예:
- 최소 제곱법 (unregularized): 불안정
- Ridge regression ($\lambda > 0$): 안정 (정리 6.3)
- SGD ($T$ 단계): 안정 (정리 6.4)

이것이 **regularization**이 왜 일반화를 돕는지의 답이다.

---

## ✏️ 엄밀한 정의

### 정의 6.1 ($\beta$-Uniform Stability)

**학습 알고리즘** $A: (\mathcal{X} \times \mathcal{Y})^n \to \mathcal{H}$가 **$\beta$-uniformly stable**이라는 것은, 모든 크기 $n$ 샘플 $S$, 모든 $i \in \{1, \ldots, n\}$, 모든 대체 샘플 $z_i' \in \mathcal{X} \times \mathcal{Y}$, 그리고 모든 테스트 포인트 $z \in \mathcal{X} \times \mathcal{Y}$에 대해:

$$\sup_S \sup_{i} \sup_{z_i'} \sup_z \left| \ell(A(S), z) - \ell(A(S^{(i)}), z) \right| \leq \beta$$

를 만족하는 것이다. 여기서 $S^{(i)} := S \setminus \{z_i\} \cup \{z_i'\}$.

**직관**: 최악의 경우(worst case) 단 하나의 샘플 변화가 임의의 테스트 포인트에서의 손실을 최대 $\beta$만큼 바꿀 수 있다는 보장.

### 정의 6.2 (Hypothesis Stability)

더 약한 형태로, **기대값** 의미에서의 안정성:

$$\mathbb{E}_{z_i'} \left| \ell(A(S), z) - \ell(A(S^{(i)}), z) \right| \leq \beta \quad \forall S, i, z$$

혹은 

$$\mathbb{E}_{S, z_i'} \left| \ell(A(S), z) - \ell(A(S^{(i)}), z) \right| \leq \beta \quad \forall i, z$$

### 정의 6.3 (Pointwise Hypothesis Stability)

가장 약한 형태로, 기대값 의미에서 평균적으로:

$$\sup_S \sup_i \sup_z \mathbb{E}_{z_i'} \left| \ell(A(S), z) - \ell(A(S^{(i)}), z) \right| \leq \beta$$

---

## 🔬 정리와 증명

### 정리 6.1 (Uniform vs Hypothesis Stability의 관계)

$\beta$-uniform stability ⇒ ($\beta$-hypothesis stability) ⇒ (pointwise hypothesis stability).

**증명**. 정의에 의해 직접 따라온다. Uniform은 "모든 경우" sup을 취하므로 hypothesis(일부 sup)는 자동으로 만족된다. □

### 정리 6.1' (Empirical 버전: Leave-One-Out)

실제로 stability를 측정할 때는 테스트 포인트를 **훈련 포인트로 대체**하는 경우도 중요하다:

$$\sup_S \sup_i \left| \ell(A(S), z_i) - \ell(A(S^{(i)}), z_i) \right| \leq \beta$$

이를 **LOO (Leave-One-Out) stability**라 부른다. 정리 6.2에서 이것만으로도 일반화 경계에 충분하다.

---

## 💻 NumPy 구현 검증

### 실험 1: 간단한 알고리즘의 Uniform Stability 경험적 측정

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. Averaging algorithm: A(S) = mean of S
# 이것은 O(1/n) stable해야 함
# ─────────────────────────────────────────────

def averaging_algo(S):
    """S = [(x_1, y_1), ..., (x_n, y_n)]: 
       A(S) = mean(y_i) (무한정규화 ERM)"""
    return np.mean(S[:, 1])  # y 값의 평균만

n = 100
n_trials = 5000
max_diffs = []

# S 샘플링, 한 점 교체, 차이 측정
for trial in range(n_trials):
    # S 생성: (x_i, y_i), y_i ∈ [0, 1]
    S = np.column_stack([
        rng.uniform(0, 1, n),  # x values
        rng.uniform(0, 1, n)   # y values
    ])
    
    # 알고리즘 출력
    pred_S = averaging_algo(S)
    
    # i-th 샘플 교체: z_i' = (x', y') random
    max_diff_trial = 0
    for i in range(n):
        # S^(i) 생성
        z_new = np.array([rng.uniform(0, 1), rng.uniform(0, 1)])
        S_replaced = S.copy()
        S_replaced[i] = z_new
        
        pred_S_replaced = averaging_algo(S_replaced)
        
        # 차이: 모든 가능한 z에서의 손실 차이
        # z = (x, y)에서 loss = (pred - y)^2
        # 하지만 모든 z를 다 시뮬레이션할 수 없으니
        # 여러 테스트 포인트에서 샘플
        test_y = rng.uniform(0, 1, 100)
        diffs = np.abs((pred_S - test_y)**2 - (pred_S_replaced - test_y)**2)
        max_diff_trial = max(max_diff_trial, np.max(diffs))
    
    max_diffs.append(max_diff_trial)

max_diffs = np.array(max_diffs)
print(f"Averaging Algorithm (n={n}):")
print(f"  Max difference (95%-ile): {np.percentile(max_diffs, 95):.6f}")
print(f"  Expected 1/n: {1/n:.6f}")
print(f"  Ratio: {np.percentile(max_diffs, 95) / (1/n):.2f}")

# ─────────────────────────────────────────────
# 2. 선형 회귀 (unregularized): 불안정
# ─────────────────────────────────────────────

def linear_regression_erm(X, y):
    """Unregularized: argmin ||Xw - y||^2"""
    try:
        w = np.linalg.lstsq(X, y, rcond=None)[0]
        return w
    except:
        return np.zeros(X.shape[1])

n = 50
d = 10
n_trials = 100
max_diffs_lr = []

for trial in range(n_trials):
    # 특성과 라벨 생성
    X = rng.standard_normal((n, d))
    y = rng.standard_normal(n)
    
    # ERM 해
    w = linear_regression_erm(X, y)
    
    # 한 샘플 교체 시 차이
    max_diff = 0
    for i in range(n):
        X_new = X.copy()
        X_new[i] = rng.standard_normal(d)
        y_new = y.copy()
        y_new[i] = rng.standard_normal()
        
        w_new = linear_regression_erm(X_new, y_new)
        
        # 테스트 포인트에서의 손실 차이 (MSE)
        test_X = rng.standard_normal((50, d))
        pred_loss = np.mean((X @ w - y) ** 2)
        pred_loss_new = np.mean((X_new @ w_new - y_new) ** 2)
        
        max_diff = max(max_diff, np.abs(pred_loss - pred_loss_new))
    
    max_diffs_lr.append(max_diff)

max_diffs_lr = np.array(max_diffs_lr)
print(f"\nLinear Regression Unregularized (n={n}):")
print(f"  Max difference (95%-ile): {np.percentile(max_diffs_lr, 95):.6f}")
print(f"  Expected O(1): unstable")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(max_diffs, bins=30, alpha=0.7, edgecolor='black')
axes[0].axvline(np.percentile(max_diffs, 95), color='r', linestyle='--', label='95%ile')
axes[0].axvline(1/n, color='g', linestyle='--', label='1/n')
axes[0].set_xlabel('Max Stability Difference')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Averaging (Stable)')
axes[0].legend()

axes[1].hist(max_diffs_lr, bins=30, alpha=0.7, edgecolor='black')
axes[1].axvline(np.percentile(max_diffs_lr, 95), color='r', linestyle='--', label='95%ile')
axes[1].set_xlabel('Max Stability Difference')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Linear Regression Unregularized (Unstable)')
axes[1].legend()

plt.tight_layout()
plt.savefig('/tmp/stability_comparison.png', dpi=100)
print("\n✓ Plot saved to /tmp/stability_comparison.png")
```

**실험 결과 해석**: Averaging은 $\beta \approx 1/n$ 정도로 안정적이고, unregularized linear regression은 $\beta = O(1)$로 불안정하다. 정규화를 추가하면 stability가 개선된다(Ch6-03).

---

## 🔗 ML 알고리즘 연결

### Ridge Regression
정규화항 $\lambda \|w\|^2$을 추가하면 strong convexity 때문에 $\beta = O(1/(\lambda n))$이 된다. (Ch6-03 정리 6.3)

### SGD (Stochastic Gradient Descent)
유한 단계 $T$와 학습률 $\eta$를 갖는 SGD는 $\beta = O(\eta T / n)$로 안정적이다. (Ch6-04 정리 6.4) 이것이 "조기 멈춤(early stopping)"이 왜 일반화를 돕는가의 근거다.

### SVM (Support Vector Machine)
Margin 최대화는 strong convexity와 관련이 있어서 안정성이 자동으로 달성된다.

---

## ⚖️ 가정과 한계

### 한계 1: Uniform Stability는 충분조건이지 필요조건이 아니다
어떤 알고리즘이 일반화하더라도 uniform stability를 만족하지 않을 수 있다. 즉:
- Stability ⇒ Generalization (정리 6.2)
- Generalization ⇏ Stability (일반적으로)

### 한계 2: 손실함수의 경계성 필요
일반화 경계(정리 6.2)를 도출할 때 $\ell \in [0, M]$ 경계성이 필수다. unbounded loss는 다른 기법이 필요.

### 한계 3: 샘플 교체의 의미
"하나의 샘플을 교체"한다는 것이 통상적으로 의미 있는 stability 분석이려면, 샘플이 어느 정도 "대표성"을 가져야 한다. 극도로 outlier인 샘플은 stability를 깨뜨릴 수 있다.

### 한계 4: Non-convex 손실의 어려움
Deep learning은 non-convex 손실을 갖는다. Uniform stability를 만족하려면 훨씬 강한 가정(e.g., 작은 step size, bounded gradient)이 필요하다.

---

## 📌 핵심 정리

1. **Uniform Stability**는 알고리즘 $A$의 **robustness** 측정: 하나의 샘플 변화가 손실을 최대 $\beta$만큼 바꾼다.

2. **$\mathcal{H}$-독립성**: VC·Rademacher와 달리, stability 경계는 가설공간 복잡도에 의존하지 않고 알고리즘 자체에만 의존한다.

3. **Regularization의 정당화**: Strong convex 정규화 (ridge)나 유한 단계 훈련(SGD)은 stability를 개선해 일반화를 돕는다.

4. **Multiple variants**: Uniform·Hypothesis·Pointwise stability가 있으며, 정리 6.2는 가장 약한 형태 (LOO)로도 충분함을 보인다.

---

## 🤔 생각해볼 문제

### 문제 6.1.1 (기초)
**문제**: 상수 알고리즘 $A(S) = c$ (모든 $S$에 대해 동일한 가설 $h_c$를 출력)는 어떤 $\beta$로 uniformly stable한가? 왜?

<details>
<summary><b>해설</b></summary>

상수 알고리즘은 $A(S) = A(S^{(i)})$이므로 모든 $S, i, z$에 대해
$$|\ell(A(S), z) - \ell(A(S^{(i)}), z)| = 0.$$
따라서 $\beta = 0$ — 완벽하게 안정적이다. 물론 이 알고리즘은 학습하지 않는다 (approximation error가 크다).

**교훈**: Stability만으로는 좋은 알고리즘을 보장하지 않는다. 3분해의 approximation 항도 작아야 한다.

</details>

### 문제 6.1.2 (심화)
**문제**: 1-Nearest Neighbor 알고리즘은 uniform stable한가? (훈련 샘플과 가장 가까운 이웃의 라벨을 출력)

<details>
<summary><b>해설</b></summary>

일반적으로 **아니다**. 이유:
- $i$-th 샘플 $z_i$를 제거하면 테스트 포인트 $z$의 "가장 가까운 이웃"이 바뀔 수 있다.
- 특히 $z$가 $z_i$에 아주 가까우면, $z_i$를 제거하는 것이 예측을 크게 바꾼다.
- 따라서 uniform stability를 만족하는 $\beta$가 존재하지 않는다 (혹은 $\beta = \infty$).

이것이 kNN이 "표본 복잡도" 면에서 분석하기 어려운 이유 중 하나이고, 대신 다른 stability variant나 다른 분석 기법(e.g., covering number)을 써야 한다.

</details>

### 문제 6.1.3 (ML 연결)
**문제**: "Early stopping이 regularization처럼 작동한다"는 주장(Ch6-04)을 stability 관점에서 직관적으로 설명하시오.

<details>
<summary><b>해설</b></summary>

SGD의 stability는 $\beta = O(\eta T / n)$로 **단계 수 $T$에 비례**한다. 즉:
- 더 많이 훈련 ($T$ 증가) → 불안정 ($\beta$ 증가)
- 일반화 gap = $\approx \beta = O(\eta T / n)$ 증가
- **조기 멈춤** ($T$를 조기에 멈춤) → $\beta$ 작음 → gap 작음

이것이 **implicit regularization**: 명시적 $\lambda$ 항을 추가하지 않아도, 훈련을 일찍 멈추는 것 자체가 알고리즘의 stability를 제어해 정규화 효과를 낸다.

Ridge regression (정리 6.3)과 비교: $\lambda$를 크게 하면 $\beta = O(1/(\lambda n))$이 작아진다. 둘 다 안정성 ↑ = 일반화 ↑을 노린다.

</details>

---

<div align="center">

◀ [이전: Ch5-06. 신경망의 Rademacher 복잡도](../ch5-rademacher/06-neural-net-rademacher.md) | [📚 README](../README.md) | [다음: Stability가 Generalization을 함의 ▶](./02-stability-implies-generalization.md)

</div>
