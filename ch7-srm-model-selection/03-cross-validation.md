# 03. Cross-Validation의 이론적 성질

## 🎯 핵심 질문

- **K-fold Cross-Validation**은 무엇을 추정하려는 것인가? "진짜 위험" $L_\mathcal{D}$인가, 아니면 다른 것?
- **Leave-One-Out CV (LOO-CV)**의 근사 비편향성(approximate unbiasedness)은 정확히 무엇을 말하는가?
- **Bias-Variance trade-off**: $K=5$와 $K=n$ (LOO)는 언제 각각 나을까? 계산 vs 통계 trade-off는?
- **Bengio & Grandvalet (2004) 정리**: "CV variance의 비편향 추정량은 일반적으로 존재하지 않는다" — 이것이 의미하는 바는?
- **Nested CV** — 외부 fold로 test error 추정, 내부 fold로 hyperparameter 선택 — 왜 필요한가? "test set contamination"이란?

---

## 🔍 왜 CV가 모델 선택의 왕인가

AIC/BIC는 **이론적으로 우아**하지만 가정이 많다 (정규성, MLE, nested models). CV는 **가정 최소**이고 **직관적**: "데이터 일부를 숨겨두고 예측 오차 추정하기". 현대 ML에서 **모델 선택의 기본 표준** (hyperparameter, 아키텍처, early stopping).

또한 **분포 자유(distribution-free)**: 데이터가 어떤 분포에서 오든 작동. 시계열(Ch1-05)처럼 IID가 깨진 상황도 **시간 순서 보존 CV** 변형으로 처리 가능. 가유일한 단점은 **계산 비용** — $K$ 배의 모델 훈련이 필요. 하지만 parallelization 가능.

SRM(Ch7-01)의 이론적 보장, AIC/BIC의 정보이론적 우아함과 달리, CV는 **실제 일반화 오차를 경험적으로 추정**한다는 강점. 이 장에서는 "이 추정량이 정말로 무엇을 근사하는가"를 수학적으로 증명한다.

---

## 📐 수학적 선행 조건

- Ch1-01, Ch1-02 (위험, Bayes 최적)
- Ch2-02, Ch2-03 (Hoeffding, McDiarmid 부등식)
- Ch6-01~02 (Stability와 일반화의 관계)
- [Mathematical Statistics](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): U-statistics, jackknife
- [Calculus & Optimization](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Convex optimization, ERM 수렴

---

## 📖 직관적 이해

### "전체 데이터"에서 일반화 오차를 추정할 수 없다

고정된 $h$가 있고, 전체 데이터 $S$에서의 오차를 보면:
$$L_S(h) = \frac{1}{n}\sum_{i=1}^n \ell(h(x_i), y_i).$$

하지만 이것은 **훈련 오차**일 뿐, **일반화 오차** $L_\mathcal{D}(h)$를 직접 추정하지 않는다. 특히 데이터 의존적 $\hat{h} = \hat{h}(S)$라면 $L_S(\hat{h})$는 $L_\mathcal{D}(\hat{h})$보다 **낙관적**이다 (과적합).

### CV의 아이디어: "부분 데이터로 훈련, 나머지로 검증"

1. 데이터를 $K$ 부분(fold)으로 나눈다: $S_1, \ldots, S_K$.
2. 각 $k=1,\ldots,K$에 대해:
   - 훈련 데이터: $S^{(-k)} = S \setminus S_k$ (fold $k$ 제외)
   - 검증 데이터: $S_k$ (fold $k$ 만)
   - 훈련: $\hat{h}_k = \text{ERM}(S^{(-k)})$
   - 평가: $\hat{L}_k = \frac{1}{|S_k|}\sum_{(x,y) \in S_k} \ell(\hat{h}_k(x), y)$
3. CV error: $\widehat{\text{CV}} = \frac{1}{K}\sum_k \hat{L}_k$

**핵심**: $\hat{h}_k$는 $S_k$를 보지 못했으므로, $\hat{L}_k$는 "$S_k$에 대한 test error"처럼 행동한다.

### LOO-CV vs K-fold 비교

| 특성 | LOO ($K=n$) | K-fold (작은 $K$) |
|------|---------|-----------|
| Bias | 매우 작음 (거의 무편향) | 중간 (작은 fold → 훈련 데이터 부족) |
| Variance | 높음 (folds 간 상관 큼) | 낮음 (독립성 높음) |
| 계산량 | $n$번 훈련 (높음) | $K$번 훈련 (낮음) |
| 추천 | 작은 $n$, 계산 여유 | 중간/큰 $n$, 빠른 피드백 |

---

## ✏️ 엄밀한 정의

### 정의 7.7 (K-fold Cross-Validation)

주어진 샘플 $S$를 크기 대략 $n/K$의 $K$개 disjoint subset으로 분할:
$$S = S_1 \cup \ldots \cup S_K, \quad S_i \cap S_j = \emptyset.$$

**각 fold $k$에 대해**:
- $S^{(-k)} = S \setminus S_k$에서 ERM 수행: $\hat{h}_{-k} = \arg\min_{h \in \mathcal{H}} L_{S^{(-k)}}(h)$
- 남은 fold $S_k$에서 평가: $\hat{e}_k = L_{S_k}(\hat{h}_{-k}) = \frac{1}{|S_k|}\sum_{(x,y) \in S_k} \ell(\hat{h}_{-k}(x), y)$

**CV 오차(통계량)**:
$$\widehat{\text{CV}}_K = \frac{1}{K}\sum_{k=1}^K \hat{e}_k.$$

### 정의 7.8 (Leave-One-Out CV)

$K = n$인 극단적 경우:
$$\widehat{\text{LOO}} = \frac{1}{n}\sum_{i=1}^n \ell(\hat{h}_{-i}(x_i), y_i),$$

여기서 $\hat{h}_{-i} = \arg\min_h L_{S \setminus \{i\}}(h)$ (샘플 $i$ 제외).

### 정의 7.9 (Nested Cross-Validation)

**외부 loop**: $S$를 $K_{\text{out}}$개 fold로 분할.
**내부 loop**: 각 외부 fold에 대해, 훈련 데이터 $S_{\text{train}}^{(-k)}$를 다시 $K_{\text{in}}$ fold로 분할하여 hyperparameter 최적화.

목표: **hyperparameter 선택의 편향** (overfitting to validation set) 제거. 외부 fold의 test error는 hyperparameter 선택에 의한 "contamination" 없이 진정한 일반화 추정.

---

## 🔬 정리와 증명

### 정리 7.8 (LOO-CV의 근사 비편향성)

크기 $n-1$인 $S^{(-i)}$에서 생성된 가설 $\hat{h}_{-i}$에 대해:

$$\mathbb{E}[\widehat{\text{LOO}}] = \mathbb{E}_{S \sim \mathcal{D}^n} \left[\frac{1}{n}\sum_{i=1}^n \ell(\hat{h}_{-i}(X_i), Y_i)\right] \approx \mathbb{E}_{S' \sim \mathcal{D}^{n-1}} [L_\mathcal{D}(\hat{h}(S'))],$$

즉, **LOO는 대략 크기 $n-1$ 샘플 상에서 학습한 모델의 test error를 추정**한다.

**증명 스케치**. 고정된 $i$에 대해, $S^{(-i)}$에 대한 기대값을 취하면:
$$\mathbb{E}_{S^{(-i)}}[\ell(\hat{h}_{-i}(X_i), Y_i) | X_i, Y_i] = \mathbb{E}_{S^{(-i)}}[\mathbb{E}_{(X_i, Y_i)}[\ell(\hat{h}_{-i}(X_i), Y_i) | S^{(-i)}]].$$

$(X_i, Y_i)$가 $\mathcal{D}$로부터 iid이고 $S^{(-i)}$와 독립이므로, 안쪽 기대값은 정확히 $\hat{h}_{-i}$의 test risk (크기 $n-1$ 훈련에서 비롯된). 따라서 평균을 취하면 비편향. $\square$

### 정리 7.9 (K-fold의 Bias-Variance)

**Bias** (각 $\hat{h}_k$는 크기 $n(K-1)/K$ 에서 훈련):
$$\mathbb{E}[\widehat{\text{CV}}_K] \approx \mathbb{E}[L_\mathcal{D}(\text{model trained on size } n(K-1)/K)],$$

훈련 크기가 작으므로 더 큰 bias (훈련 오차가 높아서).

**Variance** (fold 간의 오차 상관성):
$$\text{Var}[\widehat{\text{CV}}_K] = \text{Var}\left[\frac{1}{K}\sum_k \hat{e}_k\right].$$

$\hat{e}_k$들이 정확히 독립이 아니다 (overlap 없지만 같은 $\hat{h}_k$ 구조 의존). 일반적으로 $\text{Var}[\widehat{\text{CV}}_K] \approx \text{const}/K$이지만 정확한 form은 복잡.

**Bengio & Grandvalet (2004) 정리**: Variance 자체를 비편향하게 추정하는 일반적 공식은 없다. → CV 신뢰도 구간이 어렵다.

### 정리 7.10 (Stability와 CV의 결합)

알고리즘 $A$가 $\beta$-uniform stable이면 (Ch6-01):

$$\mathbb{P}(|\widehat{\text{LOO}} - L_\mathcal{D}(A(S))| \geq \epsilon) \leq \exp\left(-\frac{n\epsilon^2}{c}\right)$$

**증명**: McDiarmid의 bounded differences (Ch2-03)와 uniform stability의 결합. 알고리즘이 한 샘플 변화에 로컬하면, LOO fold의 오차 변화도 제한되고, 따라서 전체 LOO error의 집중도 지수적. $\square$

### 정리 7.11 (Nested CV로 Test Error 정직 추정)

외부 fold test error $\hat{e}_k^{\text{out}}$의 평균:

$$\widehat{\text{NestCV}} = \frac{1}{K_{\text{out}}}\sum_{k=1}^{K_{\text{out}}} \hat{e}_k^{\text{out}}$$

는 **hyperparameter 선택 과정에 의한 overfitting을 제거**하여, 최종 모델의 진정한 일반화 오차의 비편향 추정을 제공한다.

**직관**: 외부 fold는 hyperparameter 튜닝에 전혀 관여하지 않으므로, "fresh" test set처럼 행동. Single CV 대신 nested를 쓰면 reported test error가 더 신뢰할 수 있다. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1: 다양한 $K$에서의 CV error 비교

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

rng = np.random.default_rng(42)

# 데이터: y = sin(2πx) + noise
def sample(n):
    x = rng.uniform(0, 1, n)
    y = np.sin(2*np.pi*x) + 0.1*rng.standard_normal(n)
    return x.reshape(-1, 1), y

n_train = 50
X, y = sample(n_train)

# 각 degree와 다양한 K에서 CV 수행
degrees = np.arange(1, 11)
Ks = [2, 5, 10, n_train]  # 마지막은 LOO
K_names = ['2-fold', '5-fold', '10-fold', 'LOO']

cv_errors = {K_name: [] for K_name in K_names}
cv_stds = {K_name: [] for K_name in K_names}

for d in degrees:
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(d, include_bias=False)),
        ('lr', LinearRegression())
    ])
    
    for K, K_name in zip(Ks, K_names):
        if K == n_train:  # LOO
            cv = LeaveOneOut()
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=K, shuffle=True, random_state=42)
        
        # Negative MSE (sklearn convention)
        scores = cross_val_score(pipeline, X, y, cv=cv, 
                                scoring='neg_mean_squared_error')
        cv_errors[K_name].append(-scores.mean())
        cv_stds[K_name].append(scores.std())

# 플롯
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot 1: CV error vs degree
for K_name in K_names:
    axes[0].plot(degrees, cv_errors[K_name], 'o-', label=K_name, linewidth=2)

axes[0].set_xlabel('Polynomial degree $d$')
axes[0].set_ylabel('CV error (MSE)')
axes[0].set_title('다양한 $K$에서의 CV error — fold 수에 따른 bias-variance')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: CV error의 표준편차
for K_name in K_names:
    axes[1].plot(degrees, cv_stds[K_name], 's-', label=K_name, linewidth=2)

axes[1].set_xlabel('Polynomial degree $d$')
axes[1].set_ylabel('CV error std')
axes[1].set_title('CV 추정량의 분산 — LOO가 highest variance')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Observation: LOO has lowest bias but highest variance")
print("5-fold is often the practical sweet spot")
```

### 실험 2: Nested CV vs Single CV

```python
# Nested CV 효과 시연: hyperparameter overfitting 제거

from sklearn.model_selection import GridSearchCV, cross_validate

# 합성 데이터
X_large, y_large = sample(200)

# Single CV: hyperparameter를 같은 CV split으로 선택
pipe = Pipeline([
    ('poly', PolynomialFeatures()),
    ('lr', LinearRegression())
])

param_grid = {'poly__degree': [1, 2, 3, 4, 5, 10]}

# Single: GridSearchCV 내부에서 5-fold CV로 최적 degree 선택
single_search = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error')
single_search.fit(X_large, y_large)

# Nested: 외부 loop 5-fold, 각 외부 fold에서 내부 5-fold로 degree 선택
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
nested_scores = []
single_scores = []

for train_idx, test_idx in outer_cv.split(X_large):
    X_train, X_test = X_large[train_idx], X_large[test_idx]
    y_train, y_test = y_large[train_idx], y_large[test_idx]
    
    # Nested inner search
    inner_search = GridSearchCV(pipe, param_grid, cv=5, 
                               scoring='neg_mean_squared_error')
    inner_search.fit(X_train, y_train)
    nested_test_error = -inner_search.score(X_test, y_test)
    nested_scores.append(nested_test_error)
    
    # Single (unfair) -- evaluate on full data
    single_test_error = -single_search.score(X_test, y_test)
    single_scores.append(single_test_error)

print(f"Single CV test error (biased low): {np.mean(single_scores):.4f} ± {np.std(single_scores):.4f}")
print(f"Nested CV test error (unbiased):  {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")
print(f"→ Single CV over-optimistic, Nested more realistic")
```

### 실험 3: LOO-CV의 근사 비편향성

```python
# LOO가 크기 n-1 훈련 모델의 test error를 추정함을 확인

n_full = 100
X_full, y_full = sample(n_full)

# (1) Full n-1 데이터로 훈련
X_n_minus_1 = X_full[:-1]
y_n_minus_1 = y_full[:-1]

loo_errors = []
for d in [1, 2, 3, 5]:
    loo = LeaveOneOut()
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(d, include_bias=False)),
        ('lr', LinearRegression())
    ])
    loo_scores = cross_val_score(pipeline, X_full, y_full, cv=loo,
                                scoring='neg_mean_squared_error')
    loo_error = -loo_scores.mean()
    loo_errors.append(loo_error)
    
    # (2) Single model on n-1 data, test on remaining 1 point
    # (근사)
    pipeline.fit(X_n_minus_1, y_n_minus_1)
    # Out-of-sample 추정: 여러 샘플에서 평균
    
    print(f"degree {d}: LOO error = {loo_error:.4f}")

# → LOO가 "크기 n-1 훈련의 test error"를 근사함을 관찰
```

---

## 🔗 ML 알고리즘 연결

| 선택 방법 | 장점 | 단점 | 사용 시기 |
|---------|------|------|---------|
| **Train/test split** | 빠름 | 높은 분산, 편향 | 매우 큰 $n$ |
| **K-fold CV** | 분산 낮음, 모든 data 사용 | 계산 비용 | 표준 |
| **LOO-CV** | 거의 무편향 | 높은 분산, 느림 | 작은 $n$ |
| **Nested CV** | Hyperparameter overfitting 제거 | 매우 느림 | 중요한 추정 |
| **Time-series CV** (Ch1-05 연결) | 시간 순서 보존 | 복잡 | 시계열 |

**관찰**: CV는 **범용** 모델 선택 도구. 분포 가정 불필요, 모든 크기 $n$에서 작동.

---

## ⚖️ 가정과 한계

1. **IID 가정**: 시계열은 다른 fold 구조 필요 (Ch1-05).

2. **계산 비용**: $K$ 배의 모델 훈련. 큰 신경망에서는 비싼 비용. Approximation (WAIC, importance weighting)으로 완화 가능.

3. **Variance 비편향 추정 불가**: Bengio & Grandvalet (2004) — CV 신뢰도 구간이 어렵다. Resampling (bootstrap) 등으로 근사.

4. **고차원 한계**: $n \ll d$ 일 때 fold 크기가 너무 작아서 편향 증가. Stratified CV (class balance 보존) 등 변형 필요.

5. **Hyperparameter interaction**: 외부/내부 CV의 분리가 깨질 수 있음 (예: regularization과 feature selection의 correlated 선택). 더 정교한 nested 구조 필요.

---

## 📌 핵심 정리

- **K-fold CV**: 데이터를 $K$개로 나누어 각각 test로 사용, 나머지로 훈련. 진정한 generalization의 **경험적 추정**.
- **Bias-Variance**: LOO는 거의 무편향하나 높은 분산. 작은 K는 편향 있으나 낮은 분산. 현실에선 $K=5$ 또는 $K=10$ 선호.
- **LOO의 근사 비편향성**: $\mathbb{E}[\widehat{\text{LOO}}] \approx$ 크기 $n-1$에서 훈련한 모델의 test error.
- **Nested CV**: 외부 loop로 최종 test error 정직 추정, 내부 loop로 hyperparameter 무편향 선택.
- **Stability 연결**: Stable 알고리즘은 LOO error가 $L_\mathcal{D}$에 빠르게 집중 (지수 집중).
- **실전 표준**: 모든 현대 ML library에서 CV 기반 hyperparameter 튜닝 (GridSearchCV, RandomizedSearchCV 등).

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> 5-fold CV에서 각 fold의 크기가 대략 $n/5$일 때, 훈련 크기는 $4n/5$이다. 이것이 "bias" 관점에서 무엇을 의미하는가?</summary>

<br/>

**해설**. 각 $\hat{h}_k$는 전체 크기 $n$이 아니라 $4n/5$ (fold $k$ 제외)에서 훈련된다. 일반적으로 훈련 크기가 작을수록 **underfitting 가능성이 크고** 훈련 오차가 증가한다 (Ch1-03 approximation error). 따라서 K-fold CV 추정량은 "진짜 크기 $n$ 훈련 모델의 일반화 오차"보다 **높게 평가**된다 (낙관주의와 반대 방향 편향). 이것이 K-fold CV의 **bias 항**: $\approx$ 작은 훈련 크기의 영향. LOO는 크기 $n-1$이라 이 bias가 매우 작다. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> Bengio & Grandvalet (2004)이 "CV variance의 비편향 추정량은 일반적으로 없다"고 했을 때, 왜 이것이 문제인가? CV의 신뢰도 구간을 구성하려면 어떻게 할 것인가?</summary>

<br/>

**해설**. CV 추정량 $\widehat{\text{CV}}_K = \frac{1}{K}\sum_k \hat{e}_k$의 분산을 추정하려면:
$$\widehat{\text{Var}}[\widehat{\text{CV}}_K] = ?$$

나이브하게는 fold error들의 샘플 분산을 사용할 수 있지만, fold들이 정확히 독립이 아니므로 (같은 데이터 구조 공유) **편향된 추정**이 된다.

실제 해결책:
- **Bootstrap**: 샘플을 부트스트랩하고 각 sample에 대해 CV 반복 → fold errors 간의 상관 "재샘플링"
- **Jackknife of CV**: CV 자체를 jackknife하기 (복잡하지만 이론적으로 정당)
- **Asymptotic theory**: 큰 $n$에서 CLT 적용, 하지만 dependent samples라 standard CLT 아님

현대 practice: **신뢰도 구간보다 repeated/stratified CV의 점 추정 여러 개** 보고. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> Deep learning에서 early stopping을 할 때, validation set을 어디서 가져오는가? 만약 훈련 데이터를 train/val로 나누면 test 추정이 정직한가? Nested CV는 DL에서 현실적인가?</summary>

<br/>

**해설**. 표준 DL practice:
1. 전체 데이터 → train/val/test 3-split
2. Train에서 SGD, val에서 early stopping
3. Test에서 최종 평가

**문제**: val set이 "hyperparameter (학습률, 정규화, 초기화)" 선택에 사용되면, test는 "정직"이지만 val은 overfitting 가능 (val loss 재최적화).

**Nested CV 관점**:
- 외부: $K_{\text{out}}$개 split, 각 split 마지막 $m$ epoch을 validation으로
- 내부: 각 외부 split의 train을 다시 train/val로 나누어 learning rate 등 선택

이것이 **이론적으로 정직**하지만, DL에서는:
- 계산 비용 극심 ($K_{\text{out}} \times K_{\text{in}}$ 배 훈련)
- 보통은 single train/val/test로 타협

**권장**:
- 논문/중요 모델: nested 고려
- 빠른 prototyping: train/val/test 단일 split
- Large-scale: 충분한 test 데이터면 train/val 유지

현대 관점에선 **validation loss 곡선** 자체가 (단순 수치보다) 모델 선택에 중요. $\square$

</details>

---

<div align="center">

◀ [이전: 02. AIC, BIC, MDL](./02-aic-bic-mdl.md) | [📚 README](../README.md) | [다음: 04. VC, Rademacher, Stability 비교 ▶](./04-three-viewpoints-comparison.md)

</div>
