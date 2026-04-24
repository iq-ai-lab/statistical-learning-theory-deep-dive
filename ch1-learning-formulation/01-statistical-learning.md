# 01. 학습의 통계적 정의

## 🎯 핵심 질문

- "기계학습"을 수학적으로 정의한다는 것은 무엇인가? 왜 우리는 데이터가 **확률분포 $\mathcal{D}$로부터 iid**로 뽑혔다고 가정하는가?
- **진짜 위험(true risk)** $L_\mathcal{D}(h) = \mathbb{E}[\ell(h(X), Y)]$와 **경험 위험(empirical risk)** $L_S(h) = \frac{1}{n}\sum \ell(h(x_i), y_i)$은 무엇이 왜 다른가?
- 손실함수 $\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_+$는 왜 이 형태로 정의되는가? 0-1 loss, squared loss, cross-entropy는 각각 무엇을 최적화하는가?
- 측도론적 관점에서 "$L_\mathcal{D}(h)$가 well-defined하다"는 어떤 가정 하에 성립하는가? (가측성·적분가능성)
- 학습기(learner) $A: (\mathcal{X} \times \mathcal{Y})^n \to \mathcal{H}$를 **확률변수**로 보는 관점이 왜 중요한가?

---

## 🔍 왜 이 정식화가 현대 ML에서 중요한가

"신경망을 훈련시킨다", "Gradient descent가 수렴한다", "교차검증으로 모델을 고른다" — 이 모든 문장은 **서로 다른 양(quantity)을 최적화한다고 착각**하기 쉽다. 통계적 학습 이론은 이 혼란을 **두 개의 risk**로 정리한다: 우리가 **계산할 수 있는 것**($L_S$)과 우리가 **정말 알고 싶은 것**($L_\mathcal{D}$). 모든 SLT 정리는 이 둘의 차이, 즉 **$L_\mathcal{D}(\hat{h}) - L_S(\hat{h})$를 확률적으로 bound하는** 것을 목표로 한다.

이 정식화를 건너뛰면 "테스트 오차가 왜 훈련 오차와 다른가"라는 본질적 질문에 답할 언어 자체가 없어진다. $\mathcal{D}$·$S$·$\ell$·$\mathcal{H}$·$h^*$·$\hat{h}$의 역할을 분리하는 것은 단순히 기호가 많아지는 것이 아니라, **과적합이 왜 일어나는지**, **왜 더 많은 데이터가 도움이 되는지**, **왜 가설공간을 제한하는 것이 학습의 전제인지**를 분석 가능한 형태로 만드는 **유일한 방법**이다. 이 문서의 어휘 없이는 Ch2(집중부등식)부터 Ch7(SRM)까지 그 무엇도 진술조차 할 수 없다.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 확률공간 $(\Omega, \mathcal{F}, \mathbb{P})$, 확률변수, 기대값, iid 정의
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): 통계량, 확률표본, 추정량의 확률적 성질
- [Real Analysis](https://github.com/iq-ai-lab/real-analysis-deep-dive) *(권장)*: Lebesgue 적분, 가측함수, Fubini 정리
- 기초: 지시함수 $\mathbb{1}[\cdot]$, 조건부 확률, 기대값의 선형성

---

## 📖 직관적 이해

### "학습한다"는 것은 "분포를 추측한다"는 것

고양이/강아지 분류기를 훈련한다고 하자. 우리가 정말 원하는 것은 "세상 모든 고양이·강아지 사진을 잘 분류하는 것"이다. 하지만 "세상 모든"은 무한집합이고, 우리 손에 있는 것은 **유한한 샘플** $S = \{(x_1, y_1), \ldots, (x_n, y_n)\}$뿐이다.

SLT는 이 간극을 **확률분포 $\mathcal{D}$**로 메꾼다. "세상 모든 고양이·강아지 사진"을 **$\mathcal{X} \times \mathcal{Y}$ 위의 어떤 확률분포 $\mathcal{D}$**라고 모델링하고, 훈련 샘플이 이 $\mathcal{D}$에서 **iid로 뽑힌 것**이라고 가정한다. 그러면 "모든 사진에 대한 평균 오차"는 **기대값** $\mathbb{E}_{(X,Y) \sim \mathcal{D}}[\ell(h(X), Y)]$으로 정의되고, 이것이 바로 **진짜 위험** $L_\mathcal{D}(h)$다.

### 계산할 수 있는 것 vs 알고 싶은 것

| 양 | 기호 | 정의 | 우리는 아는가? |
|----|------|------|-------------|
| 진짜 위험 | $L_\mathcal{D}(h)$ | $\mathbb{E}_{(X,Y) \sim \mathcal{D}}[\ell(h(X), Y)]$ | ❌ $\mathcal{D}$를 모름 |
| 경험 위험 | $L_S(h)$ | $\frac{1}{n} \sum_{i=1}^n \ell(h(x_i), y_i)$ | ✅ 직접 계산 가능 |
| 일반화 gap | $L_\mathcal{D}(h) - L_S(h)$ | 두 값의 차이 | ❌ 경계로 bound |

$L_S(h)$는 **확률변수**다 — 샘플 $S$가 랜덤이기 때문이다. $\mathbb{E}_S[L_S(h)] = L_\mathcal{D}(h)$는 **큰 수의 법칙**(LLN)에 의해 $n \to \infty$일 때 수렴한다. 하지만 **유한한 $n$**에서 "얼마나 빨리?"가 집중부등식(Ch2)의 질문이다.

### 왜 손실함수는 "둘을 비교"하는가

$\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_+$은 두 번째 입력이 **진짜 라벨** $y$, 첫 번째 입력이 **예측** $h(x)$인 **비음수** 함수다. 비음수성은 기대값의 존재 보장에 쓰이고, 0에서의 값 $\ell(y, y) = 0$은 "완벽한 예측은 손실 없음"을 말한다. 대표 예:

- **0-1 loss**: $\ell(\hat{y}, y) = \mathbb{1}[\hat{y} \neq y]$ — 분류의 "맞았냐 틀렸냐"
- **Squared loss**: $\ell(\hat{y}, y) = (\hat{y} - y)^2$ — 회귀의 MSE
- **Cross-entropy**: $\ell(\hat{y}, y) = -\sum_k y_k \log \hat{y}_k$ — 확률 예측의 KL

각 loss는 **다른 최적 예측기**를 유도한다(Ch1-02).

---

## ✏️ 엄밀한 정의

### 정의 1.1 (학습 문제의 성분)

**학습 문제(learning problem)**는 다음 4-튜플 $(\mathcal{X}, \mathcal{Y}, \mathcal{D}, \ell)$로 정의된다:

- $\mathcal{X}$: **입력 공간(instance space)**, 가측공간 $(\mathcal{X}, \Sigma_\mathcal{X})$
- $\mathcal{Y}$: **라벨 공간(label space)**, 가측공간 $(\mathcal{Y}, \Sigma_\mathcal{Y})$
- $\mathcal{D}$: $\mathcal{X} \times \mathcal{Y}$ 위의 **확률측도**(생성 곱측도에 대해)
- $\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_+$: **손실함수**, 가측함수

전형적으로 $\mathcal{X} \subseteq \mathbb{R}^d$, $\mathcal{Y} \in \{\{0, 1\}, \{-1, +1\}, \{1, \ldots, K\}, \mathbb{R}\}$.

### 정의 1.2 (가설과 가설공간)

**가설(hypothesis)** $h: \mathcal{X} \to \mathcal{Y}$는 가측함수다. **가설공간(hypothesis class)** $\mathcal{H}$는 이러한 함수들의 집합이다:
$$\mathcal{H} \subseteq \mathcal{Y}^\mathcal{X}, \quad \forall h \in \mathcal{H}: h \text{는 가측}.$$

대표 예: 선형 분류기 $\mathcal{H} = \{x \mapsto \text{sign}(w^\top x + b) : w \in \mathbb{R}^d, b \in \mathbb{R}\}$, 깊이 $L$·폭 $W$ 신경망, 깊이 $d$ 결정 트리.

### 정의 1.3 (진짜 위험과 경험 위험)

가설 $h \in \mathcal{H}$에 대해:

**진짜 위험(true/population risk)**:
$$L_\mathcal{D}(h) := \mathbb{E}_{(X, Y) \sim \mathcal{D}}[\ell(h(X), Y)] = \int_{\mathcal{X} \times \mathcal{Y}} \ell(h(x), y) \, d\mathcal{D}(x, y).$$

훈련 샘플 $S = ((x_1, y_1), \ldots, (x_n, y_n)) \sim \mathcal{D}^n$에 대해 **경험 위험(empirical risk)**:
$$L_S(h) := \frac{1}{n} \sum_{i=1}^n \ell(h(x_i), y_i).$$

### 정의 1.4 (학습기)

**학습기(learner)** 또는 **학습 알고리즘**은 (유한 혹은 가변 길이) 샘플을 가설로 보내는 가측 사상이다:
$$A: \bigcup_{n \geq 1} (\mathcal{X} \times \mathcal{Y})^n \to \mathcal{H}, \quad S \mapsto \hat{h}_S = A(S).$$

$A$는 확정적(deterministic)일 수도 있고 랜덤(randomized)일 수도 있다. $\hat{h}_S$는 샘플 $S$에 의존하므로 **확률변수**다.

### 정의 1.5 (IID 가정)

훈련 샘플의 기본 가정:
$$(X_1, Y_1), \ldots, (X_n, Y_n) \overset{\text{iid}}{\sim} \mathcal{D}.$$

즉, 각 $(X_i, Y_i)$는 **$\mathcal{D}$에서 동일분포**로 뽑히고 서로 **독립**이다. 곱측도로 $S \sim \mathcal{D}^n$으로 표기한다.

---

## 🔬 정리와 증명

### 정리 1.1 (경험 위험의 비편향성)

고정된 $h \in \mathcal{H}$와 iid 샘플 $S \sim \mathcal{D}^n$에 대해,
$$\mathbb{E}_S[L_S(h)] = L_\mathcal{D}(h).$$

**증명**. 기대값의 선형성과 $(X_i, Y_i) \sim \mathcal{D}$의 동일분포성:
$$\mathbb{E}_S[L_S(h)] = \mathbb{E}_S\!\left[\frac{1}{n} \sum_{i=1}^n \ell(h(X_i), Y_i)\right] = \frac{1}{n} \sum_{i=1}^n \mathbb{E}_{(X_i, Y_i)}[\ell(h(X_i), Y_i)] = \frac{1}{n} \cdot n \cdot L_\mathcal{D}(h) = L_\mathcal{D}(h). \qquad \square$$

> **주의**: 이것은 **고정된 $h$**에서만 성립한다. **데이터 의존적** $\hat{h} = \hat{h}(S)$에 대해서는 $\mathbb{E}_S[L_S(\hat{h})]$가 $\mathbb{E}_S[L_\mathcal{D}(\hat{h})]$보다 **작을 수 있다** — 이것이 과적합의 수학적 뿌리이며, Ch3부터의 분석 전체의 동기다.

### 정리 1.2 (경험 위험의 약한 수렴 — LLN)

고정된 $h \in \mathcal{H}$에 대해 $\ell(h(X), Y)$가 **적분가능**($\mathbb{E}[|\ell(h(X), Y)|] < \infty$)하면, 강한 큰 수의 법칙(SLLN)에 의해
$$L_S(h) \xrightarrow{\text{a.s.}} L_\mathcal{D}(h) \text{ as } n \to \infty.$$

**증명**. $Z_i := \ell(h(X_i), Y_i)$라 두면 $\{Z_i\}$는 iid이고 $\mathbb{E}[|Z_i|] < \infty$. Kolmogorov SLLN에 의해
$$\frac{1}{n} \sum_{i=1}^n Z_i \xrightarrow{\text{a.s.}} \mathbb{E}[Z_1] = L_\mathcal{D}(h). \qquad \square$$

> **한계**: SLLN은 "$h$ 고정 → 수렴"만 보장한다. **$\sup_{h \in \mathcal{H}} |L_S(h) - L_\mathcal{D}(h)| \to 0$**(uniform convergence)은 훨씬 강한 주장이며, 이것이 VC 이론(Ch4)의 핵심 과제다.

### 정리 1.3 (경험 위험의 분산)

$\ell$이 $[0, M]$ 값을 갖는다면, 고정 $h$에 대해
$$\text{Var}_S[L_S(h)] = \frac{\text{Var}[\ell(h(X), Y)]}{n} \leq \frac{M^2}{4n}.$$

**증명**. 독립성에 의해 $\text{Var}\!\left[\frac{1}{n}\sum Z_i\right] = \frac{1}{n^2}\sum \text{Var}[Z_i] = \frac{\text{Var}[Z_1]}{n}$. $Z_1 \in [0, M]$이므로 $\text{Var}[Z_1] \leq (M/2)^2 = M^2/4$ (Popoviciu 부등식). $\square$

### 정리 1.4 (가측성 — well-definedness)

$h$가 가측함수이고 $\ell$이 가측함수이면, $\ell(h(\cdot), \cdot): \mathcal{X} \times \mathcal{Y} \to \mathbb{R}_+$는 $\mathcal{D}$에 관해 가측이고, 비음수성 덕분에 (가능하면 $+\infty$ 값으로) 기대값 $L_\mathcal{D}(h)$가 **항상 잘 정의된다**.

**증명**. 사상 $(x, y) \mapsto (h(x), y)$가 가측이고(component-wise), $\ell$이 가측이므로 합성 $(x, y) \mapsto \ell(h(x), y)$가 가측이다. 비음수 가측함수의 Lebesgue 적분은 $[0, \infty]$ 값으로 항상 정의된다. $\square$

### 정리 1.5 (Bayes risk 하한)

모든 (가측) 가설 $h$에 대해
$$L_\mathcal{D}(h) \geq L^* := \inf_{h': \mathcal{X} \to \mathcal{Y}} L_\mathcal{D}(h').$$

여기서 $L^*$를 **Bayes risk**라 부른다. $\mathcal{H} \subsetneq \mathcal{Y}^\mathcal{X}$일 때 $\inf_{h \in \mathcal{H}} L_\mathcal{D}(h) \geq L^*$일 수 있으며, 이 차이가 **approximation error**다.

**증명**. 정의에 의한 하한. 점별 최적 예측기의 존재는 Ch1-02에서 다룬다. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1: 경험 위험의 비편향성과 분산 감소

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# 진짜 분포 𝒟: X ~ Uniform([-1, 1]), Y = sign(X) with 10% label flip
def sample_D(n):
    X = rng.uniform(-1, 1, n)
    y_clean = np.sign(X)
    flip = rng.random(n) < 0.10
    Y = np.where(flip, -y_clean, y_clean)
    return X, Y

# 고정 가설: h(x) = sign(x)
def h(x): return np.sign(x)

# 0-1 loss
def loss(yhat, y): return (yhat != y).astype(float)

# 진짜 위험: 라벨 flip 확률 = 0.10
L_D = 0.10

# 경험 위험을 다양한 n에 대해 여러 번 계산
ns = [10, 50, 100, 500, 2000]
n_trials = 5000
means, stds = [], []
for n in ns:
    losses_n = []
    for _ in range(n_trials):
        X, Y = sample_D(n)
        L_S = loss(h(X), Y).mean()
        losses_n.append(L_S)
    means.append(np.mean(losses_n))
    stds.append(np.std(losses_n))

print(f'True risk L_D = {L_D:.4f}')
for n, m, s in zip(ns, means, stds):
    print(f'n={n:5d}: E[L_S]={m:.4f}, std={s:.4f}, 1/√n={1/np.sqrt(n):.4f}')

# 실행 예시 출력:
# True risk L_D = 0.1000
# n=   10: E[L_S]=0.1006, std=0.0953
# n=   50: E[L_S]=0.0998, std=0.0424
# n=  100: E[L_S]=0.1003, std=0.0302
# n=  500: E[L_S]=0.1001, std=0.0134
# n= 2000: E[L_S]=0.1000, std=0.0067
# 정리 1.1 확인: E[L_S] → L_D.
# 정리 1.3 확인: std ~ 1/√n.
```

### 실험 2: 경험 위험 분포의 집중(concentration) 시각화

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for n in [10, 50, 200, 1000]:
    losses = []
    for _ in range(2000):
        X, Y = sample_D(n)
        losses.append(loss(h(X), Y).mean())
    axes[0].hist(losses, bins=40, alpha=0.5, density=True, label=f'n={n}')

axes[0].axvline(L_D, color='k', linestyle='--', label='L_D=0.10')
axes[0].set_xlabel('L_S(h)'); axes[0].set_ylabel('density')
axes[0].set_title('샘플크기 n 증가에 따른 L_S(h) 분포 집중')
axes[0].legend()

# 1/√n 스케일링 확인
axes[1].loglog(ns, stds, 'o-', label='std of L_S')
axes[1].loglog(ns, [0.3/np.sqrt(n) for n in ns], '--', label='~0.3/√n')
axes[1].set_xlabel('n'); axes[1].set_ylabel('std')
axes[1].set_title('std(L_S) ∝ 1/√n — LLN의 속도')
axes[1].legend()

plt.tight_layout(); plt.show()
# → L_S의 분포가 L_D 주변으로 1/√n 속도로 집중됨을 확인.
```

### 실험 3: 손실함수에 따른 최적 예측의 차이 (Ch1-02의 선행 탐색)

```python
# 같은 (X, Y) 분포에서 어떤 상수 예측 c*가 각 loss를 최소화하는가?
# X ~ N(0, 1), Y = X + N(0, 0.5)
X = rng.standard_normal(10000)
Y = X + 0.5 * rng.standard_normal(10000)

c_grid = np.linspace(-2, 2, 200)

# 0-1 loss는 Y가 이산이어야 의미 있으므로 squared와 abs만 비교
mse  = [np.mean((c - Y) ** 2) for c in c_grid]
mae  = [np.mean(np.abs(c - Y)) for c in c_grid]

print(f'argmin MSE: c* = {c_grid[np.argmin(mse)]:.3f}, E[Y] = {Y.mean():.3f}')
print(f'argmin MAE: c* = {c_grid[np.argmin(mae)]:.3f}, median(Y) = {np.median(Y):.3f}')
# → MSE는 평균을, MAE는 중앙값을 "정답"으로 삼음.
# 손실함수가 최적 예측기를 결정한다는 중요한 관찰 (Ch1-02로 이어짐).
```

---

## 🔗 ML 알고리즘 연결

| 알고리즘 | $\mathcal{X}, \mathcal{Y}, \ell$ | 출력 $\hat{h}$ |
|---------|----------------------------------|---------------|
| **선형 회귀 (OLS)** | $\mathbb{R}^d, \mathbb{R}, (\hat{y}-y)^2$ | $\hat{h}(x) = \hat{w}^\top x$, $\hat{w} = (X^\top X)^{-1} X^\top y$ |
| **로지스틱 회귀** | $\mathbb{R}^d, \{0, 1\}, \text{cross-entropy}$ | $\hat{h}(x) = \sigma(\hat{w}^\top x)$, numerical optimization |
| **SVM** | $\mathbb{R}^d, \{\pm 1\}, \text{hinge}$ | $\hat{h}(x) = \text{sign}(\hat{w}^\top x + \hat{b})$, QP |
| **결정 트리** | $\mathbb{R}^d, \mathcal{Y}, \text{Gini/엔트로피}$ | greedy partition |
| **Neural Network** | $\mathbb{R}^d, \mathcal{Y}, \text{문제별}$ | SGD로 $\min L_S(f_\theta)$ |

이 모든 알고리즘은 **서로 다른 $\mathcal{H}$**에서 **서로 다른 $\ell$**로 **동일한 ERM 원리**를 구현한다(Ch1-03). 어떤 $\mathcal{D}$에서 어떤 조합이 잘 일반화하는지가 Ch3~Ch5의 질문이다.

---

## ⚖️ 가정과 한계

1. **IID 가정**: 현실에서 자주 깨진다 — 시계열(temporal correlation), 분포 이동(covariate shift), 계층 샘플링, active learning. Ch1-05 참조.
2. **분포 $\mathcal{D}$ 존재 가정**: "세상"을 하나의 확률분포로 모델링하는 것이 정당한가? 적대적 예제(adversarial examples)나 OOD는 이 모델을 부분적으로 깬다.
3. **손실 $\ell$의 선택**: 0-1 loss는 직관적이지만 **비볼록·비미분**이라 실용 알고리즘은 surrogate loss(hinge, log)로 대체한다. surrogate의 정당화가 **calibration**과 **contraction lemma**(Ch5-03)로 이어진다.
4. **라벨 $Y$ 관찰 가능 가정**: 지도학습만 다룸. 비지도·자기지도·강화학습은 다른 정식화가 필요하다.
5. **유한 $n$**: 모든 SLT bound는 "$n$이 충분히 클 때만 의미 있음" — 실전의 작은 $n$·큰 $\mathcal{H}$(DL 상황)에서는 고전 bound가 **vacuous**(Ch4-07).

---

## 📌 핵심 정리

- **학습 문제** = $(\mathcal{X}, \mathcal{Y}, \mathcal{D}, \ell)$ 4-튜플. 우리는 $\mathcal{D}$를 모른 채 샘플 $S \sim \mathcal{D}^n$만 관찰한다.
- **두 risk**: $L_\mathcal{D}(h) = \mathbb{E}[\ell(h(X), Y)]$ (알고 싶음), $L_S(h) = \frac{1}{n}\sum \ell(h(x_i), y_i)$ (계산 가능).
- **비편향성**: 고정 $h$에서 $\mathbb{E}_S[L_S(h)] = L_\mathcal{D}(h)$. **분산** $\propto 1/n$. SLLN으로 수렴.
- **단, 데이터 의존적 $\hat{h}$**에서는 $\mathbb{E}_S[L_S(\hat{h})]$가 $\mathbb{E}_S[L_\mathcal{D}(\hat{h})]$보다 **작을 수 있다** — SLT 전체의 출발점.
- **학습기 $A$**는 $S \mapsto \hat{h}_S$로 가설을 출력하는 **확률변수**.
- **모든 이후 챕터**는 이 정식화 위에서 $\sup_h |L_\mathcal{D}(h) - L_S(h)|$를 bound한다.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> $\mathcal{Y} = \{0, 1\}$, $\ell$이 0-1 loss일 때 $L_\mathcal{D}(h) = \mathbb{P}_{(X,Y) \sim \mathcal{D}}(h(X) \neq Y)$임을 보여라.</summary>

<br/>

**해설**. $\ell(h(X), Y) = \mathbb{1}[h(X) \neq Y]$이므로
$$L_\mathcal{D}(h) = \mathbb{E}[\mathbb{1}[h(X) \neq Y]] = \mathbb{P}(h(X) \neq Y).$$
지시함수의 기대값은 해당 사건의 확률이다(확률의 기본 성질). $\square$

즉, 0-1 loss의 risk는 **오분류 확률**과 같다. 이 등식이 이후 Hoeffding으로 **오분류 확률** 자체를 경계 짓는 기반이 된다.

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> 일반적으로 $\mathbb{E}_S[\min_{h \in \mathcal{H}} L_S(h)] \leq \min_{h \in \mathcal{H}} L_\mathcal{D}(h)$이 성립함을 보여라. 즉, ERM 해의 훈련 오차 기대값은 "최선" 진짜 오차보다 작거나 같다.</summary>

<br/>

**해설**. 고정된 $h' \in \mathcal{H}$에 대해 $L_S(h') \geq \min_{h \in \mathcal{H}} L_S(h)$. 양변에 기대값을 취하고 정리 1.1을 쓰면
$$L_\mathcal{D}(h') = \mathbb{E}[L_S(h')] \geq \mathbb{E}[\min_h L_S(h)].$$
이 부등식이 **모든 $h' \in \mathcal{H}$**에서 성립하므로 $h'$에 대해 min을 취하면
$$\min_{h' \in \mathcal{H}} L_\mathcal{D}(h') \geq \mathbb{E}[\min_h L_S(h)]. \qquad \square$$

이것은 "**훈련 오차의 기대값은 진짜 최적 오차보다 낙관적**"이라는 과적합의 수학적 씨앗이다 — Jensen 부등식의 한 형태로도 볼 수 있다.

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> Cross-entropy loss $\ell(\hat{p}, y) = -\log \hat{p}_y$ (단, $\hat{p}$는 확률벡터)에 대해 $L_\mathcal{D}(h) - L^*$가 **KL divergence**로 쓰일 수 있음을 보여라 — 즉, $L_\mathcal{D}$ 최소화가 참 조건부 분포를 매칭하는 것과 동치임을 관찰하라.</summary>

<br/>

**해설**. $p^*(y | x) := \mathbb{P}(Y = y | X = x)$라 두면
$$L_\mathcal{D}(h) = \mathbb{E}_X\!\left[\mathbb{E}_{Y|X}[-\log h(X)_Y]\right] = \mathbb{E}_X\!\left[-\sum_y p^*(y|X) \log h(X)_y\right] = \mathbb{E}_X[H(p^*(\cdot|X), h(X))]$$
여기서 $H(p, q) = -\sum p_y \log q_y$는 cross-entropy. $H(p, q) = H(p) + \text{KL}(p \| q)$ 분해에 의해
$$L_\mathcal{D}(h) = \mathbb{E}_X[H(p^*(\cdot|X))] + \mathbb{E}_X[\text{KL}(p^*(\cdot|X) \| h(X))].$$
첫째 항은 $h$에 무관하므로 $L_\mathcal{D}$ 최소화는
$$\min_h \mathbb{E}_X[\text{KL}(p^*(\cdot|X) \| h(X))] \Leftrightarrow h \equiv p^*.$$
즉 cross-entropy ERM은 **조건부 분포 추정**과 수학적으로 동치이며, Bayes risk $L^* = \mathbb{E}_X[H(p^*(\cdot|X))]$는 **조건부 엔트로피** — 제거 불가능한 불확실성. $\square$

이 관찰은 Ch1-02의 Bayes 최적 예측기 해석으로 직접 이어진다.

</details>

---

<div align="center">

◀ [이전: README](../README.md) | [📚 README](../README.md) | [다음: 02. Bayes 최적 예측기 ▶](./02-bayes-optimal.md)

</div>
