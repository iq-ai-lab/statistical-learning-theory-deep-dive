# 02. Bayes 최적 예측기와 Bayes error

## 🎯 핵심 질문

- 만약 분포 $\mathcal{D}$를 **완전히 알고 있다면**, 어떤 가설 $h^*$가 **가능한 최소 risk** $L^* = \inf_h L_\mathcal{D}(h)$를 달성하는가?
- 왜 **회귀에서는 조건부 기대값** $f^*(x) = \mathbb{E}[Y | X = x]$가, **분류에서는 사후확률 최대화** $h^*(x) = \arg\max_y \mathbb{P}(Y = y | X = x)$가 최적인가?
- **Bayes error** $L^*$가 왜 "**도달 불가능한 하한**"인가? 어떤 조건에서 $L^* = 0$이 되는가?
- 손실함수 $\ell$이 다르면 최적 예측기가 왜 달라지는가 — MSE는 평균, MAE는 중앙값, 0-1은 mode?
- 학습 알고리즘이 "잘 한다"는 것은 $L_\mathcal{D}(\hat{h}) - L^*$이 작다는 것 — 이 **excess risk**의 분해가 Ch1-03의 ERM 분석에 어떻게 이어지는가?

---

## 🔍 왜 Bayes 최적 예측기가 중요한가

"$L_\mathcal{D}(h)$를 최소화하라"는 우리의 목표다. 하지만 $\mathcal{D}$ 전체를 안다고 가정했을 때 **이론적으로 가능한 최소값**이 얼마인지 모르면, 우리 알고리즘이 "잘" 하는지 "못" 하는지 판단할 기준 자체가 없다. Bayes 최적 예측기 $h^*$와 Bayes error $L^*$는 **학습 문제의 지평선(horizon)** — 누구도 넘을 수 없는 벽이자, 우리가 평가받을 기준점이다.

이 개념은 실용적으로도 중요하다. 실전 벤치마크에서 "인간 성능"은 종종 Bayes error의 **상한 추정치**로 쓰인다. 의료 진단에서 최고 전문가가 5% 오진하는 문제라면, ML 모델의 이론 하한도 대체로 그 수준 근처다(라벨 noise 때문). "모델이 인간을 넘었다"는 주장은 실제로는 "$L^* < \text{human error}$일 가능성이 있다"는 주장이다. 또한 Bayes 예측기 유도 과정은 cross-entropy가 왜 확률 추정과 동치인지(Ch1-01 문제 3), regression에서 왜 MSE가 표준인지, 그리고 Ch1-03의 **excess risk 분해** $L_\mathcal{D}(\hat{h}) - L^* = \underbrace{\inf_\mathcal{H} L_\mathcal{D} - L^*}_{\text{approx}} + \underbrace{L_\mathcal{D}(\hat{h}) - \inf_\mathcal{H} L_\mathcal{D}}_{\text{est}}$의 논리적 뿌리다.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 조건부 기대값·조건부 확률, Tower property, 정규 조건분포
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): Bayes 규칙, 결정이론(decision theory), risk와 admissibility
- 기초: 볼록성, Jensen 부등식, $L^2$-투영의 유일성
- 선행: Ch1-01 (학습의 통계적 정의)

---

## 📖 직관적 이해

### "완벽한 정보 = 조건부 분포"

각 $x$에 대해 $Y$는 **하나로 정해지지 않을 수 있다** — 같은 흉부 X-선에도 실제로는 폐암과 양성 종양이 섞인 분포가 존재한다. $Y | X = x$의 **조건부 분포** $p(\cdot | x)$가 우리가 상상할 수 있는 "세상의 진실"의 전부이고, Bayes 최적 예측은 이 조건부 분포를 $\ell$에 맞춰 **점별로 요약**하는 작업이다.

### 손실함수가 "요약 통계"를 결정한다

| 손실 $\ell(\hat{y}, y)$ | 최적 $h^*(x)$ | 의미 |
|------------------------|---------------|------|
| $(\hat{y} - y)^2$ (squared) | $\mathbb{E}[Y \| X=x]$ | 조건부 **평균** |
| $\|\hat{y} - y\|$ (absolute) | $\text{median}(Y \| X=x)$ | 조건부 **중앙값** |
| $\mathbb{1}[\hat{y} \neq y]$ (0-1) | $\arg\max_y \mathbb{P}(Y=y \| X=x)$ | 조건부 **mode** |
| quantile loss $\rho_\tau$ | $\tau$-분위수 $Q_\tau(Y \| X=x)$ | 조건부 분위수 |
| cross-entropy | 조건부 분포 $p(\cdot \| x)$ 자체 | 분포 매칭 |

각 loss는 $Y | X$의 **다른 요약 통계**를 "정답"으로 지정하므로, 최적 예측기가 다르다. 학습 알고리즘 선택은 단지 계산 편의가 아니라 **"어떤 요약을 원하는가"의 선언**이다.

### Bayes error의 두 원천

$L^* > 0$인 이유는 두 가지다:

1. **본질적 불확실성(aleatoric)**: $X$가 주어져도 $Y$가 결정되지 않는 경우 — 이미지 해상도 부족, 센서 노이즈, 확률적 물리.
2. **라벨 noise**: 어노테이터의 실수, 주관적 기준.

**$L^* = 0$이 되는 유일한 경우**는 $Y$가 $X$의 **결정적 함수**일 때다(realizable 시나리오) — 이는 Ch3-02에서 다룬다.

---

## ✏️ 엄밀한 정의

### 정의 2.1 (Bayes risk)

학습 문제 $(\mathcal{X}, \mathcal{Y}, \mathcal{D}, \ell)$에서 **모든 가측함수** $h: \mathcal{X} \to \mathcal{Y}$에 대한 최소 risk:
$$L^* := \inf_{h \in \mathcal{Y}^\mathcal{X} \text{ 가측}} L_\mathcal{D}(h).$$

이 하한을 달성하는 $h^*$(존재 시)를 **Bayes 최적 예측기(Bayes optimal predictor)**라 한다.

### 정의 2.2 (Excess risk)

가설 $h$의 **excess risk** 또는 **regret**는
$$\mathcal{E}(h) := L_\mathcal{D}(h) - L^*.$$

학습의 궁극 목표는 $\mathcal{E}(\hat{h}) \to 0$ 이다.

### 정의 2.3 (Bayes 예측기의 예시)

$p(y | x)$를 $Y | X = x$의 조건부 분포라 하자.

**(회귀, squared loss)**:
$$h^*(x) = \mathbb{E}[Y | X = x] = \int y \, dp(y | x).$$

**(이진 분류, 0-1 loss)**:
$$h^*(x) = \mathbb{1}[\eta(x) \geq 1/2], \quad \text{where } \eta(x) := \mathbb{P}(Y = 1 | X = x).$$

**(다중 분류, 0-1 loss)**:
$$h^*(x) = \arg\max_{y \in \mathcal{Y}} \mathbb{P}(Y = y | X = x).$$

---

## 🔬 정리와 증명

### 정리 2.1 (회귀의 Bayes 최적 — 조건부 기대값)

$\mathcal{Y} = \mathbb{R}$, $\ell(\hat{y}, y) = (\hat{y} - y)^2$, $\mathbb{E}[Y^2] < \infty$일 때
$$h^*(x) = \mathbb{E}[Y | X = x]$$
이 Bayes 최적이며, Bayes risk는
$$L^* = \mathbb{E}_X\!\left[\text{Var}(Y | X)\right].$$

**증명**. 임의의 가측 $h$에 대해 $L_\mathcal{D}(h) = \mathbb{E}[(h(X) - Y)^2]$. Tower property로 조건화:
$$L_\mathcal{D}(h) = \mathbb{E}_X\!\left[\mathbb{E}_Y[(h(X) - Y)^2 | X]\right].$$
고정된 $x$에서 $h(x) = c$라 두고 "$c$에 대해 $\mathbb{E}[(c - Y)^2 | X = x]$를 최소화"하는 문제를 푼다:
$$\mathbb{E}[(c - Y)^2 | X=x] = c^2 - 2c \mathbb{E}[Y|X=x] + \mathbb{E}[Y^2|X=x].$$
$c$에 대해 미분 → $2c - 2\mathbb{E}[Y | X = x] = 0 \Rightarrow c = \mathbb{E}[Y | X = x]$. 이차식의 계수가 양수이므로 최솟값이다.

따라서 $h^*(x) = \mathbb{E}[Y | X = x]$가 **점별로** 최적이고, **기대값의 단조성**에 의해 전역 최적이다. 최솟값은
$$\mathbb{E}[(h^*(X) - Y)^2 | X] = \mathbb{E}[(Y - \mathbb{E}[Y|X])^2 | X] = \text{Var}(Y | X),$$
기대값을 취하면 $L^* = \mathbb{E}_X[\text{Var}(Y|X)]$. $\square$

### 정리 2.2 (이진 분류의 Bayes 최적 — 사후확률 임계)

$\mathcal{Y} = \{0, 1\}$, $\ell$이 0-1 loss일 때, $\eta(x) := \mathbb{P}(Y = 1 | X = x)$에 대해
$$h^*(x) = \mathbb{1}[\eta(x) \geq 1/2]$$
이 Bayes 최적이며,
$$L^* = \mathbb{E}_X[\min(\eta(X), 1 - \eta(X))].$$

**증명**. 고정된 $x$에서 $h(x) \in \{0, 1\}$. 두 선택의 조건부 risk:
$$\mathbb{E}[\mathbb{1}[h(x) \neq Y] | X = x] = \begin{cases} \eta(x), & h(x) = 0 \\ 1 - \eta(x), & h(x) = 1 \end{cases}$$

따라서 점별 최적은 $h(x) = 1$ if $1 - \eta(x) \leq \eta(x)$ (즉 $\eta(x) \geq 1/2$), 아니면 $h(x) = 0$. 각 $x$에서의 최소값은 $\min(\eta(x), 1 - \eta(x))$이고, 기대값을 취하면 $L^*$. $\square$

> **주의**: $\eta(x) = 1/2$인 점들의 집합에서는 **두 선택이 모두 최적**이므로 Bayes 예측기가 유일하지 않을 수 있다. 측도 0 집합에서의 이런 모호성은 해석에 영향을 주지 않는다.

### 정리 2.3 (다중 분류의 Bayes 최적)

$\mathcal{Y} = \{1, \ldots, K\}$, 0-1 loss일 때
$$h^*(x) = \arg\max_{k \in \{1, \ldots, K\}} \mathbb{P}(Y = k | X = x).$$

**증명**. $x$에서 $h(x) = k$ 선택의 조건부 risk는 $1 - \mathbb{P}(Y = k | X = x)$. 이를 최소화하려면 **사후확률 최대**인 $k$를 고른다. $\square$

### 정리 2.4 (절댓값 손실의 Bayes 최적 — 중앙값)

$\mathcal{Y} = \mathbb{R}$, $\ell(\hat{y}, y) = |\hat{y} - y|$일 때 $h^*(x)$는 **조건부 분포의 중앙값**이다.

**증명**. $g(c) := \mathbb{E}[|c - Y| | X = x]$. $Y | X = x$의 CDF를 $F$라 두면
$$g(c) = \int_{-\infty}^c (c - y) dF(y) + \int_c^\infty (y - c) dF(y).$$
미분(Leibniz rule):
$$g'(c) = F(c) - (1 - F(c)) = 2F(c) - 1.$$
$g'(c) = 0 \iff F(c) = 1/2$, 즉 $c$는 **중앙값**. $g''(c) = 2 F'(c) \geq 0$이므로 최솟값. $\square$

### 정리 2.5 (Excess risk 분해의 출발점)

임의의 가설공간 $\mathcal{H}$와 $\hat{h} \in \mathcal{H}$에 대해
$$\mathcal{E}(\hat{h}) = L_\mathcal{D}(\hat{h}) - L^* = \underbrace{\left[\inf_{h \in \mathcal{H}} L_\mathcal{D}(h) - L^*\right]}_{\text{approximation error}} + \underbrace{\left[L_\mathcal{D}(\hat{h}) - \inf_{h \in \mathcal{H}} L_\mathcal{D}(h)\right]}_{\text{estimation error}}.$$

**증명**. $\mathcal{H}^* := \inf_{h \in \mathcal{H}} L_\mathcal{D}(h)$를 항에 더하고 빼면 즉시 유도된다. $\square$

- **Approximation error**: $\mathcal{H}$가 Bayes 예측기를 포함하지 못할 때의 "모델 한계". $\mathcal{H}$가 클수록 작음.
- **Estimation error**: 유한 샘플로 $\mathcal{H}$ 안에서 최적을 찾지 못한 양. $n$이 클수록 작음.

**Trade-off**: $\mathcal{H}$를 키우면 approximation은 작아지지만 estimation이 커진다 — **bias-variance**의 SLT 버전. Ch7의 SRM이 이를 정식으로 다룬다.

---

## 💻 NumPy 구현 검증

### 실험 1: 회귀의 Bayes 예측기와 Bayes risk

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# 진짜 분포: X ~ U[-3, 3], Y = sin(X) + noise, noise ~ N(0, sigma(X))
# heteroscedastic — Var(Y|X)가 X에 의존
def sample_D(n):
    X = rng.uniform(-3, 3, n)
    sigma_x = 0.1 + 0.3 * np.abs(np.sin(X))          # x마다 다른 noise
    Y = np.sin(X) + sigma_x * rng.standard_normal(n)
    return X, Y, sigma_x

X, Y, sigma = sample_D(5000)

# Bayes 최적: h*(x) = sin(x) (정의상)
x_grid = np.linspace(-3, 3, 200)
h_star = np.sin(x_grid)

plt.figure(figsize=(9, 4))
plt.scatter(X[:500], Y[:500], s=8, alpha=0.3, label='data')
plt.plot(x_grid, h_star, 'r-', lw=2, label='h*(x) = sin(x)')
plt.xlabel('X'); plt.ylabel('Y')
plt.title('Bayes 최적 회귀자 h*(x) = E[Y|X=x]')
plt.legend(); plt.tight_layout(); plt.show()

# Bayes risk = E[Var(Y|X)]
# Var(Y|X=x) = sigma(x)^2
x_fine = np.linspace(-3, 3, 10000)
sigma_fine = 0.1 + 0.3 * np.abs(np.sin(x_fine))
L_star_theoretical = np.mean(sigma_fine ** 2)
print(f'L* (이론) = E[sigma(X)^2] ≈ {L_star_theoretical:.4f}')

# 경험적으로 h*(X) = sin(X)의 위험 계산
L_h_star_emp = np.mean((np.sin(X) - Y) ** 2)
print(f'L_D(h*) (경험) = {L_h_star_emp:.4f}')
# → 두 값이 일치. L* = Bayes risk.
```

### 실험 2: 손실함수 바꾸면 Bayes 예측기가 바뀐다

```python
# skewed 분포: Y | X=0은 Exponential(1) — mean=1, median=log(2)≈0.693
Y_cond = rng.exponential(1.0, 100000)

# 각 상수 예측 c에 대한 loss들
cs = np.linspace(0, 3, 300)
mse  = [np.mean((c - Y_cond) ** 2)  for c in cs]
mae  = [np.mean(np.abs(c - Y_cond)) for c in cs]

c_mse  = cs[np.argmin(mse)]
c_mae  = cs[np.argmin(mae)]

print(f'MSE 최적 c* = {c_mse:.3f},  E[Y] = {Y_cond.mean():.3f}')
print(f'MAE 최적 c* = {c_mae:.3f},  median(Y) = {np.median(Y_cond):.3f}')
# → MSE는 평균 1.00 근처, MAE는 중앙값 0.69 근처. 정리 2.1과 2.4 확인.

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cs, mse, label='MSE risk')
ax.plot(cs, mae, label='MAE risk')
ax.axvline(c_mse, color='C0', ls='--', alpha=0.6)
ax.axvline(c_mae, color='C1', ls='--', alpha=0.6)
ax.set_xlabel('c'); ax.set_ylabel('risk')
ax.set_title('같은 분포, 다른 loss → 다른 최적 예측')
ax.legend(); plt.tight_layout(); plt.show()
```

### 실험 3: 이진 분류에서 $\eta(x)$로부터 Bayes risk 수치 계산

```python
# 2D 가우시안 혼합: P(Y=1|X)를 해석적으로 계산 가능
# X|Y=0 ~ N(mu_0, I), X|Y=1 ~ N(mu_1, I), P(Y=1) = 0.5
mu_0 = np.array([-1.0, 0.0])
mu_1 = np.array([ 1.0, 0.0])

def eta(x):
    """P(Y=1|X=x) — Bayes 규칙으로 닫힌 형태"""
    p0 = np.exp(-0.5 * np.sum((x - mu_0) ** 2, axis=-1))
    p1 = np.exp(-0.5 * np.sum((x - mu_1) ** 2, axis=-1))
    return p1 / (p0 + p1)

# 몬테카를로로 L*
N = 100000
Y = rng.integers(0, 2, N)
mu = np.where(Y[:, None] == 1, mu_1, mu_0)
X = mu + rng.standard_normal((N, 2))
eta_X = eta(X)
h_star = (eta_X >= 0.5).astype(int)
L_star_mc = np.mean(h_star != Y)
L_star_formula = np.mean(np.minimum(eta_X, 1 - eta_X))
print(f'L* (MC 경험): {L_star_mc:.4f}')
print(f'L* (E[min(η, 1-η)]): {L_star_formula:.4f}')
# → 두 값 일치, 이론 확인.
```

---

## 🔗 ML 알고리즘 연결

| 알고리즘 | "진짜로" 추정하려는 것 | 정당성 |
|---------|----------------------|--------|
| **최소제곱 회귀** | $\mathbb{E}[Y \| X]$ | 정리 2.1 |
| **Quantile regression** | $Q_\tau(Y \| X)$ | pinball loss 최적화 |
| **로지스틱 회귀** | $\eta(x) = \mathbb{P}(Y=1 \| x)$ | log-loss 최적화 → 정확히 $\eta$ 모수 매칭 |
| **Softmax 분류기** | $\mathbb{P}(Y=k \| X)$ 벡터 | cross-entropy (Ch1-01 문제 3) |
| **$k$-NN 분류** | $\eta(x)$ 국소 추정 | $k \to \infty, k/n \to 0$이면 $\eta$로 수렴 → $L_\mathcal{D}(h_k) \to L^*$ (Stone 1977) |

이는 "로지스틱 회귀는 분류기이지만 실제로는 **조건부 확률**을 추정한다"는 실무적 주장의 수학적 근거다. 확률 예측이 필요한 응용(calibration, 의사결정 임계)에서 로지스틱이 선호되는 이유.

---

## ⚖️ 가정과 한계

1. **조건부 분포의 존재**: Radon-Nikodym 도함수·regular conditional distribution이 필요. 일반 Polish 공간에서는 문제없지만 병리적 경우 주의.
2. **적분가능성**: MSE의 경우 $\mathbb{E}[Y^2] < \infty$, MAE는 $\mathbb{E}[|Y|] < \infty$가 있어야 **유한** 최적 예측기가 존재.
3. **유일성**: $\eta(x) = 1/2$ 같은 boundary 지점, 이산 조건분포의 절댓값 손실 등에서는 **최적이 여러 개**. 측도 0 집합에서의 문제.
4. **계산 가능성**: Bayes 예측기는 **이론적 실체** — $\mathcal{D}$를 모르면 계산 불가. ERM(Ch1-03)은 $\mathcal{D}$ 대신 $S$로 근사한다.
5. **loss-consistency**: surrogate loss(hinge, log)로 최적화해 얻은 $\hat{h}$가 0-1 loss 관점에서도 Bayes 최적에 수렴하는가? — **calibration 이론**의 주제 (Bartlett, Jordan, McAuliffe 2006).

---

## 📌 핵심 정리

- **Bayes risk** $L^* = \inf_h L_\mathcal{D}(h)$는 학습의 **도달 불가능한 하한**.
- **회귀(MSE)**: $h^*(x) = \mathbb{E}[Y|X=x]$, $L^* = \mathbb{E}[\text{Var}(Y|X)]$.
- **분류(0-1)**: $h^*(x) = \arg\max_y \mathbb{P}(Y=y|X)$, $L^* = \mathbb{E}[\min(\eta, 1-\eta)]$(이진).
- **절댓값**: 조건부 **중앙값**. **Quantile**: 조건부 **분위수**. **Cross-entropy**: 조건부 **분포 자체**.
- **Excess risk 분해**: $L_\mathcal{D}(\hat{h}) - L^* = \text{approx} + \text{est}$. Ch1-03의 3분해(+ optimization gap)로 확장된다.
- **실용 ML 알고리즘**은 사실상 각 loss가 지정하는 **조건부 요약 통계**를 추정한다.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> Huber loss $\ell(\hat{y}, y) = \frac{1}{2}(\hat{y}-y)^2 \cdot \mathbb{1}[|\hat{y}-y| \leq \delta] + (\delta|\hat{y}-y| - \delta^2/2) \cdot \mathbb{1}[|\hat{y}-y| > \delta]$에서 Bayes 최적 예측기가 "평균"과 "중앙값" 사이를 보간함을 논증하라.</summary>

<br/>

**해설**. 조건부 기대값
$$g(c) = \mathbb{E}[\ell(c, Y) | X=x]$$
의 미분은
$$g'(c) = \mathbb{E}\!\left[(c - Y) \mathbb{1}[|c-Y| \leq \delta] + \delta \cdot \text{sign}(c - Y) \mathbb{1}[|c - Y| > \delta] \,|\, X = x\right].$$

$\delta \to \infty$이면 두 번째 항이 사라지고 **MSE의 최적 = 평균**으로 수렴; $\delta \to 0$이면 첫 번째 항이 사라지고 **MAE의 최적 = 중앙값**으로 수렴. Huber의 $\delta$가 "평균-중앙값 보간 파라미터" — **outlier-robustness**의 수학적 기원. 이 관점은 M-estimator 이론의 핵심이다.

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> $\mathcal{Y} = \{0, 1\}$에서 0-1 loss의 excess risk가 $\eta$-plugin estimator $\hat{h}(x) = \mathbb{1}[\hat{\eta}(x) \geq 1/2]$에 대해 $\mathcal{E}(\hat{h}) \leq 2 \mathbb{E}_X[|\hat{\eta}(X) - \eta(X)|]$임을 보여라(Devroye, Györfi, Lugosi 1996, Thm 2.2).</summary>

<br/>

**해설**. 고정 $x$에서 $\hat{h}(x) \neq h^*(x)$일 조건은 $\hat{\eta}$와 $\eta$가 $1/2$에 대해 반대편에 있을 때. 이 경우 조건부 excess는
$$|2\eta(x) - 1| = |(\eta - 1/2) - (1/2 - \eta)| \leq 2|\hat{\eta}(x) - \eta(x)|$$
(삼각 부등식: $\hat{\eta}$와 $\eta$가 $1/2$ 양쪽에 있으려면 $|\hat{\eta} - \eta| \geq |\eta - 1/2|$). $\hat{h}(x) = h^*(x)$이면 조건부 excess는 0이므로
$$\mathcal{E}(\hat{h}) = \mathbb{E}_X[(2\eta(X) - 1) \cdot \mathbb{1}[\hat{h}(X) \neq h^*(X)]] \leq 2 \mathbb{E}_X[|\hat{\eta}(X) - \eta(X)|]. \qquad \square$$

핵심 시사점: **확률 추정 오차**($L^1$)가 **분류 오차**(0-1)를 통제한다. 로지스틱 회귀 같은 확률 기반 방법의 이론적 정당성.

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> ImageNet Top-5 accuracy 95%는 "$L^* < 5\%$를 의미한다"는 해석이 맞는가? 아니면 "$L^* \leq 5\%$의 상한만 제공한다"인가? 둘의 차이를 논증하라.</summary>

<br/>

**해설**. 모델의 경험 오차 $L_S(\hat{h}) = 5\%$는 **특정 모델의 test set 성능**이지 Bayes error가 아니다. $L^* \leq L_\mathcal{D}(\hat{h})$는 항상 성립하고, $L_\mathcal{D}(\hat{h}) \approx L_S(\hat{h}) = 5\%$라면 **$L^* \leq 5\%$의 상한**이 얻어진다. 그러나 $L^*$가 **정확히 5%**라는 주장은 $\hat{h}$가 이미 Bayes optimal임을 의미하며, 이것은 오직 "더 나은 모델이 불가능함"을 증명해야 성립한다.

실무적으로 ImageNet의 라벨 noise(틀린 ground truth, 모호한 이미지)를 감안하면 $L^*$은 상당히 0보다 크다. Recht et al. (2019)의 ImageNet V2 연구는 같은 모델이 분포 이동(Ch1-05) 하에 3-5% 떨어지는 것을 보였고, Northcutt et al. (2021)은 ImageNet test set의 라벨 오류율을 ~6%로 추정했다. 이는 **"$L^* \approx 5$-$6\%$의 근처"**라는 추정과 잘 맞는다 — 인간 성능도 비슷하다. 즉 현대 분류기는 Bayes error의 **지평선에 상당히 근접**했다.

</details>

---

<div align="center">

◀ [이전: 01. 학습의 통계적 정의](./01-statistical-learning.md) | [📚 README](../README.md) | [다음: 03. ERM ▶](./03-erm-principle.md)

</div>
