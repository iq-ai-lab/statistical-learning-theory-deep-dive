# 04. 일반화 오차와 과적합의 수학적 정의

## 🎯 핵심 질문

- **Generalization gap** $L_\mathcal{D}(h) - L_S(h)$는 확률변수로서 어떤 성질을 갖는가 — 평균, 분산, 분포?
- **과적합(overfitting)**을 수학적으로 어떻게 정의해야 "직관적 현상"과 "엄밀한 진술"이 일치하는가?
- **No Free Lunch 정리**: $\mathcal{H}$가 제약되지 않으면 왜 **어떤 알고리즘도 보편적으로 학습할 수 없는가**? 증명의 핵심 아이디어는?
- Test error가 Training error보다 왜 **거의 항상** 크거나 같은 경향을 보이는가 — "optimism of training error" (Efron 1986)의 정확한 의미?
- "**좋은 일반화**"를 뭐라고 정의해야 — $L_\mathcal{D}(\hat{h}) \to L^*$인가, $L_\mathcal{D}(\hat{h}) - L_S(\hat{h}) \to 0$인가, 아니면 둘 다?

---

## 🔍 왜 "과적합"을 엄밀히 정의해야 하는가

실무에서 "과적합했다"는 말은 흔하지만 정확한 정의는 의외로 까다롭다. Train acc 99%, test acc 80%면 과적합? 그럼 train 99%, test 98.5%는? Train 50%, test 45%는 과적합이 아닌데도 **모델이 나쁘다**. 이런 혼동을 해결하려면 과적합을 **"generalization gap이 크다"**로 정의할 것인가, **"approximation-estimation trade-off에서 estimation이 지배한다"**로 정의할 것인가, 아니면 **"validation loss가 상승하는 훈련 점"**으로 정의할 것인가?

이 문서는 세 정의가 **같은 현상을 보지만 다른 각도에서 본다**는 것을 보인다. 또한 No Free Lunch 정리로 "과적합을 완전히 없애는 보편 알고리즘은 존재하지 않는다"는 **부정적 결과**를 증명한다 — 이것이 왜 SLT 전체가 "**$\mathcal{H}$ 제약**"에 의존하는지의 수학적 이유다. 실용적으로 이 정의들은 **regularization**, **early stopping**, **cross-validation**의 방법론적 선택을 justify하는 언어를 제공한다.

---

## 📐 수학적 선행 조건

- Ch1-01~03 (학습 정식화, Bayes 최적, ERM)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 확률변수의 기대값·분산, 집중부등식 개요
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): Optimism of training error, AIC의 유도 (Ch7-02 예고)
- 기초: 조합 argument, pigeonhole principle (NFL 증명)

---

## 📖 직관적 이해

### "알고리즘이 얼마나 정직한가"가 핵심

$L_S(\hat{h})$는 **알고리즘이 자랑하는 점수**다 — "내가 훈련 데이터에서 얼마나 잘했는지 봐!". $L_\mathcal{D}(\hat{h})$는 **실제 점수** — 배포되어 만나는 새 데이터에서의 성능. 이 둘의 차이 $L_\mathcal{D}(\hat{h}) - L_S(\hat{h})$는 **"알고리즘의 자기평가가 얼마나 낙관적인가"**. 모든 SLT는 이 낙관성을 **확률적으로 통제**하려는 시도다.

### 과적합의 세 관점

**관점 1: Gap 관점** — $L_\mathcal{D}(\hat{h}) - L_S(\hat{h})$가 크다. 훈련 정확도는 높지만 배포 성능은 낮다.

**관점 2: Excess risk 분해 관점** — $\text{err}_{\text{est}}$가 $\text{err}_{\text{approx}}$보다 훨씬 크다. 즉 $\mathcal{H}$가 너무 풍부해서 유한 샘플로 그 안의 최적을 못 찾는다.

**관점 3: Training dynamics 관점** — 훈련 진행에 따라 train loss는 계속 내려가는데 validation loss는 어느 순간부터 올라간다. "Early stopping point 이후의 구간"이 과적합.

세 관점은 **다른 양을 측정**하지만 **같은 현상을 가리킨다**. 관점 1은 $\hat{h}$의 질, 관점 2는 $\mathcal{H}$의 적합성, 관점 3은 알고리즘의 경로를 본다.

### No Free Lunch의 직관

"모든 분포 $\mathcal{D}$에서 모든 알고리즘이 잘 할 수 있는가?" 답은 **아니다**. 증명의 핵심: 유한 입력 공간에서 "$\mathcal{D}$ = 균등분포 라벨링"이 **너무 많아**서, 임의의 알고리즘 $A$에 대해 $A$가 못 맞히는 분포가 반드시 존재한다. 즉 SLT의 분석은 본질적으로 **"어떤 $\mathcal{D}$와 어떤 $\mathcal{H}$의 조합"**에 대한 주장이지, "모든 $\mathcal{D}$"에 대한 보편 주장이 아니다.

---

## ✏️ 엄밀한 정의

### 정의 4.1 (Generalization gap)

가설 $h$ (혹은 알고리즘 출력 $\hat{h}_S$)의 **generalization gap**:
$$\Delta(h) := L_\mathcal{D}(h) - L_S(h).$$

$h$가 **$S$에 의존적**이면 $\Delta$는 확률변수 ($S$에 대해).

### 정의 4.2 (과적합의 세 정의)

**(a) Gap-based**: 알고리즘 $A$가 분포 $\mathcal{D}$에서 $(\epsilon, n)$-overfitting ⇔ $\mathbb{P}_{S \sim \mathcal{D}^n}[\Delta(A(S)) \geq \epsilon] \geq 1/2$.

**(b) Excess-risk-based**: 알고리즘 $A$가 $\mathcal{D}$에서 과적합 ⇔ $\mathbb{E}_S[L_\mathcal{D}(A(S))] - L^* > 2(\inf_\mathcal{H} L_\mathcal{D} - L^*)$ (estimation이 approximation보다 지배적).

**(c) Training-curve-based**: 훈련 에폭 $t$에 대해, validation $L_{S_{\text{val}}}(\hat{h}_t)$가 $t^* < \infty$에서 최소를 달성하는 $t$가 존재하면, $t > t^*$ 구간을 "과적합 구간"이라 한다.

### 정의 4.3 (Consistency / 일관성)

학습기 $A$가 $\mathcal{D}$에서 **(약한) 일관적(weakly consistent)** ⇔
$$L_\mathcal{D}(A(S)) \xrightarrow{\text{in prob}} L^* \quad \text{as } n \to \infty.$$

**강한 일관성**은 a.s. 수렴. **보편적(universally) 일관**은 모든 $\mathcal{D}$에서 일관적.

### 정의 4.4 (No Free Lunch 설정)

$\mathcal{X}$가 유한, $|\mathcal{X}| = 2n$, $\mathcal{Y} = \{0, 1\}$, $\ell$이 0-1 loss.

---

## 🔬 정리와 증명

### 정리 4.1 (고정 $h$에서 generalization gap의 집중)

고정 가설 $h$와 $\ell \in [0, 1]$ iid 샘플에 대해, Hoeffding:
$$\mathbb{P}(|\Delta(h)| \geq \epsilon) \leq 2 e^{-2n\epsilon^2}.$$
따라서 $|\Delta(h)| = O(1/\sqrt{n})$ 고확률.

**증명**. 정리 1.1에 의해 $\mathbb{E}_S[\Delta(h)] = 0$. $Z_i = \ell(h(x_i), y_i) \in [0, 1]$ iid, $\bar{Z} = L_S(h)$, Hoeffding(Ch2-02) 적용. $\square$

> **한계**: 이는 **고정 $h$**에서만. $\hat{h}$가 $S$ 의존이면 Hoeffding을 곧바로 쓸 수 없다.

### 정리 4.2 (Training error의 낙관성 — Optimism of training error)

고정 $\mathcal{H}$와 ERM $\hat{h}^*_S$에 대해
$$\mathbb{E}_S[L_S(\hat{h}^*_S)] \leq \mathbb{E}_S[L_\mathcal{D}(\hat{h}^*_S)].$$

**증명**. Ch1-01 문제 2의 결과: $\mathbb{E}_S[\min_h L_S(h)] \leq \min_h L_\mathcal{D}(h) = \mathbb{E}_S[L_\mathcal{D}(h^*_\mathcal{H})]$. 또한 $L_\mathcal{D}(\hat{h}^*_S) \geq L_\mathcal{D}(h^*_\mathcal{H})$ (정의상). 따라서
$$\mathbb{E}_S[L_S(\hat{h}^*_S)] \leq L_\mathcal{D}(h^*_\mathcal{H}) \leq \mathbb{E}_S[L_\mathcal{D}(\hat{h}^*_S)]. \qquad \square$$

이것이 "**훈련 오차는 평균적으로 진짜 오차보다 낙관적**"의 수학적 형태. 간단 버전의 AIC·$C_p$ statistic이 이 낙관성을 추정한다 (Ch7-02).

### 정리 4.3 (No Free Lunch — Shalev-Shwartz & Ben-David Thm 5.1)

$\mathcal{X}$가 유한, $|\mathcal{X}| = 2n$, $\mathcal{Y} = \{0, 1\}$, $\ell$이 0-1 loss일 때: 임의의 학습 알고리즘 $A$에 대해 어떤 분포 $\mathcal{D}$가 존재해:
- **Realizable**: $\exists h^* \in \mathcal{Y}^\mathcal{X}, L_\mathcal{D}(h^*) = 0$.
- **그러나**: $\mathbb{P}_{S \sim \mathcal{D}^n}[L_\mathcal{D}(A(S)) \geq 1/8] \geq 1/7$.

**증명 스케치**. $\mathcal{X} = \{c_1, \ldots, c_{2n}\}$로 표기. $T = 2^{2n}$개의 가능한 라벨링 $f_1, \ldots, f_T: \mathcal{X} \to \{0, 1\}$을 고려. 각 $f_i$에 대해 분포 $\mathcal{D}_i$: $X$는 $\mathcal{X}$에서 균등, $Y = f_i(X)$ (결정적).

샘플 $S$에서 $A$가 $\hat{h} = A(S)$를 출력하면,
$$L_{\mathcal{D}_i}(\hat{h}) = \frac{1}{2n} \sum_{x \in \mathcal{X}} \mathbb{1}[\hat{h}(x) \neq f_i(x)] \geq \frac{1}{2n} \sum_{x \in \mathcal{X} \setminus S_X} \mathbb{1}[\hat{h}(x) \neq f_i(x)],$$
여기서 $S_X = \{x_1, \ldots, x_n\}$은 샘플의 입력 부분. $|\mathcal{X} \setminus S_X| \geq n$.

**핵심 관찰**: 샘플 $S$에서 관측되지 않은 $x \in \mathcal{X} \setminus S_X$에서 $\hat{h}(x)$는 $f_i(x)$와 **무관**하다 (라벨을 본 적 없으므로). 따라서 모든 $f_i$에 대해 평균을 내면 $\mathbb{E}_i[L_{\mathcal{D}_i}(\hat{h})] \geq 1/4$ (절반 확률로 틀림, $x \in \mathcal{X} \setminus S_X$에서만).

pigeonhole: 평균이 $\geq 1/4$이면 어떤 $f_i$에서 $L_{\mathcal{D}_i}(\hat{h}) \geq 1/4$. Markov 부등식으로 $\mathbb{P}[L_\mathcal{D}(A(S)) \geq 1/8] \geq 1/7$을 얻음. 자세한 상수는 SSBD Lemma B.1. $\square$

> **해석**: $\mathcal{H}$ 제약 없이 "모든 분포에서 잘 하는 알고리즘"은 불가능. SLT의 모든 주장은 **"$\mathcal{H}$가 VC 유한" 같은 제약 하에서**의 주장이다.

### 정리 4.4 (Generalization gap vs Excess risk — 서로 다른 양)

다음 두 명제는 **독립**이다:

- (a) $L_\mathcal{D}(\hat{h}) - L_S(\hat{h}) \approx 0$ (gap 작음)
- (b) $L_\mathcal{D}(\hat{h}) - L^* \approx 0$ (excess 작음)

**예시 1**(a but not b): $\hat{h}$가 상수 함수 "항상 0" 출력. 어떤 분포에서 $L_\mathcal{D}(\hat{h}) = L_S(\hat{h}) = 0.5$. Gap은 0이지만 excess는 크다.

**예시 2**(b but not a): Noisy classification에서 **훈련 샘플을 기억**한 $\hat{h}$가 $L_S = 0$, 하지만 $L_\mathcal{D} \approx L^*$도 될 수 있음 (쌍대 예). 일반적으로 memorization은 gap이 크다.

따라서 **"좋은 학습"의 정의**는 **excess risk**이지 gap이 아니다. 하지만 실무에서 $L^*$를 모르므로 **$L_S$를 대용**으로 사용해 gap을 본다.

---

## 💻 NumPy 구현 검증

### 실험 1: Optimism of training error

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

# 2D 가우시안 혼합 분류
def sample(n):
    y = rng.integers(0, 2, n)
    mu = np.where(y[:, None] == 0, [-1, 0], [1, 0])
    X = mu + rng.standard_normal((n, 2))
    return X, 2*y - 1  # {-1, +1}

# H: 랜덤 K개의 선형 분류기
K = 1000
W = rng.standard_normal((K, 3))   # (w1, w2, b)

def preds(X, W):
    Xb = np.hstack([X, np.ones((len(X), 1))])  # (n, 3)
    return np.sign(Xb @ W.T)       # (n, K)

# ERM over H
def erm(X, y, W):
    P = preds(X, W)
    err = np.mean(P != y[:, None], axis=0)  # (K,)
    return W[np.argmin(err)]

n_trials = 200
ns = [20, 50, 100, 500]
opt_gaps = []
for n in ns:
    gaps = []
    for _ in range(n_trials):
        Xtr, ytr = sample(n)
        w_hat = erm(Xtr, ytr, W)
        LS = np.mean(np.sign(np.hstack([Xtr, np.ones((n,1))]) @ w_hat) != ytr)
        # LD 근사 (큰 샘플)
        Xte, yte = sample(20000)
        LD = np.mean(np.sign(np.hstack([Xte, np.ones((20000,1))]) @ w_hat) != yte)
        gaps.append(LD - LS)
    opt_gaps.append(np.mean(gaps))

print("n       E[L_D - L_S]   (optimism)")
for n, g in zip(ns, opt_gaps):
    print(f"{n:4d}    {g:.4f}")
# 일반적으로 정리 4.2대로 양수. n이 증가하면 감소.
```

### 실험 2: No Free Lunch의 유한 재구현

```python
# |X| = 6, 이진 라벨 → 2^6 = 64개의 f_i
X_space = np.arange(6)
all_labelings = np.array([[int(b) for b in format(i, '06b')] for i in range(64)])

n_samp = 3   # 샘플 크기 (< |X|/2 = 3)
n_reps = 2000

# 알고리즘: 본 라벨 그대로 유지, 안 본 x는 항상 0
def A(sample_xs, sample_ys):
    h = np.zeros(6, dtype=int)
    for x, y in zip(sample_xs, sample_ys):
        h[x] = y
    return h

errors = []
for rep in range(n_reps):
    # 랜덤으로 f_i 선택
    f = all_labelings[rng.integers(0, 64)]
    # D_i: X 균등, Y = f(X)
    idx = rng.choice(6, n_samp, replace=True)
    S_x = X_space[idx]; S_y = f[idx]
    h = A(S_x, S_y)
    L_D = np.mean(h != f)
    errors.append(L_D)

errors = np.array(errors)
print(f"Mean L_D over random f_i: {errors.mean():.3f}")
print(f"P(L_D ≥ 1/4): {np.mean(errors >= 0.25):.3f}")
# → 평균이 정리 4.3의 1/4 근처, 고확률로 1/4 이상.
# 즉 "어떤 f에서든 잘 하는 알고리즘"은 없음을 실험적으로 관찰.
```

### 실험 3: 훈련-검증 곡선의 U-shape (관점 3의 과적합)

```python
# 심플 neural net 대신 polynomial degree로 "훈련 복잡도" 시뮬레이션
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def sample_reg(n):
    X = rng.uniform(0, 1, n).reshape(-1, 1)
    Y = np.sin(2 * np.pi * X.ravel()) + 0.3 * rng.standard_normal(n)
    return X, Y

Xtr, Ytr = sample_reg(40)
Xval, Yval = sample_reg(5000)

degrees = range(1, 20)
Ls, Lv = [], []
for d in degrees:
    poly = PolynomialFeatures(d, include_bias=False)
    reg = LinearRegression().fit(poly.fit_transform(Xtr), Ytr)
    Ls.append(np.mean((reg.predict(poly.transform(Xtr)) - Ytr) ** 2))
    Lv.append(np.mean((reg.predict(poly.transform(Xval)) - Yval) ** 2))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(degrees), Ls, 'o-', label='L_S (training)')
ax.plot(list(degrees), Lv, 's-', label='L_D (≈ validation)')
ax.axvline(np.argmin(Lv)+1, color='r', ls='--', label=f'optimal d* = {np.argmin(Lv)+1}')
ax.set_xlabel('polynomial degree'); ax.set_ylabel('MSE'); ax.set_yscale('log')
ax.set_title('훈련/검증 곡선 — 과적합의 U-shape'); ax.legend()
plt.tight_layout(); plt.show()

# → L_S는 단조 감소, L_D는 U-모양. d > d* 구간이 "과적합 구간".
```

---

## 🔗 ML 알고리즘 연결

| 실무 기법 | 과적합 제어 원리 | SLT 해석 |
|---------|----------------|---------|
| **$\ell_2$ regularization** | $\|w\|$ 작게 유지 | $\mathcal{H}$의 "효과적 크기" 감소 → estimation error ↓ |
| **$\ell_1$ regularization** | sparsity | 활성 feature만 뽑아 $\mathcal{H}$ 축소 |
| **Dropout** | 확률적 서브네트 | 평균적 $\mathcal{H}$ 축소, stability↑ (Ch6) |
| **Early stopping** | 정해진 단계 후 중단 | SGD stability bound(Ch6-04) |
| **Cross-validation** | val loss로 hyperparam 선택 | $L_\mathcal{D}$ 경험 추정 (Ch7-03) |
| **Data augmentation** | 효과적 $n$ 증가 | estimation error $1/\sqrt{n}$ 개선 |
| **Batch normalization** | 표현의 정규화 | implicit regularization, stability |

**핵심 관찰**: 이 모든 기법은 **"$\mathcal{H}$ 효과적 축소" 혹은 "효과적 $n$ 증가"**의 다른 표현이다. SLT 관점에서 기법 간 차이는 정도의 문제지 본질적 차이가 아니다.

---

## ⚖️ 가정과 한계

1. **$L_\mathcal{D}$ 관찰 불가**: 실무에서 test set으로 **추정**할 뿐. Test set에서 hyperparameter를 tune하면 "effectively another training set"이 되어 과적합 위험(**test set contamination**).
2. **NFL의 강도**: NFL은 "최악의 경우" 주장. 실전 분포는 저차원 manifold나 특수 구조가 있어 훨씬 나은 성능 가능.
3. **관점 2 vs 관점 3**: 관점 2(excess risk)는 정적, 관점 3(training curve)은 동적. SGD의 **implicit regularization**은 관점 3에서만 보임.
4. **Epoch ≠ 모델 크기**: "같은 모델, 더 오래 훈련"과 "더 큰 모델"은 다른 효과. Double descent(Nakkiran et al. 2019)는 모델 크기 축에서만 보이는 현상.
5. **Test set의 통계적 성질**: test set도 샘플 $|S_{\text{test}}|$에 따라 variance가 있음. $L_{S_{\text{test}}}(\hat{h})$의 variance는 $1/n_{\text{test}}$.

---

## 📌 핵심 정리

- **Generalization gap**: $\Delta(h) = L_\mathcal{D}(h) - L_S(h)$. 고정 $h$에서 $|\Delta| = O(1/\sqrt{n})$ (Hoeffding).
- **과적합 세 정의**: gap 크다 / est err 지배 / val curve 상승 구간. **같은 현상**의 다른 각도.
- **Optimism of training error**: $\mathbb{E}[L_S(\hat{h}^*_S)] \leq \mathbb{E}[L_\mathcal{D}(\hat{h}^*_S)]$. **평균적으로 낙관적**.
- **No Free Lunch**: $\mathcal{H}$ 제약 없이 모든 분포에서 잘 하는 알고리즘 존재하지 않음. **SLT의 $\mathcal{H}$-중심성**의 이유.
- **Generalization gap ≠ excess risk**: gap이 작아도 risk가 높을 수 있고(상수 예측), 반대도 가능. **진짜 목표는 excess risk 최소화**.
- 모든 정규화 기법은 **"$\mathcal{H}$ 효과적 축소" 혹은 "효과적 $n$ 증가"** 중 하나.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> $\hat{h}$가 $S$ 의존적이라도 $\mathbb{E}_S[L_\mathcal{D}(\hat{h})] - \mathbb{E}_S[L_S(\hat{h})] \neq 0$이 가능함을 정리 4.2로 보여라. 부등식 방향은?</summary>

<br/>

**해설**. 정리 4.2: $\mathbb{E}_S[L_S(\hat{h}^*_S)] \leq \mathbb{E}_S[L_\mathcal{D}(\hat{h}^*_S)]$. 즉 $\mathbb{E}_S[L_\mathcal{D}(\hat{h})] - \mathbb{E}_S[L_S(\hat{h})] \geq 0$.

데이터 **의존적** $\hat{h}$에서는 이 기대값 차이(= $\mathbb{E}_S[\Delta(\hat{h})]$)가 **0이 아니다**. 고정 $h$에서는 정리 1.1로 정확히 0이었음. **차이가 0에서 벗어난 정도 = "과적합의 평균적 크기"**. 이것이 AIC (Ch7-02)의 근거이며, $\mathbb{E}[\Delta]$의 추정량(covariance estimator 등)이 실무 모델 선택에 직접 쓰인다.

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> 과적합의 세 정의 중 어느 것이 **알고리즘**의 성질이고, 어느 것이 **$\mathcal{H}$**의 성질이고, 어느 것이 **샘플**의 성질인가? 각각 대응시켜라.</summary>

<br/>

**해설**. 
- **관점 1(gap)**: 알고리즘 + 샘플 + 분포의 **합성** 성질. $\hat{h} = A(S)$가 $\mathcal{D}$에 얼마나 적응했는지.
- **관점 2(excess decomposition)**: **$\mathcal{H}$ + 분포**의 성질이 주된 요인. $A$가 ERM이면 거의 오직 $\mathcal{H}$와 $\mathcal{D}$의 궁합.
- **관점 3(training curve)**: **알고리즘(경로)**의 성질. 같은 $\mathcal{H}$·$\mathcal{D}$·$S$에서도 optimizer가 다르면 다른 경로 → implicit regularization.

이 구분은 **대응 기법**을 다르게 한다: 관점 1은 **데이터 증강**(샘플 크기↑), 관점 2는 **모델 축소**($\mathcal{H}$ 변경), 관점 3은 **early stopping**(경로 제어). 실무에서 "과적합 대응"이라 하면 보통 셋을 섞어서 하지만, 이론적으로는 분리 가능하다.

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> 현대 딥러닝에서 **"double descent"**(Belkin et al. 2019, Nakkiran et al. 2019): 모델 크기를 늘릴수록 test error는 (고전 U-shape을 지나) 다시 감소한다. 이 현상이 정리 4.3의 NFL이나 관점 2의 과적합 정의와 모순되는가?</summary>

<br/>

**해설**. **모순되지 않는다**. 
- **NFL은 여전히 성립**: "모든 분포에서 잘 하는 보편 알고리즘은 없다"는 여전히 참. Double descent는 **특정 $\mathcal{D}$**에서의 현상.
- **관점 2 해석**: 고전 U-shape 이후 "interpolation threshold"($n \approx$ 모델 파라미터 수)를 넘기면 **implicit regularization 메커니즘이 바뀐다**. Over-parameterized regime에서는 **minimum-norm solution을 향한 SGD의 bias**가 효과적 $\mathcal{H}$를 축소하는 새로운 기제가 된다. 즉 관점 2의 "estimation error"가 "$\mathcal{H}$ 크기"에 선형으로 반응하지 않고, **SGD의 경로가 고르는 특정 $h$들의 효과적 복잡도**에 반응한다.

이것을 **고전 SLT로는 설명 불가능**한 DL 현상의 대표 예로, Ch4-07·Ch5-06에서 다루고, **Generalization Theory** 레포(norm-based Rademacher, NTK)에서 본격적으로 분석한다.

</details>

---

<div align="center">

◀ [이전: 03. ERM](./03-erm-principle.md) | [📚 README](../README.md) | [다음: 05. IID 가정 ▶](./05-iid-assumption.md)

</div>
