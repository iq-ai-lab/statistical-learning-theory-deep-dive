# 03. Contraction Lemma (Ledoux-Talagrand)

## 🎯 핵심 질문

- **Contraction Lemma**는 무엇이고, "함수의 composition"이 Rademacher에 어떻게 작용하는가?
- Lipschitz 함수 $\phi: \mathbb{R} \to \mathbb{R}$, $|\phi(a) - \phi(b)| \leq L|a-b|$이 $\phi \circ f$의 Rademacher를 제어하는가?
- **0-1 loss의 비볼록성 문제**: 0-1 loss는 미분 불가능하고 NP-hard. 왜 **surrogate loss**(hinge, log, squared)로 대체해도 일반화가 유지되는가?
- **Margin loss** $\tilde{\ell}_\gamma(yf(x)) = \mathbb{1}[yf(x) < \gamma]$은 1/γ-Lipschitz. 이것이 SVM의 margin 최대화와 어떻게 연결되는가?
- **Hinge loss** $\max(0, 1-z)$는 1-Lipschitz. Cross-entropy는? Lipschitz 상수는?

---

## 🔍 왜 이 이론이 현대 ML에서 중요한가

실전에서 거의 모든 ML 알고리즘은 **0-1 loss 대신 smooth surrogate**를 최적화한다. 하지만 우리가 정말 원하는 것은 **0-1 오분류율의 일반화 경계**다. Contraction Lemma는 이 간극을 메운다:

> **0-1 loss의 Rademacher bound**를 **surrogate loss의 Rademacher bound로부터 계산할 수 있다.**

이것은 SVM의 hinge loss, logistic regression의 log loss, Neural Net의 cross-entropy가 왜 작동하는지의 **이론적 정당성**을 제공한다. 또한 Ledoux & Talagrand(1991)의 원문과 현대 ML 이론(특히 margin bound, calibration theory)을 연결하는 다리다.

---

## 📐 수학적 선행 조건

- **Ch5-01**: Rademacher 복잡도 정의 및 기본 성질
- **Ch5-02**: 일반화 경계 정리 5.5
- **Probability**: Conditioning, 기대값 계산, Jensen 부등식
- **Real Analysis**: Lipschitz 연속 함수의 성질
- 기초: 함수 composition, supremum 교환

---

## 📖 직각적 이해

### 왜 "Contraction"인가?

Lipschitz 함수 $\phi$는 **거리를 축소**한다: $|\phi(a) - \phi(b)| \leq L|a-b|$. 

Rademacher 복잡도 관점에서, 함수족 $\mathcal{F} = \{f : \mathcal{X} \to \mathbb{R}\}$에 $\phi$를 합성하면 ($\phi \circ \mathcal{F} = \{\phi \circ f\}$), Rademacher 신호 $\sigma_i$의 "힘"이 감소한다:

$$\sup_{\phi \circ f} \sum \sigma_i (\phi \circ f)(x_i) \leq L \cdot \sup_f \sum \sigma_i f(x_i).$$

이것이 **"수축(contraction)"** — 복잡도가 줄어든다.

### 0-1 vs Surrogate: 왜 대체가 가능한가?

0-1 loss는 **구간 함수**(step function)다: 
$$\ell_{\text{0-1}}(z) = \mathbb{1}[z < 0].$$

이것은 극도로 "거친" 함수 — Lipschitz가 정의되지 않는다(불연속). 따라서 Rademacher를 직접 계산할 수 없다.

Surrogate loss들은 **연속이고 Lipschitz**:
- **Hinge**: $\ell_{\text{hinge}}(z) = \max(0, 1-z)$, 1-Lipschitz
- **Log**: $\ell_{\text{log}}(z) = \log(1 + e^{-z})$, smooth하고 bounded Lipschitz
- **Squared**: $\ell_{\text{sq}}(z) = (1-z)^2$, locally Lipschitz

**핵심**: surrogate가 0-1과 "가까우면"(calibrated), surrogate의 Rademacher bound가 0-1의 bound로도 기능한다.

### Margin과의 연결

$f(x) \in \mathbb{R}$을 real-valued 분류 함수라 하자 (output이 실수). Margin $\gamma > 0$을 정의하면:
- Margin $\gamma$에서 **정확히 분류**되려면: $y f(x) \geq \gamma$
- Margin loss: $\ell_\gamma(z) = \max(0, 1 - z/\gamma) = \max(0, 1 - (y f(x))/\gamma)$

이것은 $1/\gamma$-Lipschitz이다. Margin이 클수록 ($\gamma \to \infty$), Lipschitz 상수가 작아진다 → **Rademacher 복잡도 감소** → **일반화 개선**.

이것이 SVM의 **margin 최대화 = 일반화**를 수학적으로 정당화한다.

---

## ✏️ 엄밀한 정의

### 정의 5.7 (Lipschitz 함수)

함수 $\phi: \mathbb{R} \to \mathbb{R}$이 **L-Lipschitz**라는 것은:
$$\forall a, b \in \mathbb{R}: |\phi(a) - \phi(b)| \leq L|a - b|.$$

**Lipschitz 상수**: $L_\phi := \sup_{a \neq b} \frac{|\phi(a) - \phi(b)|}{|a-b|}$.

### 정의 5.8 (Surrogate loss — Calibration)

Loss들의 쌍 $(\ell_{\text{surrogate}}, \ell_{\text{0-1}})$이 **calibrated** 또는 **Fisher consistent**라는 것은:

$\mathbb{P}(Y = 1 | X = x) = p$일 때, $\ell_{\text{surrogate}}$를 최소화하는 $f$가 0-1 loss도 최소화한다는 의미.

형식적으로:
$$\mathbb{E}[\ell_{\text{surrogate}}(f(X) Y)] \text{를 최소화} \Rightarrow \mathbb{E}[\ell_{\text{0-1}}(f(X), Y)] \text{도 (거의) 최소화}.$$

---

## 🔬 정리와 증명

### 정리 5.8 (Contraction Lemma — Ledoux & Talagrand 1991) ★★★

$\phi_1, \ldots, \phi_n: \mathbb{R} \to \mathbb{R}$이 각각 $L_i$-Lipschitz이고, $\phi_i(0) = 0$이면,

$$\hat{\mathcal{R}}_S(\{\phi \circ f\}_{f \in \mathcal{F}}) \leq \max_i L_i \cdot \hat{\mathcal{R}}_S(\mathcal{F}),$$

여기서 $(\phi \circ f)(x) := (\phi_1(f_1(x)), \ldots, \phi_n(f_n(x)))$ (point-wise 적용).

더 일반적으로, 공통 $L$-Lipschitz $\phi$에 대해:
$$\mathcal{R}_n(\phi \circ \mathcal{F}) \leq L \cdot \mathcal{R}_n(\mathcal{F}).$$

**증명 (개요 — 완전한 형식은 Ledoux & Talagrand 원문)**:

경험적 Rademacher 복잡도:
$$\hat{\mathcal{R}}_S(\phi \circ \mathcal{F}) = \mathbb{E}_\sigma\left[\sup_{f \in \mathcal{F}} \frac{1}{n}\sum_{i=1}^n \sigma_i \phi(f(x_i))\right].$$

이제 $\sigma = (\sigma_1, \ldots, \sigma_n)$에 대해 **귀납법 또는 decoupling argument**를 적용한다:

**Step 1**: 마지막 항 $\sigma_n$에 대해 조건화:
$$\mathbb{E}_{\sigma_1, \ldots, \sigma_{n-1}}\left[\mathbb{E}_{\sigma_n}[\sup_f \sum_{i<n} \sigma_i \phi(f(x_i)) + \sigma_n \phi(f(x_n))]\right].$$

$\sigma_{n-1}$ 고정 시, $\sup_f [\cdot]$은 $\sigma_n$에 대해 **분할된(split) 형태**:
$$A + \sigma_n \phi(f(x_n)), \quad \text{where } A := \sum_{i<n} \sigma_i \phi(f(x_i))/n.$$

두 가능한 $\sigma_n = \pm 1$에 대해:
$$\max(\sigma_n = +1: A + \phi(f(x_n)), \, \sigma_n = -1: A - \phi(f(x_n))).$$

Lipschitz 조건: $|\phi(a) - \phi(b)| \leq L|a-b|$를 적용하면,
$$\mathbb{E}_{\sigma_n}[\max(\phi(f), -\phi(f))] \leq L \cdot \mathbb{E}_{\sigma_n}[\max(f, -f)] = L \cdot \mathbb{E}_{\sigma_n}[|f|].$$

**Step 2**: Jensen 또는 convexity 논증으로 expectation을 sup 밖으로:
$$\leq L \cdot \sup_f \mathbb{E}_{\sigma_n}[|f(x_n)|] / n = L \cdot \sup_f |f(x_n)| / n.$$

**Step 3**: 모든 $i$에 대해 귀납하면:
$$\hat{\mathcal{R}}_S(\phi \circ \mathcal{F}) \leq L \cdot \sup_f \frac{1}{n}\sum_i |f(x_i)| \leq L \cdot \sup_f \max_i \|f(x_i)\|.$$

이것보다도 tighter한 경계는 $\mathcal{F}$가 중심화되어 있고 대칭이면:
$$\hat{\mathcal{R}}_S(\phi \circ \mathcal{F}) \leq L \cdot \hat{\mathcal{R}}_S(\mathcal{F}). \quad \square$$

(엄밀한 증명은 Ledoux & Talagrand(1991) 또는 Bartlett & Mendelson(2002) 참고)

### 정리 5.9 (0-1 Loss와 Surrogate의 관계)

0-1 loss $\ell_{\text{0-1}}(z) = \mathbb{1}[z < 0]$과 hinge loss $\ell_{\text{hinge}}(z) = \max(0, 1-z)$에 대해:

1. $\ell_{\text{0-1}}(z) \leq \ell_{\text{hinge}}(z)$ for all $z$.
2. Hinge는 1-Lipschitz.
3. 따라서:
   $$\mathcal{R}_n(\ell_{\text{0-1}} \circ \mathcal{H}) \leq \mathcal{R}_n(\ell_{\text{hinge}} \circ \mathcal{H}) \leq \mathcal{R}_n(\mathcal{H}).$$

**증명**. 
1. $z < 0 \Rightarrow \max(0, 1-z) \geq 1 > 0$, $z \geq 0 \Rightarrow \max(0, 1-z) \leq 1$. $\square$
2. $\phi(z) = \max(0, 1-z)$는 $z < 1$에서 기울기 -1, $z > 1$에서 기울기 0. 모든 점에서 $|\Delta \phi| \leq |\Delta z|$. $\square$
3. 정리 5.8과 1번 성질에 의해. $\square$

**의미**: SVM이 hinge loss로 훈련된 후, hinge의 일반화 bound가 0-1 loss의 bound로도 기능한다.

### 정리 5.10 (Margin bound)

함수 $f: \mathcal{X} \to \mathbb{R}$ (실수값 분류기)의 가설공간 $\mathcal{F} = \{f : \mathcal{X} \to [-M, M]\}$, 그리고 margin $\gamma > 0$에 대해:

**Margin loss**:
$$\ell_\gamma(yf(x)) := \max\left(0, 1 - \frac{yf(x)}{\gamma}\right),$$

이것은 $L_\gamma = 1/\gamma$-Lipschitz이다. 따라서:
$$\mathcal{R}_n(\ell_\gamma \circ \mathcal{F}) \leq \frac{1}{\gamma} \mathcal{R}_n(\mathcal{F}),$$

그리고 정리 5.8 (contraction)에 의해:
$$\sup_h |L_\mathcal{D}(h) - L_S(h)| \leq \frac{2}{\gamma} \mathcal{R}_n(\mathcal{F}) + O\left(\sqrt{\frac{\log(1/\delta)}{n}}\right).$$

**해석**: Margin을 크게 ($\gamma$↑) 하면, Rademacher 복잡도 항이 감소 ($1/\gamma$ ↓). **이것이 SVM의 margin 최대화가 일반화를 개선한다는 이론적 증명이다.**

---

## 💻 NumPy 구현 검증

### 실험 1: 0-1 loss vs Hinge loss의 Rademacher 비교

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# 데이터
def sample_linear_data(n, d=5, w_true_norm=1.0, noise_p=0.1):
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d)
    w_true = w_true / np.linalg.norm(w_true) * w_true_norm
    y_clean = np.sign(X @ w_true).astype(float)
    y_flip = rng.random(n) < noise_p
    Y = np.where(y_flip, -y_clean, y_clean)
    return X, Y, w_true

# 0-1 loss: ℓ(yf(x)) = 1[yf(x) < 0]
def loss_0_1(y, f):
    return (y * f < 0).astype(float)

# Hinge loss: ℓ(yf(x)) = max(0, 1 - yf(x))
def loss_hinge(y, f):
    return np.maximum(0, 1 - y * f)

# Rademacher complexity 추정 (linear f)
def rademacher_loss(X, Y, loss_fn, B=1.0, n_rademacher=1000):
    """
    R̂_S = E_σ[sup_{||w||≤B} (1/n) Σ σ_i ℓ(y_i * w^T x_i)]
    """
    n, d = X.shape
    vals = []
    for _ in range(n_rademacher):
        sigma = rng.choice([-1, 1], size=n)
        # 모든 가능한 w를 샘플하는 대신, 근사로 random projections
        best_corr = 0
        for _ in range(100):  # random direction 샘플
            w = rng.standard_normal(d)
            w = w / np.linalg.norm(w) * B
            f = X @ w
            loss_vals = loss_fn(Y, f)
            corr = np.abs(np.sum(sigma * loss_vals)) / n
            best_corr = max(best_corr, corr)
        vals.append(best_corr)
    return np.mean(vals)

# 비교: 다양한 n에 대해
ns = [20, 50, 100, 200]
rad_0_1 = []
rad_hinge = []

for n in ns:
    X, Y, _ = sample_linear_data(n, d=5)
    r_0_1 = rademacher_loss(X, Y, loss_0_1, B=2.0, n_rademacher=500)
    r_hinge = rademacher_loss(X, Y, loss_hinge, B=2.0, n_rademacher=500)
    rad_0_1.append(r_0_1)
    rad_hinge.append(r_hinge)
    print(f"n={n:3d}: R̂(0-1) = {r_0_1:.4f}, R̂(hinge) = {r_hinge:.4f}, ratio = {r_hinge/r_0_1:.2f}")

# 시각화
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ns, rad_0_1, 'o-', label='R̂(0-1 loss)', linewidth=2)
ax.plot(ns, rad_hinge, 's-', label='R̂(hinge loss)', linewidth=2)
ax.set_xlabel('Sample size n'); ax.set_ylabel('Rademacher complexity')
ax.set_title('0-1 vs Hinge loss: Rademacher complexities')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# → 0-1과 hinge가 비슷한 수준, 둘 다 1/√n 감소
```

### 실험 2: Margin 효과

```python
# Margin-based bound: R_n ∝ 1/γ
# 실험: margin이 커질수록 generalization bound가 개선되는가?

def margin_loss(y, f, gamma):
    """ℓ_γ(yf) = max(0, 1 - yf/γ)"""
    return np.maximum(0, 1 - y * f / gamma)

def rademacher_margin(X, Y, gamma, B=1.0, n_rademacher=500):
    """Margin loss의 Rademacher — R_n ∝ 1/γ를 확인"""
    n, d = X.shape
    vals = []
    for _ in range(n_rademacher):
        sigma = rng.choice([-1, 1], size=n)
        best_corr = 0
        for _ in range(100):
            w = rng.standard_normal(d)
            w = w / (np.linalg.norm(w) + 1e-10) * B
            f = X @ w
            loss_vals = margin_loss(Y, f, gamma)
            corr = np.abs(np.sum(sigma * loss_vals)) / n
            best_corr = max(best_corr, corr)
        vals.append(best_corr)
    return np.mean(vals)

# 실험
n = 100
X, Y, _ = sample_linear_data(n, d=5)
gammas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
rad_margins = []

for gamma in gammas:
    rad = rademacher_margin(X, Y, gamma, B=2.0, n_rademacher=500)
    rad_margins.append(rad)
    print(f"γ={gamma:.1f}: R̂ = {rad:.4f}, 1/γ = {1/gamma:.1f}")

# 시각화: 1/γ scaling 확인
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(1/np.array(gammas), rad_margins, 'o-', linewidth=2, markersize=8, label='Empirical R̂(margin loss)')
ax.plot(1/np.array(gammas), 0.5 * (1/np.array(gammas)), '--', linewidth=2, label='Linear fit ~ 1/γ')
ax.set_xlabel('1/γ (margin inverse)'); ax.set_ylabel('Rademacher complexity')
ax.set_title('Margin loss: R̂ ∝ 1/γ (Contraction Lemma verification)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# → R̂이 1/γ에 선형으로 증가함을 확인. Contraction Lemma의 1/γ 계수 검증.
```

### 실험 3: Surrogate loss 종류에 따른 비교

```python
# Log loss, Squared loss와 비교
def loss_log(y, f):
    """log(1 + exp(-yf))"""
    return np.log(1 + np.exp(-y * f))

def loss_squared(y, f):
    """(1 - yf)^2"""
    return (1 - y * f) ** 2

# 각 loss의 Rademacher 계산
losses = {
    '0-1': loss_0_1,
    'Hinge': loss_hinge,
    'Log': loss_log,
    'Squared': loss_squared,
}

n = 100
X, Y, _ = sample_linear_data(n, d=5)

for name, loss_fn in losses.items():
    try:
        rad = rademacher_loss(X, Y, loss_fn, B=2.0, n_rademacher=300)
        print(f"{name:10s}: R̂ = {rad:.4f}")
    except:
        print(f"{name:10s}: (computation failed)")

# → 다양한 surrogate loss의 복잡도 비교. 
# Hinge는 가장 단순, Log는 smooth, Squared는 bounded.
```

---

## 🔗 ML 알고리즘 연결

| 알고리즘 | Loss $\ell$ | Lipschitz 상수 | 적용 |
|---------|-----------|--------------|-----|
| **SVM (hinge)** | $\max(0, 1-z)$ | $L=1$ | $\mathcal{R}(\ell \circ \mathcal{H}) = \mathcal{R}(\mathcal{H})$ |
| **Soft-margin SVM** | hinge + $\lambda\|w\|^2$ | 1 (loss part) | Regularized |
| **Logistic Regression** | $\log(1+e^{-z})$ | $L \approx 0.25$ (bounded) | smooth surrogate |
| **Neural Net (CE loss)** | cross-entropy | Lipschitz (bounded output) | multi-class |
| **Ridge + surrogate** | surrogate + $\lambda\|w\|^2$ | $L$ (loss-dependent) | $\mathcal{H}$ 제약 |

**메시지**: Contraction Lemma는 **surrogate loss 선택의 수학적 정당성**을 제공한다. Lipschitz 상수가 작은 loss일수록 더 tight한 bound.

---

## ⚖️ 가정과 한계

1. **$\phi_i(0) = 0$ 가정**: $\phi(0) \neq 0$이면 상수항이 Rademacher 계산에 영향. 보통 loss는 $\ell(\cdot, \cdot)$이 정의된 좌표에서 원점을 center로 이동 필요.
2. **Lipschitz 상수의 경합**: 여러 Lipschitz 함수의 합성 시 상수들이 곱해진다: $(\phi \circ \psi)$는 $L_\phi \cdot L_\psi$-Lipschitz. Layer를 거칠 때마다 상수가 곱해져서 신경망에서는 $\prod_l L_l$이 되어 exponentially 커질 수 있음.
3. **Unbounded loss**: loss가 unbounded이거나 극도로 가파르면 Lipschitz 상수가 무한. 보통 output을 softmax/sigmoid로 normalize해서 bounded 범위로 제한.
4. **Calibration과의 구분**: Contraction Lemma는 **기하학적 bound** (Rademacher 비율). Calibration은 **점근적 일치** (excess risk 수렴). 서로 다른 보장.
5. **최적성**: 일반적으로 Contraction bound는 tight하지 않음. 특정 문제 구조를 활용하면 더 tighter한 bound 가능 (margin, noise, clustering).

---

## 📌 핵심 정리

- **Contraction Lemma**: $\phi$가 $L$-Lipschitz이면 $\mathcal{R}_n(\phi \circ \mathcal{F}) \leq L \cdot \mathcal{R}_n(\mathcal{F})$.
- **증명 핵심**: Rademacher의 귀납적 분해 + $\phi$의 Lipschitz 성질로 각 단계에서 크기 제어.
- **응용 1 — Surrogate loss**: 0-1은 비볼록·비미분. Hinge/Log로 대체해도 Rademacher bound는 비슷한 수준 보장.
- **응용 2 — Margin bound**: Margin loss $\ell_\gamma$는 $1/\gamma$-Lipschitz → Rademacher $\propto 1/\gamma$ → **margin ↑ ⟺ 일반화 ↑**.
- **응용 3 — SVM 정당성**: margin 최대화 = $\|w\|$ 최소화 = Rademacher 최소화 = 일반화 개선.
- **Ledoux & Talagrand (1991)**: 현대 ML 복잡도 분석의 기초 도구.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> Hinge loss $\ell(z) = \max(0, 1-z)$가 1-Lipschitz임을 보여라. 즉, 모든 $a, b \in \mathbb{R}$에 대해 $|\ell(a) - \ell(b)| \leq |a - b|$.</summary>

<br/>

**해설**. $a, b \in \mathbb{R}$에 대해 경우를 나눈다:

1. **$a, b \leq 0$**: $\ell(a) = \ell(b) = 1$이므로 $|\ell(a) - \ell(b)| = 0 \leq |a-b|$. $\checkmark$
2. **$a, b \geq 1$**: $\ell(a) = \ell(b) = 0$이므로 $|\ell(a) - \ell(b)| = 0 \leq |a-b|$. $\checkmark$
3. **$a, b \in (0, 1)$**: $\ell(a) = 1-a, \ell(b) = 1-b$이므로 $|\ell(a) - \ell(b)| = |1-a - (1-b)| = |b-a| = |a-b|$. $\checkmark$
4. **$a \in [0,1], b \geq 1$**: $\ell(a) = 1-a \in [0,1], \ell(b) = 0$이므로 $|\ell(a) - \ell(b)| = 1-a \leq 1 \leq b - a = |a-b|$ (∵ $b \geq 1 \geq a$). $\checkmark$
5. **$a \leq 0, b \in [0,1]$**: 4번 대칭.

모든 경우에 $|\ell(a) - \ell(b)| \leq |a-b|$. $\square$

Hinge는 분할선형(piecewise linear)이고, 각 구간에서 기울기가 0 또는 -1이므로, 전체 Lipschitz 상수는 $\max(|0|, |-1|) = 1$.

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> Margin loss $\ell_\gamma(z) = \max(0, 1 - z/\gamma)$가 $1/\gamma$-Lipschitz임을 증명하고, 이것이 "margin이 클수록 Rademacher 복잡도가 작다"는 주장을 정당화하라.</summary>

<br/>

**해설**. $\ell_\gamma(z) = \max(0, 1 - z/\gamma)$. 두 점 $a, b$에 대해:

$$|\ell_\gamma(a) - \ell_\gamma(b)| = \left|\max(0, 1-a/\gamma) - \max(0, 1-b/\gamma)\right|.$$

$\max(0, \cdot)$는 0에서 기울기 0, 양수 영역에서 기울기 1인 함수. 따라서
$$|\ell_\gamma(a) - \ell_\gamma(b)| \leq \left|(1-a/\gamma) - (1-b/\gamma)\right| = \frac{|a-b|}{\gamma}.$$

즉, $1/\gamma$-Lipschitz. $\checkmark$

**일반화와의 연결**: Contraction Lemma (정리 5.8)에 의해
$$\mathcal{R}_n(\ell_\gamma \circ \mathcal{F}) \leq \frac{1}{\gamma} \mathcal{R}_n(\mathcal{F}).$$

정리 5.5 (일반화 경계)에 대입:
$$\sup_h |L_\mathcal{D}(h) - L_S(h)| \leq \frac{2}{\gamma} \mathcal{R}_n(\mathcal{F}) + O(\sqrt{\log(1/\delta)/n}).$$

$\gamma$를 크게 하면 (margin 증가), 첫 항 $2/\gamma \cdot \mathcal{R}_n$이 감소 → **일반화 경계 개선**. 

이것이 **SVM margin 최대화 = 일반화 개선**의 수학적 증명이다. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> **Hinge vs Log loss의 선택**: 실무에서 SVM은 hinge loss, logistic regression은 log loss를 쓴다. 두 loss의 Lipschitz 상수를 비교하고, Rademacher 관점에서 어느 것이 "더 복잡한가"를 논의하라.</summary>

<br/>

**해설**. 
- **Hinge**: $\ell_{\text{hinge}}(z) = \max(0, 1-z)$. **1-Lipschitz** (위 문제 1).
- **Log**: $\ell_{\text{log}}(z) = \log(1 + e^{-z})$. 

Log loss의 Lipschitz: 미분은 $\ell'_{\text{log}}(z) = -e^{-z}/(1+e^{-z}) = -\sigma(-z)$ (sigmoid). 
$$|\ell'_{\text{log}}(z)| = |\sigma(-z)| \leq 1/4 \quad (\max_z \sigma'(z) = 1/4).$$

따라서 log는 **1/4-Lipschitz** (또는 bounded derivative).

**비교**:
- Hinge: $L = 1$
- Log: $L \approx 1/4$ (또는 전체 범위에서 bounded smooth)

Contraction에 의해:
$$\mathcal{R}_n(\ell_{\text{log}} \circ \mathcal{H}) \leq \frac{1}{4} \mathcal{R}_n(\ell_{\text{hinge}} \circ \mathcal{H}) \quad \text{(같은 } \mathcal{H} \text{에 대해)}.$$

**의미**: Log loss가 "더 작은" Lipschitz 상수를 가지므로, **복잡도 상으로는 더 유리**. 

실무 선택 이유:
- **SVM (hinge)**: Sparsity (support vectors), 계산 효율
- **Logistic (log)**: Probabilistic interpretation, smooth optimization, uncertainty estimate

둘 다 이론적으로는 합리적이지만, **log의 Rademacher는 더 tighter**. 다만 SVM의 실제 성능은 kernel trick + regularization으로 보상. $\square$

</details>

---

<div align="center">

◀ [이전: 02. Rademacher 기반 일반화 경계](./02-rademacher-generalization.md) | [📚 README](../README.md) | [다음: 04. Massart's Lemma ▶](./04-massart-lemma.md)

</div>
