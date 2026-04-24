# 04. VC, Rademacher, Stability — 세 관점 비교

## 🎯 핵심 질문

이 문서는 **레포 전체의 종합**이다. 세 가지 근본적인 시각:

- **VC 이론** (Ch4): 가설공간의 **조합적 capacity** — "모델이 얼마나 큰가?"
- **Rademacher 복잡도** (Ch5): 데이터 의존적 **노이즈 fitting 능력** — "이 데이터와 모델의 조합이 얼마나 위험한가?"
- **Algorithmic Stability** (Ch6): 알고리즘의 **표본 민감도** — "샘플 하나 바꾸면 출력이 얼마나 달라지는가?"

이 셋을 **언제, 어느 상황에서 어느 것을 써야 하는가**? 장점·약점·적용 범위를 체계적으로 정리한다.

---

## 🔍 왜 "세 관점"인가

일반화의 근거를 제공하는 방법이 **여럿**이다:

1. **VC** (Vapnik & Chervonenkis, 1968-1971): "$\mathcal{H}$가 크면 일반화 힘들다"
2. **Rademacher** (Bartlett & Mendelson, 2002): "이 데이터와의 조합이 구체적으로 얼마나 위험한가"
3. **Stability** (Bousquet & Elisseeff, 2002): "알고리즘 자체가 샘플 변화에 강건한가"

각각은 **다른 질문**을 답한다. 현대 ML에서는 이 셋을 **상황에 맞게 선택**해야 한다. DL의 "vacuous VC bound" 문제가 나타난 이후, 더욱이 그렇다.

---

## 📐 수학적 선행 조건

- Ch1~Ch6 전체 (모든 이론 기초)
- [Probability Theory](https://github.com/iq-ai-lab/probability-theory-deep-dive): 집중부등식, MGF
- [Mathematical Statistics](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): Uniform convergence, empirical processes

---

## 📖 비교 표: 강점, 약점, 적용 범위

| 관점 | VC 차원 | Rademacher | Stability |
|------|--------|-----------|----------|
| **측정 대상** | 가설공간 크기 | 함수족+데이터 조합 | 알고리즘 로컬성 |
| **대표 경계** | $\sqrt{d/n}$ | $2\mathcal{R}_n(\mathcal{F})$ | $\beta + O(1/\sqrt{n})$ |
| **가정** | 구조 알려짐 | 데이터 분포 (실험적) | 알고리즘 정의 필요 |
| **분포 자유** | 예 | 아니오 | 예 |
| **Universal** | 예 (모든 분포 최악) | 아니오 | 부분 (stable class) |
| **계산 복잡** | 중간 | 높음 (MC 근사) | 높음 (증명) |
| **장점** | 단순, 직관, 분포 무관 | 데이터 적응, tight | 알고리즘 독립, SGD에 강함 |
| **약점** | vacuous (DL), 느슨함 | 계산 어려움 | Uniform stability 자주 불가능 |
| **적용 범위** | 고전 (선형, 작은 NN) | Kernel, 마진 | SGD, boosting, 정규화 |

---

## ✏️ 엄밀한 비교

### 정의 7.10 (세 관점의 형식)

**VC 이론**:
$$\mathbb{P}\left(\sup_{h \in \mathcal{H}} |L_\mathcal{D}(h) - L_S(h)| \geq \epsilon\right) \leq 4\Pi_\mathcal{H}(2n) \exp(-n\epsilon^2/8).$$

여기서 $\Pi_\mathcal{H}(2n) \leq \sum_{i=0}^d \binom{2n}{i} \leq (2en/d)^d$이고 $d = \text{VC}(\mathcal{H})$.

**Rademacher 복잡도**:
$$\sup_h |L_\mathcal{D}(h) - L_S(h)| \leq 2\mathcal{R}_n(\mathcal{F}) + 3\sqrt{\frac{\log(2/\delta)}{2n}} \text{ w.p. } 1-\delta,$$

여기서 $\mathcal{R}_n(\mathcal{F}) = \mathbb{E}_\sigma[\sup_f (1/n)\sum \sigma_i f(x_i)]$.

**Stability**:
$$|L_\mathcal{D}(A(S)) - L_\mathcal{D}(A(S^{(i)}))| \leq \beta \text{ for all } S, i,$$

이면 $$\mathbb{E}[L_\mathcal{D}(A(S))] - L_S(A(S)) \leq \beta + O(\sqrt{\log(1/\delta)/n}).$$

---

## 🔬 정리와 해석

### 정리 7.12 (VC와 Rademacher의 관계)

크기 $n$의 유한 샘플에 대해:

$$\mathcal{R}_n(\mathcal{F}) \leq C \cdot \frac{\sqrt{d \log n}}{n},$$

여기서 $d = \text{VC}(\mathcal{F})$. 따라서 **VC bound**와 **Rademacher bound**는 같은 $O(\sqrt{d/n})$ 스케일이지만:
- VC는 **분포 자유**
- Rademacher는 **데이터에 적응** (tight)

**결론**: 데이터가 구체적으로 주어지면 Rademacher가 VC보다 **tighter한 경계를 제공**할 가능성이 높다.

### 정리 7.13 (Stability의 장점)

**$\mathcal{H}$ 무지:** Ridge regression (Ch6-03)의 경우 $\text{VC}(\mathcal{H})$는 무한일 수 있지만, $\beta = O(1/(\lambda n))$ stable이므로:

$$\mathbb{E}[L_\mathcal{D}] - L_S \leq O(1/\lambda n),$$

이는 **VC bound 불필요**하고 **$\lambda$ 선택으로 안정성 통제 가능**.

**결론**: 알고리즘의 정규화 강도 $\lambda$가 capacity를 implicit하게 제어하는 경우, stability 관점이 가장 직관적.

### 정리 7.14 (SGD의 현대적 해석)

Hardt et al. (2016): 유한 step SGD는 stable with $\beta \leq O(\eta T)$. 따라서:

$$\mathbb{E}[L_\mathcal{D}] - L_S \leq O(\eta T) + O(\sqrt{\log(1/\delta)/n}).$$

**해석**:
- VC 관점: NN의 VC 차원 $= O(W^2 \log W)$ (거대) → vacuous bound
- Rademacher 관점: NN의 Rademacher $= O(\prod \|W_l\|/\sqrt{n})$ (spectral norm) → 가능하지만 복잡
- Stability 관점: **Step 수 $T = \min_k(\text{test loss 증가}^*)$ early stopping** → 가장 직관적, implicit regularization

현대 DL은 **stability 프레임에서 가장 잘 설명**.

---

## 🔗 ML 알고리즘 연결 — 의사결정 트리

어느 관점을 쓸 것인가? 의사결정 플로우차트:

```
┌─────────────────────────────┐
│  당신이 하는 것이 무엇인가?    │
└────────────┬────────────────┘
             │
     ┌───────┴───────┐
     ▼               ▼
  고정된 모델    알고리즘 설계
  (ERM 분석)     (SGD, 정규화)
     │               │
     ▼               ▼
┌──────────────┐ ┌──────────────┐
│ 가설공간      │ │ 데이터 주어짐│
│ 구조 명확?   │ │ 확률?        │
└──┬────────┬──┘ └──┬────────┬──┘
   │ yes   │ no    │ 예   │ 아니
   │       │       │       │
   ▼       ▼       ▼       ▼
┌────┐  ┌──────┐┌──────┐┌────┐
│VC  │  │상황  ││Radem.││Stab│
│또는│  │별    ││또는  ││또는│
│Rad.│  │      ││Margin││CV  │
└─┬──┘  │      │└──────┘└────┘
  │ (가능) │
  │ ↓     │
  │ ┌─────┘
  │ ▼
┌──────────┐
│ Contraction
│ lemma로
│ margin/
│ norm 기반
│ Rademacher
└──────────┘
```

**구체적 상황별**:

1. **"선형 분류기의 일반화 bound를 증명하고 싶다"**
   - → VC (선형 VC = $d+1$, 명확) 또는 Rademacher (margin 기반)

2. **"내 신경망이 일반화되는 이유를 설명해야 한다"**
   - → Stability (early stopping, SGD 이론) 또는 norm-based Rademacher (spectral norm)
   - **피할 것**: 고전 VC bound (vacuous)

3. **"어느 모델을 선택해야 하는가?"**
   - → CV (실전 표준) 또는 AIC/BIC (빠른 근사)
   - **부차적**: SRM의 VC bound penalty (일반적으로 보수적)

4. **"Regularization 강도 $\lambda$를 어떻게 정하는가?"**
   - → Stability (ridge: $\beta = O(1/(\lambda n))$로 직접 제어)
   - **대안**: CV, AIC/BIC

5. **"Boosting/ensemble의 일반화?"**
   - → Margin 기반 Rademacher (Schapire 1998)

---

## 💻 구현 비교: 동일 데이터에서 세 bound 계산

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.datasets import make_classification

rng = np.random.default_rng(42)

# 합성 데이터: n=200, d=10
n, d = 200, 10
X, y = make_classification(n_samples=n, n_features=d, n_informative=5,
                           n_redundant=0, random_state=42)
y = 2*y - 1  # {-1, +1}

# 선형 분류기 (hinge loss로 근사)
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X, y)

# 1. VC bound (간단한 하한)
# 선형 분류기: VC = d+1 = 11
vc_d = d + 1
# Sauer-Shelah: Π(2n) ≤ (2en/d)^d
pi_growth = (2 * np.e * n / vc_d) ** vc_d
epsilon_vc = np.sqrt((vc_d + np.log(1/0.05)) / (2*n))
vc_bound = 4 * np.sqrt(vc_d / n) + 3*np.sqrt(np.log(2/0.05) / (2*n))

# 2. Rademacher 복잡도 (몬테카를로 추정)
def rademacher_mc(X, n_trials=1000):
    """R̂_S({w·x : ‖w‖ ≤ 1}) Monte Carlo"""
    n = len(X)
    vals = []
    for _ in range(n_trials):
        sigma = rng.choice([-1, 1], size=n)
        X_sigma = X.T @ sigma  # shape (d,)
        vals.append(np.linalg.norm(X_sigma) / n)
    return np.mean(vals)

w = model.coef_[0]
B = np.linalg.norm(w)  # ‖w‖
R_empirical = rademacher_mc(X)
rademacher_bound = 2 * R_empirical * B + 3*np.sqrt(np.log(2/0.05)/(2*n))

# 3. Stability bound (Ridge로 근사)
# Ridge: β ≈ 1/(λn)
# 강정규화를 가정 (λ = 0.1)
lambda_reg = 0.1
beta_ridge = 1 / (lambda_reg * n)
stability_bound = beta_ridge + 3*np.sqrt(np.log(2/0.05) / (2*n))

# 4. 실제 test error (cross-validation)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
actual_error = 1 - cv_scores.mean()

print("=" * 60)
print(f"Sample size: n={n}, feature dim: d={d}")
print("=" * 60)
print(f"\n1. VC Bound (linear classifier, VC={vc_d}):")
print(f"   ε_VC ≈ {vc_bound:.4f}")
print(f"   Assessment: {'VACUOUS' if vc_bound > 0.5 else 'informative'}")

print(f"\n2. Rademacher Bound (‖w‖={B:.4f}, R̂_n={R_empirical:.4f}):")
print(f"   ε_Rad ≈ {rademacher_bound:.4f}")
print(f"   Assessment: {'reasonable' if rademacher_bound < 0.3 else 'loose'}")

print(f"\n3. Stability Bound (Ridge λ={lambda_reg}, β={beta_ridge:.4f}):")
print(f"   ε_Stab ≈ {stability_bound:.4f}")
print(f"   Assessment: {'tight' if stability_bound < 0.2 else 'reasonable'}")

print(f"\n4. Actual 5-fold CV error:")
print(f"   Error ≈ {actual_error:.4f} ± {cv_scores.std():.4f}")

print("\n" + "=" * 60)
print("COMPARISON:")
print(f"VC bound:        {vc_bound:.4f}")
print(f"Rademacher:      {rademacher_bound:.4f}")
print(f"Stability:       {stability_bound:.4f}")
print(f"Empirical (CV):  {actual_error:.4f}")
print("=" * 60)
# 일반적으로 Stability < Rademacher < VC 순서의 tightness
```

---

## ⚖️ 가정과 한계

세 관점의 핵심 **가정**과 그것이 깨지는 현실 상황을 체계적으로 정리.

### VC 관점의 가정과 한계
1. **iid 가정**: Ch1-05에서 다룬 시계열·분포 이동 하에선 VC 경계가 그대로 성립하지 않음.
2. **0-1 loss**: 기본 VC 이론은 binary classification 중심. Regression·multi-class·구조화 예측에는 **pseudo-dimension**·**fat-shattering** 같은 일반화 필요.
3. **분포 자유성의 대가**: 최악의 경우 분석이라 **실전 분포**에서 과도하게 보수적 — DL에서 vacuous.
4. **유한 VC 필수**: VC $= \infty$인 $\mathcal{H}$(k-NN, 어떤 RKHS)에는 적용 불가.

### Rademacher 관점의 가정과 한계
1. **$\mathcal{R}_n$ 계산 비용**: Monte Carlo로 근사해야 하고, sup-over-$\mathcal{F}$가 닫힌 형태 아닐 때 어려움.
2. **데이터 의존성의 양면**: tight하지만 **재현**이 어렵고, 새 데이터에서 다시 추정해야 함.
3. **NN 경계의 현실성**: Bartlett-Mendelson의 $\prod\|W_l\|$ 기반 bound도 실전 DL에서는 여전히 loose — norm이 매우 큼.
4. **Contraction lemma의 Lipschitz 요구**: non-Lipschitz loss에는 직접 적용 불가.

### Stability 관점의 가정과 한계
1. **Uniform stability의 가혹함**: $\sup_z$에서의 차이가 $\beta$로 유계여야. 많은 현실 알고리즘에서 증명이 어려움.
2. **Algorithm-specific**: 각 알고리즘마다 stability 분석을 새로 해야 함 — 일반화 이론으로서의 보편성 부족.
3. **Non-convex의 한계**: Hardt et al.의 SGD stability는 **convex** 가정이 기본. Non-convex NN에서는 부분적·heuristic.
4. **Stability가 optimal solution에만 적용**: early stopping의 경우 수렴 이전의 중간 가설이 주된 분석 대상인데, 이것이 β와 직접 연결되지 않을 수 있음.

### 세 관점 모두의 공통 한계
- **Asymptotic 경향**: 대부분 $n \to \infty$에서 유의미. 작은 $n$에선 constants가 지배.
- **실전 tightness**: 이론 bound가 실제 test error보다 **수십~수백 배** 큰 경우가 흔함 — 경향성 예측에는 유용하지만 수치적 guarantees로는 제한적.
- **현대 DL의 gap**: Zhang et al. (2017)이 보인 "random label을 fit"하는 NN은 **고전 SLT 세 관점 모두**에 도전적. Layer 2 Generalization Theory(NTK, PAC-Bayes, Double Descent)가 필요.

---

### 역사적 계보

SLT가 어떻게 발전해왔는가:

```
1968-1971: VC dimension
  ↓ (Vapnik, Chervonenkis)
  Shattering, growth function, VC bound 도입

1984: PAC learning
  ↓ (Valiant)
  "Probably Approximately Correct" 정의화

1989: Fundamental Theorem of SLT
  ↓ (Blumer et al.)
  PAC ↔ Uniform Convergence ↔ VC 동치성

1996: Devroye, Györfi, Lugosi
  ↓ Pattern Recognition 고전 (VC 심화)

1999: Vapnik "Nature of Statistical Learning"
  ↓ (Vapnik 종합)
  SRM, structural risk minimization 상세

2002: Rademacher Complexity (Bartlett & Mendelson)
  ↓ 혁신: data-dependent, tighter bound
  + Contraction lemma (Ledoux-Talagrand)

2002: Uniform Stability (Bousquet & Elisseeff)
  ↓ 새 관점: algorithm-centric, regularization과의 연결

2014: Understanding Machine Learning (Shalev-Shwartz, Ben-David)
  ↓ 현대 SLT 통합 교과서

2016: SGD Stability (Hardt, Recht, Singer)
  ↓ Early stopping = implicit regularization
  + Modern DL 해석의 시작

2017: Rethinking Generalization (Zhang et al.)
  ↓ DL의 "vacuous VC bound" 문제 실증적 증명
  + 현대 이론의 필요성 부각

2018+: Modern DL theory (NTK, Double Descent, etc.)
  ↓ Layer 2 "Generalization Theory Deep Dive" 레포로
  Neural Tangent Kernel, overparameterization 해석
```

---

### 언제 무엇을 쓸 것인가 — 실전 가이드

### 사례 1: 선형 SVM

**상황**: $\mathcal{X} = \mathbb{R}^d$, hinge loss, $\|w\| \leq B$.

**VC 관점**:
- VC($\{\text{halfspaces}\}$) = $d+1$
- Bound: $\sim \sqrt{d/n}$
- 결론: **분포 자유이지만 loose**

**Rademacher 관점**:
- Margin $\gamma$ 기반: $\mathcal{R}_n(\text{hinge}_\gamma) \leq B/(n\gamma)$ (Contraction lemma)
- Bound: $\sim B/(n\gamma)$
- 결론: **Margin이 크면 tight, 작으면 vacuous**

**Stability 관점**:
- Strongly convex, stability $\beta = O(1/(\lambda n))$
- 규제강도 $\lambda$로 직접 제어
- 결론: **Regularization path 선택 가이드 제공**

**추천**: **Rademacher (margin) + Stability** 결합. margin을 크게 하면 Rademacher도 좋아지고, regularization도 안정성 관점에서 정당화.

### 사례 2: Deep Neural Network

**상황**: $f_\theta(x) = W_L \sigma(...\sigma(W_1 x))$, SGD 훈련, $T$ steps.

**VC 관점**:
- VC($\mathcal{H}$) $= O(W^2 \log W)$ (거대!)
- Bound: $\sqrt{W^2 \log W / n}$ → **vacuous** (보통 $W \gg n$)
- 결론: **무용지물**

**Rademacher 관점** (spectral norm):
- $\mathcal{R}_n \lesssim \frac{\prod_l \|W_l\|}{n\sqrt{L}}$ (layer-wise 곱)
- 양호 만약 $\prod \|W_l\|$이 작다면 (implicit regularization)
- 결론: **가능하지만 bounds 계산 어려움**

**Stability 관점** (Hardt et al.):
- SGD step $t$에서: $\beta_t = O(\eta t)$ (learning rate $\eta$, step 수 $t$)
- Bound: $O(\eta T) + O(1/\sqrt{n})$
- 결론: **early stopping으로 직접 제어, 가장 직관적**

**추천**: **Stability (early stopping) + norm-based Rademacher** (spectral norm 정규화). 또는 후속 "Generalization Theory Deep Dive" 레포의 **NTK·Double Descent** 참고.

### 사례 3: 모델 선택

**상황**: 복잡도 다양한 모델들, hyperparameter 선택.

**VC 관점** (SRM, Ch7-01):
- 구조화: $\mathcal{H}_1 \subset \mathcal{H}_2 \subset \ldots$
- 각 $k$의 penalty: $\Omega_k \propto \sqrt{\text{VC}(\mathcal{H}_k)/n}$
- 추정: 느슨함, 보수적
- 결론: **이론적으로 우아하나 실전에선 loose**

**Rademacher**:
- Data-dependent penalty
- 일반적으로 더 tight
- 결론: **더 나음, 하지만 여전히 계산 어려움**

**CV 또는 AIC/BIC**:
- CV: 경험적 추정, 분포 자유, 가장 신뢰
- AIC: 예측 성능 최적
- BIC: 참 모델 선택
- 결론: **실전 표준, 가장 효과적**

**추천**: **CV 표준**. 시간 있으면 nested CV. 빠른 피드백 필요하면 AIC/BIC.

---

## 📌 핵심 정리

| 관점 | 언제 | 왜 | 유의 |
|------|------|-----|------|
| **VC** | 고전 ML, 작은 모델 | 분포 자유, 이론적 기초 | DL은 vacuous |
| **Rademacher** | Kernel, Margin 기반 | 더 tight, 데이터 적응 | 계산 비용, MC 근사 필요 |
| **Stability** | SGD, 정규화 기반 | implicit regularization 설명 | Uniform stability 증명 어려움 |
| **CV** | 모든 현대 ML | 가장 정확한 추정 | 계산 비용, 분산 추정 어려움 |
| **AIC/BIC** | 빠른 선택 필요 | 정보이론적 근거, 빠름 | 가정 민감, DL에서 unreliable |

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> VC bound와 Rademacher bound의 **스케일**을 비교하라. 둘 다 $O(\sqrt{d/n})$이지만, 상수 계수가 다르다. 어느 것이 더 일반적으로 tight한가?</summary>

<br/>

**해설**. VC bound (Sauer-Shelah):
$$4 \cdot (2en/d)^d \cdot e^{-n\epsilon^2/8} \leq \delta.$$

Rademacher bound (for binary, Massart):
$$\mathcal{R}_n \leq \sqrt{\frac{2\log|\mathcal{F}|}{n}} \approx \sqrt{\frac{2d\log(n/d)}{n}}.$$

일반적으로:
- VC의 $(2en/d)^d$ 항이 **exponentially 빠르게 증가** (high $d$에서 거대)
- Rademacher는 **polynomial** $d\log(n/d)$만 증가

따라서 **Rademacher가 고정 $n, d$에서 보통 더 tight**. 단, VC는 "분포 자유"의 대가로 보수적. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> "Zhang et al. 2017" 실험 다시 생각해보기: ResNet이 random labels도 fit하는데도 real labels에선 일반화한다. 이를 세 관점 각각에서 설명하라.</summary>

<br/>

**해설**. 

**VC 관점**: 
- Random label fit → VC가 매우 크다는 증거 (shattering large samples)
- VC bound가 vacuous (VC 차원이 huge)
- **설명 불가능**: bound가 무의미

**Rademacher 관점**:
- Random label에서도 Rademacher 복잡도가 크다는 것 (noise fitting 능력)
- Real label에서는... Rademacher 자체는 변하지 않음 (데이터 구조만 다름)
- **부분 설명**: 만약 real data가 더 구조화되어 있다면 implicit regularization 가능, 하지만 bound는 여전히 큼

**Stability 관점**:
- SGD가 수렴하는 시점 = early stopping point
- Random label: 완전 fit할 때까지 SGD 계속 → memorization
- Real label: validation loss가 증가하면 early stop → 일반화
- **최선의 설명**: early stopping이 implicit regularization, test 성능 반영

**결론**: 고전 VC/Rademacher는 **학습 상황 자체**를 반영하지 못함. Stability + SGD 이론이 가장 정확. → **Layer 2 "Generalization Theory Deep Dive"**로. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> 당신이 새 프로젝트를 시작했다. "모델 A와 B 중 어느 것이 더 일반화될까?"를 판단해야 한다. VC, Rademacher, Stability, CV 중 어느 것을 선택하겠는가? 왜?</summary>

<br/>

**해설**. 실용적 의사결정:

1. **CV 먼저 시도**: 
   - 계산 빠르고 (보통 5-10배 훈련)
   - 결과 직관적 (actual test error)
   - 분포 가정 없음
   - **5~10fold CV 실행, 오차 비교 → A vs B 결정**

2. **CV 불가능한 경우** (계산 극도로 많음, e.g., 초대형 모델):
   - → **Stability 검토**: 정규화 강도 비교
   - Model A: λ_A 강도로 β_A = O(1/(λ_A n)) → 작은 stability
   - Model B: λ_B로 β_B 더 큼
   - → A 선택

3. **이론적 정당성 필요한 경우** (논문, 중요 의사결정):
   - → **Rademacher** (margin 기반) 또는 **norm-based** 계산
   - VC는 피함 (보통 vacuous, 특히 DL)

**최종 권장**: 
- **현실**: CV 표준
- **이론 필요**: Stability (DL 시대 최적)
- **피할 것**: VC (고전, 현대에는 mostly useless)

$\square$

</details>

---

### 📚 다음 단계

이 레포는 **SLT 고전 (1968~2005)**을 완성했다. 현대 DL의 일반화 이론은:

- **[Generalization Theory Deep Dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive)** (후속 레포)
  - Neural Tangent Kernel (NTK)
  - Double Descent phenomenon
  - Norm-based Rademacher for DL
  - PAC-Bayes bounds
  - Mean-field analysis

- **관련 심화 레포**:
  - [Kernel Methods Deep Dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive) — Rademacher 응용
  - [Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive) — SGD 이론의 최적화 기초

---

<div align="center">

◀ [이전: 03. Cross-Validation](./03-cross-validation.md) | [📚 README로 돌아가기](../README.md) | [다음: Generalization Theory Deep Dive ▶](https://github.com/iq-ai-lab/generalization-theory-deep-dive)

</div>
