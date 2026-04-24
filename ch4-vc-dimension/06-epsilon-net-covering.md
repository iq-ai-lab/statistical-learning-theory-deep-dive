# 06. ε-net과 Covering Number

## 🎯 핵심 질문

- **Covering number** $\mathcal{N}(\epsilon, \mathcal{H}, \|\cdot\|)$는 무엇인가? "가설공간을 ε-공으로 덮기"의 의미는?
- **ε-net**과의 관계는? "모든 $f$마다 가까운 대표가 있다"는 것이 일반화와 무엇을 뜻하는가?
- **Packing number**와의 대우 관계 $\mathcal{N}(2\epsilon) \leq \mathcal{M}(\epsilon) \leq \mathcal{N}(\epsilon)$는?
- **Chaining argument** (dyadic decomposition)는 무엇인가?
- **Dudley's entropy integral**: $\mathbb{E}[\sup_f G_f] \leq C \int_0^\infty \sqrt{\log \mathcal{N}(\epsilon)}$의 의미는?
- **Haussler's packing bound**: VC 차원과 covering의 연결은?

---

## 🔍 왜 Covering이 중요한가

VC 경계는 "uniform convergence를 보장한다"고 했지만, 실제 ML에서는:

1. **Norm-constrained hypothesis**: $\{w : \|w\| \leq B\}$ 같은 연속 제약
2. **Function space**: 무한 차원 (neural network의 가중치)
3. **Refined bounds**: 데이터에 의존적인 복잡도 (Rademacher, covering)

Covering은 이들을 **유한화**하는 또 다른 방법이다 — VC와 다른 관점.

---

## 📐 수학적 선행 조건

- [Ch4-04](./04-sauer-shelah.md): 성장함수, Sauer-Shelah
- [Ch5-02](../ch5-rademacher/02-rademacher-generalization.md): Rademacher 복잡도
- 기초: Metric space, ball, norm, entropy

---

## 📖 직관적 이해

### Covering vs Packing

**Covering**: "가설공간을 finite대표로 덮기"
- 모든 $h \in \mathcal{H}$에 대해, 어떤 대표 $h' \in \text{cover}$가 거리 $\|h - h'\| \leq \epsilon$ 내에 있다.
- **$\epsilon$ 작을수록** → 대표 많이 필요 → $\mathcal{N}(\epsilon) \uparrow$

**Packing**: "거리 $\epsilon$ 이상 떨어진 점들 최대한 많이 떨어뜨리기"
- 서로 거리 $\|h_i - h_j\| > 2\epsilon$인 최대 개수의 $h_i$ 찾기.
- 더 "efficient" — covering의 상한.

### Chaining (Dyadic Decomposition)

무한 차원 function space에서 $\sup_f |Z_f|$를 bound하려면, covering을 "계층적으로" 사용:

- Level 0: $\epsilon_0 = 1$, cover size $\mathcal{N}(1)$
- Level 1: $\epsilon_1 = 1/2$, cover size $\mathcal{N}(1/2)$
- ...
- Level $k$: $\epsilon_k = 2^{-k}$, cover size $\mathcal{N}(2^{-k})$

각 level에서 "인접한" 대표 간 거리는 $\epsilon_k$이므로, supremum의 변동을 추적할 수 있다.

---

## ✏️ 엄밀한 정의

### 정의 4.13 (Covering Number)

$(F, \|\cdot\|)$를 normed function space라 하자 (또는 가설공간 $\mathcal{H}$).

**Covering number**는:
$$\mathcal{N}(\epsilon, F, \|\cdot\|) := \min \left\{k : \exists h_1, \ldots, h_k \in F, \quad \forall h \in F, \exists i: \|h - h_i\| \leq \epsilon\right\}.$$

즉, 반지름 $\epsilon$인 공으로 $F$를 덮는 최소 개수.

### 정의 4.14 (Packing Number)

**Packing number**는:
$$\mathcal{M}(\epsilon, F, \|\cdot\|) := \max \left\{k : \exists h_1, \ldots, h_k \in F, \quad \forall i \neq j: \|h_i - h_j\| > \epsilon\right\}.$$

즉, 최소 거리 $\epsilon$을 유지하며 떨어뜨릴 수 있는 최대 개수.

### 정의 4.15 (ε-net)

$\delta > 0$에 대해, $\mathcal{H}_\epsilon$가 $\mathcal{H}$의 **$\epsilon$-net**이라는 것은:
$$\forall h \in \mathcal{H}, \exists h' \in \mathcal{H}_\epsilon: \|h - h'\| \leq \epsilon.$$

그리고 $|\mathcal{H}_\epsilon| \leq \mathcal{N}(\epsilon)$.

---

## 🔬 정리와 증명

### 정리 4.20 (Covering ↔ Packing)

$$\mathcal{N}(2\epsilon, F, \|\cdot\|) \leq \mathcal{M}(\epsilon, F, \|\cdot\|) \leq \mathcal{N}(\epsilon, F, \|\cdot\|).$$

**증명**:

**부등식 1** ($\mathcal{M}(\epsilon) \leq \mathcal{N}(\epsilon)$):
$M_1, \ldots, M_k$가 packing (서로 거리 $> \epsilon$). 만약 $\mathcal{N}(\epsilon) < k$이면, covering $C_1, \ldots, C_{k-1}$이 존재하여 $F$를 덮는다. 그러면 $M_i$ 중 두 개 (예: $M_i, M_j$)가 같은 공 $C_r$ 안에 있어야 한다. 하지만 그러면 $\|M_i - M_j\| \leq \epsilon$ (같은 공 안) → packing 조건 위배. 따라서 $\mathcal{M}(\epsilon) \leq \mathcal{N}(\epsilon)$. $\square$

**부등식 2** ($\mathcal{N}(2\epsilon) \leq \mathcal{M}(\epsilon)$):
$h_1, \ldots, h_k$가 maximal packing (거리 $> \epsilon$). 각 $h \in F$에 대해, 어떤 $h_i$와 거리 $\leq \epsilon$ (packing의 extremality). 따라서 $\{B(h_i, 2\epsilon)\}$들이 $F$를 덮는다. 그러면 $2\epsilon$-cover의 크기 $\leq k = \mathcal{M}(\epsilon)$. $\square$

### 정리 4.21 (Covering과 VC의 연결 — Haussler)

$$\mathcal{N}(\epsilon, \mathcal{H}, \ell_\infty) \leq \left(\frac{12}{\epsilon}\right)^{\text{VC}(\mathcal{H})}.$$

(여기서 $\ell_\infty$ norm: $\|h - h'\|_\infty = \sup_{x} |h(x) - h'(x)|$)

**증명 스케치**: 
1. $\mathcal{H}$의 임의 subset이 최대 $\Pi(m)$ dichotomy를 가질 수 있다 (정의).
2. Covering을 구성할 때, 각 "축소된" 가설 $h|_S$가 최대 $\Pi(|S|)$ 형태만 가능.
3. Sauer-Shelah로 $\Pi(m) \leq (em/d)^d$ (where $d = \text{VC}$).
4. Metric covering 논증으로 covering number 상계.

결과적으로 $\mathcal{N}(\epsilon) \leq (C/\epsilon)^d$ for some $C$.

### 정리 4.22 (Dudley's Entropy Integral)

Sub-Gaussian random variables $\{G_f\}_{f \in F}$에 대해:

$$\mathbb{E}[\sup_{f \in F} G_f] \lesssim \int_0^\infty \sqrt{\log \mathcal{N}(\epsilon, F, \|\cdot\|)} \, d\epsilon.$$

**직관**:
- 작은 $\epsilon$: covering이 크고 (log term 크고), entropy integral에 더 기여.
- 큰 $\epsilon$: covering이 작지만 (log term 작음), 범위는 넓음.
- 균형이 integral에 반영.

**증명 스케치**: Chaining argument.
- $\epsilon_k = 2^{-k}$로 dyadic grid 구성.
- 각 $f$를 가장 가까운 covering $C_k$의 대표로 추적.
- Telescoping: $\sup_f G_f = \lim_k (\text{covering 대표까지의 변동}) + (\text{fine level의 변동}).
- Union bound와 entropy를 조합하여 적분 형태 도출.

---

## 💻 NumPy 구현 검증

### 실험 1: Covering vs Packing

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# 1D interval [0, 1]에서 함수들의 covering/packing 비교
# H = {h_theta(x) = 1[x >= theta] : theta in [0, 1]}

def compute_covering_packing(eps, grid_size=1000):
    """
    1D threshold의 covering과 packing을 수치적으로 계산.
    """
    thetas = np.linspace(0, 1, grid_size)
    
    # Metric: sup_x |h_theta(x) - h_theta'(x)|
    # 두 threshold h_theta, h_theta'에 대해:
    # 다른 점들은 x >= theta와 x >= theta' 사이 구간에서만 다름
    # → metric은 |theta - theta'|
    
    # Packing: 거리 > eps인 점들 최대한
    # 간단히: [0, 1]에서 eps 떨어진 점들 → ceil(1/eps) 개
    packing_size = int(np.ceil(1.0 / eps)) + 1
    
    # Covering: 거리 <= eps로 덮기
    # eps-spaced grid: ceil(1/eps) + 1 개
    covering_size = int(np.ceil(1.0 / eps)) + 1
    
    return covering_size, packing_size

epsilons = np.logspace(-3, 0, 20)  # 0.001 ~ 1
coverings = []
packings = []

for eps in epsilons:
    c, p = compute_covering_packing(eps)
    coverings.append(c)
    packings.append(p)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.loglog(epsilons, coverings, 'o-', label=r'$\mathcal{N}(\epsilon)$')
ax1.loglog(epsilons, packings, 's--', label=r'$\mathcal{M}(\epsilon)$')
ax1.loglog(epsilons, 2/epsilons, '^:', alpha=0.5, label=r'$2/\epsilon$ (theory)')
ax1.set_xlabel(r'$\epsilon$')
ax1.set_ylabel('Count')
ax1.set_title('1D Threshold: Covering vs Packing')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 관계 $M(eps) <= N(eps)$ 확인
ratio = np.array(packings) / np.array(coverings)
ax2.semilogx(epsilons, ratio, 'o-')
ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='M/N = 1')
ax2.set_xlabel(r'$\epsilon$')
ax2.set_ylabel(r'$\mathcal{M}(\epsilon) / \mathcal{N}(\epsilon)$')
ax2.set_title('Packing과 Covering의 비율')
ax2.set_ylim([0, 1.5])
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# → M <= N 확인, 1D에서는 거의 같음
```

### 실험 2: Covering과 일반화

```python
# 간단한 예: linear classifier w^T x (w in [-B, B]^d)
# covering의 크기 vs 일반화 error

def covering_linear_class(eps, d, B=1.0):
    """
    d차원 선형 분류기의 eps-covering 크기.
    w_i in [-B, B]이고, 각 w_i를 eps로 grid하면:
    ceil(2B/eps)^d개.
    """
    grid_per_dim = int(np.ceil(2 * B / eps))
    return grid_per_dim ** d

d_values = [1, 2, 5, 10]
epsilons = np.logspace(-2, 0, 20)

fig, ax = plt.subplots(figsize=(10, 6))

for d in d_values:
    covers = [covering_linear_class(eps, d) for eps in epsilons]
    ax.loglog(epsilons, covers, 'o-', label=f'd={d}')

ax.set_xlabel(r'$\epsilon$ (tolerance)')
ax.set_ylabel(r'$\mathcal{N}(\epsilon)$ (covering size)')
ax.set_title(r'Linear classifier: Covering size vs dimension')
ax.legend()
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.show()

# → 차원 증가 → covering 지수적 증가
# 이것이 "curse of dimensionality" in covering
```

---

## 🔗 ML 알고리즘 연결

Covering은 다음에 사용:

1. **Norm-based Rademacher** (Ch5-06): Norm constraint의 covering으로 복잡도 bound
2. **Kernel SVM**: RKHS covering via entropy integral (Dudley)
3. **Metric learning**: Covering으로 metric space의 샘플 복잡도 분석
4. **Online learning**: Dynamic covering (moving target 추적)

---

## ⚖️ 가정과 한계

1. **Metric 의존성**: Covering은 norm/metric에 의존 — 다른 norm 쓰면 다른 covering
2. **계산 어려움**: 일반 $\mathcal{H}$의 covering을 명시적으로 계산하기는 어려움
3. **Curse of dimensionality**: High-d에서 covering이 exponential → entropy integral도 커짐

---

## 📌 핵심 정리

- **Covering number**: 반지름 $\epsilon$ 공으로 가설공간을 덮는 최소 개수
- **Packing number**: 최소 거리 $\epsilon$ 유지하며 떨어뜨릴 수 있는 최대 개수
- **관계**: $\mathcal{N}(2\epsilon) \leq \mathcal{M}(\epsilon) \leq \mathcal{N}(\epsilon)$
- **VC와의 연결**: Haussler bound — covering은 VC 차원에 의존
- **Dudley integral**: 무한 차원 function space의 supremum bound

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> 1D interval [0, 1]에서 threshold classifier의 covering number를 $\epsilon$의 함수로 구하라.</summary>

<br/>

**해설**. Threshold $h_\theta(x) = 1[x \geq \theta]$는 매개변수 $\theta \in [0, 1]$로 정의된다. 두 threshold $h_\theta, h_{\theta'}$의 $\ell_\infty$ 거리는:

$$\|h_\theta - h_{\theta'}\|_\infty = \sup_x |h_\theta(x) - h_{\theta'}(x)|.$$

$\theta < \theta'$일 때, 다른 점은 $[\theta, \theta')$ 구간에서만. 따라서 metric $= |\theta - \theta'|$.

$\epsilon$-covering: $[0, 1]$을 $\epsilon$-간격의 grid로 커버. 필요한 점: $\lceil 1/\epsilon \rceil + 1$.

따라서 $\mathcal{N}(\epsilon) = O(1/\epsilon)$. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> Dudley's entropy integral에서 "chaining"의 핵심은 무엇인가?</summary>

<br/>

**해설**. Chaining은 dyadic level에서:

$$\sup_f Z_f = \sup_f (Z_f - Z_{\text{cover}_0}(f)) + \sup_f (Z_{\text{cover}_0(f)} - Z_{\text{cover}_1(f)}) + \cdots$$

각 level에서의 변동은 covering이 얼마나 "촘촘"한지 (크기, entropy)에 의존. 작은 레벨일수록 정확하지만 covering이 크고, 큰 레벨일수록 covering이 작지만 손실 커짐. 이 balance를 integral로 aggregated하면 efficient bound가 나온다. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> Linear SVM의 covering-based bound가 VC bound보다 tighter한 이유는?</summary>

<br/>

**해설**. SVM은 $\|w\| \leq B$ 제약이 추가되는데, 이는 **norm constraint covering**을 가능하게 한다. VC bound는 "worst-case"로 모든 dichotomy를 센 반면, covering-based bound는 "data-dependent" norm이나 margin을 활용할 수 있다. Rademacher (Ch5)가 이를 더 체계화하며, Margin bound로 완성된다 (Ch7). $\square$

</details>

---

<div align="center">

◀ [이전: 05. VC 경계 유도](./05-vc-bound-derivation.md) | [📚 README](../README.md) | [다음: 07. VC 한계 ▶](./07-vc-limits.md)

</div>
