# 05. Linear Class와 Kernel Class의 Rademacher

## 🎯 핵심 질문

- **선형 함수족** $\mathcal{F} = \{w^\top x : \|w\| \leq B\}$의 Rademacher는 얼마나 작은가? **Cauchy-Schwarz**로 어떻게 유도하는가?
- $\mathcal{R}_n(\mathcal{F}) \leq B \cdot \max_i \|x_i\| / \sqrt{n}$ — 이것이 왜 **데이터에 의존적**인가?
- **RKHS (Reproducing Kernel Hilbert Space)** 볼의 Rademacher는? $\mathcal{R}_n(\mathcal{F}) \leq B \sqrt{\text{tr}(K)/n}$ 어디서 나오는가?
- **Kernel matrix** $K_{ij} = k(x_i, x_j)$와 Rademacher의 연결은?
- **SVM과의 연결**: margin을 최대화하는 것이 왜 $\|w\|$를 최소화하는 것과 같고, 이것이 Rademacher를 줄이는가?

---

## 🔍 왜 이 이론이 현대 ML에서 중요한가

**Linear classifier**와 **Kernel method**는 SVM, Logistic Regression, 초기 Neural Networks의 이론적 기반이다. 이 문서는 다음을 보인다:

1. **선형 분류기의 일반화**: $\mathcal{R}_n$이 **norm $\|w\|$ 제약과 데이터 norm에만** 의존한다. **파라미터 수보다 norm이 중요** — 이것이 regularization의 수학적 근거.

2. **Kernel SVM의 정당성**: Kernel trick을 통한 무한차원 RKHS에서도 **trace of kernel matrix**가 복잡도를 제어한다 — kernel matrix eigenvalue들이 실제 표현력을 결정.

3. **Norm-based generalization**: VC는 "몇 개 파라미터 = 몇 개 VC차원", Rademacher는 "파라미터의 크기 = norm"을 본다 — DL에서 "고전 VC는 vacuous, norm-based Rademacher는 의미있다"는 관찰의 초석.

---

## 📐 수학적 선행 조건

- **Ch5-01 (Rademacher 정의)**: $\hat{\mathcal{R}}_S$, $\mathcal{R}_n$ 정의
- **Ch5-02 (일반화 경계)**: 정리 5.5
- **Ch5-03 (Contraction)**: Lipschitz → Rademacher 축소
- **[Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive)**: 벡터 norm, Cauchy-Schwarz, affine independence
- **[Kernel Methods Deep Dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive)**: RKHS, kernel trick, positive semi-definite
- 기초: inner product, norm의 성질, 행렬의 trace

---

## 📖 직관적 이해

### 선형 분류기: Norm 제약의 힘

$f(x) = w^\top x$의 Rademacher를 생각하자. 가장 큰 상관성을 주려면:
$$\sup_{\|w\| \leq B} \sum_i \sigma_i w^\top x_i = B \sup_{\|\hat{w}\| = 1} \hat{w}^\top \left(\sum_i \sigma_i x_i\right) = B \left\|\sum_i \sigma_i x_i\right\|.$$

기대값을 취하면:
$$\mathcal{R}_n(\mathcal{F}) = \mathbb{E}_{S, \sigma}\left[\frac{B}{n} \left\|\sum_i \sigma_i x_i\right\|\right].$$

이제 $\mathbb{E}[\|\sum \sigma_i x_i\|]$를 계산한다:
$$\mathbb{E}\left[\left\|\sum_i \sigma_i x_i\right\|^2\right] = \mathbb{E}\left[\sum_i \sigma_i^2 \|x_i\|^2 + 2\sum_{i<j} \sigma_i \sigma_j x_i^\top x_j\right] = \sum_i \|x_i\|^2$$

(교차항은 독립성으로 0, $\sigma_i^2 = 1$).

따라서:
$$\mathbb{E}\left[\left\|\sum_i \sigma_i x_i\right\|\right] \leq \sqrt{\mathbb{E}\left[\left\|\sum_i \sigma_i x_i\right\|^2\right]} = \sqrt{\sum_i \|x_i\|^2} \leq \sqrt{n} \cdot \max_i \|x_i\|.$$

결론: $\mathcal{R}_n \leq B \cdot \max_i \|x_i\| / \sqrt{n}$ — **데이터 norm과 norm 제약에만 의존**.

### Kernel: Feature map의 "대리"

Kernel method에서는 명시적 feature map $\phi(x) \in \mathcal{H}$ (무한차원)를 피하고, kernel $k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle_\mathcal{H}$만 계산한다.

RKHS 함수족 $\mathcal{F} = \{f(\cdot) = \langle w, \phi(\cdot) \rangle : \|w\|_\mathcal{H} \leq B\}$의 Rademacher:
$$\mathcal{R}_n(\mathcal{F}) \leq \frac{B}{n} \sqrt{\text{tr}(K)} = \frac{B}{n} \sqrt{\sum_i k(x_i, x_i)},$$

여기서 $K = (k(x_i, x_j))_{ij}$ (kernel matrix).

**의미**: 무한차원 RKHS인데도, Rademacher는 **kernel matrix의 대각 원소합** (trace)로만 표현된다 — "kernel method의 기적".

### Margin과 Norm의 수학적 동치

SVM에서 margin $\gamma$는:
$$\gamma = \frac{\min_i y_i f(x_i)}{\|w\|}, \quad f(x) = w^\top x.$$

최대 margin 최적화:
$$\max \gamma \Leftrightarrow \min \|w\|.$$

정리 5.10에서 본 margin loss의 Rademacher:
$$\mathcal{R}_n(\ell_\gamma \circ \mathcal{F}) \propto \frac{1}{\gamma} \mathcal{R}_n(\mathcal{F}).$$

$\mathcal{R}_n(\mathcal{F}) \propto \|w\|$ (선형)이므로, margin ↑ ↔ $\|w\|$ ↓ ↔ Rademacher ↓ → **일반화 ↑**. 이것이 margin 최대화의 이론적 정당화.

---

## ✏️ 엄밀한 정의

### 정의 5.11 (선형 함수족)

$$\mathcal{F}_B = \{f(x) = w^\top x : \|w\|_2 \leq B, w \in \mathbb{R}^d\}.$$

또는 bias 포함:
$$\mathcal{F}_B = \{f(x) = w^\top x + b : \|w\|_2 \leq B, |b| \leq B'\}.$$

### 정의 5.12 (RKHS ball)

Kernel $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ (positive semi-definite)에 대해, RKHS $\mathcal{H}_k$와:

$$\mathcal{F}_B = \{f(\cdot) = \langle w, \phi(\cdot) \rangle_{\mathcal{H}} : \|w\|_{\mathcal{H}} \leq B\},$$

여기서 $\phi(x) := k(x, \cdot) \in \mathcal{H}$ (reproducing property: $f(x) = \langle f, k(x,\cdot) \rangle$).

### 정의 5.13 (Kernel matrix)

$$K = (K_{ij})_{i,j=1}^n, \quad K_{ij} := k(x_i, x_j).$$

**Trace**:
$$\text{tr}(K) = \sum_{i=1}^n K_{ii} = \sum_{i=1}^n k(x_i, x_i).$$

---

## 🔬 정리와 증명

### 정리 5.14 (선형 함수족의 Rademacher) ★

$\mathcal{F} = \{x \mapsto w^\top x : \|w\|_2 \leq B\}$에 대해:

$$\hat{\mathcal{R}}_S(\mathcal{F}) \leq \frac{B}{n} \sqrt{\sum_{i=1}^n \|x_i\|_2^2} \leq \frac{B \cdot \max_i \|x_i\|_2}{\sqrt{n}}.$$

따라서 모집단 Rademacher (데이터 분포 $\mathcal{D}$에서):
$$\mathcal{R}_n(\mathcal{F}) \leq \frac{B \cdot M}{\sqrt{n}},$$
여기서 $M := \mathbb{E}[\|X\|_2] \leq \mathbb{E}^{1/2}[\|X\|_2^2]$ (input norm의 기대값).

**증명**. 

**Step 1**: Supremum 계산:
$$\sup_{\|w\| \leq B} \sum_{i=1}^n \sigma_i w^\top x_i = \sup_{\|w\| \leq B} w^\top \left(\sum_{i=1}^n \sigma_i x_i\right).$$

**Cauchy-Schwarz**: 
$$w^\top v = \langle w, v \rangle \leq \|w\| \cdot \|v\|.$$

따라서:
$$\sup_{\|w\| \leq B} w^\top \left(\sum_i \sigma_i x_i\right) = B \left\|\sum_i \sigma_i x_i\right\|.$$

**Step 2**: 기대값:
$$\hat{\mathcal{R}}_S(\mathcal{F}) = \mathbb{E}_\sigma\left[\frac{1}{n} B \left\|\sum_i \sigma_i x_i\right\|\right] = \frac{B}{n} \mathbb{E}_\sigma\left[\left\|\sum_i \sigma_i x_i\right\|\right].$$

**Step 3**: Norm의 기대값 bound. Cauchy-Schwarz (다시):
$$\mathbb{E}\left[\left\|\sum_i \sigma_i x_i\right\|\right] \leq \sqrt{\mathbb{E}\left[\left\|\sum_i \sigma_i x_i\right\|^2\right]}.$$

내부:
$$\mathbb{E}\left[\left\|\sum_i \sigma_i x_i\right\|^2\right] = \mathbb{E}\left[\left\langle \sum_i \sigma_i x_i, \sum_j \sigma_j x_j \right\rangle\right] = \sum_i \mathbb{E}[\sigma_i^2] \|x_i\|^2 + \sum_{i \neq j} \mathbb{E}[\sigma_i \sigma_j] \langle x_i, x_j \rangle.$$

독립성과 $\mathbb{E}[\sigma_i] = 0$, $\mathbb{E}[\sigma_i^2] = 1$:
$$= \sum_i \|x_i\|^2.$$

따라서:
$$\mathbb{E}\left[\left\|\sum_i \sigma_i x_i\right\|\right] \leq \sqrt{\sum_i \|x_i\|^2}.$$

**Step 4**: 종합:
$$\hat{\mathcal{R}}_S(\mathcal{F}) \leq \frac{B}{n} \sqrt{\sum_i \|x_i\|^2}. \quad \square$$

추가: $\sqrt{\sum_i \|x_i\|^2} \leq \sqrt{n} \cdot \max_i \|x_i\|$이므로 두 번째 부등식도 성립.

### 정리 5.15 (Kernel class의 Rademacher)

RKHS kernel $k$, ball 반지름 $B$에 대해:

$$\mathcal{R}_n(\mathcal{F}_B) \leq \frac{B}{n} \sqrt{\sum_{i=1}^n k(x_i, x_i)} = \frac{B \sqrt{\text{tr}(K)}}{n}.$$

**증명**. 정리 5.14의 직접 일반화.

Kernel의 reproducing property: $f(x) = \langle w, k(x, \cdot) \rangle$. 이것을 feature map $\phi(x) = k(x, \cdot) \in \mathcal{H}$로 보면 선형:
$$f(x) = \langle w, \phi(x) \rangle_\mathcal{H}.$$

정리 5.14를 $\phi(x_i)$ (RKHS 내 벡터)에 적용:
$$\hat{\mathcal{R}}_S \leq \frac{B}{n} \sqrt{\sum_i \|\phi(x_i)\|_\mathcal{H}^2} = \frac{B}{n} \sqrt{\sum_i k(x_i, x_i)}.$$

(RKHS norm: $\|\phi(x)\|_\mathcal{H}^2 = k(x,x)$, reproducing property). $\square$

### 정리 5.16 (SVM margin bound)

Soft-margin SVM에서, 훈련 데이터가 margin $\gamma > 0$으로 분리되면 ($y_i f(x_i) \geq \gamma$ for all $i$),

$$\sup_h |L_\mathcal{D}(h) - L_S(h)| \leq O\left(\frac{B \max_i \|x_i\|}{\gamma \sqrt{n}} + \sqrt{\frac{\log(1/\delta)}{n}}\right).$$

(여기서 $\|w\| \leq B$, kernel bound $\leq B\sqrt{\text{tr}(K)/n}$ for kernel SVM).

**증명 스케치**. 
1. Margin loss $\ell_\gamma(z) = \max(0, 1 - z/\gamma)$는 $1/\gamma$-Lipschitz (정리 5.10).
2. Contraction lemma: $\mathcal{R}_n(\ell_\gamma \circ \mathcal{H}) \leq (1/\gamma) \mathcal{R}_n(\mathcal{H})$.
3. 선형 SVM: $\mathcal{R}_n(\mathcal{H}) \leq B \max_i \|x_i\| / \sqrt{n}$ (정리 5.14).
4. 정리 5.5 적용. $\square$

**의미**: **Margin을 크게 유지하면서 norm $\|w\|$를 작게** → Rademacher ↓ → 일반화 ↑.

---

## 💻 NumPy 구현 검증

### 실험 1: 선형 함수족의 Rademacher

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# 데이터 생성
def sample_linear_data(n, d=10, noise_std=0.1):
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d)
    w_true = w_true / np.linalg.norm(w_true)
    y = (X @ w_true) + noise_std * rng.standard_normal(n)
    return X, y, w_true

# 경험적 Rademacher 복잡도 (선형)
def rademacher_linear_emp(X, B=1.0, n_rademacher=1000):
    n, d = X.shape
    vals = []
    for _ in range(n_rademacher):
        sigma = rng.choice([-1, 1], size=n)
        # E_σ[sup_{||w||≤B} (1/n) Σ σ_i w^T x_i]
        # = (B/n) E_σ[||Σ σ_i x_i||]
        weighted_sum = (sigma[:, None] * X).sum(axis=0)
        norm = np.linalg.norm(weighted_sum)
        vals.append(B * norm / n)
    return np.mean(vals)

# 이론적 상한
def rademacher_linear_upper(X, B=1.0):
    n = len(X)
    # R ≤ (B/n) * sqrt(sum ||x_i||^2)
    norms_sq = np.sum(np.linalg.norm(X, axis=1) ** 2)
    return B * np.sqrt(norms_sq) / n

# 실험: 다양한 n
ns = [10, 20, 50, 100, 200, 500]
d = 10
B = 1.0

empirical_rads = []
theoretical_bounds = []

for n in ns:
    X, y, w_true = sample_linear_data(n, d)
    rad_emp = rademacher_linear_emp(X, B, n_rademacher=500)
    rad_theory = rademacher_linear_upper(X, B)
    empirical_rads.append(rad_emp)
    theoretical_bounds.append(rad_theory)
    print(f"n={n:3d}: Empirical R̂={rad_emp:.4f}, Theory={rad_theory:.4f}, ratio={rad_emp/rad_theory:.2f}")

# 시각화
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ns, empirical_rads, 'o-', label='Empirical R̂_S', linewidth=2, markersize=8)
ax.plot(ns, theoretical_bounds, 's--', label='Theoretical upper bound', linewidth=2, markersize=8)
ax.set_xlabel('Sample size n'); ax.set_ylabel('Rademacher complexity')
ax.set_title('Linear class: empirical vs theoretical Rademacher')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

### 실험 2: Kernel matrix의 Trace와 Rademacher

```python
from sklearn.metrics.pairwise import rbf_kernel

# RBF kernel: k(x, x') = exp(-γ||x-x'||^2)
def rademacher_kernel_emp(X, gamma=1.0, B=1.0, n_rademacher=1000):
    """
    RKHS ball의 Rademacher 추정
    이론: R ≤ (B/n) sqrt(tr(K))
    """
    K = rbf_kernel(X, gamma=gamma)
    trace_K = np.trace(K)
    theory_bound = (B / len(X)) * np.sqrt(trace_K)
    return theory_bound

# 실험
n = 100
d = 5
X = rng.standard_normal((n, d))

gammas = [0.1, 0.5, 1.0, 2.0, 5.0]
for gamma in gammas:
    K = rbf_kernel(X, gamma=gamma)
    trace_K = np.trace(K)
    rad_bound = (1.0 / n) * np.sqrt(trace_K)
    print(f"γ={gamma:.1f}: tr(K)={trace_K:.2f}, R_bound={rad_bound:.4f}")

# 의미: 더 복잡한 kernel (작은 γ → 부드러운) → 더 큰 trace → 더 큰 Rademacher
```

### 실험 3: Norm 제약의 효과

```python
# Linear SVM: 다양한 norm bound B에 대한 Rademacher
def rademacher_vs_norm_bound(X, B_values=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """
    R ∝ B 확인
    """
    n = len(X)
    norms_sq = np.sum(np.linalg.norm(X, axis=1) ** 2)
    base = np.sqrt(norms_sq) / n
    
    results = []
    for B in B_values:
        rad = B * base
        results.append(rad)
    
    return B_values, results

n = 100
X = rng.standard_normal((n, 5))
Bs, rads = rademacher_vs_norm_bound(X)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Bs, rads, 'o-', linewidth=2, markersize=8)
ax.plot(Bs, 0.1 * np.array(Bs), '--', alpha=0.5, label='Linear: R ∝ B')
ax.set_xlabel('Norm bound B'); ax.set_ylabel('Rademacher complexity')
ax.set_title('Linear class: R ∝ B (norm bound effect)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# → B가 커질수록 Rademacher도 선형으로 증가
# 이것이 SVM의 soft-margin trade-off: large B → large R → large error bound
```

---

## 🔗 ML 알고리즘 연결

| 알고리즘 | 함수족 | Rademacher |가정 |
|---------|------|-----------|------|
| **Linear SVM** | $\{w \cdot x : \|w\| \leq B\}$ | $O(B \max\|x\| / \sqrt{n})$ | Data는 bounded norm |
| **Kernel SVM** | RKHS ball | $O(B \sqrt{\text{tr}(K)/n})$ | Kernel matrix trace bounded |
| **Logistic Regression** | Linear + log loss | $O(B \max\|x\| / \sqrt{n})$ | Surrogate loss Lipschitz |
| **Ridge Regression** | Linear + squared loss | $O(B \max\|x\| / \sqrt{n})$ | Bounded output |
| **Gaussian Process** | RKHS (infinite-dim) | Kernel-dependent | Usually small (smooth kernel) |

**의미**: 
- **SVM의 margin 최대화** = norm 최소화 = Rademacher 최소화 = 일반화 개선.
- **Kernel method의 강점**: 무한차원 RKHS인데도 Rademacher는 trace(kernel matrix)로 제어 — "kernel method의 매직".
- **Regularization의 수학**: $\|w\|$ 제약이 generalization bound를 직접 감소 — L2 regularization의 이론적 정당화.

---

## ⚖️ 가정과 한계

1. **Data norm 가정**: $\max_i \|x_i\|$ 또는 $\mathbb{E}[\|X\|]$이 bounded. Unbounded 데이터면 bound가 의미 없음.
2. **Norm 제약의 정당성**: $\|w\| \leq B$는 임의 상수. 실전에서는 regularization parameter $\lambda$로 조정해서 implicit bound.
3. **RKHS의 무한성**: 정리 5.15는 무한차원 RKHS인데도 유한 bound를 준다는 것이 강점. 하지만 이는 "bounded norm" 가정 덕분.
4. **Margin bound의 practicality**: 정리 5.16의 bound는 여전히 loose할 수 있음 — margin $\gamma$가 작으면 $1/\gamma$배 증가.
5. **Non-linear 함수족**: linear와 kernel의 경계. 임의의 non-linear (예: 깊은 NN)은 이 bound를 벗어남 (Ch5-06).

---

## 📌 핵심 정리

- **선형 함수족**: $\mathcal{R}_n = O(B \max\|x\| / \sqrt{n})$ — norm과 data norm 제약만 의존.
- **증명**: Cauchy-Schwarz로 $\sup_{\|w\| \leq B} w^\top v = B\|v\|$, 기대값 계산.
- **Kernel class**: RKHS ball → $\mathcal{R}_n = O(B \sqrt{\text{tr}(K)/n})$ — kernel matrix trace로 표현.
- **SVM margin bound**: margin $\gamma$ ↑ → $(1/\gamma) \mathcal{R}_n$ ↓ → 일반화 개선.
- **핵심 인사이트**: Norm 제약이 복잡도를 제어. VC는 파라미터 수, Rademacher는 norm의 크기.
- **Kernel Methods Deep Dive**로의 다리: RKHS의 이론적 근거.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> Cauchy-Schwarz 부등식을 사용해서 $\sup_{\|w\| \leq B} w^\top v = B\|v\|$임을 보여라.</summary>

<br/>

**해설**. Cauchy-Schwarz: $w^\top v = \langle w, v \rangle \leq \|w\| \cdot \|v\|$.

$\|w\| \leq B$이면:
$$\sup_{\|w\| \leq B} w^\top v \leq B \|v\|.$$

등호 달성: $w = B \cdot \frac{v}{\|v\|}$ (혹은 $v$ 방향 단위벡터)로 놓으면, $\|w\| = B$이고
$$w^\top v = B \frac{v}{\|v\|} \cdot v = B \|v\|.$$

따라서 $\sup = B\|v\|$. $\square$

**직관**: 최대 내적은 같은 방향에서 달성된다 — 최적의 $w$는 $v$ 방향으로 최대 크기.

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> 정리 5.14의 증명에서 $\mathbb{E}[\|\sum \sigma_i x_i\|^2] = \sum \|x_i\|^2$가 성립하는 이유를 자세히 설명하라. 특히 교차항 $\mathbb{E}[\sigma_i \sigma_j x_i^\top x_j]$가 왜 0이 되는가?</summary>

<br/>

**해설**. Expanding:
$$\left\|\sum_i \sigma_i x_i\right\|^2 = \left\langle \sum_i \sigma_i x_i, \sum_j \sigma_j x_j \right\rangle = \sum_i \sum_j \sigma_i \sigma_j \langle x_i, x_j \rangle.$$

두 경우로 나눈다:

1. **$i = j$**: $\sigma_i^2 \langle x_i, x_i \rangle = 1 \cdot \|x_i\|^2 = \|x_i\|^2$ (∵ $\sigma_i^2 = 1$).
   
   기대값: $\mathbb{E}[\sigma_i^2 \|x_i\|^2] = \mathbb{E}[1 \cdot \|x_i\|^2] = \|x_i\|^2$.

2. **$i \neq j$**: $\sigma_i \sigma_j \langle x_i, x_j \rangle$.

   기대값: $\mathbb{E}[\sigma_i \sigma_j \langle x_i, x_j \rangle] = \mathbb{E}[\sigma_i] \cdot \mathbb{E}[\sigma_j] \cdot \langle x_i, x_j \rangle$ (독립성).
   
   $\mathbb{E}[\sigma_i] = \frac{1}{2}(+1) + \frac{1}{2}(-1) = 0$.
   
   따라서 $= 0$.

전체:
$$\mathbb{E}\left[\left\|\sum_i \sigma_i x_i\right\|^2\right] = \sum_i \|x_i\|^2 + 0 = \sum_i \|x_i\|^2. \quad \square$$

**의미**: Rademacher의 **직교성** — 서로 다른 $\sigma$들의 교차항은 기대값에서 0이 된다. 따라서 제곱의 합은 각 제곱들의 합.

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> **SVM의 margin 최대화**: 훈련 데이터 $(x_i, y_i)$에서 모든 점이 margin $\gamma$로 분리된다면 ($y_i w^\top x_i \geq \gamma$ for all $i$), Rademacher bound를 이용해서 왜 $\|w\|$를 작게 해야 일반화가 좋은지 설명하라.</summary>

<br/>

**해설**. 

**Step 1**: Margin loss (정리 5.10):
$$\ell_\gamma(z) = \max(0, 1 - z/\gamma).$$

이것은 $1/\gamma$-Lipschitz.

**Step 2**: Contraction lemma (정리 5.8):
$$\mathcal{R}_n(\ell_\gamma \circ \mathcal{H}) \leq \frac{1}{\gamma} \mathcal{R}_n(\mathcal{H}).$$

**Step 3**: 선형 분류기 (정리 5.14):
$$\mathcal{R}_n(\mathcal{H}) \leq \frac{\|w\| \cdot M}{\sqrt{n}},$$
여기서 $M = \max_i \|x_i\|$.

**Step 4**: 조합:
$$\mathcal{R}_n(\ell_\gamma \circ \mathcal{H}) \leq \frac{\|w\| \cdot M}{\gamma \sqrt{n}}.$$

**Step 5**: 정리 5.5로 일반화 경계:
$$\sup_h |L_\mathcal{D}(h) - L_S(h)| \leq \frac{2\|w\| M}{\gamma \sqrt{n}} + O(\sqrt{\log(1/\delta)/n}).$$

**결론**: 
- Margin $\gamma$ 고정 시, $\|w\|$ 작음 → Rademacher ↓ → bound ↓ → 일반화 ↑.
- $\|w\|$ 고정 시, margin $\gamma$ 큼 → Rademacher ↓ → bound ↓ → 일반화 ↑.

따라서 **margin 최대화 = norm 최소화**, 둘 다 일반화를 개선한다. SVM의 목적함수 $\min \|w\|$ s.t. $y_i w^\top x_i \geq 1$은 **실은 Rademacher 복잡도 최소화**와 동등하다! $\square$

</details>

---

<div align="center">

◀ [이전: 04. Massart's Lemma](./04-massart-lemma.md) | [📚 README](../README.md) | [다음: 06. Neural Network의 Rademacher 복잡도 ▶](./06-neural-net-rademacher.md)

</div>
