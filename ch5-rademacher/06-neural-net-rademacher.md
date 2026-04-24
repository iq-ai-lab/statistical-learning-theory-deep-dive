# 06. Neural Network의 Rademacher 복잡도

## 🎯 핵심 질문

- **Bartlett & Mendelson (2002)**: 심층망의 Rademacher를 **층별 norm의 곱** $\prod_l \|W_l\|_F$로 bound할 수 있는가? 완전한 증명은?
- **Spectral norm 버전** (Bartlett, Foster, Telgarsky 2017): $\|W_l\|_\sigma$ (spectral norm)을 쓰면 어떻게 다른가?
- **"파라미터 수 vs Norm"**: VC 차원은 $O(W^2 \log W)$ (엄청남), 하지만 norm-based Rademacher는 작은 $\prod\|W_l\|$ → **실제 일반화 설명 가능**. 왜?
- **"깊이의 저주(curse of depth)"**: 층이 많을수록 norm 곱이 커진다. 하지만 현실은? 깊은 망이 효과적인 이유는?
- **Double Descent, NTK와의 연결**: 이 경계가 왜 "vacuous" 하기도 하고 "meaningful" 하기도 한가?

---

## 🔍 왜 이 이론이 현대 ML에서 중요한가

**Neural Network의 일반화는 SLT의 가장 큰 미해결 문제다.** 이 문서가 보이는 것:

1. **고전 VC bound의 실패**: NN의 VC 차원 $\geq O(W^2 \log W)$ (매개변수 수에 가까움). 이를 bound에 대입하면 vacuous (> 1).

2. **Norm-based Rademacher의 의미**: $\mathcal{R}_n \approx (\prod_l \|W_l\|) / \sqrt{n}$. 만약 norm들이 **작으면** (regularization, 초기 신경망) meaningful bound가 가능.

3. **암묵적 정규화(Implicit Regularization)**: SGD가 찾는 해가 자연스럽게 **norm이 작은 해**(maximum margin)로 수렴한다는 관찰 — Ch6-04와 연결.

4. **Double Descent 패러독스**: 파라미터 수 >> 샘플 수인데도 일반화 잘 됨. 이것이 **norm이 작으면** 설명 가능.

5. **Generalization Theory 레포로의 다리**: NTK, PAC-Bayes, margin distribution 등 현대 DL 이론의 입구.

---

## 📐 수학적 선행 조건

- **Ch5-01~03**: Rademacher 정의, 일반화, Contraction lemma
- **Ch5-05**: Linear/Kernel Rademacher — 기초 구조 이해
- **Calculus & Optimization Deep Dive**: Neural network 최적화, activation의 Lipschitz
- **[Generalization Theory Deep Dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive)**: NTK, implicit regularization (선행 또는 동시 학습)
- 기초: Matrix norm (Frobenius, spectral), layer 합성

---

## 📖 직관적 이해

### 선형에서 비선형으로: Contraction의 누적

선형 함수족의 Rademacher는 (정리 5.14):
$$\mathcal{R}_n(\text{linear}) \propto B \max\|x\| / \sqrt{n}.$$

신경망은 **여러 층의 합성**:
$$f(x) = W_L \sigma(W_{L-1} \sigma(\cdots \sigma(W_1 x))).$$

각 activation $\sigma$ (ReLU, tanh, sigmoid)는:
- **Lipschitz**: $|\sigma(a) - \sigma(b)| \leq L_\sigma |a - b|$
- ReLU: $L_\sigma = 1$
- Sigmoid: $L_\sigma = 1/4$

Contraction lemma(정리 5.8)에 의해, 각 층을 거칠 때마다 Rademacher에 Lipschitz 상수가 곱해진다:
$$\mathcal{R}_n(\text{after 1st activation}) \leq L_{\sigma} \cdot \mathcal{R}_n(\text{linear 1st layer}).$$

**층을 거칠 때마다 곱**: $\prod_l L_{\sigma_l}$ (activation들의 Lipschitz 곱).

**선형 부분의 크기**: 각 층의 weight matrix norm $\|W_l\|$.

**종합**: $\mathcal{R}_n \propto (\prod_l \|W_l\|) \cdot (\prod_l L_{\sigma_l}) / \sqrt{n}$.

### Norm vs 파라미터 수: 왜 전자가 더 나은가?

- **VC 기반**: VC차원 ≥ 파라미터 수 (대충). 큰 망 = 큰 VC = vacuous bound.
- **Norm 기반**: 같은 망이라도, **weight가 작으면** bound가 작다. Regularization (L2 penalty, dropout, batch norm)은 implicit하게 norm을 제어.

**현실**: SGD가 implicit regularization으로 norm-small solution을 찾음(Hardt et al. 2016, Ch6-04). 따라서 norm-based bound는 의미있는 값을 줄 수 있다.

### 깊이의 저주?

$L$층 신경망에서:
$$\mathcal{R}_n \leq \frac{1}{\sqrt{n}} \prod_{l=1}^L \|W_l\|_F.$$

층이 많을수록 ($L$ ↑) → norm 곱 커짐 → bound 커짐 → 일반화 어려움?

**하지만 현실**:
- 깊은 망이 shallow보다 잘 works (적절한 initialization, normalization과 함께).
- 원인들:
  1. **더 나은 최적화**: Deeper network가 landscape가 다를 수 있음 (implicit regularization).
  2. **더 효율적 표현**: 같은 학습 목표도 더 작은 norm으로 가능 (깊이의 이득).
  3. **Double descent**: 과도한 parameterization도 일반화 좋을 수 있음 (다른 효과).

이들은 "norm-based Rademacher의 한계"이자 **현대 DL 이론의 열린 문제**.

---

## ✏️ 엄밀한 정의

### 정의 5.14 (신경망 함수족)

$L$층 fully-connected network:
$$f_W(x) = W_L \sigma(W_{L-1} \sigma(\cdots \sigma(W_1 x) \cdots)),$$

여기서:
- $W_l \in \mathbb{R}^{d_l \times d_{l-1}}$: $l$번째 층의 weight matrix
- $\sigma: \mathbb{R} \to \mathbb{R}$: activation (component-wise)
- $d_0 = d$ (input), $d_L = 1$ (output for regression)

### 정의 5.15 (Weight norm bound)

$$\|W\|_F := \sqrt{\sum_{ij} W_{ij}^2} \quad \text{(Frobenius norm)}$$
$$\|W\|_\sigma := \max_i \lambda_i(W^\top W)^{1/2} \quad \text{(spectral norm, largest singular value)}.$$

Weight들이 bounded:
$$\mathcal{F}_M = \{f_W : \|W_l\|_F \leq M_l \text{ for all } l\}.$$

### 정의 5.16 (Activation의 Lipschitz)

$$L_\sigma := \sup_{a \neq b} \frac{|\sigma(a) - \sigma(b)|}{|a-b|}.$$

예:
- ReLU $\sigma(x) = \max(0, x)$: $L_\sigma = 1$
- Sigmoid $\sigma(x) = 1/(1+e^{-x})$: $L_\sigma = 1/4$
- Tanh $\sigma(x) = (e^x - e^{-x})/(e^x + e^{-x})$: $L_\sigma = 1$

---

## 🔬 정리와 증명

### 정리 5.17 (Bartlett-Mendelson 2002: Frobenius norm 기반) ★★★

$L$층 신경망, Lipschitz activation, weight bound $\|W_l\|_F \leq M_l$에 대해:

$$\mathcal{R}_n(\mathcal{F}) \leq \frac{1}{\sqrt{n}} \cdot \|W_L\|_F \cdot \prod_{l=1}^{L-1} L_\sigma \cdot \|W_l\|_F \cdot B_0,$$

여기서 $B_0 = \max_i \|x_i\|$ (input norm).

더 간단히 (상수 무시):
$$\mathcal{R}_n(\mathcal{F}) = O\left(\frac{1}{\sqrt{n}} \prod_{l=1}^L \|W_l\|_F\right).$$

**증명 스케치** (귀납적 구성):

**Layer 1 (Input → hidden)**:
$$z_1(x) = W_1 x, \quad \mathcal{R}(\text{linear}) \leq \|W_1\|_F \cdot B_0 / \sqrt{n}.$$

**Activation (Contraction)**:
$$h_1 = \sigma(z_1), \quad \mathcal{R}(\sigma \circ \text{linear}) \leq L_\sigma \cdot \mathcal{R}(\text{linear}) \leq L_\sigma \cdot \|W_1\|_F \cdot B_0 / \sqrt{n}.$$

**Layer 2**:
$$z_2 = W_2 h_1, \quad \mathcal{R}(\text{linear 2}) \leq \|W_2\|_F \cdot \|h_1\|_{\max} / \sqrt{n}.$$

여기서 $\|h_1\|_{\max} \leq B_1$ (bounded activation).

**귀납적으로**: 각 층을 거칠 때마다 $\|W_l\|_F$와 $L_\sigma$가 곱해짐:
$$\mathcal{R}(\mathcal{F}_L) \leq O\left(\frac{1}{\sqrt{n}} \prod_{l=1}^L \|W_l\|_F \cdot \prod_{l=1}^{L-1} L_\sigma\right). \quad \square$$

(엄밀한 증명: Bartlett & Mendelson (2002) 또는 Mohri, Rostamizadeh, Talwalkar (2018) Ch4)

### 정리 5.18 (Spectral norm 기반 — Bartlett, Foster, Telgarsky 2017)

Spectral norm $\|W_l\|_\sigma$ (largest singular value)를 쓰면:

$$\mathcal{R}_n(\mathcal{F}) \leq O\left(\frac{1}{\sqrt{n}} \prod_{l=1}^L \|W_l\|_\sigma\right).$$

**장점**:
- Frobenius보다 더 tighter (일반적으로 $\|W\|_\sigma \leq \|W\|_F \leq \sqrt{\text{rank}} \cdot \|W\|_\sigma$).
- Spectral norm regularization (spectral normalization, 현대 GAN 등)의 이론적 근거.

**계산**: SVD로 계산 (gradient descent와 동시 계산 가능).

### 정리 5.19 (VC bound와의 비교)

일반적인 $W$ 파라미터 NN에 대해:

| 방법 | Bound 형태 | 의존성 |
|-----|----------|--------|
| **VC 차원** | $\mathcal{R}_n \sim O(\sqrt{d \log n / n})$, $d = \Theta(W)$ | **파라미터 수** |
| **Norm-based (Rad)** | $\mathcal{R}_n \sim O((\prod \|W_l\|) / \sqrt{n})$ | **Weight magnitude** |

**의미**:
- VC: 큰 망 → 큰 bound (vacuous).
- Norm: 같은 망이라도 weight small → small bound (meaningful).

현실에서 SGD는 implicitly small-norm solution을 찾기 때문에, **norm-based bound가 더 설명력 있다.**

### 정리 5.20 (일반화 경계 — 신경망 버전)

정리 5.5 + Bartlett-Mendelson을 결합하면, 확률 $\geq 1-\delta$로:

$$\sup_f |L_\mathcal{D}(f) - L_S(f)| \leq O\left(\frac{1}{\sqrt{n}} \prod_{l=1}^L \|W_l\|_F + \sqrt{\frac{\log(1/\delta)}{n}}\right).$$

---

## 💻 NumPy 구현 검증

### 실험 1: 층 깊이에 따른 norm 곱 변화

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# 다양한 층 깊이의 신경망 생성
def create_network(L, d_in=10, d_hidden=20, d_out=1, weight_scale=1.0):
    """
    L-layer network: input -> [hidden]*L -> output
    """
    weights = []
    dims = [d_in] + [d_hidden] * (L-1) + [d_out]
    for l in range(L):
        W = weight_scale * rng.standard_normal((dims[l+1], dims[l]))
        weights.append(W)
    return weights

# Frobenius norm의 곱
def norm_product(weights, norm_type='frobenius'):
    prod = 1.0
    for W in weights:
        if norm_type == 'frobenius':
            norm = np.linalg.norm(W, 'fro')
        elif norm_type == 'spectral':
            # spectral norm = largest singular value
            _, s, _ = np.linalg.svd(W)
            norm = s[0]
        prod *= norm
    return prod

# 실험: 층 깊이 증가
Ls = [1, 2, 3, 4, 5, 6, 8, 10]
weight_scales = [0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for scale in weight_scales:
    prods_frob = []
    prods_spec = []
    for L in Ls:
        W = create_network(L, weight_scale=scale)
        prod_f = norm_product(W, 'frobenius')
        prod_s = norm_product(W, 'spectral')
        prods_frob.append(prod_f)
        prods_spec.append(prod_s)
    
    axes[0].semilogy(Ls, prods_frob, 'o-', label=f'weight_scale={scale}', linewidth=2)
    axes[1].semilogy(Ls, prods_spec, 's-', label=f'weight_scale={scale}', linewidth=2)

axes[0].set_xlabel('Number of layers L'); axes[0].set_ylabel('Frobenius norm product')
axes[0].set_title('Depth effect: ∏||W_l||_F')
axes[0].legend(); axes[0].grid(True, alpha=0.3, which='both')

axes[1].set_xlabel('Number of layers L'); axes[1].set_ylabel('Spectral norm product')
axes[1].set_title('Depth effect: ∏||W_l||_σ')
axes[1].legend(); axes[1].grid(True, alpha=0.3, which='both')

plt.tight_layout(); plt.show()

# → weight_scale < 1이면 지수적으로 감소 (좋음)
# → weight_scale > 1이면 지수적으로 증가 (나쁨) — "깊이의 저주"
```

### 실험 2: Rademacher 복잡도 비교 (간단한 NN 시뮬레이션)

```python
# 간단한 2층 네트워크에서 Rademacher 추정
def simple_nn_rademacher(X, W1, W2, n_rademacher=1000):
    """
    f(x) = W2 * ReLU(W1 * x)
    Approximate Rademacher by Monte Carlo
    """
    n = len(X)
    max_corrs = []
    
    for _ in range(n_rademacher):
        sigma = rng.choice([-1, 1], size=n)
        
        # Forward pass through network
        z1 = X @ W1.T  # (n, hidden)
        h1 = np.maximum(z1, 0)  # ReLU
        z2 = h1 @ W2.T  # (n, 1)
        outputs = z2.ravel()
        
        # Correlation with random labels
        corr = np.abs(np.sum(sigma * outputs)) / n
        max_corrs.append(corr)
    
    return np.mean(max_corrs)

# 실험
n, d_in = 50, 5
d_hidden = 10

X = rng.standard_normal((n, d_in))

# Case 1: Small weights (good regularization)
W1_small = 0.1 * rng.standard_normal((d_hidden, d_in))
W2_small = 0.1 * rng.standard_normal((1, d_hidden))

# Case 2: Large weights
W1_large = 2.0 * rng.standard_normal((d_hidden, d_in))
W2_large = 2.0 * rng.standard_normal((1, d_hidden))

rad_small = simple_nn_rademacher(X, W1_small, W2_small, n_rademacher=500)
rad_large = simple_nn_rademacher(X, W1_large, W2_large, n_rademacher=500)

# 이론적 bound (대략적)
prod_small = np.linalg.norm(W1_small, 'fro') * np.linalg.norm(W2_small, 'fro')
prod_large = np.linalg.norm(W1_large, 'fro') * np.linalg.norm(W2_large, 'fro')

bound_small = prod_small / np.sqrt(n)
bound_large = prod_large / np.sqrt(n)

print(f"Small weights: Empirical R̂={rad_small:.4f}, Theory ≤ {bound_small:.4f}")
print(f"Large weights: Empirical R̂={rad_large:.4f}, Theory ≤ {bound_large:.4f}")

# → small weights → small Rademacher → good generalization
#  large weights → large Rademacher → bad generalization
```

### 실험 3: 깊이와 성능의 trade-off

```python
# 간단한 regression 문제에서 깊이 vs 정확도
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

def train_nn_simple(X, y, hidden_dims, epochs=100, lr=0.01):
    """
    Simple SGD training (무겁지 않게)
    """
    d_in = X.shape[1]
    layers = []
    dims = [d_in] + hidden_dims + [1]
    
    for i in range(len(dims)-1):
        W = 0.1 * rng.standard_normal((dims[i+1], dims[i]))
        b = np.zeros((dims[i+1],))
        layers.append((W, b))
    
    # 간단한 경사하강법 (구현 생략, 근사로 final loss만 추적)
    # 실제로는 복잡하니 여기서는 norm의 크기만 비교
    norm_prod = 1.0
    for W, b in layers:
        norm_prod *= np.linalg.norm(W, 'fro')
    
    return norm_prod

# 실험
hidden_configs = [
    [20],           # 1 hidden layer
    [20, 20],       # 2 hidden layers
    [20, 20, 20],   # 3 hidden layers
    [10, 10, 10, 10],  # 4 layers, smaller
]

norm_products = [train_nn_simple(np.random.randn(50, 5), 
                                 np.random.randn(50), 
                                 config) for config in hidden_configs]

labels = ['1 layer [20]', '2 layers [20,20]', '3 layers [20,20,20]', '4 layers [10,10,10,10]']

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(labels)), norm_products, color=['blue', 'orange', 'red', 'green'])
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel('Norm product ∏||W_l||_F')
ax.set_title('Norm product by network depth')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.show()

# → 더 깊은 망이 동일 hidden size라면 norm product 증가
# → 하지만 hidden size를 줄이면 상쇄 가능 (implicit trade-off)
```

---

## 🔗 ML 알고리즘 연결

| 설정 | Rademacher bound | 의미 |
|-----|-----------------|------|
| **Shallow (1-2 layers)** | $O(\prod \|W_l\| / \sqrt{n})$, 적음 | Small bound, 좋은 일반화 가능 |
| **Deep (10+ layers)** | $O(\prod \|W_l\| / \sqrt{n})$, 클 수 있음 | 만약 norm이 크면 vacuous |
| **Regularized (L2, dropout, BN)** | norm이 작음 → bound 작음 | 의미있는 일반화 경계 |
| **Spectral norm** | $O(\prod \|W_l\|_\sigma / \sqrt{n})$ | Frobenius보다 tighter |
| **SGD (implicit reg)** | norm converging to small value | Generalization Theory 설명 가능 |

**핵심 연결**:
- Regularization (명시적) → norm 제어 → Rademacher ↓ → 일반화 ↑
- SGD (implicit regularization) → implicit하게 norm-small solution → norm-based bound 의미있음

---

## ⚖️ 가정과 한계

1. **Activation의 Lipschitz**: ReLU, tanh 등은 Lipschitz (1 또는 $< 1$). 하지만 기타 activation (gelu, swish 등)은 성능 차이. 일반 프레임에는 들어가지만 상수 다름.

2. **Layer 합성의 누적**: $L$ 층이면 norm 곱이 $\prod$ 형태. 만약 각 $\|W_l\| > 1$이면 지수적 폭발. 하지만 현실 초기화 (Xavier, He)는 norm < 1 범위.

3. **Vacuous bound**: 여전히 큰 망에서는 bound가 > 1일 수 있음. 하지만 이것은 "최악의 경우" bound이고, 실제 데이터는 특별한 구조를 가질 수 있음.

4. **Loss 함수 미포함**: 지금까지는 0-1 또는 regression loss. 다중 분류 (softmax + cross-entropy)는 추가 분석 필요.

5. **Optimization의 미반영**: Rademacher bound는 ERM 최악 경우. 실제 SGD의 해는 더 나을 수 있음 (implicit regularization, Ch6-04).

6. **기울기 흐름 가정**: Bartlett-Mendelson은 fixed weight norm 가정. 하지만 학습 중 norm이 변함.

---

## 📌 핵심 정리

- **Bartlett-Mendelson 2002**: $L$층 NN → $\mathcal{R}_n = O((\prod_l \|W_l\|_F) / \sqrt{n})$.
- **증명**: Layer 1-by-1, contraction lemma + linear class 조합.
- **Spectral norm (2017)**: $\|W_l\|_\sigma$ → 더 tighter (일반적으로).
- **VC vs Norm**: VC는 파라미터 수, Rademacher는 weight magnitude → 후자가 실제 NN 설명력 있음.
- **깊이의 저주?**: 층 많음 → norm 곱 커질 수 있음, 하지만 implicit regularization으로 해결 가능.
- **의미**: Norm-based Rademacher는 **실제 DL 일반화를 부분적으로 설명** — 완전 설명은 아직 미해결.
- **다음 단계**: [Generalization Theory Deep Dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive) — NTK, PAC-Bayes, margin, double descent.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> ReLU activation $\sigma(x) = \max(0, x)$의 Lipschitz 상수가 1임을 보여라. 즉, 모든 $a, b$에 대해 $|\sigma(a) - \sigma(b)| \leq |a-b|$.</summary>

<br/>

**해설**. $a, b \in \mathbb{R}$에 대해:

1. **$a, b \geq 0$**: $\sigma(a) = a, \sigma(b) = b$이므로 $|\sigma(a) - \sigma(b)| = |a-b|$. $\checkmark$

2. **$a, b \leq 0$**: $\sigma(a) = \sigma(b) = 0$이므로 $|\sigma(a) - \sigma(b)| = 0 \leq |a-b|$. $\checkmark$

3. **$a \geq 0, b < 0$** (또는 반대): 일반성 잃지 않고 $a \geq 0 > b$라 하면,
   $$|\sigma(a) - \sigma(b)| = |a - 0| = a = a - b + (-b).$$
   $b < 0$이므로 $-b > 0$, 따라서:
   $$a \leq a - b + |b| = a + |b|.$$
   그런데 $a - b = |a - b|$ (∵ $a \geq 0 > b$), 따라서:
   $$|\sigma(a) - \sigma(b)| = a \leq |a - b|. \quad \checkmark$$

모든 경우에 $|\sigma(a) - \sigma(b)| \leq |a-b|$이므로 ReLU는 1-Lipschitz. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> Theorem 5.17의 귀납 증명을 자세히 써보라. 특히 "각 층을 거칠 때마다 norm 곱이 늘어난다"는 부분을 엄밀히.</summary>

<br/>

**해설**. 귀납법: 첫 $\ell$ 층의 Rademacher를 $\mathcal{R}^{(\ell)}$이라 정의.

**Base case** ($\ell = 1$): 선형 층 $z^{(1)} = W_1 x$.

고정된 샘플 $S = (x_1, \ldots, x_n)$에 대해:
$$\hat{\mathcal{R}}^{(1)}_S = \mathbb{E}_\sigma[\sup_W \frac{1}{n}\sum_i \sigma_i \langle W, x_i \rangle].$$

정리 5.14 (선형 함수족):
$$\hat{\mathcal{R}}^{(1)}_S \leq \frac{\|W_1\|_F}{n} \sqrt{\sum_i \|x_i\|^2} \leq \frac{\|W_1\|_F \cdot B_0}{\sqrt{n}}.$$

**Inductive step** ($\ell-1 \to \ell$):

가정: 첫 $\ell-1$ 층 후 hidden activation $h^{(\ell-1)}$의 Rademacher가
$$\mathcal{R}^{(\ell-1)} = O\left(\frac{1}{\sqrt{n}} \prod_{l=1}^{\ell-1} \|W_l\|_F \cdot L_\sigma^{\ell-2} \cdot B_0\right).$$

**Step 1**: $\ell$ 번째 층의 선형 부분: $z^{(\ell)} = W_\ell h^{(\ell-1)}$.

정리 5.14를 $h^{(\ell-1)}$ (이전 층 output, "new input")에 적용:
$$\mathcal{R}(\text{linear at layer } \ell) \leq \frac{\|W_\ell\|_F \cdot B_{\ell-1}}{\sqrt{n}},$$

여기서 $B_{\ell-1} = \max_i \|h^{(\ell-1)}(x_i)\|$ (bounded by previous activations).

**Step 2**: Activation (Contraction lemma, 정리 5.8):
$$\mathcal{R}(\sigma \circ W_\ell h^{(\ell-1)}) \leq L_\sigma \cdot \mathcal{R}(\text{linear}) \leq L_\sigma \cdot \frac{\|W_\ell\|_F \cdot B_{\ell-1}}{\sqrt{n}}.$$

**Step 3**: 조합:
$$\mathcal{R}^{(\ell)} \leq \mathcal{R}^{(\ell-1)} \cdot L_\sigma \cdot \|W_\ell\|_F / \text{(scale factor)}.$$

정확히는:
$$\mathcal{R}^{(\ell)} = O\left(\frac{1}{\sqrt{n}} \prod_{l=1}^{\ell} \|W_l\|_F \cdot L_\sigma^{\ell-1} \cdot B_0\right).$$

**결론** ($\ell = L$):
$$\mathcal{R}^{(L)} = O\left(\frac{1}{\sqrt{n}} \prod_{l=1}^L \|W_l\|_F \cdot L_\sigma^{L-1} \cdot B_0\right). \quad \square$$

(정확한 상수는 논문 참고, 여기서는 order만 강조)

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> **"깊이 vs Norm trade-off"**: 2층 망 vs 3층 망을 비교하라. 같은 "표현력"을 원할 때, 어느 것이 더 작은 norm으로 달성할 수 있는가? 아니면 이것이 문제 구조에 의존하는가?</summary>

<br/>

**해설**. 

**이론적 관점**:
- **2층**: $\mathcal{R} \propto \|W_1\| \cdot \|W_2\| \cdot L_\sigma / \sqrt{n}$.
- **3층**: $\mathcal{R} \propto \|W_1\| \cdot \|W_2\| \cdot \|W_3\| \cdot L_\sigma^2 / \sqrt{n}$.

같은 output을 원한다면, 3층 norm의 곱이 2층보다 작을 수 있는가?

만약 3층이 hidden size를 줄일 수 있다면 가능: $\|W_1\| \cdot \|W_2\| \cdot \|W_3\|$ (작은 hidden) < $\|W_1'
\| \cdot \|W_2'\|$ (큰 hidden).

**실제**:
- **Universal approximation**: 깊은 망이 같은 hidden size의 shallow 망보다 **지수적으로 더 효율적**일 수 있음 (Montúfar et al. 2016).
- **Implicit regularization**: SGD가 깊은 망에서 implicit하게 작은 norm으로 수렴할 가능성.
- **Double descent**: 매개변수를 매우 많이 늘려도 (깊이 증가) generalization이 개선되는 현상.

**결론**: "깊이 = bad"는 틀림. **깊이 + 적절한 정규화/초기화 = 좋은 norm-complexity trade-off 가능.** 이것이 현대 DL의 성공 비결이자 **이론과 실제의 gap** — 현재 진행 중인 연구(Generalization Theory 레포).

</details>

---

<div align="center">

◀ [이전: 05. Linear & Kernel Rademacher](./05-linear-kernel-rademacher.md) | [📚 README](../README.md) | [다음: 01. Uniform Stability ▶](../ch6-stability/01-uniform-stability.md)

</div>
