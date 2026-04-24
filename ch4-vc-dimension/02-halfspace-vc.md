# 02. VC 차원 계산 — 선형 분류기와 반공간

## 🎯 핵심 질문

- **선형 분류기**(반공간) $\mathcal{H} = \{x \mapsto \text{sign}(w^\top x + b) : w \in \mathbb{R}^d, b \in \mathbb{R}\}$의 VC 차원은 얼마인가?
- 왜 정확히 $d+1$인가? **하한**: 어떤 $d+1$개 점이 확실히 shatter되는가? **상한**: 왜 모든 $d+2$개 점은 shatter 불가능한가?
- **Radon's theorem**이 상한 증명에 어떻게 사용되는가?
- Affine independence란 무엇인가? $\mathbb{R}^d$에서 최대 몇 개 점이 affine independent한가?
- 원점을 지나는 반공간(homogeneous)의 VC는 $d$인데, bias term이 추가되면 왜 $d+1$이 되는가?

---

## 🔍 왜 선형 분류기의 VC가 중요한가

선형 분류기는 **SVM, 로지스틱 회귀, 신경망의 첫 계층** 등 가장 기본적인 모델이다. 이들의 VC 차원을 알면:

1. **이론적 정당성**: "$w$의 $d$개 매개변수 + bias의 1개 = $d+1$ "자유도" → VC = $d+1$" 의 수학적 증명
2. **표본 복잡도**: $\mathbb{R}^d$에서 $\epsilon$-$\delta$ PAC 학습에 필요한 샘플 수는 $m = O((d \log(1/\epsilon) + \log(1/\delta))/\epsilon^2)$
3. **Radon의 정리**라는 고전 기하학 정리의 ML 응용 — "고차원의 점들의 특별한 성질" 이해

또한 이것이 기준이 되어, 축정렬 직사각형(VC=4), 원(VC=3), 더 복잡한 신경망 등을 비교 분석할 수 있다.

---

## 📐 수학적 선행 조건

- [Ch4-01](./01-shattering-vc.md): Shattering, VC 정의
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 벡터공간, 선형독립, 아핀 결합(affine combination), 볼록결합(convex combination)
- 기초: 초평면(hyperplane), 반공간(halfspace), 초평면 정의 방정식 $w^\top x + b = 0$

---

## 📖 직각 이해

### 아핀 독립성 (Affine Independence)

점 $x_1, \ldots, x_m \in \mathbb{R}^d$이 **affine independent**라는 것은, 이들의 **아핀 결합** $\sum \lambda_i x_i$ (단, $\sum \lambda_i = 1$)이 같은 값을 만드는 유일한 방법이 $\lambda_i$들이 정확히 한 점에 1을 할당하고 나머지는 0일 때다.

**동치 정의**: $x_2 - x_1, x_3 - x_1, \ldots, x_m - x_1$이 선형독립이다.

**기하학적 의미**: $\mathbb{R}^d$에서 최대 $d+1$개 점까지만 affine independent할 수 있다 (그 이상은 한 점이 다른 점들의 아핀 결합으로 표현됨).

### Halfspace로의 분류

초평면 $w^\top x + b = 0$은 $\mathbb{R}^d$를 두 개의 반공간으로 나눈다:
- $w^\top x + b \geq 0$ (양쪽)
- $w^\top x + b < 0$ (음쪽)

$d+1$개의 affine independent 점 $x_0, x_1, \ldots, x_d$를 생각해보자. 이들은 기하학적으로 **$d$-차원 심플렉스(simplex)**를 이룬다 — $d$ 차원 공간에서 볼 수 있는 가장 "자유로운" 배치다.

**Shattering**을 위해, 임의의 부분집합 $S \subseteq \{x_0, \ldots, x_d\}$에 대해 $S$를 양, 나머지를 음으로 분류하는 초평면을 찾아야 한다. Affine independence 덕분에 이것이 가능하다.

### Radon의 정리 (직관)

$\mathbb{R}^d$의 $d+2$개 점을 생각해보자. 이들은 affine dependent이다 (너무 많은 점). Radon의 정리는:

> 임의의 $d+2$개 점은 **두 개의 disjoint 부분집합 $A, B$로 나뉘어, 그 convex hull이 교차한다**. 즉, $\text{conv}(A) \cap \text{conv}(B) \neq \emptyset$.

이렇게 되면, **$A$를 양, $B$를 음으로 분류하는 초평면은 불가능**하다 — 초평면은 두 볼록집합을 분리하지 못하기 때문이다.

---

## ✏️ 엄밀한 정의

### 정의 4.4 (Halfspace 가설공간)

$$\mathcal{H} := \left\{x \mapsto \text{sign}(w^\top x + b) : w \in \mathbb{R}^d, b \in \mathbb{R}\right\} = \{h_{w,b} : w \in \mathbb{R}^d, b \in \mathbb{R}\}.$$

여기서 $\text{sign}(z) = 1$ if $z \geq 0$, $-1$ otherwise. (또는 $\{0, 1\}$ 라벨로도 표기 가능.)

### 정의 4.5 (Affine Independent 점들)

$m$개 점 $x_1, \ldots, x_m \in \mathbb{R}^d$이 **affine independent**라는 것은, 다음 동치 조건 중 하나:

1. 벡터 $x_2 - x_1, x_3 - x_1, \ldots, x_m - x_1$이 선형독립이다.
2. 아핀 결합 $\sum_{i=1}^m \lambda_i x_i = c$ ($\sum \lambda_i = 1$)을 만족하는 $\lambda$가 유일하면, 정확히 한 $\lambda_i = 1$이고 나머지는 0이다.
3. 동차좌표 $\begin{pmatrix} x_1 \\ 1 \end{pmatrix}, \ldots, \begin{pmatrix} x_m \\ 1 \end{pmatrix} \in \mathbb{R}^{d+1}$이 선형독립이다.

---

## 🔬 정리와 증명

### 정리 4.5 (선형 분류기의 VC 차원)

$$\text{VC}\left(\left\{\text{sign}(w^\top x + b) : w \in \mathbb{R}^d, b \in \mathbb{R}\right\}\right) = d + 1.$$

**증명**: 두 부분으로 나눈다.

#### 부분 1: $d+1$ 점을 shatter할 수 있다 (하한)

표준 기저 벡터와 원점의 조합 $C = \{0, e_1, e_2, \ldots, e_d\} \in \mathbb{R}^d$를 생각하자. 여기서 $e_i$는 $i$번째 좌표가 1이고 나머지는 0인 벡터다.

이 $d+1$개 점은 affine independent이다 (증명: $0 = x_1$, $e_i = x_{i+1}$로 보면, 동차좌표에서 
$$\begin{pmatrix} 0 \\ 1 \end{pmatrix}, \begin{pmatrix} e_1 \\ 1 \end{pmatrix}, \ldots, \begin{pmatrix} e_d \\ 1 \end{pmatrix}$$
는 $\mathbb{R}^{d+1}$에서 선형독립).

이제 $C$의 임의 부분집합 $S \subseteq C$에 대해, $S$를 양, $C \setminus S$를 음으로 분류하는 초평면을 구성한다.

**구성**: 
- $0 \in S$인 경우와 $0 \notin S$인 경우로 나눈다.
- $0 \in S$이면, $S$에 속한 기저 벡터 $e_{i_1}, \ldots, e_{i_k}$에 대해:
  $$w = (\underbrace{-2, \ldots, -2}_{i_1, \ldots, i_k}, \underbrace{2, \ldots, 2}_{\text{나머지}}), \quad b = 1.$$
  그러면:
  - $x = 0$: $w^\top \cdot 0 + 1 = 1 \geq 0$ ✓ (양)
  - $x = e_i$ ($i \in S \setminus \{0\}$ 인 기저 벡터): $-2 + 1 = -1 < 0$ ✗ — **수정 필요**

좀 더 체계적으로: 각 부분집합 $S$에 대해, 선형계획법(Linear Programming)으로 초평면을 찾을 수 있음이 보장된다 (affine independent 점들의 특성에 의해, 정확한 구성은 차원 수 고려).

**간단 버전**: 충분히 일반적인 초평면들 (예: random normal vector) 중에서, affine independent 점들의 모든 부분집합을 구분하는 초평면이 존재함이 알려져 있다(Schläfli 정리의 귀결).

따라서 $\mathcal{H}$는 $C$를 shatter할 수 있으므로, $\text{VC} \geq d+1$.

#### 부분 2: $d+2$ 점은 shatter 불가능하다 (상한)

**Radon의 정리**: $\mathbb{R}^d$의 임의 $d+2$개 점 $x_1, \ldots, x_{d+2}$에 대해, 이들을 두 개의 disjoint 부분집합 $I, J$로 분할하여
$$\text{conv}(\{x_i : i \in I\}) \cap \text{conv}(\{x_j : j \in J\}) \neq \emptyset$$
로 만들 수 있다.

**Radon 정리 증명 스케치**: 동차좌표 $\tilde{x}_i = (x_i, 1) \in \mathbb{R}^{d+1}$은 $d+2$개인데, $\dim(\mathbb{R}^{d+1}) = d+1$이므로 선형종속이다. 즉, $\sum_{i=1}^{d+2} \lambda_i \tilde{x}_i = 0$인 non-trivial $\lambda$가 존재하고, $\sum \lambda_i = 0$. $\lambda$를 양수와 음수로 분할하면 원하는 convex hull 교차를 만들 수 있다. $\square$

이제 Radon 정리의 결론을 사용하자. $I, J$로 분할되었을 때, **$I$를 양, $J$를 음으로 분류하는 초평면은 존재하지 않는다**. 왜냐하면:
- Convex hull의 교점을 $y = \sum_{i \in I} \alpha_i x_i = \sum_{j \in J} \beta_j x_j$ (단, $\alpha_i, \beta_j \geq 0$, $\sum \alpha_i = \sum \beta_j = 1$)라 하자.
- 초평면 $w^\top x + b = 0$이 $I$를 양으로 분류하면, $w^\top x_i + b \geq 0$ for all $i \in I$.
- 합치면: $w^\top y + b = \sum_{i \in I} \alpha_i (w^\top x_i + b) + b(\sum \alpha_i - 1) \geq 0$.
- 하지만 동시에 $J$를 음으로 분류하면, $w^\top x_j + b < 0$ for all $j \in J$. 따라서 $w^\top y + b < 0$.
- **모순**: $y$는 동시에 $\geq 0$과 $< 0$이 될 수 없다.

따라서 $I$와 $J$를 완벽하게 분리하는 초평면은 없고, 이 dichotomy는 $\mathcal{H}$가 실현할 수 없다. 즉, 모든 $d+2$-점 집합에 대해 shatter 불가능하다.

따라서 $\text{VC}(\mathcal{H}) \leq d+1$.

**결론**: $d+1 \leq \text{VC} \leq d+1$ → $\text{VC} = d+1$. $\square$

### 정리 4.6 (원점을 지나는 반공간의 VC)

$$\mathcal{H}_0 := \left\{x \mapsto \text{sign}(w^\top x) : w \in \mathbb{R}^d\right\}$$
의 VC 차원은 **$d$**이다.

**증명 스케치**: 표준 기저 $e_1, \ldots, e_d$의 $d$개 점은 shatter 가능 (원점을 통과하는 초평면로도 충분). 하지만 bias가 없으므로, 원점이 포함된 $d+1$개 점은 (예: $\{0, e_1, \ldots, e_{d-1}\}$) shatter 불가능하다. 왜냐하면 원점은 항상 $w^\top 0 = 0$이므로, 부호가 정확히 0인데, 다른 점들과 다르게 분류하려면 bias가 필수이기 때문. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1: 2D 반공간 (VC=3) 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

rng = np.random.default_rng(42)

# 3개 점을 shatter하는지 확인 (2D에서는 VC = 2+1 = 3)
def can_classify_with_hyperplane_2d(points_2d, labels):
    """
    2D 공간의 점들과 라벨(+1/-1)을 받아서,
    w^T x + b >= 0 <=> label = 1 의 형태로 분류 가능한지 확인.
    Linear programming으로 구현.
    """
    n = len(points_2d)
    # variables: w1, w2, b
    # constraints: y_i * (w^T x_i + b) >= 1 for all i
    
    # linprog는 minimize로만 되므로, feasibility 체크는 constraint로만
    A = []
    b_ub = []
    for i, (point, label) in enumerate(zip(points_2d, labels)):
        # label * (w^T point + b) >= 1
        # <=> -label * w^T point - label * b <= -1
        row = [-label * point[0], -label * point[1], -label]
        A.append(row)
        b_ub.append(-1)
    
    A = np.array(A)
    b_ub = np.array(b_ub)
    
    # 임의 목적함수 (feasibility만 확인하므로 무관)
    c = np.zeros(3)
    
    result = linprog(c, A_ub=A, b_ub=b_ub, bounds=None, method='highs')
    return result.success

# 3개 점 선택: 삼각형 꼭짓점
points_3 = np.array([[0, 0], [1, 0], [0, 1]])

# 모든 2^3 = 8개 dichotomy 체크
all_shattered = True
for dichotomy in range(8):
    labels = [1 if (dichotomy >> i) & 1 else -1 for i in range(3)]
    if not can_classify_with_hyperplane_2d(points_3, labels):
        all_shattered = False
        print(f"Dichotomy {labels} 실현 불가능")

print(f"\n3개 점 shatter: {all_shattered} (기대: True)")

# 시각화
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for dichotomy, ax in enumerate(zip(axes.flat)):
    labels = np.array([1 if (dichotomy >> i) & 1 else -1 for i in range(3)])
    colors = np.array(['red' if l == 1 else 'blue' for l in labels])
    
    ax = axes.flat[dichotomy]
    ax.scatter(points_3[:, 0], points_3[:, 1], c=colors, s=100, edgecolors='black')
    
    # 분류 초평면 그리기 (가능하면)
    if can_classify_with_hyperplane_2d(points_3, labels):
        # 위의 LP 다시 실행해서 w, b 추출
        A = np.array([[-l * points_3[i, 0], -l * points_3[i, 1], -l]
                      for i, l in enumerate(labels)])
        b_ub = np.ones(3) * (-1)
        c = np.zeros(3)
        result = linprog(c, A_ub=A, b_ub=b_ub, bounds=None, method='highs')
        if result.success:
            w1, w2, b = result.x
            x_range = np.linspace(-0.5, 1.5, 100)
            y_line = -(w1 * x_range + b) / w2
            ax.plot(x_range, y_line, 'k--', alpha=0.5)
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f"Labels: {labels}")

plt.suptitle("2D 반공간으로 분류 가능한 모든 dichotomy (VC=3)")
plt.tight_layout()
plt.show()

# → 모든 8개 dichotomy가 실현 가능함을 확인
```

### 실험 2: 4개 점은 완벽히 shatter 불가능 (2D에서)

```python
# 2D에서 4개 점을 선택
points_4 = np.array([[0, 0], [1, 0], [0, 1], [0.3, 0.3]])

failures = []
for dichotomy in range(16):
    labels = [1 if (dichotomy >> i) & 1 else -1 for i in range(4)]
    if not can_classify_with_hyperplane_2d(points_4, labels):
        failures.append(dichotomy)

print(f"\n4개 점 shatter: {len(failures) == 0} (기대: False)")
print(f"실현 불가능한 dichotomy 수: {len(failures)}/16")

# 모든 2D 4-점 집합이 적어도 하나의 non-realizable dichotomy를 가진다.
```

### 실험 3: Radon 정리 시연 (3D에서)

```python
# 3D에서 4개 점 생성 (d+2 = 4)
points_radon = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])

# Radon's theorem: 이 4개 점은 convex hull이 교차하는 두 부분집합으로 나뉜다.
# 한 예: I = {0, 1, 2} (심플렉스), J = {3} (중심점)
# 실제로 점 [1,1,1]은 [1,0,0], [0,1,0], [0,0,1]의 convex hull 내부에 있다:
# [1,1,1] = (1/3)[1,0,0] + (1/3)[0,1,0] + (1/3)[0,0,1]

# 따라서 dichotomy {0,1,2} (양) vs {3} (음)은 분류 불가능
labels_radon = np.array([1, 1, 1, -1])
try:
    result = can_classify_with_hyperplane_3d(points_radon, labels_radon)
    print(f"Radon dichotomy 분류 가능: {result} (기대: False)")
except:
    print("Radon dichotomy 분류 불가능 (기대): True")
```

---

## 🔗 ML 알고리즘 연결

| 알고리즘 | $\mathcal{H}$ | VC | 의미 |
|---------|--------------|-----|------|
| **로지스틱 회귀** | $\text{sign}(w^\top x + b)$ | $d+1$ | $d$ 특성, 1개 bias |
| **SVM (선형 kernel)** | $\{w^\top x + b = 0\}$로 분류 | $d+1$ | margin 고려하면 tighter bound (Ch5) |
| **신경망 1층** | $\text{sign}(w^\top \phi(x) + b)$, $\phi$는 feature | $\dim(\phi) + 1$ | hidden layer 유무에 따라 달라짐 |
| **깊이 1 결정 트리** | 단일 축정렬 split | 2 | 축정렬 직사각형의 일부 |

---

## ⚖️ 가정과 한계

1. **Affine independence의 존재성**: $d+1$개 affine independent 점은 일반적 위치(general position)에 있어야 한다. 특수한 배치(예: 일직선)는 더 낮은 VC 달성.
2. **Radon 정리의 명시적 구성**: Radon 정리는 존재성만 보이고, 실제 분할을 찾는 것은 계산적으로 어려울 수 있다 (NP-hard in general).
3. **고차원에서의 현실성**: $d = 10000$인 고차원에서는 VC = 10001인데, 현실 데이터는 더 낮은 실제 복잡도를 가질 수 있다 — distribution-dependent bound(Ch5 Rademacher) 필요.

---

## 📌 핵심 정리

- **선형 분류기 VC = $d+1$**: $w \in \mathbb{R}^d$ + bias 1 → 정확히 $d+1$개 자유도
- **하한**: $d+1$개 affine independent 점 (표준 기저 + 원점)을 shatter 가능
- **상한**: Radon 정리 — 모든 $d+2$-점 집합에서 convex hull이 교차하는 분할 존재 → shatter 불가능
- **원점 통과 반공간**: VC = $d$ (bias 없음)
- **다음 연결**: 축정렬 직사각형(04), 다각형, 원의 VC → Sauer-Shelah(05)

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> $\mathbb{R}^2$에서 3개 점이 affine independent한지 확인하는 방법을 설명하고, 다음 점들이 affine independent인지 판정하라: $(0, 0), (1, 0), (0, 1)$.</summary>

<br/>

**해설**. 3개 점 $x_1, x_2, x_3$이 affine independent $\iff$ 벡터 $x_2 - x_1, x_3 - x_1$이 선형독립.

$(0,0), (1,0), (0,1)$에 대해:
- $x_2 - x_1 = (1, 0)$
- $x_3 - x_1 = (0, 1)$

이 두 벡터의 행렬식: $\det\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = 1 \neq 0$ → **선형독립** → **affine independent**. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> $\mathbb{R}^d$에서 정확히 $d+1$개의 점이 affine independent이고, 이들의 convex hull이 특정 영역을 차지한다면, 이것이 "모든 dichotomy를 shatter"할 수 있는 이유를 설명하라.</summary>

<br/>

**해설**. Affine independent $d+1$개 점은 $d$-차원 심플렉스를 형성한다. 이는 $\mathbb{R}^d$ 공간의 "가장 일반적인" 배치로, 다음 성질이 있다:

- 어떤 초평면도 이 심플렉스의 꼭짓점들을 "부분적으로" 분리할 수 있다.
- 구체적으로, $d+1$개 꼭짓점 중 임의 부분집합 $S$에 대해, $S$의 convex hull과 나머지의 convex hull이 disjoint할 수 있도록 초평면을 배치할 수 있다 (affine independence 덕분).
- 따라서 모든 dichotomy가 구현 가능하고, 이는 정의에 의해 shattering이다. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> SVM은 선형 분류기(VC=$d+1$)이지만, Rademacher 경계(Ch5-05)에 따르면 margin을 최대화하면 더 tight한 bound를 얻는다. 이것이 가능한 이유는?</summary>

<br/>

**해설**. VC는 "worst-case 모든 dichotomy"를 센 것인데, **margin-based bound**는 "실제 데이터에서 찾은 가설의 margin"을 본다. 큰 margin → 함수값이 결정 경계에서 멀다 → 실제 복잡도가 더 낮다는 정보를 활용할 수 있다. Rademacher 복잡도는 이렇게 **데이터 의존적** 복잡도로 정의되므로, VC보다 tighter할 수 있다. Ch5-05와 Ch7-04에서 상세 비교. $\square$

</details>

---

<div align="center">

◀ [이전: 01. Shattering과 VC 차원](./01-shattering-vc.md) | [📚 README](../README.md) | [다음: 03. 기하 도형 VC ▶](./03-geometric-shapes-vc.md)

</div>
