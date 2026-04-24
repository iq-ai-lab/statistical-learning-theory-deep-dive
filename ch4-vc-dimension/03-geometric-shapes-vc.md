# 03. VC 차원 계산 — 기하학적 가설공간

## 🎯 핵심 질문

- **축정렬 직사각형** (axis-aligned rectangle)의 VC는 몇인가? 왜 4인가?
- **회전된 직사각형** (rotated rectangle)의 VC는 왜 5인가? 축정렬과의 차이는?
- **원(disc)** $\{\|x - c\| \leq r\}$의 VC는 3인 이유는?
- **볼록 다각형** ($k$-gon)의 VC = $2k+1$이라는 공식은 어디서 나오는가?
- **무한 VC**: Convex set (무제약)의 VC는 왜 무한인가?

---

## 🔍 왜 기하학적 가설공간이 중요한가

선형 분류기(VC=$d+1$) 이후, 다음으로 자연스러운 질문은 **기하학적 형태**(직사각형, 원, 다각형)의 복잡도다. 이들은:

1. **실전 모델**: 이미지에서 객체 탐지 (bounding box = 직사각형), 클러스터링 (원 기반 거리), 지역 표현 (convex region)
2. **VC의 다양성**: 같은 차원에서도 기하 형태에 따라 VC가 크게 다름 — $d$-차원 공간에서 선형은 VC=$d+1$, 직사각형은 VC=$2d$
3. **composition**: 많은 알고리즘이 이 기본 형태들을 조합

이 문서는 각 형태의 VC를 **heuristic과 정확한 증명**으로 함께 제시한다.

---

## 📐 수학적 선행 조건

- [Ch4-01](./01-shattering-vc.md): Shattering, VC 정의
- [Ch4-02](./02-halfspace-vc.md): 선형 분류기, Radon 정리
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Convex set, convex hull, normal vector
- 기초: 원(circle), 직사각형, 다각형

---

## 📖 직관적 이해

### 축정렬 직사각형 (VC=4)

$\mathbb{R}^2$에서 축정렬 직사각형은 $\mathcal{H} = \{(x, y) \mapsto \mathbb{1}[a \leq x \leq b, c \leq y \leq d]\}$.

**4개 점 shattering**: "다이아몬드" 배치
```
      •(2)
      |
(1)•--+--•(3)
      |
      •(4)
```

이 4개 점은 모두 분류 가능:
- 모두 양: $[x_{\min}-\epsilon, x_{\max}+\epsilon] \times [y_{\min}-\epsilon, y_{\max}+\epsilon]$
- $\{1\}$만: $[x_1-\epsilon, x_1+\epsilon] \times [y_1-\epsilon, y_1+\epsilon]$
- 모든 부분집합: 각각에 대응하는 직사각형 존재

**5개 점 불가능**: "+"-형태의 5개 점을 추가하면, "위쪽과 아래쪽 동시에 양"은 직사각형으로 불가능 — 가운데를 무조건 포함해야 하기 때문.

### 원 (VC=3)

원은 $\mathcal{H} = \{\|x - c\| \leq r\}$ (중심 $c \in \mathbb{R}^2$, 반지름 $r \geq 0$)

**3개 점 shattering**: 정삼각형 배치
```
      •
     / \
    •---•
```

이 3개 점에 대해 모든 부분집합을 구현 가능:
- 중심과 반지름을 조정하면 임의 부분집합을 "원 안"으로 설정 가능

**4개 점 불가능**: 일반적 배치 (예: 정사각형)에서, "대각 쌍을 각각 포함하되 중간 두 점은 제외"는 원으로 불가능 — 한 원이 두 점을 포함하면 그 사이 거리는 한정되어, 다른 부분을 정확히 제외하기 어려움.

---

## ✏️ 엄밀한 정의

### 정의 4.7 (축정렬 직사각형)

$$\mathcal{H}_{\text{rect,aligned}} := \{(x_1, \ldots, x_d) \mapsto \mathbb{1}[\forall j: a_j \leq x_j \leq b_j] : a_j, b_j \in \mathbb{R}, a_j \leq b_j\}.$$

각 차원마다 상한과 하한을 독립적으로 설정.

### 정의 4.8 (회전된 직사각형)

$$\mathcal{H}_{\text{rect,rotated}} := \{x \mapsto \mathbb{1}[\text{$x$가 직사각형 내부}] : \text{rotation, translation 포함}\}.$$

중심과 방향(각도), 크기를 모두 조정 가능.

### 정의 4.9 (원판)

$$\mathcal{H}_{\text{disc}} := \{x \mapsto \mathbb{1}[\|x - c\| \leq r] : c \in \mathbb{R}^d, r \geq 0\}.$$

중심(d개 매개변수)과 반지름(1개) → 총 $d+1$개 매개변수.

### 정의 4.10 (볼록 k-각형)

$$\mathcal{H}_{k\text{-gon}} := \{\text{k-edge convex polygon의 내부}\}.$$

---

## 🔬 정리와 증명

### 정리 4.7 (축정렬 직사각형, 2D)

$$\text{VC}(\mathcal{H}_{\text{rect,aligned}}) = 4 \quad \text{(in } \mathbb{R}^2\text{)}.$$

**증명**:

**하한**: 다이아몬드 4개 점 $\{(1, 0), (0, 1), (-1, 0), (0, -1)\}$을 shatter 가능함을 보인다. 각 부분집합에 대해:
- $\emptyset$: 불가능한 직사각형 (예: $[2, 1]$)
- 단일 점 $\{(1, 0)\}$: $[0.5, 1.5] \times [-0.5, 0.5]$
- 인접 두 점 (예: $\{(1, 0), (0, 1)\}$): $[0, 1] \times [0, 1]$
- 모든 부분집합: 대응하는 직사각형 찾기 가능

각 경우에 대해 명시적으로 $(a, b, c, d)$를 구성할 수 있다.

**상한**: 임의 5개 점을 생각해보자. 

$x$ 좌표의 순서를 $x_{(1)} \leq x_{(2)} \leq x_{(3)} \leq x_{(4)} \leq x_{(5)}$라 하자. 직사각형은 상한 $b$를 하나 정하면, $[a, b]$는 연속 구간. 따라서 직사각형이 포함하는 점들의 $x$ 좌표도 **연속 구간**을 이룬다.

만약 dichotomy가 "점 1과 3은 포함, 점 2는 미포함"이면, 점 1과 3을 포함하는 연속 $x$-구간은 필히 점 2도 포함해야 한다 — **모순**.

따라서 모든 5-점 집합은 shatter 불가능. $\square$

### 정리 4.8 ($d$-차원 축정렬 직사각형)

$$\text{VC}(\mathcal{H}_{\text{rect,aligned}}) = 2d \quad \text{(in } \mathbb{R}^d\text{)}.$$

**증명 스케치**: 각 차원에서 "구간" 2개 (하한과 상한)를 독립적으로 정하므로, 이것은 각 차원의 "threshold" 2개씩의 조합. 1D에서 interval의 VC=2이므로, $d$개 차원에서는 $2d$. (정확한 construction과 upper bound는 각 차원의 독립성 argument로 유도.)

### 정리 4.9 (원판, 2D)

$$\text{VC}(\mathcal{H}_{\text{disc}}) = 3 \quad \text{(in } \mathbb{R}^2\text{)}.$$

**증명**:

**하한**: 정삼각형 $\{(0, 0), (1, 0), (1/2, \sqrt{3}/2)\}$을 shatter 가능. 중심과 반지름을 조정하면:
- 모두 양: 큰 원
- 단일 점: 작은 원
- 임의 부분집합: 기하학적으로 구현 가능

**상한**: 4개 점이 시작된다고 가정하자. 

일반적 배치 (4개 점이 convex position에 있음)에서, Radon-like argument가 성립한다. 원은 **중심(2개) + 반지름(1개) = 3개 자유도**를 가지는데, 4개 점의 제약 조건은 "이들을 특정 방식으로 포함/제외"이므로, 일반적으로 3개 자유도로는 모두 만족 불가능한 dichotomy가 존재한다.

(정확한 증명은 non-degeneracy argument와 dimension counting을 사용.)

### 정리 4.10 (볼록 k-각형)

$$\text{VC}(\mathcal{H}_{k\text{-gon}}) = 2k + 1.$$

**증명 스케치**: Convex k-polygon은 k개 edge로 정의되고, 각 edge는 2개 매개변수(위치, 기울기), 더하기 전체 위치 제약. 대충 $2k + 1$개 자유도로 $2k+1$개의 affine-independent 점을 shatter 가능. (정확한 construction과 upper bound는 Shalev-Shwartz & Ben-David, Ex 6.8 참조)

### 정리 4.11 (Convex set — 무한 VC)

일반 convex set $\{\text{convex region } C\}$의 VC 차원은 **무한**이다.

**증명**: 임의 $n$개 점 $\{x_1, \ldots, x_n\}$이 convex position에 있다고 가정하자 (즉, 각 점이 다른 점들의 convex hull 내부가 아님). 이 점들의 임의 부분집합 $S$에 대해, $S$의 convex hull과 나머지의 convex hull은 disjoint이다 (convex position의 성질). 

그러면 $S$를 포함하고 나머지를 제외하는 convex 영역을 만들 수 있다 (예: $\text{conv}(S)$를 약간 확대). 따라서 모든 부분집합이 구현 가능하고, 이는 모든 크기 점집합을 shatter함을 의미한다 → **VC = ∞**. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1: 축정렬 직사각형 (VC=4, 2D)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def check_rect_shattering(points, labels):
    """2D 축정렬 직사각형으로 분류 가능한지 확인"""
    # points: (n, 2), labels: {1, -1}
    pos = points[np.array(labels) == 1]
    neg = points[np.array(labels) == -1]
    
    if len(pos) == 0:
        return True  # 공 직사각형
    if len(neg) == 0:
        return True  # 전체 포함
    
    # pos를 포함하되 neg를 제외하는 직사각형 찾기
    a = max(0, np.min(pos[:, 0]) - 0.01)
    b = np.max(pos[:, 0]) + 0.01
    c = np.min(pos[:, 1]) - 0.01
    d = np.max(pos[:, 1]) + 0.01
    
    # neg가 이 직사각형에 포함되는지 확인
    inside_neg = np.all((neg[:, 0] >= a) & (neg[:, 0] <= b) & 
                        (neg[:, 1] >= c) & (neg[:, 1] <= d))
    return not inside_neg

# 다이아몬드 4개 점
points_4 = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)

all_shattered = True
for dichotomy in range(16):
    labels = np.array([1 if (dichotomy >> i) & 1 else -1 for i in range(4)])
    if not check_rect_shattering(points_4, labels):
        all_shattered = False
        print(f"Dichotomy {labels} 불가능")

print(f"4개 점 (다이아몬드) shattered: {all_shattered} (기대: True)")

# 시각화
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for dichotomy in range(16):
    ax = axes.flat[dichotomy]
    labels = np.array([1 if (dichotomy >> i) & 1 else -1 for i in range(4)])
    colors = np.array(['red' if l == 1 else 'blue' for l in labels])
    
    ax.scatter(points_4[:, 0], points_4[:, 1], c=colors, s=100, edgecolors='black')
    
    # 분류 직사각형 그리기
    pos = points_4[labels == 1]
    if len(pos) > 0 and len(pos) < 4:
        a, b = np.min(pos[:, 0]) - 0.2, np.max(pos[:, 0]) + 0.2
        c, d = np.min(pos[:, 1]) - 0.2, np.max(pos[:, 1]) + 0.2
        rect = plt.Rectangle((a, c), b-a, d-c, fill=False, edgecolor='green', linestyle='--')
        ax.add_patch(rect)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f"Dichotomy {dichotomy}")

plt.suptitle("축정렬 직사각형 — 4개 점 모두 shatter")
plt.tight_layout()
plt.show()
```

### 실험 2: 원 (VC=3, 2D)

```python
def check_circle_shattering(points, labels):
    """2D 원으로 분류 가능한지 확인 (수치적 근사)"""
    pos = points[np.array(labels) == 1]
    neg = points[np.array(labels) == -1]
    
    if len(pos) == 0 or len(neg) == 0:
        return True
    
    # 간단 휴리스틱: pos의 중심과 neg의 중심의 중간점 주변에서 탐색
    from scipy.optimize import minimize
    
    def objective(params):
        cx, cy, r = params
        c = np.array([cx, cy])
        # pos는 원 안에, neg는 원 밖에
        loss = 0
        for p in pos:
            loss += max(0, np.linalg.norm(p - c) - r)  # 원 밖이면 penalty
        for n in neg:
            loss += max(0, r - np.linalg.norm(n - c))  # 원 안이면 penalty
        return loss
    
    # 초기값
    cx0, cy0 = np.mean(points, axis=0)
    r0 = 1.0
    result = minimize(objective, [cx0, cy0, r0], method='Nelder-Mead')
    
    return result.fun < 1e-3

# 정삼각형
points_3 = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]], dtype=float)

all_shattered = True
for dichotomy in range(8):
    labels = np.array([1 if (dichotomy >> i) & 1 else -1 for i in range(3)])
    if not check_circle_shattering(points_3, labels):
        all_shattered = False
        print(f"Dichotomy {labels} 불가능")

print(f"3개 점 (정삼각형) shattered: {all_shattered} (기대: True)")

# 4개 점: 정사각형
points_4_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
failures_4 = []
for dichotomy in range(16):
    labels = np.array([1 if (dichotomy >> i) & 1 else -1 for i in range(4)])
    if not check_circle_shattering(points_4_square, labels):
        failures_4.append(dichotomy)

print(f"4개 점 (정사각형) shattered: {len(failures_4) == 0} (기대: False)")
print(f"실현 불가능 dichotomy: {len(failures_4)}/16")
```

---

## 🔗 ML 알고리즘 연결

| 모델 | 형태 | VC (2D) | 응용 |
|------|------|--------|------|
| **Axis-aligned rectangle** | 2개 interval의 곱 | 4 | Object detection (bounding box) |
| **Rotated rectangle** | 임의 방향 직사각형 | 5 | 회전된 ROI 탐지 |
| **Disk/Circle** | 중심+반지름 | 3 | Clustering, RBF kernel 기초 |
| **k-gon** | k-edge convex polygon | $2k+1$ | 복잡한 region 근사 |
| **Convex region** | 일반 convex set | ∞ | 이론적 한계 — uniform convergence 불가능 |

---

## ⚖️ 가정과 한계

1. **축정렬 제약**: Axis-aligned rectangle은 실제 문제(회전된 object)에 덜 적합 → rotated 필요하지만 VC 증가
2. **기하학적 배치**: 같은 크기 점집합이라도 배치에 따라 shattering 가능성이 달라질 수 있음
3. **차원 의존성**: Axis-aligned rectangle의 VC = $2d$는 차원이 증가하면서 급증
4. **Convex의 무한 VC**: Convex set은 이론적으로는 매우 powerful하지만, 실제로는 데이터의 분포 구조로 복잡도가 제한됨 (distribution-dependent bound 필요)

---

## 📌 핵심 정리

- **축정렬 직사각형 (2D)**: VC = 4, $\mathbb{R}^d$에서는 VC = $2d$ (각 차원에서 interval)
- **회전된 직사각형**: VC = 5 (추가 회전 자유도)
- **원판 (2D)**: VC = 3 (중심 2개 + 반지름 1개 매개변수)
- **k-각형**: VC = $2k+1$
- **무한 convex**: 제약 없는 convex region은 VC = ∞
- **다음**: Sauer-Shelah로 성장함수 상계 (Ch4-04), VC 경계 유도 (Ch4-05)

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> $\mathbb{R}^2$에서 축정렬 직사각형의 VC=4임을 보이기 위해, 5개 점 중 하나가 항상 shatter 불가능함을 보여라.</summary>

<br/>

**해설**. 5개 점의 $x$ 좌표를 순서대로 $x_{(1)} \leq x_{(2)} \leq x_{(3)} \leq x_{(4)} \leq x_{(5)}$라 하자. 직사각형은 $x$ 범위로 $[a, b]$를 선택하므로, 포함되는 점들의 $x$ 좌표도 연속 구간을 이룬다.

따라서 dichotomy가 "점 1과 3을 양, 점 2를 음으로 분류"하면, 점 1(최소)과 3을 포함하는 $x$ 구간은 필히 점 2도 포함해야 한다 → **모순**. 이런 식의 dichotomy가 반드시 하나 이상 존재하므로, 5개 점은 완벽히 shatter 불가능. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> 원판의 VC=3을 증명하기 위해, "일반적 위치의 4개 점"에 대해 실현 불가능한 dichotomy가 존재함을 보여라.</summary>

<br/>

**해설**. 원은 중심(2개) + 반지름(1개) = 3개 자유도. 4개 점의 constraint는 일반적으로 4개보다 많은 비선형 조건 (각 점이 원 안/밖인지)을 만들 수 있다.

구체적으로, convex position의 4개 점 (정사각형 등)을 생각하면, dichotomy "점 1,3만 양"은 원이 점 1과 3을 포함하되 점 2,4를 제외해야 하는데, 원의 대칭성으로 인해 **수치적으로 완벽한 분리가 불가능**함을 보일 수 있다. (정확한 증명은 dimension-counting argument와 algebraic geometry 필요.) $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> Convex region의 VC=∞라는 것이 실전에서 어떤 함의를 가지는가? 왜 SVM(반공간, VC=$d+1$)이 convex polytope(VC=∞)보다 실제로 더 잘 일반화할까?</summary>

<br/>

**해설**. VC=∞이면 uniform convergence를 보장할 수 없다 — 고전 SLT 관점에서는 배울 수 없는 가설공간이 된다. 하지만 SVM이 더 잘 일반화하는 이유는:

1. **Margin**: SVM은 단순히 "분류하는 초평면" 중에서 **margin이 가장 큰 것**을 찾는다 → 이는 데이터에 적응한 복잡도 감소.
2. **Rademacher 기반**: Ch5의 margin-based Rademacher 경계는 $O(\text{margin}^{-1})$로, "margin이 크면 복잡도 낮음"을 정식화.
3. **PAC-Bayes 등**: Convex region도 적절한 prior/margin 고려하면 bound 가능.

즉, **pure VC bound는 convex의 무한함을 감당 못하지만, margin·regularization·stability 등 다른 관점을 도입하면 가능**. 이것이 Ch5-07, Ch6, Ch7의 핵심 메시지. $\square$

</details>

---

<div align="center">

◀ [이전: 02. 반공간](./02-halfspace-vc.md) | [📚 README](../README.md) | [다음: 04. Sauer-Shelah ▶](./04-sauer-shelah.md)

</div>
