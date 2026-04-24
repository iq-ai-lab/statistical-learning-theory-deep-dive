# 01. Shattering과 VC 차원의 정의

## 🎯 핵심 질문

- **Shattering(분쇄)**는 기하학적으로 무엇인가? 가설공간 $\mathcal{H}$가 점집합을 "모든 방식으로 분류"한다는 것의 수학적 의미는?
- **VC 차원** $\text{VC}(\mathcal{H})$는 무엇인가? 왜 "한 개의 수"로 가설공간의 복잡도를 나타낼 수 있는가?
- Threshold ($\mathbb{R}$ 위의 구간 분류), Interval, Axis-aligned rectangle의 VC 차원은 각각 몇인가?
- VC가 유한하면 "uniform convergence"가 성립한다는 Fundamental Theorem과 어떻게 연결되는가?
- 신경망(k-NN)의 VC 차원이 $\infty$라는 것이 의미하는 바는?

---

## 🔍 왜 VC 차원이 현대 ML에서 중요한가

PAC Learning(Ch3)은 "유한 $|\mathcal{H}|$일 때만" $m = O((\log|\mathcal{H}| + \log(1/\delta))/\epsilon^2)$ 샘플로 배울 수 있다고 했다. 하지만 신경망, 의사결정 트리, SVM 같은 실용 모델의 가설공간은 **연속 매개변수** 때문에 무한하다. 그렇다면 어떻게 이들을 분석할 것인가?

VC 차원은 "무한 $\mathcal{H}$라도, **유한 크기의 부분집합에 대해서는** finite complexity를 가질 수 있다"는 통찰에서 출발한다. 이를 통해:

1. **$m$ 개 샘플에 대한 성장함수** $\Pi_\mathcal{H}(m)$를 정의 → 무한 $\mathcal{H}$를 유한으로 상계 (Sauer-Shelah)
2. **Union Bound를 확장** → 무한 경우에도 적용 가능 (정리 4.5)
3. **현대 ML의 모든 분류기를 단일 척도로 비교** → SVM (VC=$d+1$), 트리 (VC=$\sim 2^d$), NN (VC=$\sim W \log W$)

또한 VC가 유한한지/무한한지는 Fundamental Theorem(Ch3-04)에서 "uniform convergence 존재성"과 동치이므로, **배울 수 있는 것과 배울 수 없는 것의 경계**를 정의한다.

---

## 📐 수학적 선행 조건

- [Ch1-01](./01-statistical-learning.md): 가설공간, 손실, 경험위험 $L_S(h)$
- [Ch1-03](./03-erm-principle.md): ERM, 3분해
- [Ch3-01~03](../ch3-pac-learning/): PAC 정의, 유한 $\mathcal{H}$의 샘플 복잡도
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab): Uniform convergence 개념
- 기초: Dichotomy(이진분류), 집합론, 조합론

---

## 📖 직관적 이해

### Dichotomy와 Realization

$m$개 점의 집합 $C = \{x_1, \ldots, x_m\} \subseteq \mathcal{X}$를 가정하자. 각 점에 대해 가설 $h$는 "양의 라벨" 또는 "음의 라벨"을 할당한다. 이렇게 만들어지는 라벨 조합은 **dichotomy** 또는 **partition**으로 불린다.

| 점 | $h_1$ | $h_2$ | $h_3$ | ... |
|----|-------|-------|-------|-----|
| $x_1$ | + | + | - | ... |
| $x_2$ | + | - | - | ... |
| $x_3$ | - | + | + | ... |

$m$개 점에 대해 가능한 dichotomy는 최대 $2^m$개다. 하지만 가설공간 $\mathcal{H}$에 의해 "실현 가능(realizable)"한 dichotomy는 그보다 적을 수 있다.

**Shattering의 핵심**: $\mathcal{H}$가 $C$를 shatter한다는 것은 "$\mathcal{H}$의 가설들이 $2^m$개의 모든 dichotomy를 만들 수 있다"는 뜻이다.

### 예시: Thresholds vs Intervals on $\mathbb{R}$

**1. Threshold classifiers**: $h_\theta(x) = \mathbb{1}[x \geq \theta]$

3개 점 $\{a, b, c\}$ (단, $a < b < c$)를 생각해보자. Threshold는 하나의 "끝"으로만 나눌 수 있으므로, 예를 들어 $\{a\}$을 양, $\{b, c\}$를 음으로 분류할 수 없다. 최대 1개 점을 shatter할 수 있다 → **VC = 1**.

**2. Interval classifiers**: $h_{a,b}(x) = \mathbb{1}[a \leq x \leq b]$

3개 점 $\{0, 1, 2\}$를 생각해보자:
- $\emptyset$ (모두 음): interval = 공집합
- $\{1\}$ (음, 양, 음): interval = $[1, 1]$
- $\{0, 1, 2\}$ (모두 양): interval = $[0, 2]$ 이상
- 하지만 $\{0, 2\}$ (양, 음, 양): 불가능 — interval은 연속 구간이므로 $1$도 포함해야 함

따라서 4개 점 중 일부는 shatter 불가능. 2개 점 $\{0, 2\}$는 모두 shatter 가능 → **VC = 2**.

### VC 차원의 직관

**$\text{VC}(\mathcal{H})$는 "$\mathcal{H}$가 완벽하게 분류할 수 있는 최대 크기의 점집합"이다.** 이는:
- 가설공간의 "자유도(degrees of freedom)"를 나타낸다
- 더 큰 VC = 더 flexible하고 powerful한 모델
- 하지만 더 큰 VC = uniform convergence를 위해 더 많은 샘플 필요

---

## ✏️ 엄밀한 정의

### 정의 4.1 (Dichotomy와 Realization)

점집합 $C = \{x_1, \ldots, x_m\} \subseteq \mathcal{X}$에 대해, **dichotomy**는 부분집합 $S \subseteq C$다 (양의 라벨인 점들). 

가설 $h: \mathcal{X} \to \{0, 1\}$은 dichotomy $S$를 **realize**한다는 것은
$$h(x_i) = 1 \iff x_i \in S$$
를 의미한다.

$\mathcal{H}$에 의해 **realize된 dichotomy들의 집합**을 $\mathcal{H}|_C$ 또는 $\mathcal{H}_C$라 표기한다:
$$\mathcal{H}|_C := \{h(C) : h \in \mathcal{H}\} = \{(h(x_1), \ldots, h(x_m)) : h \in \mathcal{H}\}.$$

### 정의 4.2 (Shattering)

가설공간 $\mathcal{H}$가 점집합 $C$를 **shatter(분쇄)**한다는 것은
$$|\mathcal{H}|_C| = 2^{|C|}$$
를 의미한다. 즉, $C$의 **모든 부분집합**이 어떤 $h \in \mathcal{H}$에 의해 realize될 수 있다.

더 직관적으로: $C$의 각 점에 대해 양/음을 독립적으로 할당하는 $2^{|C|}$가지 방식 모두가 $\mathcal{H}$의 어떤 가설로 표현 가능하다.

### 정의 4.3 (VC 차원)

가설공간 $\mathcal{H}$의 **VC 차원(Vapnik-Chervonenkis dimension)**은
$$\text{VC}(\mathcal{H}) := \max\{m : \exists C \subseteq \mathcal{X}, |C| = m, \mathcal{H} \text{ shatters } C\}.$$

**주의**: 
1. "$m$개 점이 shatter되는 것**이 존재**하면 충분" — 모든 $m$-점 집합이 shatter될 필요는 없다.
2. $\text{VC}(\mathcal{H})$가 $d$라면, $d+1$개 점은 shatter 불가능하다 (정의에 의해).
3. VC가 정의되려면, shatter 가능한 최대 크기가 존재해야 한다. VC가 무한하면 모든 크기의 점집합을 shatter할 수 있다.

---

## 🔬 정리와 증명

### 정리 4.1 (VC 차원의 단조성)

$\mathcal{H}_1 \subseteq \mathcal{H}_2$이면 $\text{VC}(\mathcal{H}_1) \leq \text{VC}(\mathcal{H}_2)$.

**증명**. $\mathcal{H}_1$이 크기 $m$의 점집합 $C$를 shatter하면, $\mathcal{H}_2 \supseteq \mathcal{H}_1$도 $C$를 shatter한다. 따라서 shattering 가능 최대 크기도 증가하거나 같다. $\square$

### 정리 4.2 (VC가 유한한 필요충분조건)

$\text{VC}(\mathcal{H}) < \infty$ $\iff$ 어떤 $m^*$ 이상의 크기 점집합은 모두 shatter 불가능하다.

**증명**. VC의 정의에 의해, $\text{VC}(\mathcal{H}) = d < \infty$라면 정의상 어떤 $d+1$-점 집합 (더 나아가 모든 $d+1$-점 집합)을 shatter 불가능하다 — 이것이 최대값이 존재한다는 뜻이기 때문. 역은 자명. $\square$

### 정리 4.3 (상한: $|\mathcal{H}|_C| \leq 2^{|C|}$)

고정 $C$에 대해 항상 $|\mathcal{H}|_C| \leq 2^{|C|}$이고, 등호는 $\mathcal{H}$가 $C$를 shatter할 때만 성립.

**증명**. $\mathcal{H}|_C$는 $C$에서 $\{0,1\}$로의 함수들의 부분집합인데, 총 함수의 개수는 $2^{|C|}$이다. 따라서 $|\mathcal{H}|_C| \leq 2^{|C|}$. Shattering은 정확히 등호 조건. $\square$

### 정리 4.4 (VC와 Realizability의 기하학적 의미)

$\text{VC}(\mathcal{H}) = d$라면:
- (존재) 어떤 $d$-점 집합 $C^*$가 존재하여 $\mathcal{H}$가 $C^*$를 shatter.
- (불존재) 모든 $(d+1)$-점 집합 $C$에 대해, $\mathcal{H}$가 실현하지 못하는 dichotomy가 **최소 하나** 존재.

**증명**. 정의에 의한 직접 귀결. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1: Threshold classifiers on $\mathbb{R}$ (VC=1)

```python
import numpy as np
from itertools import combinations

def check_shattering(points, hypothesis_class, name=""):
    """
    점 집합 points와 가설공간을 받아서,
    모든 2^|points|개의 dichotomy를 실현할 수 있는지 확인.
    """
    n = len(points)
    all_dichotomies = []
    for r in range(n+1):
        for subset in combinations(range(n), r):
            dichotomy = tuple(1 if i in subset else 0 for i in range(n))
            all_dichotomies.append(dichotomy)
    
    # 가설공간에 의해 실현 가능한 dichotomy
    realized = set()
    for h in hypothesis_class:
        prediction = tuple(h(x) for x in points)
        realized.add(prediction)
    
    shattered = (len(realized) == 2**n)
    print(f"{name}: {len(realized)}/{2**n} dichotomies realized. Shattered: {shattered}")
    return shattered

# Threshold classifiers: h_θ(x) = 1 iff x >= θ
def make_threshold_classifiers(num_params=100):
    thetas = np.linspace(-10, 10, num_params)
    return [lambda x, t=theta: 1 if x >= t else 0 for theta in thetas]

# 1개 점은 shatter 가능
points_1 = np.array([0.5])
h_class_1 = make_threshold_classifiers()
check_shattering(points_1, h_class_1, "Threshold on 1 point")

# 2개 점은 shatter 불가능
points_2 = np.array([-1.0, 1.0])
h_class_2 = make_threshold_classifiers()
check_shattering(points_2, h_class_2, "Threshold on 2 points")
# → 실행 결과: "Threshold on 2 points: 3/4 dichotomies realized"
# (양수, 음수 → 불가능)

# VC(threshold) = 1 확인
```

### 실험 2: Intervals on $\mathbb{R}$ (VC=2)

```python
# Interval classifiers: h_{a,b}(x) = 1 iff a <= x <= b
def make_interval_classifiers(num_params=50):
    grid = np.linspace(-5, 5, num_params)
    classifiers = []
    for a in grid:
        for b in grid:
            if a <= b:
                classifiers.append(lambda x, al=a, br=b: 1 if al <= x <= br else 0)
    return classifiers

# 2개 점은 shatter 가능
points_2 = np.array([-1.0, 1.0])
h_class_int = make_interval_classifiers()
check_shattering(points_2, h_class_int, "Interval on 2 points")
# → "Interval on 2 points: 4/4 dichotomies realized. Shattered: True"

# 3개 점은 shatter 불가능
points_3 = np.array([-2.0, 0.0, 2.0])
check_shattering(points_3, h_class_int, "Interval on 3 points")
# → "Interval on 3 points: ?/8 dichotomies realized" (4~6개, 특히 {-2, 2} 불가능)

# VC(interval) = 2 확인
```

### 실험 3: 작은 유한 $\mathcal{H}$의 VC 계산

```python
def compute_vc_dimension_brute_force(hypothesis_class, max_points=10):
    """
    작은 유한 가설공간의 VC 차원을 brute force로 계산.
    """
    vc = 0
    for m in range(1, max_points + 1):
        found_shattered_set = False
        # 임의의 m개 점 샘플
        for trial in range(50):
            points = np.random.uniform(-10, 10, m)
            if check_shattering_silent(points, hypothesis_class):
                found_shattered_set = True
                break
        if not found_shattered_set:
            return vc
        vc = m
    return vc

def check_shattering_silent(points, hypothesis_class):
    """shatter 여부만 True/False로 반환"""
    n = len(points)
    realized = set()
    for h in hypothesis_class:
        prediction = tuple(h(x) for x in points)
        realized.add(prediction)
    return len(realized) == 2**n

# 예시: XOR-like 유한 가설공간
h_xor = [
    lambda x: 1 if x[0] > 0 else 0,                      # h1
    lambda x: 1 if x[1] > 0 else 0,                      # h2
    lambda x: 1 if (x[0] > 0) != (x[1] > 0) else 0,      # h3 (XOR)
    lambda x: 0,                                          # h4 (all negative)
]
# 이 작은 H의 VC는 대략 1~2 정도로 예상
```

---

## 🔗 ML 알고리즘 연결

| 모델 | 형태 | VC 차원 |
|------|------|--------|
| **Threshold on $\mathbb{R}$** | $\{x \mapsto \mathbb{1}[x \geq \theta]\}$ | 1 |
| **Interval on $\mathbb{R}$** | $\{x \mapsto \mathbb{1}[a \leq x \leq b]\}$ | 2 |
| **Halfspace in $\mathbb{R}^d$** | $\{x \mapsto \mathbb{1}[w^\top x + b \geq 0]\}$ | $d+1$ |
| **Axis-aligned rectangle in $\mathbb{R}^2$** | $\{x \mapsto \mathbb{1}[x_1 \in [a, b], x_2 \in [c, d]]\}$ | 4 |
| **Decision tree depth $d$** | Tree with $2^d$ leaves | $\leq 2^d$ |
| **Neural network ($W$ params)** | 다층 신경망 | $O(W \log W)$ |
| **1-NN** | 최근접 이웃 | $\infty$ |

VC 차원이 무한한 모델(k-NN)은 uniform convergence 보장 불가능. Ch4-07에서 다시 다룬다.

---

## ⚖️ 가정과 한계

1. **Binary classification만**: VC는 원래 $\{0,1\}$ 라벨용. 다중 클래스나 회귀는 확장 필요 (VC-like bounds).
2. **정의의 최악의 경우**: VC를 달성하는 점집합이 반드시 "자연스러운" 분포에 나타나는 것은 아니다. 특정 분포 하에서는 실제 복잡도가 더 낮을 수 있다.
3. **discrete vs continuous $\mathcal{H}$**: 유한 가설공간은 VC가 자동 유한이지만, 무한 $\mathcal{H}$에서 VC가 무한할 수도 있고 유한할 수도 있다.
4. **Sauer-Shelah 없이는 bound 사용 불가**: VC만으로는 정리 4.5의 일반화 경계를 쓸 수 없다. Growth function $\Pi_\mathcal{H}(m)$과 Sauer-Shelah lemma가 필수.

---

## 📌 핵심 정리

- **Shattering**: 가설공간 $\mathcal{H}$가 점집합 $C$의 **모든 부분집합**을 분류할 수 있다는 의미. 정식으로 $|\mathcal{H}|_C| = 2^{|C|}$.
- **VC 차원**: "$\mathcal{H}$가 완벽하게 분류할 수 있는 최대 크기의 점집합"의 크기. 가설공간의 자유도/복잡도를 단일 정수로 나타낸다.
- **예시**: Threshold(VC=1), Interval(VC=2), 반공간(VC=$d+1$), 축정렬 직사각형(VC=4).
- **무한 VC의 의미**: 모든 크기의 점집합을 shatter 가능 → **uniform convergence 불가능** → PAC 불학습 가능.
- **다음 단계**: Sauer-Shelah로 성장함수 bound (04), 이를 이용한 VC 일반화 경계 (05).

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> 다음을 보여라: 만약 $\mathcal{H}$가 크기 $d+1$의 어떤 점집합 $C$를 shatter하면, $d+1 \leq \text{VC}(\mathcal{H})$이다.</summary>

<br/>

**해설**. VC의 정의에 의해, $\text{VC}(\mathcal{H})$는 shatter 가능한 점집합의 최대 크기다. 크기 $d+1$의 점집합이 shatter되므로, 최대 크기는 최소 $d+1$이다. 따라서 $\text{VC}(\mathcal{H}) \geq d+1$. $\square$

이 부등식은 "VC 하한을 증명하는 방법"이다 — 특정 크기 점집합이 shatter되는 예시 하나만 찾으면 충분하다.

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> $\mathcal{H}_1 \cap \mathcal{H}_2$의 VC 차원은 $\min(\text{VC}(\mathcal{H}_1), \text{VC}(\mathcal{H}_2))$ 이상이거나 이하인가? 반례를 찾거나 부등식을 증명하라.</summary>

<br/>

**해설**. 일반적으로 관계가 없다. 

**반례**: $\mathcal{H}_1 = \{h(x) = 1\}$ (모두 양, VC=0), $\mathcal{H}_2 = \{h(x) = 0\}$ (모두 음, VC=0). 그러면 $\mathcal{H}_1 \cap \mathcal{H}_2 = \emptyset$인데, 공집합의 VC를 0으로 정의하면 $\text{VC}(\mathcal{H}_1 \cap \mathcal{H}_2) = 0 = \min(0, 0)$.

하지만 더 일반적으로, intersection은 더 제한적이므로 $\text{VC}(\mathcal{H}_1 \cap \mathcal{H}_2) \leq \min(\text{VC}(\mathcal{H}_1), \text{VC}(\mathcal{H}_2))$ 정도가 기대되지만, 엄밀한 일반 명제는 아니다. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> 신경망이 VC 무한을 가지는 한 가지 방법을 기술하라 — 즉, 임의로 큰 점집합을 shatter할 수 있음을 직관적으로 설명하라.</summary>

<br/>

**해설**. 신경망의 매개변수가 무한히 많다고 가정하거나, 매개변수는 유한하지만 깊이를 임의로 조정 가능하면, 각 점에 대해 독립적으로 출력을 제어할 수 있다. 예를 들어, 충분한 은닉 노드를 가진 1-hidden-layer network는 $n$개 샘플에 대해 $n$개의 distinct feature를 학습하고, 출력층이 이들을 선형 결합하면 모든 dichotomy를 구현 가능. 따라서 VC = $\infty$. 이것이 "왜 deep learning은 고전 VC bound로 설명 안 되는가"의 일부 이유. Ch4-07에서 상세. $\square$

</details>

---

<div align="center">

◀ [이전: Ch3-05. Occam's Razor와 MDL](../ch3-pac-learning/05-occam-mdl.md) | [📚 README](../README.md) | [다음: 02. VC 차원 계산 — 선형 분류기 ▶](./02-halfspace-vc.md)

</div>
