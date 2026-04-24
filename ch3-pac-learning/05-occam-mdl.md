# 05. Occam's Razor와 MDL 원리

## 🎯 핵심 질문

- **Occam's Razor**는 학습 이론에서 정확히 무엇을 말하는가? "더 짧은 설명이 더 잘 일반화한다"는 주장의 수학적 형태는?
- **정리 (Blumer et al. 1987)**: 가설 $h$의 설명 길이가 $d$ 비트이면, $m = O((d + \log(1/\delta))/\epsilon)$개 샘플로 PAC learn 가능 — 왜 $1/\epsilon$ (not $1/\epsilon^2$)?
- **Description length**를 어떻게 형식화하는가? Kraft 부등식으로 $\sum 2^{-|h|} \leq 1$이 왜 중요한가?
- **MDL(Minimum Description Length) 원리**는 무엇인가? Rissanen(1978)의 정의와 통계적 해석.
- **Bayesian view**: MDL과 prior $p(h) \propto 2^{-|h|}$는 정확히 어떻게 동치인가?
- **압축과 학습의 등가성** (Littlestone-Warmuth): 데이터 압축 능력과 generalization이 같은 능력인가?

---

## 🔍 왜 이 원리가 현대 ML에서 중요한가

"더 간단한 모델이 더 잘 일반화한다"는 직관은 모든 ML 실무자가 안다. 하지만 **왜**인가? 

1. **SRM(Ch7-01)의 이론적 기초**: "모델 복잡도"를 정량화하는 공식적 방법.

2. **Information-theoretic view**: Generalization이 **compression**과 연결된다는 통찰. 데이터를 잘 압축하려면, 참 분포의 구조를 이해해야 한다. 즉, 일반화 능력.

3. **Bayesian interpretation**: 머신러닝을 베이지안 모델 선택 문제로 보는 관점. Prior $\propto 2^{-|h|}$가 자연스럽게 등장.

4. **현대적 재조명**: 신경망에서도 "더 적은 비트로 가중치를 표현할 수 있는 모델이 일반화한다"는 empirical 관찰 (pruning, quantization).

---

## 📐 수학적 선행 조건

- Ch3-01, 3-02: PAC learnability, sample complexity
- Ch2-02: Hoeffding 부등식
- 기초: 정보 이론 (entropy, compression, prefix codes), Kraft 부등식

---

## 📖 직관적 이해

### Occam의 직관

Medieval 철학자 William of Ockham: "필요하지 않은 존재를 가정하지 말라" (multiplicitas non est ponenda sine necessitate).

ML 버전: "같은 정도로 데이터를 잘 설명하는 두 모델이 있다면, **더 간단한 것**을 선택하자. 왜냐하면 더 복잡한 것은 우연히 훈련 데이터를 맞춘 가능성이 높기 때문."

### Description length의 의미

$h$를 인코딩하는데 $|h|$ 비트가 필요하다는 것은:
- 더 많은 매개변수? ⟹ 더 큰 $|h|$
- 더 큰 매개변수 크기? ⟹ 더 큰 $|h|$
- More complex decision boundary? ⟹ 일반적으로 더 큰 $|h|$

따라서 **짧은 description** ⟺ **간단한 모델**.

### Kraft 부등식과 확률의 합

직관: prefix-free code (어떤 code word도 다른 code word의 prefix가 아님)에 대해,

$$\sum_{h} 2^{-|h|} \leq 1.$$

이는 "설명 길이들이 확률처럼 행동한다"는 의미. $p(h) := 2^{-|h|}$로 정의하면 합이 1 (정규화된 분포).

---

## ✏️ 엄밀한 정의

### 정의 3.5.1 (Description Length)

가설 $h$를 인코딩하는데 필요한 비트 수를 **description length** $|h|$라 하자. 정확히는:
- 가설공간 $\mathcal{H}$를 고정
- 각 $h$를 유일하게 인코딩하는 **prefix-free code** 선택
- Code word 길이가 $|h|$ 비트

### 정의 3.5.2 (Kraft 부등식)

길이 $|h|$의 prefix-free code가 존재하려면

$$\sum_{h \in \mathcal{H}} 2^{-|h|} \leq 1.$$

이것을 **Kraft inequality**라 부른다. 역도 성립 (Kraft-McMillan).

### 정의 3.5.3 (MDL 원리 — 두 부분)

주어진 샘플 $S$에 대해, 다음을 최소화하는 $h$를 선택하라:

$$\text{MDL}(h, S) := -\log p(S|h) - \log p(h),$$

또는 비트로 표현하면

$$\text{MDL bits}(h, S) := -\log_2 p(S|h) - \log_2 p(h).$$

분해:
- $-\log p(S|h)$: 가설 $h$ 하에서 **데이터를 설명하는 비트 수** (data description length)
- $-\log p(h)$: **가설 자체의 비트 수** (hypothesis description length)

---

## 🔬 정리와 증명

### 정리 3.5.1 (Occam's Razor Bound — Blumer et al. 1987)

**가정**:
- 각 가설 $h \in \mathcal{H}$에 정수 길이 $|h| \in \mathbb{N}$ 배정 (prefix-free code)
- $\sum_h 2^{-|h|} \leq 1$ (Kraft 부등식)
- Realizable: $\exists h^* \in \mathcal{H}, L_\mathcal{D}(h^*) = 0$
- iid 샘플 $S$, 0-1 loss

**결론**: 확률 $\geq 1-\delta$로

$$L_\mathcal{D}(h) \leq \frac{|h| + \log(1/\delta)}{n}$$

for any $h \in \mathcal{H}$ 만족 $L_S(h) = 0$.

**증명**:

**Step 1. "Bad event" 정의**

고정 $h$에 대해:

$$B_h := \{L_\mathcal{D}(h) > (|h| + \log(1/\delta))/n\} \cap \{L_S(h) = 0\}.$$

**Step 2. Probability bound for one $h$**

$L_\mathcal{D}(h) > (|h| + \log(1/\delta))/n$이면, $L_S(h) = 0$일 확률은

$$\mathbb{P}[L_S(h) = 0 | L_\mathcal{D}(h) > (|h| + \log(1/\delta))/n] \leq (1-\epsilon)^n$$

where $\epsilon := (|h| + \log(1/\delta))/n$.

따라서

$$\mathbb{P}[B_h] \leq \left(1 - \frac{|h| + \log(1/\delta)}{n}\right)^n \leq e^{-(|h| + \log(1/\delta))}.$$

(마지막 부등식: $1-x \leq e^{-x}$, $x = (|h| + \log(1/\delta))/n$.)

**Step 3. Union bound with Kraft**

$$\mathbb{P}[\exists h: B_h] \leq \sum_{h} e^{-(|h| + \log(1/\delta))} = e^{-\log(1/\delta)} \sum_h e^{-|h|}.$$

Kraft 부등식: $\sum_h 2^{-|h|} \leq 1$ ⟹ $\sum_h e^{-|h| \ln 2} \leq 1$ ⟹ $\sum_h e^{-|h|} \leq \sum_h 2^{-|h|} \leq 1$. 

따라서

$$\mathbb{P}[\exists h: B_h] \leq e^{-\log(1/\delta)} \cdot 1 = \delta. \quad \square$$

### 정리 3.5.2 (MDL과 MAP의 동치성)

**MDL 원리**:

$$\min_h [-\log p(S|h) - \log p(h)]$$

**MAP estimation** with prior $p(h) \propto 2^{-|h|}$:

$$\min_h [-\log p(S|h) - \log p(h)]$$

이 둘은 **동일한 최적화 문제**.

**해석**:
- 과정 1: 데이터 $S$에 주어진 $h$ 하에서 likelihood $p(S|h)$를 최대화.
- 과정 2: Prior $p(h) \propto 2^{-|h|}$ (간단한 모델 선호)를 곱한다.
- 결과: "데이터를 잘 설명 + 간단한 모델" 균형.

**주의**: MDL은 Bayesian이 아니라 **정보 이론적 원칙**이다. 하지만 **Bayesian interpretation으로도 읽을 수 있다**.

### 정리 3.5.3 (Compression과 Generalization의 등가성)

**Littlestone & Warmuth (1986)**:

학습자가 "샘플을 잘 압축할 수 있다" ⟺ "일반화할 수 있다".

정확히는:

- **Compression scheme**: 훈련 샘플의 부분집합 $A \subseteq S$를 압축해 "$S \setminus A$의 라벨을 복원할 수 있다면"
- 학습자는 PAC learnable.

---

## 💻 NumPy 구현 검증

### 실험 1: Description length와 Generalization의 관계

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

rng = np.random.default_rng(42)

# 1D threshold classifiers, discretized to different precisions
# 정밀도에 따라 description length가 달라진다

precisions = [5, 8, 10, 12, 16]  # bits to encode threshold
description_lengths = precisions  # |h| = bits per threshold

epsilon_vals = [0.1, 0.15, 0.2]
delta = 0.05

def sample_D_1d_sep(n):
    """Data separable by threshold at 0.5"""
    X = rng.uniform(-1, 1, n)
    Y = (X >= 0.5).astype(int)
    return X, Y

# For each precision, train and test
fig, ax = plt.subplots(figsize=(10, 5))

for eps in epsilon_vals:
    test_errors = []
    
    for prec in precisions:
        desc_len = prec
        # Occam bound: m = (|h| + log(1/δ)) / ε
        theory_m = (desc_len + np.log(1/delta)) / eps
        
        # Simulate actual learning
        n = int(theory_m) + 10  # add some margin
        n_trials = 50
        errors = []
        
        for _ in range(n_trials):
            X_train, Y_train = sample_D_1d_sep(n)
            
            # ERM: try all quantized thresholds
            n_quant = 2 ** prec
            thresholds = np.linspace(-1, 1, n_quant)
            
            losses = []
            for t in thresholds:
                pred = (X_train >= t).astype(int)
                loss = (pred != Y_train).sum() / len(Y_train)
                losses.append(loss)
            
            best_t = thresholds[np.argmin(losses)]
            
            # Test
            X_test, Y_test = sample_D_1d_sep(5000)
            pred_test = (X_test >= best_t).astype(int)
            error = (pred_test != Y_test).sum() / len(Y_test)
            errors.append(error)
        
        test_errors.append(np.mean(errors))
    
    ax.plot(precisions, test_errors, 'o-', 
            label=f'ε={eps}, Occam bound (|h|+log(1/δ))/ε', linewidth=2)

ax.set_xlabel('Description length |h| (bits)'); ax.set_ylabel('Test error')
ax.set_title('Occam Razor: Description length vs Generalization error')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# → Description length가 작을수록 (더 낮은 정밀도) 일반화 오차가 작음 (충분한 샘플이 있으면)
```

### 실험 2: Kraft 부등식 검증 및 Prior 기반 bound

```python
# 가설공간에 다양한 길이 배정, Kraft 부등식 확인
# 그리고 "짧은 설명"에 높은 prior를 주기

# Example: 이진 트리 구조의 가설 (depth d마다 2^d 개 가설)
# 깊이 0: 1 가설 (no split), |h| = 1
# 깊이 1: 2 가설 (one split), |h| = 3
# 깊이 2: 4 가설, |h| = 5
# ...
# 일반: depth d에서 2^d개 가설, |h| = 2d+1

depths = range(1, 6)
total_kraft = 0
priors = []

for d in depths:
    n_hyp_at_d = 2 ** d
    desc_len = 2 * d + 1
    kraft_contrib = n_hyp_at_d * (2 ** (-desc_len))
    total_kraft += kraft_contrib
    
    prior = 2 ** (-desc_len)
    priors.append(prior)
    
    print(f'Depth {d}: {n_hyp_at_d:3d} hypotheses, |h|={desc_len:2d}, '
          f'Kraft contribution = {kraft_contrib:.4f}')

print(f'\nTotal Kraft sum (should be ≤ 1): {total_kraft:.4f}')

# Visualize priors
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.semilogy(depths, priors, 'o-', linewidth=2)
ax1.set_xlabel('Description length |h|'); ax1.set_ylabel('Prior p(h) = 2^(-|h|)')
ax1.set_title('Prior distribution: Shorter hypotheses have higher prior')
ax1.grid(alpha=0.3)

# Kraft constraint
ax2.bar(range(1, 6), [2**(-（2*d+1)) for d in range(1, 6)], alpha=0.7)
ax2.set_xlabel('Hypothesis'); ax2.set_ylabel('Kraft weight 2^(-|h|)')
ax2.set_title(f'Kraft inequality: sum = {total_kraft:.4f} ≤ 1')
ax2.axhline(0, color='k', linewidth=0.5)
plt.tight_layout(); plt.show()
```

---

## 🔗 ML 알고리즘 연결

| 알고리즘 | Complexity measure | MDL 해석 |
|---------|-----------------|---------|
| **Decision Trees** | Tree 크기 (nodes, depth) | Shorter tree = shorter encoding |
| **Neural Networks** | Weight magnitude / sparsity | Smaller weights = shorter encoding |
| **Regularized ERM** | $\min_h [L_S(h) + \lambda \|h\|_2]$ | Loss + encoding cost |
| **Model selection** | AIC/BIC (Ch7-02) | Data encoding + model encoding |

---

## ⚖️ 가정과 한계

1. **Description length의 모호성**: $|h|$를 정확히 어떻게 정의할 것인가? 가중치를 몇 비트로 인코딩? 구현에 따라 다름. 표준화된 정의 부족.

2. **Kraft 부등식의 현실성**: 실전 가설공간에서 $\sum 2^{-|h|} \leq 1$을 만족시키려면 매우 특정한 인코딩 체계 필요. 자연스럽지 않을 수 있음.

3. **Fast rate ($1/\epsilon$)의 가정**: Occam bound는 realizable case를 가정. 노이즈 있으면 더 느린 $1/\epsilon^2$ 추가.

4. **정보 이론 vs 계산**: Occam bound는 정보 이론적이지만, ERM 최적화의 계산 복잡도는 보장하지 않음.

5. **Prior의 선택**: Bayesian 버전에서 prior $p(h) \propto 2^{-|h|}$가 "자연스러운가"? 다른 prior를 쓸 수도 있음.

---

## 📌 핵심 정리

- **Occam's Razor**: 간단한 모델(짧은 description length)이 더 잘 일반화한다 — 수학적으로 bound로 증명 가능.

- **Occam bound** (realizable, prefix-free code): $L_\mathcal{D}(h) \leq (|h| + \log(1/\delta)) / n$ w.p. $1-\delta$.

- **Sample complexity**: $m = O([d + \log(1/\delta)] / \epsilon)$ — **$1/\epsilon$ fast rate** (description length 활용).

- **MDL 원리**: $\min_h [-\log p(S|h) - \log p(h)]$ — hypothesis + data의 **total encoding** 최소화.

- **Bayesian equivalence**: MDL ⟺ MAP with prior $p(h) \propto 2^{-|h|}$.

- **Kraft 부등식**: Description lengths가 확률처럼 행동. Prefix-free code가 존재하려면 만족해야 함.

- **압축과 일반화**: 데이터 압축 능력과 generalization이 본질적으로 같은 능력 (Littlestone-Warmuth).

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> Occam bound와 realizable PAC bound (Ch3-02의 정리 3.2.1)를 비교하라. "Description length $|h|$"와 "$\log |\mathcal{H}|$"가 어떻게 다른가? 언제 더 낫다고 할 수 있는가?</summary>

<br/>

**해설**. 

| 측면 | Realizable PAC (유한) | Occam bound |
|------|-----|------|
| Sample complexity | $\log |\mathcal{H}| / \epsilon$ | $(\|h\| + \log(1/\delta))/\epsilon$ |
| 가정 | 가설공간 크기 $\|\mathcal{H}\|$ 알아야 함 | 각 가설의 길이만 알면 됨 |
| 적용 | 유한 $\mathcal{H}$ 전용 | 무한 $\mathcal{H}$도 가능 (Kraft 조건만) |
| 타이트함 | Fixed — 모든 가설을 같게 봄 | Individual — 짧은 가설에 유리 |

**언제 Occam이 더 나은가?**: 가설공간이 크지만 **간단한 해를 기대**할 때. 예: linear classifier ($\log |\mathcal{H}|$ 매우 크지만, 간단한 linear 가설이 $|h|$ 작음). $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> Kraft 부등식 $\sum_h 2^{-|h|} \leq 1$을 만족하면서, 동시에 모든 $h$를 "fairly하게" 인코딩하려면 어떻게 해야 하는가? (Hint: Huffman encoding)</summary>

<br/>

**해설**. Kraft 부등식은 "가능한 code word 길이의 합"이 확률 분포 같아야 한다"는 제약. 이를 달성하는 방법:

1. **Huffman encoding** (information theory): 자주 나오는 $h$에는 짧은 code, 드문 $h$에는 긴 code 배정. 이렇게 하면 평균 길이가 최소 (entropy 달성).

2. **Fixed-length + variable-length**: 작은 $\mathcal{H}$는 fixed-length binary (예: 100개 가설면 7비트), 큰 부분은 variable-length Huffman.

3. **Optimal분배**: 어떤 $h$를 자주 쓰는지 알면, Kraft constraint 하에서 전체 bound를 최소화 가능. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> 신경망의 "description length" $|h|$를 어떻게 정의할 것인가? (a) 전체 weight를 floating-point로 인코딩? (b) Sparsity 고려? (c) Weight magnitude 정규화?</description>

<br/>

**해설**. 신경망에 MDL을 적용하려면 여러 선택지:

1. **Weight magnitude**: Smaller weights = shorter encoding (quantization). 예: heuristic $|h| = \sum |\text{non-zero weights}| \cdot \text{bits per weight}$.

2. **VC dimension**: $|h| \approx O(W^2 \log W)$ 기반. 하지만 vacuous.

3. **Compression-based**: 실제로 weight를 압축해 몇 비트 필요한지 측정. 더 정직하지만 계산 비용 높음.

4. **Empirical**: 최근 추세는 가중치의 **spectral norm** $\|W\|_{\text{spectral}}$를 정규화 (Ch5-05의 spectral bound). 간접적 complexity measure.

결론: 신경망의 "정확한" MDL은 아직 표준화되지 않음. 하지만 직관 ("간단한 가중치 = 잘 일반화")은 여전히 강력. $\square$

</details>

---

<div align="center">

◀ [이전: 04. Fundamental Theorem of Statistical Learning](./04-fundamental-theorem.md) | [📚 README](../README.md) | [다음: Ch4-01. Shattering과 VC 차원 ▶](../ch4-vc-dimension/01-shattering-vc.md)

</div>
