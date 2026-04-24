# 05. VC 경계의 유도

## 🎯 핵심 질문

- **VC bound**의 완전한 형태는? $\mathbb{P}(\sup_h |L_\mathcal{D}(h) - L_S(h)| \geq \epsilon)$을 어떻게 유도하는가?
- **Symmetrization lemma**(ghost sample trick)는 무엇인가? 왜 sample을 두 배로 만들면 bound가 나오는가?
- **Random swap**: Rademacher sign $\sigma_i \in \{\pm 1\}$로 swap할 때, supremum이 왜 유한 집합으로 제한되는가?
- 최종 bound $4 \Pi_\mathcal{H}(2n) e^{-n\epsilon^2/8}$에서 상수들은 어디서 오는가?
- **Sample complexity**: 이 bound로부터 $m = O((d \log(1/\epsilon) + \log(1/\delta))/\epsilon^2)$을 어떻게 얻는가?

---

## 🔍 왜 VC 경계 유도가 중요한가

Ch3에서 **유한 $\mathcal{H}$**에 대해 Union Bound로 경계를 얻었다. 이제 **무한 $\mathcal{H}$**에서도 가능한가?

핵심 인사이트: 무한이어도, **고정 샘플 $S$에서는** $\mathcal{H}|_S$는 유한이다! 따라서:
1. Ghost sample $S'$를 도입 (symmetrization)
2. $\mathcal{H}$의 supremum을 $\mathcal{H}|_{S \cup S'}$ (크기 $\leq \Pi(2n)$)로 제한
3. Hoeffding + Union → 확률 경계 완성

이것이 **무한을 유한으로 바꾸는** VC 이론의 마술이다.

---

## 📐 수학적 선행 조건

- [Ch1-03](../ch1-learning-formulation/03-erm-principle.md): ERM, uniform convergence 개념
- [Ch2-02](../ch2-concentration/02-hoeffding.md): Hoeffding 부등식
- [Ch3-03](../ch3-pac-learning/03-agnostic-pac.md): Union Bound
- [Ch4-04](./04-sauer-shelah.md): Sauer-Shelah, 성장함수

---

## 📖 직관적 이해

### Ghost Sample Trick

원래 샘플: $S = \{(x_1, y_1), \ldots, (x_n, y_n)\}$ → $L_S(h)$ 계산

Ghost sample: $S' = \{(x'_1, y'_1), \ldots, (x'_n, y'_n)\}$ (독립 추가 샘플)

**핵심 관찰**:
$$\mathbb{E}_{S'} [L_{S'}(h)] = L_\mathcal{D}(h).$$

따라서:
$$\mathbb{E}_{S'} [|L_S(h) - L_\mathcal{D}(h)|] \approx \mathbb{E}_{S'} [|L_S(h) - L_{S'}(h)|].$$

이렇게 하면 "$\mathcal{D}$를 알 필요 없이" empirical quantity로만 bound 가능해진다!

### Random Swap

$\sigma = (\sigma_1, \ldots, \sigma_n)$, $\sigma_i \in \{\pm 1\}$ 균등 랜덤.

$S$와 $S'$의 $i$번째 샘플을 swap: $(x_i, y_i) \leftrightarrow (x'_i, y'_i)$ (if $\sigma_i = +1$).

결과: $\tilde{S}$ (swapped sample)

$$|L_S(h) - L_{S'}(h)|$$가 $\sigma$에 대해 non-smooth하면, swap은 이를 "smooth화"하는 역할을 한다. 그 결과 **bounded differences** 조건을 만족하여 McDiarmid가 적용 가능해진다.

---

## ✏️ 엄밀한 정의

### 정의 4.12 (Symmetrization)

독립 샘플 $S, S' \sim \mathcal{D}^n$에 대해,
$$L_S(h) := \frac{1}{n} \sum_{i=1}^n \ell(h(x_i), y_i), \quad L_{S'}(h) := \frac{1}{n} \sum_{i=1}^n \ell(h(x'_i), y'_i).$$

---

## 🔬 정리와 증명

### 정리 4.15 (Symmetrization Lemma)

$$\mathbb{P}_{S}(\sup_{h \in \mathcal{H}} |L_S(h) - L_\mathcal{D}(h)| \geq \epsilon) \leq 2 \mathbb{P}_{S, S'}(\sup_{h \in \mathcal{H}} |L_S(h) - L_{S'}(h)| \geq \epsilon/2).$$

**증명**:

조건부로 $S$를 고정하자. $\ell(h(x), y) \in [0, 1]$이므로 $L(h)$도 $[0, 1]$.

$$|L_S(h) - L_\mathcal{D}(h)| = |\mathbb{E}_{S'}[L_S(h)] - \mathbb{E}_{S'}[L_{S'}(h)|$$
(정리 1.1: $\mathbb{E}[L_S(h)] = L_\mathcal{D}(h)$)

삼각부등식:
$$\leq \mathbb{E}_{S'}[|L_S(h) - L_{S'}(h)|].$$

따라서 (조건부):
$$\sup_h |L_S(h) - L_\mathcal{D}(h)| \leq \mathbb{E}_{S'}[\sup_h |L_S(h) - L_{S'}(h)|].$$

Markov를 적용하면:
$$\mathbb{P}_{S'}(\sup_h |L_S(h) - L_{S'}(h)| \geq \epsilon/2 | S) \geq \mathbb{P}_{S'}(\sup_h |L_S(h) - L_\mathcal{D}(h)| \geq \epsilon | S).$$

$S$에 대해 기대값을 취하면:
$$\mathbb{P}_{S, S'}(\sup_h |L_S(h) - L_{S'}(h)| \geq \epsilon/2) \geq \frac{1}{2} \mathbb{P}_S(\sup_h |L_S(h) - L_\mathcal{D}(h)| \geq \epsilon).$$

(정확한 상수 $2$는 Markov 부등식의 loose함을 설명.)

$\square$

### 정리 4.16 (Random Swap & Rademacher)

Rademacher $\sigma_i \in \{\pm 1\}$ 균등 독립에 대해,

$$\mathbb{P}_{S, S', \sigma}(\sup_h \left| \frac{1}{n} \sum_i \sigma_i [\ell(h(x_i), y_i) - \ell(h(x'_i), y'_i)] \right| \geq \epsilon) \leq 2 \mathbb{P}_{S \cup S'}(\sup_h |L_S(h) - L_{S'}(h)| \geq \epsilon/2).$$

**증명 스케치**: Rademacher swap은 "unbiased estimator of difference"를 만든다.

**정리 4.17** (Union Bound over Finiteduced Set)

$$\mathbb{P}_{S, S'}(\sup_{h \in \mathcal{H}} |L_S(h) - L_{S'}(h)| \geq \epsilon) \leq |\mathcal{H}|_{S \cup S'}| \cdot \max_{h \in \mathcal{H}|_{S \cup S'}} \mathbb{P}(|L_S(h) - L_{S'}(h)| \geq \epsilon).$$

이제 $\mathcal{H}|_{S \cup S'}$는 고정된 $2n$개 샘플에서 정의된 **유한 가설공간**이고, $|\mathcal{H}|_{S \cup S'}| \leq \Pi_\mathcal{H}(2n)$.

각 고정 $h$에 대해 Hoeffding을 적용:
$$\mathbb{P}(|L_S(h) - L_{S'}(h)| \geq \epsilon) \leq 2 e^{-2n\epsilon^2}.$$

따라서:
$$\mathbb{P}(\sup_h |L_S(h) - L_{S'}(h)| \geq \epsilon) \leq \Pi_\mathcal{H}(2n) \cdot 2 e^{-2n\epsilon^2}.$$

### 정리 4.18 (VC 일반화 경계)

$$\mathbb{P}_S(\sup_{h \in \mathcal{H}} |L_\mathcal{D}(h) - L_S(h)| \geq \epsilon) \leq 4 \Pi_\mathcal{H}(2n) e^{-n\epsilon^2/8}.$$

**증명**: 정리 4.15, 4.16, 4.17을 결합.

- Symmetrization: 좌변 $\leq 2 \times$ (RHS of 4.15)
- Random swap (Hoeffding): RHS of 4.15 $\leq \Pr[\text{Rademacher 항} \geq \epsilon/4]$
- Union 오버 유한: $\leq \Pi_\mathcal{H}(2n) \times 2e^{-2n(\epsilon/4)^2} = \Pi_\mathcal{H}(2n) \times 2e^{-n\epsilon^2/8}$

상수 정리: $2 \times 2 = 4$.

$\square$

### 정리 4.19 (Sample Complexity from VC Bound)

오차 $\epsilon$과 확률 실패 $\delta$를 원한다면, $n$이 충분히 크면:
$$4 \Pi_\mathcal{H}(2n) e^{-n\epsilon^2/8} \leq \delta.$$

Sauer-Shelah ($\Pi_\mathcal{H}(2n) \leq (2en/d)^d$)를 사용하면:
$$4 (2en/d)^d e^{-n\epsilon^2/8} \leq \delta.$$

로그를 취하고 정리하면:
$$n = O\left(\frac{d \log(1/\epsilon) + \log(1/\delta)}{\epsilon^2}\right).$$

(정확한 상수는 생략.)

---

## 💻 NumPy 구현 검증

### 실험 1: Symmetrization 효과

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# 간단한 H: threshold classifiers
# D: X ~ U[0, 1], Y = sign(X - 0.5) with 10% noise

def sample_D(n):
    X = rng.uniform(0, 1, n)
    y_clean = (X >= 0.5).astype(int)
    flip = rng.random(n) < 0.1
    Y = np.where(flip, 1 - y_clean, y_clean)
    return X, Y

ns = [50, 100, 200, 500, 1000]
n_trials = 500

gaps_S_D = []     # |L_S(h) - L_D(h)|
gaps_S_Sp = []    # |L_S(h) - L_S'(h)|

for n in ns:
    gap_sd, gap_ssprime = [], []
    
    for _ in range(n_trials):
        X, Y = sample_D(n)
        X2, Y2 = sample_D(n)
        
        # 최적 threshold를 "알고 있다"고 가정 (h*(x) = 1 iff x >= 0.5)
        h = lambda x: (x >= 0.5).astype(int)
        
        L_S = np.mean(Y != h(X))
        L_D = 0.1  # 알려진 Bayes risk
        L_Sp = np.mean(Y2 != h(X2))
        
        gap_sd.append(abs(L_S - L_D))
        gap_ssprime.append(abs(L_S - L_Sp))
    
    gaps_S_D.append(np.mean(gap_sd))
    gaps_S_Sp.append(np.mean(gap_ssprime))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(ns, gaps_S_D, 'o-', label=r'$\mathbb{E}[|L_S - L_D|]$')
ax1.plot(ns, gaps_S_Sp, 's-', label=r'$\mathbb{E}[|L_S - L_{S\'}|]$')
ax1.plot(ns, [1/np.sqrt(n) for n in ns], '--', label=r'$1/\sqrt{n}$')
ax1.set_xlabel('n')
ax1.set_ylabel('Expected gap')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('Symmetrization: $|L_S - L_D|$ vs $|L_S - L_{S\'}|$')

# 상관관계 시각화
ax2.scatter(gaps_S_D, gaps_S_Sp, alpha=0.3)
ax2.set_xlabel(r'$|L_S - L_D|$')
ax2.set_ylabel(r'$|L_S - L_{S\'}|$')
ax2.set_title('Symmetrization의 근사 품질')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# → gaps_S_Sp가 gaps_S_D와 유사함을 확인
```

### 실험 2: VC bound vs 실제 gap

```python
# Threshold (VC=1): Π(2n) = 2n+1

ns = [10, 50, 100, 500, 1000]
epsilons = np.linspace(0.05, 0.3, 10)

for n in ns:
    # Hoeffding+Union bound for 유한 H (union over Π(2n) dichotomies)
    pi_2n = 2*n + 1
    bound = lambda eps: 4 * pi_2n * np.exp(-n * eps**2 / 8)
    
    # 경험적: 실제 supremum gap 측정
    empirical_gaps = []
    for trial in range(100):
        X, Y = sample_D(n)
        gap = abs(np.mean(Y) - 0.9)  # L_D의 근사
        empirical_gaps.append(gap)
    
    emp_max = np.max(empirical_gaps)
    
    # Bound에서 역으로 필요한 ε 계산
    bound_eps = [eps for eps in epsilons if bound(eps) < 0.1]
    bound_epsilon = bound_eps[0] if bound_eps else 0.3
    
    print(f"n={n:5d}: 경험 gap ≈ {emp_max:.4f}, "
          f"이론 bound({bound_epsilon:.3f}) ≈ 0.1")

# → 경험적 gap이 이론 bound 안에 있음을 확인
```

---

## 🔗 ML 알고리즘 연결

VC bound는 **모든 ERM 기반 알고리즘**에 적용:

- **Linear regression/SVM**: VC = $d+1$ → $m = O(d/\epsilon^2)$
- **Decision tree**: VC = $O(\text{leaves})$ → regularized tree depth
- **Neural network**: VC = $O(W \log W)$ → **bound가 vacuous** (W >> n)
- **Ensemble methods**: 각 기본 학습기의 VC 조합

---

## ⚖️ 가정과 한계

1. **Symmetrization의 slack**: "worst-case bound"이므로 constant 2가 들어감 — 실제 data에선 더 tight할 수 있음.
2. **High-dimensional regimes**: $d >> n$이면 $\Pi(2n) \approx n^d \gg 1$ → bound가 의미 없음.
3. **Distribution-dependent**: VC bound는 worst-case 분포를 가정 — 특정 분포에선 Rademacher(Ch5)나 margin bound가 더 좋음.

---

## 📌 핵심 정리

- **Symmetrization**: Ghost sample $S'$를 도입해 $L_\mathcal{D}$ 의존을 제거 → empirical bound로 전환.
- **유한화**: $\sup_{h \in \mathcal{H}}$를 $\sup_{h \in \mathcal{H}|_{S \cup S'}}$로 제한 (크기 $\leq \Pi(2n)$).
- **Union Bound**: 유한 집합 위에서 Hoeffding 적용 → $4 \Pi_\mathcal{H}(2n) e^{-n\epsilon^2/8}$.
- **Sample complexity**: Sauer-Shelah + 역산 → $m = O((d \log(1/\epsilon) + \log(1/\delta))/\epsilon^2)$.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> Symmetrization Lemma에서 상수 2가 나오는 이유를 설명하라.</summary>

<br/>

**해설**. Markov 부등식의 적용:
$$\mathbb{E}[X] \geq t \cdot \mathbb{P}(X \geq t).$$

$X = \sup_h |L_S(h) - L_{S'}(h)|$, $t = \epsilon/2$라 하면:
$$\mathbb{E}[|L_S(h) - L_\mathcal{D}(h)|] \geq (\epsilon/2) \cdot \mathbb{P}(|L_S(h) - L_{S'}(h)| \geq \epsilon/2).$$

삼각부등식으로 $\mathbb{E}[|L_S - L_\mathcal{D}|] \leq \mathbb{E}[|L_S - L_{S'}|]$이므로, 상수 2가 emerge한다. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> "Union over finite dichotomies" 단계에서 $|\mathcal{H}|_{S \cup S'}| \leq \Pi_\mathcal{H}(2n)$이 성립하는 이유를 설명하라.</summary>

<br/>

**해설**. $\mathcal{H}|_{S \cup S'}$는 $S \cup S'$ (크기 $2n$) 위에서 실현되는 dichotomy들의 집합이다. 성장함수의 정의에 의해:

$$\Pi_\mathcal{H}(2n) = \max_{|C|=2n} |\mathcal{H}|_C|.$$

따라서 어떤 특정 점집합 $S \cup S'$에서도 $|\mathcal{H}|_{S \cup S'}| \leq \Pi_\mathcal{H}(2n)$. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> VC bound에서 "고정된 가설 $h$에 대해 Hoeffding"을 쓸 수 있지만, "ERM 해 $\hat{h}$"에는 Hoeffding을 직접 못 쓰는 이유는?</summary>

<br/>

**해설**. $\hat{h} = \hat{h}(S)$는 **데이터에 의존적(data-dependent)**이기 때문이다. Hoeffding은 "$h$ 고정" 가정 하에 $\mathbb{E}_{(X,Y) \sim \mathcal{D}}[\ell(h(X), Y)]$가 $L_\mathcal{D}(h)$라는 것에 기반한다. 

하지만 $\hat{h}$를 찾기 위해 $S$를 봤으므로, $\hat{h}$가 $S$에 "최적화"되어 있다 → bias 발생. 이를 통제하려면 **Union Bound** (모든 가능한 $h$에 대해) 또는 **symmetrization** (empirical quantity로 보정)이 필요하다. 이것이 Ch3부터의 핵심 동기. $\square$

</details>

---

<div align="center">

◀ [이전: 04. Sauer-Shelah](./04-sauer-shelah.md) | [📚 README](../README.md) | [다음: 06. Covering ▶](./06-epsilon-net-covering.md)

</div>
