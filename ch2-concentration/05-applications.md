# 05. 집중부등식의 ML 응용 정리

## 🎯 핵심 질문

- **Cross-validation error**: 테스트 폴드의 오차가 CV 추정을 얼마나 정확히 대표하는가? 어떤 부등식을 쓰는가?
- **Bootstrap confidence interval**: Bootstrap이 수렴하는 이유가 집중부등식인가? 어떤 형태인가?
- **Online learning regret**: Bandit·강화학습의 regret bound는 어떤 집중 부등식을 기반으로 하는가?
- **Random forest splitting**: Hoeffding tree가 왜 split 결정에 Hoeffding을 쓰는가?
- **부등식 선택의 의사결정**: 어떤 상황에서 Hoeffding/Bernstein/McDiarmid를 쓸지, 무엇을 고려하는가?

---

## 🔍 왜 이 정리가 필요한가

**"어떤 집중부등식을 쓸지"** — ML 실전에서 가장 자주 하는 질문이다. Ch2-01부터 04까지 다섯 개의 부등식을 배웠지만, 각각이 언제 나타나고 왜 선택되는지를 **정리되지 않은 상태**로는 알 수 없다.

이 문서는:
1. **5개 부등식의 적용 시나리오** 명확화
2. **상황별 비교표**: 어느 것이 tighter, 계산이 용이, 실무 적용 가능?
3. **실제 알고리즘 예시**: CV, bootstrap, online learning, random forest에서의 정확한 적용
4. **고차 정리**: "부등식의 위계"와 "언제 upgrade되는가"

---

## 📐 수학적 선행 조건

- Ch1-01, Ch1-03: 위험, ERM 정의
- Ch2-01부터 04: 모든 집중부등식
- 기초: 교차검증, bootstrap, online learning의 개념 수준

---

## 📖 직관적 이해

### 부등식의 "위계"

```
Markov (O(1/t))
  ↓ (분산 정보 추가)
Chebyshev (O(1/t²))
  ↓ (구간 정보 + Chernoff)
Hoeffding (e^{-2nt²})
  ↓ (특정 함수 구조 + martingale)
McDiarmid (e^{-2t²/Σc_i²}) — 더 일반적
  ↓ (분산 정보 다시)
Bernstein (e^{-nt²/(2σ²+...)}) — 가장 정교함
```

**낮은 단계 부등식**: 조건 느슨, bound 약함. **높은 단계**: 조건 까다로움, bound 강함.

### 각 부등식의 "신호"

| 부등식 | 마주할 상황 | 힌트 |
|--------|-----------|------|
| **Markov** | "분포를 전혀 모른다" + 비음수 | 조건이 거의 없음 → bound 약함 |
| **Chebyshev** | 분산은 알지만 분포 모른다 | 분산 값만 집어넣기 |
| **Hoeffding** | 표본 평균, 모든 범위 알려짐 | "경험 위험" 하면 자동 떠올림 |
| **McDiarmid** | 일반 함수, 안정성 확인 가능 | CV 오차, bootstrap, 최댓값 같은 복잡 함수 |
| **Bernstein** | 분산이 작은 특수 상황 | "우리 데이터는 분산 작아요" → 더 tight 필요 |

---

## ✏️ 엄밀한 정의

### 정의 2.9 (집중부등식의 분류)

| 형태 | 설명 | 적용 |
|-----|-----|-----|
| **확률 형식** | $\mathbb{P}(\|X - \mu\| \geq t)$ | 구체적 값 bound 필요 |
| **고확률 형식** | $\mathbb{P}(\|X - \mu\| \geq t) \leq \delta$ → $t = f(\delta)$ | 신뢰 구간 유도 |
| **기대값 형식** | $\mathbb{E}[\|X - \mu\|^p] \leq g(p)$ | 고차 적률 제어 |

---

## 🔬 응용 사례 및 정리

### 응용 1: Cross-Validation Error Bound

**상황**: $K$-fold CV로 모델 평가. 각 폴드의 test error $L_k$.

**사용 부등식**: **McDiarmid (bounded differences)**

**이유**: CV 오차 $\hat{L}_{\text{CV}} = \frac{1}{K}\sum L_k$는 하나의 폴드를 바꿔도 최대 $1/K$ 변함. Bounded diff 조건 만족.

**공식**:
$$\mathbb{P}(\|L_{\text{CV}} - \mathbb{E}[L_{\text{CV}}]\| \geq t) \leq 2\exp\left(-\frac{2K^2 t^2}{\sum c_i^2}\right) = 2\exp(-2Kt^2)$$

(모든 $c_i = 1/K$이므로)

**신뢰 구간** (high-prob form): 확률 $\geq 1 - \delta$로
$$L_{\text{CV}} \in \left[\mathbb{E}[L_{\text{CV}}] - \sqrt{\frac{\log(2/\delta)}{2K}}, \mathbb{E}[L_{\text{CV}}] + \sqrt{\frac{\log(2/\delta)}{2K}}\right].$$

**Code 예시**:
```python
# K-fold CV
scores = cross_val_score(clf, X, y, cv=K)
cv_mean = scores.mean()
delta = 0.05
bound = np.sqrt(np.log(2/delta) / (2*K))
ci_lower, ci_upper = cv_mean - bound, cv_mean + bound
# 확률 ≥ 95%로 진짜 error는 [ci_lower, ci_upper]
```

---

### 응용 2: Bootstrap Confidence Interval

**상황**: Bootstrap으로 통계량 $T(X)$ (예: 중앙값)의 분포 추정.

**사용 부등식**: **McDiarmid** (bootstrap sample의 한 점 변화)

**이유**: Bootstrap statistics $T^*(X^*)$도 bounded differences. 한 resampling에서 샘플 하나 변화 → 통계량 변화 ≤ $c$ (e.g., 중앙값은 천천히 변함).

**공식** (Efron & Tibshirani):
$$\mathbb{P}(\|T^* - T\| \geq t) \leq 2\exp(-2t^2 / \text{empirical variance}).$$

실제로는 **empirical bootstrap distribution**에서 percentile 직접 사용 (더 간단).

---

### 응용 3: Online Learning / Bandit Regret

**상황**: $T$ 라운드 온라인 의사결정. 각 라운드 손실 $\ell_t \in [0, 1]$.

**누적 regret**: $\text{Regret} = \sum_{t=1}^T (\ell_t - \ell_t^*)$

**사용 부등식**: **Bernstein** (variance-dependent regret)

**공식** (Variance-aware MAB, Audibert et al.):
$$\mathbb{E}[\text{Regret}] \leq O(\sqrt{VT \log K} + \log K)$$

여기서 $V = \sum_{t=1}^T \text{Var}(\ell_t)$ (손실 분산의 합).

**비교**: Hoeffding 기반 regret $O(\sqrt{T \log K})$보다 variance-aware $O(\sqrt{VT \log K})$이 low-variance 상황에서 우월.

**직관**: 손실이 안정적(분산 작음)이면 빠르게 수렴.

---

### 응용 4: Hoeffding Tree (Random Forest의 기초)

**상황**: Streaming data에서 트리를 온라인으로 증가. Split 결정 시 어느 feature가 좋은가?

**사용 부등식**: **Hoeffding**

**공식**: Feature $a, b$의 정보이득 차이 $\Delta = IG(a) - IG(b)$. Hoeffding으로
$$\mathbb{P}(\|\Delta - \hat{\Delta}\| \geq \epsilon) \leq 2\exp(-2n\epsilon^2/R^2)$$

여기서 $R = \log(\text{# classes})$.

Split 결정: $\hat{\Delta} - \tau > 0$일 때 (여기서 $\tau$는 Hoeffding bound). 그러면 "충분히 높은 확률로 feature $a$가 진짜 더 좋다"는 보증.

**why Hoeffding?**: 정보이득은 범위가 bounded, 한 번에 한 샘플만 추가되므로 Hoeffding 형태.

---

### 응용 5: Concentration of Rademacher Complexity (Ch5-02로 연결)

**상황**: 가설공간 $\mathcal{H}$의 Rademacher 복잡도 $\mathcal{R}_n(\mathcal{H})$ 추정.

**사용 부등식**: **McDiarmid** (한 샘플 영향 bounded)

**공식**: Empirical Rademacher $\hat{\mathcal{R}}_S(\mathcal{H}) = \frac{1}{n}\max_{h \in \mathcal{H}} \sum \sigma_i h(x_i)$에 대해
$$\mathbb{P}(\|\hat{\mathcal{R}}_S - \mathbb{E}[\hat{\mathcal{R}}_S]\| \geq t) \leq 2\exp\left(-2n^2 t^2 / n^2\right) = 2\exp(-2t^2)$$

($c_i = 1/n$ 각각, $\sum c_i^2 = 1/n$)

**의미**: Rademacher complexity **자체의 집중**을 보장. 이것이 Ch5-02에서 "일반화 bound를 Rademacher로 표현"할 수 있게 하는 이유.

---

## 💻 비교 표 및 의사결정 가이드

### 부등식 선택 체크리스트

```
Q1: 분포를 아는가?
├─ 아니오 → Q2로
└─ 예 (정규분포 등) → 분포별 특화 bound (이 문서 범위 밖)

Q2: 대상이 표본 평균인가?
├─ 예 → Q3로
└─ 아니오 (일반 함수) → Q4로

Q3: 구간 범위 [a, b]를 아는가?
├─ 예 → Hoeffding (기본)
│   └─ 분산도 작은가? → Bernstein (더 tight)
└─ 아니오 → Chebyshev (분산만 사용)

Q4: 함수가 한 좌표 변화에 bounded되는가?
├─ 예 (CV, bootstrap, Rademacher) → McDiarmid
└─ 아니오 → Markov (최후의 수단)
```

### 실전 비교표

| 상황 | 추천 부등식 | 이유 | 구현 난이도 |
|-----|-----------|------|-----------|
| **학습 이론 (PAC)** | Hoeffding | 표본 평균, 범위 알려짐 | ⭐ |
| **CV score** | McDiarmid | 한 폴드 변화 bounded | ⭐⭐ |
| **Bootstrap** | McDiarmid | Resampling stability | ⭐⭐ |
| **Online regret** | Bernstein | Variance-aware | ⭐⭐⭐ |
| **Hoeffding tree** | Hoeffding | Information gain | ⭐⭐ |
| **Rademacher conc.** | McDiarmid | 샘플별 영향 bound | ⭐⭐ |
| **분산 모를 때** | Chebyshev | 조건 느슨 | ⭐ |
| **아무 정보 없을 때** | Markov | 최악의 경우 | ⭐ |

---

## 🔗 ML 알고리즘 연결 (정리)

```
┌─────────────────────────────────────────────────────┐
│     집중부등식의 ML 응용 흐름도                      │
└─────────────────────────────────────────────────────┘

1단계: 고정 h의 위험 (Ch1, Ch2 이 문서)
   ↓ Hoeffding: P(|L_S(h) - L_𝒟(h)| ≥ ε) ≤ 2e^{-2nε²}

2단계: 유한 H의 균일 수렴 (Ch3, PAC)
   ↓ Union bound: P(sup_h |...| ≥ ε) ≤ |H|·2e^{-2nε²}
   ↓ m = O((log|H| + log(1/δ))/ε²)

3단계: 무한 H의 균일 수렴 (Ch4, VC)
   ↓ Sauer-Shelah: |H| → π_H(m) ≤ (em/d)^d
   ↓ VC bound: P(sup_h |...| ≥ ε) ≤ 4π_H(2n)e^{-nε²/8}

4단계: Rademacher 복잡도 (Ch5)
   ↓ McDiarmid: Rademacher 자체의 집중
   ↓ Symmetrization + Contraction
   ↓ gap ≤ 2R_n + O(√(log(1/δ)/n))

5단계: Algorithmic stability (Ch6)
   ↓ Hoeffding + stability definition
   ↓ β-stable ⟹ gap ≤ β

6단계: SRM & model selection (Ch7)
   ↓ Penalty 유도 (AIC, BIC, MDL)
```

---

## ⚖️ 가정과 한계

1. **iid 가정**: 시계열·분포 이동 상황에서는 다른 부등식 필요 (mixing coefficient 등).
2. **구간 정보 필수**: Hoeffding은 범위를 알아야 함. 알 수 없으면 truncation (손실 발생).
3. **tight하지 않을 수 있음**: 모든 부등식이 "최악의 경우"를 본다. 실제 데이터는 훨씬 타이트할 수 있음.
4. **분포 자유의 대가**: 분포를 활용 안 하는 대신 bound가 약함 (예: DL 상황에서 vacuous).
5. **고차 확률**: 여러 hypothesis, 여러 parameter 튜닝하면 union bound로 exponential blow-up.

---

## 📌 핵심 정리

### 부등식 간단 정리

| 이름 | 공식 | 조건 | 적용 |
|-----|------|------|-----|
| **Markov** | $P(X \geq t) \leq \mathbb{E}[X]/t$ | $X \geq 0$ | 최후 수단 |
| **Chebyshev** | $P(\|X-\mu\| \geq t) \leq \sigma^2/t^2$ | 분산 알려짐 | 분포 무관 |
| **Hoeffding** | $P(\|\bar{X}-\mu\| \geq t) \leq 2e^{-2nt^2}$ | 범위 $[a,b]$ | 표본 평균 |
| **McDiarmid** | $P(\|f-\mathbb{E}[f]\| \geq t) \leq 2e^{-2t^2/\sum c_i^2}$ | Bounded diff | 일반 함수 |
| **Bernstein** | $P(\|\bar{X}-\mu\| \geq t) \leq 2e^{-nt^2/(2\sigma^2+2Mt/3)}$ | 범위+분산 | 저분산 regime |

### 선택 기준

1. **표본 평균**: Hoeffding → (분산 작으면) Bernstein
2. **일반 함수**: McDiarmid (bounded differences 확인)
3. **아무 정보**: Markov
4. **분포 구체적**: 분포별 특화 부등식

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> 5-fold CV에서 각 폴드의 오차 $L_k \in [0, 1]$. 평균 CV score $\bar{L} = \frac{1}{5}\sum L_k$의 95% confidence interval을 McDiarmid로 유도하라.</summary>

<br/>

**해설**. McDiarmid: 한 폴드 $L_k$를 바꿔도 $\bar{L}$은 최대 $1/5$ 변함. 따라서 $c_i = 1/5$. $\sum c_i^2 = 5 \cdot (1/5)^2 = 1/5$.

확률 95%로 $\delta = 0.05$:
$$t: 2e^{-2 \cdot 5 \cdot t^2} = 0.05 \Rightarrow e^{-10t^2} = 0.025 \Rightarrow t \approx \sqrt{-\ln(0.025)/10} \approx 0.31.$$

신뢰 구간: $[\bar{L} - 0.31, \bar{L} + 0.31]$ — 단위가 [0, 1]이므로 이는 매우 넓다. 더 나은 bound는 Hoeffding version using individual fold errors. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> Bernoulli(p=0.01)에서 표본 크기 $n$에 따라 Hoeffding과 Bernstein의 bound가 crossover하는 지점을 찾아라. 언제부터 Bernstein이 더 tight한가?</summary>

<br/>

**해설**. $\sigma^2 = 0.01 \cdot 0.99 \approx 0.01$, $M = 1$, $t = 0.01$.

- Hoeffding: $2e^{-2n(0.01)^2} = 2e^{-0.0002n}$
- Bernstein: $2e^{-n(0.01)^2/(2 \cdot 0.01 + 2 \cdot 1 \cdot 0.01/3)} = 2e^{-n \cdot 0.0001/(0.02 + 0.00667)} \approx 2e^{-0.00345n}$

Bernstein이 약 17배 더 빠르게 감소. 모든 $n$에서 Bernstein이 이기는 상황. 일반적으로 $\sigma^2 \ll M^2/4$이면 항상 Bernstein 우월. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> Random forest의 split 선택: 1000 샘플, 두 feature의 information gain 차이 $\hat{\Delta} = 0.05$. Hoeffding으로 이 feature가 진짜로 더 나은지 (95% 확률로) 결정할 수 있는가?</summary>

<br/>

**해설**. Hoeffding: 정보이득 범위는 $[0, \log(K)]$ (K = class 수). 보수적으로 $R = \log 2 = 0.693$.

$$2e^{-2 \cdot 1000 \cdot \epsilon^2 / (0.693)^2} = 0.05 \Rightarrow \epsilon \approx 0.024.$$

$\hat{\Delta} - \epsilon = 0.05 - 0.024 = 0.026 > 0$ → 충분히 높은 확률로 feature가 더 좋다. 하지만 여러 feature를 시도하면 (예: 50개), union bound로 bound가 50배 느슨해진다. 이 문제가 **Hoeffding tree가 practical하려면 "충분히 큰 n"이나 "early stopping" 필요**한 이유.

**개선**: Bonferroni correction이나 adaptive significance level (e.g., Holm-Bonferroni) 사용. $\square$

</details>

---

<div align="center">

◀ [이전: 04. Bernstein 부등식](./04-bernstein.md) | [📚 README](../README.md) | [다음: Ch3 PAC Learning ▶](../ch3-pac-learning/01-pac-definition.md)

</div>
