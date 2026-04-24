# 05. IID 가정과 그것이 깨지는 경우

## 🎯 핵심 질문

- 왜 SLT의 모든 경계는 **iid 가정**에서 출발하는가 — 이 가정이 깨지면 어떤 단계에서 정확히 무엇이 고장 나는가?
- **시계열 데이터**(stock price, sensor)에서 iid가 깨지는 방식과 이를 완화하는 **mixing coefficient** 이론은 무엇인가?
- **Covariate shift**(훈련·테스트 분포의 $p(X)$는 다르지만 $p(Y|X)$는 같음)와 **concept drift**($p(Y|X)$가 변함)의 구분이 왜 실무적으로 중요한가?
- **Domain adaptation**·**importance weighting**·**adversarial training**은 각각 iid 위반의 어떤 양상을 다루는 해법인가?
- **Non-iid concentration**(Bernstein for martingales, block-wise Hoeffding)의 기본 아이디어는?

---

## 🔍 왜 iid 가정의 위반이 중요한가

이 레포의 Ch2~Ch7 모든 정리는 "**iid 샘플 $S \sim \mathcal{D}^n$**"으로 시작한다. 이것이 틀리면 어떻게 되는가? Hoeffding은 $\mathbb{E}[e^{\lambda(X_i - \mu)}] \leq e^{\lambda^2(b-a)^2/8}$의 **독립 곱 분해** $\mathbb{E}[e^{\lambda \sum(X_i - \mu)}] = \prod \mathbb{E}[e^{\lambda(X_i - \mu)}]$에서 유도된다 — 독립성이 무너지면 **첫 줄부터 깨진다**. VC bound의 symmetrization도 "ghost sample $S'$가 $S$와 같은 분포"라는 동일분포 가정에 의존한다.

실전에서 iid가 완전히 성립하는 경우는 **매우 드물다** — 뉴스 기사는 시간 순서가 있고, 의료 데이터는 병원별 편향이 있으며, 이미지는 수집 편향(데이터 증강 전후가 iid 아님)이 있다. 그럼에도 SLT 경계가 **근사적으로 유효**한 이유는 (1) **효과적 iid**를 만드는 전처리(shuffle, batch sampling, temporal split), (2) **mixing 조건**으로 "느린 상관"을 허용하는 일반화 이론, (3) **분포 이동 특화 이론**(covariate/label shift)이 발달했기 때문이다. 이 문서는 iid 위반의 **정확한 분류**와 각 경우의 **수학적 대응**을 정리한다 — Ch2~Ch7 모든 경계의 **적용 범위**를 명확히 하기 위해.

---

## 📐 수학적 선행 조건

- Ch1-01~04 전체
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): martingale, stationary process, $\beta$-mixing
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): Likelihood ratio, importance weighting
- 기초: 조건부 기대값, Markov property

---

## 📖 직관적 이해

### iid의 두 얼굴

"iid"는 두 개의 가정이다:
- **Independent**: $S$의 샘플들이 서로 정보를 주지 않음. $(X_i, Y_i)$가 $(X_j, Y_j)$에 대해 독립 ($i \neq j$).
- **Identically distributed**: 모든 샘플이 **같은** $\mathcal{D}$에서 뽑힘. 훈련·테스트 간 분포 동일.

두 가정은 **독립적으로** 깨질 수 있고, 깨지는 방식이 다르면 대응이 다르다.

| 위반 유형 | 예시 | 무엇이 깨지는가 | 대응 |
|----------|------|----------------|------|
| **Temporal dependence** (독립 $\times$) | 주가, 언어 | Hoeffding의 $\prod \mathbb{E}[e^{\lambda X_i}]$ | mixing, block-wise |
| **Covariate shift** ($p(X)$ 변화) | 훈련: 맑은 날, 테스트: 비 | $\mathbb{E}_{\mathcal{D}_\text{test}}$ 계산 | importance weighting |
| **Label shift** ($p(Y)$ 변화) | 팬데믹 전후 증상 분포 | base rate 변화 | label shift 보정 (Lipton 2018) |
| **Concept drift** ($p(Y\|X)$ 변화) | 스팸 메일의 진화 | 고정 $\mathcal{D}$ 가정 자체 | online learning, 지속 학습 |
| **Selection bias** | 생존 편향 | $S \sim \mathcal{D}^n$ 가정 | causal inference, propensity |

### 왜 "약간의 위반"은 SLT를 부수지 않는가

SLT는 종종 **점근적**으로 robust하다. 예를 들어 $\beta$-mixing sequence에서 **effective sample size** $n_{\text{eff}} \approx n/(1 + \tau)$ ($\tau$는 mixing time)로 바꾸면 많은 Hoeffding-류 bound이 생존한다. Rate는 $\sqrt{\log|\mathcal{H}|/n_{\text{eff}}}$. 즉 iid 위반은 "**속도 감소**"로 나타난다. 완전한 independence를 요구하는 것이 아니라 "**상관이 빠르게 감쇠**"하면 된다.

---

## ✏️ 엄밀한 정의

### 정의 5.1 (IID 샘플)

$\{(X_i, Y_i)\}_{i=1}^n$가 iid $\sim \mathcal{D}$ ⇔:
- **Independent**: 결합분포가 주변분포의 곱 — $\mathbb{P}((X_1, Y_1), \ldots, (X_n, Y_n)) = \prod_{i=1}^n \mathbb{P}(X_i, Y_i)$.
- **Identically distributed**: 모든 $i$에 대해 $(X_i, Y_i) \sim \mathcal{D}$.

### 정의 5.2 (시계열과 Stationarity)

$\{Z_t\}_{t \in \mathbb{Z}}$가 **strictly stationary** ⇔ 모든 $k, t_1, \ldots, t_m \in \mathbb{Z}$에 대해
$$(Z_{t_1}, \ldots, Z_{t_m}) \stackrel{d}{=} (Z_{t_1 + k}, \ldots, Z_{t_m + k}).$$

"동일분포"의 일반화. 그러나 **독립이 아님**.

### 정의 5.3 ($\beta$-mixing coefficient)

시점 간격 $\tau$에서의 의존도:
$$\beta(\tau) := \sup_{t} \mathbb{E}\!\left[\sup_{A \in \sigma(Z_{s}: s \geq t+\tau)} |\mathbb{P}(A | \mathcal{F}_t) - \mathbb{P}(A)|\right],$$
여기서 $\mathcal{F}_t = \sigma(Z_s: s \leq t)$는 과거 $\sigma$-field. 시퀀스가 $\beta$-mixing ⇔ $\beta(\tau) \to 0$ as $\tau \to \infty$ (geometric/polynomial rate).

**직관**: $\tau$ 떨어진 시점들이 얼마나 "거의 독립"인가.

### 정의 5.4 (분포 이동의 분류)

훈련 분포 $\mathcal{D}^{\text{tr}}(X, Y)$, 테스트 분포 $\mathcal{D}^{\text{te}}(X, Y)$ 둘 다 존재. 다음 분해
$$\mathcal{D}(X, Y) = p(X) \cdot p(Y | X) = p(Y) \cdot p(X | Y)$$
의 어느 요인이 변하는지에 따라:
- **Covariate shift**: $p^{\text{tr}}(X) \neq p^{\text{te}}(X)$, $p^{\text{tr}}(Y|X) = p^{\text{te}}(Y|X)$.
- **Label shift**: $p^{\text{tr}}(Y) \neq p^{\text{te}}(Y)$, $p^{\text{tr}}(X|Y) = p^{\text{te}}(X|Y)$.
- **Concept drift**: $p^{\text{tr}}(Y|X) \neq p^{\text{te}}(Y|X)$.
- **Joint shift**: 위 셋의 조합.

---

## 🔬 정리와 증명

### 정리 5.1 (독립성이 Hoeffding 증명의 어느 단계에서 쓰이는가)

독립 iid $X_i \in [a, b]$, Hoeffding:
$$\mathbb{E}[e^{\lambda \sum(X_i - \mu)}] = \prod_{i=1}^n \mathbb{E}[e^{\lambda (X_i - \mu)}] \leq \prod_{i=1}^n e^{\lambda^2(b-a)^2/8} = e^{n \lambda^2 (b-a)^2/8}.$$

**독립성이 없어지면**: 첫 등호 $\mathbb{E}[\prod] = \prod \mathbb{E}[\cdot]$가 깨진다.

**동일분포성이 없어지면**: 각 $\mathbb{E}[e^{\lambda(X_i - \mu)}]$를 동일하게 bound할 수 없음 ($\mu_i$가 다름).

**증명**: Ch2-02 참조. 여기서는 "어디가 깨지는가"만 짚는 관찰. $\square$

### 정리 5.2 ($\beta$-mixing sequence의 Bernstein-type 부등식 (Yu 1994))

$\{Z_t\}$가 stationary, $\beta$-mixing with coefficient $\beta(\tau)$, $Z_t \in [0, M]$, $\mu = \mathbb{E}[Z_t]$. Blocking technique:
$$\mathbb{P}\!\left(\left|\frac{1}{n}\sum_{t=1}^n Z_t - \mu\right| \geq \epsilon\right) \leq 2 \exp\!\left(-\frac{n_{\text{eff}} \epsilon^2}{2 M^2}\right) + n \beta(\tau_n),$$
여기서 $n_{\text{eff}} = n/(2\tau_n)$, $\tau_n$는 선택 파라미터.

**증명 스케치**. 시퀀스를 **길이 $\tau_n$의 블록 $B_1, \ldots, B_k$**로 자르고 홀수/짝수 블록만 본다. "$\tau_n$ 간격의 블록은 거의 독립"이라는 $\beta$-mixing 정의 → Berbee's coupling으로 **iid 블록 시퀀스로 교체**. iid Hoeffding 적용 후 coupling error $n \beta(\tau_n)$ 더함. $\square$

> **해석**: $\tau_n$을 기하적으로 선택(예: $\tau_n = \log n$)하면 $n \beta(\tau_n) = o(1)$이 되고, $n_{\text{eff}} = n/\log n$. **rate가 $\log n$ 인수만 손해**. 즉 약한 의존성에서 SLT는 대체로 살아남는다.

### 정리 5.3 (Covariate shift 하의 importance-weighted ERM)

$p^{\text{tr}}(X)$·$p^{\text{te}}(X)$는 다르고 $p(Y|X)$는 같다고 가정. 테스트 risk:
$$L_{\mathcal{D}^{\text{te}}}(h) = \mathbb{E}_{X \sim p^{\text{te}}, Y | X}[\ell(h(X), Y)] = \mathbb{E}_{X \sim p^{\text{tr}}, Y | X}\!\left[\frac{p^{\text{te}}(X)}{p^{\text{tr}}(X)} \ell(h(X), Y)\right].$$

따라서 **importance-weighted ERM**:
$$\hat{h} = \arg\min_h \frac{1}{n} \sum_{i=1}^n \frac{p^{\text{te}}(X_i)}{p^{\text{tr}}(X_i)} \ell(h(X_i), Y_i)$$
은 $L_{\mathcal{D}^{\text{te}}}(h)$의 비편향 경험 추정량이다 (가중치가 known일 때).

**증명**. 측도의 Radon-Nikodym 유도: $p^{\text{te}}(X) / p^{\text{tr}}(X)$이 $p^{\text{tr}} \to p^{\text{te}}$ 변환의 likelihood ratio. $\int f \, dp^{\text{te}} = \int f \, (dp^{\text{te}}/dp^{\text{tr}}) \, dp^{\text{tr}}$. 경험추정은 LLN. $\square$

> **실무 함정**: 가중치 분산 $\text{Var}(p^{\text{te}}/p^{\text{tr}})$이 크면 ("overlap이 나쁨") 경험 추정의 분산이 폭발. 이 경우 **domain-invariant features** (Ganin et al. 2015) 같은 alternatives가 필요.

### 정리 5.4 (Concept drift 하의 불가능성)

$p^{\text{tr}}(Y|X) \neq p^{\text{te}}(Y|X)$에 관한 가정 없으면, $n \to \infty$라도 $L_{\mathcal{D}^{\text{te}}}$ 최소화는 **불가능**.

**증명 스케치**. 반례: $\mathcal{X} = \{0, 1\}$, 훈련에서 $Y = X$, 테스트에서 $Y = 1 - X$. 어떤 $\hat{h}: \{0,1\} \to \{0,1\}$도 $L_{\mathcal{D}^{\text{te}}}(\hat{h}) \geq 1/2$ (훈련이 테스트의 **반대**). NFL와 유사한 구조. $\square$

이것이 **concept drift**에서 iid-SLT가 **본질적 실패**하는 이유. 대응책은:
- **Online learning**: 매 라운드 손실을 관측 → regret bound (drift rate와 연계)
- **Continual learning**: 과거 지식 유지하며 적응

---

## 💻 NumPy 구현 검증

### 실험 1: AR(1) 시계열에서 Hoeffding bound의 loosening

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

def ar1(n, phi, sigma=1.0):
    """AR(1): Z_{t+1} = phi * Z_t + sigma * eps_t"""
    Z = np.zeros(n)
    Z[0] = rng.standard_normal() * sigma / np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        Z[t] = phi * Z[t-1] + sigma * rng.standard_normal()
    return Z

n = 200
phis = [0.0, 0.3, 0.7, 0.95]
n_trials = 5000
t_grid = np.linspace(0, 1, 40)

fig, ax = plt.subplots(figsize=(9, 4.5))
for phi in phis:
    # 각 phi에서 (1/n) Σ Z_t가 0 근처에 집중되는 분포를 경험적으로 관찰
    means = np.array([ar1(n, phi).mean() for _ in range(n_trials)])
    emp = np.array([np.mean(np.abs(means) >= t) for t in t_grid])
    ax.semilogy(t_grid, emp, 'o-', label=f'φ={phi} emp', alpha=0.7)

# iid Hoeffding bound (scale σ_eff ≈ 1/(1-φ^2))
# 단 유계 iid에만 적용이므로 참고 — 실제는 sub-Gaussian
hoeff = 2 * np.exp(-2 * n * t_grid ** 2 / 4)  # [−1,1] 대략
ax.semilogy(t_grid, hoeff, 'k--', label='iid Hoeffding (참고)', alpha=0.8)
ax.set_xlabel('t'); ax.set_ylabel('P(|mean| ≥ t)')
ax.set_title('AR(1) 의존성 증가 → Hoeffding의 암묵적 iid 가정이 깨짐')
ax.legend(fontsize=8); plt.tight_layout(); plt.show()

# → φ가 0에서 0.95로 가면 같은 n이어도 mean의 분산이 훨씬 큼.
#   "iid Hoeffding rate"는 강한 의존성에서 과도하게 optimistic.
#   Effective sample size: n_eff ≈ n(1-φ)/(1+φ).
for phi in phis:
    n_eff = n * (1 - phi) / (1 + phi)
    print(f'φ={phi}: n_eff ≈ {n_eff:.0f}')
```

### 실험 2: Covariate shift와 importance weighting

```python
# 훈련: X ~ N(0, 1), 테스트: X ~ N(1.5, 1). Y | X는 둘 다 sign(X + 0.3).
def p_tr(x): return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
def p_te(x): return np.exp(-0.5 * (x - 1.5) ** 2) / np.sqrt(2 * np.pi)

n = 500
X_tr = rng.standard_normal(n)
Y_tr = np.sign(X_tr + 0.3)

# 후보 분류기: h_theta(x) = sign(x - theta)
thetas = np.linspace(-1, 2, 200)

# Naive ERM (훈련 분포에서 loss 최소화) — covariate shift 무시
loss_naive = np.array([np.mean(np.sign(X_tr - th) != Y_tr) for th in thetas])
theta_naive = thetas[np.argmin(loss_naive)]

# Importance-weighted ERM
w = p_te(X_tr) / p_tr(X_tr)
loss_iw = np.array([np.mean(w * (np.sign(X_tr - th) != Y_tr)) for th in thetas])
theta_iw = thetas[np.argmin(loss_iw)]

# 진짜 테스트 risk (큰 테스트 샘플로 근사)
X_te = 1.5 + rng.standard_normal(50000)
Y_te = np.sign(X_te + 0.3)
test_risk = lambda th: np.mean(np.sign(X_te - th) != Y_te)

print(f'Naive ERM theta  = {theta_naive:.3f}, test risk = {test_risk(theta_naive):.4f}')
print(f'IW ERM theta     = {theta_iw:.3f}, test risk = {test_risk(theta_iw):.4f}')
print(f'True optimal theta = -0.3, test risk = {test_risk(-0.3):.4f}')
# → IW가 훨씬 test 최적 theta=-0.3에 가깝게 수렴.
```

### 실험 3: Mixing sequence에서 block resampling의 효과

```python
# AR(1) 시계열에서 "블록 bootstrap"으로 CI
def block_bootstrap(Z, block_size, n_resamples):
    n = len(Z)
    means = []
    n_blocks = n // block_size
    for _ in range(n_resamples):
        idx = rng.integers(0, n - block_size, n_blocks)
        resample = np.concatenate([Z[i:i+block_size] for i in idx])
        means.append(resample.mean())
    return np.array(means)

Z = ar1(1000, phi=0.7)
# iid bootstrap
iid_boot = np.array([rng.choice(Z, len(Z), replace=True).mean() for _ in range(2000)])
# block bootstrap (block size ~ correlation length)
blk_boot = block_bootstrap(Z, block_size=20, n_resamples=2000)

print(f'IID bootstrap std: {iid_boot.std():.4f}  (과소추정, 독립 가정)')
print(f'Block bootstrap std: {blk_boot.std():.4f}  (올바른 의존성 반영)')
print(f'True sample mean std: computed from multiple AR(1) realizations would show the truth')
# → block bootstrap이 시계열의 참 variance를 더 정확히 반영.
```

---

## 🔗 ML 알고리즘 연결

| 실무 시나리오 | iid 위반 유형 | 수학적 대응 |
|-------------|--------------|-----------|
| **시계열 예측** | temporal dependence | Temporal CV, blocked CV |
| **의료 이미지** (병원별) | covariate shift | Domain adaptation, DANN |
| **이메일 스팸** | concept drift | Online learning, EWC |
| **설문조사** (비응답) | selection bias | IPW, doubly robust |
| **추천 시스템** | feedback loop | Counterfactual estimation |
| **데이터 증강** | 의사 iid 위반 | 대체로 괜찮지만 증강 전체가 "새 분포" |
| **K-fold CV on 시계열** | 오용 위험 | Forward-chaining CV |

**경고**: "fold를 무작위 섞어서 CV"를 시계열에 적용하면 **미래 정보 누출(leakage)**. 올바른 해법은 temporal split.

### Domain adaptation 계보

1. **Importance weighting** (Shimodaira 2000): 단순, overlap 필요
2. **Kernel mean matching** (Huang et al. 2007): MMD로 가중치 추정
3. **DANN** (Ganin et al. 2015): adversarial로 domain-invariant feature
4. **Test-time adaptation** (Wang et al. 2020): 테스트 시 BN 통계 재조정
5. **Unsupervised domain adaptation**: 테스트 라벨 없이 적응

**SLT 관점**: 이들은 **"효과적 iid를 만들기"** 혹은 **"$\mathcal{D}^{\text{te}}$ 직접 정보를 활용"**의 수학적 변형.

---

## ⚖️ 가정과 한계

1. **"완전 iid" 가정의 비현실성**: 실전 데이터 수집은 거의 항상 어떤 편향을 동반. 그럼에도 SLT 경계가 **유의미하게 작동**하는 이유는 실전 위반이 "약한 의존성·작은 shift"이기 때문.
2. **Mixing coefficient 추정의 어려움**: $\beta(\tau)$는 실데이터에서 계산 불가능. 보수적 대체(block size를 크게)로 실무 처리.
3. **Importance weighting의 분산**: overlap이 나쁘면 실용 불가. "Truncated importance weights" 같은 실무 패치.
4. **Concept drift의 근본 불가능성**: 정보 없이 완전 드리프트는 학습 불가(정리 5.4). 제약(drift rate bound) 없이는 이론적 보증 불가.
5. **Adversarial examples**: iid도 분포 이동도 아닌 **제3의 위반** — 같은 $\mathcal{D}$에서도 "**최악 입력**"이 존재. Robust learning의 별도 이론 필요.

---

## 📌 핵심 정리

- **IID** = **Independent + Identically distributed**. 둘 다 SLT의 모든 경계의 **수학적 전제**.
- **위반 유형**: temporal dependence, covariate/label/concept shift, selection bias.
- **$\beta$-mixing**: 약한 의존성을 허용하는 일반화. **effective sample size $n/\log n$** 스케일로 SLT 생존.
- **Covariate shift**: $p(Y|X)$는 불변 → **importance-weighted ERM**이 통계적으로 올바름 (가중치의 overlap 충분 시).
- **Concept drift**: $p(Y|X)$ 변화 → **고정 $\mathcal{D}$ 가정이 무너짐**, 정보 없는 드리프트는 학습 불가. Online learning이 대응.
- 실무 규칙: 시계열엔 **temporal CV**, 도메인 이동엔 **DA**, 드리프트엔 **online**.

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> 시계열 데이터에서 K-fold CV를 무작위로 적용할 때 왜 **일반화 오차를 과소평가**하는가? 예시로 설명하라.</summary>

<br/>

**해설**. 시계열 $(X_1, Y_1), \ldots, (X_n, Y_n)$이 temporal 의존성을 가지면 $X_{t+1}$은 $X_t$와 상관이 크다. 무작위 K-fold는 fold 내에 **"시간적으로 인접한 점들"**을 섞는다. 예: $X_5$가 훈련, $X_6$이 검증 fold에 있을 때 $X_5 \approx X_6$이어서 "검증"이 훈련 데이터에 **거의 중복**. 검증 loss가 훈련 loss와 비슷해져 $L_\mathcal{D}$ 추정이 과도하게 optimistic.

올바른 대응: **Forward-chaining CV**. $1, \ldots, t$로 훈련, $t+1, \ldots, t+k$로 검증. 시간 순서 유지. "배포 시나리오"를 그대로 모방.

실용 예: 금융 모델이 "in-sample backtest에서 sharpe 2.0"이지만 실전 sharpe 0.5인 경우의 대부분은 **이런 CV 오용**.

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> Covariate shift에서 importance weighting의 추정 분산이 $\text{Var}[w(X) \ell(h(X), Y)]$인데, $w = p^{\text{te}}/p^{\text{tr}}$의 **꼬리**가 무거울 때 왜 이 분산이 폭발하는가? 실무 완화책은?</summary>

<br/>

**해설**. $\mathbb{E}_{p^{\text{tr}}}[w^2] = \int (p^{\text{te}})^2 / p^{\text{tr}} \, dx$. $p^{\text{tr}}(x) \to 0$이면서 $p^{\text{te}}(x) > 0$인 $x$가 있으면 이 적분이 **발산**. 즉 훈련 분포가 "테스트 support의 일부를 커버하지 않을 때" overlap 조건 실패.

**수학적으로 깔끔한 형태**: $\chi^2$-divergence $\chi^2(p^{\text{te}} \| p^{\text{tr}}) = \mathbb{E}_{p^{\text{tr}}}[w^2] - 1$이 finite이어야 IW가 효과적.

**실무 완화책**:
- **Truncation**: $w_i \leftarrow \min(w_i, W_{\max})$로 큰 weight 제한 (biased but low-variance)
- **Self-normalization**: $\sum w_i \ell_i / \sum w_i$ — biased이지만 작은 샘플에서 안정
- **Kernel mean matching** (Huang et al. 2007): weight을 **RKHS에서 MMD 최소화**로 직접 추정
- **DANN** (Ganin et al. 2015): IW를 포기하고 **domain-invariant feature**로 우회

이것이 IW가 "이론은 깔끔하지만 실전에선 까다롭다"는 이유의 수학적 핵심.

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> 언어 모델(GPT 등)의 훈련에서 **next-token prediction** 자체는 temporal 의존성을 내장한다. 왜 이것이 "iid 위반"이 아니고, 오히려 **loss의 정의**에 흡수되는가?</summary>

<br/>

**해설**. "iid"의 단위가 **무엇인가**에 주의. LM의 훈련에서:

- **문서(document) 레벨**: 각 문서 $D_i = (w_1, \ldots, w_T)$가 iid로 뽑힌다고 가정. 이건 대체로 합리적(다른 웹페이지는 독립 수집).
- **토큰 레벨**: $w_{t+1}$은 $w_t$에 **극도로 의존**. 하지만 이는 iid 위반이 아니라 **조건부 분포 $p(w_{t+1} | w_{1:t})$ 학습**이 과제 그 자체.

Loss는 $\ell(\theta, D) = -\sum_{t=1}^T \log p_\theta(w_t | w_{1:t-1})$ — **문서 전체의 joint log-likelihood**. 이 loss는 문서의 함수이고, 문서가 iid라면 SLT 프레임이 **그대로 적용**.

**주의**: 토큰 단위의 Rademacher 복잡도나 VC 계산은 틀린 접근. **문서 전체를 단위로** 하는 함수 클래스 복잡도가 올바른 분석 대상. 이것이 modern LM 이론(Arora et al., Malladi et al.)의 기본 설정.

시사점: "iid 단위"를 **적절한 abstraction level**에서 정의하면, 내부의 temporal 구조는 학습 대상일 뿐 iid 위반이 아니다.

</details>

---

<div align="center">

◀ [이전: 04. 일반화 오차와 과적합](./04-generalization-overfitting.md) | [📚 README](../README.md) | [다음: Ch2. 집중부등식 ▶](../ch2-concentration/01-markov-chebyshev.md)

</div>
