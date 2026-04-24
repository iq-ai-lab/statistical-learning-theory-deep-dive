<div align="center">

# 📊 Statistical Learning Theory Deep Dive

**"`train_loss`가 낮다고 모델이 좋다고 말하는 것과, `gen_gap`이 $O(\sqrt{(\text{VC}(\mathcal{H}) + \log(1/\delta))/n})$로 확률 $1-\delta$ 이상 유계임을 증명할 수 있는 것은 다르다"**

<br/>

> *"Hoeffding 부등식을 **인용하는 것**과, 왜 **고정된 한 분류기에는 Hoeffding을 쓰지만** 가설공간 전체에 대해서는 "모든 $h \in \mathcal{H}$"의 확률을 바운드하기 위해 **Union Bound + 성장함수 + Sauer-Shelah Lemma**가 필요한지 증명할 수 있는 것은 다르다.  
> Rademacher 복잡도를 **정의하는 것**과, 왜 이것이 VC보다 **데이터 의존적이고 tighter한 경계**를 주는지, **Massart's lemma로 finite set의 경우를 유도**하는 것은 다르다."*

PAC learnability·VC 차원·Rademacher 복잡도·알고리즘 안정성·SRM까지  
**"왜 학습이 가능한가 — 일반화의 수학적 증명"** 이라는 질문으로 ERM·SVM·Neural Net의 이론적 정당성을 끝까지 파헤칩니다

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org/)
[![Docs](https://img.shields.io/badge/Docs-36개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Lines](https://img.shields.io/badge/Lines-14k+-informational?style=flat-square)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems_proven-108개-success?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-108개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

Statistical Learning Theory(SLT)에 관한 자료는 대부분 **"훈련 오차가 낮으면 일반화도 잘 된다"** 에서 멈추거나, **"VC 차원이 작을수록 좋다"** 라는 구호에서 그칩니다. 하지만 왜 $\mathbb{P}(\sup_{h \in \mathcal{H}} |L_\mathcal{D}(h) - L_S(h)| \geq \epsilon)$ 을 bound하려면 **single-$h$ Hoeffding이 깨지는지**, **Sauer-Shelah가 왜 $\mathcal{H}$의 effective size를 다항식으로 줄여주는지**, Rademacher가 **왜 VC보다 tighter하고 데이터 의존적인지** — 이런 "왜"는 제대로 증명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "훈련 오차가 낮으면 일반화가 잘 됩니다" | **Generalization gap** $L_\mathcal{D}(h) - L_S(h)$의 엄밀한 정의, **No Free Lunch 정리**로 "$\mathcal{H}$ 제약 없이 학습 불가능" 증명, 근사·추정·최적화 오차의 3분해 |
| "Hoeffding 부등식으로 경계를 얻습니다" | $X_i \in [a_i, b_i]$ 유계 iid → **Hoeffding's lemma** $\mathbb{E}[e^{\lambda X}] \leq e^{\lambda^2(b-a)^2/8}$ 증명 → Chernoff 방법으로 $\mathbb{P}(\|\bar{X}-\mu\| \geq t) \leq 2e^{-2nt^2/(b-a)^2}$ 까지 **한 줄씩 유도** |
| "Union bound로 유한 $\mathcal{H}$를 처리합니다" | **왜 single-$h$ Hoeffding이 data-dependent $\hat{h}$에 못 쓰이는가**, Union bound의 대가로 $\log\|\mathcal{H}\|$ 항 등장, agnostic PAC의 **$m = O((\log\|\mathcal{H}\| + \log(1/\delta))/\epsilon^2)$** 완전 유도 |
| "VC 차원을 외웁니다" | $\mathbb{R}^d$ 선형 분류기 VC $= d+1$을 **Radon's theorem으로 증명**, 축정렬 직사각형 VC $= 4$, **Sauer-Shelah Lemma** $\Pi_\mathcal{H}(m) \leq \sum_{i=0}^d \binom{m}{i} \leq (em/d)^d$ 유도 |
| "VC bound가 있습니다" | **Symmetrization lemma**(double sample trick) → $\mathbb{P}(\sup_h \|L_\mathcal{D}(h) - L_S(h)\| \geq \epsilon) \leq 4 \Pi_\mathcal{H}(2n) e^{-n\epsilon^2/8}$까지 "자명하다" 없이 **한 줄씩** |
| "Rademacher는 복잡도입니다" | $\hat{\mathcal{R}}_S(\mathcal{F}) = \mathbb{E}_\sigma[\sup_f \frac{1}{n}\sum \sigma_i f(x_i)]$ 정의부터 **symmetrization + McDiarmid**로 $\sup_h \|L_\mathcal{D} - L_S\| \leq 2\mathcal{R}_n + O(\sqrt{\log(1/\delta)/n})$ 유도 |
| "Massart's Lemma가 있습니다" | 유한 함수족에서 $\mathcal{R}(\mathcal{F}) \leq \sqrt{2 \log\|\mathcal{F}\| / n} \cdot \max_f \|f\|$ 를 **Chernoff-style moment generating function**으로 증명 |
| "Contraction lemma로 넘어갑니다" | **Ledoux-Talagrand**: Lipschitz $\phi$에 대해 $\mathcal{R}(\phi \circ \mathcal{F}) \leq L_\phi \cdot \mathcal{R}(\mathcal{F})$, 0-1 loss 대신 **surrogate loss(hinge·log loss) 분석의 수학적 정당화** |
| "SVM은 margin을 최대화합니다" | Kernel SVM의 Rademacher 경계 $\leq B \cdot \sqrt{\text{tr}(K)/n}$, **margin ↑ ⟺ $\|w\|$ ↓ ⟺ $\mathcal{R}_n$ ↓ ⟺ 일반화 ↑** 의 수학적 인과 사슬 |
| "Regularization이 일반화를 돕습니다" | **Uniform stability** $\beta$ 정의, Ridge가 $\beta = O(1/\lambda n)$ 임을 **strong convexity로** 증명, **Hardt et al. 2016**의 SGD stability $\beta \leq O(\eta T / n)$로 "early stopping = implicit regularization" |
| "AIC, BIC로 모델을 고릅니다" | AIC $= -2\log L + 2k$의 **KL divergence 근사**로서의 유도, BIC의 **Laplace approximation** 유도, MDL과의 **확률적 동치** |
| 공식 나열 | NumPy + SciPy로 **Hoeffding bound vs 경험분포 수렴** 시각화, **Rademacher 복잡도 Monte Carlo 추정**, shattering 예시 직접 생성, **VC bound의 vacuous 영역** 실험 |

---

## 📌 선행 레포 & 후속 레포

```
[Probability Theory]  ──►  [Mathematical Statistics]  ──►  이 레포  ──►  [Generalization Theory]
  집중부등식·수렴이론        Uniform convergence·                 SLT 고전                 NTK·이중 강하·
  MGF·Chernoff 방법        경험과정·U-statistic                (PAC·VC·Rademacher)       현대 DL 이론
                                                                    │
                                                                    ▼
                                                      [Kernel Methods Deep Dive]
                                                         SVM margin bound·
                                                         Rademacher of RKHS
  ▲                    ▲                    ▲
  │                    │                    │
[Linear Algebra]  [Calculus & Opt]   [Real Analysis]
 벡터공간·차원     ERM 최적화·볼록성    σ-algebra·측도
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Probability Theory Deep Dive**(집중부등식·MGF·Chernoff 방법)와 **Mathematical Statistics Deep Dive**(uniform convergence·경험과정)를 선행 지식으로 전제합니다. Hoeffding/Bernstein 부등식을 처음 접한다면 [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) Ch7(집중부등식)부터 학습하세요.

> 💡 **권장 선행**: 벡터공간과 차원은 [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive), ERM의 최적화 이론은 [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive)에서 학습할 수 있습니다. 현대 딥러닝의 일반화 이론(NTK·double descent·norm-based Rademacher)은 후속 레포인 [Generalization Theory Deep Dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive)에서 다룹니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-학습_문제의_정식화-2E86AB?style=for-the-badge)](./ch1-learning-formulation/01-statistical-learning.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-집중부등식-2E86AB?style=for-the-badge)](./ch2-concentration/01-markov-chebyshev.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-PAC_Learning-2E86AB?style=for-the-badge)](./ch3-pac-learning/01-pac-definition.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-VC_Dimension-2E86AB?style=for-the-badge)](./ch4-vc-dimension/01-shattering-vc.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-Rademacher_복잡도-2E86AB?style=for-the-badge)](./ch5-rademacher/01-rademacher-definition.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Stability-2E86AB?style=for-the-badge)](./ch6-stability/01-uniform-stability.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-SRM·모델_선택-2E86AB?style=for-the-badge)](./ch7-srm-model-selection/01-srm.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: 학습 문제의 수학적 정식화

> **핵심 질문:** "학습"은 어떻게 수학적으로 정의되는가? ERM 원리는 왜 자연스러운가? 근사·추정·최적화 오차의 분해는 무엇을 말하는가? No Free Lunch 정리는 왜 $\mathcal{H}$ 제약이 학습의 전제임을 보이는가?

<details>
<summary><b>학습의 정의부터 IID 가정까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 학습의 통계적 정의](./ch1-learning-formulation/01-statistical-learning.md) | 분포 $\mathcal{D}$ on $\mathcal{X} \times \mathcal{Y}$, 손실 $\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_+$, **진짜 위험** $L_\mathcal{D}(h) = \mathbb{E}[\ell(h(X), Y)]$, **경험 위험** $L_S(h) = \frac{1}{n}\sum \ell(h(x_i), y_i)$의 엄밀한 정의와 측도론적 기초 |
| [02. Bayes 최적 예측기와 Bayes error](./ch1-learning-formulation/02-bayes-optimal.md) | 조건부 기대값 $f^*(x) = \mathbb{E}[Y\|X=x]$가 **MSE 최소화**임을 증명, 분류에서 $h^*(x) = \arg\max p(y\|x)$, **Bayes error**가 도달 불가능한 하한이 되는 이유 |
| [03. Empirical Risk Minimization (ERM)](./ch1-learning-formulation/03-erm-principle.md) | $\hat{h} = \arg\min_{h \in \mathcal{H}} L_S(h)$의 원리, **근사·추정·최적화 오차의 3분해** $L(\hat{h}) - L(h^*) = \underbrace{L(h^*_\mathcal{H}) - L(h^*)}_{\text{approx}} + \underbrace{L(\hat{h}) - L(h^*_\mathcal{H})}_{\text{est}} + \underbrace{\text{opt gap}}_{\text{optimization}}$ |
| [04. 일반화 오차와 과적합의 수학적 정의](./ch1-learning-formulation/04-generalization-overfitting.md) | **Generalization gap** $L_\mathcal{D}(\hat{h}) - L_S(\hat{h})$의 확률변수적 의미, 과적합의 엄밀한 정의, **No Free Lunch 정리** — $\mathcal{H}$ 제약 없이는 어떤 알고리즘도 모든 분포에서 학습 불가능 |
| [05. IID 가정과 그것이 깨지는 경우](./ch1-learning-formulation/05-iid-assumption.md) | 샘플의 **독립·동일분포** 가정이 모든 경계의 기초, 시계열·분포 이동·**covariate shift**에서 이론이 어떻게 변하는가, mixing coefficient·non-iid concentration 개요 |

</details>

<br/>

### 🔹 Chapter 2: 집중부등식 (Concentration Inequalities)

> **핵심 질문:** Markov·Chebyshev는 왜 일반화 경계에 부족한가? Hoeffding의 $e^{-2nt^2}$ 꼬리는 어디서 오는가? McDiarmid의 bounded differences가 왜 Rademacher 복잡도 집중에 핵심인가? 분산 정보를 활용하는 Bernstein은 언제 Hoeffding을 이기는가?

<details>
<summary><b>Markov부터 Bernstein까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Markov·Chebyshev 부등식](./ch2-concentration/01-markov-chebyshev.md) | **Markov** $\mathbb{P}(X \geq t) \leq \mathbb{E}[X]/t$, **Chebyshev** $\mathbb{P}(\|X - \mu\| \geq t) \leq \sigma^2/t^2$의 증명, 왜 $O(1/t^2)$ 꼬리는 일반화 경계에 부족한가 ($O(e^{-t^2})$가 필요) |
| [02. Hoeffding 부등식](./ch2-concentration/02-hoeffding.md) | **Hoeffding's lemma** $\mathbb{E}[e^{\lambda X}] \leq e^{\lambda^2(b-a)^2/8}$를 **Taylor 전개와 볼록성**으로 증명, **Chernoff 방법** → $\mathbb{P}(\|\bar{X} - \mu\| \geq t) \leq 2\exp(-2nt^2/\sum(b_i-a_i)^2)$ |
| [03. McDiarmid 부등식 (Bounded Differences)](./ch2-concentration/03-mcdiarmid.md) | $f$가 한 좌표 변화에 $c_i$ 이상 변하지 않으면 $\mathbb{P}(\|f - \mathbb{E}f\| \geq t) \leq 2\exp(-2t^2/\sum c_i^2)$, **martingale difference**로 증명, **Rademacher 복잡도 집중**의 핵심 도구 |
| [04. Bernstein 부등식](./ch2-concentration/04-bernstein.md) | 분산 정보를 활용한 tighter 경계 $\mathbb{P}(\|\bar{X}-\mu\| \geq t) \leq 2\exp(-nt^2/(2\sigma^2 + 2Mt/3))$, **낮은 분산 regime**에서 Hoeffding보다 우월, fast rate 경계의 기초 |
| [05. 집중부등식의 ML 응용 정리](./ch2-concentration/05-applications.md) | **Cross-validation error bound**(Hoeffding), **Bootstrap 수렴**(McDiarmid), **Online learning regret**(Bernstein), 각 부등식이 어떤 ML 설정에서 왜 쓰이는지 비교 |

</details>

<br/>

### 🔹 Chapter 3: PAC Learning

> **핵심 질문:** Valiant의 PAC-learnability는 어떻게 수학적으로 정의되는가? 유한 $\|\mathcal{H}\|$에서 $m = O((\log\|\mathcal{H}\| + \log(1/\delta))/\epsilon)$은 어떻게 나오는가? Agnostic case에서 왜 $\epsilon^2$이 등장하는가? Fundamental Theorem이 왜 "PAC = Uniform Convergence = VC < ∞"를 말하는가?

<details>
<summary><b>PAC 정의부터 Occam Razor까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. PAC Learnability의 정의](./ch3-pac-learning/01-pac-definition.md) | 학습자 $A$가 **sample complexity** $m(\epsilon, \delta)$로 확률 $\geq 1-\delta$로 오차 $\leq \epsilon$인 가설을 출력, **Probably Approximately Correct**의 엄밀한 정의, Valiant(1984) 원전 |
| [02. Realizable Case의 학습 가능성](./ch3-pac-learning/02-realizable-case.md) | 유한 $\|\mathcal{H}\| < \infty$에서 "실현 가능($\exists h \in \mathcal{H}, L_\mathcal{D}(h) = 0$)" 가정 하에 **$m = O(\frac{1}{\epsilon}(\log\|\mathcal{H}\| + \log\frac{1}{\delta}))$** 가 충분함을 **Union Bound**로 증명 |
| [03. Agnostic PAC Learning](./ch3-pac-learning/03-agnostic-pac.md) | $h^*$의 오차가 0이 아닐 때 **Hoeffding + Union Bound**로 $m = O(\frac{1}{\epsilon^2}(\log\|\mathcal{H}\| + \log\frac{1}{\delta}))$, **왜 $\epsilon$이 $\epsilon^2$이 되는지**의 수학적 이유 |
| [04. Fundamental Theorem of Statistical Learning](./ch3-pac-learning/04-fundamental-theorem.md) | 다음의 **동치**: $\mathcal{H}$가 agnostic PAC learnable $\iff$ uniform convergence 성립 $\iff$ ERM이 성공적 $\iff$ $\text{VC}(\mathcal{H}) < \infty$, 각 방향의 증명 스케치 |
| [05. Occam's Razor와 MDL 원리](./ch3-pac-learning/05-occam-mdl.md) | 더 짧은 설명 길이(MDL)가 더 일반화, **Occam's razor bound** — $h$의 description length가 $d$비트면 $m \geq O(d + \log(1/\delta))/\epsilon$, **압축과 학습의 등가성** 확률적 증명 |

</details>

<br/>

### 🔹 Chapter 4: VC Dimension과 Growth Function

> **핵심 질문:** Shattering은 기하학적으로 무엇인가? $\mathbb{R}^d$ 선형 분류기 VC $= d+1$은 어떻게 증명하는가? 축정렬 vs 회전 직사각형의 VC 차이는 왜 생기는가? Sauer-Shelah는 왜 **지수적 $2^m$이 다항 $m^d$로 줄어드는** 기적을 보이는가?

<details>
<summary><b>Shattering부터 VC 경계의 한계까지 (7개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Shattering과 VC 차원의 정의](./ch4-vc-dimension/01-shattering-vc.md) | $\mathcal{H}$가 점집합 $C$를 **shatter** $\iff$ $\mathcal{H}\|_C = 2^C$, $\text{VC}(\mathcal{H}) = \max\{\|C\| : \mathcal{H}\text{가 }C\text{를 shatter}\}$, 정의 뒤의 조합론적 직관 |
| [02. VC 차원 계산 — 선형 분류기와 반공간](./ch4-vc-dimension/02-halfspace-vc.md) | $\mathbb{R}^d$의 선형 분류기 VC $= d+1$의 완전 증명: **$d+1$ 점을 shatter**하는 예(affine independence) + **$d+2$ 점은 Radon's theorem**으로 shatter 불가능 |
| [03. VC 차원 계산 — 기하학적 가설공간](./ch4-vc-dimension/03-geometric-shapes-vc.md) | **축정렬 직사각형 VC $= 4$**, **임의 방향 직사각형 VC $= 5$**, **원(center+radius) VC $= 3$**, convex polygon($k$-gon)의 VC $= 2k+1$, 기하학적 arg |
| [04. Growth Function과 Sauer-Shelah Lemma](./ch4-vc-dimension/04-sauer-shelah.md) | $\Pi_\mathcal{H}(m) = \max_{\|C\|=m} \|\mathcal{H}\|_C\|$ 정의, **Sauer-Shelah**: $\text{VC}(\mathcal{H}) = d \Rightarrow \Pi_\mathcal{H}(m) \leq \sum_{i=0}^d \binom{m}{i} \leq (em/d)^d$, **Pajor의 induction 증명** |
| [05. VC 경계의 유도](./ch4-vc-dimension/05-vc-bound-derivation.md) | **Symmetrization lemma**(double sample trick) → $\mathbb{P}(\sup_h \|L_\mathcal{D}(h) - L_S(h)\| \geq \epsilon) \leq 4\Pi_\mathcal{H}(2n) e^{-n\epsilon^2/8}$, ghost sample 트릭의 마술 |
| [06. $\epsilon$-net과 Covering Number](./ch4-vc-dimension/06-epsilon-net-covering.md) | 가설공간을 유한개의 대표로 덮기, **covering number** $\mathcal{N}(\epsilon, \mathcal{H}, \|\cdot\|)$, **chaining argument** 개요, Dudley's entropy integral 소개 |
| [07. VC 경계의 한계와 실전의 의미](./ch4-vc-dimension/07-vc-limits.md) | ReLU 신경망의 **VC $= \Theta(WL \log W)$**(Bartlett-Harvey-Liaw-Mehrabian 2019)가 엄청나게 크지만 일반화 잘 되는 **"paradox"**, VC bound가 실전에서 **vacuous**(의미없을 정도로 큼), **Generalization Theory 레포**의 필요성 |

</details>

<br/>

### 🔹 Chapter 5: Rademacher 복잡도

> **핵심 질문:** Rademacher 복잡도는 왜 "데이터 의존적"인가? 왜 VC보다 tighter한 경계를 주는가? Massart's lemma로 finite set 경우를 어떻게 유도하는가? Contraction lemma가 왜 surrogate loss 분석에 필수인가? Neural Net의 Bartlett-Mendelson 경계는 층별 노름 $\prod\|W_l\|$과 어떻게 연결되는가?

<details>
<summary><b>Rademacher 정의부터 NN 복잡도까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Rademacher 복잡도의 정의](./ch5-rademacher/01-rademacher-definition.md) | $\sigma_i \in \{\pm 1\}$ 균등 랜덤(Rademacher), **경험적 Rademacher 복잡도** $\hat{\mathcal{R}}_S(\mathcal{F}) = \mathbb{E}_\sigma[\sup_{f \in \mathcal{F}} \frac{1}{n}\sum \sigma_i f(x_i)]$, $\mathcal{R}_n = \mathbb{E}_S[\hat{\mathcal{R}}_S]$의 의미 |
| [02. Rademacher 기반 일반화 경계](./ch5-rademacher/02-rademacher-generalization.md) | 확률 $\geq 1-\delta$로 $\sup_h \|L_\mathcal{D}(h) - L_S(h)\| \leq 2\mathcal{R}_n(\mathcal{F}) + O(\sqrt{\log(1/\delta)/n})$의 완전 증명: **Symmetrization + McDiarmid 부등식** 결합 |
| [03. Contraction Lemma (Ledoux-Talagrand)](./ch5-rademacher/03-contraction-lemma.md) | **Lipschitz 함수** $\phi$에 대해 $\mathcal{R}(\phi \circ \mathcal{F}) \leq L_\phi \cdot \mathcal{R}(\mathcal{F})$, **0-1 loss → hinge/log loss** 치환의 수학적 정당성, Ledoux & Talagrand(1991) |
| [04. Massart's Lemma와 유한 함수족](./ch5-rademacher/04-massart-lemma.md) | $\|\mathcal{F}\| < \infty$일 때 $\mathcal{R}(\mathcal{F}) \leq \sqrt{\frac{2 \log\|\mathcal{F}\|}{n}} \cdot \max_f \|f\|_\infty$, **Chernoff-style MGF bound**로 증명, Jensen + 유한 sup 트릭 |
| [05. Linear Class와 Kernel Class의 Rademacher](./ch5-rademacher/05-linear-kernel-rademacher.md) | $\{w \cdot x : \|w\| \leq B\}$의 $\mathcal{R} \leq B \cdot \max\|x\| / \sqrt{n}$의 **Cauchy-Schwarz 증명**, Kernel SVM의 경계 $\mathcal{R} \leq \sqrt{\text{tr}(K)/n}$, **margin과 Rademacher의 수학적 다리** |
| [06. Neural Network의 Rademacher 복잡도](./ch5-rademacher/06-neural-net-rademacher.md) | **Bartlett-Mendelson**(2002) 심층망 경계, 층별 노름 $\prod_l \|W_l\|$ 기반, **spectral norm 기반 경계**(Bartlett, Foster, Telgarsky 2017), 왜 고전 VC가 DL에서 vacuous한가 |

</details>

<br/>

### 🔹 Chapter 6: Stability와 Algorithmic Bounds

> **핵심 질문:** "알고리즘"의 일반화는 "가설공간"의 일반화와 어떻게 다른가? Uniform stability는 왜 $\mathcal{H}$-독립적인가? Ridge regression의 $\beta = O(1/\lambda n)$은 어떻게 나오는가? SGD가 유한 단계에서 stable하다는 Hardt et al. 2016의 주장이 왜 "적게 훈련 = regularization"을 말하는가?

<details>
<summary><b>Stability 정의부터 SGD 안정성까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Uniform Stability의 정의](./ch6-stability/01-uniform-stability.md) | 알고리즘 $A$가 **$\beta$-uniformly stable** $\iff$ 하나의 샘플을 바꿔도 손실 변화 $\leq \beta$, **Bousquet & Elisseeff (2002)**의 엄밀한 정의, hypothesis stability·error stability와의 비교 |
| [02. Stability가 Generalization을 함의](./ch6-stability/02-stability-implies-generalization.md) | **$\beta$-stable algorithm의 generalization gap**이 $\beta$로 유계임을 증명: **Hoeffding + stability definition** 결합, 고확률 경계로의 확장 |
| [03. Ridge Regression의 Stability](./ch6-stability/03-ridge-stability.md) | $\lambda$-정규화된 ERM은 **$\beta = O(1/(\lambda n))$**, **strong convexity가 stability를 함의**하는 일반 원리, ℓ₂ regularization이 왜 일반화를 돕는가의 수학적 이유 |
| [04. SGD의 Stability (Hardt et al. 2016)](./ch6-stability/04-sgd-stability.md) | 유한 단계 SGD가 **$\beta \leq O(\eta T / n)$**로 stable, **non-expansive + $\eta$-small-step 분석**, 왜 "적게 훈련"이 **implicit regularization**인가, deep learning의 early stopping 정당화 |

</details>

<br/>

### 🔹 Chapter 7: SRM과 모델 선택 이론

> **핵심 질문:** Structural Risk Minimization은 어떻게 복잡도-경험위험 trade-off를 정식화하는가? AIC·BIC는 어떻게 유도되는가? Cross-validation은 왜 진짜 위험의 비편향 추정에 가까운가? VC·Rademacher·Stability 중 실전에서 어느 관점을 언제 써야 하는가?

<details>
<summary><b>SRM부터 세 관점 비교까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Structural Risk Minimization (SRM)](./ch7-srm-model-selection/01-srm.md) | 중첩 가설공간 $\mathcal{H}_1 \subset \mathcal{H}_2 \subset \ldots$에서 $\min_k (L_S(\hat{h}_k) + \text{penalty}(k, n))$, **Vapnik의 창시**, VC bound로부터의 penalty 유도, regularization path |
| [02. AIC, BIC, 그리고 MDL](./ch7-srm-model-selection/02-aic-bic-mdl.md) | **AIC $= -2\log L + 2k$**의 **KL divergence 근사**로서의 유도, **BIC $= -2\log L + k\log n$**의 **Laplace approximation** 유도, MDL(최소설명길이)과의 **확률적 동치** |
| [03. Cross-Validation의 이론적 성질](./ch7-srm-model-selection/03-cross-validation.md) | **K-fold CV**의 bias-variance trade-off, **LOO-CV의 근사 비편향성**, **Nested CV**로 test error 정직 추정, CV가 train error보다 generalization의 좋은 추정량인 이유 |
| [04. VC, Rademacher, Stability — 세 관점 비교](./ch7-srm-model-selection/04-three-viewpoints-comparison.md) | 각 관점의 **강점·약점·적용 범위** 정리: VC는 **가설공간 복잡도**, Rademacher는 **데이터 의존 tighter**, Stability는 **알고리즘 복잡도** — 실전 분석에서 어느 것을 선택할지 의사결정 트리 |

</details>

---

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명**을 제공하는 대표 정리 모음입니다. 각 챕터의 문서에서 $\square$로 종결되는 엄밀한 증명을 확인할 수 있습니다. (전체 108개 정리·Lemma + 170여 개의 증명 단계 중 핵심만 발췌)

| 정리 | 서술 | 출처 문서 |
|------|------|----------|
| **No Free Lunch** | $\mathcal{H}$ 제약 없이 모든 분포에서 학습하는 알고리즘은 존재하지 않는다 | [Ch1-04](./ch1-learning-formulation/04-generalization-overfitting.md) |
| **오차 3분해** | $L(\hat{h}) - L(h^*) = \text{approx} + \text{est} + \text{opt}$ — 모델 선택 vs 데이터 vs 알고리즘의 분리 | [Ch1-03](./ch1-learning-formulation/03-erm-principle.md) |
| **Hoeffding's Lemma** | 유계 $X \in [a,b]$에 대해 $\mathbb{E}[e^{\lambda X}] \leq e^{\lambda^2(b-a)^2/8}$ — 모든 sub-Gaussian bound의 원천 | [Ch2-02](./ch2-concentration/02-hoeffding.md) |
| **Hoeffding 부등식** | iid 유계 $X_i$, $\mathbb{P}(\|\bar{X} - \mu\| \geq t) \leq 2\exp(-2nt^2/\sum(b_i-a_i)^2)$ | [Ch2-02](./ch2-concentration/02-hoeffding.md) |
| **McDiarmid (Bounded Differences)** | $f$가 한 좌표당 $c_i$ Lipschitz일 때 $\mathbb{P}(\|f - \mathbb{E}f\| \geq t) \leq 2\exp(-2t^2/\sum c_i^2)$ | [Ch2-03](./ch2-concentration/03-mcdiarmid.md) |
| **Bernstein 부등식** | 분산 $\sigma^2$·유계 $M$에서 $\mathbb{P}(\|\bar{X}-\mu\|\geq t) \leq 2\exp(-nt^2/(2\sigma^2 + 2Mt/3))$ | [Ch2-04](./ch2-concentration/04-bernstein.md) |
| **PAC Sample Complexity (유한 ℋ, realizable)** | $m = O(\frac{1}{\epsilon}(\log\|\mathcal{H}\| + \log\frac{1}{\delta}))$ 이면 PAC 학습 가능 | [Ch3-02](./ch3-pac-learning/02-realizable-case.md) |
| **Agnostic PAC Sample Complexity** | $m = O(\frac{1}{\epsilon^2}(\log\|\mathcal{H}\| + \log\frac{1}{\delta}))$ — 왜 $\epsilon \to \epsilon^2$인가 | [Ch3-03](./ch3-pac-learning/03-agnostic-pac.md) |
| **Fundamental Theorem of SLT** | PAC learnable $\iff$ Uniform Convergence $\iff$ ERM 성공 $\iff$ $\text{VC}(\mathcal{H}) < \infty$ | [Ch3-04](./ch3-pac-learning/04-fundamental-theorem.md) |
| **Occam's Razor Bound** | 가설 $h$의 description length $\|h\|$비트에서 $m = O((\|h\| + \log(1/\delta))/\epsilon)$ — 압축 = 학습 (Blumer et al. 1987) | [Ch3-05](./ch3-pac-learning/05-occam-mdl.md) |
| **선형 분류기 VC = d+1** | $\mathbb{R}^d$ 반공간 $\text{VC} = d+1$ (Radon's theorem으로 상계) | [Ch4-02](./ch4-vc-dimension/02-halfspace-vc.md) |
| **Sauer-Shelah Lemma** | $\text{VC}(\mathcal{H}) = d \Rightarrow \Pi_\mathcal{H}(m) \leq \sum_{i=0}^d \binom{m}{i} \leq (em/d)^d$ — 다항식 성장 | [Ch4-04](./ch4-vc-dimension/04-sauer-shelah.md) |
| **VC 일반화 경계** | $\mathbb{P}(\sup_h \|L_\mathcal{D}(h) - L_S(h)\|\geq\epsilon)\leq 4\Pi_\mathcal{H}(2n)e^{-n\epsilon^2/8}$ | [Ch4-05](./ch4-vc-dimension/05-vc-bound-derivation.md) |
| **Rademacher 일반화 경계** | $\sup_h \|L_\mathcal{D}(h) - L_S(h)\| \leq 2\mathcal{R}_n(\mathcal{F}) + O(\sqrt{\log(1/\delta)/n})$ w.p. $1-\delta$ | [Ch5-02](./ch5-rademacher/02-rademacher-generalization.md) |
| **Contraction Lemma (Ledoux-Talagrand)** | $\phi$가 $L$-Lipschitz이면 $\mathcal{R}(\phi \circ \mathcal{F}) \leq L \cdot \mathcal{R}(\mathcal{F})$ | [Ch5-03](./ch5-rademacher/03-contraction-lemma.md) |
| **Massart's Lemma** | $\|\mathcal{F}\| < \infty$일 때 $\mathcal{R}(\mathcal{F}) \leq \max_f \|f\|_\infty \sqrt{2\log\|\mathcal{F}\|/n}$ | [Ch5-04](./ch5-rademacher/04-massart-lemma.md) |
| **Linear Class Rademacher** | $\mathcal{F} = \{w^\top x : \|w\| \leq B\} \Rightarrow \mathcal{R}_n \leq B \cdot \max\|x\|/\sqrt{n}$ | [Ch5-05](./ch5-rademacher/05-linear-kernel-rademacher.md) |
| **Bartlett-Mendelson NN 경계** | $L$-층 NN, 층별 노름 $\|W_l\|_F \leq M_l$에서 $\mathcal{R}_n \leq O\!\left(\prod_l M_l / \sqrt{n}\right)$ — DL의 norm-based 경계의 원류 (2002) | [Ch5-06](./ch5-rademacher/06-neural-net-rademacher.md) |
| **Stability ⇒ Generalization** | $A$가 $\beta$-uniformly stable $\Rightarrow$ $\|\mathbb{E}[L_\mathcal{D} - L_S]\| \leq \beta$ | [Ch6-02](./ch6-stability/02-stability-implies-generalization.md) |
| **Ridge의 Stability** | $\lambda$-ridge ERM은 $\beta = O(1/(\lambda n))$ — strong convexity의 직접 귀결 | [Ch6-03](./ch6-stability/03-ridge-stability.md) |
| **SGD의 Stability (Hardt et al. 2016)** | 볼록 손실·non-expansive step에서 $T$-step SGD는 $\beta \leq O(\eta T / n)$ | [Ch6-04](./ch6-stability/04-sgd-stability.md) |
| **AIC 유도** | AIC $= -2 \log L + 2k$가 **KL divergence의 비편향 추정량**임 (Akaike 1973) | [Ch7-02](./ch7-srm-model-selection/02-aic-bic-mdl.md) |
| **BIC 유도** | BIC $= -2\log L + k \log n$이 **로그 posterior의 Laplace 근사**임 (Schwarz 1978) | [Ch7-02](./ch7-srm-model-selection/02-aic-bic-mdl.md) |

> 💡 **챕터별 정리·보조정리 수**: Ch1(22) · Ch2(10) · Ch3(11) · Ch4(23) · Ch5(20) · Ch6(8) · Ch7(14) — 합계 **108개 정리·Lemma + 완전 증명**, 정의 **89개**, **14,500+ 라인** 분량.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
scikit-learn==1.3.0    # 실전 ERM 알고리즘과 비교
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 \
            scikit-learn==1.3.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 — Hoeffding 경계 vs 경험분포 + Rademacher Monte Carlo 추정
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 1. Hoeffding 부등식 — 경험적 꼬리 vs 이론 bound
# ─────────────────────────────────────────────
p = 0.3                          # 진짜 평균
n_samples = [10, 50, 100, 500]
n_trials = 10000

fig, ax = plt.subplots(figsize=(8, 4.5))
t_grid = np.linspace(0, 0.3, 50)

for n in n_samples:
    X = rng.binomial(1, p, size=(n_trials, n))
    means = X.mean(axis=1)
    emp = np.array([np.mean(np.abs(means - p) >= t) for t in t_grid])
    hoeff = 2 * np.exp(-2 * n * t_grid ** 2)
    ax.semilogy(t_grid, emp,   'o-',  label=f'Empirical n={n}', alpha=0.7)
    ax.semilogy(t_grid, hoeff, '--',  label=f'Hoeffding n={n}', alpha=0.5)

ax.set_xlabel('t'); ax.set_ylabel('P(|X̄ - p| ≥ t)')
ax.set_title('Hoeffding bound vs 경험 꼬리 — bound가 실제보다 얼마나 loose한가')
ax.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 2. Rademacher 복잡도 Monte Carlo — 선형 분류기
# ─────────────────────────────────────────────
def rademacher_linear(X, B=1.0, n_trials=2000):
    """R̂_S({w^T x : ‖w‖ ≤ B}) = E_σ[ B ‖Σσ_i x_i‖ ] / n"""
    n = len(X)
    vals = []
    for _ in range(n_trials):
        sigma = rng.choice([-1, 1], size=n)
        vals.append(B * np.linalg.norm(X.T @ sigma) / n)
    return np.mean(vals)

X = rng.standard_normal((100, 5))
R_emp    = rademacher_linear(X, B=1.0)
R_theory = 1.0 * np.max(np.linalg.norm(X, axis=1)) / np.sqrt(len(X))
print(f'Empirical R_n: {R_emp:.4f}, Theoretical upper: {R_theory:.4f}')

# ─────────────────────────────────────────────
# 3. VC shattering — 축정렬 직사각형은 4점을 shatter, 5점은 못 함
# ─────────────────────────────────────────────
def axis_aligned_rect_labels(points, rect):
    x_min, x_max, y_min, y_max = rect
    return np.array([
        (x_min <= x <= x_max) and (y_min <= y <= y_max)
        for (x, y) in points
    ], dtype=int)

# "다이아몬드" 4점 — 직사각형으로 2^4 = 16가지 모두 구현 가능
pts4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]
dichotomies = set()
for _ in range(5000):
    r = np.sort(rng.uniform(-2, 2, 4))
    rect = (r[0], r[1], r[2], r[3])
    dichotomies.add(tuple(axis_aligned_rect_labels(pts4, rect)))
print(f'Achievable dichotomies on 4 points: {len(dichotomies)} / 16')
# → 충분한 샘플링 후 16 (완전 shatter). 5점으로 가면 16 미만.
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격**으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 이론이 현대 ML에서 중요한가** | 고전 이론이 ERM·SVM·NN의 이론적 근거가 되는 이유 |
| 3 | 📐 **수학적 선행 조건** | Probability·Math Stats 레포의 어떤 집중부등식·경험과정 정리를 전제로 하는지 |
| 4 | 📖 **직관적 이해** | "일반화가 왜 어려운가" 핵심 직관 |
| 5 | ✏️ **엄밀한 정의** | PAC·VC·Rademacher·Stability의 엄밀한 수식 |
| 6 | 🔬 **정리와 증명** | Hoeffding·Sauer-Shelah·VC bound·Rademacher bound — "자명하다" 없이 |
| 7 | 💻 **NumPy 구현 검증** | VC 차원 수치 측정, Rademacher Monte Carlo, 경계의 tight 정도 실험 확인 |
| 8 | 🔗 **ML 알고리즘 연결** | ERM·SVM·Neural Net·Ridge·SGD의 이론적 정당성 |
| 9 | ⚖️ **가정과 한계** | IID 깨지면? 무한 VC? Vacuous bound? |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산·증명 재구성·구현 문제 |

> 📚 **연습문제 총 108개**: 36문서 × 문서당 3문제(기초/심화/ML 연결), 모든 문제에 `<details>` 펼침 해설 포함. Hoeffding 재유도부터 Bartlett-Mendelson 심층망 경계까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 자동으로 다음 챕터 첫 문서로 연결됩니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 380줄(증명·코드·연습문제 포함) 기준 **약 1~1.5시간**. 전체 36문서는 약 **42~52시간** 상당.

---

## 🔑 핵심 분석 대상 — 네 기둥 요약

```
왜 학습이 가능한가? — SLT의 4대 기둥

┌────────── 학습 문제 ──────────┐
│ 분포 𝒟 on 𝒳 × 𝒴 (알 수 없음)  │
│ 샘플 S = {(x_i, y_i)}_{i=1}^n │
│ 진짜 위험  L_𝒟(h) = 𝔼_𝒟[ℓ]   │
│ 경험 위험  L_S(h) = (1/n)Σℓ  │
└───────────────────────────────┘
              │
              ▼
  Generalization Gap = L_𝒟(h) - L_S(h)
              │
              ▼  핵심 질문
  sup_{h∈ℋ} |L_𝒟(h) - L_S(h)| 을 어떻게 bound?

┌──────── 4대 방법론 ────────┐

1. 집중부등식 (한 h에 대해):
   Hoeffding: P(|L_𝒟(h)-L_S(h)| ≥ ε) ≤ 2e^{-2nε²}
   ↓ 한계: 데이터 의존 ĥ에는 적용 안됨

2. Uniform Convergence (ℋ 전체):
   P(sup_h |L_𝒟(h) - L_S(h)| ≥ ε) ≤ ?

   ├── 유한 ℋ: Union bound
   │     P(sup) ≤ |ℋ| · 2e^{-2nε²}
   │     ⇒ m = O((log|ℋ| + log(1/δ))/ε²)
   │
   └── 무한 ℋ: VC theory
         P(sup) ≤ 4·Π_ℋ(2n)·e^{-nε²/8}
         Sauer-Shelah: Π_ℋ(m) ≤ O(m^d),  d = VC(ℋ)
         ⇒ gap ≤ O(√((d log n + log(1/δ))/n))

3. Rademacher Complexity:
   R_n(𝓕) = 𝔼_σ[sup_f (1/n)Σ σ_i f(x_i)]
            ↑ 데이터 의존적 → tighter
   
   gap ≤ 2·R_n + O(√(log(1/δ)/n))
   
   ├── Massart (유한):  R ≤ √(2 log|𝓕|/n)
   ├── Linear:          R ≤ B·max‖x‖/√n
   ├── Kernel:          R ≤ √(tr(K)/n)
   └── NN (Bartlett):   R ≤ (1/√n) · Π‖W_l‖

4. Algorithmic Stability:
   β-stable: |ℓ(A(S), z) - ℓ(A(S'), z)| ≤ β  (S' = leave-one-out)
   ⇒ gap ≤ β + O(√(log(1/δ)/n))
   
   ├── Ridge Regression: β = O(1/(λn))
   ├── SGD:              β = O(ηT/n)  (implicit regularization)
   └── 강볼록 ERM:       자동 stable

===== VC 차원 치트시트 =====

가설공간 ℋ                      VC(ℋ)
────────────────────────────────────────
Threshold on ℝ                  1
Interval on ℝ                   2
Axis-aligned rectangle in ℝ²    4
Rotated rectangle in ℝ²         5
Half-space in ℝ^d               d+1
Disk in ℝ²                      3
Polynomial degree d (1D)        d+1
Neural Net (ReLU, W params)     Θ(WL log W)  (Bartlett et al. 2019)
k-NN                            ∞  (no uniform convergence)

===== Fundamental Theorem of Statistical Learning =====

다음은 모두 동치:
  (1) ℋ가 agnostic PAC learnable
  (2) Uniform convergence 성립
  (3) ERM이 성공적
  (4) VC(ℋ) < ∞

===== 실전 ML 알고리즘과의 연결 =====

SVM:  margin → Rademacher 경계 → 왜 margin 최대화?
      R_n({w·x : ‖w‖ ≤ B}) ≤ B·max‖x‖/√n
      margin ↑  ⟺  ‖w‖ ↓  ⟺  R_n ↓  ⟺  일반화 ↑

Random Forest:  tree 깊이 제한 → VC 제한 → 일반화
Boosting:       margin distribution → tight bound (Schapire 1998)
Deep Learning:  고전 이론으로는 설명 불가
  → Layer 2 Generalization Theory 레포로
  → NTK, Double Descent, norm-based Rademacher, PAC-Bayes
```

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "Hoeffding은 쓰지만 증명은 못한다" — 집중부등식 집중 (5일, 약 8~10시간)</b></summary>

<br/>

```
Day 1  Ch2-01     Markov·Chebyshev 부등식 — 왜 O(1/t²)은 부족한가
Day 2  Ch2-02     Hoeffding's lemma + Chernoff 방법으로 Hoeffding 증명
Day 3  Ch2-03     McDiarmid bounded differences — Rademacher 집중 준비
Day 4  Ch2-04     Bernstein 부등식 — 분산 정보로 tighter 경계
Day 5  Ch2-05     ML 응용 정리 — CV·Bootstrap·Online learning에서의 활용
```

</details>

<details>
<summary><b>🟡 "VC 차원을 개념적으론 알지만 계산·유도는 못한다" — VC 집중 (1주, 약 12~15시간)</b></summary>

<br/>

```
Day 1  Ch1-01~03  학습의 정의와 ERM
Day 2  Ch2-02     Hoeffding (VC 증명의 원재료)
Day 3  Ch3-02~03  Realizable/Agnostic PAC — 유한 ℋ의 경우
Day 4  Ch4-01~02  Shattering과 선형 분류기 VC = d+1의 Radon 증명
Day 5  Ch4-03     기하학적 가설공간의 VC (사각형·원·다각형)
Day 6  Ch4-04     Sauer-Shelah Lemma 완전 증명
Day 7  Ch4-05~07  VC bound 유도와 한계 — DL에서 왜 vacuous인가
```

</details>

<details>
<summary><b>🔴 "SLT의 수학적 기반을 완전 정복한다" — 전체 정복 (7주, 약 45~55시간)</b></summary>

<br/>

```
1주차  Chapter 1 전체 — 학습 문제 정식화
        → ERM·Bayes optimal·No Free Lunch 정리
        → 근사·추정·최적화 오차의 3분해 체화

2주차  Chapter 2 전체 — 집중부등식
        → Markov → Chebyshev → Hoeffding → McDiarmid → Bernstein
        → 각 부등식의 증명을 직접 재구성

3주차  Chapter 3 전체 — PAC Learning
        → Valiant의 원전 정의, Realizable vs Agnostic
        → Fundamental Theorem의 4중 동치 증명

4주차  Chapter 4 전체 — VC Dimension
        → Shattering의 기하학적 직관
        → Sauer-Shelah Lemma 직접 증명
        → VC bound 완전 유도 + 실전에서의 vacuous 문제

5주차  Chapter 5 전체 — Rademacher Complexity
        → Symmetrization + McDiarmid의 결합
        → Massart + Contraction + Linear/Kernel 경계
        → Bartlett-Mendelson 심층망 경계로 DL의 이론적 다리

6주차  Chapter 6 전체 — Algorithmic Stability
        → Uniform stability 정의와 Bousquet-Elisseeff
        → Ridge의 strong convexity 기반 β
        → Hardt et al.의 SGD stability — early stopping 정당화

7주차  Chapter 7 전체 — SRM과 모델 선택
        → Vapnik의 SRM 원리
        → AIC·BIC·MDL의 통합 이해
        → VC·Rademacher·Stability 세 관점 비교 — 언제 무엇을 쓸지
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | MGF·Chernoff·sub-Gaussian·martingale 집중 | Ch2 전체(집중부등식), Ch5-02(McDiarmid) |
| [mathematical-statistics-deep-dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive) | Uniform convergence, 경험과정, U-statistic | Ch3-04(Fundamental Theorem), Ch4-05(Symmetrization) |
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | 벡터공간, 차원, affine independence | Ch4-02(선형분류기 VC), Ch5-05(linear class Rademacher) |
| [calculus-optimization-deep-dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive) | 볼록성, strong convexity, gradient flow | Ch1-03(ERM), Ch6-03(Ridge stability), Ch6-04(SGD) |
| [kernel-methods-deep-dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive) | RKHS, SVM, kernel trick, Representer 정리 | Ch5-05(Kernel class Rademacher), Ch7-04(SVM 경계 비교) |
| [generalization-theory-deep-dive](https://github.com/iq-ai-lab/generalization-theory-deep-dive) | NTK·이중 강하·norm-based Rademacher·PAC-Bayes | Ch4-07·Ch5-06 — 이 레포의 **직접적 후속** |

> 💡 이 레포는 **고전 SLT(1968~2005)의 수학적 기반**에 집중합니다. Probability의 MGF·Chernoff를 선행하면 Ch2가 훨씬 자연스럽고, Math Stats의 uniform convergence를 선행하면 Ch3·Ch4의 VC 경계가 "왜 그 형태인지" 완전 투명해집니다. 현대 딥러닝이 왜 고전 VC bound로는 설명되지 않는가(Zhang et al. 2017의 "rethinking generalization")는 후속 **Generalization Theory** 레포에서 다룹니다.

---

## 📖 Reference

### 🏛️ Statistical Learning Theory 표준 교재
- **Understanding Machine Learning: From Theory to Algorithms** (Shalev-Shwartz & Ben-David, 2014) — **현대 SLT 표준**, PAC·VC·Rademacher의 통합 정리
- **Foundations of Machine Learning** (Mohri, Rostamizadeh & Talwalkar, 2018) — 수학적으로 가장 엄밀한 ML 이론 교재
- **The Nature of Statistical Learning Theory** (Vapnik, 1999) — **Vapnik 본인의 정리**, SRM 원리의 원전
- **A Probabilistic Theory of Pattern Recognition** (Devroye, Györfi & Lugosi, 1996) — 확률론적 관점의 고전

### 🎯 집중부등식 심화
- **High-Dimensional Statistics: A Non-Asymptotic Viewpoint** (Wainwright, 2019) — 집중부등식 현대 교과서
- **Concentration Inequalities: A Nonasymptotic Theory of Independence** (Boucheron, Lugosi & Massart, 2013) — 집중 특화 레퍼런스
- **Probability Inequalities for Sums of Bounded Random Variables** (Hoeffding, 1963) — **Hoeffding 원전**

### 🧩 PAC Learning 원전과 발전
- **A Theory of the Learnable** (Valiant, 1984) — **PAC 원전**
- **Occam's Razor** (Blumer, Ehrenfeucht, Haussler & Warmuth, 1987) — PAC + MDL
- **Learnability and the Vapnik-Chervonenkis Dimension** (Blumer et al., 1989) — VC와 PAC의 연결

### 📐 VC 이론
- **On the Uniform Convergence of Relative Frequencies of Events to Their Probabilities** (Vapnik & Chervonenkis, 1971) — **VC 원전**
- **Necessary and Sufficient Conditions for Uniform Convergence of Means to Their Expectations** (Vapnik & Chervonenkis, 1981)
- **Shattering All Sets of k Points in General Position** (Sauer, 1972) — **Sauer's lemma**

### 🌀 Rademacher·현대 일반화 경계
- **Rademacher and Gaussian Complexities: Risk Bounds and Structural Results** (Bartlett & Mendelson, 2002) — **Rademacher 경계 원전**
- **Local Rademacher Complexities** (Bartlett, Bousquet & Mendelson, 2005) — fast rate
- **Spectrally-normalized Margin Bounds for Neural Networks** (Bartlett, Foster & Telgarsky, 2017) — 심층망 Rademacher
- **Probability in Banach Spaces** (Ledoux & Talagrand, 1991) — **Contraction lemma** 원전

### ⚖️ Algorithmic Stability
- **Stability and Generalization** (Bousquet & Elisseeff, 2002) — **Uniform stability 원전**
- **Train Faster, Generalize Better: Stability of Stochastic Gradient Descent** (Hardt, Recht & Singer, 2016) — **SGD stability 원전**

### 🔧 모델 선택 이론
- **A New Look at the Statistical Model Identification** (Akaike, 1974) — **AIC 원전**
- **Estimating the Dimension of a Model** (Schwarz, 1978) — **BIC 원전**
- **Modeling by Shortest Data Description** (Rissanen, 1978) — **MDL 원전**
- **The Elements of Statistical Learning** (Hastie, Tibshirani & Friedman, 2009) — Ch7 Model Assessment and Selection

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"일반화 경계를 인용하는 것과, 왜 single-$h$ Hoeffding에서 Union Bound로, 유한 $\mathcal{H}$에서 VC·Sauer-Shelah로, 그리고 VC에서 Rademacher의 data-dependent tighter 경계로 — 한 계단씩 올라가야 하는지를 증명할 수 있는 것은 다르다"*

</div>
