# 07. VC 경계의 한계와 실전의 의미

## 🎯 핵심 질문

- 신경망의 VC 차원은 얼마인가? Bartlett-Harvey-Liaw-Mehrabian(2019)의 ReLU 결과는?
- **Vacuous bound**의 예: ResNet-50 (W ≈ 25M, n = 1M)에서 VC bound가 왜 완전히 의미 없는가?
- **Zhang et al. 2017 "Rethinking Generalization"**: 신경망이 랜덤 라벨을 memorize하면서도 일반화되는 "역설"은?
- VC 이론이 DL을 설명하지 못하는 이유는 정확히 무엇인가?
- **돌파구**: Norm-based Rademacher (Ch5-06), SGD implicit regularization (Ch6-04), Margin theory, PAC-Bayes, NTK
- 후속 이론들 (Generalization Theory 레포)로의 연결?

---

## 🔍 왜 VC가 실전 DL을 설명 못하는가

VC 경계의 기본 형태:
$$\mathbb{P}(\text{gen gap} \geq \epsilon) \leq 4 \Pi_\mathcal{H}(2n) e^{-n\epsilon^2/8}.$$

Sauer-Shelah:
$$\Pi_\mathcal{H}(2n) \leq (2en/d)^d, \quad d = \text{VC}(\mathcal{H}).$$

여기서 $d$가 매개변수 수 $W$ 수준이면 → $\Pi(2n) \geq (2en/W)^W$ → **bound가 1보다 커짐** (vacuous).

현대 DL: $W = 10^6 \sim 10^8$, $n = 10^3 \sim 10^6$ → $W \gg n$ → 고전 bound 완전 실패.

---

## 📐 수학적 선행 조건

- [Ch4-04](./04-sauer-shelah.md): Sauer-Shelah, 성장함수
- [Ch4-05](./05-vc-bound-derivation.md): VC 경계 유도
- [Ch5](../ch5-rademacher/): Rademacher 복잡도 (이미 학습하지 않았다면 병렬 학습 권장)
- [Ch6-04](../ch6-stability/04-sgd-stability.md): SGD stability (예고)

---

## 📖 직관적 이해

### 신경망의 VC 차원

**경험적 관찰** (Zhang et al. 2017):
- ResNet이 ImageNet (1.2M 샘플) 랜덤 라벨을 모두 외울 수 있다.
- 이는 $\text{VC}(\text{ResNet}) \geq 1.2 \times 10^6$.

**이론적 상한** (Bartlett, Harvey, Liaw, Mehrabian 2019):
- Width $W$, depth $L$인 ReLU 신경망: $\text{VC} = \Theta(WL \log W)$.
- ResNet-50: $W \approx 25 \times 10^6$, $L \approx 50$ → VC $\sim 10^8 \log W \sim 10^9$.

### Vacuous Bound의 예

**모델**: ResNet-50, W = 25M
**데이터**: ImageNet, n = 1.2M
**기대**: 일반화 gap $\approx 10\%$ (실제 관찰)

**VC bound**:
$$\text{bound} = 4 (2en/\text{VC})^{\text{VC}} e^{-n\epsilon^2/8}.$$

$n = 1.2 \times 10^6$, $\text{VC} = 10^9$를 대입하면 $n/\text{VC} \approx 10^{-3} \ll 1$ → 결과 bound가 **1 이상** (또는 극도로 loose).

실제로는 $\epsilon = 0.1$이 가능한데, bound는 $\epsilon > 0.9$ 정도에서만 의미 있음.

---

## ✏️ 엄밀한 정의

### 정의 4.7 (Vacuous Bound)

일반화 경계 $B(n, \mathcal{H}, \delta)$가 **vacuous** ⇔ $B \geq 1$ (확률적으로 자명한 경계). 즉, "gap은 1 이하"라는 trivial 진술 이상의 정보를 주지 못함.

실용적으로는 다음 중 하나를 "vacuous"의 기준으로 삼는다:
- $B \geq 1$ (극단)
- $B$가 관측되는 실제 test error보다 수십 배 큰 경우
- $B$의 **감소 속도가 $n$에 대해 실용 범위에서 의미 없는** 경우

### 정의 4.8 (효과적 복잡도 — Norm-based)

파라미터 수 $W$ 대신 **가중치 노름의 곱** $\prod_l \|W_l\|$ 또는 $\prod_l \|W_l\|_\sigma$(spectral norm)를 복잡도 척도로 쓰는 경계(Bartlett-Foster-Telgarsky 2017). 이 척도 하에선 NN이 **effective 복잡도**가 훨씬 작을 수 있다.

### 정의 4.9 (Implicit Regularization)

알고리즘 $A$(특히 SGD)가 명시적 정규화 항 없이도 **특정 종류의 해**(e.g., minimum-norm, flat minimum)를 선호하는 현상. 결과적으로 "효과적 $\mathcal{H}$"가 축소되어 일반화 개선. Ch6-04에서 formal 분석.

---

## 🔬 정리와 증명

### 정리 4.23 (ReLU 신경망의 VC 차원)

Width $W$, depth $L$인 ReLU 신경망에 대해 (activation 단위로):
$$\text{VC}(\text{ReLU net}) = \Theta(WL \log W).$$

**증명 스케치**:
- **하한**: 각 뉴런이 decision boundary 생성 → 개수 $O(WL)$만큼의 dichotomy 가능.
- **상한**: VC-dimension 계산에 linear algebraic rank argument + log factor (hyperplane 조합).

(정확한 증명은 Bartlett et al. 2019 원문 참조.)

### 정리 4.24 (Vacuous Bound의 특성)

고전 VC bound가 "의미 있으려면" (< 1):
$$4 (2en/d)^d e^{-n\epsilon^2/8} < 1$$
$$\Rightarrow (2en/d)^d e^{-n\epsilon^2/8} < 1/4$$
$$\Rightarrow d \log(2en/d) < n\epsilon^2/8.$$

$d \approx n$인 경우 (많은 파라미터):
$$n \log(2e) < n\epsilon^2/8 \Rightarrow \epsilon > \sqrt{8 \log(2e)} \approx 2.6.$$

즉, $\epsilon > 260\%$일 때만 bound < 1 → **완전히 vacuous**.

---

## 💻 NumPy 구현 검증

### 실험 1: VC bound의 vacuousness 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

def vc_bound(n, d, epsilon):
    """고전 VC bound 계산"""
    pi_2n = (2 * np.e * n / d) ** d  # Sauer-Shelah 상계
    exp_term = np.exp(-n * epsilon ** 2 / 8)
    return 4 * pi_2n * exp_term

# ResNet-50 상황
W = 25e6
L = 50
d = int(W * L * np.log(W))  # VC ≈ WL log W
n = 1.2e6

# ε에 따른 bound
epsilons = np.linspace(0.01, 3, 100)
bounds = [vc_bound(n, min(d, 1e9), eps) for eps in epsilons]  # overflow 방지

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 선형 스케일
ax1.semilogy(epsilons, bounds, 'b-', linewidth=2)
ax1.axhline(y=1, color='r', linestyle='--', label='Meaningful bound (< 1)')
ax1.axvline(x=0.1, color='g', linestyle=':', alpha=0.5, label='실제 일반화 gap ≈ 10%')
ax1.set_xlabel(r'$\epsilon$ (error tolerance)')
ax1.set_ylabel('Bound value')
ax1.set_title('ResNet-50: VC Bound Vacuousness')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([1e-3, 1e2])

# 필요한 n 계산 (고정 ε에서 bound < 1이 되려면)
epsilons_fixed = [0.1, 0.5, 1.0]
ns_needed = []
for eps in epsilons_fixed:
    # d log(2en/d) = n*eps^2/8
    # 수치적으로 풀기
    for n_trial in np.logspace(3, 9, 100):
        if vc_bound(n_trial, min(d, 1e9), eps) < 1:
            ns_needed.append(n_trial)
            break
    else:
        ns_needed.append(np.inf)

ax2.barh(range(len(epsilons_fixed)), [n if n < 1e9 else 1e9 for n in ns_needed])
ax2.set_yticks(range(len(epsilons_fixed)))
ax2.set_yticklabels([f'ε={e}' for e in epsilons_fixed])
ax2.set_xlabel('Required sample size n')
ax2.set_xscale('log')
ax2.set_title('VC bound < 1이 되려면 필요한 샘플 수')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# → vacuousness 시각적 확인
```

### 실험 2: Zhang et al. 2017 실험 재현 (간단 버전)

```python
# 작은 CNN: MNIST 크기 네트워크
# 실제 라벨 vs 랜덤 라벨에서의 수렴 차이

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# MNIST 로드
digits = load_digits()
X, Y = digits.data, digits.target

# 간단 CNN 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 파라미터 수
model = SimpleCNN()
W = sum(p.numel() for p in model.parameters())
print(f"모델 파라미터: {W}")

# 실험
n_samples = 500
X_train = torch.from_numpy(X[:n_samples]).float()
Y_train_real = torch.from_numpy(Y[:n_samples]).long()
Y_train_random = torch.randint(0, 10, (n_samples,))

def train_and_evaluate(Y_train, label_type):
    model = SimpleCNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(50):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, Y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses

losses_real = train_and_evaluate(Y_train_real, "Real")
losses_random = train_and_evaluate(Y_train_random, "Random")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(losses_real, label='실제 라벨', linewidth=2)
ax.plot(losses_random, label='랜덤 라벨', linewidth=2, linestyle='--')
ax.set_xlabel('Epoch')
ax.set_ylabel('Training loss')
ax.set_yscale('log')
ax.set_title('Zhang et al. 2017: 실제 vs 랜덤 라벨 학습')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# → 두 경우 모두 0에 수렴하지만, 일반화는 실제 라벨에서만 가능
```

---

## 🔗 DL 이론의 돌파구들

### 1. Norm-Based Rademacher (Ch5-06)

**아이디어**: VC 대신 실제 가중치 norm $\|W_l\|$을 사용.

Bartlett, Foster, Telgarsky (2017):
$$\text{Rademacher complexity} \sim \prod_l \|W_l\| / \sqrt{n}.$$

**이득**: 
- Margin을 최대화하면 norm이 작아짐 → complexity 감소
- VC보다 훨씬 tighter bound

### 2. Implicit Regularization via SGD (Ch6-04)

**Hardt, Recht, Singer (2016)**:

SGD는 finite step에서 **stable**하다:
$$\beta \leq O(\eta T).$$

따라서:
$$\text{gen gap} \leq O(\eta T) = O(\text{step size} \times \text{time}). $$

**직관**: Early stopping은 **implicit regularization** — 충분히 오래 훈련하지 않으면 복잡한 패턴 학습 불가능.

### 3. Margin Theory

**Schapire et al. (1998)**:

Large margin at classification boundary → tighter Rademacher bound.

PAC-Bayes + margin 결합하면 DL 설명 가능할 수 있음.

### 4. Neural Tangent Kernel (Generalization Theory 레포)

무한 너비 극한에서:
- NN이 kernel method (NTK)처럼 동작
- Kernel의 Rademacher complexity로 분석 가능

### 5. Double Descent (Generalization Theory 레포)

**Bartlett, Hastie, Montanari (2020)**:

- **Classical regime**: $n \gg d$ → bias-variance trade-off
- **Interpolation threshold**: $n \approx d$ → valley
- **Overparameterized regime**: $n \ll d$ → bias 감소, implicit regularization

---

## ⚖️ 가정과 한계

1. **VC는 worst-case**: 모든 분포에서 성립하려 함 → 타이트한 bound 불가능
2. **DL의 특수성**: 현대 아키텍처(ResNet, Attention)는 VC 분석에 부적합
3. **계산과 최적화**: 이론은 "배울 수 있는가"만 다루지, "어떻게 찾는가"는 미흡
4. **Empirical vs Theoretical**: 실전 성공은 이론보다 휠씬 복잡한 요소(hardware, scale, data quality) 포함

---

## 📌 핵심 정리

- **신경망 VC**: ReLU net VC = $\Theta(WL \log W)$ — 파라미터 수에 거의 비례
- **Vacuous bound**: 현대 DL에서 고전 VC bound는 1 이상 (무의미)
- **역설**: 신경망이 random label 외워도, real label에서 일반화 → VC alone으로는 설명 불가
- **돌파구**: Norm-based Rademacher, SGD stability, Margin, NTK, PAC-Bayes, Double descent
- **다음**: Generalization Theory 레포에서 현대 DL 이론 습득

---

## 🤔 생각해볼 문제

<details>
<summary><b>문제 1 (기초):</b> Vacuous bound가 정의상 무엇을 의미하는가? Bound가 "1 이상"이면 왜 무의미한가?</summary>

<br/>

**해설**. 확률부등식 $\mathbb{P}(A) \leq B$는 $B < 1$일 때만 nontrivial하다. $B \geq 1$이면 "$\mathbb{P}(A) \leq 1$" = 자명한 사실 (모든 확률은 1 이하). 따라서 bound ≥ 1이면 정보 0. 

고전 VC bound $4 \Pi(2n) e^{-\cdots}$가 1 이상이면, 일반화 gap을 어떤 값으로도 bound 못한다는 뜻. $\square$

</details>

<br/>

<details>
<summary><b>문제 2 (심화):</b> Zhang et al. (2017)이 "신경망은 random label을 memorize할 수 있다"고 보인 것이 VC 이론에 무엇을 시사하는가?</summary>

<br/>

**해설**. VC = $d$는 "최악의 경우" 정의다 — "어떤 $d+1$-점 배치도 shatter 불가능"이 존재한다는 뜻. 하지만 NN이 모든 n-point dichotomy를 실현할 수 있으면, 실제 VC ≥ n (또는 무한).

이렇게 되면:
1. Sauer-Shelah 상계 $(en/d)^d$가 의미 없음
2. Union Bound penalty $\log \Pi \sim d \log n \approx n \log n$ → 지수적으로 커짐
3. 결과 bound가 worst-case 실패

따라서 **VC bound는 DL의 실제 일반화를 설명할 수 없다**. 고전 theory의 한계를 명확히 보여주는 증거. $\square$

</details>

<br/>

<details>
<summary><b>문제 3 (ML 연결):</b> Norm-based Rademacher (Ch5-06)가 VC를 왜 "개선"하는가? ResNet-50에서 bound가 tight해질까?</summary>

<br/>

**해설**. Norm-based Rademacher는 "actual parameters의 크기"를 고려한다:
$$\mathcal{R}_n \sim \prod_l \|W_l\| / \sqrt{n}.$$

ResNet-50의 $\|W_l\|$을 크게 유지하는 훈련 (weight decay, batch norm) → norm 작음 → Rademacher 작음 → tighter bound.

실제로 모든 layer의 norm을 곱하면 여전히 큰 수이지만, **margin 대기 추가 제약** (misclassified margin)을 고려하면 더 tight할 수 있다.

하지만 여전히 완벽히 tight하진 않으며, NTK·implicit regularization 등 multiple factors가 결합되어 실제 일반화를 설명한다. $\square$

</details>

---

<div align="center">

◀ [이전: 06. Covering](./06-epsilon-net-covering.md) | [📚 README](../README.md) | [다음: Ch5-01. Rademacher 복잡도의 정의 ▶](../ch5-rademacher/01-rademacher-definition.md)

</div>
