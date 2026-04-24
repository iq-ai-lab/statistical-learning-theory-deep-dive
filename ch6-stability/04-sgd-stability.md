# 04. SGD의 Stability (Hardt et al. 2016)

## 🎯 핵심 질문

- **SGD (Stochastic Gradient Descent)**가 왜 generalize하는가? 랜덤 샘플링만으로 안정성이 개선되는가?
- **"적게 훈련 = implicit regularization"**이라는 주장을 수학적으로 정당화하는 방법은?
- Hardt et al. (2016)의 **$\beta \leq O(\eta T / n)$** bound는 어떻게 유도되는가? (여기서 $\eta$ = step size, $T$ = iterations)
- Deep learning에서 **조기 멈춤(early stopping)**이 왜 일반화를 돕는가?

---

## 🔍 왜 이 정리가 현대 DL 이론의 분수령인가

Zhang et al. (2017)의 "Rethinking Generalization"은 충격적인 주장을 했다: **신경망은 라벨을 "기억"할 수 있으므로, VC 차원은 엄청 크다. 그런데도 왜 일반화하는가?** (Ch4-07의 paradox)

**Hardt et al. 2016**은 이에 대한 부분적 답을 제시한다:
- VC/Rademacher: "가설공간이 작으면 일반화" → 신경망에는 적용 불가 (VC 너무 크다)
- **Stability**: "알고리즘이 안정적이면 일반화" → **SGD는 유한 단계에서 $\beta = O(\eta T / n)$로 안정적**

따라서:
- SGD는 처음 $T$ 단계에서는 안정적으로 학습하고,
- $T$ 증가 (또는 $\eta$ 증가)하면 불안정해지므로 과적합 시작,
- 이것이 "조기 멈춤"의 수학적 정당화.

---

## 📐 수학적 선행 조건

- Ch1-01~03: ERM, 위험, 3분해
- Ch2-02~03: Hoeffding, McDiarmid 부등식
- Ch6-01~02: Uniform stability, stability ⇒ generalization
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive):
  - Convex/smooth 함수
  - Gradient descent 수렴
  - **Non-expansive mappings**: $\|T(x) - T(y)\| \leq \|x - y\|$

---

## 📖 직관적 이해

### "작은 스텝 = 큰 변화 어려움"

SGD의 업데이트:
$$w_{t+1} = w_t - \eta \nabla \ell(w_t, z_{i_t})$$

여기서 $z_{i_t}$는 $t$-번째에 무작위로 뽑은 샘플.

**직관**:
- 학습률 $\eta$가 **작다** → 각 스텝의 변화가 작다
- $T$가 **작다** (조기 멈춤) → 총 누적 변화도 작다
- 따라서 샘플 하나 변화 → 모델의 change도 작음
- **안정적**

반대로:
- $\eta$ 크거나 $T$ 크다 → 모델이 크게 변할 수 있다
- 샘플 변화에 sensitive → **불안정**

### "Non-expansive" Step의 의미

Smooth convex 손실 $\ell$에서, gradient step
$$w' = w - \eta \nabla \ell(w, z)$$

은 수축 성질(contraction)을 만족한다:
$$\|w' - w'\| \leq (1 - c \eta) \|w - w''\|$$

(여기서 $c \geq 0$는 convexity parameter) — 이것을 **non-expansive**라 부른다.

직관: 두 서로 다른 "경로" (다른 샘플로 훈련한 두 모델)가 훈련이 진행되어도 **거리가 증가하지 않는다**는 뜻. 이것이 stability의 원천.

---

## ✏️ 엄밀한 정의

### 정의 6.4.1 (SGD 알고리즘)

**입력**: 샘플 $S = \{z_1, \ldots, z_n\}$, 손실 $\ell$, 초기점 $w_0$, 학습률 $\eta$, iteration $T$

**출력**: $w_T$

$$\text{for } t = 0, 1, \ldots, T-1: \quad w_{t+1} = w_t - \eta \nabla \ell(w_t, z_{i_t})$$

여기서 $i_t$는 **균등 무작위로** $\{1, \ldots, n\}$에서 선택됨 (with replacement).

### 정의 6.4.2 (Non-expansive Operator)

연산자 $T: \mathbb{R}^d \to \mathbb{R}^d$가 **non-expansive**라는 것은:
$$\|T(w) - T(w')\| \leq \|w - w'\| \quad \forall w, w'.$$

---

## 🔬 정리와 증명

### 정리 6.4 (SGD의 Uniform Stability — Hardt et al. 2016)

**가정**:
- 손실 $\ell(\cdot, z): \mathbb{R}^d \to \mathbb{R}$가 모든 $z$에 대해:
  - **$L$-Lipschitz**: $|\ell(w, z) - \ell(w', z)| \leq L \|w - w'\|$
  - **$\beta$-smooth**: $\ell(w', z) \leq \ell(w, z) + \langle \nabla \ell(w, z), w' - w \rangle + \frac{\beta}{2}\|w' - w\|^2$
- SGD: step size $\eta \leq \frac{2}{\beta}$, iterations $T$

**결론**: SGD는 다음 uniform stability를 만족한다:
$$\beta_{\text{stab}} \leq \frac{2 L^2 \eta T}{n}.$$

더 일반적으로 (Hardt et al. Thm 2.2):
$$\mathbb{E}[L_\mathcal{D}(w_T) - L_S(w_T)] \leq \frac{2 L^2 \eta T}{n}.$$

**증명 스케치**:

**Step 1**: 두 경로의 거리 추적.

두 개의 SGD trajectory를 생각하자:
- **경로 1**: 샘플 $S = \{z_1, \ldots, z_n\}$에서 훈련 → $w_0, w_1, \ldots, w_T$
- **경로 2**: 샘플 $S^{(i)} = S \setminus \{z_i\} \cup \{z_i'\}$에서 훈련 → $w'_0, w'_1, \ldots, w'_T$

둘 다 같은 초기점 $w_0$에서 시작, 같은 순서로 샘플을 뽑되 (**동일한 random seed**), $i$-번째 스텝에서만 다른 샘플을 본다.

**Step 2**: Gradient step의 non-expansive 성질.

Smoothness 가정 하에서, gradient step
$$w_{t+1} = w_t - \eta \nabla \ell(w_t, z_{i_t})$$

은 다음을 만족한다:
$$\|w_{t+1} - w'_{t+1}\| \leq (1 - \eta \mu) \|w_t - w'_t\| \quad \text{if } i_t \neq i$$

(여기서 $\mu$는 strong convexity parameter, convex 손실이면 $\mu = 0$이므로):
$$\|w_{t+1} - w'_{t+1}\| \leq \|w_t - w'_t\| \quad \text{if } i_t \neq i$$

즉 거리가 유지된다 (non-expansive).

**Step 3**: $i$-번째 스텝에서의 변화.

$t = i-1$일 때 (즉, $i$번째 스텝에서):
$$\|w_i - w'_i\| \leq \|w_{i-1} - \nabla \ell(w_{i-1}, z_i) - (w'_{i-1} - \eta \nabla \ell(w'_{i-1}, z_i'))\|$$

Lipschitz를 이용하면:
$$\|w_i - w'_i\| \leq \|w_{i-1} - w'_{i-1}\| + 2\eta L$$

(첫 번째 항은 non-expansiveness, 두 번째 항은 다른 샘플 때문의 최대 변화)

**Step 4**: 누적 바운드.

$\|w_0 - w'_0\| = 0$에서 시작하여, 각 스텝마다 최대 $2\eta L$이 더해질 수 있으므로:
$$\|w_T - w'_T\| \leq T \cdot 2\eta L.$$

**Step 5**: 손실 차이로의 변환.

Lipschitz 손실을 이용하면, 임의의 테스트 포인트 $z$에서:
$$|\ell(w_T, z) - \ell(w'_T, z)| \leq L \|w_T - w'_T\| \leq L \cdot 2\eta L T = 2L^2 \eta T.$$

이것이 $n$개 샘플의 "평균 변화"이므로 $\beta_{\text{stab}} = \frac{2L^2 \eta T}{n}$. □

### 정리 6.5 (고확률 경계 — 정리 6.2와 결합)

정리 6.4와 정리 6.2를 결합하면, 확률 $\geq 1 - \delta$로:
$$L_\mathcal{D}(w_T) - L_S(w_T) \leq \frac{2L^2 \eta T}{n} + O\left(\sqrt{\frac{\log(1/\delta)}{n}}\right).$$

**직관**: 
- 작은 $\eta T$ → 작은 gap → 좋은 일반화
- **Early stopping**: $T$를 줄이면 gap이 감소 → 조기 멈춤의 정당화

---

## 💻 NumPy 구현 검증

### 실험: SGD의 Trajectory 안정성 측정

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────
# 설정: Logistic Regression
# 손실: ℓ(w, z) = log(1 + exp(-y*w^T x))
# ─────────────────────────────────────────────

def logistic_loss(w, x, y):
    """Logistic loss: log(1 + exp(-y*w^T*x))"""
    z = y * (w @ x)
    return np.log(1 + np.exp(-z))

def logistic_gradient(w, x, y):
    """Gradient of logistic loss"""
    z = y * (w @ x)
    return -y * x / (1 + np.exp(z))

class LogisticSGD:
    def __init__(self, eta=0.01, T=100):
        self.eta = eta
        self.T = T
        self.trajectory = []
    
    def fit(self, X, y):
        """
        SGD on logistic regression
        X: (n, d), y: (n,) in {-1, +1}
        """
        n, d = X.shape
        w = np.zeros(d)
        self.trajectory = [w.copy()]
        
        for t in range(self.T):
            i = rng.integers(0, n)  # 무작위 샘플
            grad = logistic_gradient(w, X[i], y[i])
            w = w - self.eta * grad
            self.trajectory.append(w.copy())
        
        self.w = w
        return self
    
    def predict(self, X):
        return np.sign(X @ self.w)

# ─────────────────────────────────────────────
# 1. 두 경로의 거리 추적 (한 샘플 다를 때)
# ─────────────────────────────────────────────

# 데이터 생성
n, d = 100, 10
X = rng.standard_normal((n, d))
y = rng.choice([-1, 1], n)

eta = 0.01
T = 200

# 경로 1: 원본 데이터 S
sgd1 = LogisticSGD(eta=eta, T=T)
sgd1.fit(X, y)
traj1 = np.array(sgd1.trajectory)

# 경로 2: 데이터 S^{(i)} (i-번째 샘플 교체)
i_replace = 0
X_replaced = X.copy()
X_replaced[i_replace] = rng.standard_normal(d)  # 새 샘플로 교체

# 같은 random seed로 다시 훈련 (except for i-th step)
rng_fixed = np.random.default_rng(42)

class LogisticSGDControlled:
    """동일한 random seed를 사용하지만 i-th 샘플만 다르게"""
    def __init__(self, eta=0.01, T=100, seed=42, replace_idx=None, X_replace=None):
        self.eta = eta
        self.T = T
        self.seed = seed
        self.replace_idx = replace_idx
        self.X_replace = X_replace
        self.trajectory = []
        self.distances = []
    
    def fit(self, X, y, traj_reference=None):
        n, d = X.shape
        w = np.zeros(d)
        self.trajectory = [w.copy()]
        self.distances = [0.0]
        
        rng_local = np.random.default_rng(self.seed)
        
        for t in range(self.T):
            # Fixed random sequence
            i = rng_local.integers(0, n)
            
            # 원본 또는 교체 데이터 선택
            if self.replace_idx is not None and i == self.replace_idx:
                x_sample = self.X_replace[i]
            else:
                x_sample = X[i]
            
            grad = logistic_gradient(w, x_sample, y[i])
            w = w - self.eta * grad
            self.trajectory.append(w.copy())
            
            # 거리 계산 (reference trajectory 있으면)
            if traj_reference is not None:
                dist = np.linalg.norm(w - traj_reference[t+1])
                self.distances.append(dist)
        
        self.w = w
        return self

sgd2 = LogisticSGDControlled(eta=eta, T=T, seed=42, replace_idx=0, X_replace=X_replaced)
sgd2.fit(X, y, traj_reference=traj1)

# 거리 추적 그래프
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (1) 두 경로의 거리
axes[0].plot(sgd2.distances, 'b-', linewidth=2, label='||w_t - w\'_t||')
axes[0].axhline(y=2 * eta * np.max([100, 200]) / n, color='r', linestyle='--', label=f'Theory: 2ηLT/n ≈ {2*eta*T/n:.3f}')
axes[0].set_xlabel('Iteration t')
axes[0].set_ylabel('Distance ||w - w\'||')
axes[0].set_title(f'SGD Trajectory Distance (η={eta}, T={T}, n={n})')
axes[0].legend()
axes[0].grid(True)

# (2) Eta와 T의 영향
etas = [0.001, 0.01, 0.1]
Ts = [50, 100, 200]
stability_matrix = np.zeros((len(etas), len(Ts)))

for i_eta, eta_val in enumerate(etas):
    for i_T, T_val in enumerate(Ts):
        sgd1 = LogisticSGDControlled(eta=eta_val, T=T_val, seed=42)
        sgd1.fit(X, y)
        
        sgd2 = LogisticSGDControlled(eta=eta_val, T=T_val, seed=42, replace_idx=0, X_replace=X_replaced)
        sgd2.fit(X, y)
        
        # 최종 거리
        final_dist = np.linalg.norm(sgd1.w - sgd2.w)
        stability_matrix[i_eta, i_T] = final_dist

im = axes[1].imshow(stability_matrix, cmap='YlOrRd', aspect='auto')
axes[1].set_xticks(range(len(Ts)))
axes[1].set_yticks(range(len(etas)))
axes[1].set_xticklabels([f'T={T}' for T in Ts])
axes[1].set_yticklabels([f'η={eta}' for eta in etas])
axes[1].set_xlabel('Iterations T')
axes[1].set_ylabel('Learning rate η')
axes[1].set_title('||w_T - w\'_T|| (Final Distance)')
plt.colorbar(im, ax=axes[1])

for i in range(len(etas)):
    for j in range(len(Ts)):
        axes[1].text(j, i, f'{stability_matrix[i, j]:.3f}', ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig('/tmp/sgd_stability.png', dpi=100)
print("✓ Plot saved: /tmp/sgd_stability.png")

# ─────────────────────────────────────────────
# 2. Early Stopping과 Generalization
# ─────────────────────────────────────────────

print("\n" + "="*70)
print("Early Stopping: Generalization Gap vs Iteration T")
print("="*70)

# 더 큰 데이터셋
n_train, n_test = 200, 1000
d = 20

X_train = rng.standard_normal((n_train, d))
y_train = rng.choice([-1, 1], n_train)
X_test = rng.standard_normal((n_test, d))
y_test = rng.choice([-1, 1], n_test)

eta = 0.01
T_max = 500
Ts = list(range(10, T_max, 20))

train_losses = []
test_losses = []

for T in Ts:
    sgd = LogisticSGD(eta=eta, T=T)
    sgd.fit(X_train, y_train)
    
    # 훈련 손실
    train_loss = np.mean([logistic_loss(sgd.w, X_train[i], y_train[i]) for i in range(n_train)])
    train_losses.append(train_loss)
    
    # 테스트 손실
    test_loss = np.mean([logistic_loss(sgd.w, X_test[i], y_test[i]) for i in range(n_test)])
    test_losses.append(test_loss)

gaps = np.array(test_losses) - np.array(train_losses)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Ts, train_losses, 'o-', label='Train Loss', markersize=6)
ax.plot(Ts, test_losses, 's-', label='Test Loss', markersize=6)
ax.fill_between(Ts, train_losses, test_losses, alpha=0.2, label='Gap')
ax.set_xlabel('Iterations T')
ax.set_ylabel('Loss')
ax.set_title('Early Stopping: Train vs Test Loss (Implicit Regularization)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('/tmp/early_stopping.png', dpi=100)
print("\n✓ Plot saved: /tmp/early_stopping.png")

print(f"\nEarly Stopping Results:")
print(f"{'T':>5} | {'Train Loss':>12} | {'Test Loss':>12} | {'Gap':>10}")
print("-"*50)
for T, train, test, gap in zip(Ts[::5], train_losses[::5], test_losses[::5], gaps[::5]):
    print(f"{T:5d} | {train:12.6f} | {test:12.6f} | {gap:10.6f}")

print(f"\nOptimal T (minimum gap): T = {Ts[np.argmin(gaps)]}")
```

**결과 해석**:
1. 거리가 $O(\eta T)$로 증가 (정리 6.4와 일치)
2. 더 큰 $\eta$ 또는 $T$ → 거리 증가 → 불안정
3. Early stopping: 적절한 $T$에서 gap 최소화

---

## 🔗 ML 알고리즘 연결

### Implicit Regularization
정리 6.4: $\beta = O(\eta T / n)$

이것의 의미:
- 명시적 정규화항 (Ridge의 $\lambda \|w\|^2$) 없이도
- SGD의 "적게 훈련"이 정규화 역할
- 이를 **implicit regularization**이라 부른다

### Deep Learning의 일반화 설명 (부분적)
Zhang et al. "Rethinking Generalization":
- 신경망의 VC는 크지만 일반화 잘 된다
- 이유 (부분적): SGD는 유한 단계에서 stable

### 시간 복잡도 vs 정확도 Trade-off
더 오래 훈련 → 더 정확하지만 덜 안정 (over-fitting)

---

## ⚖️ 가정과 한계

### 한계 1: Convex 손실에 특화
정리 6.4는 convex 손실에 대한 증명이다. Non-convex (deep NN)은 다른 기법 필요.

### 한계 2: 상수항 무시
정리의 상수 $2L^2$는 손실함수의 Lipschitz 상수이므로, 실제로는 더 큰 bound일 수 있다.

### 한계 3: 동일한 Random Seed 가정
증명에서 두 경로가 "동일한 순서로 샘플을 본다"고 가정했다. 실제는 샘플링이 독립이므로 더 복잡.

### 한계 4: Step Size에 대한 가정
$\eta \leq 2/\beta$ 가정은 손실함수의 smoothness를 알아야 한다.

---

## 📌 핵심 정리

1. **정리 6.4**: SGD는 $\beta = O(\eta T / n)$-uniformly stable
   - Proof: non-expansive gradient step + Lipschitz loss

2. **직관**: 작은 스텝 ($\eta$ 작음) + 짧은 훈련 ($T$ 작음) = 안정적

3. **Early Stopping**: 조기 멈춤으로 $T$ 제한 → stability 개선 → generalization 개선

4. **Deep Learning 연결**: "적게 훈련 = implicit regularization"의 수학적 근거

---

## 🤔 생각해볼 문제

### 문제 6.4.1 (기초)
**문제**: 정리 6.4의 증명에서 "non-expansive" 성질이 왜 중요한가? $\eta = 0$ (스텝 없음)일 때 두 경로의 거리는 어떻게 되는가?

<details>
<summary><b>해설</b></summary>

Non-expansive: $\|T(w) - T(w')\| \leq \|w - w'\|$

이것이 의미하는 바:
- Gradient step을 밟아도 두 점 간의 거리가 유지되거나 줄어든다
- "거리 폭발"이 없다

$\eta = 0$이면:
- $w_{t+1} = w_t$ (업데이트 없음)
- 따라서 $\|w_t - w'_t\| = 0$ for all $t$
- 완벽하게 안정적이지만, 학습도 안 됨

Non-expansiveness 덕분에 현실의 $\eta > 0$에서도 거리가 선형으로만 증가 ($O(\eta T)$), exponential이 아니다.

</details>

### 문제 6.4.2 (심화)
**문제**: "Smooth convex 손실에서 gradient step은 non-expansive"라고 했다. $\eta = 1/\beta$ (smooth parameter)일 때와 $\eta = 2/\beta$일 때 수축(contraction)의 정도가 다른 이유는 무엇인가?

<details>
<summary><b>해설</b></summary>

Gradient step의 contraction:
$$w_{t+1} = w_t - \eta \nabla \ell(w_t)$$

Smoothness $\ell(w') \leq \ell(w) + \langle \nabla \ell(w), w' - w \rangle + \frac{\beta}{2}\|w'-w\|^2$를 이용.

Gradient descent가 convex 손실을 최소화한다면:
$$\|w_{t+1} - w^*\| \leq \|w_t - w^*\| \cdot (1 - \eta \cdot \text{strong convexity})$$

Smooth만 있고 (strong convexity 없으면) $\eta \leq 2/\beta$이어야만 수렴이 보장된다.

- $\eta = 1/\beta$: 조금 더 conservative, safe
- $\eta = 2/\beta$: 최대 step, just at boundary

둘 다 non-expansive이지만, constant가 다르다.

</details>

### 문제 6.4.3 (ML 연결)
**문제**: "Early stopping이 implicit regularization"이라는 주장을 정리 6.4와 실험 결과로 설명하시오. Ridge regression의 explicit $\lambda$와 SGD의 implicit "적게 훈련"의 차이는?

<details>
<summary><b>해설</b></summary>

**Ridge (explicit)**:
$$\min L_S(w) + \lambda \|w\|^2$$
정규화를 **목적함수에 직접** 추가.

**SGD (implicit)**:
- 정규화항 없음
- 하지만 $T$ 단계에서 멈추므로 $\beta = O(\eta T / n)$
- 정리 6.2: Gap = $O(\eta T / n) + O(\sqrt{\log(1/\delta)/n})$
- $T$를 줄이면 (early stopping) gap ↓

**차이**:
- Ridge: 직접적으로 $\lambda$ 조정
- SGD: "훈련을 덜" 함으로써 간접적으로 정규화

**연결**:
- Ridge: gap $\approx O(1/\lambda)$ + optimization cost
- SGD: gap $\approx O(\eta T / n)$ + 계산 효율적

실전에서는 **둘을 결합**: SGD로 훈련하면서 early stopping (implicit) + L2 정규화 (explicit)

</details>

---

<div align="center">

◀ [이전: Ridge Regression의 Stability](./03-ridge-stability.md) | [📚 README](../README.md) | [다음: Ch7-01. Structural Risk Minimization ▶](../ch7-srm-model-selection/01-srm.md)

</div>
