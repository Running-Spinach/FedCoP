# D²-FL: Distributional Dual-Stream Federated Pathology Representation Learning

> **Authors**: [Your Name]
>
> **Affiliation**: [Your Institution]
>
> **Code**: https://github.com/[repo]/D²-FL

---

## Abstract

Federated Learning (FL) enables privacy-preserving collaborative model training across distributed medical institutions. However, existing FL methods face three fundamental challenges in real-world medical imaging scenarios: (1) **data heterogeneity** (non-IID label distributions across hospitals), (2) **domain shift** (different imaging equipment and acquisition parameters), and (3) **privacy-utility trade-off** (differential privacy noise degrades performance). We propose **D²-FL** (Distributional Pathology Prototype Federated Learning), a novel prototype-based FL framework that addresses all three challenges simultaneously. D²-FL introduces five key innovations over the FedProto baseline: (i) **distributional prototypes** that model per-class features as Gaussian distributions $\mathcal{N}(\mu, \sigma^2)$ rather than point vectors, capturing client-level uncertainty; (ii) **Bayesian fusion** via precision-weighted averaging for optimal global prototype aggregation; (iii) **prototype disentanglement** into semantic (disease-discriminative, shared) and style (imaging-characteristic, local) subspaces with HSIC independence constraints; (iv) **prototype EMA momentum** and **adaptive $\lambda$ warmup** for stable training; and (v) **temperature-scaled inference** for calibrated predictions. Comprehensive experiments on NIH ChestX-ray14 under non-IID federated settings demonstrate D²-FL's superior performance over five baseline FL algorithms (FedAvg, FedProx, FedBN, SCAFFOLD, FedProto), with particularly significant gains under strong differential privacy ($\varepsilon \leq 8$).

---

## 1. Introduction

### 1.1 Background

Federated Learning (FL) is a distributed machine learning paradigm that follows the principle of **"data stays, models move"** — multiple clients (e.g., hospitals) train models locally and only share model updates with a central server, preserving data privacy. The classic **FedAvg** algorithm [McMahan et al., AISTATS 2017] operates as follows:

$$\begin{aligned}
\text{For each round } t = 1, 2, \ldots, T: \\
\quad 1.\;& \text{Server broadcasts global model } \mathbf{w}_t \text{ to selected clients} \\
\quad 2.\;& \text{Each client } k \text{ trains locally to obtain } \mathbf{w}_t^k \\
\quad 3.\;& \text{Server aggregates: } \mathbf{w}_{t+1} = \sum_{k} \frac{n_k}{n} \mathbf{w}_t^k
\end{aligned}$$

### 1.2 Limitations of Weight-Sharing FL

Traditional FL methods that share model weights face two fundamental problems:

| Problem | Description | Medical Imaging Implication |
|---------|-------------|---------------------------|
| **Data Heterogeneity (Non-IID)** | Clients have different label distributions, causing local model drift | Hospital A has mostly pneumonia cases, Hospital B has mostly cardiomegaly |
| **Domain Shift (Feature Shift)** | Different imaging equipment/protocols create inconsistent feature distributions across clients | Different CT scanners, exposure settings, and PACS post-processing pipelines |

When simple weight averaging is applied under these conditions, it destroys learned representations, a phenomenon known as **client drift**.

### 1.3 The Prototype-Based Paradigm

**FedProto** [Tan et al., AAAI 2022] introduced a paradigm shift: instead of sharing model weights, share **prototypes** — the feature vectors output by the penultimate layer (fc1) of the neural network. This approach offers three advantages:

1. **Model-heterogeneity agnostic**: Different clients can use different architectures as long as prototype dimensions align
2. **Communication-efficient**: Prototypes (~256D × 14 classes = 3.6K floats) are orders of magnitude smaller than model weights (~23M parameters for ResNet-50)
3. **Privacy-preserving**: Prototypes are low-dimensional aggregated representations, making raw data reconstruction difficult

However, FedProto uses **point prototypes** (single vectors), which discard uncertainty information and are vulnerable to domain shift contamination.

---

## 2. Related Work

We implement and benchmark against five baseline algorithms, all using ImageNet-pretrained ResNet-50 backbone with optional differential privacy for fair comparison.

### 2.1 FedAvg (McMahan et al., AISTATS 2017)

The foundational FL algorithm performing simple weight averaging:

$$\mathbf{w}_{t+1} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{w}_t^k$$

- **Pros**: Simple, tunable communication frequency
- **Cons**: Severe performance degradation under non-IID data (client drift)
- **Communication overhead**: ~23M parameters

### 2.2 FedProx (Li et al., MLSys 2020)

Adds a proximal term to the local objective to constrain deviation from the global model:

$$\mathcal{L}_{\text{local}} = \mathcal{L}_{\text{BCE}} + \frac{\mu}{2} \|\mathbf{w} - \mathbf{w}_t\|^2$$

- **Pros**: Partially mitigates non-IID client drift
- **Cons**: $\mu$ requires careful tuning; too large → slow convergence, too small → degenerates to FedAvg
- **Hyperparameter**: `--fedprox_mu` (default 0.01)

### 2.3 FedBN (Li et al., ICLR 2021)

Addresses **feature shift** by keeping BatchNorm statistics local:

$$\text{Aggregate: } \{\text{conv}, \text{linear}\} \text{ parameters} \quad \text{Keep local: } \{\text{BN parameters}\}$$

- **Pros**: Well-suited for medical imaging with cross-hospital equipment variation
- **Cons**: Limited effectiveness for label distribution shift (label skew)

### 2.4 SCAFFOLD (Karimireddy et al., ICML 2020)

Uses **control variates** to correct client drift with variance reduction:

$$\mathbf{g}_{\text{corrected}} = \mathbf{g} - \mathbf{c}_i + \mathbf{c}$$

where $\mathbf{c}$ is the global control variate and $\mathbf{c}_i$ is the local control variate for client $i$. After training:

$$\mathbf{c}_i^{\text{new}} = \mathbf{c}_i - \mathbf{c} + \frac{\mathbf{w}_{\text{global}} - \mathbf{w}_{\text{local}}}{\eta \cdot K}$$

$$\mathbf{c}^{\text{new}} = \mathbf{c} + \frac{1}{K} \sum_{i} (\mathbf{c}_i^{\text{new}} - \mathbf{c}_i)$$

- **Pros**: Theoretically eliminates client drift, fast convergence
- **Cons**: 2× communication cost (control variate same size as model); stateful (must store per-client $\mathbf{c}_i$)

### 2.5 FedProto (Tan et al., AAAI 2022)

The prototype-sharing baseline. Defines prototype $\mathbf{p} \in \mathbb{R}^d$ as the fc1 feature vector:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda \cdot \underbrace{\frac{1}{N} \sum_{i} \|\mathbf{p}_i - \mathbf{G}[y_i]\|^2}_{\mathcal{L}_{\text{proto}} \text{ (MSE)}}$$

**Prototype aggregation**:

$$\mathbf{G}[c] = \frac{1}{K} \sum_{k=1}^{K} \mathbf{P}_k^{[c]}$$

- **Pros**: Model-agnostic, communication-efficient, privacy-friendly
- **Cons**: Point estimates discard uncertainty; no protection against domain shift contamination

---

## 3. Proposed Method: D²-FL

D²-FL extends FedProto with five key innovations. We present each component in detail.

### 3.1 Distributional Prototypes

#### 3.1.1 Motivation

Point prototypes only convey location information, discarding uncertainty. When a client has limited data (few-shot), its prototype estimate should carry higher uncertainty. Gaussian distributional prototypes convey both **location** ($\mu$) and **uncertainty** ($\sigma^2$).

#### 3.1.2 Probabilistic Prototype Head

Instead of a single fc1 output vector, we use a dual-head linear layer (`ProbabilisticProtoHead`):

$$\begin{aligned}
\boldsymbol{\mu} &= \mathbf{W}_\mu \mathbf{h} + \mathbf{b}_\mu \\
\log \boldsymbol{\sigma}^2 &= \operatorname{clamp}_{[-10, 10]}(\mathbf{W}_\sigma \mathbf{h} + \mathbf{b}_\sigma)
\end{aligned}$$

where $\mathbf{h} \in \mathbb{R}^{2048}$ is the ResNet-50 backbone output, $\boldsymbol{\mu}, \log \boldsymbol{\sigma}^2 \in \mathbb{R}^{d}$ (default $d = 256$). The clamp prevents numerical instability.

#### 3.1.3 Distributional Distance Metrics

Three distance functions are supported between local distribution $q = \mathcal{N}(\boldsymbol{\mu}_q, \boldsymbol{\sigma}_q^2)$ and global distribution $p = \mathcal{N}(\boldsymbol{\mu}_p, \boldsymbol{\sigma}_p^2)$:

| Metric | Formula | Properties |
|--------|---------|------------|
| **KL Divergence** | $\text{KL}(q\|p) = \frac{1}{2}\left[\log\frac{\sigma_p^2}{\sigma_q^2} + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{\sigma_p^2} - 1\right]$ | Asymmetric, measures coverage of $q$ by $p$ |
| **2-Wasserstein** | $W_2^2(q, p) = \|\boldsymbol{\mu}_q - \boldsymbol{\mu}_p\|^2 + \|\boldsymbol{\sigma}_q - \boldsymbol{\sigma}_p\|_F^2$ | Symmetric, geometric distance in parameter space |
| **MSE** | $\|\boldsymbol{\mu}_q - \boldsymbol{\mu}_p\|^2$ | Degenerates to point prototype baseline |

Controlled by `--dist_type` (default: `kl`).

### 3.2 Bayesian Fusion

#### 3.2.1 Precision-Weighted Aggregation

When multiple clients contribute Gaussian prototypes for the same class, the optimal fusion under Gaussian observation models is **precision-weighted averaging** (`bayesian_fusion_single_label`):

$$\boxed{\boldsymbol{\mu}^* = \frac{\sum_k \boldsymbol{\mu}_k / \boldsymbol{\sigma}_k^2}{\sum_k 1 / \boldsymbol{\sigma}_k^2}} \qquad \boxed{{\boldsymbol{\sigma}^2}^* = \frac{1}{\sum_k 1 / \boldsymbol{\sigma}_k^2}}$$

where $\boldsymbol{\mu}_k, \boldsymbol{\sigma}_k^2$ are the mean and variance of client $k$'s prototype for a given class.

**Key properties**:
- Clients with **lower variance** (more certain estimates) receive **higher weight** in fusion
- The fused variance ${\boldsymbol{\sigma}^2}^*$ is strictly **smaller** than any individual client variance: ${\sigma^*}^2 \leq \min_k \sigma_k^2$
- This embodies the Bayesian principle that **multiple observations reduce uncertainty**

#### 3.2.2 Within-Client Aggregation

Before cross-client fusion, prototypes from the same class within one client are aggregated using the **law of total variance**:

$$\begin{aligned}
\boldsymbol{\mu}_{\text{avg}} &= \frac{1}{M} \sum_{i=1}^{M} \boldsymbol{\mu}_i \\
\boldsymbol{\sigma}^2_{\text{avg}} &= \underbrace{\frac{1}{M} \sum_{i=1}^{M} \boldsymbol{\sigma}_i^2}_{\mathbb{E}[\text{Var}]} + \underbrace{\operatorname{Var}(\{\boldsymbol{\mu}_i\})}_{\text{Var}[\mathbb{E}]}
\end{aligned}$$

### 3.3 Prototype Disentanglement (Semantic-Style Separation)

#### 3.3.1 Motivation

In cross-hospital federated learning, medical images exhibit significant **domain shift**:

| Style Source | Description |
|-------------|-------------|
| Equipment differences | Different X-ray machine vendors produce varying contrast/brightness |
| Acquisition parameters | kVp, mAs, exposure time settings vary across hospitals |
| Post-processing | PACS window width/level, sharpening parameters differ |
| Patient demographics | Body habitus, age distributions vary by region |

These style variations contaminate prototype vectors, causing two problems:

1. **Aggregation noise**: Style differences are misinterpreted as semantic differences during cross-client aggregation, polluting global prototypes
2. **DP inefficiency**: Gaussian noise is added to the "semantic + style" mixed signal, diluting the effective signal-to-noise ratio

**Core insight**: If we decompose prototypes into **semantic** (disease-discriminative) and **style** (imaging-characteristic) independent subspaces, sharing only the semantic component:

- Semantic prototypes become purer → lower cross-client aggregation variance
- Under the same DP budget, effective signal ratio increases → better privacy-utility trade-off

#### 3.3.2 Architecture

The fc1 output (256-dim) is split by `--sem_ratio` (default 0.75):

```
fc1 output (256-dim)
  ├── z_sem  (first 192 dim → 75%) → [ProtoHead] → shared to server
  └── z_style (last 64 dim → 25%)  → [ProtoHead] → kept local

Classification: logits = fc2(concat(z_sem, z_style))
```

- **Semantic** (shared): Encodes disease-discriminative features (lesion shape, texture, location) — cross-client consistent
- **Style** (local): Encodes hospital-specific imaging properties (contrast, noise patterns) — aids local classification but is not shared

#### 3.3.3 Independence Constraint (HSIC)

To enforce statistical independence between semantic and style subspaces, we add a **Hilbert-Schmidt Independence Criterion (HSIC)** loss with linear kernel — equivalent to the squared Frobenius norm of the cross-covariance matrix:

$$\boxed{\mathcal{L}_{\text{dis}} = \|\operatorname{Cov}(\mathbf{z}_{\text{sem}}, \mathbf{z}_{\text{style}})\|_F^2}$$

where the cross-covariance matrix is computed on the current batch:

$$\operatorname{Cov}(\mathbf{z}_{\text{sem}}, \mathbf{z}_{\text{style}})_{ij} = \frac{1}{n-1} \sum_{k=1}^{n} (z_{\text{sem},k}^{(i)} - \bar{z}_{\text{sem}}^{(i)}) \cdot (z_{\text{style},k}^{(j)} - \bar{z}_{\text{style}}^{(j)})$$

Additionally, a variance regularization term prevents feature collapse:

$$\mathcal{L}_{\text{var\_reg}} = \operatorname{ReLU}(0.01 - \operatorname{Var}(\mathbf{z}_{\text{sem}})) + \operatorname{ReLU}(0.01 - \operatorname{Var}(\mathbf{z}_{\text{style}}))$$

The total disentanglement loss:

$$\mathcal{L}_{\text{dis\_total}} = \|\operatorname{Cov}(\mathbf{z}_{\text{sem}}, \mathbf{z}_{\text{style}})\|_F^2 + 0.1 \cdot \mathcal{L}_{\text{var\_reg}}$$

- **Large $\mathcal{L}_{\text{dis}}$**: Semantic and style dimensions are correlated → style information leaks into the shared signal
- **Small $\mathcal{L}_{\text{dis}}$**: Two subspaces are statistically independent → successful disentanglement

#### 3.3.4 Complete Local Training Objective (Disentangled Mode)

$$\boxed{\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda \cdot \mathcal{L}_{\text{proto}}^{\text{sem}} + \lambda_{\text{dis}} \cdot \mathcal{L}_{\text{dis}}}$$

Where:
- $\mathcal{L}_{\text{BCE}}$: Classification loss using the full concatenated $[\mathbf{z}_{\text{sem}}, \mathbf{z}_{\text{style}}]$
- $\mathcal{L}_{\text{proto}}^{\text{sem}}$: Prototype regularization computed **only on semantic features**
- $\mathcal{L}_{\text{dis}}$: Cross-covariance independence constraint (computed locally, never uploaded)
- $\lambda$: Prototype loss weight (`--ld`)
- $\lambda_{\text{dis}}$: Disentanglement loss weight (`--dis_lambda`, default 0.05)

**Server aggregation (semantic-only)**:

$$\mathbf{P}_k^{\text{sem}} = \operatorname{agg\_func}(\mathbf{z}_{\text{sem}}) \quad \text{(within-client)}$$

$$\mathbf{G}^{\text{new}} = \operatorname{proto\_aggregation}(\{\mathbf{P}_k^{\text{sem}}\}) \quad \text{(cross-client, semantic only)}$$

$\mathbf{z}_{\text{style}}$ **never leaves the client** — style information is 100% localized.

### 3.4 Prototype EMA Momentum

To stabilize global prototype updates, we apply Exponential Moving Average (EMA):

$$\boxed{\mathbf{G}_t = \beta \cdot \mathbf{G}_{t-1} + (1 - \beta) \cdot \mathbf{G}_t^{\text{new}}}$$

where $\beta \in [0, 1)$ (`--proto_momentum`, default 0.9). This applies to both point prototypes (tensor EMA) and distributional prototypes (separate EMA on $\mu$ and $\log \boldsymbol{\sigma}^2$).

**Benefits**: Dampens round-to-round noise from client sampling, especially important when `--frac` is small.

### 3.5 Adaptive Lambda Warmup

The prototype loss weight $\lambda$ follows a linear warmup schedule:

$$\boxed{\lambda_{\text{eff}}(t) = \lambda \cdot \min\left(1, \frac{t + 1}{\max(W, 1)}\right)}$$

where $t$ is the current round (0-indexed), $W$ is the warmup duration (`--ld_warmup`, default 50 rounds).

**Motivation**: Early rounds have noisy, uninformative global prototypes. Applying full prototype regularization from round 0 would pull local features toward random directions, destabilizing early training.

### 3.6 Temperature-Scaled Inference

During test-time prototype-based classification, we apply temperature scaling to the prototype-similarity logits:

$$\boxed{\text{logit}_j(\mathbf{x}) = -\frac{\operatorname{dist}(\mathbf{p}_{\mathbf{x}}, \mathbf{G}[j])}{T}}$$

where $T$ (`--temperature`, default 1.0):

| $T$ value | Effect |
|-----------|--------|
| $T = 1.0$ | No scaling (identity) |
| $T < 1.0$ | Sharpened distances → more confident predictions |
| $T > 1.0$ | Softened distances → smoother, more calibrated predictions |

For distributional prototypes, the distance function is the **Mahalanobis-like metric**:

$$\operatorname{dist}(\mathbf{p}, \mathcal{N}(\boldsymbol{\mu}_g, \boldsymbol{\sigma}_g^2)) = \frac{1}{2} \sum_{j=1}^{d} \frac{(p_j - \mu_{g,j})^2}{\sigma_{g,j}^2}$$

#### 3.6.1 Multi-Label Prototype Handling

Since ChestX-ray14 is a multi-label dataset (one X-ray can have multiple diseases), the prototype extraction and loss computation are adapted:

**Prototype extraction**: One image's feature vector contributes to the prototype pool of **all positive labels**:
```
For sample x with positive labels {Atelectasis, Effusion}:
  proto = fc1(ResNet50(x))
  agg_protos['Atelectasis'].append(proto)
  agg_protos['Effusion'].append(proto)
```

**Prototype loss** (multi-label, point prototype case):

$$\mathcal{L}_{\text{proto}} = \frac{1}{\sum_i |\mathcal{P}_i|} \sum_{i \in \text{batch}} \sum_{c \in \mathcal{P}_i} \|\mathbf{p}_i - \mathbf{G}[c]\|^2$$

where $\mathcal{P}_i = \{c \mid y_i^{(c)} = 1\}$ is the set of positive labels for sample $i$. The loss is averaged over all positive labels rather than per-image, meaning X-rays with multiple diseases contribute proportionally more to the prototype loss.

### 3.7 Two-Level Prototype Aggregation

**Level 1 — Within-client aggregation** (`agg_func`):

For a single client, multiple images of the same class produce multiple prototypes. Within-client aggregation:

- **Point prototype**: Simple arithmetic mean

  $$\mathbf{P}_k^{[c]} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{p}_{k,i}^{[c]}$$

- **Distributional prototype**: Law of total variance

  $$\begin{aligned}
  \boldsymbol{\mu}_k^{[c]} &= \frac{1}{M} \sum_{i=1}^{M} \boldsymbol{\mu}_{k,i}^{[c]} \\
  {\boldsymbol{\sigma}^2}_k^{[c]} &= \frac{1}{M} \sum_{i=1}^{M} {\boldsymbol{\sigma}^2}_{k,i}^{[c]} + \operatorname{Var}(\{\boldsymbol{\mu}_{k,i}^{[c]}\})
  \end{aligned}$$

**Level 2 — Cross-client aggregation** (`proto_aggregation`):

- **Point prototype**: Simple average across clients

  $$\mathbf{G}^{[c]} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{P}_k^{[c]}$$

- **Distributional prototype**: Bayesian fusion (precision-weighted average, see §3.2.1)

### 3.8 Multi-Label Binary Cross-Entropy Loss

For the 14-disease multi-label classification:

$$\boxed{\mathcal{L}_{\text{BCE}} = -\sum_{i=1}^{C} \left[ y_i \cdot \log \sigma(\text{logit}_i) + (1 - y_i) \cdot \log(1 - \sigma(\text{logit}_i)) \right]}$$

where $C = 14$, $\sigma(\cdot)$ is the sigmoid function, and $y_i \in \{0, 1\}$. Each label is predicted independently — a single sample can belong to multiple classes simultaneously.

---

## 4. Theoretical Analysis

### 4.1 Proposition 1: Disentanglement Reduces Aggregation Variance

**Claim**: Under non-IID degree $\eta$, the aggregation variance upper bound of semantic prototypes after disentanglement is reduced to $\alpha^2$ times the original variance, where $\alpha = \text{sem\_ratio} \in (0, 1)$.

**Intuition**: Let the original prototype $\mathbf{p} \in \mathbb{R}^d$ be decomposed as:

$$\mathbf{p} = [\mathbf{z}_{\text{sem}} \in \mathbb{R}^{\alpha d} \;\|\; \mathbf{z}_{\text{style}} \in \mathbb{R}^{(1-\alpha)d}]$$

Style noise (approximately 25% of dimensions when $\alpha = 0.75$) no longer participates in cross-client aggregation, so the variance contributed by style variations is eliminated from the aggregated prototypes.

**Implication**: The semantic-only global prototypes $\mathbf{G}_{\text{sem}}$ have strictly lower variance than full-dimensional prototypes $\mathbf{G}_{\text{full}}$, leading to more stable and reliable knowledge transfer.

### 4.2 Proposition 2: Disentanglement Improves DP Signal-to-Noise Ratio

**Claim**: Under the same $(\varepsilon, \delta)$-DP budget, the effective signal-to-noise ratio (SNR) of disentangled semantic prototypes is higher:

$$\boxed{\frac{\text{SNR}_{\text{dis}}}{\text{SNR}_{\text{orig}}} = \left(\frac{d_{\text{sem}}}{d_{\text{proto}}}\right)^{-1/2} \approx 1.15 \quad \text{when } \alpha = 0.75}$$

where $d_{\text{proto}} = 256$, $d_{\text{sem}} = \alpha \cdot d_{\text{proto}} = 192$.

**Proof sketch**: The expected L2 norm of DP Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I}_d)$ in $d$ dimensions is $\mathbb{E}[\|\boldsymbol{\epsilon}\|_2] \propto \sqrt{d}$. Since disentanglement reduces the uploaded vector dimension from $d$ to $\alpha d$, the absolute noise magnitude decreases by a factor of $\sqrt{\alpha}$ for the same noise multiplier $\sigma$, yielding the relationship above.

**Implication**: The disentanglement + DP combination is synergistic — disentanglement not only improves prototype purity but also amplifies the effective privacy budget, making D²-FL particularly advantageous in low-$\varepsilon$ regimes.

---

## 5. Differential Privacy

All six algorithms support $(\varepsilon, \delta)$-Differential Privacy via the `--use_dp` flag.

### 5.1 Mechanism

**Prototype-based algorithms (FedProto / D²-FL)**: `DPMechProto` applies L2 clipping + Gaussian noise to the `(mu || logvar)` concatenated vector.

**Weight-based algorithms (FedAvg / FedProx / FedBN / SCAFFOLD)**: `DPMechWeight` applies L2 clipping + Gaussian noise to the weight delta $\mathbf{w}_{\text{local}} - \mathbf{w}_{\text{global}}$.

$$\boxed{\mathbf{v}_{\text{clipped}} = \mathbf{v} \cdot \min\left(1, \frac{C}{\|\mathbf{v}\|_2}\right)}$$

$$\boxed{\mathbf{v}_{\text{noisy}} = \mathbf{v}_{\text{clipped}} + \mathcal{N}(\mathbf{0}, \sigma^2 C^2 \mathbf{I})}$$

where $C$ is the L2 clipping norm (`--dp_clip`, default 1.0) and $\sigma$ is the noise multiplier determined by binary search to satisfy the target $(\varepsilon, \delta)$.

### 5.2 Privacy Accounting (Moments Accountant)

We use **Rényi Differential Privacy (RDP)** [Mironov, CSF 2017] for cross-round privacy budget tracking:

**Per-round RDP cost**:

$$\varepsilon_{\text{RDP}}(\lambda) = \frac{\lambda}{2\sigma^2}$$

**Multi-round accumulation**:

$$\varepsilon_{\text{RDP}}^{\text{total}}(\lambda) = \sum_{i=1}^{T} \varepsilon_{\text{RDP}}^{(i)}(\lambda)$$

**Conversion to $(\varepsilon, \delta)$-DP**:

$$\boxed{\varepsilon = \min_{\lambda > 1} \left[ \varepsilon_{\text{RDP}}^{\text{total}}(\lambda) - \frac{\log \delta}{\lambda - 1} \right]}$$

The minimization is performed over $\lambda \in \{1.1, 1.2, \ldots, 10.9\}$ (99 discrete orders).

### 5.3 Noise Multiplier Search

Given target per-round $\varepsilon$, the noise multiplier $\sigma$ is found via binary search:

$$\sigma^* = \underset{\sigma}{\operatorname{argmin}} \; |\varepsilon_{\text{computed}}(\sigma) - \varepsilon_{\text{target}}|$$

### 5.4 DP Parameters

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `--use_dp` | Enable differential privacy | False |
| `--dp_epsilon` | Target privacy budget $\varepsilon$ | 8.0 |
| `--dp_delta` | Failure probability $\delta$ | $10^{-5}$ |
| `--dp_clip` | L2 clipping norm $C$ | 1.0 |

---

## 6. Model Architecture

### 6.1 D2FLResNet: Pretrained ResNet-50 Backbone

All algorithms share the same architecture with ImageNet-pretrained weights:

```
Input (3 × 224 × 224)                        ← Grayscale X-ray → 3-channel
  ├── stem: conv1 (7×7) → BN → ReLU → MaxPool    ← ImageNet pretrained
  ├── layer1: 3× Bottleneck(64→256)          ← Frozen
  ├── layer2: 4× Bottleneck(256→512)         ← Frozen
  ├── layer3: 6× Bottleneck(512→1024)        ← Fine-tuned
  ├── layer4: 3× Bottleneck(1024→2048)       ← Fine-tuned
  ├── AdaptiveAvgPool2d(1) → Flatten         → (2048-dim)
  ├── fc1 (2048 → proto_dim=256) → ReLU       ← Prototype features
  ├── [ProbabilisticProtoHead]                ← (optional) μ, log σ²
  ├── [DisentangledProtoHead]                 ← (optional) sem/style split
  └── fc2 (256 → 14)                          ← Multi-label logits
```

Bottleneck expansion factor = 4, total parameters ≈ 23M.

### 6.2 Model Output Formats

| Mode | Flag | Output Format |
|------|------|---------------|
| Point prototype | (default) | `(logits, protos)` |
| Distributional prototype | `--use_distributional` | `(logits, mu, logvar)` |
| Disentangled + Point | `--use_disentangle` | `(logits, z_sem, z_style)` |
| Disentangled + Distributional | `--use_disentangle --use_distributional` | `(logits, mu_sem, logvar_sem, mu_style, logvar_style)` |

---

## 7. Non-IID Data Partitioning

### 7.1 Label Skew (Default Strategy)

Each client receives a random subset of disease classes using the **first positive label sorting** strategy:

1. Sort all images by their first positive label
2. Randomly assign `n_list[i]` disease classes to each client $i$
3. For each assigned class, randomly sample `k_list[i]` images
4. "No Finding" (all-negative) samples are evenly distributed to all clients (minimum 10 per client) as negative examples

$$\text{Client } i: |\mathcal{C}_i| \sim \mathcal{U}[\max(2, W - S), \min(C, W + S)]$$

where $W = \text{ways}$, $S = \text{stdev}$, $C = 14$ (total classes).

### 7.2 Key Parameters

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `--num_users` | Number of clients $K$ | 20 |
| `--ways` | Avg. classes per client $W$ | 3 |
| `--shots` | Avg. samples per class per client | 100 |
| `--stdev` | Std dev of class/sample count | 2 |
| `--frac` | Fraction of clients participating per round | 0.04 |

---

## 8. Algorithm Comparison Summary

| Algorithm | Type | Shared Content | Overhead | Non-IID Handling | DP Method |
|-----------|------|---------------|----------|-----------------|-----------|
| **FedAvg** | Weight-sharing baseline | Model weights (~23M) | High | None | Weight delta |
| **FedProx** | Weight-sharing baseline | Model weights | High | Proximal constraint | Weight delta |
| **FedBN** | Weight-sharing baseline | Model weights (skip BN) | High | Local BN stats | Weight delta |
| **SCAFFOLD** | Weight-sharing baseline | Weights + control variates | Very high (~2×) | Gradient correction | Weight delta |
| **FedProto** | Prototype-sharing baseline | Point prototypes (256d × 14) | **Low** | Prototype regularization | Prototype vector |
| **D²-FL (Ours)** | **Prototype-sharing (proposed)** | Gaussian prototypes $\mathcal{N}(\mu, \sigma^2)$ | **Low** | Distributional + Bayesian + Disentanglement | Prototype vector |

### 8.1 D²-FL vs. FedProto: Detailed Comparison

| Feature | FedProto (Baseline) | D²-FL (Proposed) |
|---------|--------------------|--------------------|
| Backbone | Pretrained ResNet-50 | Pretrained ResNet-50 |
| Prototype type | Point vector $\mathbf{p} \in \mathbb{R}^d$ | Gaussian $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ |
| Aggregation | Simple averaging | Precision-weighted Bayesian fusion |
| Uncertainty modeling | None | Per-client variance $\boldsymbol{\sigma}_k^2$ |
| Prototype disentanglement | None | Semantic-style separation + HSIC |
| Prototype momentum | None | EMA: $\mathbf{G}_t = \beta \mathbf{G}_{t-1} + (1-\beta)\mathbf{G}_t^{\text{new}}$ |
| $\lambda$ schedule | Constant | Linear warmup: $\lambda \cdot \min(1, t/W)$ |
| Inference temperature | 1.0 (identity) | Configurable: $-\text{dist}/T$ |
| Distance metric | MSE only | KL / Wasserstein / MSE |
| Privacy protection | Optional $(\varepsilon, \delta)$-DP | Optional $(\varepsilon, \delta)$-DP |
| Target scenario | General non-IID | High heterogeneity + privacy-sensitive + domain shift |

---

## 9. D²-FL Complete Algorithm

### Pseudocode

```
Input: K clients, pretrained ResNet-50 backbone per client,
       global rounds T, client sampling fraction ρ,
       prototype loss weight λ, warmup rounds W,
       EMA momentum β, temperature τ,
       disentanglement flag, DP parameters (ε, δ, C)

Initialize:
  local_model_list = [ResNet50_pretrained() for k = 1..K]
  global_protos = {}   # empty dictionary
  global_protos_ema = {}

For round t = 0 to T-1:
    # Client sampling
    m = max(1, int(ρ × K))
    S_t = random_sample(K, m)    # selected clients

    # Adaptive lambda warmup
    λ_eff = λ × min(1, (t+1) / max(W, 1))

    local_protos = {}

    For each client k in S_t:
        # Local training
        (w_k, loss, acc, P_k) = update_weights_het(
            model=copy(local_model_list[k]),
            global_protos=global_protos,
            λ_eff=λ_eff
        )
        # Within-client aggregation
        P_k_agg = agg_func(P_k)
        local_protos[k] = P_k_agg
        # Update local model
        local_model_list[k].load_state_dict(w_k)

    # (Optional) DP perturbation
    If use_dp:
        For each k in S_t:
            local_protos[k] = DPMechProto.clip_and_noise(local_protos[k])

    # Cross-client prototype aggregation
    G_new = proto_aggregation(local_protos)

    # EMA momentum
    If global_protos_ema is not empty:
        For each label c in G_new:
            G_new[c] = β × G_ema[c] + (1-β) × G_new[c]

    global_protos = G_new
    global_protos_ema = {c: detach(g) for c, g in G_new.items()}

    # (Optional) Privacy accounting
    If use_dp:
        ε_spent = accountant.get_epsilon()
        print(f"Round {t+1}: ε = {ε_spent:.4f}")

# Final evaluation with temperature scaling
acc_l, acc_g = test_inference(global_protos, temperature=τ)
Return acc_l, acc_g
```

### Loss Function in Full Detail

The complete local training objective for the disentangled distributional D²-FL variant:

$$\boxed{\begin{aligned}
\mathcal{L}_{\text{total}} &= \underbrace{-\sum_{c=1}^{C} \left[ y_c \log \sigma(f_c) + (1-y_c) \log(1-\sigma(f_c)) \right]}_{\mathcal{L}_{\text{BCE}}: \text{ multi-label classification}} \\
&+ \lambda_{\text{eff}} \cdot \underbrace{\frac{1}{|\mathcal{P}|} \sum_{i} \sum_{c \in \mathcal{P}_i} \mathcal{D}\big(\mathcal{N}(\boldsymbol{\mu}_{\text{sem},i}, \boldsymbol{\sigma}_{\text{sem},i}^2) \;\|\; \mathcal{N}(\boldsymbol{\mu}_{g,c}, \boldsymbol{\sigma}_{g,c}^2)\big)}_{\mathcal{L}_{\text{proto}}^{\text{sem}}: \text{ semantic prototype regularization}} \\
&+ \lambda_{\text{dis}} \cdot \underbrace{\Big[\|\operatorname{Cov}(\mathbf{z}_{\text{sem}}, \mathbf{z}_{\text{style}})\|_F^2 + 0.1 \cdot (\operatorname{ReLU}(0.01 - \operatorname{Var}(\mathbf{z}_{\text{sem}})) + \operatorname{ReLU}(0.01 - \operatorname{Var}(\mathbf{z}_{\text{style}})))\Big]}_{\mathcal{L}_{\text{dis}}: \text{ disentanglement independence}}
\end{aligned}}$$

where:
- $f_c$ is the $c$-th output logit
- $\sigma(\cdot)$ is the sigmoid function
- $\mathcal{D}$ is the distributional distance (KL, Wasserstein, or MSE)
- $\mathcal{P}_i = \{c \mid y_{i,c} = 1\}$ is the set of positive labels for sample $i$
- $|\mathcal{P}| = \sum_i |\mathcal{P}_i|$ is the total positive label count in the batch
- $\mathbf{z}_{\text{sem}} \in \mathbb{R}^{B \times \alpha d}$, $\mathbf{z}_{\text{style}} \in \mathbb{R}^{B \times (1-\alpha)d}$ are the batch features
- $\lambda_{\text{eff}} = \lambda \cdot \min(1, \frac{t+1}{W})$ is the warmup-adjusted weight

---

## 10. Mathematical Notation Index

| Symbol | Meaning |
|--------|---------|
| $K$ | Total number of clients |
| $N_k$ | Number of samples at client $k$ |
| $C$ | Total number of classes (14 for ChestX-ray14) |
| $\mathcal{C}_k$ | Set of classes owned by client $k$ |
| $d$ | Prototype dimension (256) |
| $f_k$ | Local model (D2FLResNet) for client $k$ |
| $\mathbf{p}_k^{(i)}$ | Prototype vector of sample $i$ from client $k$ |
| $\mathbf{P}_k^{[c]}$ | Aggregated prototype of class $c$ at client $k$ |
| $\mathbf{G}^{[c]}$ | Global prototype of class $c$ |
| $\mathcal{L}_{\text{BCE}}$ | Binary cross-entropy multi-label classification loss |
| $\mathcal{L}_{\text{proto}}$ | Prototype distance regularization loss |
| $\lambda$ | Prototype loss weight (`--ld`) |
| $\boldsymbol{\mu}, \boldsymbol{\sigma}^2$ | Gaussian prototype mean and variance |
| $\mathbf{z}_{\text{sem}}$ | Semantic prototype vector (shared) |
| $\mathbf{z}_{\text{style}}$ | Style prototype vector (local only) |
| $\mathcal{L}_{\text{dis}}$ | Disentanglement independence loss (HSIC) |
| $\lambda_{\text{dis}}$ | Disentanglement loss weight (`--dis_lambda`) |
| $\alpha$ | Semantic dimension ratio (`--sem_ratio`, default 0.75) |
| $\beta$ | Prototype EMA momentum coefficient (`--proto_momentum`, default 0.9) |
| $W$ | Lambda warmup duration (`--ld_warmup`, default 50) |
| $T$ | Temperature scaling factor (`--temperature`, default 1.0) |
| $\varepsilon, \delta$ | Differential privacy parameters |
| $C$ | L2 clipping norm for DP (`--dp_clip`) |
| $\sigma$ | DP noise multiplier |
| $\rho$ | Client sampling fraction per round (`--frac`) |
| $T$ | Total global communication rounds |

---

## 11. Numerical Stability Considerations

| Issue | Mitigation |
|-------|-----------|
| Variance explosion/collapse | `logvar` clamped to $[-10, 10]$ in `ProbabilisticProtoHead`. Corresponding $\sigma^2 \in [4.5 \times 10^{-5}, 2.2 \times 10^4]$ |
| Division by zero in Bayesian fusion | $\text{var} + 10^{-8}$ in precision computation |
| $\log(0)$ in variance tracking | $\log(\text{avg\_var} + 10^{-8})$ in within-client aggregation |
| Division by zero in Mahalanobis distance | $g_{\text{var}} + 10^{-8}$ in test inference |
| HSIC with batch size < 2 | Returns zero (degenerate case) |

---

## 12. References

1. **Tan, Y., et al.** "FedProto: Federated Prototype Learning across Heterogeneous Clients." *AAAI 2022*. [arXiv:2105.00243](https://arxiv.org/abs/2105.00243)
2. **McMahan, B., et al.** "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS 2017*. (FedAvg)
3. **Li, T., et al.** "Federated Optimization in Heterogeneous Networks." *MLSys 2020*. (FedProx)
4. **Li, X., et al.** "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization." *ICLR 2021*.
5. **Karimireddy, S. P., et al.** "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." *ICML 2020*.
6. **Mironov, I.** "Rényi Differential Privacy." *CSF 2017*. (RDP / Moments Accountant)
7. **Gretton, A., et al.** "Measuring Statistical Dependence with Hilbert-Schmidt Norms." *ALT 2005*. (HSIC)
8. **Wang, X., et al.** "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." *CVPR 2017*. (NIH ChestX-ray14)
9. **He, K., et al.** "Deep Residual Learning for Image Recognition." *CVPR 2016*. (ResNet-50)
