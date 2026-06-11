# D²-FL: Distributional Dual-Stream Federated Pathology Representation Learning

> **Authors**: [Your Name]
>
> **Affiliation**: [Your Institution]
>
> **Code**: https://github.com/[repo]/D²-FL

---

## Abstract

Federated Learning (FL) enables privacy-preserving collaborative model training across distributed medical institutions. However, existing FL methods face three fundamental challenges in real-world medical imaging scenarios: (1) **data heterogeneity** (non-IID label distributions across hospitals), (2) **domain shift** (different imaging equipment and acquisition parameters), and (3) **privacy-utility trade-off** (differential privacy noise degrades performance). We propose **D²-FL** (Distributional Dual-Stream Federated Pathology Representation Learning), a novel prototype-based FL framework that addresses all three challenges simultaneously. D²-FL introduces seven key innovations over the FedProto baseline: (i) **distributional prototypes** that model per-class features as Gaussian distributions $\mathcal{N}(\mu, \sigma^2)$ rather than point vectors, capturing client-level uncertainty; (ii) **Bayesian fusion** via precision-weighted averaging for optimal global prototype aggregation; (iii) **learnable-gate prototype disentanglement** into semantic (disease-discriminative, shared) and style (imaging-characteristic, local) subspaces, enforced by five complementary mechanisms — HSIC independence, gate entropy regularization, orthogonal constraints, adversarial domain invariance via gradient reversal, and contrastive semantic alignment via InfoNCE; (iv) **prototype calibration** through Huber loss aligning log-variance with actual prediction error; (v) **entropy regularization** to prevent variance collapse back to point prototypes; (vi) **prototype EMA momentum** and **adaptive $\lambda$ warmup** for stable training; and (vii) **per-class temperature-scaled inference** for calibrated predictions. Comprehensive experiments on NIH ChestX-ray14 under non-IID federated settings benchmark D²-FL against eight FL algorithms (FedAvg, FedProx, FedBN, SCAFFOLD, FedProto, FedGMKD, FedBCS, FedSeProto), with particularly significant gains under strong differential privacy ($\varepsilon \leq 8$).

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

We implement and benchmark against eight baseline algorithms, all using ImageNet-pretrained ResNet-50 backbone with optional differential privacy for fair comparison.

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

### 2.6 FedGMKD (NeurIPS 2024)

Uses Gaussian Mixture Models (GMM) fitted via EM algorithm instead of single point prototypes. Each class is represented by `gmm_components` (default 3) Gaussian components. Aggregation uses **discrepancy-aware weighting** where each client's contribution is weighted by $1 / \text{mean\_variance}$ (quality score).

### 2.7 FedBCS (AAAI 2026)

Performs frequency-domain style recalibration via an InstanceNorm-style 1D feature recalibration module. Aims to normalize cross-client style variations before prototype extraction, making prototypes more consistent across different imaging equipment.

### 2.8 FedSeProto (ECAI 2024)

Implements a hard-split two-branch MLP architecture that separates features into semantic and domain subspaces. Uses HSIC mutual information minimization to enforce independence between the two branches. Only the semantic branch prototypes are shared with the server.

---

## 3. Proposed Method: D²-FL

D²-FL extends FedProto with seven key innovations. We present each component in detail.

### 3.1 Distributional Prototypes

#### 3.1.1 Motivation

Point prototypes only convey location information, discarding uncertainty. When a client has limited data (few-shot), its prototype estimate should carry higher uncertainty. Gaussian distributional prototypes convey both **location** ($\mu$) and **uncertainty** ($\sigma^2$).

#### 3.1.2 Probabilistic Prototype Head

Instead of a single fc1 output vector, we use a dual-head MLP (`ProbabilisticProtoHead`) with hidden layer, ReLU activation, and LayerNorm:

$$\begin{aligned}
\mathbf{h}' &= \operatorname{LayerNorm}(\operatorname{ReLU}(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1)) \\
\boldsymbol{\mu} &= \mathbf{W}_{\mu 2} \mathbf{h}' + \mathbf{b}_{\mu 2} \\
\log \boldsymbol{\sigma}^2 &= \operatorname{clamp}_{[-10, 10]}(\mathbf{W}_{\sigma 2} \mathbf{h}' + \mathbf{b}_{\sigma 2})
\end{aligned}$$

where $\mathbf{h} \in \mathbb{R}^{2048}$ is the ResNet-50 backbone output, $\boldsymbol{\mu}, \log \boldsymbol{\sigma}^2 \in \mathbb{R}^{d}$ (default $d = 256$). The clamp prevents numerical instability. Log-variance bias is initialized to $-2.3$ (corresponding to $\sigma^2 \approx 0.1$).

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

#### 3.3.2 LearnableGate Architecture

Instead of a hard dimensional split (e.g., first 75% = semantic, last 25% = style), D²-FL uses a **learnable soft gate** that allows the network to discover which dimensions encode semantic vs. style information:

```
fc1 output h (256-dim)
    │
    ├── gate = sigmoid(Linear_{256→256}(h))     →  soft assignment per dimension
    │
    ├── z_sem  = gate ⊙ h                        →  semantic features (gate → 1)
    └── z_style = (1 - gate) ⊙ h                 →  style features (gate → 0)

Classification: logits = fc2(concat(z_sem, z_style))
Upload: z_sem only (style never leaves the client)
```

- **gate[j] ≈ 1.0**: Dimension $j$ encodes disease-discriminative information → shared
- **gate[j] ≈ 0.0**: Dimension $j$ encodes imaging-style information → kept local
- **gate[j] ≈ 0.5**: Undecided dimension (training in progress)

#### 3.3.3 Five Complementary Disentanglement Mechanisms

The disentanglement is enforced through five complementary loss components, each addressing a different failure mode:

**Mechanism 1 — HSIC Independence (Statistical Decorrelation)**

The Hilbert-Schmidt Independence Criterion with linear kernel — equivalent to the squared Frobenius norm of the cross-covariance matrix:

$$\boxed{\mathcal{L}_{\text{HSIC}} = \|\operatorname{Cov}(\mathbf{z}_{\text{sem}}, \mathbf{z}_{\text{style}})\|_F^2}$$

where the cross-covariance matrix is computed on the current batch:

$$\operatorname{Cov}(\mathbf{z}_{\text{sem}}, \mathbf{z}_{\text{style}})_{ij} = \frac{1}{n-1} \sum_{k=1}^{n} (z_{\text{sem},k}^{(i)} - \bar{z}_{\text{sem}}^{(i)}) \cdot (z_{\text{style},k}^{(j)} - \bar{z}_{\text{style}}^{(j)})$$

- Large $\mathcal{L}_{\text{HSIC}}$: semantic and style dimensions are correlated → information leakage
- Small $\mathcal{L}_{\text{HSIC}}$: two subspaces are statistically independent → successful disentanglement

**Mechanism 2 — Gate Entropy Regularization (Decisive Assignment)**

The soft gate $g \in [0, 1]^d$ can stagnate at 0.5, yielding ambiguous assignments. Binary entropy regularization pushes each dimension toward a decisive 0 or 1:

$$\boxed{\mathcal{L}_{\text{gate}} = -\frac{1}{d}\sum_{j=1}^{d} \left[ g_j \log g_j + (1 - g_j) \log(1 - g_j) \right]}$$

- $g = 0.5$: maximum entropy → maximum penalty
- $g = 0$ or $1$: zero entropy → no penalty

**Mechanism 3 — Orthogonal Constraint (Leakage Prevention)**

Even with low HSIC, information can leak between subspaces through non-linear transformations. Directly constraining the L2-normalized features to be orthogonal provides an additional safeguard:

$$\boxed{\mathcal{L}_{\text{orth}} = \|\mathbf{Z}_{\text{sem}}^{\text{norm}} \cdot \mathbf{Z}_{\text{style}}^{\text{norm},T}\|_F^2}$$

where $\mathbf{Z}^{\text{norm}}$ denotes L2-normalized feature matrices along the feature dimension.

**Mechanism 4 — Adversarial Domain Invariance (Gradient Reversal)**

Statistical independence (HSIC) is necessary but not sufficient — the network could still encode domain information in semantically meaningful ways. A domain classifier with gradient reversal actively penalizes any domain-detectable signal in semantic features:

```
z_sem (256-dim)
    │
    ├── Gradient Reversal Layer (forward: identity, backward: ×(-λ_adv))
    │   └── DomainClassifier: Linear(128) → ReLU → Linear(1) → Sigmoid
    │       └── Predicts: which client (domain) does this feature come from?
    │
    └── L_adv = BCE(domain_prediction, uniform_distribution)
```

$$\boxed{\mathcal{L}_{\text{adv}} = -\sum_{k=1}^{K} \frac{1}{K} \log \hat{y}_k^{\text{domain}}}$$

The target is the **uniform distribution** (not the true domain) — training drives the domain classifier to chance-level performance, meaning semantic features have become domain-invariant.

**Mechanism 5 — Contrastive Semantic Alignment (Disease Clustering)**

HSIC and adversarial training ensure semantic features are "clean," but don't guarantee they encode disease information well. Contrastive learning (InfoNCE) clusters semantic features of images sharing the same diseases:

$$\boxed{\mathcal{L}_{\text{contra}} = -\log \frac{\sum_{j \in \mathcal{P}_i} \exp(\text{sim}(\mathbf{z}_i^{\text{sem}}, \mathbf{z}_j^{\text{sem}}) / \tau_c)}{\sum_{j \neq i} \exp(\text{sim}(\mathbf{z}_i^{\text{sem}}, \mathbf{z}_j^{\text{sem}}) / \tau_c)}}$$

Positive pairs are defined by **Jaccard similarity > 0.5** (samples sharing at least half their positive disease labels), making this loss naturally adapted to the multi-label setting.

**Total Disentanglement Loss**:

$$\boxed{\mathcal{L}_{\text{dis}} = \mathcal{L}_{\text{HSIC}} + \lambda_g \cdot \mathcal{L}_{\text{gate}} + \lambda_o \cdot \mathcal{L}_{\text{orth}}}$$

Controlled by `--dis_lambda` (default 0.05).

### 3.4 Prototype Calibration Loss

#### 3.4.1 Motivation

The variance $\sigma^2$ output by the ProbabilisticProtoHead is learned freely. The network could "cheat" by outputting consistently small variances (appearing confident) regardless of actual prediction quality. We need the variance to honestly reflect prediction uncertainty.

#### 3.4.2 Huber Calibration

$$\boxed{\mathcal{L}_{\text{cal}} = \operatorname{Huber}\left(\operatorname{mean}\left(|\log \sigma^2 - \log \|\boldsymbol{\mu} - \boldsymbol{\mu}_g\|^2 + \epsilon|\right),\; \delta=0.5\right)}$$

- $\|\boldsymbol{\mu} - \boldsymbol{\mu}_g\|^2$: actual distance between local and global mean (target variance)
- $\log \sigma^2$: network's predicted log-variance
- The difference penalizes miscalibration: variance that doesn't match actual error
- Huber loss ($\delta=0.5$) provides robustness against outliers
- Weight $\lambda_{\text{cal}} = 0.01$: auxiliary regularizer, does not dominate training

### 3.5 Entropy Regularization

#### 3.5.1 The Variance Collapse Problem

In distributional prototype learning, there is a natural degenerate direction:

$$\log \sigma^2 \to -\infty \quad\Rightarrow\quad \sigma^2 \to 0 \quad\Rightarrow\quad \text{Gaussian prototypes collapse to point prototypes}$$

Once variance collapses to zero, the advantages of distributional prototypes (uncertainty encoding, Bayesian fusion) are entirely lost.

#### 3.5.2 Entropy Maximization

$$\boxed{\mathcal{L}_{\text{ent}} = \operatorname{mean}(-\log \sigma^2)}$$

- When $\sigma^2 \to 0$ ($\log \sigma^2 \to -\infty$): $-\log \sigma^2 \to +\infty$ → strong penalty
- When $\sigma^2$ is reasonably large: small penalty
- This maximizes the entropy of the Gaussian, encouraging the model to retain meaningful uncertainty

**Weight is deliberately tiny** ($\lambda_{\text{ent}} = 0.001$): only prevents collapse, doesn't dominate training. Active only when `--use_distributional` is enabled.

### 3.6 Prototype EMA Momentum

To stabilize global prototype updates, we apply Exponential Moving Average (EMA):

$$\boxed{\mathbf{G}_t = \beta \cdot \mathbf{G}_{t-1} + (1 - \beta) \cdot \mathbf{G}_t^{\text{new}}}$$

where $\beta \in [0, 1)$ (`--proto_momentum`, default 0.9). This applies to both point prototypes (tensor EMA) and distributional prototypes (separate EMA on $\mu$ and $\log \boldsymbol{\sigma}^2$).

**Benefits**: Dampens round-to-round noise from client sampling, especially important when `--frac` is small.

### 3.7 Adaptive Lambda Warmup

The prototype loss weight $\lambda$ follows a linear warmup schedule:

$$\boxed{\lambda_{\text{eff}}(t) = \lambda \cdot \min\left(1, \frac{t + 1}{\max(W, 1)}\right)}$$

where $t$ is the current round (0-indexed), $W$ is the warmup duration (`--ld_warmup`, default 50 rounds).

**Motivation**: Early rounds have noisy, uninformative global prototypes. Applying full prototype regularization from round 0 would pull local features toward random directions, destabilizing early training.

### 3.8 Temperature-Scaled Inference

#### 3.8.1 Per-Class Learnable Temperature

D²-FL supports **per-class learnable temperature** (`PerClassTemperature`) for prototype-based inference:

$$\boxed{\text{logit}_j(\mathbf{x}) = -\frac{\operatorname{dist}(\mathbf{p}_{\mathbf{x}}, \mathbf{G}[j])}{T_j}}$$

where $T_j = \exp(\tau_j)$ is the learnable temperature for class $j$, with $\tau_j$ initialized to $\log(1.0) = 0$. A global temperature can also be set via `--temperature` (overrides per-class).

| $T$ value | Effect |
|-----------|--------|
| $T = 1.0$ | No scaling (identity) |
| $T < 1.0$ | Sharpened distances → more confident predictions |
| $T > 1.0$ | Softened distances → smoother, more calibrated predictions |

#### 3.8.2 Distributional Prototype Distance

For distributional prototypes, the distance function is the **Mahalanobis-like metric**:

$$\operatorname{dist}(\mathbf{p}, \mathcal{N}(\boldsymbol{\mu}_g, \boldsymbol{\sigma}_g^2)) = \frac{1}{2} \sum_{j=1}^{d} \frac{(p_j - \mu_{g,j})^2}{\sigma_{g,j}^2}$$

Dimensions with high variance (uncertain) contribute less to the distance — the network naturally discounts unreliable feature dimensions.

### 3.9 Multi-Label Prototype Handling

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

### 3.10 Two-Level Prototype Aggregation

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

### 3.11 Multi-Label Binary Cross-Entropy Loss

For the 14-disease multi-label classification:

$$\boxed{\mathcal{L}_{\text{BCE}} = -\sum_{i=1}^{C} \left[ y_i \cdot \log \sigma(\text{logit}_i) + (1 - y_i) \cdot \log(1 - \sigma(\text{logit}_i)) \right]}$$

where $C = 14$, $\sigma(\cdot)$ is the sigmoid function, and $y_i \in \{0, 1\}$. Each label is predicted independently — a single sample can belong to multiple classes simultaneously.

---

## 4. Complete Loss Function

The full D²-FL local training objective with all components enabled (7 loss terms):

$$\boxed{\begin{aligned}
\mathcal{L}_{\text{total}} &= \underbrace{-\sum_{c=1}^{C} \left[ y_c \log \sigma(f_c) + (1-y_c) \log(1-\sigma(f_c)) \right]}_{\mathcal{L}_{\text{BCE}}: \text{ multi-label classification}} \\
&+ \lambda_{\text{eff}} \cdot \underbrace{\frac{1}{|\mathcal{P}|} \sum_{i} \sum_{c \in \mathcal{P}_i} \mathcal{D}\big(\mathcal{N}(\boldsymbol{\mu}_{\text{sem},i}, \boldsymbol{\sigma}_{\text{sem},i}^2) \;\|\; \mathcal{N}(\boldsymbol{\mu}_{g,c}, \boldsymbol{\sigma}_{g,c}^2)\big)}_{\mathcal{L}_{\text{proto}}^{\text{sem}}: \text{ semantic prototype regularization}} \\
&+ \lambda_{\text{dis}} \cdot \underbrace{\Big[\|\operatorname{Cov}(\mathbf{z}_{\text{sem}}, \mathbf{z}_{\text{style}})\|_F^2 + \mathcal{L}_{\text{gate}} + \mathcal{L}_{\text{orth}}\Big]}_{\mathcal{L}_{\text{dis}}: \text{ disentanglement independence}} \\
&+ \lambda_{\text{cal}} \cdot \underbrace{\operatorname{Huber}\big(|\log \sigma^2 - \log \|\boldsymbol{\mu} - \boldsymbol{\mu}_g\|^2|, \delta=0.5\big)}_{\mathcal{L}_{\text{cal}}: \text{ variance calibration}} \\
&+ \lambda_{\text{contra}} \cdot \underbrace{\mathcal{L}_{\text{InfoNCE}}(\mathbf{z}^{\text{sem}}, \text{Jaccard} > 0.5)}_{\mathcal{L}_{\text{contra}}: \text{ contrastive semantic alignment}} \\
&+ \lambda_{\text{adv}} \cdot \underbrace{\operatorname{BCE}(\operatorname{GRL}(\operatorname{DomainClassifier}(\mathbf{z}^{\text{sem}})), \text{uniform})}_{\mathcal{L}_{\text{adv}}: \text{ adversarial domain invariance}} \\
&+ \lambda_{\text{ent}} \cdot \underbrace{\operatorname{mean}(-\log \sigma^2)}_{\mathcal{L}_{\text{ent}}: \text{ entropy regularization}}
\end{aligned}}$$

where:
- $f_c$ is the $c$-th output logit
- $\sigma(\cdot)$ is the sigmoid function
- $\mathcal{D}$ is the distributional distance (KL, Wasserstein, or MSE)
- $\mathcal{P}_i = \{c \mid y_{i,c} = 1\}$ is the set of positive labels for sample $i$
- $|\mathcal{P}| = \sum_i |\mathcal{P}_i|$ is the total positive label count in the batch
- $\mathbf{z}_{\text{sem}} \in \mathbb{R}^{B \times d}$, $\mathbf{z}_{\text{style}} \in \mathbb{R}^{B \times d}$ are the batch features
- $\lambda_{\text{eff}} = \lambda \cdot \min(1, \frac{t+1}{W})$ is the warmup-adjusted weight

**Default weights**: $\lambda=1.0$, $\lambda_{\text{dis}}=0.05$, $\lambda_{\text{cal}}=0.01$, $\lambda_{\text{contra}}=0.05$, $\lambda_{\text{adv}}=0.01$, $\lambda_{\text{ent}}=0.001$. All auxiliary weights are deliberately small to prevent auxiliary losses from dominating the primary classification task during early training.

---

## 5. Theoretical Analysis

### 5.1 Proposition 1: Disentanglement Reduces Aggregation Variance

**Claim**: Under non-IID degree $\eta$, the aggregation variance upper bound of semantic prototypes after disentanglement is reduced to $\alpha^2$ times the original variance, where $\alpha = \text{sem\_ratio} \in (0, 1)$.

**Intuition**: Let the original prototype $\mathbf{p} \in \mathbb{R}^d$ be decomposed via the learnable gate. Style-related dimensions (approximately $(1-\alpha)d$ when gates converge) no longer participate in cross-client aggregation, so the variance contributed by style variations is eliminated from the aggregated prototypes.

**Implication**: The semantic-only global prototypes $\mathbf{G}_{\text{sem}}$ have strictly lower variance than full-dimensional prototypes $\mathbf{G}_{\text{full}}$, leading to more stable and reliable knowledge transfer.

### 5.2 Proposition 2: Disentanglement Improves DP Signal-to-Noise Ratio

**Claim**: Under the same $(\varepsilon, \delta)$-DP budget, the effective signal-to-noise ratio (SNR) of disentangled semantic prototypes is higher:

$$\boxed{\frac{\text{SNR}_{\text{dis}}}{\text{SNR}_{\text{orig}}} = \left(\frac{d_{\text{sem}}}{d_{\text{proto}}}\right)^{-1/2} \approx 1.15 \quad \text{when } \alpha = 0.75}$$

where $d_{\text{proto}} = 256$, $d_{\text{sem}} \approx \alpha \cdot d_{\text{proto}}$ (effective dimension after gate sparsification).

**Proof sketch**: The expected L2 norm of DP Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I}_d)$ in $d$ dimensions is $\mathbb{E}[\|\boldsymbol{\epsilon}\|_2] \propto \sqrt{d}$. Since disentanglement reduces the uploaded vector's effective dimension, the absolute noise magnitude decreases by a factor of $\sqrt{\alpha}$ for the same noise multiplier $\sigma$, yielding the relationship above.

**Implication**: The disentanglement + DP combination is synergistic — disentanglement not only improves prototype purity but also amplifies the effective privacy budget, making D²-FL particularly advantageous in low-$\varepsilon$ regimes.

---

## 6. Differential Privacy

All algorithms support $(\varepsilon, \delta)$-Differential Privacy via the `--use_dp` flag.

### 6.1 Mechanism

**Prototype-based algorithms (FedProto / FedGMKD / FedBCS / FedSeProto / D²-FL)**: `DPMechProto` applies L2 clipping + Gaussian noise to the `(mu || logvar)` concatenated vector.

**Weight-based algorithms (FedAvg / FedProx / FedBN / SCAFFOLD)**: `DPMechWeight` applies L2 clipping + Gaussian noise to the weight delta $\mathbf{w}_{\text{local}} - \mathbf{w}_{\text{global}}$.

$$\boxed{\mathbf{v}_{\text{clipped}} = \mathbf{v} \cdot \min\left(1, \frac{C}{\|\mathbf{v}\|_2}\right)}$$

$$\boxed{\mathbf{v}_{\text{noisy}} = \mathbf{v}_{\text{clipped}} + \mathcal{N}(\mathbf{0}, \sigma^2 C^2 \mathbf{I})}$$

where $C$ is the L2 clipping norm (`--dp_clip`, default 1.0) and $\sigma$ is the noise multiplier determined by binary search to satisfy the target $(\varepsilon, \delta)$.

### 6.2 Privacy Accounting (Moments Accountant)

We use **Rényi Differential Privacy (RDP)** [Mironov, CSF 2017] for cross-round privacy budget tracking:

**Per-round RDP cost**:

$$\varepsilon_{\text{RDP}}(\lambda) = \frac{\lambda}{2\sigma^2}$$

**Multi-round accumulation**:

$$\varepsilon_{\text{RDP}}^{\text{total}}(\lambda) = \sum_{i=1}^{T} \varepsilon_{\text{RDP}}^{(i)}(\lambda)$$

**Conversion to $(\varepsilon, \delta)$-DP**:

$$\boxed{\varepsilon = \min_{\lambda > 1} \left[ \varepsilon_{\text{RDP}}^{\text{total}}(\lambda) - \frac{\log \delta}{\lambda - 1} \right]}$$

The minimization is performed over $\lambda \in \{1.1, 1.2, \ldots, 10.9\}$ (99 discrete orders).

### 6.3 DP Parameters

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `--use_dp` | Enable differential privacy | False |
| `--dp_epsilon` | Target privacy budget $\varepsilon$ | 8.0 |
| `--dp_delta` | Failure probability $\delta$ | $10^{-5}$ |
| `--dp_clip` | L2 clipping norm $C$ | 1.0 |

---

## 7. Model Architecture

### 7.1 D2FLResNet: Pretrained ResNet-50 Backbone

All algorithms share the same architecture with ImageNet-pretrained weights (`IMAGENET1K_V2`):

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
  │   └── hidden(Linear+ReLU+LayerNorm) → μ_head + logvar_head
  ├── [DisentangledProtoHead]                 ← (optional) sem/style split
  │   ├── LearnableGate (Linear+sigmoid)
  │   ├── Semantic branch → [ProbabilisticProtoHead]
  │   ├── Style branch → [ProbabilisticProtoHead]
  │   └── DomainClassifier (Linear(128)+ReLU+Linear(1))
  ├── [PerClassTemperature]                   ← (optional) per-class τ
  └── fc2 (256 → 14)                          ← Multi-label logits
```

Bottleneck expansion factor = 4, total parameters ≈ 23M.

### 7.2 Model Output Formats

| Mode | Flags | Output Format |
|------|-------|---------------|
| Point prototype | (default) | `(logits, protos)` |
| Distributional prototype | `--use_distributional` | `(logits, mu, logvar)` |
| Disentangled + Point | `--use_disentangle` | `(logits, z_full, z_sem, z_style, gate)` |
| Disentangled + Distributional | `--use_distributional --use_disentangle` | `(logits, mu_full, logvar_full, mu_sem, logvar_sem, mu_style, logvar_style, gate)` |

---

## 8. Non-IID Data Partitioning

### 8.1 Label Skew (Default Strategy)

Each client receives a random subset of disease classes using the **first positive label sorting** strategy:

1. Sort all images by their first positive label
2. Randomly assign `n_list[i]` disease classes to each client $i$
3. For each assigned class, randomly sample `k_list[i]` images
4. "No Finding" (all-negative) samples are evenly distributed to all clients (minimum 10 per client) as negative examples

$$\text{Client } i: |\mathcal{C}_i| \sim \mathcal{U}[\max(2, W - S), \min(C, W + S)]$$

where $W = \text{ways}$, $S = \text{stdev}$, $C = 14$ (total classes).

### 8.2 Key Parameters

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `--num_users` | Number of clients $K$ | 20 |
| `--ways` | Avg. classes per client $W$ | 3 |
| `--shots` | Avg. samples per class per client | 100 |
| `--stdev` | Std dev of class/sample count | 2 |
| `--frac` | Fraction of clients participating per round | 0.25 |

---

## 9. Algorithm Comparison Summary

| Algorithm | Type | Shared Content | Overhead | Non-IID Handling | DP Method |
|-----------|------|---------------|----------|-----------------|-----------|
| **FedAvg** | Weight-sharing baseline | Model weights (~23M) | High | None | Weight delta |
| **FedProx** | Weight-sharing baseline | Model weights | High | Proximal constraint | Weight delta |
| **FedBN** | Weight-sharing baseline | Model weights (skip BN) | High | Local BN stats | Weight delta |
| **SCAFFOLD** | Weight-sharing baseline | Weights + control variates | Very high (~2×) | Gradient correction | Weight delta |
| **FedProto** | Prototype-sharing baseline | Point prototypes (256d × 14) | **Low** | Prototype regularization | Prototype vector |
| **FedGMKD** | Prototype-sharing baseline | GMM prototypes (3 comp × 256d × 14) | Low | GMM + discrepancy-aware aggregation | Prototype vector |
| **FedBCS** | Prototype-sharing baseline | Frequency-calibrated prototypes | Low | Freq-domain style recalibration | Prototype vector |
| **FedSeProto** | Prototype-sharing baseline | Semantic-only prototypes (128d × 14) | **Lowest** | Hard-split + HSIC MI min | Prototype vector |
| **D²-FL (Ours)** | **Prototype-sharing (proposed)** | Gaussian prototypes $\mathcal{N}(\mu, \sigma^2)$ (semantic-only) | **Low** | Distributional + Bayesian + 5-mechanism disentanglement + calibration + entropy reg | Prototype vector |

### 9.1 D²-FL vs. FedProto: Detailed Comparison

| Feature | FedProto (Baseline) | D²-FL (Proposed) |
|---------|--------------------|--------------------|
| Backbone | Pretrained ResNet-50 | Pretrained ResNet-50 |
| Prototype type | Point vector $\mathbf{p} \in \mathbb{R}^d$ | Gaussian $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ |
| Aggregation | Simple averaging | Precision-weighted Bayesian fusion |
| Uncertainty modeling | None | Per-client variance $\boldsymbol{\sigma}_k^2$ |
| Prototype disentanglement | None | LearnableGate: semantic vs. style |
| HSIC independence | None | Cross-covariance Frobenius norm |
| Gate entropy reg. | None | $-g\log g - (1-g)\log(1-g)$ |
| Orthogonal constraint | None | $\|\mathbf{Z}_{\text{sem}}^{\text{norm}} \mathbf{Z}_{\text{style}}^{\text{norm},T}\|_F^2$ |
| Adversarial domain inv. | None | Gradient reversal + domain classifier |
| Contrastive alignment | None | InfoNCE with Jaccard > 0.5 |
| Calibration loss | None | Huber(logvar, log(actual distance)) |
| Entropy regularization | None | mean(-logvar) to prevent collapse |
| Prototype momentum | None | EMA: $\mathbf{G}_t = \beta \mathbf{G}_{t-1} + (1-\beta)\mathbf{G}_t^{\text{new}}$ |
| $\lambda$ schedule | Constant | Linear warmup: $\lambda \cdot \min(1, t/W)$ |
| Inference temperature | 1.0 (identity) | Per-class learnable or global configurable |
| Distance metric | MSE only | KL / Wasserstein / MSE |
| Privacy protection | Optional $(\varepsilon, \delta)$-DP | Optional $(\varepsilon, \delta)$-DP |
| Target scenario | General non-IID | High heterogeneity + privacy-sensitive + domain shift |

---

## 10. D²-FL Complete Algorithm

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
        # Local training with full D²-FL objective
        (w_k, loss, acc, P_k) = update_weights_het(
            model=copy(local_model_list[k]),
            global_protos=global_protos,
            λ_eff=λ_eff
        )
        # Within-client aggregation (semantic prototypes only if disentangled)
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
    # → Bayesian fusion if distributional, simple averaging if point

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

---

## 11. Mathematical Notation Index

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
| $\mathcal{L}_{\text{dis}}$ | Disentanglement loss (HSIC + gate entropy + orthogonal) |
| $\mathcal{L}_{\text{cal}}$ | Calibration loss (Huber) |
| $\mathcal{L}_{\text{contra}}$ | Contrastive semantic alignment loss (InfoNCE) |
| $\mathcal{L}_{\text{adv}}$ | Adversarial domain invariance loss (GRL + BCE) |
| $\mathcal{L}_{\text{ent}}$ | Entropy regularization loss (-logvar mean) |
| $\lambda$ | Prototype loss weight (`--ld`) |
| $\lambda_{\text{dis}}$ | Disentanglement loss weight (`--dis_lambda`) |
| $\lambda_{\text{cal}}$ | Calibration loss weight (`--cal_lambda`) |
| $\lambda_{\text{contra}}$ | Contrastive loss weight (`--contra_lambda`) |
| $\lambda_{\text{adv}}$ | Adversarial loss weight (`--adv_lambda`) |
| $\lambda_{\text{ent}}$ | Entropy regularization weight (`--ent_lambda`) |
| $\boldsymbol{\mu}, \boldsymbol{\sigma}^2$ | Gaussian prototype mean and variance |
| $\mathbf{z}_{\text{sem}}$ | Semantic prototype vector (shared) |
| $\mathbf{z}_{\text{style}}$ | Style prototype vector (local only) |
| $\mathbf{g}$ | Learnable gate values (per-dimension [0,1]) |
| $\alpha$ | Semantic dimension ratio (`--sem_ratio`, default 0.75) |
| $\beta$ | Prototype EMA momentum coefficient (`--proto_momentum`, default 0.9) |
| $W$ | Lambda warmup duration (`--ld_warmup`, default 50) |
| $T_j$ | Per-class temperature scaling factor |
| $\varepsilon, \delta$ | Differential privacy parameters |
| $C$ | L2 clipping norm for DP (`--dp_clip`) |
| $\sigma$ | DP noise multiplier |
| $\rho$ | Client sampling fraction per round (`--frac`) |
| $T$ | Total global communication rounds |

---

## 12. Loss Weight Summary

| Loss Term | Symbol | Lambda Param | Default | Active Condition |
|-----------|--------|-------------|---------|-----------------|
| Classification | $\mathcal{L}_{\text{BCE}}$ | (always 1.0) | 1.0 | Always |
| Prototype Alignment | $\mathcal{L}_{\text{proto}}$ | `--ld` | 1.0 | After warmup |
| Disentanglement | $\mathcal{L}_{\text{dis}}$ | `--dis_lambda` | 0.05 | `--use_disentangle` |
| Calibration | $\mathcal{L}_{\text{cal}}$ | `--cal_lambda` | 0.01 | `--use_distributional` |
| Contrastive | $\mathcal{L}_{\text{contra}}$ | `--contra_lambda` | 0.05 | `--use_disentangle` |
| Adversarial | $\mathcal{L}_{\text{adv}}$ | `--adv_lambda` | 0.01 | `--use_disentangle` |
| Entropy Reg. | $\mathcal{L}_{\text{ent}}$ | `--ent_lambda` | 0.001 | `--use_distributional` |

Setting any lambda to 0 disables that loss term, enabling fine-grained ablation studies.

---

## 13. Numerical Stability Considerations

| Issue | Mitigation |
|-------|-----------|
| Variance explosion/collapse | `logvar` clamped to $[-10, 10]$ in `ProbabilisticProtoHead`. Corresponding $\sigma^2 \in [4.5 \times 10^{-5}, 2.2 \times 10^4]$ |
| Division by zero in Bayesian fusion | $\text{var} + 10^{-8}$ in precision computation |
| $\log(0)$ in variance tracking | $\log(\text{avg\_var} + 10^{-8})$ in within-client aggregation |
| Division by zero in Mahalanobis distance | $g_{\text{var}} + 10^{-8}$ in test inference |
| HSIC with batch size < 2 | Returns zero (degenerate case) |
| Gate stagnation at 0.5 | Binary entropy regularization pushes toward 0 or 1 |

---

## 14. References

1. **Tan, Y., et al.** "FedProto: Federated Prototype Learning across Heterogeneous Clients." *AAAI 2022*. [arXiv:2105.00243](https://arxiv.org/abs/2105.00243)
2. **McMahan, B., et al.** "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS 2017*. (FedAvg)
3. **Li, T., et al.** "Federated Optimization in Heterogeneous Networks." *MLSys 2020*. (FedProx)
4. **Li, X., et al.** "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization." *ICLR 2021*.
5. **Karimireddy, S. P., et al.** "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." *ICML 2020*.
6. **FedGMKD.** "Federated Learning with Gaussian Mixture Knowledge Distillation." *NeurIPS 2024*.
7. **FedBCS.** "Federated Learning with Broadcast Calibration and Style Harmonization." *AAAI 2026*.
8. **FedSeProto.** "Federated Semantic Prototype Learning for Domain Generalization." *ECAI 2024*.
9. **Mironov, I.** "Rényi Differential Privacy." *CSF 2017*. (RDP / Moments Accountant)
10. **Gretton, A., et al.** "Measuring Statistical Dependence with Hilbert-Schmidt Norms." *ALT 2005*. (HSIC)
11. **Goodfellow, I., et al.** "Generative Adversarial Networks." *NeurIPS 2014*. (Adversarial training / GRL)
12. **Chen, T., et al.** "A Simple Framework for Contrastive Learning of Visual Representations." *ICML 2020*. (SimCLR / InfoNCE)
13. **Wang, X., et al.** "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." *CVPR 2017*. (NIH ChestX-ray14)
14. **He, K., et al.** "Deep Residual Learning for Image Recognition." *CVPR 2016*. (ResNet-50)
15. **Guo, C., et al.** "On Calibration of Modern Neural Networks." *ICML 2017*. (Temperature scaling / calibration)
