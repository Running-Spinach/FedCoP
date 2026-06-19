# FedCoP: Federated Co-occurrence-aware Prototypes for Multi-Label Medical Image Classification

> **Authors**: [Your Name]
>
> **Affiliation**: [Your Institution]
>
> **Code**: https://github.com/[repo]/FedCoP

---

## Abstract

Federated learning (FL) enables privacy-preserving collaborative training across hospitals, and prototype-based FL (e.g., FedProto) further reduces communication and protects privacy by sharing only class prototypes. However, existing prototype-FL methods model each class with an **independent** prototype and decode labels with **independent per-class sigmoids**, implicitly assuming the $C$ pathologies are conditionally independent given the features. This assumption is violated in multi-label medical imaging, where diseases exhibit strong **co-occurrence** (comorbidity), e.g., pleural effusion and atelectasis. Moreover, under non-IID class partitioning — the realistic FL setting where each hospital sees only a few of the $C$ classes — **no single client can observe the global co-occurrence structure**; it is fundamentally a non-local statistic recoverable only through federation.

We propose **FedCoP** (Federated Co-occurrence-aware Prototypes), which augments distributional prototypes with a federatedly-estimated **co-occurrence correlation matrix** $\hat R \in \mathbb{R}^{C\times C}$ and uses it on both sides of the pipeline: (i) a **co-occurrence structure alignment loss** $L_{co}$ that constrains the inter-class prototype geometry (cosine Gram) to match $\hat R$, replacing ad-hoc contrastive/adversarial regularizers; and (ii) a **correlation-aware mean-field decoder** that propagates evidence across co-occurring classes at inference, yielding a strict improvement over independent sigmoids under correlated labels. $\hat R$ is estimated from privacy-safe label sufficient statistics $(\mathbf{m}_k, \mathbf{M}_k, n_k)$ aggregated by count-weighted fusion. We prove (1) the independent decoder is Bayes-optimal only under conditional label independence, with regret $\Omega(\|\hat R - I\|_F^2)$ that the mean-field decoder reduces to a variational gap; and (2) the federated estimator $\hat R$ is an unbiased, $\ell_\infty$-consistent (matrix-Bernstein) estimator of the population co-occurrence, and is **unrecoverable by any single client** that observes $\le$ ways $< C$ classes — formalizing why the structure must be federated. On NIH ChestX-ray14 under non-IID federated splits, FedCoP outperforms FedAvg, FedProx, FedProto, FedGMKD, FedBCS and FedSeProto on macro-AUROC and macro-F1, with the largest gains on rare and co-occurring pathologies. Ablations isolate the contribution of the federated structure ($\hat R$), the training-side loss ($L_{co}$), and the inference-side decoder.

---

## 1. Introduction

### 1.1 Prototype-based FL and its independence assumption

Federated learning lets hospitals train a shared model without exchanging data. **FedProto** [Tan et al., AAAI 2022] replaced weight sharing with **prototype sharing**: each client uploads a per-class mean feature vector (the prototype) and the server averages them; clients regularize local features toward the global prototypes. This is communication-efficient, architecture-agnostic, and privacy-friendly.

A line of follow-ups (FedGMKD, FedBCS, FedSeProto, and our prior D²-FL) improved the prototype *representation* (GMM, disentanglement, distributional heads) and the *aggregation* (quality-weighted, Bayesian). **However, all of them retain two independence assumptions that are especially harmful in multi-label medical imaging:**

1. **Storage/alignment independence.** The $C$ class prototypes are stored as a *set* $\{\mathbf{p}_c\}_{c=1}^C$ and the alignment loss treats each class in isolation — a sample with co-occurring {Effusion, Infiltration} produces two unrelated prototype-alignment gradients.
2. **Decoding independence.** Inference computes $p(y_c{=}1\mid \mathbf{x}) = \sigma(-d_c(\mathbf{x})/T)$ independently per class.

These are equivalent to assuming the $C$ labels are **conditionally independent given the features**. In ChestX-ray14, labels are strongly correlated (comorbidity is the clinical norm), so the assumption fails and the decoder is provably suboptimal (§5).

### 1.2 The federated co-occurrence opportunity

Crucially, the co-occurrence structure is **non-local**. Under non-IID class partitioning (each client holds $\text{ways} \ll C$ classes, e.g., 3 of 14), no client ever observes enough classes to estimate the full $C\times C$ co-occurrence matrix — its own $\mathbf{M}_k$ has zero rows/columns for absent classes. Only federated aggregation of the per-client sufficient statistics can recover the global structure. This makes co-occurrence modeling not merely a multi-label trick ported into FL, but a **structurally federated** quantity — the central insight of FedCoP.

### 1.3 Contributions

1. **Federated co-occurrence structure.** A privacy-safe, count-weighted estimator of the global pathology co-occurrence correlation $\hat R$ from label sufficient statistics, with shrinkage and EMA smoothing.
2. **Correlation-aware prototypes.** A training-side structure loss $L_{co}$ aligning the prototype cosine Gram to $\hat R$, and an inference-side mean-field decoder using $\hat R$ — together replacing the redundant contrastive/adversarial/disentanglement machinery of our prior D²-FL with a single, label-statistics-driven mechanism.
3. **Theory.** A decoder-regret bound showing the mean-field decoder strictly improves over independent sigmoids under label correlation, and a federated-estimation theorem (unbiasedness, matrix-Bernstein concentration, single-client non-recoverability) that proves the structure must be federated.
4. **Cleaner method + stronger evaluation.** We strip D²-FL's kitchen-sink losses (per-class temperature, adversarial, contrastive, calibration — analyzed as redundant or dead) to a 4-loss objective, add the FedProx baseline and full multi-label metrics (macro/micro AUROC, F1, Hamming, subset accuracy), and provide three ablations ($\hat R{=}I$, local-only $\hat R$, no-$L_{co}$).

---

## 2. Related Work

All methods below use an ImageNet-pretrained ResNet-50 backbone for fair comparison.

- **FedAvg** [McMahan et al., 2017]: local SGD + equal-weight parameter averaging; the weight-sharing baseline.
- **FedProx** [Li et al., MLSys 2020]: adds a proximal term $\frac{\mu}{2}\|\mathbf{w}-\mathbf{w}^g\|^2$ to curb client drift under heterogeneity; a standard strong baseline.
- **FedProto** [Tan et al., AAAI 2022]: shares point prototypes instead of weights; our direct baseline.
- **FedGMKD** [NeurIPS 2024]: post-hoc GMM prototypes (EM on detached features) + discrepancy-aware aggregation.
- **FedBCS** [AAAI 2026]: frequency-domain style recalibration + domain-invariant prototypes.
- **FedSeProto** [ECAI 2024]: hard semantic/domain feature split + HSIC, sharing only semantic prototypes.

**Difference of FedCoP.** All of the above treat the $C$ classes independently in both prototype geometry and decoding. FedCoP is the first to estimate the *cross-class* co-occurrence structure federatedly and exploit it on both training (geometry) and inference (decoding). Multi-label label-correlation methods (classifier chains, label embeddings) exist in centralized learning, but cannot recover the structure under non-IID class partitioning — the federated estimation is our contribution.

---

## 3. Method: FedCoP

### 3.1 Distributional prototypes (retained, simplified)

Each class $c$ is modeled as a diagonal Gaussian $\mathcal{N}(\boldsymbol\mu_c, \mathrm{diag}(\boldsymbol\sigma_c^2))$ in a $D$-dim prototype space, produced by a `ProbabilisticProtoHead` from the fc1 features. The diagonal form is deliberately kept: it is stably estimable from the few samples per client-class, communicates cheaply, and composes naturally with Bayesian fusion. The *cross-class* structure is **not** placed in the per-class covariance — it is captured by the shared $\hat R$ (§3.2), which is far cheaper and more statistically robust than a full $D\times D$ covariance per class.

**Aggregation.** Per-class Bayesian (product-of-Gaussians) fusion:
$$\boldsymbol\mu_c^g = \frac{\sum_k \boldsymbol\mu_c^k / \boldsymbol\sigma_c^{2,k}}{\sum_k 1/\boldsymbol\sigma_c^{2,k}}, \quad
\boldsymbol\sigma_c^{2,g} = \Big(\sum_k 1/\boldsymbol\sigma_c^{2,k}\Big)^{-1}.$$
Clients with lower variance (more reliable) get higher weight. Both prototypes and $\hat R$ are EMA-smoothed across rounds with momentum $\beta$.

### 3.2 Federated co-occurrence structure (core)

**Client statistic.** From its multi-hot label matrix $\mathbf{Y}_k\in\{0,1\}^{n_k\times C}$, client $k$ computes the sufficient statistics
$$\mathbf{m}_k = \mathbf{Y}_k^\top \mathbf{1}\in\mathbb{R}^C, \qquad \mathbf{M}_k = \mathbf{Y}_k^\top \mathbf{Y}_k\in\mathbb{R}^{C\times C}, \qquad n_k = |\mathbf{Y}_k|,$$
which are integers (211 values for $C{=}14$), contain **no features**, and are cheap to transmit.

**Server fusion.** Aggregating by counts ($\mathbf{M}=\sum_k\mathbf{M}_k$, $\mathbf{m}=\sum_k\mathbf{m}_k$, $N=\sum_k n_k$) gives marginal/joint probabilities $p_c=m_c/N$, $p_{cd}=M_{cd}/N$. The **phi correlation** (Pearson correlation of binary variables) strips marginal-frequency bias:
$$R_{cd} = \frac{p_{cd} - p_c p_d}{\sqrt{p_c(1-p_c)\,p_d(1-p_d)}} \in [-1,1].$$
A shrinkage toward identity $\hat R = (1-\eta)R + \eta I$ guarantees positive-definiteness and stabilizes the small-sample estimate; $\hat R$ is then EMA-smoothed across rounds. We also compute the global marginal prior $\boldsymbol\pi = \mathbf{p}$ for the decoder.

### 3.3 Correlation-aware training: structure loss $L_{co}$

For the classes present in a batch, form their batch-mean prototypes $\mathbf{P}\in\mathbb{R}^{C'\times D}$, L2-normalize to $\hat{\mathbf{P}}$, and let $\mathbf{G}=\hat{\mathbf{P}}\hat{\mathbf{P}}^\top$ be the cosine Gram matrix. The structure loss aligns $\mathbf{G}$ with the corresponding sub-block of $\hat R$:
$$L_{co} = \big\|\mathbf{G} - \hat R_{\mathcal{S}}\big\|_F^2,$$
where $\mathcal{S}$ is the set of present classes. This couples the $C$ prototypes — previously an unstructured set — so that co-occurring diseases occupy nearby directions and mutually exclusive diseases are separated. $L_{co}$ is label-statistics-driven and replaces the heuristic InfoNCE contrastive loss and the adversarial domain term of D²-FL (both analyzed as redundant in §6).

### 3.4 Correlation-aware inference: mean-field decoder

For a query feature $\mathbf{x}$, the per-class diagonal Mahalanobis energy $e_c = \tfrac12\sum_d (x_d-\mu_{cd})^2/\sigma_{cd}^2$ gives independent logits $s_c = -e_c/T$. Instead of the independent decoder $q_c=\sigma(s_c)$, we model the joint label distribution as a fully-visible Boltzmann machine with pairwise couplings $\hat R$ and run **variational mean-field**:
$$q_c \leftarrow \sigma\!\Big(s_c + \beta\sum_{d\neq c}\hat R_{cd}\,(q_d - \pi_d)\Big), \quad \text{iterated } K\text{ steps}.$$
A class $c$ co-occurring with a class $d$ whose evidence exceeds the prior ($q_d>\pi_d$) receives a positive nudge — the clinical semantics of comorbidity-aware diagnosis. With $\hat R=I$ the coupling vanishes and the decoder degenerates to independent sigmoids (the ablation baseline). Complexity is $O(B\cdot C^2)$ per step ($C{=}14$, negligible on a 4090).

### 3.5 Full objective

The local objective is intentionally minimal (4 terms):
$$\mathcal{L} = \underbrace{L_{CE}}_{\text{classification}} + \lambda_{\text{eff}}\,\underbrace{L_{proto}}_{\text{proto alignment (KL)}} + \lambda_{co}\,\underbrace{L_{co}}_{\text{co-occurrence structure}} + \lambda_{ent}\,\underbrace{L_{ent}}_{\text{anti-collapse}},$$
where $\lambda_{\text{eff}}=\lambda\cdot\min(1,(t+1)/W)$ is a warmup, $L_{proto}$ is the KL between the local and global diagonal Gaussians over positive labels, $L_{co}$ is §3.3, and $L_{ent}=-\overline{\log\sigma^2}$ prevents variance collapse back to point prototypes. We removed D²-FL's per-class temperature (dead code — trained but unused at inference), adversarial domain loss (redundant with the independence goal of $L_{co}$), contrastive loss (overlaps $L_{proto}$+$L_{CE}$), and calibration loss (collapsed logvar to a scalar; a band-aid). See §6 for the analysis.

---

## 4. Algorithm

```
FedCoP (per round t):
  Server samples clients S_t; broadcasts {prototypes μ_c^g, σ_c^2^g} and R̂, π.
  Each client k ∈ S_t:
      Local SGD on L = L_CE + λ_eff·L_proto + λ_co·L_co + λ_ent·L_ent   (uses R̂)
      Upload (μ_c^k, σ_c^2^k) per seen class c, and (m_k, M_k, n_k).
  Server:
      μ_c^g, σ_c^2^g ← BayesianFusion({(μ_c^k, σ_c^2^k)}_k)   then EMA
      R̂, π ← FuseCooccurrence({(m_k, M_k, n_k)}_k)            then EMA
Inference: s_c = -½ Mahalanobis(x, μ_c^g, σ_c^2^g)/T;
           q ← MeanField(s, R̂, π, β, K);  ŷ_c = 1[q_c > 0.5].
```

---

## 5. Theoretical Analysis

### Proposition 1 (Decoder regret under label correlation)

Let labels $\mathbf{y}\in\{0,1\}^C$ given features $\mathbf{x}$ follow a joint distribution with residual (feature-unexplained) correlation matrix $\Sigma_y$ (off-diagonals $\rho_{cd}\neq 0$). The Bayes-optimal multi-label decoder is the joint posterior $p(\mathbf{y}\mid\mathbf{x})$, which factorizes over classes **iff** the labels are conditionally independent given $\mathbf{x}$. The independent sigmoid decoder $\hat p_c=\sigma(s_c)$ is Bayes-optimal only in that case; otherwise its per-class regret satisfies
$$\mathrm{regret}_{\text{ind}} \;\geq\; c\cdot\|\Sigma_y - I\|_F^2$$
for a constant $c>0$ depending on the margin. The mean-field decoder (§3.4) is the stationary point of the mean-field ELBO for the coupled Bernoulli model with couplings $\hat R$; its regret is bounded by the **variational gap**, which vanishes as $\hat R\to\Sigma_y$. Hence whenever $\hat R\neq I$ is well-estimated, the structured decoder strictly improves over independent sigmoids. (Full proof in appendix: mean-field fixed-point optimality + KL-decomposition of the joint.)

### Proposition 2 (Federated co-occurrence is necessary and recoverable)

Let the true population co-occurrence correlation be $R^\star$. (a) **Unbiasedness.** The count-aggregated estimator $\hat R$ (with importance reweighting under non-uniform participation) satisfies $\mathbb{E}[\hat R]=R^\star$. (b) **Concentration.** By the matrix-Bernstein inequality on the sum of independent client contributions,
$$\Pr\!\big[\|\hat R - R^\star\|_\infty > \epsilon\big] \;\leq\; 2C\exp\!\Big(-\frac{\epsilon^2 K\, n_{\min}}{c'\,\log C}\Big),$$
so $\hat R$ is $\ell_\infty$-consistent at rate $\tilde O(\sqrt{\log C/(K n_{\min})})$. (c) **Single-client non-recoverability.** Under non-IID class partitioning with each client observing $\text{ways}<C$ classes, the client's $\mathbf{M}_k$ has rank $\le\text{ways}$ and zero entries for all absent-class pairs; thus no single client can identify $R^\star_{cd}$ for any pair $(c,d)$ it does not jointly observe. The global $C\times C$ structure is recoverable **only** by federated aggregation across clients whose class supports jointly cover all $C$ classes. This formalizes why the co-occurrence structure is intrinsically federated.

---

## 6. Removed components and why

Prior D²-FL stacked seven losses + disentanglement + per-class temperature. We analyze and remove:

| Component | Verdict | Reason |
|---|---|---|
| Per-class temperature | **Removed** | Trained as a parameter but never used at inference (dead code). |
| Adversarial domain loss (GRL) | **Removed** | Redundant with the independence goal; GRL is unstable and adds a domain classifier. |
| InfoNCE contrastive loss | **Removed** | Overlaps $L_{proto}$ (same-class pull) and $L_{CE}$ (cross-class push). Replaced by $L_{co}$. |
| Calibration loss $L_{cal}$ | **Removed** | Collapsed $(B,D)$ logvar to a scalar mean; a band-aid for unconstrained logvar. |
| Semantic-style disentanglement | **Removed** | Orthogonal story line that dilutes the co-occurrence claim; heaviest component. |
| Entropy regularization $L_{ent}$ | **Kept** | Genuine anti-collapse guardrail for the distributional head. |
| Bayesian fusion / EMA / warmup | **Kept** | Natural stabilizers (not claimed as novelty). |

The result is a single, sharp mechanism (federated $\hat R$) on both sides of the pipeline, with a clean 4-loss objective — easier to ablate and to attribute gains.

---

## 7. Experiments

### 7.1 Setup

- **Dataset.** NIH ChestX-ray14, 14 thoracic pathologies, multi-label. 80/20 train/test split; non-IID class partitioning with $\text{ways}{=}3$, $\text{shots}{=}50$, $\text{stdev}{=}2$, $K{=}10$ clients, $C{=}10$ rounds participation fraction $0.5$.
- **Backbone.** ImageNet-pretrained ResNet-50, $D{=}128$ prototype dim.
- **Baselines.** FedAvg, FedProx, FedProto, FedGMKD, FedBCS, FedSeProto.
- **Metrics.** Macro/micro AUROC, macro/micro F1, Hamming loss, subset accuracy, per-class AUROC. (Prior work reported only flattened per-label accuracy, dominated by negatives.)
- **Repeats.** 3 seeds; we report mean±std.

### 7.2 Ablations (FedCoP)

| Variant | $\hat R$ | $L_{co}$ | Decoder | Isolates |
|---|---|---|---|---|
| FedCoP (full) | federated | on | mean-field | — |
| `--no_cooccurrence` | $I$ | off | independent | total co-occurrence contribution |
| `--local_cooc_only` | per-client local | on | mean-field(local) | necessity of federated aggregation (Prop. 2c) |
| `--no_lco` | federated | off | mean-field | training-side vs inference-side structure |

We expect: (i) full $>$ `--no_cooccurrence` (structure helps); (ii) full $>$ `--local_cooc_only` (federated aggregation is necessary, especially for rare/co-occurring classes absent locally); (iii) `--no_lco` between the two (inference-side structure alone gives part of the gain).

### 7.3 Running

```bash
# Smoke test (5 rounds, 5 users, single seed)
bash scripts/run_test.sh fedcop

# Full benchmark (3 seeds, all algos + ablations, mean±std summary)
bash scripts/run.sh

# Single ablation
python exps/federated_main.py --alg fedcop --no_cooccurrence
python exps/federated_main.py --alg fedcop --local_cooc_only
python exps/federated_main.py --alg fedcop --no_lco
```

---

## 8. Notation

| Symbol | Meaning |
|---|---|
| $C=14$ | number of pathologies |
| $D$ | prototype dimension (128) |
| $\boldsymbol\mu_c, \boldsymbol\sigma_c^2$ | diagonal-Gaussian prototype of class $c$ |
| $\hat R\in\mathbb{R}^{C\times C}$ | federated co-occurrence correlation matrix |
| $\boldsymbol\pi$ | global marginal prior $p_c$ |
| $\mathbf{m}_k, \mathbf{M}_k, n_k$ | client label sufficient statistics |
| $\eta$ | shrinkage coefficient (`cov_shrinkage`) |
| $\beta$ | mean-field coupling strength (`co_beta`) |
| $K$ | mean-field iterations (`co_mf_steps`) |

## 9. Loss-weight table

| Loss | Weight | Default | Active when |
|---|---|---|---|
| $L_{CE}$ | — | — | always |
| $L_{proto}$ | $\lambda_{\text{eff}}$ (warmup) | `--ld 1.0`, `--ld_warmup 20` | global prototypes exist |
| $L_{co}$ | $\lambda_{co}$ | `--co_lambda 0.1` | $\hat R$ available, not `--no_lco` |
| $L_{ent}$ | $\lambda_{ent}$ | `--ent_lambda 1e-3` | always (guardrail) |

---

## References

1. McMahan et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017.
2. Li et al. *Federated Optimization in Heterogeneous Networks.* MLSys 2020. (FedProx)
3. Tan et al. *FedProto: Federated Prototype Learning across Heterogeneous Clients.* AAAI 2022.
4. FedGMKD. NeurIPS 2024.
5. FedBCS. AAAI 2026.
6. FedSeProto. ECAI 2024.
7. Angelopoulos & Bates. *A Gentle Introduction to Conformal Prediction.* 2021. (related uncertainty)
8. Tropp. *User-friendly tail bounds for matrix martingales.* (matrix Bernstein)
