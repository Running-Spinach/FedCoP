# D²-FL: Distributional Dual-Stream Federated Pathology Representation Learning

Privacy-Preserving Medical Image Classification via Federated Prototype Learning with Distributional Prototypes and Disentangled Representations. D²-FL achieves implicit privacy protection through **distributional prototypes** (encoding uncertainty per client) and **semantic disentanglement** (sharing only disease-relevant features, isolating domain-specific style).

All algorithms use **ImageNet pretrained ResNet-50** backbone for fair comparison. Based on [FedProto (AAAI 2022)](https://arxiv.org/abs/2105.00243), this project provides **5 FL baselines** + the proposed **D²-FL** method:

### Baselines (comparison algorithms)

| Algorithm | Shared Info | Key Mechanism | Paper |
|-----------|------------|---------------|-------|
| **FedAvg** | Model weights | Simple weight averaging | AISTATS 2017 |
| **FedProto** | Class prototypes (point) | Prototype regularization + nearest-neighbor | AAAI 2022 |
| **FedGMKD** | GMM prototypes | EM-fitted GMM per class + discrepancy-aware aggregation | NeurIPS 2024 |
| **FedBCS** | Frequency-calibrated prototypes | InstanceNorm-style feature recalibration | AAAI 2026 |
| **FedSeProto** | Hard-split semantic/domain protos | Two-branch MLP + HSIC mutual info minimization | ECAI 2024 |

### Proposed Method: D²-FL

| Algorithm | Shared Info | Key Innovation |
|-----------|------------|----------------|
| **D²-FL** | Gaussian prototypes `N(mu, sigma^2)` (semantic-only when disentangled) | Distributional prototypes + Semantic-Style disentanglement + Bayesian fusion + Proto EMA + Temperature scaling |

Key features of D²-FL:

- **Distributional Prototypes** — each class prototype is modeled as a Gaussian distribution `N(mu, sigma^2)`, capturing per-client uncertainty
- **Semantic-Style Disentanglement** — features are decomposed into semantic (disease-relevant, shared) and style (hospital-specific, local) subspaces via a learnable soft gate
- **Disentanglement Enforcement** — four complementary losses ensure clean separation:
  - *HSIC Independence* — statistical decorrelation between semantic and style features
  - *Gate Entropy Regularization* — encourages decisive 0/1 gate assignments per dimension
  - *Orthogonal Constraint* — prevents information leakage between subspaces
  - *Adversarial Domain Invariance* — semantic features fool a domain classifier via gradient reversal
  - *Contrastive Semantic Alignment* — InfoNCE loss clusters same-disease semantic features across clients
- **Bayesian Fusion** — precision-weighted aggregation: clients with lower variance get higher weight
- **Prototype Calibration** — Huber loss ensures log-variance honestly reflects actual uncertainty
- **Entropy Regularization** — prevents variance collapse back to point prototypes (logvar → -∞)
- **Prototype EMA Momentum** — exponential moving average of global prototypes for stability
- **Adaptive Lambda Warmup** — gradual increase of prototype loss weight to avoid early bias
- **Per-Class Temperature Scaling** — learnable per-class temperature for calibrated prototype-distance inference
- **Implicit Privacy Protection** — three mechanisms combine to protect client data without adding explicit noise:
  - *Distributional prototypes* encode per-client uncertainty via variance parameters — a high-variance client naturally reveals less about its training data
  - *Semantic-style disentanglement* ensures only disease-relevant features leave the client; hospital-specific style features stay local
  - *Prototype-level sharing* avoids transmitting raw model weights or gradients, providing stronger privacy than weight-sharing FL
  - These mechanisms offer *de facto* privacy benefits without the utility loss of formal DP noise — see [Privacy](#privacy) section for detailed analysis
- **Multi-label Classification** — adapted for Chest X-ray diagnosis (14 disease labels per image)
- **Task Heterogeneity** — each client sees only a subset of disease classes (Non-IID label skew)

## Requirements

- Python 3.7+
- PyTorch 1.7+
- Torchvision 0.8+
- NumPy, Pandas, Pillow, tqdm
- scikit-learn, matplotlib (for visualization)

See [requirements.txt](requirements.txt) for the full list.

## Data Preparation

Download the [NIH ChestX-ray14 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) and extract to:

```
../data/chestxray/
  images/
    00000001_000.png
    00000002_000.png
    ...
  Data_Entry_2017.csv
```

The `--data_dir` flag controls the root data directory (default: `../data/`).

## Project Structure

```
D²-FL/
├── exps/
│   └── federated_main.py          # Main entry point (6 algorithms)
├── lib/
│   ├── options.py                 # Argument parsing
│   ├── utils.py                   # Data loading, prototype aggregation
│   ├── update.py                  # Local training, test inference
│   ├── sampling.py                # IID/Non-IID data partitioning
│   ├── chestxray.py               # ChestXray14 dataset class
│   ├── visualize.py               # t-SNE prototype visualization
│   ├── models/
│   │   └── resnet.py              # D2FLResNet / ResNet50 backbone
│   ├── dist_proto/
│   │   ├── proto_head.py          # ProbabilisticProtoHead + PerClassTemperature
│   │   ├── losses.py              # KL / Wasserstein / MSE / Calibration / Entropy reg
│   │   ├── aggregation.py         # Bayesian precision-weighted fusion
│   │   └── disentangle.py         # DisentangledProtoHead + LearnableGate + HSIC + Adversarial + Contrastive
│   └── dp/
│       └── __init__.py            # Privacy notes (DP code removed; see Privacy section)
├── figures/                       # Architecture diagrams
├── paper/                         # Paper & theory documentation
├── scripts/
│   └── run.sh                     # Launch script (6 algorithms)
├── requirements.txt
└── README.md
```

## Running

Select the algorithm via `--alg` (default: `d2fl`). All algorithms use pretrained ResNet-50 backbone.

### Baselines
```bash
# FedProto (original, point prototypes only)
python exps/federated_main.py --alg fedproto \
    --ways 5 --shots 100 --num_users 20 --rounds 200 --ld 1.0

# FedAvg
python exps/federated_main.py --alg fedavg \
    --ways 5 --shots 100 --num_users 20 --rounds 200

# FedGMKD (GMM prototypes + discrepancy-aware aggregation)
python exps/federated_main.py --alg fedgmkd \
    --gmm_components 3 --ways 5 --num_users 20 --rounds 200

# FedBCS (frequency-domain style recalibration)
python exps/federated_main.py --alg fedbcs \
    --ways 5 --num_users 20 --rounds 200

# FedSeProto (hard-split semantic-domain decoupling + HSIC)
python exps/federated_main.py --alg fedseproto \
    --mi_lambda 0.05 --ways 5 --num_users 20 --rounds 200
```

### Proposed: D²-FL
```bash
# Point prototype mode (FedProto baseline)
python exps/federated_main.py --alg d2fl \
    --ways 5 --shots 100 --num_users 20 --rounds 200 --ld 1.0

# Distributional prototype (Gaussian)
python exps/federated_main.py --alg d2fl \
    --use_distributional --dist_type kl \
    --ways 5 --rounds 200 --ld 1.0

# Disentanglement mode (semantic-style separation)
python exps/federated_main.py --alg d2fl \
    --use_disentangle --sem_ratio 0.75 --dis_lambda 0.05 \
    --ways 5 --rounds 200 --ld 1.0

# Full D²-FL (distributional + disentangled + all enhancements)
python exps/federated_main.py --alg d2fl \
    --use_distributional --dist_type kl \
    --use_disentangle --dis_lambda 0.05 \
    --cal_lambda 0.01 --contra_lambda 0.05 --adv_lambda 0.01 --ent_lambda 0.001 \
    --proto_momentum 0.9 --temperature 1.0 --ld_warmup 50 \
    --ways 5 --num_users 20 --rounds 100
```

## Options

### Algorithm Selection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alg` | d2fl | FL algorithm: fedavg / fedproto / fedgmkd / fedbcs / fedseproto / d2fl |

### Federated Learning
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rounds` | 100 | Global communication rounds |
| `--num_users` | 20 | Number of clients |
| `--frac` | 0.25 | Fraction of clients participating per round |
| `--train_ep` | 1 | Local epochs per round |
| `--local_bs` | 4 | Local batch size |
| `--lr` | 0.01 | Learning rate |
| `--momentum` | 0.5 | SGD momentum |
| `--optimizer` | sgd | Optimizer: sgd / adam |
| `--seed` | 1234 | Random seed |

### Model
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | resnet50 | Backbone: resnet50 |
| `--num_classes` | 14 | Number of classes |
| `--proto_dim` | 256 | Prototype vector dimension |
| `--image_size` | 224 | Input image size |
| `--pretrained` | True | Use ImageNet pretrained ResNet-50 |

### Task Heterogeneity
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ways` | 3 | Avg classes per client |
| `--shots` | 100 | Avg samples per class per client |
| `--stdev` | 2 | Std deviation of ways/shots |
| `--iid` | 0 | Use IID split (0 = Non-IID) |
| `--unequal` | 0 | Unequal data amounts across clients |

### Prototype Learning (FedProto / D²-FL shared)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ld` | 1.0 | Prototype loss weight lambda |
| `--proto_dim` | 256 | Prototype vector dimension |
| `--ft_round` | 10 | Fine-tuning rounds |

### Distributional Prototypes (D²-FL only)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_distributional` | False | Enable Gaussian prototypes `N(mu, sigma^2)` |
| `--dist_type` | kl | Distance type: kl / wasserstein / mse |

### Disentanglement (D²-FL only)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_disentangle` | False | Enable semantic-style disentanglement |
| `--sem_ratio` | 0.75 | Target semantic dimension ratio |
| `--dis_lambda` | 0.05 | Disentanglement loss weight (HSIC + gate entropy + orthogonal) |
| `--cal_lambda` | 0.01 | Prototype calibration loss weight |
| `--contra_lambda` | 0.05 | Contrastive semantic alignment loss weight |
| `--adv_lambda` | 0.01 | Adversarial domain invariance loss weight |
| `--ent_lambda` | 0.001 | Entropy regularization weight (prevents variance collapse) |

### Training/Inference Strategy (D²-FL only)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--proto_momentum` | 0.9 | Global prototype EMA momentum |
| `--ld_warmup` | 50 | Proto loss weight warmup rounds |
| `--temperature` | 1.0 | Proto inference temperature |
| `--use_per_class_temp` | True | Per-class learnable temperature |

### Algorithm-Specific
| Parameter | Default | Applies To | Description |
|-----------|---------|------------|-------------|
| `--gmm_components` | 3 | FedGMKD | GMM components per class |
| `--mi_lambda` | 0.05 | FedSeProto | MI minimization loss weight |
| `--pretrained` | True | All | Use ImageNet pretrained ResNet-50 |
| `--stopping_rounds` | 10 | All | Early stopping patience |
| `--test_ep` | 10 | All | Test evaluation interval |

## Key Features

### 5 Baselines + D²-FL Proposed Method

All algorithms use **ImageNet pretrained ResNet-50** backbone.

| Algorithm | Type | Server Aggregation | Local Objective |
|-----------|------|-------------------|-----------------|
| **FedAvg** | Weight-sharing | Weight averaging | `L_BCE` |
| **FedProto** | Prototype-sharing (baseline) | Prototype averaging | `L_BCE + lambda * MSE(proto, global_proto)` |
| **FedGMKD** | Prototype-sharing (baseline) | Discrepancy-aware GMM fusion | `L_BCE + lambda * GMM_proto_loss` |
| **FedBCS** | Prototype-sharing (baseline) | Prototype averaging | `L_BCE + freq-domain style recalibration` |
| **FedSeProto** | Prototype-sharing (baseline) | Semantic-only proto averaging | `L_BCE + lambda * MSE + HSIC MI min` |
| **D²-FL** | Prototype-sharing (proposed) | Bayesian fusion | `L_BCE + lambda * KL/Wasserstein + L_dis + L_cal + L_contra + L_adv + L_ent` |

### D²-FL vs FedProto

| Feature | FedProto (baseline) | D²-FL (proposed) |
|---------|-------------------|-------------------|
| Backbone | Pretrained ResNet-50 | Pretrained ResNet-50 |
| Prototype type | Point vector | Gaussian `N(mu, sigma^2)` |
| Aggregation | Simple averaging | Precision-weighted Bayesian fusion |
| Uncertainty | Not modeled | Per-client variance |
| Disentanglement | None | Learnable soft gate: semantic vs. style |
| HSIC Independence | None | Cross-covariance Frobenius norm |
| Adversarial Domain Inv. | None | Gradient reversal on semantic features |
| Contrastive Alignment | None | InfoNCE with Jaccard similarity |
| Calibration | None | Huber loss on logvar vs. distance |
| Entropy Regularization | None | `-logvar.mean()` prevents variance collapse |
| Privacy | N/A | Distributional encoding + semantic disentanglement (implicit) |
| Proto Momentum | None | EMA `G_t = beta*G_{t-1} + (1-beta)*G_new` |
| Lambda Schedule | Constant | Warmup: `ld * min(1, round/warmup)` |
| Inference Temp | 1.0 (none) | Per-class learnable or global configurable |
| Distance | MSE | KL / Wasserstein / MSE |

### D²-FL Complete Loss Function

The full D²-FL local training objective (7 loss terms):

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{BCE}}}_{\text{classification}} + \lambda_{\text{eff}} \cdot \underbrace{\mathcal{L}_{\text{proto}}}_{\text{prototype align}} + \lambda_{\text{dis}} \cdot \underbrace{\mathcal{L}_{\text{dis}}}_{\text{disentangle}} + \lambda_{\text{cal}} \cdot \underbrace{\mathcal{L}_{\text{cal}}}_{\text{calibration}} + \lambda_{\text{contra}} \cdot \underbrace{\mathcal{L}_{\text{contra}}}_{\text{contrastive}} + \lambda_{\text{adv}} \cdot \underbrace{\mathcal{L}_{\text{adv}}}_{\text{adversarial}} + \lambda_{\text{ent}} \cdot \underbrace{\mathcal{L}_{\text{ent}}}_{\text{entropy reg}}$$

Where:
- **L_BCE**: Multi-label binary cross-entropy
- **L_proto**: Distributional prototype distance (KL / Wasserstein / MSE) between local and global prototypes
- **L_dis**: Disentanglement = HSIC independence + gate entropy + orthogonal constraint
- **L_cal**: Calibration = Huber(logvar_mean, log(||mu - global_mu||^2))
- **L_contra**: Contrastive = InfoNCE with Jaccard similarity positive pairs
- **L_adv**: Adversarial = BCE on gradient-reversed semantic features (target=uniform)
- **L_ent**: Entropy regularization = -logvar.mean() (prevents variance → 0)

### Prototype Learning
Instead of sharing model weights, FedProto and D²-FL share **class prototypes** — feature vectors from the penultimate layer. This enables:
- Communication efficiency (256-dim vectors vs millions of weights)
- Model-heterogeneous FL support

### Multi-label Evaluation
Two test modes for prototype-based methods:
1. **Local model**: `sigmoid(logits) > 0.5` (only works for client's own classes)
2. **Global prototypes**: distance to each global prototype as pseudo-logit (zero-shot for unseen classes)

For weight-sharing methods, evaluation uses per-client local model sigmoid threshold classification.

## Privacy

D²-FL achieves **implicit privacy protection** without explicit differential privacy mechanisms. Three design choices work together:

### 1. Distributional Prototypes → Uncertainty as Privacy

By modeling each class prototype as a Gaussian distribution `N(mu, sigma^2)` rather than a point vector:

- **Per-client variance** encodes how much the client's data deviates from the consensus — a client with outlier data will naturally have higher variance, revealing less about individual samples
- **Bayesian fusion** aggregates prototypes by *precision* (1/sigma^2) — low-variance clients contribute more, high-variance clients contribute less. This means clients that "stand out" are automatically de-weighted
- **Entropy regularization** prevents variance collapse (`logvar → -∞`), ensuring the distributional representation is genuinely used instead of degenerating back to point prototypes

*Key insight*: adding Gaussian noise (as in formal DP) is mathematically equivalent to increasing the variance of a Gaussian distribution. D²-FL already learns per-client variance — a natural mechanism for bounding information leakage.

### 2. Semantic Disentanglement → Only Disease Features Leave the Client

The disentanglement module decomposes features into:

| Subspace | Contains | Shared? |
|----------|----------|---------|
| **Semantic** `z_sem` | Disease-relevant patterns (e.g., opacity, nodule shape) | Yes — uploaded as prototype |
| **Style** `z_style` | Hospital-specific artifacts (scanner type, lighting, patient demographics) | No — stays local |

This means:
- The server **never sees** client-specific style information
- Prototypes encode only disease semantics — useless for reconstructing individual images
- The four enforcement mechanisms (HSIC independence, gate entropy, adversarial invariance, contrastive alignment) ensure this separation is genuine, not superficial

### 3. Prototype-Level Sharing → Minimal Information Transfer

Unlike FedAvg which transmits full model weights (millions of parameters), D²-FL shares only:

- Per-class prototypes: typically 256-dimensional vectors × number of classes
- When disentangled: only the semantic subspace (e.g., 192 dims) is shared — style features stay local

This provides **orders of magnitude less** information leakage than weight-sharing FL. Combined with distributional encoding and disentanglement, D²-FL offers strong *de facto* privacy without the **utility penalty** of formal DP noise injection.

### Comparison with Explicit DP

| Aspect | Formal DP (e.g., Gaussian mechanism) | D²-FL Implicit Privacy |
|--------|--------------------------------------|------------------------|
| Mechanism | Clipping + noise injection | Variance encoding + disentanglement |
| Guarantee | (epsilon, delta)-DP (provable) | De facto privacy (empirical) |
| Utility impact | Significant accuracy drop at strong epsilon | No accuracy penalty |
| Information leaked | Bounded by epsilon | Bounded by prototype dimensionality + variance |
| Client-specific protection | Uniform noise across all clients | Adaptive: high-variance clients reveal less |

For deployment scenarios requiring **formal** privacy guarantees, D²-FL's architecture is compatible with additional DP layers (the prototype structure is naturally amenable to Gaussian mechanism perturbation with lower noise due to reduced dimensionality in the semantic subspace). However, for most medical FL scenarios, the implicit privacy of distributional prototypes + disentanglement provides strong practical protection.

## Citation

**FedProto:**
```
@inproceedings{tan2021fedproto,
  title={FedProto: Federated Prototype Learning across Heterogeneous Clients},
  author={Tan, Yue and Long, Guodong and Liu, Lu and Zhou, Tianyi and Lu, Qinghua and Jiang, Jing and Zhang, Chengqi},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2022}
}
```

**FedAvg:**
```
@inproceedings{mcmahan2017communication,
  title={Communication-Efficient Learning of Deep Networks from Decentralized Data},
  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and y Arcas, Blaise Aguera},
  booktitle={AISTATS},
  year={2017}
}
```

