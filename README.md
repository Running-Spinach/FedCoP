# DPP-FL: Distributional Dual-Stream Federated Pathology Representation Learning

Privacy-Preserving Medical Image Classification via Federated Prototype Learning with Distributional Prototypes and Differential Privacy.

All algorithms use **ImageNet pretrained ResNet-50** backbone + **Differential Privacy** for fair comparison. Based on [FedProto (AAAI 2022)](https://arxiv.org/abs/2105.00243), this project provides **5 FL baselines** + the proposed **DPP-FL** method:

### Baselines (comparison algorithms)

| Algorithm | Shared Info | Key Mechanism | Paper |
|-----------|------------|---------------|-------|
| **FedAvg** | Model weights | Simple weight averaging + DP | AISTATS 2017 |
| **FedProx** | Model weights | Proximal term `mu/2 * ||w - w_t||^2` + DP | MLSys 2020 |
| **FedBN** | Model weights (no BN) | Local BN stats, global conv/linear + DP | ICLR 2021 |
| **SCAFFOLD** | Model weights + control variates | Gradient correction `g - c_i + c` + DP | ICML 2020 |
| **FedProto** | Class prototypes (point) | Prototype regularization + nearest-neighbor + DP | AAAI 2022 |

### Proposed Method: DPP-FL

| Algorithm | Shared Info | Key Innovation |
|-----------|------------|----------------|
| **DPP-FL** | Gaussian prototypes `N(mu, sigma^2)` | Distributional prototypes + Bayesian fusion + Proto EMA + Temperature scaling |

Key features of DPP-FL (all algorithms share pretrained backbone + DP for fairness):

- **Distributional Prototypes** — each class prototype is modeled as a Gaussian distribution `N(mu, sigma^2)`, capturing per-client uncertainty
- **Bayesian Fusion** — precision-weighted aggregation: clients with lower variance get higher weight
- **Prototype EMA Momentum** — exponential moving average of global prototypes for stability
- **Adaptive Lambda Warmup** — gradual increase of prototype loss weight to avoid early bias
- **Temperature-Scaled Inference** — sharpened prototype distances for better class separation
- **Differential Privacy** — Gaussian noise perturbation + Moments Accountant for formal (epsilon, delta)-DP guarantees (applied to weights for weight-sharing methods, prototypes for prototype-sharing methods)
- **Multi-label Classification** — adapted for Chest X-ray diagnosis (14 disease labels per image)
- **Task Heterogeneity** — each client sees only a subset of disease classes (Non-IID label skew)

## Requirements

- Python 3.7+
- PyTorch 1.7+
- Torchvision 0.8+
- NumPy, Pandas, Pillow, tqdm
- scikit-learn, matplotlib (for visualization)
- tensorboardX (for logging)

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
DPP-FL/
├── exps/
│   └── federated_main.py          # Main entry point
├── lib/
│   ├── options.py                 # Argument parsing
│   ├── utils.py                   # Data loading, prototype aggregation
│   ├── update.py                  # Local training, test inference
│   ├── sampling.py                # IID/Non-IID data partitioning
│   ├── chestxray.py               # ChestXray14 dataset class
│   ├── visualize.py               # t-SNE prototype visualization
│   ├── models/
│   │   └── resnet.py              # DPPFLResNet / ResNet50 backbone
│   ├── dist_proto/
│   │   ├── proto_head.py          # ProbabilisticProtoHead (mu, logvar)
│   │   ├── losses.py              # KL / Wasserstein / MSE distances
│   │   ├── aggregation.py         # Bayesian precision-weighted fusion
│   │   └── disentangle.py         # DisentangledProtoHead + HSIC loss
│   └── dp/
│       ├── mechanisms.py          # DPMechProto, MomentsAccountant
│       └── __init__.py            # DP module exports
├── figures/                       # Architecture diagrams
├── paper/                         # Paper & theory documentation
├── scripts/
│   └── run.sh                     # Launch script (6 algorithms)
├── requirements.txt
└── README.md
```

## Running

Select the algorithm via `--alg` (default: `dppfl`). All algorithms use pretrained ResNet-50 backbone. Add `--use_dp` to enable differential privacy.

### Baselines
```bash
# FedProto (original, point prototypes only)
python exps/federated_main.py --alg fedproto \
    --ways 5 --shots 100 --num_users 20 --rounds 200 --ld 1.0

# FedProto + DP
python exps/federated_main.py --alg fedproto \
    --use_dp --dp_epsilon 8.0 --dp_clip 1.0 \
    --ways 5 --num_users 20 --rounds 30

# FedAvg
python exps/federated_main.py --alg fedavg \
    --ways 5 --shots 100 --num_users 20 --rounds 200 --frac 1.0

# FedAvg + DP
python exps/federated_main.py --alg fedavg \
    --use_dp --dp_epsilon 8.0 --dp_clip 1.0 \
    --ways 5 --num_users 20 --rounds 30

# FedProx
python exps/federated_main.py --alg fedprox \
    --fedprox_mu 0.01 --ways 5 --num_users 20 --rounds 200

# FedBN
python exps/federated_main.py --alg fedbn \
    --ways 5 --shots 100 --num_users 20 --rounds 200 --frac 1.0

# SCAFFOLD
python exps/federated_main.py --alg scaffold \
    --ways 5 --shots 100 --num_users 20 --rounds 200
```

### Proposed: DPP-FL
```bash
# 点原型模式
python exps/federated_main.py --alg dppfl \
    --ways 5 --shots 100 --num_users 20 --rounds 200 --ld 1.0

# 分布原型 (Gaussian)
python exps/federated_main.py --alg dppfl \
    --use_distributional --dist_type kl \
    --ways 5 --rounds 200 --ld 1.0

# 分布原型 + DP
python exps/federated_main.py --alg dppfl \
    --use_distributional --dist_type wasserstein \
    --use_dp --dp_epsilon 8.0 --dp_clip 1.0 \
    --rounds 30

# 完整 DPP-FL (分布原型 + DP + 温度缩放 + 动量)
python exps/federated_main.py --alg dppfl \
    --use_distributional --dist_type kl \
    --use_dp --dp_epsilon 8.0 --dp_clip 1.0 \
    --proto_momentum 0.9 --temperature 0.5 --ld_warmup 50 \
    --ways 5 --num_users 20 --rounds 100
```

## Options

### Algorithm Selection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alg` | dppfl | FL algorithm: fedavg / fedprox / fedbn / scaffold / fedproto / dppfl |

### Federated Learning
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rounds` | 100 | Global communication rounds |
| `--num_users` | 20 | Number of clients |
| `--frac` | 0.04 | Fraction of clients participating per round |
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

### Task Heterogeneity
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ways` | 3 | Avg classes per client |
| `--shots` | 100 | Avg samples per class per client |
| `--stdev` | 2 | Std deviation of ways/shots |
| `--iid` | 0 | Use IID split (0 = Non-IID) |
| `--unequal` | 0 | Unequal data amounts across clients |

### Prototype Learning (FedProto / DPP-FL shared)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ld` | 1.0 | Prototype loss weight lambda |
| `--proto_dim` | 256 | Prototype vector dimension |

### Distributional Prototypes (DPP-FL only)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_distributional` | False | Enable Gaussian prototypes `N(mu, sigma^2)` |
| `--dist_type` | kl | Distance type: kl / wasserstein / mse |

### Differential Privacy (all algorithms)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_dp` | False | Enable DP on uploaded weights/prototypes |
| `--dp_epsilon` | 8.0 | Target epsilon |
| `--dp_delta` | 1e-5 | Target delta |
| `--dp_clip` | 1.0 | L2 clipping norm |

### Algorithm-Specific
| Parameter | Default | Applies To | Description |
|-----------|---------|------------|-------------|
| `--fedprox_mu` | 0.01 | FedProx | Proximal term coefficient |
| `--scaffold_lr` | None | SCAFFOLD | Global LR (defaults to `--lr`) |
| `--pretrained` | True | All | Use ImageNet pretrained ResNet-50 |
| `--proto_momentum` | 0.9 | DPP-FL | Global prototype EMA momentum |
| `--ld_warmup` | 50 | DPP-FL | Proto loss weight warmup rounds |
| `--temperature` | 1.0 | DPP-FL | Proto inference temperature |

## Key Features

### 5 Baselines + DPP-FL Proposed Method

All algorithms use **ImageNet pretrained ResNet-50** backbone + optional **DP** (weight-level for weight-sharing, prototype-level for prototype-sharing).

| Algorithm | Type | Server Aggregation | Local Objective |
|-----------|------|-------------------|-----------------|
| **FedAvg** | Weight-sharing | Weight averaging (+ DP) | `L_BCE` |
| **FedProx** | Weight-sharing | Weight averaging (+ DP) | `L_BCE + (mu/2) * ||w - w_global||^2` |
| **FedBN** | Weight-sharing | Weight avg (skip BN) (+ DP) | `L_BCE` |
| **SCAFFOLD** | Weight-sharing | Weight avg + control variate (+ DP) | `L_BCE` with grad correction |
| **FedProto** | Prototype-sharing (baseline) | Prototype averaging (+ DP on protos) | `L_BCE + lambda * MSE(proto, global_proto)` |
| **DPP-FL** | Prototype-sharing (proposed) | Bayesian fusion (+ DP on protos) | `L_BCE + lambda * KL/Wasserstein(N_loc, N_global)` |

### DPP-FL vs FedProto

| Feature | FedProto (baseline) | DPP-FL (proposed) |
|---------|-------------------|-------------------|
| Backbone | Pretrained ResNet-50 | Pretrained ResNet-50 |
| Prototype type | Point vector | Gaussian `N(mu, sigma^2)` |
| Aggregation | Simple averaging | Precision-weighted Bayesian fusion |
| Uncertainty | Not modeled | Per-client variance |
| Privacy | Optional (epsilon, delta)-DP | Optional (epsilon, delta)-DP |
| Proto Momentum | None | EMA `G_t = beta*G_{t-1} + (1-beta)*G_new` |
| Lambda Schedule | Constant | Warmup: `ld * min(1, round/warmup)` |
| Inference Temp | 1.0 (none) | Configurable: `-dist / T` |
| Distance | MSE | KL / Wasserstein / MSE |

### Prototype Learning
Instead of sharing model weights, FedProto and DPP-FL share **class prototypes** — feature vectors from the penultimate layer. This enables:
- Communication efficiency (256-dim vectors vs millions of weights)
- Model-heterogeneous FL support
- Privacy preservation (raw data never leaves the client)

### Multi-label Evaluation
Two test modes for FedProto/DPP-FL:
1. **Local model**: `sigmoid(logits) > 0.5` (only works for client's own classes)
2. **Global prototypes**: distance to each global prototype as pseudo-logit (zero-shot for unseen classes)

For FedAvg/FedProx/FedBN/SCAFFOLD, evaluation uses per-client local model sigmoid threshold classification.

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

**FedProx:**
```
@inproceedings{li2020federated,
  title={Federated Optimization in Heterogeneous Networks},
  author={Li, Tian and Sahu, Anit Kumar and Zaheer, Manzil and Sanjabi, Maziar and Talwalkar, Ameet and Smith, Virginia},
  booktitle={MLSys},
  year={2020}
}
```

**FedBN:**
```
@inproceedings{li2021fedbn,
  title={FedBN: Federated Learning on Non-IID Features via Local Batch Normalization},
  author={Li, Xiaoxiao and Jiang, Meirui and Zhang, Xiaofei and Kamp, Michael and Dou, Qi},
  booktitle={ICLR},
  year={2021}
}
```

**SCAFFOLD:**
```
@inproceedings{karimireddy2020scaffold,
  title={SCAFFOLD: Stochastic Controlled Averaging for Federated Learning},
  author={Karimireddy, Sai Praneeth and Kale, Satyen and Mohri, Mehryar and Reddi, Sashank and Stich, Sebastian and Suresh, Ananda Theertha},
  booktitle={ICML},
  year={2020}
}
```
