# FedCoP: Federated Co-occurrence-aware Prototypes for Multi-Label Medical Image Classification

Prototype-based federated learning on **NIH ChestX-ray14** (14 thoracic pathologies, chest radiography) and **MuReD** (20 retinal pathologies, fundus), where **pathology co-occurrence (comorbidity) is recovered federatedly** and used to constrain prototype geometry (training) and propagate evidence across co-occurring classes (inference). FedCoP breaks the per-class-independence assumption shared by all prior prototype-FL methods.

All algorithms use an **ImageNet-pretrained ResNet-50** backbone for fair comparison. Built on [FedProto (AAAI 2022)](https://arxiv.org/abs/2105.00243), this project provides **6 FL baselines** + the proposed **FedCoP** method + 3 ablations.

## Why FedCoP

Existing prototype-FL methods (FedProto, FedGMKD, FedSeProto, …) model each of the 14 pathologies with an **independent** prototype and decode labels with **independent per-class sigmoids** — assuming the labels are conditionally independent given features. This fails in ChestX-ray14, where diseases strongly co-occur. Worse, under non-IID class partitioning (each hospital sees only ~3/14 classes), **no single client can observe the global co-occurrence structure** — it is recoverable only by federated aggregation of label statistics. FedCoP:

1. Estimates a global co-occurrence correlation matrix `R̂` from privacy-safe label sufficient statistics `(m_k, M_k, n_k)` aggregated by count-weighted fusion.
2. **Training-side:** a structure loss `L_co` aligns the prototype cosine-Gram to `R̂` (co-occurring diseases → nearby prototype directions).
3. **Inference-side:** a correlation-aware **mean-field decoder** propagates evidence across co-occurring classes (strict improvement over independent sigmoids under correlated labels).

## Algorithms

| Algorithm | Shared Info | Key Mechanism | Paper |
|-----------|------------|---------------|-------|
| **FedAvg** | Model weights | Weight averaging | AISTATS 2017 |
| **FedProx** | Model weights | Weight averaging + proximal term | MLSys 2020 |
| **FedProto** | Point prototypes | Prototype regularization + nearest-neighbor | AAAI 2022 |
| **FedGMKD** | GMM prototypes | EM-fitted GMM + discrepancy-aware aggregation | NeurIPS 2024 |
| **FedSeProto** | Hard-split semantic/domain protos | Two-branch MLP + HSIC | ECAI 2024 |
| **FedCoP** ★ | Gaussian prototypes + co-occurrence `R̂` | Federated co-occurrence structure + mean-field decode | **Proposed** |

### FedCoP ablations (isolate the contribution)

| Variant | Flag | Isolates |
|---|---|---|
| `fedcop_nocoo` | `--no_cooccurrence` | total co-occurrence contribution (`R̂=I`) |
| `fedcop_local` | `--local_cooc_only` | necessity of federated `R̂` aggregation |
| `fedcop_nolco` | `--no_lco` | training-side `L_co` vs inference-side decoder |

## Requirements

- Python 3.8+, PyTorch 2.0+, Torchvision 0.15+
- NumPy, Pandas, Pillow, tqdm, scikit-learn (metrics)

See [requirements.txt](requirements.txt). Tested on a single NVIDIA 4090 (Linux).

## Data Preparation

### ChestX-ray14 (default, 14 classes)

Download the [NIH ChestX-ray14 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) and extract to:

```
./data/chestxray/
  images/
    00000001_000.png
    ...
  Data_Entry_2017.csv
```

### MuReD (20 classes, fundus — cross-modality generalization)

Place the MuReD (Multi-label Retinal Diseases) dataset under:

```
./data/Multi-Label Retinal Diseases (MuReD) Dataset/
  images/images/            # double-layer dir; png + tif mixed
    <ID>.png | <ID>.tif
  train_data.csv            # ID,DR,NORMAL,MH,...,OTHER  (20 label columns)
  val_data.csv
```

The loader maps each CSV `ID` to `<ID>.png` or `<ID>.tif` automatically. Images are converted to RGB (color fundus, unlike the grayscale chest films).

`--data_dir` controls the root (default `./data/`). Select the dataset with `--dataset chestxray14|mured`, or via the `DATASET` env var in the run scripts.

## Project Structure

```
FedCoP/
├── code/
│   ├── exps/
│   │   └── federated_main.py      # Main entry point (7 algorithms + dispatch)
│   ├── lib/
│   │   ├── options.py             # Argument parsing
│   │   ├── utils.py               # Data loading, prototype aggregation, exp_details
│   │   ├── update.py              # Local training, test inference (all algorithms)
│   │   ├── metrics.py             # Multi-label metrics (AUROC/F1/Hamming/subset)
│   │   ├── sampling.py            # IID/Non-IID data partitioning
│   │   ├── chestxray.py           # ChestXray14 dataset class
│   │   ├── mured.py               # MuReD dataset class (20-class fundus)
│   │   ├── visualize.py           # t-SNE prototype visualization
│   │   ├── models/
│   │   │   └── resnet.py          # FedCoPResNet / ResNet50 backbone
│   │   └── dist_proto/
│   │       ├── proto_head.py      # ProbabilisticProtoHead (Gaussian μ/logvar)
│   │       ├── losses.py          # KL / Wasserstein / MSE / Entropy reg
│   │       ├── aggregation.py     # Bayesian precision-weighted fusion
│   │       └── structured.py      # ★ Co-occurrence stat / fusion / L_co / mean-field decode
│   └── scripts/
│       ├── run.sh                 # Full benchmark (3 seeds, mean±std)
│       └── run_test.sh            # Smoke test (5 rounds, single seed)
├── data/                          # Datasets (gitignored)
├── logs/                          # Run logs (gitignored)
├── protos_vis/                    # t-SNE prototype npy/pdf (gitignored)
├── paper/                         # Paper & theory documentation
├── requirements.txt
└── README.md
```

## Running

Select the algorithm via `--alg` (default: `fedcop`). All algorithms use a pretrained ResNet-50 backbone.

### Quick smoke test
```bash
bash code/scripts/run_test.sh           # all algorithms, 5 rounds, single seed
bash code/scripts/run_test.sh fedcop    # only FedCoP
```

### Full benchmark (3 seeds, mean±std summary)
```bash
bash code/scripts/run.sh                # all algorithms + ablations
bash code/scripts/run.sh fedcop         # only FedCoP across 3 seeds
```

### Dataset selection (ChestX-ray14 default; MuReD for cross-modality)
```bash
DATASET=mured bash code/scripts/run_test.sh fedcop    # smoke test on MuReD
DATASET=mured bash code/scripts/run.sh fedcop         # full benchmark on MuReD
```
`DATASET` auto-sets `--dataset` and `--num_classes` (14 for chestxray14, 20 for mured).

### Individual algorithms
```bash
# Baselines
python code/exps/federated_main.py --alg fedavg
python code/exps/federated_main.py --alg fedprox --fedprox_mu 0.01
python code/exps/federated_main.py --alg fedproto
python code/exps/federated_main.py --alg fedgmkd --gmm_components 3
python code/exps/federated_main.py --alg fedseproto --mi_lambda 0.05

# FedCoP (proposed)
python code/exps/federated_main.py --alg fedcop \
    --co_lambda 0.1 --cov_shrinkage 0.1 --co_beta 1.0 --co_mf_steps 2 \
    --ent_lambda 1e-3 --proto_momentum 0.9 --temperature 1.0 --ld_warmup 20

# FedCoP ablations
python code/exps/federated_main.py --alg fedcop --no_cooccurrence    # R̂ = I
python code/exps/federated_main.py --alg fedcop --local_cooc_only    # per-client local R̂
python code/exps/federated_main.py --alg fedcop --no_lco             # training-side L_co off
```

## Options

### Algorithm & Federated
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alg` | fedcop | fedavg / fedprox / fedproto / fedgmkd / fedseproto / fedcop |
| `--rounds` | 100 | Global communication rounds |
| `--num_users` | 20 | Number of clients |
| `--frac` | 0.25 | Fraction of clients per round |
| `--train_ep` | 1 | Local epochs per round |
| `--local_bs` | 4 | Local batch size |
| `--lr` | 0.01 | Learning rate |
| `--seed` | 1234 | Random seed |

### Model & Data
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | chestxray14 | chestxray14 / mured |
| `--num_classes` | 14 | 14 (chestxray14) / 20 (mured) |
| `--proto_dim` | None→256 | Prototype dimension |
| `--image_size` | 224 | Input image size |
| `--pretrained` | True | ImageNet pretrained ResNet-50 |
| `--ways` | 3 | Avg classes per client (Non-IID) |
| `--shots` | 100 | Avg samples per class per client |
| `--stdev` | 2 | Std of ways/shots across clients |

### Distributional Prototypes
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_distributional` | False | Gaussian prototypes (FedCoP forces True) |
| `--dist_type` | kl | kl / wasserstein / mse |
| `--ld` | 1.0 | Prototype loss weight λ |
| `--ld_warmup` | 50 | Proto loss warmup rounds |
| `--proto_momentum` | 0.9 | EMA momentum (prototypes + R̂) |
| `--temperature` | 1.0 | Inference temperature T |
| `--ent_lambda` | 1e-3 | Entropy reg weight (anti-collapse) |

### FedCoP Co-occurrence (core)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--co_lambda` | 0.1 | Structure loss L_co weight |
| `--cov_shrinkage` | 0.1 | R̂ shrinkage η toward identity |
| `--co_rank` | 0 | Low-rank approx rank (0 = full) |
| `--co_beta` | 1.0 | Mean-field coupling strength β |
| `--co_mf_steps` | 2 | Mean-field iterations |
| `--no_cooccurrence` | False | Ablation: R̂=I, structure off |
| `--local_cooc_only` | False | Ablation: per-client local R̂ (no federation) |
| `--no_lco` | False | Ablation: training-side L_co off |

### Baseline-specific
| Parameter | Default | Applies To | Description |
|-----------|---------|------------|-------------|
| `--fedprox_mu` | 0.01 | FedProx | Proximal term coefficient |
| `--gmm_components` | 3 | FedGMKD | GMM components per class |
| `--mi_lambda` | 0.05 | FedSeProto | MI minimization weight |

## Method summary

FedCoP's local objective is intentionally minimal (4 losses):

$$\mathcal{L} = L_{CE} + \lambda_{\text{eff}}\,L_{proto} + \lambda_{co}\,L_{co} + \lambda_{ent}\,L_{ent}$$

- **L_CE**: multi-label BCE classification.
- **L_proto**: KL between local and global diagonal-Gaussian prototypes (per positive label).
- **L_co**: co-occurrence structure alignment — prototype cosine-Gram ↔ federated `R̂`.
- **L_ent**: anti-collapse entropy regularization.

The co-occurrence matrix `R̂` is estimated from label counts `(m_k, M_k, n_k)` (privacy-safe, no features), aggregated by count-weighted phi-correlation + shrinkage + EMA. At inference, a mean-field decoder uses `R̂` to propagate evidence across co-occurring classes.

**Omitted regularizers** (analyzed as redundant/dead): per-class temperature (dead code), adversarial domain loss (redundant), InfoNCE contrastive (overlaps L_proto+L_CE), calibration loss (collapsed logvar to scalar), and semantic-style disentanglement (orthogonal story line). See `paper/` for the full analysis and theory (decoder-regret bound + federated-estimation theorem).

## Evaluation metrics

Beyond the legacy per-label accuracy, FedCoP reports full multi-label metrics (macro/micro AUROC, macro/micro F1, Hamming loss, subset accuracy, per-class AUROC) via `lib/metrics.py`, computed on the correlation-aware prototype-decode probabilities.

## Citation

```bibtex
@inproceedings{tan2022fedproto,
  title={FedProto: Federated Prototype Learning across Heterogeneous Clients},
  author={Tan, Yue and Long, Guodong and Liu, Lu and Zhou, Tianyi and Lu, Qinghua and Jiang, Jing and Zhang, Chengqi},
  booktitle={AAAI}, year={2022}
}
@inproceedings{li2020fedprox,
  title={Federated Optimization in Heterogeneous Networks},
  author={Li, Tian and Sahu, Anit Kumar and Zaheer, Manzil and Sanjabi, Maziar and Talwalkar, Ameet and Smith, Virginia},
  booktitle={MLSys}, year={2020}
}
```
