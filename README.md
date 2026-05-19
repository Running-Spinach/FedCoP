# DPP-FL: Distributional Pathology Prototype Federated Learning

Privacy-Preserving Medical Image Classification via Federated Prototype Learning with Distributional Prototypes and Differential Privacy.

Based on [FedProto (AAAI 2022)](https://arxiv.org/abs/2105.00243), this project extends the framework with:

- **Distributional Prototypes** — each class prototype is modeled as a Gaussian distribution `N(mu, sigma^2)` instead of a point vector, capturing per-client uncertainty
- **Differential Privacy** — Gaussian noise perturbation + Moments Accountant for formal (epsilon, delta)-DP guarantees on uploaded prototypes
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
│   └── federated_main.py         # Main entry point
├── lib/
│   ├── options.py                # Argument parsing
│   ├── utils.py                  # Data loading, prototype aggregation
│   ├── update.py                 # Local training, test inference
│   ├── sampling.py               # IID/Non-IID data partitioning
│   ├── chestxray.py              # ChestXray14 dataset class
│   ├── visualize.py              # t-SNE prototype visualization
│   ├── models/
│   │   ├── models.py             # CNNMnist (for MNIST)
│   │   └── resnet.py             # ResNet50 backbone
│   ├── dist_proto/
│   │   ├── proto_head.py         # ProbabilisticProtoHead (mu, logvar)
│   │   ├── losses.py             # KL / Wasserstein / MSE distances
│   │   └── aggregation.py        # Bayesian precision-weighted fusion
│   └── dp/
│       └── mechanisms.py         # DPMechProto, MomentsAccountant
├── scripts/
│   └── run.sh                    # Example launch script
├── requirements.txt
├── README.md
└── THEORY.md                     # Detailed theory & algorithm docs (Chinese)
```

## Running

### Basic FedProto (point prototypes)
```bash
python exps/federated_main.py \
    --ways 5 --shots 100 --num_users 20 --rounds 200 --ld 1.0
```

### Distributional Prototypes (Gaussian)
```bash
python exps/federated_main.py \
    --use_distributional --dist_type kl \
    --ways 5 --rounds 200 --ld 1.0
```

### Distributional + Differential Privacy
```bash
python exps/federated_main.py \
    --use_distributional --dist_type wasserstein \
    --use_dp --dp_epsilon 8.0 --dp_clip 1.0 \
    --rounds 30
```

### Quick test (MNIST, IID)
```bash
python exps/federated_main.py \
    --model cnn --num_classes 10 --iid 1 \
    --rounds 50 --num_users 10
```

## Options

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
| `--model` | resnet50 | Backbone: resnet50 / cnn |
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

### Distributional Prototypes
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_distributional` | False | Enable Gaussian prototypes |
| `--dist_type` | kl | Distance type: kl / wasserstein / mse |
| `--ld` | 1.0 | Prototype loss weight lambda |

### Differential Privacy
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_dp` | False | Enable DP on uploaded prototypes |
| `--dp_epsilon` | 8.0 | Target epsilon |
| `--dp_delta` | 1e-5 | Target delta |
| `--dp_clip` | 1.0 | L2 clipping norm for prototypes |

## Key Features

### Prototype Learning
Instead of sharing model weights (FedAvg), clients share **class prototypes** — feature vectors from the penultimate layer. This enables:
- Communication efficiency (256-dim vectors vs millions of weights)
- Model-heterogeneous FL support
- Privacy preservation (raw data never leaves the client)

### Distributional Prototypes
When `--use_distributional` is enabled, prototypes become Gaussian distributions:
- **Mean (mu)**: the expected feature representation
- **Variance (sigma^2)**: uncertainty estimate (higher for clients with less data)
- **Bayesian Fusion**: precision-weighted aggregation on the server — clients with lower variance get higher weight

### Differential Privacy
Client-side Gaussian noise perturbation with Moments Accountant:
- L2-clip prototype vectors to bound sensitivity
- Add calibrated Gaussian noise before upload
- Track (epsilon, delta)-DP budget across rounds via Rényi DP

### Multi-label Evaluation
Two test modes per client:
1. **Local model**: `sigmoid(logits) > 0.5` (only works for client's own classes)
2. **Global prototypes**: distance to each global prototype as pseudo-logit (zero-shot for unseen classes)

## Citation

```
@inproceedings{tan2021fedproto,
  title={FedProto: Federated Prototype Learning across Heterogeneous Clients},
  author={Tan, Yue and Long, Guodong and Liu, Lu and Zhou, Tianyi and Lu, Qinghua and Jiang, Jing and Zhang, Chengqi},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
