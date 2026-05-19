# FedProto: Federated Prototype Learning across Heterogeneous Clients

Implementation of the paper accepted by AAAI 2022: [FedProto: Federated Prototype Learning across Heterogeneous Clients](https://arxiv.org/abs/2105.00243).

This fork uses **ResNet-50** backbone with **ChestX-ray14** multi-label dataset (14 disease classification).

## Requirements
* Python 3.6+
* PyTorch 1.6+
* Torchvision
* Numpy
* Pandas
* PIL (Pillow)

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

## Running

```bash
python exps/federated_main.py --num_classes 14 --num_users 20 --ways 5 --shots 100 --stdev 2 --rounds 100 --ld 1
```

## Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rounds` | 100 | Global communication rounds |
| `--num_users` | 20 | Number of clients |
| `--train_ep` | 1 | Local epochs per round |
| `--local_bs` | 4 | Local batch size |
| `--lr` | 0.01 | Learning rate |
| `--momentum` | 0.5 | SGD momentum |
| `--num_classes` | 14 | Number of classes (ChestX-ray14 diseases) |
| `--ways` | 3 | Avg classes per client |
| `--shots` | 100 | Avg samples per class |
| `--stdev` | 2 | Std of ways/shots |
| `--ld` | 1.0 | Prototype loss weight |
| `--image_size` | 224 | Input image size |
| `--seed` | 1234 | Random seed |
| `--model` | resnet50 | Model backbone: resnet50 / cnn |
| `--proto_dim` | 256 | Prototype vector dimension |
| `--use_distributional` | False | Enable Gaussian prototypes |
| `--dist_type` | kl | kl / wasserstein / mse |
| `--use_dp` | False | Enable differential privacy |
| `--dp_epsilon` | 8.0 | Target epsilon |
| `--dp_clip` | 1.0 | L2 clip norm |

## Citation
```
@inproceedings{tan2021fedproto,
  title={FedProto: Federated Prototype Learning across Heterogeneous Clients},
  author={Tan, Yue and Long, Guodong and Liu, Lu and Zhou, Tianyi and Lu, Qinghua and Jiang, Jing and Zhang, Chengqi},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
