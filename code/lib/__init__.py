# =============================================================================
# FedCoP Core Library
# =============================================================================
# 联邦学习核心模块:数据加载、模型定义、训练更新、原型聚合、分布原型、共现结构
#
# 子包:
#   dist_proto/  — 分布原型(Gaussian head / Bayesian fusion / 结构化共现)
#   models/      — 骨干网络(ResNet50 / FedCoPResNet)
#
# 顶层模块:
#   options.py   — 命令行参数解析
#   utils.py     — 数据集加载、原型聚合、实验信息打印
#   update.py    — 本地训练循环、测试推理(各算法的客户端 + 评估逻辑)
#   metrics.py   — 多标签评估指标(AUROC/F1/Hamming/subset)
#   sampling.py  — IID / Non-IID 客户端数据划分
#   chestxray.py — ChestX-ray14 多标签数据集类
#   visualize.py — t-SNE 原型可视化
# =============================================================================

# ── 顶层 API(federated_main.py 直接使用的入口)──
from .options import args_parser
from .utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func
from .update import (
    LocalUpdate,
    DatasetSplit,
    test_inference_new_het_lt,
    test_inference_FedCoP,
    eval_clients_multilabel,
    # FedGMKD 组件
    _update_weights_FedGMKD,
    _agg_func_FedGMKD,
    _proto_aggregation_FedGMKD,
    # FedSeProto 组件
    _update_weights_FedSeProto,
)
from .sampling import chestxray_noniid, chestxray_noniid_lt, chestxray_iid
from .chestxray import ChestXray14
from .metrics import compute_multilabel_metrics, format_metrics

# ── 子包 ──
from . import dist_proto
from . import models

__all__ = [
    # options
    'args_parser',
    # utils
    'get_dataset', 'average_weights', 'exp_details', 'proto_aggregation', 'agg_func',
    # update
    'LocalUpdate', 'DatasetSplit',
    'test_inference_new_het_lt', 'test_inference_FedCoP',
    'eval_clients_multilabel',
    '_update_weights_FedGMKD', '_agg_func_FedGMKD', '_proto_aggregation_FedGMKD',
    '_update_weights_FedSeProto',
    # metrics
    'compute_multilabel_metrics', 'format_metrics',
    # sampling
    'chestxray_noniid', 'chestxray_noniid_lt', 'chestxray_iid',
    # chestxray
    'ChestXray14',
    # subpackages
    'dist_proto',
    'models',
]
