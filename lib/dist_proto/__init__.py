# 功能:分布原型子包 —— 概率原型头、分布损失、贝叶斯融合、结构化共现
#
# FedCoP 的分布原型 + 共现结构核心组件:
#   proto_head.py    — ProbabilisticProtoHead(特征 → 高斯 μ/logvar)
#   losses.py        — 分布原型对齐损失 L_proto、熵正则 L_ent
#   aggregation.py   — 逐类贝叶斯精度加权融合
#   structured.py    — 联邦共现结构估计、L_co 结构对齐、mean-field 解码(核心创新)

from .proto_head import ProbabilisticProtoHead
from .losses import (
    kl_divergence_gaussian,
    wasserstein2_gaussian,
    distributional_proto_loss,
    entropy_regularization,
)
from .aggregation import bayesian_fusion, bayesian_fusion_single_label
from .structured import (
    compute_local_cooc,
    fuse_cooccurrence,
    ema_correlation,
    cos_gram_structure_loss,
    mean_field_decode,
)
