# 功能：分布原型子模块，提供概率原型头、分布损失函数和贝叶斯融合聚合

from .proto_head import ProbabilisticProtoHead
from .losses import (
    kl_divergence_gaussian,
    wasserstein2_gaussian,
    distributional_proto_loss,
)
from .aggregation import bayesian_fusion, bayesian_fusion_single_label
