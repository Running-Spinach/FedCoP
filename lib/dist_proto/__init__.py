# 功能：分布原型子模块，提供概率原型头、分布损失函数、
#       贝叶斯融合聚合和增强原型解耦

from .proto_head import ProbabilisticProtoHead, PerClassTemperature
from .losses import (
    kl_divergence_gaussian,
    wasserstein2_gaussian,
    distributional_proto_loss,
    prototype_calibration_loss,
    entropy_regularization,
)
from .aggregation import bayesian_fusion, bayesian_fusion_single_label
from .disentangle import (
    DisentangledProtoHead,
    disentanglement_loss,
    contrastive_semantic_loss,
    adversarial_disentanglement_loss,
    GradientReversal,
    grad_reverse,
    LearnableGate,
)
