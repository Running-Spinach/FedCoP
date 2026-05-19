# 功能：概率原型头模块，将特征向量转换为高斯分布的均值和方差参数

import torch
import torch.nn as nn


class ProbabilisticProtoHead(nn.Module):
    """
    概率原型头，将特征向量转换为高斯分布 N(mu, sigma^2) 的参数

    参数:
        in_features: 输入特征向量的维度
        proto_dim: 原型分布的维度（默认与输入相同）
    """

    def __init__(self, in_features, proto_dim=None):
        super().__init__()
        if proto_dim is None:
            proto_dim = in_features
        self.mu_head = nn.Linear(in_features, proto_dim)
        self.logvar_head = nn.Linear(in_features, proto_dim)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 特征张量，形状为 (batch_size, in_features)

        返回:
            mu: 均值张量，形状为 (batch_size, proto_dim)
            log_var: 对数方差张量，形状为 (batch_size, proto_dim)
        """
        mu = self.mu_head(x)
        log_var = self.logvar_head(x)#两个线性层，分别输出均值和对数方差
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)# 对数方差限制在合理范围内，防止数值不稳定
        return mu, log_var
