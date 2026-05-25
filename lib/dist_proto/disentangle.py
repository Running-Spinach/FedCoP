# 功能：原型解耦模块 — 将原型分解为语义（共享）与风格（本地）两部分
# 原理：最小化语义-风格互信息，仅共享语义原型，保留风格于本地
# 效果：(1) 跨客户端语义原型更纯净  (2) 同等 DP budget 下信噪比更高

import torch
import torch.nn as nn
import torch.nn.functional as F

from .proto_head import ProbabilisticProtoHead


class DisentangledProtoHead(nn.Module):
    """解耦原型头：将 fc1 特征分解为语义和风格两个独立子空间。

    语义空间 (semantic)：疾病判别特征，跨客户端共享
    风格空间 (style)：医院成像特性（对比度、设备差异等），保留本地

    实现方式：将 fc1 输出的前 sem_dim 维作为语义，剩余维度作为风格。
    通过独立性损失（HSIC）约束两部分统计独立。

    参数:
        proto_dim: fc1 输出总维度（默认 256）
        sem_ratio: 语义维度占比（默认 0.75，即 192/256）
        use_distributional: 是否使用分布式原型头
    """

    def __init__(self, proto_dim=256, sem_ratio=0.75, use_distributional=False):
        super().__init__()
        self.sem_dim = int(proto_dim * sem_ratio)
        self.style_dim = proto_dim - self.sem_dim
        self.proto_dim = proto_dim
        self.use_distributional = use_distributional

        if use_distributional:
            self.sem_head = ProbabilisticProtoHead(self.sem_dim, proto_dim=self.sem_dim)
            self.style_head = ProbabilisticProtoHead(self.style_dim, proto_dim=self.style_dim)
        else:
            self.sem_head = None
            self.style_head = None

    def forward(self, proto_features):
        """将原型特征分解为语义和风格两部分。

        参数:
            proto_features: fc1 输出的特征张量 (B, proto_dim)

        返回:
            点原型模式:
              (z_full, z_sem, z_style) — z_full 用于分类，z_sem 用于共享，z_style 用于本地
            分布原型模式:
              (mu_full, logvar_full, mu_sem, logvar_sem, mu_style, logvar_style)
        """
        z_sem = proto_features[:, :self.sem_dim]        # (B, sem_dim)
        z_style = proto_features[:, self.sem_dim:]        # (B, style_dim)

        if self.use_distributional:
            mu_sem, logvar_sem = self.sem_head(z_sem)
            mu_style, logvar_style = self.style_head(z_style)
            mu_full = torch.cat([mu_sem, mu_style], dim=1)
            logvar_full = torch.cat([logvar_sem, logvar_style], dim=1)
            return (mu_full, logvar_full,
                    mu_sem, logvar_sem,
                    mu_style, logvar_style)
        else:
            return proto_features, z_sem, z_style


def disentanglement_loss(z_sem, z_style):
    """基于交叉协方差的独立性损失（线性核 HSIC）。

    最小化语义和风格之间的所有成对线性相关性，
    迫使两部分编码相互独立的信息。

    L_dis = ||Cov(z_sem, z_style)||_F^2

    参数:
        z_sem: 语义特征 (B, D_sem)
        z_style: 风格特征 (B, D_style)

    返回:
        标量独立性损失
    """
    n = z_sem.size(0)
    if n < 2:
        return torch.tensor(0.0, device=z_sem.device)

    z_sem_c = z_sem - z_sem.mean(dim=0, keepdim=True)
    z_style_c = z_style - z_style.mean(dim=0, keepdim=True)

    # 交叉协方差矩阵 C_ij = Cov(z_sem_i, z_style_j)
    cross_cov = torch.mm(z_sem_c.T, z_style_c) / (n - 1)

    # Frobenius 范数平方：Σ_i Σ_j C_ij^2
    loss = torch.sum(cross_cov ** 2)

    # 额外正则：约束语义和风格各自的方差不要坍缩
    var_sem = z_sem_c.var(dim=0).mean()
    var_style = z_style_c.var(dim=0).mean()
    var_reg = F.relu(0.01 - var_sem) + F.relu(0.01 - var_style)

    return loss + 0.1 * var_reg
