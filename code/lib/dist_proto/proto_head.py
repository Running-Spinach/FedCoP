# =============================================================================
# 概率原型头 — 把特征向量参数化为高斯分布 N(μ, σ²)
# =============================================================================
# 这是 FedCoP 分布原型的产生模块。在 FedProto 中原型只是一个固定向量(点原型),
# 只能编码"这个类长什么样"。真实医疗数据中同一种病的 X 光片差异很大(不同患者、
# 不同病程),一个点无法表达类内多样性。
#
# 分布原型的优势:
#   - μ(均值):编码"典型特征",相当于点原型
#   - σ²(方差):编码"类内变异性/不确定性",告诉聚合算法"这个原型有多可靠"
#     方差大 → 不同患者差异大 → 聚合时权重小;方差小 → 特征稳定 → 权重大
#
# FedCoP 沿用对角高斯(逐维独立方差):估计稳定、通信廉价,且与贝叶斯融合天然兼容。
# 跨类的"结构"不在这里建模——那由 structured.py 的共现相关矩阵 R̂ 负责。
# =============================================================================

import torch
import torch.nn as nn


class ProbabilisticProtoHead(nn.Module):
    """概率原型头:特征向量 → 高斯分布参数 (μ, logvar)

    架构:
        x (B, in_features)
            │
            ▼
        Linear(in → hidden) + ReLU + LayerNorm   ← 隐藏层(非线性 + 稳定训练)
            │
            ├──→ mu_head     (Linear(hidden → proto_dim))  → μ
            └──→ logvar_head (Linear(hidden → proto_dim))  → logvar

    关键设计:
        1. logvar 而非 var:var 必须 >0,logvar 无约束,exp 后自然得正;数值范围友好。
        2. LayerNorm:联邦下各客户端分布差异大,归一化避免某维度数值爆炸主导梯度。
        3. logvar 偏置初始化 -2.3(σ²≈0.1):折中的初始不确定性,既不"过于自信"
           也不"完全不确定",利于冷启动。
        4. μ/logvar 共享隐藏层:二者描述同一分布,共享底层减少参数 + 防过拟合。

    参数:
        in_features: 输入维度(fc1 输出,如 128)
        proto_dim:   原型输出维度(默认与 in_features 相同)
        hidden_dim:  隐藏层维度(默认 256)
        init_logvar: logvar 偏置初值,默认 -2.3 = log(0.1)
    """

    def __init__(self, in_features, proto_dim=None, hidden_dim=256,
                 init_logvar=-2.3):
        super().__init__()
        if proto_dim is None:
            proto_dim = in_features

        # 隐藏层:非线性变换 + LayerNorm 稳定训练
        self.hidden = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )

        # μ(均值)头与 logvar(对数方差)头,共享隐藏层
        self.mu_head = nn.Sequential(nn.Linear(hidden_dim, proto_dim))
        self.logvar_head = nn.Sequential(nn.Linear(hidden_dim, proto_dim))

        # 校准初始化:冷启动关键
        with torch.no_grad():
            last_linear = self.logvar_head[0]
            nn.init.constant_(last_linear.bias, init_logvar)   # 初始 σ²≈0.1
            nn.init.xavier_uniform_(last_linear.weight)
            last_linear = self.mu_head[0]
            nn.init.xavier_uniform_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(self, x):
        """前向:特征 → 隐藏层 → (μ, logvar)

        参数:
            x: (B, in_features)

        返回:
            mu:     (B, proto_dim) 均值
            logvar: (B, proto_dim) 对数方差,裁剪到 [-10, 10] 防溢出
        """
        h = self.hidden(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar
