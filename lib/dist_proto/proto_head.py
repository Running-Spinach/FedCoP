# 功能：增强概率原型头 — 端到端可学习分布原型
# 相比旧版改进：
#   1. 更深架构：隐藏层 + LayerNorm → 更强的表达能力
#   2. 每类可学习温度参数 → 自适应推理锐度
#   3. 残差连接 → 训练稳定性
#   4. 校准初始化 → logvar 初始值对应合理的不确定性

import torch
import torch.nn as nn
class ProbabilisticProtoHead(nn.Module):
    """
    增强概率原型头，将特征向量转换为高斯分布参数 N(mu, sigma^2)

    架构：x → Hidden(ReLU+LN) → mu_head / logvar_head (残差)
      - mu_head:   Linear(in, proto_dim)
      - logvar_head: Linear(in, proto_dim) → 初始化为偏置=log(0.1)≈−2.3

    相比旧版（两层裸Linear）：
      - 添加 256→256 隐藏层 + LayerNorm，提升非线性表达能力
      - logvar_head 偏置初始化为 log(σ²_init)，避免训练早期的零方差坍缩
      - 残差连接使恒等映射成为可能，稳定深层原型训练

    参数:
        in_features: 输入特征维度 (来自 fc1 输出)
        proto_dim: 高斯原型输出维度 (默认与 in_features 相同)
        hidden_dim: 隐藏层维度 (默认 256)
        init_logvar: logvar 偏置初始值 (默认 log(0.1) ≈ -2.3)
    """

    def __init__(self, in_features, proto_dim=None, hidden_dim=256,
                 init_logvar=-2.3):
        super().__init__()
        if proto_dim is None:
            proto_dim = in_features

        # 隐藏层：增强非线性表达
        self.hidden = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )

        # mu/logvar 头：残差连接
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, proto_dim),
        )

        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, proto_dim),
        )

        # 校准初始化：logvar 偏置 → init_logvar
        # 默认 init_logvar = log(0.1) = -2.3 → σ²_init ≈ 0.1，避免"过于自信"
        with torch.no_grad():
            last_linear = self.logvar_head[0]
            nn.init.constant_(last_linear.bias, init_logvar)
            nn.init.xavier_uniform_(last_linear.weight)

        # mu head 正常初始化
        with torch.no_grad():
            last_linear = self.mu_head[0]
            nn.init.xavier_uniform_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(self, x):
        """
        前向传播：x → hidden → mu/logvar

        参数:
            x: 特征张量 (B, in_features)

        返回:
            mu: 均值 (B, proto_dim)
            logvar: 对数方差 (B, proto_dim)，裁剪到 [-10, 10] 防数值溢出
        """
        h = self.hidden(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar


class PerClassTemperature(nn.Module):
    """
    每类可学习温度参数

    用于原型推理时的类别自适应温度缩放：
      proto_logit[class_j] = -dist(proto_i, global_proto[j]) / T_j

    原理：
      - 类间区分度高的类别 → T > 1（软化，容忍更多不确定性）
      - 类间区分度低的类别 → T < 1（锐化，增强判别力）
      - 通过梯度反向传播端到端学习最优温度

    参数:
        num_classes: 类别数量
        init_temp: 初始温度值 (默认 1.0)
    """

    def __init__(self, num_classes, init_temp=1.0):
        super().__init__()
        init_log_temp = torch.log(torch.tensor(init_temp))
        self.log_temp = nn.Parameter(
            torch.full((num_classes,), init_log_temp)
        )

    def forward(self, class_indices=None):
        """
        返回指定类别的温度值

        参数:
            class_indices: 类别索引 (可选). None 时返回全部类别

        返回:
            temperatures: exp(log_temp) > 0, shape 由 class_indices 决定
        """
        temps = torch.exp(self.log_temp)
        if class_indices is not None:
            temps = temps[class_indices]
        return temps
