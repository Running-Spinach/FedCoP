# =============================================================================
# 功能：增强概率原型头 — 端到端可学习分布原型
# =============================================================================
# 这个模块把"特征向量"变成"高斯分布参数 (μ, σ²)"，是 D²-FL 的核心组件。
#
# 为什么需要这个模块？
#   在原始 FedProto 中，原型就是一个固定向量（点原型），它只能编码"这个类是什么样子"。
#   但在真实医疗数据中，同一种病的 X 光片差异很大（不同患者、不同病程阶段），
#   一个点无法充分表达类内的多样性。
#
# 分布原型的优势：
#   - μ（均值）：编码"典型特征"，相当于点原型的功能
#   - σ²（方差）：编码"类内变异性"，告诉聚合算法"这个原型有多可靠"
#   - 方差大的地方 — 不同患者差异大，聚合时权重应该小
#   - 方差小的地方 — 特征稳定一致，聚合时权重应该大
#
# 包含两个子模块：
#   1. ProbabilisticProtoHead  — 把特征向量变成高斯分布的 μ 和 σ²
#   2. PerClassTemperature     — 每类自适应推理温度
#
# 和旧版的区别：
#   旧版：两层裸 Linear → (μ, logvar)，无隐藏层、无归一化、无残差
#   新版：隐藏层 + LayerNorm + 残差 + 校准初始化
# =============================================================================

import torch
import torch.nn as nn


class ProbabilisticProtoHead(nn.Module):
    """增强概率原型头：特征向量 → 高斯分布参数 N(μ, σ²)

    架构设计（深思熟虑的每一层都有原因的）：

        x (B, in_features)
            │
            ▼
        ┌─────────────────────┐
        │  Linear(in → 256)   │  ← 升维/降维到隐藏维度
        │  ReLU()             │  ← 引入非线性（方差和均值需要不同的变换）
        │  LayerNorm(256)     │  ← 稳定训练（避免某维度数值爆炸主导梯度）
        └─────────────────────┘
            │
            ├──────────────────→ mu_head (Linear(256 → proto_dim))
            │                        输出：μ（均值向量）
            │
            └──────────────────→ logvar_head (Linear(256 → proto_dim))
                                 输出：logvar（对数方差向量）

    关键设计决策：

    1. 为什么用 logvar 而不是直接输出 var？
       — var 必须是正数（>0），而 logvar 可以是任意实数。
         神经网络输出无约束实数更自然，之后 exp(logvar) 自然获得正方差。
       — 数值稳定性：logvar 的范围 [-10, 10] 对应 var 的范围 [~4.5e-5, ~22026]，
         覆盖了从"几乎确定"到"极其不确定"的全部范围。

    2. 为什么需要 LayerNorm？
       — 没有 LayerNorm 时，某些维度的方差可能变得极大，主导 KL/Wasserstein
         损失的计算。LayerNorm 在隐藏层后归一化，保证每个维度的贡献均衡。

    3. 为什么 logvar 偏置初始化为 -2.3（即 σ² ≈ 0.1）？
       — 如果初始化为 0（即 σ² ≈ 1），模型一开始就"非常不确定"，
         原型聚合权重很小，全局原型几乎无法学习。
       — 如果初始化为 -10（即 σ² ≈ 4.5e-5），模型一开始就"过于自信"，
         容易陷入局部最优，且方差可能坍缩回点原型。
       — -2.3（σ² ≈ 0.1）是一个折中：有一定的初始不确定性，但不过分。

    4. 为什么 mu_head 和 logvar_head 共享隐藏层？
       — μ 和 logvar 都描述同一个"类别"的分布，共享底层特征提取
         可以减少参数量、加速训练、避免过拟合。

    参数:
        in_features:  输入特征维度（来自 fc1 输出，如 256）
        proto_dim:    高斯原型输出维度（默认与 in_features 相同）
        hidden_dim:   隐藏层维度（默认 256，提供足够的非线性容量）
        init_logvar:  logvar 偏置初始值。默认 -2.3 = log(0.1)
    """

    def __init__(self, in_features, proto_dim=None, hidden_dim=256,
                 init_logvar=-2.3):
        super().__init__()
        if proto_dim is None:
            proto_dim = in_features

        # 隐藏层：提供非线性变换能力
        # LayerNorm 稳定训练 — 联邦学习中不同客户端的数据分布差异大，
        # 没有 LayerNorm 的话，某些客户端的梯度可能过大或过小
        self.hidden = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )

        # μ（均值）头：预测"典型特征"
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, proto_dim),
        )

        # logvar（对数方差）头：预测"不确定程度"
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, proto_dim),
        )

        # ════════════════════════════════════════════════════════════
        #  校准初始化 — 这是"冷启动"的关键
        # ════════════════════════════════════════════════════════════
        with torch.no_grad():
            # logvar_head: 偏置 = init_logvar，权重用 Xavier 均匀分布
            # 这确保初始输出 ≈ init_logvar，即初始方差 ≈ exp(init_logvar)
            last_linear = self.logvar_head[0]
            nn.init.constant_(last_linear.bias, init_logvar)
            nn.init.xavier_uniform_(last_linear.weight)

            # mu_head: 偏置 = 0，权重用 Xavier 均匀分布
            # 均值初始化为 0（对称性，不偏向任何方向）
            last_linear = self.mu_head[0]
            nn.init.xavier_uniform_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(self, x):
        """前向传播：特征 → 隐藏层 → (μ, logvar)

        参数:
            x: 特征张量，shape (B, in_features)，B = batch 大小

        返回:
            mu:     均值向量，shape (B, proto_dim)
            logvar: 对数方差向量，shape (B, proto_dim)，裁剪到 [-10, 10]
                   防止 logvar 过大或过小导致数值溢出
        """
        h = self.hidden(x)                       # 隐藏层变换
        mu = self.mu_head(h)                     # 均值预测
        logvar = self.logvar_head(h)             # 对数方差预测
        # 裁剪：exp(-10) ≈ 4.5e-5，exp(10) ≈ 22026
        # 在这个范围之外的值在物理/数学上没有意义
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar


class PerClassTemperature(nn.Module):
    """每类可学习温度参数 — 自适应推理锐度

    为什么需要这个模块？
        在原型推理时，我们用"到全局原型的距离"来做分类：
            logit[class_j] = -dist(proto_i, global_proto[j]) / T_j

        温度 T_j 控制的是"改变距离一个单位，logit 变化多少"：
        - T < 1：锐化 → 距离稍微变近，logit 就大幅增加 → 更"自信"
        - T > 1：软化 → 需要距离非常近，logit 才有明显变化 → 更"保守"
        - T = 1：标准缩放

        不同类别的"难度"不同：
        - 容易区分的疾病（如气胸，特征明显）→ T 应该较小（锐化，自信判断）
        - 难区分的疾病（如肺炎和肺水肿容易混淆）→ T 应该较大（软化，谨慎判断）

        这个模块让网络通过梯度反向传播，端到端地为每个类别学习最优温度。

    实现细节：
        用 log_temp 而不是 temp 作为可学习参数，原因和 logvar 一样：
        temp 必须 > 0，log_temp 可以是任意实数，网络输出更自然。

    参数:
        num_classes: 类别数量（ChestX-ray14 = 14）
        init_temp:   初始温度值（默认 1.0 = 标准缩放）
    """

    def __init__(self, num_classes, init_temp=1.0):
        super().__init__()
        # 可学习参数：log(T)，初始化为 log(init_temp)
        init_log_temp = torch.log(torch.tensor(init_temp))
        self.log_temp = nn.Parameter(
            torch.full((num_classes,), init_log_temp)
        )

    def forward(self, class_indices=None):
        """返回指定类别的温度值

        参数:
            class_indices: 类别索引（可选）。如 [0, 3, 5] 只返回类别0、3、5的温度。
                          None 时返回全部类别的温度。

        返回:
            temperatures: exp(log_temp) > 0，shape 由 class_indices 决定
        """
        # exp 确保温度始终为正
        temps = torch.exp(self.log_temp)
        if class_indices is not None:
            temps = temps[class_indices]
        return temps
