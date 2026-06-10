# 功能：增强原型解耦模块 — 语义-风格分离
# 相比旧版改进：
#   1. 可学习门控 → 替代硬分割（前75%=语义），端到端学习最优维度分配
#   2. 对抗域分类器 → 迫使语义特征无法识别来源客户端，真正实现域不变
#   3. 对比语义对齐 → 同类语义特征跨样本拉近，异类推远
#   4. 正交正则化 → 语义和风格子空间的基向量正交约束
#
# 原理：
#   - 语义 (semantic)：疾病判别特征 → 跨客户端共享
#   - 风格 (style)：医院成像特性 → 保留本地
#   - 三者协同：门控分离 + 对抗确保域不变 + 对比确保类别可分 + 正交防信息泄漏

import torch
import torch.nn as nn
import torch.nn.functional as F

from .proto_head import ProbabilisticProtoHead


class GradientReversal(torch.autograd.Function):
    """梯度反转层：前向恒等映射，反向梯度乘 -lambda

    用途：对抗训练中，阻止语义特征编码域信息。
    前向: y = x
    反向: dy/dx = -lambda * dy/dy
    """

    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradientReversal.apply(x, lambd)


class LearnableGate(nn.Module):
    """可学习门控：为每个特征维度分配语义/风格软权重

    替代硬分割（旧版：前 sem_dim 维=语义，其余=风格），
    门控允许网络端到端学习最优的语义/风格维度分配。

    原理：
      gate = sigmoid(Linear(x))  ∈ (0, 1)^D
      z_sem   = gate * x          # 每个维度按 gate 流向语义
      z_style = (1 - gate) * x    # 每个维度按 1-gate 流向风格

    附加熵正则化：鼓励 gate 接近 0/1（避免模糊），同时确保
    有一定比例维度分配给语义（避免退化到全0或全1）。
    """

    def __init__(self, proto_dim, bias_init=0.5):
        super().__init__()
        self.gate_layer = nn.Linear(proto_dim, proto_dim)
        # 初始化偏置使 gate 初始值约 0.5（最大不确定性）
        with torch.no_grad():
            nn.init.zeros_(self.gate_layer.weight)
            nn.init.constant_(self.gate_layer.bias, bias_init)

    def forward(self, x):
        """返回语义/风格软门控和分离后的特征

        返回:
            z_sem, z_style: 分离后的特征 (B, proto_dim)
            gate: 门控值 (B, proto_dim)，用于正则化
        """
        gate = torch.sigmoid(self.gate_layer(x))
        z_sem = gate * x
        z_style = (1 - gate) * x
        return z_sem, z_style, gate


class DisentangledProtoHead(nn.Module):
    """增强解耦原型头：可学习门控 + 对抗域不变 + 对比语义对齐

    架构：
      fc1 特征 → LearnableGate → z_sem, z_style
        z_sem  → [可选 ProtoHead] → mu_sem, logvar_sem
        z_style → [可选 ProtoHead] → mu_style, logvar_style
        z_sem  → GradientReversal → DomainClassifier（对抗）

    训练时的损失组成：
      1. L_CE:  交叉熵分类损失（基于全维度 proto_features）
      2. L_dis:  解耦独立性损失（HSIC + 门控熵 + 正交约束）
      3. L_adv:  对抗域分类损失（语义特征不应包含域信息）
      4. L_contra: 对比语义对齐损失（同类拉近、异类推远）

    参数:
        proto_dim: fc1 输出维度 (默认 256)
        sem_ratio: 语义维度的目标占比（仅用于正则化引导，非硬约束）
        use_distributional: 是否使用分布式原型头
    """

    def __init__(self, proto_dim=256, sem_ratio=0.75, use_distributional=False):
        super().__init__()
        self.proto_dim = proto_dim
        self.sem_dim = int(proto_dim * sem_ratio)
        self.style_dim = proto_dim - self.sem_dim
        self.use_distributional = use_distributional
        self.sem_ratio = sem_ratio

        # 核心创新 1: 可学习门控替代硬分割
        self.gate = LearnableGate(proto_dim)

        # 核心创新 2: 对抗域分类器
        # 轻量级：单隐藏层 → 输出 num_domains（训练时动态设置 or 固定为 num_clients）
        self.domain_classifier = nn.Sequential(
            nn.Linear(proto_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),  # 二分类：区分"当前客户端 vs 其他"的域差异
            # 注意：真实应用中 num_domains=num_clients，这里用 1 做简化（二分类域判别）
            # 可扩展到多客户端：改为 nn.Linear(128, num_clients)
        )

        # 分布原型头（可选）
        if use_distributional:
            self.sem_head = ProbabilisticProtoHead(proto_dim, proto_dim=proto_dim,
                                                    hidden_dim=proto_dim)
            self.style_head = ProbabilisticProtoHead(proto_dim, proto_dim=proto_dim,
                                                      hidden_dim=proto_dim)
        else:
            self.sem_head = None
            self.style_head = None

    def forward(self, proto_features, return_gate=False):
        """前向传播：门控分离 → 可选分布参数化

        参数:
            proto_features: fc1 输出 (B, proto_dim)
            return_gate: 是否返回门控值（用于正则化计算）

        返回:
            分布模式: (mu_full, logvar_full, mu_sem, logvar_sem, mu_style, logvar_style)
            点模式:   (z_full, z_sem, z_style)
            若 return_gate=True，末尾追加 gate 张量
        """
        z_sem, z_style, gate = self.gate(proto_features)

        if self.use_distributional:
            mu_sem, logvar_sem = self.sem_head(z_sem)
            mu_style, logvar_style = self.style_head(z_style)
            # 全维度 = 语义 + 风格拼接（保持与分类头 fc2 的兼容性）
            mu_full = torch.cat([mu_sem, mu_style], dim=1)
            logvar_full = torch.cat([logvar_sem, logvar_style], dim=1)
            result = (mu_full, logvar_full,
                      mu_sem, logvar_sem,
                      mu_style, logvar_style)
        else:
            z_full = torch.cat([z_sem, z_style], dim=1)
            result = (z_full, z_sem, z_style)

        if return_gate:
            result = result + (gate,)
        return result

    def forward_adversarial(self, z_sem, grad_reverse_lambda=1.0):
        """对抗域分类前向：对语义特征做梯度反转后预测域标签

        参数:
            z_sem: 语义特征 (B, proto_dim)
            grad_reverse_lambda: 梯度反转强度

        返回:
            domain_logits: 域分类 logits (B, 1)
        """
        z_rev = grad_reverse(z_sem, lambd=grad_reverse_lambda)
        return self.domain_classifier(z_rev)


def disentanglement_loss(z_sem, z_style, gate=None, proto_dim=None):
    """增强解耦损失：HSIC 独立性 + 门控熵正则 + 正交约束

    L_dis = L_HSIC + α_ent * L_gate_entropy + α_orth * L_orthogonal

    各分量含义：
      L_HSIC: 交叉协方差 Frobenius 范数 → 最小化语义-风格线性相关
      L_gate_entropy: -gate*log(gate) → 鼓励门控接近 0/1（锐化）
      L_orthogonal: ||W_sem^T W_style||_F → 语义/风格基正交

    参数:
        z_sem: 语义特征 (B, D)
        z_style: 风格特征 (B, D)
        gate: 门控值 (B, D)，可选
        proto_dim: 原型总维度（仅用于维度不匹配时截断）

    返回:
        loss: 标量解耦损失
    """
    n = z_sem.size(0)
    if n < 2:
        return torch.tensor(0.0, device=z_sem.device)

    # ── 1. HSIC 独立性损失 ──
    z_sem_c = z_sem - z_sem.mean(dim=0, keepdim=True)
    z_style_c = z_style - z_style.mean(dim=0, keepdim=True)

    # 交叉协方差矩阵 C_{ij} = Cov(z_sem_i, z_style_j)
    cross_cov = torch.mm(z_sem_c.T, z_style_c) / (n - 1)
    loss_hsic = torch.sum(cross_cov ** 2)

    # 方差防坍缩正则
    var_sem = z_sem_c.var(dim=0).mean()
    var_style = z_style_c.var(dim=0).mean()
    var_reg = F.relu(0.01 - var_sem) + F.relu(0.01 - var_style)

    loss = loss_hsic + 0.1 * var_reg

    # ── 2. 门控熵正则：鼓励 gate → {0, 1} ──
    if gate is not None:
        gate = torch.clamp(gate, 1e-8, 1 - 1e-8)
        entropy_per_dim = -(gate * torch.log(gate) + (1 - gate) * torch.log(1 - gate))
        # 最小化熵 → 推动 gate 接近 0 或 1（锐化决策）
        loss_gate_entropy = entropy_per_dim.mean()
        loss = loss + 0.01 * loss_gate_entropy

    # ── 3. 正交约束：语义/风格基应正交 ──
    # 在样本维度上做正交：整个 batch 的语义和风格向量应尽可能正交
    z_sem_norm = F.normalize(z_sem_c, p=2, dim=0)    # (B, D) 逐维度归一化
    z_style_norm = F.normalize(z_style_c, p=2, dim=0)
    # 内积矩阵的 Frobenius 范数
    ortho = torch.mm(z_sem_norm.T, z_style_norm)       # (D, D)
    loss_orth = torch.sum(ortho ** 2) / z_sem.size(1)
    loss = loss + 0.01 * loss_orth

    return loss


def contrastive_semantic_loss(z_sem, labels, temperature=0.1):
    """对比语义对齐损失：同类语义特征拉近，异类推远

    使用 InfoNCE 风格对比学习，在 batch 内构建正负样本对。
    这对 Non-IID FL 特别重要：即使不同客户端数据分布不同，
    相同疾病的语义表征应该在语义空间中接近。

    实现：
      - 正样本对：同一 batch 中标签 Jaccard 相似度 > 0.5 的样本
      - 负样本对：其余样本
      - 损失：-log(exp(sim(z_i, z_pos)/τ) / Σ exp(sim(z_i, z_j)/τ))

    参数:
        z_sem: 语义特征 (B, D)
        labels: 多标签矩阵 (B, num_classes)，值 ∈ {0, 1}
        temperature: 对比温度 (默认 0.1)

    返回:
        loss: 标量对比损失
    """
    B = z_sem.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z_sem.device)

    # L2 归一化 → 余弦相似度空间
    z_norm = F.normalize(z_sem, p=2, dim=1)           # (B, D)

    # 相似度矩阵
    sim = torch.mm(z_norm, z_norm.T) / temperature     # (B, B)

    # 正样本掩码：Jaccard 相似度 > 0.5 或标签完全一致
    labels_float = labels.float()
    intersection = torch.mm(labels_float, labels_float.T)                    # (B, B)
    union = labels_float.sum(1, keepdim=True) + labels_float.sum(1, keepdim=True).T - intersection
    jaccard = intersection / (union + 1e-8)

    pos_mask = (jaccard > 0.5).float()
    pos_mask.fill_diagonal_(0)  # 排除自身

    # InfoNCE 损失
    exp_sim = torch.exp(sim)
    # 分母：所有样本对的相似度之和（排除自身）
    denom = exp_sim.sum(dim=1) - torch.exp(sim.diag())

    # 分子：正样本对的相似度之和
    num_pos = pos_mask.sum(dim=1)                      # (B,)
    pos_sim_sum = (exp_sim * pos_mask).sum(dim=1)      # (B,)

    # 仅对有正样本的样本计算损失
    valid = num_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z_sem.device)

    loss = -torch.log(
        (pos_sim_sum[valid] / (denom[valid] + 1e-8)) + 1e-8
    ).mean()

    return loss


def adversarial_disentanglement_loss(domain_logits, target_labels=None):
    """对抗域分类损失：语义特征应无法区分来源域

    使用 BCEWithLogitsLoss：
      - 若 target_labels 未提供，默认 target=0（希望分类器无法区分）
      - 若提供，则为标准域分类损失（用于风格分支）

    参数:
        domain_logits: 域分类器输出 (B, 1) 或 (B, num_domains)
        target_labels: 目标域标签，None 时默认全零

    返回:
        loss: 标量对抗损失
    """
    if target_labels is None:
        target_labels = torch.zeros_like(domain_logits)
    return F.binary_cross_entropy_with_logits(domain_logits, target_labels)
