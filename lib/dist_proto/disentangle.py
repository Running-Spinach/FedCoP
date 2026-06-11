# =============================================================================
# 功能：D²-FL 增强原型解耦模块 — 语义-风格分离
# =============================================================================
# 这个模块是 D²-FL 的核心创新之一，解决联邦学习中的一个关键问题：
# 不同医院（客户端）的 X 光片虽然拍的是同一种病，但因为机器型号、拍摄参数、
# 图像处理流程不同，导致图片"风格"差异很大。这种风格差异会污染原型，
# 让全局原型变模糊。
#
# 解决思路：把特征分成两类——
#   - 语义特征（semantic）：疾病本身的特征，如病灶形状、位置、密度。
#                         这些是跨医院通用的，应该共享。
#   - 风格特征（style）：医院特有的成像特征，如对比度、亮度、噪声模式。
#                       这些是各医院自己的事，应该保留本地。
#
# 三个子模块协同工作：
#   1. LearnableGate   — 可学习门控，自动决定每个维度该分给语义还是风格
#   2. GradientReversal — 梯度反转，对抗训练的核心，让语义特征"洗掉"域信息
#   3. DomainClassifier  — 域分类器，尝试从语义特征猜出来源客户端（对抗目标）
#
# 四个损失函数确保解耦质量：
#   1. disentanglement_loss    — HSIC 独立性 + 门控熵 + 正交约束
#   2. contrastive_semantic_loss — 对比学习：同类疾病语义要相似
#   3. adversarial_disentanglement_loss — 对抗损失：语义特征不应包含域信息
#
# 和 FedSeProto 的区别：
#   - FedSeProto：硬分割（前75%维度=语义，后25%=域）+ 仅 HSIC
#   - D²-FL：软门控（可学习分配）+ HSIC + 门控熵 + 正交 + 对抗 + 对比
#
# 通俗理解：
#   就像一个合唱团，每个人有自己的音色（风格），但唱的是同一首歌（语义）。
#   语义-风格解耦就是：只分享"歌谱"（语义原型），不分享"嗓音"（风格原型）。
#   对抗训练则确保：别人拿到歌谱后，猜不出是哪个歌手唱的。
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from .proto_head import ProbabilisticProtoHead


# =============================================================================
#  梯度反转层 — 对抗训练的"魔法开关"
# =============================================================================

class GradientReversal(torch.autograd.Function):
    """梯度反转层（Gradient Reversal Layer, GRL）

    这是一个"两面派"操作：
    - 前向传播：什么都不做，x 原样输出（y = x）
    - 反向传播：梯度乘 -λ，方向反转（dx = -λ * dy）

    用途：
        在对抗训练中，GRL 夹在语义特征和域分类器之间。
        前向时，域分类器正常看到语义特征并尝试分类。
        反向时，梯度反转"欺骗"特征提取器，让它朝着
        "让域分类器猜错"的方向更新。

    直观理解：
        就像给语义特征穿上"隐身衣"——
        域分类器想从语义特征猜出来源医院，
        但梯度反转会让语义特征越来越"医院无关"，
        最终域分类器只能瞎猜（50%准确率≈随机）。

    参数:
        lambd: 梯度反转强度。λ=0=不反转，λ=1=完全反转。
               通常训练初期小，后期逐渐增大。
    """

    @staticmethod
    def forward(ctx, x, lambd=1.0):
        # ctx 是上下文，用来在 forward 和 backward 之间传话
        ctx.lambd = lambd
        return x  # 前向：原样通过

    @staticmethod
    def backward(ctx, grad_output):
        # 反向：梯度乘 -λ（方向反转 + 强度缩放）
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    """梯度反转的便捷调用接口"""
    return GradientReversal.apply(x, lambd)


# =============================================================================
#  可学习门控 — 软分割替代硬分割
# =============================================================================

class LearnableGate(nn.Module):
    """可学习门控：为每个特征维度自动分配语义/风格软权重

    旧版做法（FedSeProto）：
        前 sem_dim 维（如 256 × 0.75 = 192 维）固定分给语义，
        后 style_dim 维固定分给风格。
        问题是：哪些维度是语义、哪些是风格，可能不是按索引排列的。

    本模块的做法（D²-FL）：
        通过一个可学习的线性层 + sigmoid，为每个维度输出一个
        (0, 1) 之间的软权重 gate：
        - gate[d] ≈ 1：第 d 维主要流向语义
        - gate[d] ≈ 0：第 d 维主要流向风格
        - gate[d] ≈ 0.5：第 d 维两边都流（不鼓励，有熵正则惩罚）

    分离公式：
        z_sem[d]   = gate[d]      × x[d]    # 每个维度被门控按比例分配
        z_style[d] = (1 - gate[d]) × x[d]    # 总信息量守恒

    直观理解：
        这就像调音台上的推子——每个维度的信息不是"全给语义"或"全给风格"，
        而是按比例分配到两边。网络自己学习最优的分配方案。

    初始化策略：
        偏置初始化为 0.5，使 sigmoid(0.5) ≈ 0.62，接近 0.5（最大不确定性）。
        这样可以避免训练初期网络"先入为主"地把所有维度都分给一边。
    """

    def __init__(self, proto_dim, bias_init=0.5):
        """
        参数:
            proto_dim: 原型特征的总维度（如 256）
            bias_init: 偏置初始值。0.5 使得 gate 初始值约 0.62
        """
        super().__init__()
        self.gate_layer = nn.Linear(proto_dim, proto_dim)
        # 特殊初始化：权重=0（初始时各维度独立判断），偏置=0.5（初始时不确定）
        with torch.no_grad():
            nn.init.zeros_(self.gate_layer.weight)
            nn.init.constant_(self.gate_layer.bias, bias_init)

    def forward(self, x):
        """前向传播：输入特征 → 门控分配 → 语义+风格分离

        参数:
            x: 输入特征，shape (B, proto_dim)，B = batch 大小

        返回:
            z_sem:   语义特征，shape (B, proto_dim)
            z_style: 风格特征，shape (B, proto_dim)
            gate:    门控值，shape (B, proto_dim)，用于后续的正则化损失计算
        """
        # sigmoid 将输出压缩到 (0, 1) 区间，作为软权重
        gate = torch.sigmoid(self.gate_layer(x))

        # 按门控比例分配信息
        z_sem = gate * x           # 信息流向语义
        z_style = (1 - gate) * x   # 剩余信息流向风格

        return z_sem, z_style, gate


# =============================================================================
#  增强解耦原型头 — D²-FL 的"大脑"
# =============================================================================

class DisentangledProtoHead(nn.Module):
    """增强解耦原型头：可学习门控 + 对抗域不变 + 分布原型

    这是 D²-FL 解耦模式的核心组件，负责：
    1. 把 fc1 输出分离成语义和风格
    2. 可选地将语义/风格特征参数化为高斯分布
    3. 提供对抗域分类接口，用于训练时的域不变性约束

    架构流程：
        fc1 特征 (B, 256)
            │
            ▼
        LearnableGate ─── gate (门控值，用于正则化)
            │
            ├── z_sem (语义分支)
            │     │
            │     ├── [可选] ProbabilisticProtoHead → (μ_sem, logvar_sem)
            │     └── GradientReversal → DomainClassifier（对抗训练用）
            │
            └── z_style (风格分支)
                  │
                  └── [可选] ProbabilisticProtoHead → (μ_style, logvar_style)

    训练时的四重损失：
        1. L_CE:    分类损失（基于 fc2 输出，fc2 输入仍是 fc1 全维度特征）
        2. L_dis:   解耦独立性损失（HSIC + 门控熵 + 正交）
        3. L_adv:   对抗域分类损失（通过 GRL 反向）
        4. L_contra: 对比语义对齐损失（同类别拉近、异类别推远）

    和 FedSeProto 的区别：
        - FedSeProto: 两个独立 MLP 头把特征硬分成语义/域两部分（硬分割）
        - D²-FL: 可学习门控软分割 + 对抗训练 + 对比学习 + 分布原型

    参数:
        proto_dim:           fc1 输出维度（默认 256）
        sem_ratio:           语义维度目标占比（仅用于正则化引导，实际分配由门控决定）
        use_distributional:  是否使用分布式原型头输出 (μ, logvar)
    """

    def __init__(self, proto_dim=256, sem_ratio=0.75, use_distributional=False):
        super().__init__()
        self.proto_dim = proto_dim
        self.sem_dim = int(proto_dim * sem_ratio)       # 目标语义维度数（参考值）
        self.style_dim = proto_dim - self.sem_dim        # 目标风格维度数（参考值）
        self.use_distributional = use_distributional
        self.sem_ratio = sem_ratio

        # ── 创新1: 可学习门控替代硬分割 ──
        # 和 FedSeProto 的核心区别：不是固定前N维=语义，而是端到端学习
        self.gate = LearnableGate(proto_dim)

        # ── 创新2: 对抗域分类器 ──
        # 这个分类器的任务是"从语义特征猜出来自哪个客户端"。
        # 但通过梯度反转，语义特征会被训练成"让分类器猜不出来"。
        # 当分类器只能瞎猜时 → 语义特征成功实现了域不变。
        self.domain_classifier = nn.Sequential(
            nn.Linear(proto_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),  # 输出1维logit → BCEWithLogitsLoss
        )

        # ── 分布原型头（可选）──
        # 将语义/风格特征参数化为高斯分布 N(μ, σ²)
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
            proto_features: fc1 输出，shape (B, proto_dim)
            return_gate:    是否返回门控值。训练时需要（用于熵正则），
                           推理/原型聚合时不需要。

        返回（一个元组，根据模式不同内容不同）:
            分布模式: (μ_full, logvar_full, μ_sem, logvar_sem, μ_style, logvar_style)
            点模式:   (z_full, z_sem, z_style)
            若 return_gate=True，末尾追加 gate 张量

            其中 "full" 是拼接了 sem+style 的完整维度（保持和 fc2 兼容）
        """
        # 步骤1：门控分离
        z_sem, z_style, gate = self.gate(proto_features)

        if self.use_distributional:
            # 步骤2a：分布参数化（语义和风格分别过 ProtoHead）
            mu_sem, logvar_sem = self.sem_head(z_sem)
            mu_style, logvar_style = self.style_head(z_style)

            # 拼接全维度：保持和分类头 fc2 的维度兼容
            mu_full = torch.cat([mu_sem, mu_style], dim=1)
            logvar_full = torch.cat([logvar_sem, logvar_style], dim=1)

            result = (mu_full, logvar_full,
                      mu_sem, logvar_sem,
                      mu_style, logvar_style)
        else:
            # 步骤2b：点原型模式，直接拼接
            z_full = torch.cat([z_sem, z_style], dim=1)
            result = (z_full, z_sem, z_style)

        # 按需返回门控值（训练时需要用于熵正则）
        if return_gate:
            result = result + (gate,)
        return result

    def forward_adversarial(self, z_sem, grad_reverse_lambda=1.0):
        """对抗域分类前向：语义特征 → GRL → 域分类器

        这个函数只在训练时调用，用于计算对抗损失。
        通过梯度反转层，让语义特征朝着"无法被域分类器识别"的方向优化。

        参数:
            z_sem: 语义特征，shape (B, proto_dim)
            grad_reverse_lambda: 梯度反转强度（默认1.0=完全反转）

        返回:
            domain_logits: 域分类 logits，shape (B, 1)
                          正值→分类器认为来自"其他域"，负值→认为是"当前域"
        """
        # GRL 在这起作用：前向原样通过，反向梯度反转
        z_rev = grad_reverse(z_sem, lambd=grad_reverse_lambda)
        return self.domain_classifier(z_rev)


# =============================================================================
#  解耦独立性损失 — L_dis
# =============================================================================

def disentanglement_loss(z_sem, z_style, gate=None, proto_dim=None):
    """增强解耦损失：三合一确保语义和风格真正分离

    L_dis = L_HSIC + α_ent × L_gate_entropy + α_orth × L_orthogonal

    三个分量各司其职：

    1. L_HSIC（Hilbert-Schmidt Independence Criterion）
       — 核心：最小化语义和风格的"统计相关性"
       — 做法：计算交叉协方差矩阵的 Frobenius 范数
       — 目标：让语义和风格彼此独立（知道语义推不出风格，反之亦然）
       — 通俗理解：如果从"病灶形状"能推断出"是哪个医院的机器拍的"，
         那说明解耦不彻底，HSIC 就会惩罚。

    2. L_gate_entropy（门控熵正则）
       — 核心：鼓励门控"态度鲜明"，不要模棱两可
       — 做法：最小化二元熵 -[g·log(g) + (1-g)·log(1-g)]
       — 目标：gate → 0 或 1（清晰分配），而不是 0.5（模糊分配）
       — 通俗理解：每个维度要么归语义、要么归风格，不要"两边都沾一点"

    3. L_orthogonal（正交约束）
       — 核心：语义和风格在"方向"上应该正交
       — 做法：计算归一化语义向量和风格向量的内积，惩罚非零值
       — 目标：两个子空间垂直，信息不会"泄漏"
       — 通俗理解：就像两堵互相垂直的墙，在一堵墙上移动不会改变在另一堵墙上的位置

    参数:
        z_sem:    语义特征，shape (B, D)
        z_style:  风格特征，shape (B, D)
        gate:     门控值，shape (B, D)，可选（None 则跳门控熵）
        proto_dim: 原型总维度（保留参数，兼容旧版）

    返回:
        loss: 标量解耦损失
    """
    n = z_sem.size(0)
    # batch 太小（比如只有1个样本）无法计算协方差，直接返回0
    if n < 2:
        return torch.tensor(0.0, device=z_sem.device)

    # ════════════════════════════════════════════════════════════
    #  1. HSIC 独立性损失 — 核心解耦约束
    # ════════════════════════════════════════════════════════════
    # 首先中心化（减去均值），这是计算协方差的前提
    z_sem_c = z_sem - z_sem.mean(dim=0, keepdim=True)
    z_style_c = z_style - z_style.mean(dim=0, keepdim=True)

    # 交叉协方差矩阵 C[i][j] = Cov(z_sem 的第i维, z_style 的第j维)
    # 形状 (D, D)，除以 (n-1) 是无偏估计
    cross_cov = torch.mm(z_sem_c.T, z_style_c) / (n - 1)

    # HSIC = ||C||_F² = 交叉协方差所有元素的平方和
    # 如果语义和风格独立，C ≈ 0 → HSIC ≈ 0
    loss_hsic = torch.sum(cross_cov ** 2)

    # 方差防坍缩正则：如果语义或风格的方差变得极小，
    # HSIC 也会变小（但这是退化，不是真正的解耦）。
    # 加一个 relu 惩罚，确保每个维度至少有 0.01 的方差。
    var_sem = z_sem_c.var(dim=0).mean()
    var_style = z_style_c.var(dim=0).mean()
    var_reg = F.relu(0.01 - var_sem) + F.relu(0.01 - var_style)

    loss = loss_hsic + 0.1 * var_reg

    # ════════════════════════════════════════════════════════════
    #  2. 门控熵正则 — 鼓励"态度鲜明"
    # ════════════════════════════════════════════════════════════
    if gate is not None:
        # clamp 防止 log(0) 导致 NaN
        gate = torch.clamp(gate, 1e-8, 1 - 1e-8)
        # 二元熵公式：H(g) = -[g·log(g) + (1-g)·log(1-g)]
        # 当 g=0.5 时熵最大（≈0.69），g=0或1时熵=0
        entropy_per_dim = -(gate * torch.log(gate) + (1 - gate) * torch.log(1 - gate))
        # 最小化熵 → 推动 gate 接近 0 或 1
        loss_gate_entropy = entropy_per_dim.mean()
        loss = loss + 0.01 * loss_gate_entropy

    # ════════════════════════════════════════════════════════════
    #  3. 正交约束 — 防止信息泄漏
    # ════════════════════════════════════════════════════════════
    # 在特征维度间做正交：归一化后计算内积矩阵
    z_sem_norm = F.normalize(z_sem_c, p=2, dim=0)       # 逐维度 L2 归一化
    z_style_norm = F.normalize(z_style_c, p=2, dim=0)

    # 内积矩阵 O[i][j] = <sem的第i维, style的第j维>
    # 如果正交，O ≈ 零矩阵
    ortho = torch.mm(z_sem_norm.T, z_style_norm)         # (D, D)

    # 最小化内积矩阵的 Frobenius 范数 → 推动正交
    loss_orth = torch.sum(ortho ** 2) / z_sem.size(1)
    loss = loss + 0.01 * loss_orth

    return loss


# =============================================================================
#  对比语义对齐损失 — L_contra
# =============================================================================

def contrastive_semantic_loss(z_sem, labels, temperature=0.1):
    """对比语义对齐损失 — 同类疾病语义特征要相似，异类要不同

    为什么需要这个损失？
        在 Non-IID 联邦学习中，不同客户端的数据分布差异很大。
        比如医院A只有"肺结节"和"肺炎"的病例，医院B只有"气胸"和"水肿"。
        如果没有对比损失，两个医院对同一种病的语义特征可能因"缺乏对照"
        而漂移到不同的方向。

    实现方式（InfoNCE 风格的对比学习）：
        1. 计算 batch 内所有样本对的语义相似度
        2. 用标签的 Jaccard 相似度建立正/负样本关系：
           - 正样本对：Jaccard > 0.5（共享至少一半的疾病标签）
           - 负样本对：其余所有样本
        3. 损失 = -log(exp(sim(正样本)) / Σ exp(sim(所有样本)))
           → 拉近正样本、推远负样本

    为什么用 Jaccard 相似度而不是标签完全一致？
        因为是多标签分类，两个样本可能有"肺结节+肺炎"和"肺结节+水肿"，
        它们共享"肺结节"，语义上应该有部分相似。Jaccard > 0.5 是一个
        合理的阈值，允许部分重叠的样本也作为正样本对。

    参数:
        z_sem:       语义特征，shape (B, D)
        labels:      多标签矩阵，shape (B, num_classes)，值 ∈ {0, 1}
        temperature: 对比温度 τ（默认 0.1）。越小 → 对相似度差异越敏感

    返回:
        loss: 标量对比损失
    """
    B = z_sem.size(0)
    if B < 2:
        return torch.tensor(0.0, device=z_sem.device)

    # L2 归一化 → 余弦相似度空间（相似度 = 归一化向量的内积）
    z_norm = F.normalize(z_sem, p=2, dim=1)             # (B, D)

    # 相似度矩阵除以温度 → 温度控制"锐度"
    # τ 小：高相似度和低相似度的差距被放大（更"尖锐"）
    # τ 大：差距被缩小（更"平滑"）
    sim = torch.mm(z_norm, z_norm.T) / temperature       # (B, B)

    # 正样本掩码：用 Jaccard 相似度判断两个样本是否"同类"
    labels_float = labels.float()
    # intersection[i][j] = 样本i和样本j共同拥有的疾病数
    intersection = torch.mm(labels_float, labels_float.T)  # (B, B)
    # union[i][j] = 样本i的疾病数 + 样本j的疾病数 - 共同的疾病数
    union = (labels_float.sum(1, keepdim=True)
             + labels_float.sum(1, keepdim=True).T
             - intersection)
    # Jaccard = 交集 / 并集
    jaccard = intersection / (union + 1e-8)

    # Jaccard > 0.5 → 视为同类 → 正样本对
    pos_mask = (jaccard > 0.5).float()
    pos_mask.fill_diagonal_(0)  # 自己和自己的相似度不算正样本

    # InfoNCE 损失计算
    exp_sim = torch.exp(sim)                              # (B, B)

    # 分母 = 对所有负样本（+ 正样本）的 exp(sim) 求和（排除自身）
    denom = exp_sim.sum(dim=1) - torch.exp(sim.diag())   # (B,)

    # 分子 = 对正样本的 exp(sim) 求和
    num_pos = pos_mask.sum(dim=1)                         # (B,)  每个样本有几个正样本
    pos_sim_sum = (exp_sim * pos_mask).sum(dim=1)         # (B,)  正样本的 exp(sim) 之和

    # 只对有正样本的样本计算损失（没有正样本的跳过）
    valid = num_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z_sem.device)

    # -log(分子/分母) → InfoNCE 损失
    loss = -torch.log(
        (pos_sim_sum[valid] / (denom[valid] + 1e-8)) + 1e-8
    ).mean()

    return loss


# =============================================================================
#  对抗域不变损失 — L_adv
# =============================================================================

def adversarial_disentanglement_loss(domain_logits, target_labels=None):
    """对抗域分类损失 — 语义特征应"骗过"域分类器

    这个损失配合 GradientReversal 使用：
    - GRL 在前向时原样传递语义特征
    - 域分类器尝试从语义特征判断"来自哪个客户端"
    - GRL 在反向时反转梯度，让语义特征朝着"让分类器猜错"的方向更新

    目标标签的设置：
    - target=0（默认）：希望分类器输出 0（无法区分 = 50%概率）
      这意味着语义特征成功"洗掉"了域信息
    - 如果 target 不为 0：则是标准域分类（用于判别域差异，而非消除）

    和 disentanglement_loss 的关系：
    - HSIC 确保"统计独立性"（线性不相关）
    - 对抗损失确保"对抗不变性"（非线性也无法区分）
    - 二者互补，提供更强的解耦保证

    参数:
        domain_logits: 域分类器输出，shape (B, 1) 或 (B, num_domains)
        target_labels: 目标标签。None 时默认全零（希望分类器无法区分）

    返回:
        loss: 标量对抗损失
    """
    if target_labels is None:
        # 默认：希望分类器输出全是 0（猜不出来源域）
        target_labels = torch.zeros_like(domain_logits)
    # BCEWithLogitsLoss = sigmoid + binary cross entropy
    # 当 domain_logits → 0（分类器猜不出来），loss → 最小
    return F.binary_cross_entropy_with_logits(domain_logits, target_labels)
