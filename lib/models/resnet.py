# =============================================================================
# 模型定义：ResNet-50 骨干网络，适配 ChestX-ray14 多标签分类与原型提取
# =============================================================================
# 包含两个模型类：
#   1. ResNet50    — FedProto 基线模型（从头训练，Kaiming 初始化）
#   2. D2FLResNet  — D²-FL 提出方法（ImageNet 预训练 + 增强原型头）
#
# 两个模型的核心区别：
#   - ResNet50：纯从头训练，原型来自 fc1 输出（点原型）或浅层 ProtoHead（分布原型）
#   - D2FLResNet：ImageNet 预训练骨干 + 深度 ProtoHead + 可选的语义-风格解耦头 +
#                 每类可学习温度。相当于给 FedProto 加了三层"buff"
#
# 输出格式（取决于 use_distributional 和 use_disentangle 的组合）：
#
#   配置                       | 输出元组内容
#   ────────────────────────  | ──────────────────────────────────────
#   点原型（基线）              | (logits, proto_features)
#   分布原型                   | (logits, mu, logvar)
#   解耦 + 点原型              | (logits, z_full, z_sem, z_style)  [+ gate]
#   解耦 + 分布原型            | (logits, mu_full, logvar_full,
#                              |   mu_sem, logvar_sem, mu_style, logvar_style)  [+ gate]
#
# 其中：
#   - logits：分类输出 (B, 14)，配合 BCEWithLogitsLoss 做多标签分类
#   - proto_features / mu / z_sem：用于原型聚合的"共享特征"
#   - gate：门控值，仅在 return_gate=True 时返回，用于训练时的解耦正则化
# =============================================================================

import sys
from pathlib import Path
lib_dir = (Path(__file__).parent / "..").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from dist_proto import ProbabilisticProtoHead
from dist_proto.proto_head import PerClassTemperature
from dist_proto.disentangle import DisentangledProtoHead as EnhancedDisentangledProtoHead


# ═══════════════════════════════════════════════════════════════════════════════
#  基础卷积层
# ═══════════════════════════════════════════════════════════════════════════════

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积 — Bottleneck 中用于通道压缩和扩展

    在 ResNet Bottleneck 中，1x1 卷积承担两个角色：
    - 第一个 1x1：把通道数从 in_planes 压缩到 planes（降维，节省计算）
    - 第三个 1x1：把通道数从 planes 扩展到 planes*4（升维，恢复表达力）
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 卷积 — Bottleneck 中的空间特征提取

    这是 ResNet 中唯一做空间操作的卷积层（1x1 只做通道变换）。
    padding=1 保持空间尺寸不变（当 stride=1 时）。
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  ResNet-50 瓶颈残差块
# ═══════════════════════════════════════════════════════════════════════════════

class Bottleneck(nn.Module):
    """ResNet-50 瓶颈残差块

    结构：1x1(压缩) → 3x3(空间特征) → 1x1(扩展) → + 恒等映射 → ReLU

    这种"窄-宽-窄"的设计（Bottleneck = 瓶颈）的优势：
    - 先压缩通道 → 减少 3x3 卷积的计算量（3x3 在低维空间运行）
    - 再扩展回去 → 恢复高维特征表达力
    - 相比直接在高维做 3x3 卷积，计算量大幅降低

    参数说明：
        in_planes: 输入通道数
        planes:    瓶颈中间通道数（3x3 卷积的通道数）
        stride:    3x3 卷积的步长。stride=2 时特征图尺寸减半。
        downsample: 当输入输出维度不匹配时，对恒等映射做 1x1 卷积对齐维度。

    维度变化：
        输入: (B, in_planes, H, W)
        conv1: (B, planes, H, W)          — 压缩
        conv2: (B, planes, H/s, W/s)      — 空间处理（s=stride）
        conv3: (B, planes*4, H/s, W/s)    — 扩展到 4×planes
        输出: (B, planes*4, H/s, W/s)
    """
    expansion = 4  # 输出通道 = planes × 4，这是 ResNet-50 的标准设置

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 恒等映射的下采样模块
        self.stride = stride

    def forward(self, x):
        """前向传播：主路径 → + 恒等映射 → ReLU

        残差连接（恒等映射）的本质：
        - 主路径学习"需要改变的部分"
        - 恒等映射保留"不需要改变的部分"
        - 加起来 = 在原始特征基础上做微调
        - 这解决了深层网络的梯度消失问题
        """
        identity = x  # 保存恒等映射

        # 主路径：1x1 → 3x3 → 1x1
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # 如果维度变化，对恒等映射也做相应变换
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接："主路径的输出" + "原始输入"
        out += identity
        out = self.relu(out)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  ResNet-50 — FedProto 基线模型
# ═══════════════════════════════════════════════════════════════════════════════

class ResNet50(nn.Module):
    """ResNet-50 骨干网络，适配 FedProto 原型提取

    这是 FedProto 基线使用的模型。从头训练（Kaiming 初始化），
    或加载 ImageNet 预训练权重（仅骨干层）。

    架构概览：
        [输入 224×224×3]
            │
            ▼
        conv1 + bn1 + relu + maxpool     → (B, 64, 56, 56)
            │
            ▼
        layer1 (3×Bottleneck, 64→256)    → (B, 256, 56, 56)
        layer2 (4×Bottleneck, 256→512)   → (B, 512, 28, 28)
        layer3 (6×Bottleneck, 512→1024)  → (B, 1024, 14, 14)
        layer4 (3×Bottleneck, 1024→2048) → (B, 2048, 7, 7)
            │
            ▼
        AdaptiveAvgPool2d(1,1) → flatten → (B, 2048)
            │
            ▼
        fc1 (2048 → proto_dim)           → (B, proto_dim)  ← 原型特征
            │
            ├── [可选] ProtoHead → (μ, logvar)  ← 分布原型
            │
            ▼
        fc2 (proto_dim → num_classes)    → (B, 14)        ← 分类 logits

    参数:
        args: 配置对象，包含 num_classes, proto_dim, use_distributional, pretrained 等
    """

    def __init__(self, args):
        super().__init__()
        self.in_planes = 64  # 第一个 Bottleneck 的输入通道数
        self.num_classes = args.num_classes
        self.proto_dim = getattr(args, 'proto_dim', None) or 256
        self.use_distributional = getattr(args, 'use_distributional', False)
        pretrained = getattr(args, 'pretrained', True)

        # ── 初始卷积层（Stem）──
        # ChestX-ray14 的灰度图在 transform 中已转为 3 通道（Grayscale(3)）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── 四个残差层 ──
        # 参数依次为：瓶颈中间通道数, 块数量, 第一个块的步长
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        # ── 全局池化 ──
        # 输出 (B, 2048, 1, 1) → flatten → (B, 2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ── 原型提取层 ──
        # 这是联邦学习的关键：fc1 的输出被用作"原型特征"
        self.fc1 = nn.Linear(512 * Bottleneck.expansion, self.proto_dim)

        # ── 分布原型头（可选）──
        if self.use_distributional:
            self.proto_head = ProbabilisticProtoHead(self.proto_dim, proto_dim=self.proto_dim)
        else:
            self.proto_head = None

        # ── 分类头 ──
        # 输出 14 维 logits，配合 BCEWithLogitsLoss（多标签分类）
        self.fc2 = nn.Linear(self.proto_dim, self.num_classes)

        # ── 加载预训练权重 ──
        if pretrained:
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
            pretrained_model = tv_models.resnet50(weights=weights)
            pretrained_dict = pretrained_model.state_dict()
            # strict=False：只加载匹配的层（骨干层），fc1/fc2/proto_head 随机初始化
            self.load_state_dict(pretrained_dict, strict=False)
        else:
            self._init_weights()

    def _make_layer(self, planes, blocks, stride):
        """构建一个 ResNet 残差层

        一个"层"包含 blocks 个 Bottleneck 块。第一个块的 stride 控制空间下采样，
        其余块的 stride=1（保持尺寸不变）。

        参数:
            planes: 瓶颈中间通道数
            blocks: 该层中 Bottleneck 块的数量（ResNet-50: [3, 4, 6, 3]）
            stride: 第一个 Bottleneck 的步长。stride=2 时特征图尺寸减半。

        返回:
            nn.Sequential: 组合后的残差层
        """
        downsample = None
        # 如果维度不匹配，需要加一个 1x1 卷积来对齐恒等映射的维度
        if stride != 1 or self.in_planes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        # 第一个块（可能带下采样）
        layers = [Bottleneck(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * Bottleneck.expansion  # 更新输入通道数
        # 剩余块（无下采样）
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Kaiming 正态初始化（不加载预训练时使用）

        Kaiming 初始化专为 ReLU 设计：
        - 权重：N(0, sqrt(2/fan_out))，保持 ReLU 后的方差稳定
        - BatchNorm：gamma=1, beta=0（恒等映射开始）
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播：骨干 → fc1(原型特征) → [ProtoHead] → fc2(分类 logits)

        参数:
            x: 输入图像，shape (B, 3, 224, 224)

        返回:
            点原型模式: (logits, proto_features)
            分布原型模式: (logits, mu, logvar)
        """
        # Backbone
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)            # (B, 2048)

        # 原型空间
        proto_features = F.relu(self.fc1(x))  # (B, proto_dim)

        # 分类
        logits = self.fc2(proto_features)     # (B, num_classes)

        if self.use_distributional and self.proto_head is not None:
            # 分布原型模式：输出高斯参数
            mu, logvar = self.proto_head(proto_features)
            return logits, mu, logvar
        else:
            # 点原型模式：直接输出原型特征
            return logits, proto_features


# ═══════════════════════════════════════════════════════════════════════════════
#  D2FLResNet — D²-FL 专用模型（提出方法）
# ═══════════════════════════════════════════════════════════════════════════════

class D2FLResNet(nn.Module):
    """D²-FL 专用模型：ImageNet 预训练 ResNet-50 + 增强原型头 + 解耦头

    D2FLResNet 是 D²-FL 的"硬件"，和 ResNet50（FedProto 基线）相比，
    多了三个关键组件：

    1. ProbabilisticProtoHead（增强版）
       — 更深的隐藏层 + LayerNorm + 校准初始化
       — 输出高质量的高斯原型参数 N(μ, σ²)

    2. DisentangledProtoHead（解耦头）
       — 可学习门控：自动区分语义和风格特征
       — 对抗域分类器：确保语义特征不含域信息
       — 支持分布原型参数化

    3. PerClassTemperature（每类温度）
       — 推理时对每个疾病类别使用不同的温度缩放
       — 端到端学习：网络自己决定每个类的"锐度"

    完整架构：
        [输入 224×224×3]
            │
            ▼
        ┌─ ImageNet 预训练 ResNet-50 ─┐
        │  Stem (conv1+bn1+relu+mp)   │
        │  layer1 → layer2 → layer3 → layer4
        │  avgpool → flatten (2048维)  │
        └─────────────────────────────┘
            │
            ▼
        fc1 (2048 → proto_dim) + ReLU  → (B, proto_dim)
            │
            ├──→ fc2 (proto_dim → 14)  → logits（分类损失用）
            │
            ├──→ [DisentangledProtoHead]  → 语义/风格分离特征（解耦损失用）
            │     ├── LearnableGate
            │     ├── [ProbabilisticProtoHead(sem)]
            │     ├── [ProbabilisticProtoHead(style)]
            │     └── DomainClassifier（对抗训练用）
            │
            └──→ [ProbabilisticProtoHead]  → (μ, logvar)（无解耦时）

    关键设计理念：
    - fc2 始终接收全维度 fc1 特征做分类（保证分类性能不因解耦下降）
    - 解耦仅影响原型聚合：只上传语义原型，过滤掉风格噪声
    - 对抗训练通过 GRL 反向，不影响分类任务的正向梯度
    """

    def __init__(self, args):
        """
        参数:
            args: 配置对象，需包含 num_classes, proto_dim, use_distributional,
                  use_disentangle, pretrained, sem_ratio 等字段
        """
        super().__init__()
        self.num_classes = args.num_classes
        self.proto_dim = getattr(args, 'proto_dim', None) or 256
        self.use_distributional = getattr(args, 'use_distributional', False)
        self.use_disentangle = getattr(args, 'use_disentangle', False)

        # ── 加载 ImageNet 预训练 ResNet-50 ──
        # 使用 torchvision 的官方预训练模型，分开保存各层以便后续访问
        pretrained = getattr(args, 'pretrained', True)
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tv_models.resnet50(weights=weights)

        # 分离 stem（初始层）+ 四个残差层 + 池化
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc_backbone = backbone.fc  # 原始 ImageNet 的 fc（1000类），保留但不直接使用

        # ── 原型提取层 ──
        # 2048 = ResNet-50 最终特征维度（512 × Bottleneck.expansion(4) = 2048）
        self.fc1 = nn.Linear(2048, self.proto_dim)

        # ── 增强分布原型头（可选）──
        # 仅在未启用解耦时使用（解耦模式下，其内部的 sem_head/style_head 替代此功能）
        if self.use_distributional:
            self.proto_head = ProbabilisticProtoHead(
                self.proto_dim, proto_dim=self.proto_dim,
                hidden_dim=self.proto_dim)
        else:
            self.proto_head = None

        # ── 分类头 ──
        # 始终从全维度 fc1 特征分类，保证分类性能
        self.fc2 = nn.Linear(self.proto_dim, self.num_classes)

        # ── 增强原型解耦（D²-FL 专属组件）──
        if self.use_disentangle:
            sem_ratio = getattr(args, 'sem_ratio', 0.75)
            # DisentangledProtoHead 内部包含：
            #   LearnableGate（软门控）
            #   DomainClassifier（对抗域分类器）
            #   [可选] ProbabilisticProtoHead × 2（语义和风格分布参数化）
            self.dis_head = EnhancedDisentangledProtoHead(
                proto_dim=self.proto_dim,
                sem_ratio=sem_ratio,
                use_distributional=self.use_distributional,
            )
        else:
            self.dis_head = None

        # ── 每类可学习温度（用于原型推理）──
        # 推理时：logit[class_j] = -dist(proto, global_proto[j]) / T_j
        # T_j 通过梯度反向传播学习最优值
        use_per_class_temp = getattr(args, 'use_per_class_temp', True)
        if use_per_class_temp:
            init_temp = getattr(args, 'temperature', 1.0)
            self.class_temp = PerClassTemperature(self.num_classes, init_temp=init_temp)
        else:
            self.class_temp = None

    def forward(self, x, return_gate=False):
        """前向传播：预训练骨干 → fc1 → [ProtoHead/DisHead] → fc2

        参数:
            x: 输入图像，shape (B, 3, 224, 224)
            return_gate: 是否返回门控值。训练时需要（用于解耦正则化损失），
                        推理时不需要（减少计算开销）。

        返回:
            根据配置返回不同内容的元组，详见文件头部的输出格式表。

            注意：无论何种模式，返回元组的第一个元素始终是 logits。
        """
        # ── 步骤1: 骨干网络提取特征 ──
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)            # (B, 2048)

        # ── 步骤2: fc1 → 原型特征 ──
        proto_features = F.relu(self.fc1(x))  # (B, proto_dim)

        # ── 步骤3: 根据模式分叉处理 ──
        if self.use_disentangle and self.dis_head is not None:
            # 解耦模式：门控分离 → 语义/风格/全维度
            # 注意：fc2 仍用全维度 proto_features 做分类（不解耦分类）
            result = self.dis_head(proto_features, return_gate=return_gate)
            logits = self.fc2(proto_features)
            return (logits,) + result
        else:
            # 非解耦模式：直接分类 + 可选分布参数化
            logits = self.fc2(proto_features)
            if self.use_distributional and self.proto_head is not None:
                # 分布原型：输出高斯参数
                mu, logvar = self.proto_head(proto_features)
                return logits, mu, logvar
            else:
                # 点原型：直接输出特征
                return logits, proto_features

    def forward_adversarial(self, z_sem, grad_reverse_lambda=1.0):
        """对抗域分类前向：语义特征 → GRL → 域分类器

        这个函数仅在训练时调用，用于计算对抗域不变损失（L_adv）。
        通过梯度反转，迫使语义特征变得"无法被域分类器识别来源"。

        工作原理：
        1. 前向：语义特征正常通过域分类器，预测域标签
        2. 反向：梯度反转 (× -λ)，"欺骗"语义特征朝着反方向更新
        3. 效果：语义特征越来越"域无关"，域分类器准确率趋向随机水平

        参数:
            z_sem: 语义特征，shape (B, proto_dim)
            grad_reverse_lambda: 梯度反转强度。λ=1.0 为完全反转。

        返回:
            domain_logits: 域分类 logits，shape (B, 1)
        """
        if self.dis_head is not None:
            return self.dis_head.forward_adversarial(z_sem, grad_reverse_lambda)
        return None

    def get_class_temperatures(self, class_indices=None):
        """获取每类可学习温度参数

        用于原型推理时的温度缩放。温度在训练过程中端到端学习。

        参数:
            class_indices: 类别索引（可选）。None 返回全部类别。

        返回:
            temperatures: shape (num_classes,) 或 (len(class_indices),)
        """
        if self.class_temp is not None:
            return self.class_temp(class_indices)
        # fallback：如果未启用可学习温度，返回全1（即不做缩放）
        return torch.ones(self.num_classes)
