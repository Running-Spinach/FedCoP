# ResNet-50 骨干网络，适配 ChestX-ray14 多标签分类与原型提取
# 架构：ResNet-50 → fc1(2048→proto_dim) → [ProtoHead] → fc2(proto_dim→14)
# 原型来自 fc1 输出（点原型）或 ProtoHead 输出（分布原型）
#
# FedProto 基线: ResNet50 (from scratch, Kaiming init)
# D²-FL 提出方法: D2FLResNet (ImageNet pretrained backbone)

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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积层（Bottleneck 中用于通道压缩和扩展）"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 卷积层（Bottleneck 中的空间特征提取）"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    """ResNet-50 瓶颈残差块（1x1→3x3→1x1 结构 + 恒等映射）

    参数:
        in_planes: 输入通道数
        planes: 瓶颈中间通道数（输出通道 = planes * expansion）
        stride: 3x3卷积步长（用于下采样）
        downsample: 恒等映射的1x1卷积下采样模块（维度不匹配时使用）
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """前向传播：x → 1x1(压缩) → 3x3(空间) → 1x1(扩展) → + 恒等映射 → ReLU"""
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    """ResNet-50 骨干网络，适配 FedProto 原型提取

    输出格式（取决于 use_distributional）：
      - 点原型: (logits, protos)       — protos 来自 fc1 输出，shape (B, proto_dim)
      - 分布原型: (logits, mu, logvar) — 来自 ProbabilisticProtoHead，shape (B, proto_dim)

    ImageNet 预训练：当 pretrained=True 时，骨干层（conv1/bn1/layer1~4）
    从 torchvision 预训练权重初始化，fc1/fc2/proto_head 随机初始化。
    """

    def __init__(self, args):
        """
        参数:
            args: 配置对象，需包含 num_classes, proto_dim, use_distributional,
                  pretrained 等字段
        """
        super().__init__()
        self.in_planes = 64
        self.num_classes = args.num_classes
        self.proto_dim = getattr(args, 'proto_dim', None) or 256
        self.use_distributional = getattr(args, 'use_distributional', False)
        pretrained = getattr(args, 'pretrained', True)

        # ── ResNet-50 骨干 ──
        # 输入 3 通道（ChestX-ray14 灰度图通过 transform 转为 3 通道）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))#输出 (B, 2048, 1, 1) → flatten → (B, 2048)

        # ── 原型提取层（类比 CNNMnist 的 fc1）──
        self.fc1 = nn.Linear(512 * Bottleneck.expansion, self.proto_dim)

        # ── 分布原型头（可选）──
        if self.use_distributional:
            self.proto_head = ProbabilisticProtoHead(self.proto_dim, proto_dim=self.proto_dim)
        else:
            self.proto_head = None

        # ── 分类头（输出 14 维 logits，配合 BCEWithLogitsLoss）──
        self.fc2 = nn.Linear(self.proto_dim, self.num_classes)

        # ── 加载预训练权重 ──
        if pretrained:
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
            pretrained_model = tv_models.resnet50(weights=weights)
            pretrained_dict = pretrained_model.state_dict()
            # 仅加载骨干层（conv1, bn1, layer1~layer4），跳过 fc（ImageNet 1000 类）
            # fc1, fc2, proto_head 保持随机初始化
            self.load_state_dict(pretrained_dict, strict=False)
        else:
            self._init_weights()

    def _make_layer(self, planes, blocks, stride):
        """构建一个 ResNet 残差层（包含 blocks 个 Bottleneck 块）

        参数:
            planes: 瓶颈中间通道数
            blocks: 该层中 Bottleneck 块的数量
            stride: 第一个 Bottleneck 的步长（控制空间下采样）

        返回:
            nn.Sequential: 组合后的残差层
        """
        downsample = None
        if stride != 1 or self.in_planes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = [Bottleneck(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Kaiming 正态初始化卷积层，BatchNorm 初始化为 gamma=1, beta=0"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播：骨干 → fc1(原型特征) → [ProtoHead] → fc2(分类 logits)

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
        x = torch.flatten(x, 1)          # (B, 2048) 保留第 0 维，从第 1 维开始往后所有维度合并成一维。

        # 原型空间
        proto_features = F.relu(self.fc1(x))  # (B, proto_dim)

        # 分类
        logits = self.fc2(proto_features)     # (B, num_classes)

        if self.use_distributional and self.proto_head is not None:
            mu, logvar = self.proto_head(proto_features)
            return logits, mu, logvar
        else:
            return logits, proto_features


class D2FLResNet(nn.Module):
    """D²-FL 专用模型：ImageNet 预训练 ResNet-50 + 增强原型头 + 分类头

    核心创新（相比 FedProto 基线）：
      1. 端到端可学习分布原型 — 深度 ProtoHead 输出 N(μ, σ²)，校准初始化
      2. 语义-风格解耦 — 可学习门控 + 对抗域不变 + 对比语义对齐
      3. 每类自适应温度 — 端到端学习最佳推理锐度

    输出格式（取决于 use_distributional 和 use_disentangle）：
      - 点原型: (logits, protos)
      - 分布原型: (logits, mu, logvar)
      - 解耦+点原型: (logits, z_sem, z_style) 或带 gate
      - 解耦+分布原型: (logits, mu_sem, logvar_sem, mu_style, logvar_style) 或带 gate
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
        pretrained = getattr(args, 'pretrained', True)
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tv_models.resnet50(weights=weights)

        # 分离 stem + layers + avgpool
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc_backbone = backbone.fc

        # ── 原型提取层 ──
        self.fc1 = nn.Linear(2048, self.proto_dim)

        # ── 增强分布原型头（可选）──
        if self.use_distributional:
            self.proto_head = ProbabilisticProtoHead(
                self.proto_dim, proto_dim=self.proto_dim,
                hidden_dim=self.proto_dim)
        else:
            self.proto_head = None

        # ── 分类头 ──
        self.fc2 = nn.Linear(self.proto_dim, self.num_classes)

        # ── 增强原型解耦（D²-FL 专属）──
        if self.use_disentangle:
            sem_ratio = getattr(args, 'sem_ratio', 0.75)
            self.dis_head = EnhancedDisentangledProtoHead(
                proto_dim=self.proto_dim,
                sem_ratio=sem_ratio,
                use_distributional=self.use_distributional,
            )
        else:
            self.dis_head = None

        # ── 每类可学习温度（用于原型推理）──
        use_per_class_temp = getattr(args, 'use_per_class_temp', True)
        if use_per_class_temp:
            init_temp = getattr(args, 'temperature', 1.0)
            self.class_temp = PerClassTemperature(self.num_classes, init_temp=init_temp)
        else:
            self.class_temp = None

    def forward(self, x, return_gate=False):
        """前向传播：预训练骨干 → fc1(原型特征) → [ProtoHead/DisHead] → fc2(分类 logits)

        参数:
            x: 输入图像 (B, C, H, W)
            return_gate: 是否返回门控值（用于解耦正则化损失）

        返回:
            解耦+分布: (logits, mu_sem, logvar_sem, mu_style, logvar_style)  [+ gate]
            解耦+点: (logits, z_sem, z_style)  [+ gate]
            分布: (logits, mu, logvar)
            点: (logits, proto_features)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)          # (B, 2048)

        proto_features = F.relu(self.fc1(x))  # (B, proto_dim)

        if self.use_disentangle and self.dis_head is not None:
            # 增强解耦模式：可学习门控 + 对抗域不变 + 对比语义对齐
            result = self.dis_head(proto_features, return_gate=return_gate)
            logits = self.fc2(proto_features)
            return (logits,) + result
        else:
            # 原始分布原型模式
            logits = self.fc2(proto_features)
            if self.use_distributional and self.proto_head is not None:
                mu, logvar = self.proto_head(proto_features)
                return logits, mu, logvar
            else:
                return logits, proto_features

    def forward_adversarial(self, z_sem, grad_reverse_lambda=1.0):
        """对抗域分类前向：对语义特征做梯度反转后预测域标签

        用于训练时检测语义特征是否泄漏了域信息。
        """
        if self.dis_head is not None:
            return self.dis_head.forward_adversarial(z_sem, grad_reverse_lambda)
        return None

    def get_class_temperatures(self, class_indices=None):
        """获取每类可学习温度参数

        返回:
            temperatures: shape (num_classes,) 或 (len(class_indices),)
        """
        if self.class_temp is not None:
            return self.class_temp(class_indices)
        return torch.ones(self.num_classes)  # fallback: 均匀温度
