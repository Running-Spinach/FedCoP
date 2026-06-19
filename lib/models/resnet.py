# =============================================================================
# 模型定义:ResNet-50 骨干,适配 ChestX-ray14 多标签分类与原型提取
# =============================================================================
# 两个模型类:
#   1. ResNet50     — 基线模型(FedAvg/FedProto/FedProx/FedGMKD/FedBCS/FedSeProto 用)
#   2. FedCoPResNet — FedCoP 提出方法(ImageNet 预训练 + 概率原型头)
#
# FedCoPResNet 的设计刻意精简:骨干 → fc1(原型特征)→ ProbabilisticProtoHead(μ,logvar)
#   + fc2(分类 logits)。forward 只返回 (logits, mu, logvar) 单一模式。
# 跨类的"共现结构"不在这里建模——那由 lib/dist_proto/structured.py 的 R̂ 负责。
#
# 输出格式:
#   ResNet50     点原型: (logits, proto_features)  /  分布: (logits, mu, logvar)
#   FedCoPResNet 分布原型: (logits, mu, logvar)     —— 始终分布,单一模式
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


# ═══════════════════════════════════════════════════════════════════════════════
#  基础卷积层
# ═══════════════════════════════════════════════════════════════════════════════

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积 — Bottleneck 中用于通道压缩/扩展"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 卷积 — Bottleneck 中唯一做空间操作的卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  ResNet-50 瓶颈残差块
# ═══════════════════════════════════════════════════════════════════════════════

class Bottleneck(nn.Module):
    """ResNet-50 瓶颈残差块:1x1(压缩) → 3x3(空间) → 1x1(扩展) + 恒等映射"""

    expansion = 4  # 输出通道 = planes × 4

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
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  ResNet50 — 基线模型
# ═══════════════════════════════════════════════════════════════════════════════

class ResNet50(nn.Module):
    """ResNet-50 骨干,基线算法(FedAvg/FedProto/FedProx 等)使用

    架构:Stem → layer1-4 → avgpool(2048) → fc1(proto_dim) → [ProtoHead] → fc2(14)
    可选分布原型头(基线若开启 --use_distributional 时用)。

    参数:
        args: 含 num_classes, proto_dim, use_distributional, pretrained 等
    """

    def __init__(self, args):
        super().__init__()
        self.in_planes = 64
        self.num_classes = args.num_classes
        self.proto_dim = getattr(args, 'proto_dim', None) or 256
        self.use_distributional = getattr(args, 'use_distributional', False)
        pretrained = getattr(args, 'pretrained', True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * Bottleneck.expansion, self.proto_dim)

        if self.use_distributional:
            self.proto_head = ProbabilisticProtoHead(self.proto_dim, proto_dim=self.proto_dim)
        else:
            self.proto_head = None

        self.fc2 = nn.Linear(self.proto_dim, self.num_classes)

        if pretrained:
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
            pretrained_model = tv_models.resnet50(weights=weights)
            self.load_state_dict(pretrained_model.state_dict(), strict=False)
        else:
            self._init_weights()

    def _make_layer(self, planes, blocks, stride):
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
        """Kaiming 正态初始化(不加载预训练时)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向:骨干 → fc1(原型特征) → [ProtoHead] → fc2(分类 logits)

        返回:
            点原型模式: (logits, proto_features)
            分布原型模式: (logits, mu, logvar)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)            # (B, 2048)

        proto_features = F.relu(self.fc1(x))  # (B, proto_dim)
        logits = self.fc2(proto_features)     # (B, num_classes)

        if self.use_distributional and self.proto_head is not None:
            mu, logvar = self.proto_head(proto_features)
            return logits, mu, logvar
        else:
            return logits, proto_features


# ═══════════════════════════════════════════════════════════════════════════════
#  FedCoPResNet — FedCoP 提出方法
# ═══════════════════════════════════════════════════════════════════════════════

class FedCoPResNet(nn.Module):
    """FedCoP 专用模型:ImageNet 预训练 ResNet-50 + 概率原型头

    设计刻意精简(砍掉了旧 D²-FL 的解耦头/对抗分类器/每类温度等冗余组件):
        [输入 224×224×3]
            │
            ▼
        ImageNet 预训练 ResNet-50(stem + layer1-4 + avgpool) → (B, 2048)
            │
            ▼
        fc1(2048 → proto_dim) + ReLU → (B, proto_dim)  ← 原型特征
            │
            ├──→ fc2(proto_dim → 14) → logits(分类损失用)
            │
            └──→ ProbabilisticProtoHead → (μ, logvar)  ← 分布原型(共享/聚合用)

    跨类共现结构不在此模型内,由服务器端的共现相关矩阵 R̂(structured.py)建模,
    并通过 L_co(训练侧)和 mean-field 解码(推理侧)作用于原型。

    参数:
        args: 含 num_classes, proto_dim, pretrained 等
    """

    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes
        self.proto_dim = getattr(args, 'proto_dim', None) or 256

        # ── 加载 ImageNet 预训练 ResNet-50,分离各层 ──
        pretrained = getattr(args, 'pretrained', True)
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tv_models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # 原型提取层 + 概率原型头 + 分类头
        self.fc1 = nn.Linear(2048, self.proto_dim)
        self.proto_head = ProbabilisticProtoHead(
            self.proto_dim, proto_dim=self.proto_dim, hidden_dim=self.proto_dim)
        self.fc2 = nn.Linear(self.proto_dim, self.num_classes)

    def forward(self, x):
        """前向:预训练骨干 → fc1 → ProtoHead → fc2

        参数:
            x: (B, 3, 224, 224)

        返回:(logits, mu, logvar)
            logits: (B, num_classes) 分类 logits(BCE 用)
            mu:     (B, proto_dim) 分布原型均值(对齐/聚合/推理用)
            logvar: (B, proto_dim) 分布原型对数方差(不确定性/贝叶斯融合用)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)            # (B, 2048)

        proto_features = F.relu(self.fc1(x))   # (B, proto_dim)
        logits = self.fc2(proto_features)      # (B, num_classes)
        mu, logvar = self.proto_head(proto_features)
        return logits, mu, logvar
