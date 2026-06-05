# ResNet-50 骨干网络，适配 ChestX-ray14 多标签分类与原型提取
# 架构：ResNet-50 → fc1(2048→proto_dim) → [ProtoHead] → fc2(proto_dim→14)
# 原型来自 fc1 输出（点原型）或 ProtoHead 输出（分布原型）
#
# FedProto 基线: ResNet50 (from scratch, Kaiming init)
# DPP-FL 提出方法: DPPFLResNet (ImageNet pretrained backbone)

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


class DPPFLResNet(nn.Module):
    """DPP-FL 专用模型：ImageNet 预训练 ResNet-50 + 原型头 + 分类头

    与 FedProto 基线的关键区别：
      - 使用 torchvision 预训练权重（ImageNet），大幅提升有限医学数据下的特征质量
      - 仅微调 layer3/layer4 + 原型/分类头（stem + layer1/layer2 冻结）
      - 支持分布原型 ProbabilisticProtoHead
      - 可选原型解耦 (DisentangledProtoHead)，分离语义与风格特征

    输出格式（取决于 use_distributional 和 use_disentangle）：
      - 点原型: (logits, protos)
      - 分布原型: (logits, mu, logvar)
      - 解耦+点原型: (logits, z_sem, z_style)
      - 解耦+分布原型: (logits, mu_sem, logvar_sem, mu_style, logvar_style)
    """

    def __init__(self, args):
        """
        参数:
            args: 配置对象，需包含 num_classes, proto_dim, use_distributional,
                  use_disentangle, pretrained 等字段
        """
        super().__init__()
        self.num_classes = args.num_classes
        self.proto_dim = getattr(args, 'proto_dim', None) or 256
        self.use_distributional = getattr(args, 'use_distributional', False)

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
        self.fc_backbone = backbone.fc  # 保留引用，实际不用

        # ── 原型提取层 ──
        self.fc1 = nn.Linear(2048, self.proto_dim)

        # ── 分布原型头（可选）──
        if self.use_distributional:
            self.proto_head = ProbabilisticProtoHead(self.proto_dim, proto_dim=self.proto_dim)
        else:
            self.proto_head = None

        # ── 分类头 ──
        self.fc2 = nn.Linear(self.proto_dim, self.num_classes)

        # ── 原型解耦（DPP-FL 专属）──
        self.use_disentangle = getattr(args, 'use_disentangle', False)
        if self.use_disentangle:
            from dist_proto.disentangle import DisentangledProtoHead
            sem_ratio = getattr(args, 'sem_ratio', 0.75)
            self.dis_head = DisentangledProtoHead(
                proto_dim=self.proto_dim,
                sem_ratio=sem_ratio,
                use_distributional=self.use_distributional,
            )
        else:
            self.dis_head = None

    def forward(self, x):
        """前向传播：预训练骨干 → fc1(原型特征) → [ProtoHead/DisHead] → fc2(分类 logits)

        返回:
            解耦+分布: (logits, mu_sem, logvar_sem, mu_style, logvar_style)
            解耦+点: (logits, z_sem, z_style)
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
            # 解耦模式：仅共享语义原型，风格原型保留本地
            if self.use_distributional:
                (_mu_full, _logvar_full,
                 mu_sem, logvar_sem,
                 mu_style, logvar_style) = self.dis_head(proto_features)
                logits = self.fc2(proto_features)
                return logits, mu_sem, logvar_sem, mu_style, logvar_style
            else:
                _z_full, z_sem, z_style = self.dis_head(proto_features)
                logits = self.fc2(proto_features)
                return logits, z_sem, z_style
        else:
            # 原始模式
            logits = self.fc2(proto_features)
            if self.use_distributional and self.proto_head is not None:
                mu, logvar = self.proto_head(proto_features)
                return logits, mu, logvar
            else:
                return logits, proto_features
