# ResNet-50 骨干网络，适配 ChestX-ray14 多标签分类与原型提取
# 架构：ResNet-50 → fc1(2048→proto_dim) → [ProtoHead] → fc2(proto_dim→14)
# 原型来自 fc1 输出（点原型）或 ProtoHead 输出（分布原型）

import sys
from pathlib import Path
lib_dir = (Path(__file__).parent / "..").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
from dist_proto import ProbabilisticProtoHead


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
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
    """

    def __init__(self, args):
        super().__init__()
        self.in_planes = 64
        self.num_classes = args.num_classes
        self.proto_dim = getattr(args, 'proto_dim', None) or 256
        self.use_distributional = getattr(args, 'use_distributional', False)

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

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ── 原型提取层（类比 CNNMnist 的 fc1）──
        self.fc1 = nn.Linear(512 * Bottleneck.expansion, self.proto_dim)

        # ── 分布原型头（可选）──
        if self.use_distributional:
            self.proto_head = ProbabilisticProtoHead(self.proto_dim, proto_dim=self.proto_dim)
        else:
            self.proto_head = None

        # ── 分类头（输出 14 维 logits，配合 BCEWithLogitsLoss）──
        self.fc2 = nn.Linear(self.proto_dim, self.num_classes)

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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Backbone
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)          # (B, 2048)

        # 原型空间
        proto_features = F.relu(self.fc1(x))  # (B, proto_dim)

        # 分类
        logits = self.fc2(proto_features)     # (B, num_classes)

        if self.use_distributional and self.proto_head is not None:
            mu, logvar = self.proto_head(proto_features)
            return logits, mu, logvar
        else:
            return logits, proto_features
