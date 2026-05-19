# 功能：MNIST CNN模型，提取原型向量用于联邦原型学习

import sys
from pathlib import Path
lib_dir = (Path(__file__).parent / "..").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from torch import nn
import torch.nn.functional as F
from dist_proto import ProbabilisticProtoHead


class CNNMnist(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320 / 20 * args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)

        self.use_distributional = getattr(args, 'use_distributional', False)
        if self.use_distributional:
            proto_dim = getattr(args, 'proto_dim', None) or 50
            self.proto_head = ProbabilisticProtoHead(50, proto_dim=proto_dim)
        else:
            self.proto_head = None

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x1 = F.relu(self.fc1(x))

        if self.use_distributional and self.proto_head is not None:
            mu, log_var = self.proto_head(x1)
        else:
            protos = x1

        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)

        if self.use_distributional and self.proto_head is not None:
            return F.log_softmax(x, dim=1), mu, log_var
        else:
            return F.log_softmax(x, dim=1), protos
