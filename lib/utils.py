# 功能：工具函数模块，包含数据集加载、权重聚合、原型聚合和实验详情输出等

import copy
import numpy as np
import torch
from torchvision import transforms
from sampling import chestxray_iid, chestxray_noniid, chestxray_noniid_lt
import sys
from pathlib import Path
lib_dir = (Path(__file__).parent).resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from dist_proto.aggregation import bayesian_fusion_single_label
from chestxray import ChestXray14


class TransformedSubset:
    """带独立 transform 的数据子集包装器，避免 Subset 共享底层 dataset.transform

    参数:
        dataset: 原始完整数据集
        indices: 要提取的样本索引列表
        transform: 独立应用于每个样本的图像变换
    """

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        """返回子集样本总数"""
        return len(self.indices)

    def __getitem__(self, idx):
        """返回指定索引的 (图像, 标签)，自动应用 transform"""
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataset(args, n_list, k_list):
    """加载数据集并按 Non-IID/IID 划分给各客户端

    参数:
        args: 配置对象（model, data_dir, num_users, iid 等）
        n_list: 每个客户端拥有的类别数量列表
        k_list: 每个客户端每类拥有的样本数量列表

    返回:
        train_dataset: 训练集
        test_dataset: 测试集
        user_groups: 训练数据客户端划分 {client_idx: data indices}
        user_groups_lt: 测试数据客户端划分（Non-IID 模式有值，IID 为 None）
        classes_list: 每个客户端的类别列表
        classes_list_gt: 用于测试的客户端类别列表
    """
    if args.dataset != 'chestxray14':
        raise ValueError(f"Unsupported dataset: {args.dataset}. Only 'chestxray14' is supported.")

    # ── ChestX-ray14: 多标签数据集 ──
    data_dir = args.data_dir + 'chestxray'
    image_size = getattr(args, 'image_size', 224)

    train_transform = transforms.Compose([
        transforms.Grayscale(3),#3通道灰度图
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = ChestXray14(data_dir, transform=None, image_size=image_size)
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    idxs = np.random.RandomState(args.seed).permutation(n_total)
    train_idxs, test_idxs = idxs[:n_train], idxs[n_train:]

    train_dataset = TransformedSubset(full_dataset, train_idxs, train_transform)
    test_dataset = TransformedSubset(full_dataset, test_idxs, test_transform)

    if args.iid:
        user_groups = chestxray_iid(train_dataset, args.num_users)
        user_groups_lt = None
        classes_list = None
        classes_list_gt = None
    else:
        user_groups, classes_list = chestxray_noniid(
            args, train_dataset, args.num_users, n_list, k_list)
        user_groups_lt = chestxray_noniid_lt(
            args, test_dataset, args.num_users, n_list, k_list, classes_list)
        classes_list_gt = classes_list

    return train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt


def average_weights(w):
    """
    标准 FedAvg 权重聚合：对所有参数取均值（等权平均）
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights_fedbn(w):
    """
    FedBN 权重聚合：跳过 BatchNorm 层参数，仅平均 conv/linear 层

    BN 层参数包括：running_mean, running_var, weight, bias
    （这些保留各客户端本地值，不做聚合）

    返回:
        w_avg: 聚合后的 state_dict（BN 参数 = 第一个客户端的值作为占位）
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if 'bn' in key or 'running_mean' in key or 'running_var' in key:
            continue  # 跳过 BN 参数，保留客户端本地值
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def agg_func(protos, use_distributional=False):
    """对单个客户端内同一标签的多个原型取平均（本地聚合）

    参数:
        protos: {label: [proto_val, ...]} 单客户端原型字典
        use_distributional: True 则对 (mu, logvar) 做混合高斯平均，
                           False 则对点原型做算术平均

    返回:
        protos: 聚合后的字典 {label: single_proto}
    """
    for label, proto_list in protos.items():
        if len(proto_list) > 1:
            if use_distributional:
                mus = torch.stack([p[0] for p in proto_list])
                logvars = torch.stack([p[1] for p in proto_list])
                vars_ = torch.exp(logvars)

                mu_avg = mus.mean(dim=0)
                avg_var = vars_.mean(dim=0) + mus.var(dim=0, unbiased=False)
                logvar_avg = torch.log(avg_var + 1e-8)

                protos[label] = (mu_avg, logvar_avg)
            else:
                proto = 0 * proto_list[0].data
                for p in proto_list:
                    proto += p.data
                protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


def proto_aggregation(local_protos_list, use_distributional=False):
    """跨客户端全局原型聚合

    将多个客户端上传的同类原型进行融合：
      - 点原型: 算术平均
      - 分布原型: 贝叶斯精度加权融合

    参数:
        local_protos_list: {client_idx: {label: proto}}
        use_distributional: True 则使用贝叶斯融合，False 则算术平均

    返回:
        聚合后的全局原型字典 {label: fused_proto}
    """
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for label, proto_list in agg_protos_label.items():
        if len(proto_list) > 1:
            if use_distributional:
                mu_fused, logvar_fused = bayesian_fusion_single_label(proto_list)
                agg_protos_label[label] = (mu_fused, logvar_fused)
            else:
                proto = 0 * proto_list[0].data
                for p in proto_list:
                    proto += p.data
                agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0]

    return agg_protos_label


def exp_details(args):
    """打印实验配置详情：算法、模型、优化器、联邦参数、隐私参数等"""
    print('\nExperimental details:')
    print(f'    Algorithm : {args.alg}')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.rounds}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.train_ep}\n')

    if args.alg == 'fedprox':
        print(f'    FedProx mu         : {args.fedprox_mu}')
    if args.alg == 'scaffold':
        scaffold_lr = args.scaffold_lr or args.lr
        print(f'    SCAFFOLD global lr : {scaffold_lr}')

    if args.alg == 'fedproto':
        print('    Prototype mode     : Point (baseline)')
    if args.alg == 'dppfl':
        if getattr(args, 'use_distributional', False):
            print('    Prototype mode     : Distributional')
            print(f'    Distribution type   : {args.dist_type}')
        else:
            print('    Prototype mode     : Distributional (disabled, using point)')
        if getattr(args, 'proto_dim', None):
            print(f'    Proto dim           : {args.proto_dim}')

        print(f'    Proto momentum      : {getattr(args, "proto_momentum", 0.9)}')
        print(f'    LD warmup rounds    : {getattr(args, "ld_warmup", 50)}')
        print(f'    Temperature         : {getattr(args, "temperature", 1.0)}')

        if getattr(args, 'use_disentangle', False):
            sem_ratio = getattr(args, 'sem_ratio', 0.75)
            print(f'    Prototype disentangle : Enabled (sem={sem_ratio:.0%}, style={(1-sem_ratio):.0%})')
            print(f'    Disentangle lambda    : {getattr(args, "dis_lambda", 0.05)}')

    print(f'    Pretrained backbone : {getattr(args, "pretrained", True)}')

    if getattr(args, 'use_dp', False):
        print('    Differential Privacy: Enabled (all algorithms)')
        print(f'    Target epsilon      : {args.dp_epsilon}')
        print(f'    Target delta        : {args.dp_delta}')
        print(f'    Clip norm           : {args.dp_clip}')
    else:
        print('    Differential Privacy: Disabled')
    print()
    return
