# =============================================================================
# 功能：工具函数模块 — 数据集加载、原型聚合、实验详情
# =============================================================================
# 这个文件包含了联邦学习实验中用到的所有"辅助"功能：
#   1. TransformedSubset — 带独立 transform 的数据子集包装器
#   2. get_dataset — 加载 ChestX-ray14 并按 Non-IID/IID 划分
#   3. average_weights / average_weights_fedbn — 模型权重聚合
#   4. agg_func — 单客户端本地原型聚合
#   5. proto_aggregation — 跨客户端全局原型聚合（D²-FL 使用贝叶斯融合）
#   6. exp_details — 打印实验配置详情
#
# D²-FL 和 FedProto 的核心区别在这里体现：
#   proto_aggregation 根据 use_distributional 标志选择：
#   - False: 算术平均（FedProto 方式，点原型取均值）
#   - True:  贝叶斯精度加权融合（D²-FL 方式，方差小的原型权重更大）
# =============================================================================

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
    """带独立 transform 的数据子集包装器

    为什么需要这个类？
        PyTorch 自带的 Subset 不创建新的 dataset 对象，而是持有对原始
        dataset 的引用。这意味着所有 Subset 共享同一个 transform。
        但在联邦学习中，不同客户端可能需要不同的预处理（例如：不同的
        数据增强策略）。TransformedSubset 允许每个子集独立设置 transform。

    参数:
        dataset:   原始完整数据集（包含图像和标签）
        indices:   要提取的样本索引列表
        transform: 独立应用于每个样本的图像变换（如 Normalize, RandomFlip 等）
    """

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """返回指定索引的 (图像, 标签)，自动应用 transform"""
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def labels(self):
        """代理底层 dataset 的 labels，只返回当前子集的标签"""
        return self.dataset.labels[self.indices]


# ═══════════════════════════════════════════════════════════════════════════════
#  数据集加载
# ═══════════════════════════════════════════════════════════════════════════════

def get_dataset(args, n_list, k_list):
    """加载 ChestX-ray14 数据集并按 Non-IID/IID 划分给各客户端

    划分逻辑：
        Non-IID 模式：
        - 每个客户端随机获得 n_list[i] 个疾病类别（类别数不同 → Non-IID）
        - 每个类别随机获得 k_list[i] 个样本（样本数不同 → 更 Non-IID）
        - 测试集也按相同方式划分（长尾分布）

        IID 模式：
        - 所有客户端随机分得等量数据（每个样本随机分配给某个客户端）

    参数:
        args:   配置对象（model, data_dir, num_users, iid, image_size 等）
        n_list: 每个客户端拥有的类别数量列表
        k_list: 每个客户端每类拥有的样本数量列表

    返回:
        train_dataset:   训练集（带 ImageNet 标准化的 transform）
        test_dataset:    测试集
        user_groups:     训练数据客户端划分 {client_idx: data indices}
        user_groups_lt:  测试数据客户端划分（Non-IID 有值，IID 为 None）
        classes_list:    每个客户端拥有的类别列表
        classes_list_gt: 用于测试的客户端类别列表
    """
    if args.dataset != 'chestxray14':
        raise ValueError(f"Unsupported dataset: {args.dataset}. Only 'chestxray14' is supported.")

    data_dir = args.data_dir + 'chestxray'
    image_size = getattr(args, 'image_size', 224)

    # ── 训练数据增强 + 标准化 ──
    # 使用 ImageNet 标准均值和标准差（配合预训练 ResNet-50）
    train_transform = transforms.Compose([
        transforms.Grayscale(3),  # 灰度图复制为 3 通道（模拟 RGB），使 ResNet 可处理
        transforms.Resize((image_size, image_size)),  # 统一尺寸
        transforms.RandomHorizontalFlip(),  # 随机水平翻转（数据增强）
        transforms.ToTensor(),  # [0, 255] → [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 标准归一化
                             std=[0.229, 0.224, 0.225]),
    ])

    # ── 测试 transform（无数据增强）──
    test_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 加载完整数据集（无 transform，由 TransformedSubset 按需应用）
    full_dataset = ChestXray14(data_dir, transform=None, image_size=image_size)
    n_total = len(full_dataset)
    # 80/20 训练/测试划分（固定 seed 保证可复现）
    n_train = int(0.8 * n_total)
    idxs = np.random.RandomState(args.seed).permutation(n_total)
    train_idxs, test_idxs = idxs[:n_train], idxs[n_train:]

    train_dataset = TransformedSubset(full_dataset, train_idxs, train_transform)
    test_dataset = TransformedSubset(full_dataset, test_idxs, test_transform)

    # ── 按 IID/Non-IID 分配客户端数据 ──
    if args.iid:
        # IID：所有客户端随机等分数据
        user_groups = chestxray_iid(train_dataset, args.num_users)
        user_groups_lt = None
        classes_list = None
        classes_list_gt = None
    else:
        # Non-IID：按 n_list/k_list 分配不同类别和样本数
        user_groups, classes_list = chestxray_noniid(
            args, train_dataset, args.num_users, n_list, k_list)
        user_groups_lt = chestxray_noniid_lt(
            args, test_dataset, args.num_users, n_list, k_list, classes_list)
        classes_list_gt = classes_list

    return train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt


# ═══════════════════════════════════════════════════════════════════════════════
#  模型权重聚合
# ═══════════════════════════════════════════════════════════════════════════════

def average_weights(w):
    """标准 FedAvg 权重聚合：所有客户端参数等权平均

    这是最基础的联邦聚合方式：每个客户端不管数据量多少、模型好坏，
    一律享有相同的投票权。

    参数:
        w: state_dict 列表

    返回:
        w_avg: 等权平均后的 state_dict
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights_fedbn(w):
    """FedBN 权重聚合：跳过 BatchNorm 参数，仅平均 conv/linear 层

    FedBN 的核心思想：BatchNorm 层的 running_mean 和 running_var
    记录了各客户端本地的数据统计信息，不应该被跨客户端平均。
    只共享 conv/linear 层的参数。

    跳过的参数：名称中包含 'bn', 'running_mean', 'running_var' 的参数

    返回:
        w_avg: 聚合后的 state_dict（BN 参数保留第一个客户端的值作为占位）
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if 'bn' in key or 'running_mean' in key or 'running_var' in key:
            continue  # 保留本地 BN 统计信息
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


# ═══════════════════════════════════════════════════════════════════════════════
#  原型聚合 — D²-FL 和 FedProto 的核心区别在此
# ═══════════════════════════════════════════════════════════════════════════════

def agg_func(protos, use_distributional=False):
    """单客户端本地原型聚合 — 将多个样本的原型合并为一个

    一个客户端内可能有很多个样本都属于同一类，训练时会为每个样本
    生成一个原型。上传前需要把同一类的所有原型"压缩"成一个。

    聚合方式取决于原型类型：
    - 点原型：直接取平均（所有原型的算术平均）
    - 分布原型：混合高斯平均（方差 = 组内方差 + 组间方差）

    为什么分布原型的方差聚合要用"组内方差 + 组间方差"？
    - 组内方差（vars_.mean()）：每个样本自己的不确定性
    - 组间方差（mus.var(dim=0)）：不同样本之间的差异
    - 两者相加 → 正确编码"这一类有多分散"

    参数:
        protos:              单客户端原型字典 {label: [proto_val, ...]}
        use_distributional:  True=分布原型聚合，False=点原型平均

    返回:
        protos: 聚合后的字典 {label: single_proto}
    """
    for label, proto_list in protos.items():
        if len(proto_list) > 1:
            if use_distributional:
                # ── 分布原型聚合（混合高斯）──
                mus = torch.stack([p[0] for p in proto_list])        # (N, D)
                logvars = torch.stack([p[1] for p in proto_list])    # (N, D)
                vars_ = torch.exp(logvars)                            # (N, D)

                # 融合均值 = 各样本均值的算术平均
                mu_avg = mus.mean(dim=0)                               # (D,)

                # 融合方差 = 组内平均方差 + 组间方差
                # vars_.mean(dim=0)：各样本自身不确定性的平均
                # mus.var(dim=0)：不同样本均值的离散程度
                avg_var = vars_.mean(dim=0) + mus.var(dim=0, unbiased=False)
                logvar_avg = torch.log(avg_var + 1e-8)                # (D,)

                protos[label] = (mu_avg, logvar_avg)
            else:
                # ── 点原型聚合（算术平均）──
                proto = 0 * proto_list[0].data
                for p in proto_list:
                    proto += p.data
                protos[label] = proto / len(proto_list)
        else:
            # 只有一个样本 → 无需聚合
            protos[label] = proto_list[0]

    return protos


def proto_aggregation(local_protos_list, use_distributional=False):
    """跨客户端全局原型聚合

    将多个客户端上传的同类原型进行融合，这是服务器端的"聚合"步骤。

    聚合方式：
    - 点原型（use_distributional=False）：
      → 算术平均。公平但忽略质量差异。
    - 分布原型（use_distributional=True）：
      → 贝叶斯精度加权融合。方差小的客户端（数据质量高/样本多）
      在聚合中有更大的权重。这是 D²-FL 相比 FedProto 的关键优势。

    参数:
        local_protos_list:    {client_idx: {label: proto}} 各客户端上传的原型
        use_distributional:   True=贝叶斯融合，False=算术平均

    返回:
        聚合后的全局原型字典 {label: fused_proto}
    """
    # 步骤1：按标签收集所有客户端的原型
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    # 步骤2：对每个标签进行融合
    for label, proto_list in agg_protos_label.items():
        if len(proto_list) > 1:
            if use_distributional:
                # ── 贝叶斯精度加权融合 ──
                # 公式：μ_global = Σ(μ_i/σ²_i) / Σ(1/σ²_i)
                #       σ²_global = 1 / Σ(1/σ²_i)
                mu_fused, logvar_fused = bayesian_fusion_single_label(proto_list)
                agg_protos_label[label] = (mu_fused, logvar_fused)
            else:
                # ── 点原型算术平均 ──
                proto = 0 * proto_list[0].data
                for p in proto_list:
                    proto += p.data
                agg_protos_label[label] = proto / len(proto_list)
        else:
            # 只有一个客户端有该标签 → 直接使用
            agg_protos_label[label] = proto_list[0]

    return agg_protos_label


# ═══════════════════════════════════════════════════════════════════════════════
#  实验配置打印
# ═══════════════════════════════════════════════════════════════════════════════

def exp_details(args):
    """打印实验配置详情

    在训练开始前输出所有关键配置参数，方便日志记录和实验管理。
    包括：算法名称、模型、优化器、联邦参数、隐私设置等。
    """
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

    # 基线算法专属配置
    if args.alg == 'fedgmkd':
        print(f'    GMM components     : {getattr(args, "gmm_components", 3)}')
    if args.alg == 'fedbcs':
        print('    Style recalibration : AdaptiveIN (1D)')
    if args.alg == 'fedseproto':
        print(f'    MI lambda           : {getattr(args, "mi_lambda", 0.05)}')

    # FedProto / D²-FL 专属配置
    if args.alg == 'fedproto':
        print('    Prototype mode     : Point (baseline)')
    if args.alg == 'd2fl':
        if getattr(args, 'use_distributional', False):
            print('    Prototype mode     : Distributional')
            print(f'    Distribution type   : {args.dist_type}')
        else:
            print('    Prototype mode     : Distributional (disabled, using point)')
        if getattr(args, 'proto_dim', None):
            print(f'    Proto dim           : {args.proto_dim}')

        # D²-FL 增强特性
        print(f'    Proto momentum      : {getattr(args, "proto_momentum", 0.9)}')
        print(f'    LD warmup rounds    : {getattr(args, "ld_warmup", 50)}')
        print(f'    Temperature         : {getattr(args, "temperature", 1.0)}')

        if getattr(args, 'use_disentangle', False):
            sem_ratio = getattr(args, 'sem_ratio', 0.75)
            print(f'    Prototype disentangle : Enabled (sem={sem_ratio:.0%}, style={(1-sem_ratio):.0%})')
            print(f'    Disentangle lambda    : {getattr(args, "dis_lambda", 0.05)}')

    print(f'    Pretrained backbone : {getattr(args, "pretrained", True)}')

    print()
    return
