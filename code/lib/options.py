# =============================================================================
# 功能:命令行参数解析模块 — FedCoP 联邦学习实验的"遥控器"
# =============================================================================
# 所有实验配置都通过命令行参数传入，方便做消融实验和超参搜索。
#
# 参数分为 6 组：
#   1. 联邦学习参数 — 训练轮数、客户端数、本地 epoch/batch 等
#   2. 模型参数 — 骨干网络、算法选择、归一化方式等
#   3. 数据参数 — 数据集路径、类别数、图像尺寸等
#   4. Non-IID 参数 — 每客户类别数、每类样本数等
#   5. FedCoP 专属参数 — 共现结构、分布原型等(提出方法的创新点)
#   6. 差分隐私参数 — DP 保护的可选开关
#
# 使用示例:
#   # FedCoP 完整方法(默认)
#   python federated_main.py --alg fedcop
#
#   # 消融:关闭共现结构(R=I,隔离核心贡献)
#   python federated_main.py --alg fedcop --no_cooccurrence
#
#   # 消融:只用本地共现,不联邦聚合(证明联邦聚合必要性)
#   python federated_main.py --alg fedcop --local_cooc_only
#
#   # 消融:关训练侧 L_co,保留推理侧 R̂
#   python federated_main.py --alg fedcop --no_lco
#
#   # 基线对比
#   python federated_main.py --alg fedprox --fedprox_mu 0.01
# =============================================================================

import argparse


def args_parser():
    """解析命令行参数，返回包含所有配置的参数对象

    返回:
        args: 包含联邦学习实验全部配置参数的 Namespace 对象
    """
    parser = argparse.ArgumentParser(
        description='FedCoP: Federated Co-occurrence-aware Prototypes'
    )

    # ═══════════════════════════════════════════════════════════════════════════
    #  1. 联邦学习基础参数
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--rounds', type=int, default=100,
                        help="全局训练轮数（通信轮次）。Non-IID 场景建议 ≥100。")
    parser.add_argument('--num_users', type=int, default=20,
                        help="客户端数量 K（即联邦中的医院/机构数）")
    parser.add_argument('--frac', type=float, default=0.25,
                        help='每轮参与的客户端比例 C（如 0.25 表示每轮随机选 25%% 的客户端）')
    parser.add_argument('--train_ep', type=int, default=1,
                        help="本地训练轮数 E（每个客户端每轮做几个 epoch 的 SGD）")
    parser.add_argument('--local_bs', type=int, default=4,
                        help="本地批次大小 B（ChestX-ray14 图像大，建议 4~8）")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率（SGD/Adam 共用）')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD 动量系数（默认 0.5，仅 SGD 使用）')

    # ═══════════════════════════════════════════════════════════════════════════
    #  2. 模型与算法参数
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--model', type=str, default='resnet50',
                        help='模型名称: resnet50 / cnn')
    parser.add_argument('--alg', type=str, default='fedcop',
                        choices=['fedproto', 'fedavg', 'fedprox', 'fedgmkd',
                                 'fedseproto', 'fedcop'],
                        help='FL 算法选择。fedavg=经典平均, fedprox=近端正则基线, '
                             'fedproto=点原型基线, fedgmkd=GMM原型(NeurIPS2024), '
                             'fedseproto=语义域解耦(ECAI2024), '
                             'fedcop=FedCoP 提出方法(默认,共现感知分布原型)')
    parser.add_argument('--num_channels', type=int, default=3,
                        help="图像通道数。ChestX-ray14 原始为灰度图，通过 Grayscale(3) 转为 3 通道。")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="归一化方式: batch_norm（推荐）, layer_norm, 或 None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="卷积核数量（仅 CNN 模型使用，ResNet 忽略）")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="是否使用最大池化（仅 CNN 模型使用）")

    # ═══════════════════════════════════════════════════════════════════════════
    #  3. 数据集参数
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--image_size', type=int, default=224,
                        help="输入图像尺寸。ResNet-50 标准输入为 224×224。")
    parser.add_argument('--dataset', type=str, default='chestxray14',
                        choices=['chestxray14', 'mured'],
                        help="数据集名称: chestxray14 (NIH 胸片, 14 类) / "
                             "mured (MuReD 眼底多标签, 20 类)")
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help="数据集根目录（其下应有 chestxray/ 或 "
                             "Multi-Label Retinal Diseases (MuReD) Dataset/ 子目录）")
    parser.add_argument('--num_classes', type=int, default=14,
                        help="类别数量。chestxray14=14, mured=20。须与所选数据集匹配。")

    # ═══════════════════════════════════════════════════════════════════════════
    #  4. 硬件与训练配置
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--gpu', default=0, help="GPU 设备 ID（如 0, 1）。默认使用 CPU。")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="优化器类型: sgd（推荐，联邦学习标准）/ adam")
    parser.add_argument('--iid', type=int, default=0,
                        help='是否 IID 数据分布。0=Non-IID（默认，更真实），1=IID。')
    parser.add_argument('--unequal', type=int, default=0,
                        help='是否使用不等量数据划分（1=各客户端样本数不同）')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='早停容忍轮数（连续 N 轮不提升则停止）')
    parser.add_argument('--verbose', type=int, default=1,
                        help='是否详细输出训练日志（0=静默，1=详细）')
    parser.add_argument('--seed', type=int, default=1234,
                        help='随机种子（保证实验可复现）')
    parser.add_argument('--test_ep', type=int, default=10,
                        help="测试评估间隔轮数（每 N 轮评估一次）")

    # ═══════════════════════════════════════════════════════════════════════════
    #  5. Non-IID 数据划分参数
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--ways', type=int, default=3,
                        help="每客户端拥有的类别数（Non-IID 程度的核心参数）"
                             "ways 越小 → Non-IID 程度越严重。")
    parser.add_argument('--shots', type=int, default=100,
                        help="每类大约的样本数（每个客户端）")
    parser.add_argument('--train_shots_max', type=int, default=200,
                        help="每类最大训练样本数（上限约束）")
    parser.add_argument('--test_shots', type=int, default=15,
                        help="每类测试样本数")
    parser.add_argument('--stdev', type=int, default=2,
                        help="类别数和样本数的标准差（增大 → 客户端间差异更大）")

    # ═══════════════════════════════════════════════════════════════════════════
    #  6. 原型学习通用参数(FedProto / FedCoP 共用)
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--ld', type=float, default=1,
                        help="原型损失权重 λ。控制 L_proto 在总损失中的占比。"
                             "0=不使用原型损失，1=等权，>1=更重视原型对齐。")
    parser.add_argument('--ft_round', type=int, default=10,
                        help="微调轮数（实验性功能，通常不使用）")

    # ═══════════════════════════════════════════════════════════════════════════
    #  7. 分布原型参数(FedCoP / 基线共用)
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--use_distributional', action='store_true',
                        help='启用高斯分布原型(替代点原型)。开启后原型变为 N(μ, σ²),'
                             '支持贝叶斯精度加权融合与不确定性编码。FedCoP 始终启用。')
    parser.add_argument('--dist_type', type=str, default='kl',
                        choices=['kl', 'wasserstein', 'mse'],
                        help='原型距离类型。kl=KL散度(推荐),'
                             'wasserstein=2-Wasserstein距离,mse=仅均值MSE(退化回FedProto)。')
    parser.add_argument('--proto_dim', type=int, default=None,
                        help='原型向量维度。None 时自动使用 fc1 输出维度。')

    # ═══════════════════════════════════════════════════════════════════════════
    #  8. FedCoP 专属:共现结构参数(核心创新)
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--co_lambda', type=float, default=0.1,
                        help='[FedCoP] 共现结构对齐损失 L_co 的权重 λ_co。'
                             '把各类原型余弦 Gram 对齐到联邦共现相关矩阵 R̂。'
                             '设为 0 等价于关闭训练侧结构(--no_lco)。')
    parser.add_argument('--co_warmup', type=int, default=10,
                        help='[FedCoP] L_co 的 warmup 轮数。前 N 轮 R̂ 还是噪声,'
                             '不下发 R̂(L_co 退化为 0),等 R̂ 稳定后再开。'
                             '0=从第一轮就开(旧行为)。修复"完整版反不如 nocoo"的倒挂。')
    parser.add_argument('--cov_shrinkage', type=float, default=0.1,
                        help='[FedCoP] 共现相关矩阵收缩系数 η ∈ [0,1]。'
                             'R̂=(1−η)R+ηI,拉向单位阵保证正定+小样本稳定。'
                             'η=1 退化回独立(关闭共现结构)。')
    parser.add_argument('--co_rank', type=int, default=0,
                        help='[FedCoP] R̂ 低秩近似的秩。0=全秩(默认,14×14 无需降秩)。')
    parser.add_argument('--co_beta', type=float, default=1.0,
                        help='[FedCoP] mean-field 解码的耦合强度 β。'
                             'β 越大类间证据传播越强;β=0 退化为独立 sigmoid。')
    parser.add_argument('--co_mf_steps', type=int, default=2,
                        help='[FedCoP] mean-field 解码迭代步数(通常 1~3 步即收敛)。')
    parser.add_argument('--ent_lambda', type=float, default=0.001,
                        help='[FedCoP] 熵正则 L_ent 权重。防止方差坍缩回点原型。'
                             '设为 0 可禁用。')

    # ── FedCoP 消融开关(顶会必需,证明各创新点贡献)──
    parser.add_argument('--no_cooccurrence', action='store_true',
                        help='[FedCoP 消融] 关闭共现结构:R̂=I 且 L_co 关。'
                             '隔离"共现结构"的总贡献(对应论文 R=I 消融)。')
    parser.add_argument('--local_cooc_only', action='store_true',
                        help='[FedCoP 消融] 推理时只用各客户端本地共现,不联邦聚合。'
                             '证明"联邦聚合共现结构"的必要性(核心 FL 论点)。')
    parser.add_argument('--no_lco', action='store_true',
                        help='[FedCoP 消融] 关闭训练侧 L_co,但保留推理侧 R̂ 解码。'
                             '隔离训练侧 vs 推理侧结构的作用。')

    # ═══════════════════════════════════════════════════════════════════════════
    #  9. 训练与推理策略
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='使用 ImageNet 预训练 ResNet-50 骨干(默认开启)。')
    parser.add_argument('--proto_momentum', type=float, default=0.9,
                        help='全局原型/共现矩阵 EMA 动量系数。'
                             '0=无动量(每轮完全替换),0.9=90%%保留旧值。')
    parser.add_argument('--ld_warmup', type=int, default=50,
                        help='原型损失权重 warmup 轮数。'
                             '前 N 轮 ld 从 0 线性增长到 args.ld,避免初期误导训练。')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='原型推理温度 T。logit = -dist / T。'
                             'T<1 锐化(更敏感),T>1 软化(更平滑)。')
    parser.add_argument('--fuse_alpha', type=float, default=0.5,
                        help='[FedCoP] 推理时分类器 logit 与原型 logit 的融合权重 α。'
                             'fused = α·logit_cls + (1−α)·logit_proto,再进 mean-field。'
                             'α=1 纯分类器,α=0 纯原型,0.5 等权(默认)。')

    # ═══════════════════════════════════════════════════════════════════════════
    #  11. 基线算法专属参数
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--gmm_components', type=int, default=3,
                        help='[FedGMKD] GMM 高斯分量数。每个类别的原型用 K 个高斯分量表示。'
                             'K=1 时退化为单高斯。')
    parser.add_argument('--mi_lambda', type=float, default=0.05,
                        help='[FedSeProto] 互信息最小化损失权重。控制 HSIC 独立性约束的强度。')
    parser.add_argument('--fedprox_mu', type=float, default=0.01,
                        help='[FedProx] 近端正则系数 μ。损失 = L_CE + (μ/2)·‖w−w_global‖²。'
                             'μ 越大本地模型越接近全局模型,Non-IID 下更稳定。')

    args = parser.parse_args()
    return args
