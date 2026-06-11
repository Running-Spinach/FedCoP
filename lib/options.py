# =============================================================================
# 功能：命令行参数解析模块 — D²-FL 联邦学习实验的"遥控器"
# =============================================================================
# 所有实验配置都通过命令行参数传入，方便做消融实验和超参搜索。
#
# 参数分为 6 组：
#   1. 联邦学习参数 — 训练轮数、客户端数、本地 epoch/batch 等
#   2. 模型参数 — 骨干网络、算法选择、归一化方式等
#   3. 数据参数 — 数据集路径、类别数、图像尺寸等
#   4. Non-IID 参数 — 每客户类别数、每类样本数等
#   5. D²-FL 专属参数 — 分布原型、解耦、温度等（提出方法的创新点）
#   6. 差分隐私参数 — DP 保护的可选开关
#
# 使用示例：
#   # 完整 D²-FL 模式
#   python federated_main.py --alg d2fl --use_distributional --use_disentangle
#
#   # 消融实验：去掉解耦
#   python federated_main.py --alg d2fl --use_distributional
#
#   # 消融实验：去掉分布原型（退化回 FedProto + 预训练）
#   python federated_main.py --alg d2fl
#
#   # 带差分隐私
#   python federated_main.py --alg d2fl --use_dp --dp_epsilon 8
# =============================================================================

import argparse


def args_parser():
    """解析命令行参数，返回包含所有配置的参数对象

    返回:
        args: 包含联邦学习实验全部配置参数的 Namespace 对象
    """
    parser = argparse.ArgumentParser(
        description='D²-FL: Distributional Pathology Prototype Federated Learning'
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
    parser.add_argument('--alg', type=str, default='d2fl',
                        choices=['fedproto', 'fedavg', 'fedgmkd', 'fedbcs', 'fedseproto', 'd2fl'],
                        help='FL 算法选择。fedproto=点原型基线, fedavg=经典平均, '
                             'fedgmkd=GMM原型(NeurIPS2024), fedbcs=频域风格重校准(AAAI2026), '
                             'fedseproto=语义域解耦(ECAI2024), d2fl=D²-FL提出方法(默认)')
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
                        help="数据集名称（当前仅支持 ChestX-ray14 多标签数据集）")
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help="数据集根目录（其下应有 chestxray/ 子目录）")
    parser.add_argument('--num_classes', type=int, default=14,
                        help="类别数量。ChestX-ray14 含 14 种常见胸腔疾病。")

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
    #  6. 原型学习通用参数（FedProto / D²-FL 共用）
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--ld', type=float, default=1,
                        help="原型损失权重 λ。控制 L_proto 在总损失中的占比。"
                             "0=不使用原型损失，1=等权，>1=更重视原型对齐。")
    parser.add_argument('--ft_round', type=int, default=10,
                        help="微调轮数（实验性功能，通常不使用）")

    # ═══════════════════════════════════════════════════════════════════════════
    #  7. D²-FL 专属：分布原型参数
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--use_distributional', action='store_true',
                        help='[D²-FL] 启用高斯分布原型（替代点原型）。'
                             '开启后，原型从单向量变为 N(μ, σ²) 高斯分布，'
                             '支持贝叶斯精度加权融合和不确定性编码。')
    parser.add_argument('--dist_type', type=str, default='kl',
                        choices=['kl', 'wasserstein', 'mse'],
                        help='[D²-FL] 原型距离类型。kl=KL散度（推荐，信息论最优），'
                             'wasserstein=2-Wasserstein距离（几何视角），'
                             'mse=仅均值MSE（退化回FedProto行为）。')
    parser.add_argument('--proto_dim', type=int, default=None,
                        help='[D²-FL] 原型向量维度。None 时自动使用 fc1 输出维度（256）。')

    # ═══════════════════════════════════════════════════════════════════════════
    #  8. D²-FL 专属：原型解耦参数
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--use_disentangle', action='store_true',
                        help='[D²-FL] 启用增强原型解耦（语义-风格分离）。'
                             '包含：可学习门控 + 对抗域不变 + 对比语义对齐。'
                             '仅上传语义原型，风格特征保留本地。')
    parser.add_argument('--sem_ratio', type=float, default=0.75,
                        help='[D²-FL] 语义维度目标占比（0-1）。如 0.75 表示约 75%% 维度分配给语义。'
                             '注意：这只是正则化引导目标，实际分配由可学习门控决定。')
    parser.add_argument('--dis_lambda', type=float, default=0.05,
                        help='[D²-FL] 解耦独立性损失权重 λ_dis。'
                             '控制 HSIC + 门控熵 + 正交约束在总损失中的占比。')

    # ═══════════════════════════════════════════════════════════════════════════
    #  9. D²-FL 专属：增强损失权重
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--cal_lambda', type=float, default=0.01,
                        help='[D²-FL] 原型校准损失权重 λ_cal。'
                             '鼓励方差反映真实不确定性（logvar ≅ log(distance)）。'
                             '设为 0 可禁用校准损失。')
    parser.add_argument('--contra_lambda', type=float, default=0.05,
                        help='[D²-FL] 对比语义对齐损失权重 λ_ctr。'
                             '让同类疾病的语义特征在空间中聚集。'
                             'Non-IID 场景下特别重要。设为 0 可禁用。')
    parser.add_argument('--adv_lambda', type=float, default=0.01,
                        help='[D²-FL] 对抗域不变损失权重 λ_adv。'
                             '确保语义特征不包含域信息（通过梯度反转训练）。'
                             '设为 0 可禁用对抗训练。')
    parser.add_argument('--ent_lambda', type=float, default=0.001,
                        help='[D²-FL] 熵正则损失权重 λ_ent。'
                             '防止方差坍缩回点原型（logvar → -∞）。'
                             '设为 0 可禁用熵正则。')

    # ═══════════════════════════════════════════════════════════════════════════
    #  10. D²-FL 专属：训练与推理策略
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='[D²-FL] 使用 ImageNet 预训练 ResNet-50 骨干（默认开启）。'
                             '预训练骨干提供更好的初始特征表达，加速收敛。')
    parser.add_argument('--proto_momentum', type=float, default=0.9,
                        help='[D²-FL] 全局原型 EMA 动量系数。'
                             '0=无动量（每轮完全替换），0.9=90%%保留旧值。'
                             '动量越大 → 全局原型越平滑，但更新越慢。')
    parser.add_argument('--ld_warmup', type=int, default=50,
                        help='[D²-FL] 原型损失权重 warmup 轮数。'
                             '前 N 轮 ld 从 0 线性增长到 args.ld。'
                             '避免训练初期分类器未收敛时，原型损失误导训练。')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='[D²-FL] 原型推理温度系数 T。'
                             'logit = -dist / T。T<1 锐化（更敏感），T>1 软化（更平滑）。')
    parser.add_argument('--use_per_class_temp', action='store_true', default=True,
                        help='[D²-FL] 启用每类可学习温度参数（默认开启）。'
                             '每个疾病类别学习自己的最优推理温度。')

    # ═══════════════════════════════════════════════════════════════════════════
    #  11. 差分隐私参数（所有算法共用）
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--use_dp', action='store_true',
                        help='启用差分隐私保护。对上传的原型（或权重 delta）进行裁剪 + 加高斯噪声。'
                             '所有算法（FedAvg/D²-FL/FedProto等）均支持。')
    parser.add_argument('--dp_epsilon', type=float, default=8.0,
                        help='目标 ε 值（(ε, δ)-DP）。ε 越小 → 隐私保护越强，但效用越低。')
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                        help='目标 δ 值（(ε, δ)-DP）。通常设为 1/数据集大小的量级。')
    parser.add_argument('--dp_clip', type=float, default=1.0,
                        help='原型的 L2 范数裁剪界限。上传前将原型的 L2 范数限制在此值内。'
                             'clip 越小 → 噪声相对越大，但隐私保护更强。')

    # ═══════════════════════════════════════════════════════════════════════════
    #  12. 基线算法专属参数
    # ═══════════════════════════════════════════════════════════════════════════

    parser.add_argument('--gmm_components', type=int, default=3,
                        help='[FedGMKD] GMM 高斯分量数。每个类别的原型用 K 个高斯分量表示。'
                             'K=1 时退化为单高斯。')
    parser.add_argument('--mi_lambda', type=float, default=0.05,
                        help='[FedSeProto] 互信息最小化损失权重。控制 HSIC 独立性约束的强度。')

    args = parser.parse_args()
    return args
