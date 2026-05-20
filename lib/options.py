# 功能：命令行参数解析模块，定义联邦学习实验的所有可配置参数

import argparse


def args_parser():
    """
    解析命令行参数，返回包含所有配置的参数对象

    返回:
        args: 包含联邦学习实验全部配置参数的对象
    """
    parser = argparse.ArgumentParser()

    # 联邦学习参数
    parser.add_argument('--rounds', type=int, default=100,
                        help="全局训练轮数")
    parser.add_argument('--num_users', type=int, default=20,
                        help="客户端数量: K")
    parser.add_argument('--frac', type=float, default=0.04,
                        help='每轮参与的客户端比例: C')
    parser.add_argument('--train_ep', type=int, default=1,
                        help="本地训练轮数: E")
    parser.add_argument('--local_bs', type=int, default=4,
                        help="本地批次大小: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD动量 (默认: 0.5)')

    # 模型参数
    parser.add_argument('--model', type=str, default='resnet50', help='模型名称: resnet50 / cnn')
    parser.add_argument('--alg', type=str, default='dppfl',
                        choices=['fedproto', 'fedavg', 'fedprox', 'fedbn', 'scaffold', 'dppfl'],
                        help="FL算法: fedproto(点原型基线), fedavg, fedprox, fedbn, scaffold, dppfl(分布原型+DP)")
    parser.add_argument('--num_channels', type=int, default=3, help="图像通道数（ChestX-ray14灰度图转3通道）")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="归一化方式: batch_norm, layer_norm, 或 None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="卷积核数量")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="是否使用最大池化")

    # 其他参数
    parser.add_argument('--image_size', type=int, default=224,
                        help="输入图像尺寸（ChestX-ray14: 224）")
    parser.add_argument('--data_dir', type=str, default='./data/', help="数据集根目录")
    parser.add_argument('--num_classes', type=int, default=14, help="类别数量（ChestX-ray14: 14种疾病）")
    parser.add_argument('--gpu', default=0, help="GPU设备ID，默认使用CPU")
    parser.add_argument('--optimizer', type=str, default='sgd', help="优化器类型")
    parser.add_argument('--iid', type=int, default=0,
                        help='是否IID分布，0表示Non-IID')
    parser.add_argument('--unequal', type=int, default=0,
                        help='是否使用不等量数据划分')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='早停轮数')
    parser.add_argument('--verbose', type=int, default=1, help='是否详细输出')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    parser.add_argument('--test_ep', type=int, default=10, help="测试评估轮数")

    # 本地训练参数（小样本学习相关）
    parser.add_argument('--ways', type=int, default=3, help="每客户类别数")
    parser.add_argument('--shots', type=int, default=100, help="每类样本数")
    parser.add_argument('--train_shots_max', type=int, default=200, help="每类最大训练样本数")
    parser.add_argument('--test_shots', type=int, default=15, help="每类测试样本数")
    parser.add_argument('--stdev', type=int, default=2, help="类别数标准差")
    parser.add_argument('--ld', type=float, default=1, help="原型损失权重")
    parser.add_argument('--ft_round', type=int, default=10, help="微调轮数")

    # 分布原型参数
    parser.add_argument('--use_distributional', action='store_true',
                        help='是否使用高斯分布原型')
    parser.add_argument('--dist_type', type=str, default='kl',
                        choices=['kl', 'wasserstein', 'mse'],
                        help='原型损失距离类型: kl, wasserstein, mse')
    parser.add_argument('--proto_dim', type=int, default=None,
                        help='原型向量维度（默认使用fc1输出维度）')

    # 差分隐私参数
    parser.add_argument('--use_dp', action='store_true',
                        help='是否对原型上传启用差分隐私保护')
    parser.add_argument('--dp_epsilon', type=float, default=8.0,
                        help='目标epsilon值 (epsilon, delta)-DP')
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                        help='目标delta值 (epsilon, delta)-DP')
    parser.add_argument('--dp_clip', type=float, default=1.0,
                        help='原型的L2范数裁剪界限')

    # FedProx / SCAFFOLD 专属参数
    parser.add_argument('--fedprox_mu', type=float, default=0.01,
                        help='FedProx 近端项系数 mu')
    parser.add_argument('--scaffold_lr', type=float, default=None,
                        help='SCAFFOLD 全局学习率（默认等于 --lr）')

    # DPP-FL 专属参数（提出方法）
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='DPP-FL: 使用 ImageNet 预训练 ResNet-50 (默认开启)')
    parser.add_argument('--proto_momentum', type=float, default=0.9,
                        help='DPP-FL: 全局原型 EMA 动量系数 (0=无动量)')
    parser.add_argument('--ld_warmup', type=int, default=50,
                        help='DPP-FL: 原型损失权重 warmup 轮数')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='DPP-FL: 原型推理温度系数 (越小越尖锐)')

    args = parser.parse_args()
    return args
