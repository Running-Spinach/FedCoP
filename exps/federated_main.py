# 功能：FedProto联邦学习主程序入口（任务异构模式，ChestX-ray14 + ResNet-50）
# 核心思想：通过共享原型（prototypes）来促进客户端间的知识迁移

import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from options import args_parser
from update import LocalUpdate, LocalTest, test_inference_new_het_lt
from models.resnet import ResNet50
from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func
from dp import DPMechProto, MomentsAccountant, compute_noise_multiplier_from_epsilon


def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    """
    FedProto 任务异构联邦学习主循环

    每轮通信分为三个阶段：
      阶段一：本地训练 —— 各客户端用全局原型作为正则化目标，在本地数据上更新模型
      阶段二：原型上传 —— 客户端提取同类样本的平均原型，可选加 DP 噪声后上传
      阶段三：全局聚合 —— 服务器按标签融合所有本地原型，生成新一轮的全局原型

    最终测试：分别评估"只用本地模型 softmax" vs "用全局原型做最近邻分类"的准确率
    """

    # ── 读取扩展功能开关 ───────────────────────────────────────────
    use_dist = getattr(args, 'use_distributional', False)  # True: 高斯分布原型；False: 点原型
    use_dp = getattr(args, 'use_dp', False)                # True: 对上传原型添加差分隐私噪声

    # ── 初始化 TensorBoard 日志 ─────────────────────────────────────
    summary_writer = SummaryWriter('../tensorboard/chestxray_fedproto_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    # ── 全局状态初始化 ─────────────────────────────────────────────
    # global_protos: 所有客户端共享的全局原型字典 {label: proto}
    # 第 0 轮时为空列表，客户端在本地训练时会跳过原型正则化项 (loss2 = 0)
    global_protos = []
    idxs_users = np.arange(args.num_users)          # 所有客户端都参与训练（不使用部分采样）
    train_loss, train_accuracy = [], []

    # ── 差分隐私组件初始化 ──────────────────────────────────────────
    # 核心原理：对每个客户端上传的 (mu, logvar) 做 L2 范数裁剪 + 高斯噪声，
    #          并用 Moments Accountant 跨轮追踪 (ε, δ) 隐私预算
    if use_dp:
        accountant = MomentsAccountant(delta=args.dp_delta)
        # 将总 ε 平分到每一轮，二分搜索找到对应的噪声乘数 σ
        per_round_noise = compute_noise_multiplier_from_epsilon(
            args.dp_epsilon / args.rounds,
            sample_rate=1.0,
            delta=args.dp_delta
        )
        dp_mech = DPMechProto(
            clip_norm=args.dp_clip,          # L2 裁剪阈值 C
            noise_multiplier=per_round_noise, # 噪声乘数 σ（噪声标准差 = σ * C）
            use_dp=True
        )
        print(f'DP enabled: target_epsilon={args.dp_epsilon}, noise_multiplier={per_round_noise:.4f}')
    else:
        accountant = None
        dp_mech = DPMechProto(use_dp=False)

    # ═══════════════════════════════════════════════════════════════
    #  全局通信循环（FedProto 核心训练流程）
    # ═══════════════════════════════════════════════════════════════
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        # ── 阶段一：本地训练（遍历每个客户端） ─────────────────────
        # 每个客户端独立执行：
        #   1. 用本地数据 + 当前全局原型 → 计算 L = L_CE + λ * L_proto
        #   2. 反向传播更新模型参数
        #   3. 提取同类样本的平均原型（客户端内聚合）
        proto_loss = 0
        for idx in idxs_users:
            # 创建该客户端的本地数据加载器
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])

            # 核心：FedProto 本地训练
            #   - global_protos 作为正则化目标：拉近本地原型与全局原型的距离
            #   - 返回: w=更新后的权重, loss={'total','1'(CE),'2'(proto)}, acc, protos
            w, loss, acc, protos = local_model.update_weights_het(
                args, idx, global_protos,
                model=copy.deepcopy(local_model_list[idx]),  # 深拷贝，避免原地修改
                global_round=round
            )

            # 客户端内原型聚合：同一 label 的多个样本原型 → 取平均得到一个代表向量
            # 点原型：直接求均值；分布原型：合并方差 E[Var] + Var[E]
            agg_protos = agg_func(protos, use_distributional=use_dist)

            # 收集本轮结果
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos                         # {client_idx: {label: proto}}

            # 记录每个客户端的训练曲线到 TensorBoard
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)   # 分类损失
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)   # 原型损失
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # ── 阶段二：差分隐私扰动（可选） ────────────────────────────
        # 在原型离开客户端之前，对 (mu, logvar) 联合向量做 L2 裁剪 + 高斯噪声
        # 注意：噪声在客户端本地添加，服务器只能看到扰动后的原型
        if use_dp:
            for idx in idxs_users:
                local_protos[idx] = dp_mech.clip_and_noise(local_protos[idx])

        # ── 更新各客户端本地模型 ──────────────────────────────────
        # 将本轮训练好的权重写回 local_model_list
        local_weights_list = local_weights
        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model


        # ── 阶段三：全局原型聚合（服务器端） ─────────────────────────
        # 收集所有客户端的本地原型，按 label 分组后融合：
        #   点原型：直接跨客户端取平均
        #   分布原型：精度加权贝叶斯融合 μ* = Σ(μ_i/σ²_i) / Σ(1/σ²_i)
        global_protos = proto_aggregation(local_protos, use_distributional=use_dist)

        # ── 更新差分隐私预算追踪 ──────────────────────────────────
        # 用 Rényi DP 追踪每轮隐私消耗，累加后转为 (ε, δ)-DP
        if use_dp and accountant is not None:
            sample_rate = dp_mech.sample_rate(len(idxs_users), args.num_users)
            rdp_eps = accountant.compute_rdp_gaussian(dp_mech.noise_multiplier, sample_rate)
            accountant.accumulate(rdp_eps)
            current_eps = accountant.get_epsilon()
            print(f'| Round {round+1} | DP epsilon: {current_eps:.4f}')

        # ── 记录本轮平均损失 ──────────────────────────────────────
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)


    # ═══════════════════════════════════════════════════════════════
    #  最终测试：在本地测试集上评估所有客户端（多标签 per-label 准确率）
    # ═══════════════════════════════════════════════════════════════
    # test_inference_new_het_lt 对每个客户端做两种测试：
    #   1. 不使用全局原型 (w/o protos)：sigmoid(logits) > 0.5 多标签分类
    #   2. 使用全局原型 (with protos)：负原型距离 → sigmoid → 二值预测
    # 对比两者的 per-label 准确率，体现全局原型带来的跨客户端知识迁移效果

    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
        args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos
    )

    print('For all users (with protos), mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list_g), np.std(acc_list_g)))
    print('For all users (w/o protos), mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std is {:.5f}'.format(
        np.mean(loss_list), np.std(loss_list)))


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    n_list = np.random.randint(
        max(2, args.ways - args.stdev),
        min(args.num_classes, args.ways + args.stdev + 1),
        args.num_users
    )

    k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev - 1, args.num_users)

    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)

    local_model_list = []
    for i in range(args.num_users):
        local_model = ResNet50(args=args)
        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
