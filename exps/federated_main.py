# =============================================================================
# D²-FL 联邦学习主程序入口（ChestX-ray14 + 预训练 ResNet-50）
# =============================================================================
# 这个文件是实验的"总调度"。
# 一句话概括：给定配置 → 加载数据 → 创建模型 → 选算法 → 训练 → 评估。
#
# 支持 6 种 FL 算法的公平对比（全部使用预训练骨干 + 可选差分隐私）：
#   1. FedAvg    — 经典联邦平均（McMahan et al., 2017）
#   2. FedProto  — 联邦原型学习（Tan et al., 2022），D²-FL 的直接基线
#   3. D²-FL     — 提出的方法 ★
#   4. FedGMKD   — GMM 原型 + 差异感知聚合（NeurIPS 2024）
#   5. FedBCS    — 频域风格重校准（AAAI 2026）
#   6. FedSeProto — 语义-域特征解耦（ECAI 2024）
#
# 运行示例：
#   # 最简运行（D²-FL 完整模式）
#   python federated_main.py --alg d2fl --use_distributional --use_disentangle
#
#   # FedProto 基线
#   python federated_main.py --alg fedproto
#
#   # 带差分隐私
#   python federated_main.py --alg d2fl --use_dp --dp_epsilon 8
#
#   # 消融实验：仅分布原型，无解耦
#   python federated_main.py --alg d2fl --use_distributional
# =============================================================================

import copy, sys
import time
from pathlib import Path

# sys.path 必须在其他 lib import 之前设置
lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random

from lib.update import test_inference_new_het_lt_D2FL
from options import args_parser
from update import (LocalUpdate, test_inference_new_het_lt,
                    eval_clients_multilabel,
                    _update_weights_FedGMKD, _agg_func_FedGMKD,
                    _proto_aggregation_FedGMKD,
                    _update_weights_FedBCS,
                    _update_weights_FedSeProto)
from models.resnet import D2FLResNet, ResNet50
from utils import (get_dataset, average_weights,
                   exp_details, proto_aggregation, agg_func)
from dp import (DPMechProto, DPMechWeight, MomentsAccountant,
                compute_noise_multiplier_from_epsilon)


# ═══════════════════════════════════════════════════════════════════════════════
#  FedProto — 联邦原型学习基线
# ═══════════════════════════════════════════════════════════════════════════════

def FedProto_taskheter(args, train_dataset, test_dataset, user_groups,
                       user_groups_lt, local_model_list, classes_list):
    """FedProto 联邦原型学习基线

    这是 D²-FL 的直接对比基线。和 D²-FL 的核心区别：
    - 点原型（向量取平均）vs 分布原型（贝叶斯融合）
    - 无原型解耦 vs 语义-风格分离
    - 无 EMA 动量 vs 原型平滑
    - 无温度缩放 vs 自适应温度

    每轮联邦学习流程：
        服务器端（本函数内）：
          1. 随机采样 C×K 个客户端参与本轮
          2. 将全局原型下发给这些客户端
          3. 等待客户端本地训练完成
          4. 收集并聚合所有客户端上传的本地原型 → 新的全局原型
          5. （可选）对上传原型加差分隐私噪声

        客户端（LocalUpdate.update_weights_FedP）：
          1. 接收全局原型
          2. 本地训练：BCE loss + λ × MSE(本地原型, 全局原型)
          3. 聚合本地原型 → 上传

    参数:
        args:             全局配置
        train_dataset:    训练数据集
        test_dataset:     测试数据集
        user_groups:      训练数据客户端划分
        user_groups_lt:   测试数据客户端划分
        local_model_list: 所有客户端的模型列表（就地更新）
        classes_list:     每个客户端拥有的类别列表

    返回:
        acc_list_l: 各客户端不使用全局原型的准确率
        acc_list_g: 各客户端使用全局原型最近邻分类的准确率
    """
    use_dist = getattr(args, 'use_distributional', False)
    use_dp = getattr(args, 'use_dp', False)

    # TensorBoard 日志目录
    summary_writer = SummaryWriter(
        f'../tensorboard/chestxray_{args.alg}_'
        f'{args.ways}w{args.shots}s{args.stdev}e_'
        f'{args.num_users}u_{args.rounds}r'
    )

    global_protos = []  # 全局原型列表（初始为空，第一轮不计算原型损失）
    train_loss = []
    m = max(1, int(args.frac * args.num_users))  # 每轮参与的客户端数

    # ── 差分隐私初始化 ──
    if use_dp:
        accountant = MomentsAccountant(delta=args.dp_delta)
        per_round_noise = compute_noise_multiplier_from_epsilon(
            args.dp_epsilon / args.rounds, sample_rate=1.0, delta=args.dp_delta)
        dp_mech = DPMechProto(clip_norm=args.dp_clip,
                              noise_multiplier=per_round_noise, use_dp=True)
        print(f'DP enabled: target_epsilon={args.dp_epsilon}, noise_multiplier={per_round_noise:.4f}')
    else:
        accountant = None
        dp_mech = DPMechProto(use_dp=False)

    # ═══════════════════════════════════════════════════════════════
    #  联邦训练主循环
    # ═══════════════════════════════════════════════════════════════
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        # 步骤1：随机采样 C×K 个参与客户端
        idxs_users = np.random.choice(args.num_users, m, replace=False)
        proto_loss = 0

        # 步骤2：每个参与客户端进行本地训练
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_FedP(
                args, idx, global_protos,
                model=copy.deepcopy(local_model_list[idx]),
                global_round=round, ld=args.ld)

            # 本地原型聚合（同一客户端的多个样本 → 每个标签一个原型）
            agg_protos = agg_func(protos, use_distributional=use_dist)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos

            # 记录到 TensorBoard
            summary_writer.add_scalar(f'Train/Loss/user{idx+1}', loss['total'], round)
            summary_writer.add_scalar(f'Train/Loss1/user{idx+1}', loss['1'], round)
            summary_writer.add_scalar(f'Train/Loss2/user{idx+1}', loss['2'], round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)
            proto_loss += loss['2']

        # 步骤3（可选）：对上传的原型加差分隐私噪声
        if use_dp:
            for idx in idxs_users:
                local_protos[idx] = dp_mech.clip_and_noise(local_protos[idx])

        # 步骤4：更新本地模型（加载本地训练后的权重）
        for i_w, idx in enumerate(idxs_users):
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights[i_w], strict=True)
            local_model_list[idx] = local_model

        # 步骤5：服务器端全局原型聚合
        # 点原型 → 算术平均，分布原型 → 贝叶斯融合
        global_protos = proto_aggregation(local_protos, use_distributional=use_dist)

        # 步骤6（可选）：更新差分隐私追踪
        if use_dp and accountant is not None:
            sample_rate = dp_mech.sample_rate(len(idxs_users), args.num_users)
            rdp_eps = accountant.compute_rdp_gaussian(dp_mech.noise_multiplier, sample_rate)
            accountant.accumulate(rdp_eps)
            print(f'| Round {round+1} | DP epsilon: {accountant.get_epsilon():.4f}')

        train_loss.append(sum(local_losses) / len(local_losses))

    # ── 最终测试 ──
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
        args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)

    print('For all users (with protos), mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list_g), np.std(acc_list_g)))
    print('For all users (w/o protos), mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std is {:.5f}'.format(
        np.mean(loss_list), np.std(loss_list)))

    return acc_list_l, acc_list_g


# ═══════════════════════════════════════════════════════════════════════════════
#  ★★★ D²-FL — 提出方法 ★★★
# ═══════════════════════════════════════════════════════════════════════════════

def D2FL_taskheter(args, train_dataset, test_dataset, user_groups,
                    user_groups_lt, local_model_list, classes_list):
    """D²-FL: Distributional Pathology Prototype Federated Learning

    这是本项目的核心算法函数，相比 FedProto 基线增加了 6 项 SOTA 提升：

    ┌─────────────────────────────────────────────────────────────┐
    │ 组件                         | 作用                        │
    ├─────────────────────────────────────────────────────────────┤
    │ 1. ImageNet 预训练骨干        | 更好的初始特征（迁移学习）    │
    │ 2. 分布原型 + 贝叶斯融合      | 编码不确定性，精度加权聚合    │
    │ 3. 原型 EMA 动量              | 平滑全局原型更新（防震荡）    │
    │ 4. 自适应权重 warmup          | 逐步增加原型损失权重          │
    │ 5. 温度缩放原型推理           | 控制推理锐度                  │
    │ 6. 原型解耦（语义-风格分离）   | 过滤客户端特有噪声            │
    │    ├─ 可学习门控              | 软分割替代硬分割              │
    │    ├─ HSIC 独立性             | 统计去相关                    │
    │    ├─ 对抗域不变              | 洗掉域信息                    │
    │    └─ 对比语义对齐            | 同类拉近、异类推远            │
    │ 7. 每类可学习温度             | 自适应推理锐度               │
    │ 8. 可选差分隐私               | 保护上传的原型               │
    └─────────────────────────────────────────────────────────────┘

    和 FedProto 的流程区别：
      FedProto: 点原型 MSE → 算术平均聚合 → 直接分类
      D²-FL:   分布原型 KL → 贝叶斯融合 → EMA平滑 → 温度缩放 → 解耦过滤

    参数:
        args, train_dataset, test_dataset, user_groups,
        user_groups_lt, local_model_list, classes_list
        （与 FedProto_taskheter 相同）

    返回:
        acc_list_l: per-label 准确率（模型自身分类器）
        acc_list_g: per-label 准确率（全局原型最近邻分类）
    """
    # ── 读取配置 ──
    use_dist = getattr(args, 'use_distributional', True)    # 分布原型（核心创新）
    use_dp = getattr(args, 'use_dp', False)                 # 差分隐私
    use_dis = getattr(args, 'use_disentangle', True)         # 原型解耦（核心创新）
    dis_lambda = getattr(args, 'dis_lambda', 0.05)          # 解耦损失权重
    proto_momentum = getattr(args, 'proto_momentum', 0.9)   # EMA 动量系数
    ld_warmup = getattr(args, 'ld_warmup', 50)              # 原型损失 warmup 轮数
    temperature = getattr(args, 'temperature', 1.0)          # 推理温度

    
    cal_lambda = getattr(args, 'cal_lambda', 0.01)          # 校准损失权重
    contra_lambda = getattr(args, 'contra_lambda', 0.05)     # 对比损失权重
    adv_lambda = getattr(args, 'adv_lambda', 0.01)           # 对抗损失权重
    ent_lambda = getattr(args, 'ent_lambda', 0.001)          # 熵正则权重

    # TensorBoard 日志目录
    suffix = '_dis' if use_dis else ''  # 解耦模式加后缀区分
    summary_writer = SummaryWriter(
        f'../tensorboard/chestxray_{args.alg}{suffix}_'
        f'{args.ways}w{args.shots}s{args.stdev}e_'
        f'{args.num_users}u_{args.rounds}r'
    )

    if use_dis:
        sem_dim = int(getattr(args, 'proto_dim', 256) * getattr(args, 'sem_ratio', 0.75))
        print(f'Prototype Disentanglement ENABLED: sem_dim={sem_dim}, dis_lambda={dis_lambda}')

    # ── 初始化全局原型 ──
    # global_protos: 当前轮的全局原型（用于本地训练的原型损失计算）
    # global_protos_ema: EMA 累积的全局原型（用于稳定的聚合更新）
    # 两者分开：训练用 global_protos，聚合时用 EMA 平滑版本
    global_protos = []
    global_protos_ema = {}
    train_loss = []
    m = max(1, int(args.frac * args.num_users))  # 每轮参与的客户端数

    # ── 差分隐私初始化 ──
    if use_dp:
        accountant = MomentsAccountant(delta=args.dp_delta)
        # 每轮预算 = 总预算 / 总轮数
        per_round_noise = compute_noise_multiplier_from_epsilon(
            args.dp_epsilon / args.rounds, sample_rate=1.0, delta=args.dp_delta)
        dp_mech = DPMechProto(clip_norm=args.dp_clip,
                              noise_multiplier=per_round_noise, use_dp=True)
        print(f'DP enabled: target_epsilon={args.dp_epsilon}, noise_multiplier={per_round_noise:.4f}')
    else:
        accountant = None
        dp_mech = DPMechProto(use_dp=False)

    # ═══════════════════════════════════════════════════════════════
    #  联邦训练主循环
    # ═══════════════════════════════════════════════════════════════
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        # 步骤1：随机采样 C×K 个参与客户端
        idxs_users = np.random.choice(args.num_users, m, replace=False)
        # 累计各类损失（用于 TensorBoard）
        proto_loss = 0
        dis_loss_sum = 0
        cal_loss_sum = 0
        contra_loss_sum = 0
        adv_loss_sum = 0
        ent_loss_sum = 0

        # ── 自适应原型损失权重 warmup ──
        # 训练初期分类器还不稳定，原型也还没收敛。
        # 如果一开始就给大的原型损失权重，会导致模型被"拉偏"。
        # warmup 策略：前 ld_warmup 轮线性增加到 args.ld，之后保持不变。
        ld = args.ld * min(1.0, (round + 1) / max(ld_warmup, 1))

        # 步骤2：每个参与客户端进行 D²-FL 本地训练
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            # ★ 核心调用：D²-FL 的 7 合 1 损失函数 + 解耦
            w, loss, acc, protos = local_model.update_weights_D2FL(
                args, idx, global_protos,
                model=copy.deepcopy(local_model_list[idx]),
                global_round=round, ld=ld)

            # 本地原型聚合（同一客户端的多个样本 → 每个标签一个原型）
            agg_protos = agg_func(protos, use_distributional=use_dist)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos

            # 记录到 TensorBoard（每个客户端独立曲线）
            summary_writer.add_scalar(f'Train/Loss/user{idx+1}', loss['total'], round)
            summary_writer.add_scalar(f'Train/Loss1-CE/user{idx+1}', loss['1'], round)
            summary_writer.add_scalar(f'Train/Loss2-Proto/user{idx+1}', loss['2'], round)
            summary_writer.add_scalar(f'Train/Loss3-Dis/user{idx+1}', loss['3'], round)
            summary_writer.add_scalar(f'Train/Loss-Cal/user{idx+1}', loss.get('cal', 0), round)
            summary_writer.add_scalar(f'Train/Loss-Contra/user{idx+1}', loss.get('contra', 0), round)
            summary_writer.add_scalar(f'Train/Loss-Adv/user{idx+1}', loss.get('adv', 0), round)
            summary_writer.add_scalar(f'Train/Loss-Ent/user{idx+1}', loss.get('ent', 0), round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)
            proto_loss += loss['2']
            dis_loss_sum += loss['3']
            cal_loss_sum += loss.get('cal', 0)
            contra_loss_sum += loss.get('contra', 0)
            adv_loss_sum += loss.get('adv', 0)
            ent_loss_sum += loss.get('ent', 0)

        # 步骤3（可选）：差分隐私
        # 解耦模式的优势：仅对语义原型加噪 → 语义维度更小 → DP 信噪比更高
        if use_dp:
            for idx in idxs_users:
                local_protos[idx] = dp_mech.clip_and_noise(local_protos[idx])

        # 步骤4：更新本地模型
        for i_w, idx in enumerate(idxs_users):
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights[i_w], strict=True)
            local_model_list[idx] = local_model

        # 步骤5：服务器端全局原型聚合（解耦模式下仅聚合语义原型）
        new_global_protos = proto_aggregation(local_protos, use_distributional=use_dist)

        # ── 步骤6：原型 EMA 动量平滑 ──
        # 为什么需要 EMA？
        # 每轮参与的客户端不同（随机采样），如果直接替换全局原型，
        # 会产生"震荡"——这轮的全局原型可能和上轮差很多。
        # EMA 平滑：新原型 = momentum × 旧原型 + (1-momentum) × 本轮聚合结果
        # momentum=0.9 意味着"90%保留旧值，10%加入新信息"
        if proto_momentum > 0 and len(global_protos_ema) > 0:
            for label in new_global_protos:
                if label in global_protos_ema:
                    if use_dist:
                        old_mu, old_logvar = global_protos_ema[label]
                        new_mu, new_logvar = new_global_protos[label]
                        mu_ema = proto_momentum * old_mu + (1 - proto_momentum) * new_mu
                        logvar_ema = proto_momentum * old_logvar + (1 - proto_momentum) * new_logvar
                        new_global_protos[label] = (mu_ema, logvar_ema)
                    else:
                        new_global_protos[label] = (proto_momentum * global_protos_ema[label]
                                                     + (1 - proto_momentum) * new_global_protos[label])

        # 更新全局原型
        global_protos = new_global_protos
        # 保存 EMA 副本（detach 切断梯度，clone 防止原地修改）
        global_protos_ema = {k: (v[0].detach().clone() if isinstance(v, tuple) else v.detach().clone())
                              for k, v in global_protos.items()}

        # 步骤7（可选）：更新差分隐私追踪
        if use_dp and accountant is not None:
            sample_rate = dp_mech.sample_rate(len(idxs_users), args.num_users)
            rdp_eps = accountant.compute_rdp_gaussian(dp_mech.noise_multiplier, sample_rate)
            accountant.accumulate(rdp_eps)
            print(f'| Round {round+1} | DP epsilon: {accountant.get_epsilon():.4f}')

        # 记录全局指标到 TensorBoard
        summary_writer.add_scalar('Train/ld', ld, round)
        if use_dis:
            summary_writer.add_scalar('Train/Loss3-Dis/mean', dis_loss_sum / len(idxs_users), round)
        if cal_lambda > 0:
            summary_writer.add_scalar('Train/Loss-Cal/mean', cal_loss_sum / len(idxs_users), round)
        if contra_lambda > 0:
            summary_writer.add_scalar('Train/Loss-Contra/mean', contra_loss_sum / len(idxs_users), round)
        if adv_lambda > 0:
            summary_writer.add_scalar('Train/Loss-Adv/mean', adv_loss_sum / len(idxs_users), round)
        if ent_lambda > 0:
            summary_writer.add_scalar('Train/Loss-Ent/mean', ent_loss_sum / len(idxs_users), round)
        train_loss.append(sum(local_losses) / len(local_losses))

    # ── 最终评估（带温度缩放 + 解耦感知）──
    # 使用 D²-FL 专用测试函数（支持分布原型马氏距离 + 温度缩放 + 解耦特征提取）
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt_D2FL(
        args, local_model_list, test_dataset, classes_list, user_groups_lt,
        global_protos, temperature=temperature)

    print('For all users (with protos), mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list_g), np.std(acc_list_g)))
    print('For all users (w/o protos), mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std is {:.5f}'.format(
        np.mean(loss_list), np.std(loss_list)))

    return acc_list_l, acc_list_g


# ═══════════════════════════════════════════════════════════════════════════════
#  FedAvg — 经典联邦平均
# ═══════════════════════════════════════════════════════════════════════════════

def FedAvg_taskheter(args, train_dataset, test_dataset, user_groups,
                     user_groups_lt, local_model_list, _classes_list):
    """FedAvg 联邦平均：本地 SGD + 服务器等权平均

    每轮流程：
      1. 采样客户端 → 本地多 epoch SGD
      2. 收集本地模型参数 → 服务器等权平均
      3.（可选）差分隐私（对权重 delta 裁剪+加噪）

    这是最基础的联邦学习算法，不涉及任何原型机制。
    """
    use_dp = getattr(args, 'use_dp', False)

    summary_writer = SummaryWriter(
        f'../tensorboard/chestxray_{args.alg}_'
        f'{args.ways}w{args.shots}s_'
        f'{args.num_users}u_{args.rounds}r'
    )

    global_model = copy.deepcopy(local_model_list[0])
    global_model.to(args.device)
    train_loss = []
    m = max(1, int(args.frac * args.num_users))

    if use_dp:
        accountant = MomentsAccountant(delta=args.dp_delta)
        per_round_noise = compute_noise_multiplier_from_epsilon(
            args.dp_epsilon / args.rounds, sample_rate=1.0, delta=args.dp_delta)
        dp_mech = DPMechWeight(clip_norm=args.dp_clip,
                               noise_multiplier=per_round_noise, use_dp=True)
        print(f'DP enabled: target_epsilon={args.dp_epsilon}, noise_multiplier={per_round_noise:.4f}')
    else:
        accountant = None
        dp_mech = DPMechWeight(use_dp=False)

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        idxs_users = np.random.choice(args.num_users, m, replace=False)
        global_state = global_model.state_dict()

        for idx in idxs_users:
            local_update = LocalUpdate(args=args, dataset=train_dataset,
                                       idxs=user_groups[idx])
            w, loss, acc = local_update.update_weights(
                idx, copy.deepcopy(global_model), global_round=round)

            # DP: 对权重 delta 裁剪 + 加噪
            w = dp_mech.clip_and_noise(w, global_state)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            summary_writer.add_scalar(f'Train/Loss/user{idx+1}', loss, round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)

        # 全局等权平均
        global_weight = average_weights(local_weights)
        global_model.load_state_dict(global_weight)

        # 所有客户端同步为全局模型（FedAvg 的标准做法）
        for i in range(args.num_users):
            local_model_list[i] = copy.deepcopy(global_model)

        if use_dp and accountant is not None:
            sample_rate = dp_mech.sample_rate(len(idxs_users), args.num_users)
            rdp_eps = accountant.compute_rdp_gaussian(dp_mech.noise_multiplier, sample_rate)
            accountant.accumulate(rdp_eps)
            print(f'| Round {round+1} | DP epsilon: {accountant.get_epsilon():.4f}')

        train_loss.append(sum(local_losses) / len(local_losses))

    acc_list = eval_clients_multilabel(
        args, local_model_list, test_dataset, user_groups_lt)

    print('For all users, mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list), np.std(acc_list)))
    return acc_list


# ═══════════════════════════════════════════════════════════════════════════════
#  FedGMKD (NeurIPS 2024): GMM-based Prototype Federated Learning
#  对比基线 vs D²-FL:
#    - GMM 后处理（EM拟合）vs 端到端 NN 输出分布参数
#    - 多分量高斯原型 vs 单高斯 + 校准
#    - Discrepancy-Aware Aggregation (质量+数量) vs Bayesian Fusion
# ═══════════════════════════════════════════════════════════════════════════════

def FedGMKD_taskheter(args, train_dataset, test_dataset, user_groups,
                       user_groups_lt, local_model_list, classes_list):
    """FedGMKD: GMM 后处理原型 + Discrepancy-Aware 聚合

    每轮流程:
      1. 客户端本地训练（L_CE + 原型对齐）
      2. 每类特征用 EM 拟合 GMM（后处理，不可端到端学习）
      3. 上传 (weights, means, logvars) 三元组
      4. 服务器 Quality-weighted 聚合（方差小→质量高→权重大）
    """
    use_dist = getattr(args, 'use_distributional', True)
    use_dp = getattr(args, 'use_dp', False)
    n_gmm = getattr(args, 'gmm_components', 3)

    summary_writer = SummaryWriter(
        f'../tensorboard/chestxray_{args.alg}_gmm{n_gmm}_'
        f'{args.ways}w{args.shots}s_'
        f'{args.num_users}u_{args.rounds}r'
    )

    global_protos = []
    train_loss = []
    m = max(1, int(args.frac * args.num_users))

    if use_dp:
        accountant = MomentsAccountant(delta=args.dp_delta)
        per_round_noise = compute_noise_multiplier_from_epsilon(
            args.dp_epsilon / args.rounds, sample_rate=1.0, delta=args.dp_delta)
        dp_mech = DPMechProto(clip_norm=args.dp_clip,
                              noise_multiplier=per_round_noise, use_dp=True)
        print(f'DP enabled: target_epsilon={args.dp_epsilon}, noise_multiplier={per_round_noise:.4f}')
    else:
        accountant = None
        dp_mech = DPMechProto(use_dp=False)

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        idxs_users = np.random.choice(args.num_users, m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w, loss, acc, protos = _update_weights_FedGMKD(
                local_model, args, idx, global_protos,
                model=copy.deepcopy(local_model_list[idx]),
                global_round=round, ld=args.ld)

            # GMM 后处理聚合（核心区别于 D²-FL 的端到端方式）
            agg_protos = _agg_func_FedGMKD(protos, n_components=n_gmm)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos

            summary_writer.add_scalar(f'Train/Loss/user{idx+1}', loss['total'], round)
            summary_writer.add_scalar(f'Train/Loss1-CE/user{idx+1}', loss['1'], round)
            summary_writer.add_scalar(f'Train/Loss2-Proto/user{idx+1}', loss['2'], round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)

        if use_dp:
            for idx in idxs_users:
                local_protos[idx] = dp_mech.clip_and_noise(local_protos[idx])

        for i_w, idx in enumerate(idxs_users):
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights[i_w], strict=True)
            local_model_list[idx] = local_model

        # Discrepancy-Aware 聚合（质量+数量联合加权）
        global_protos = _proto_aggregation_FedGMKD(local_protos)

        if use_dp and accountant is not None:
            sample_rate = dp_mech.sample_rate(len(idxs_users), args.num_users)
            rdp_eps = accountant.compute_rdp_gaussian(dp_mech.noise_multiplier, sample_rate)
            accountant.accumulate(rdp_eps)
            print(f'| Round {round+1} | DP epsilon: {accountant.get_epsilon():.4f}')

        train_loss.append(sum(local_losses) / len(local_losses))

    acc_list = eval_clients_multilabel(
        args, local_model_list, test_dataset, user_groups_lt)

    print('For all users, mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list), np.std(acc_list)))
    return acc_list


# ═══════════════════════════════════════════════════════════════════════════════
#  FedBCS (AAAI 2026): Frequency-Domain Style Recalibration
#  对比基线 vs D²-FL:
#    - 频域风格-内容分离（AdaptiveIN）vs 可学习门控解耦
#    - 单一风格重校准 vs HSIC + 对抗 + 对比三重机制
#    - 点原型 vs 分布原型
# ═══════════════════════════════════════════════════════════════════════════════

def FedBCS_taskheter(args, train_dataset, test_dataset, user_groups,
                      user_groups_lt, local_model_list, classes_list):
    """FedBCS: 频域风格重校准 + 域不变原型

    每轮流程:
      1. 客户端本地训练（L_CE + 风格重校准原型对齐）
      2. 上传重校准后的原型（已去除客户端风格偏差）
      3. 服务器标准原型聚合
    """
    use_dist = getattr(args, 'use_distributional', False)
    use_dp = getattr(args, 'use_dp', False)

    summary_writer = SummaryWriter(
        f'../tensorboard/chestxray_{args.alg}_'
        f'{args.ways}w{args.shots}s_'
        f'{args.num_users}u_{args.rounds}r'
    )

    global_protos = []
    train_loss = []
    m = max(1, int(args.frac * args.num_users))

    if use_dp:
        accountant = MomentsAccountant(delta=args.dp_delta)
        per_round_noise = compute_noise_multiplier_from_epsilon(
            args.dp_epsilon / args.rounds, sample_rate=1.0, delta=args.dp_delta)
        dp_mech = DPMechProto(clip_norm=args.dp_clip,
                              noise_multiplier=per_round_noise, use_dp=True)
        print(f'DP enabled: target_epsilon={args.dp_epsilon}, noise_multiplier={per_round_noise:.4f}')
    else:
        accountant = None
        dp_mech = DPMechProto(use_dp=False)

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        idxs_users = np.random.choice(args.num_users, m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w, loss, acc, protos = _update_weights_FedBCS(
                local_model, args, idx, global_protos,
                model=copy.deepcopy(local_model_list[idx]),
                global_round=round, ld=args.ld)

            agg_protos = agg_func(protos, use_distributional=use_dist)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos

            summary_writer.add_scalar(f'Train/Loss/user{idx+1}', loss['total'], round)
            summary_writer.add_scalar(f'Train/Loss1-CE/user{idx+1}', loss['1'], round)
            summary_writer.add_scalar(f'Train/Loss2-Proto/user{idx+1}', loss['2'], round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)

        if use_dp:
            for idx in idxs_users:
                local_protos[idx] = dp_mech.clip_and_noise(local_protos[idx])

        for i_w, idx in enumerate(idxs_users):
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights[i_w], strict=True)
            local_model_list[idx] = local_model

        global_protos = proto_aggregation(local_protos, use_distributional=use_dist)

        if use_dp and accountant is not None:
            sample_rate = dp_mech.sample_rate(len(idxs_users), args.num_users)
            rdp_eps = accountant.compute_rdp_gaussian(dp_mech.noise_multiplier, sample_rate)
            accountant.accumulate(rdp_eps)
            print(f'| Round {round+1} | DP epsilon: {accountant.get_epsilon():.4f}')

        train_loss.append(sum(local_losses) / len(local_losses))

    acc_list = eval_clients_multilabel(
        args, local_model_list, test_dataset, user_groups_lt)

    print('For all users, mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list), np.std(acc_list)))
    return acc_list


# ═══════════════════════════════════════════════════════════════════════════════
#  FedSeProto (ECAI 2024): Semantic-Domain Feature Decoupling
#  对比基线 vs D²-FL:
#    - 硬分割（独立编码器）+ 互信息最小化 vs 软门控 + 对抗 + 对比
#    - 点原型（语义特征均值）vs 端到端分布原型
#    - HSIC 独立性 vs HSIC + 门控熵 + 正交 + 对抗 + 对比
# ═══════════════════════════════════════════════════════════════════════════════

def FedSeProto_taskheter(args, train_dataset, test_dataset, user_groups,
                          user_groups_lt, local_model_list, classes_list):
    """FedSeProto: 语义-域特征解耦 + 仅共享语义原型

    每轮流程:
      1. 客户端本地训练: L_CE + L_proto(semantic) + L_MI(HSIC)
      2. 仅上传语义原型（域特征完全保留本地）
      3. 服务器标准原型聚合（点原型算术平均）
    """
    use_dp = getattr(args, 'use_dp', False)

    summary_writer = SummaryWriter(
        f'../tensorboard/chestxray_{args.alg}_'
        f'{args.ways}w{args.shots}s_'
        f'{args.num_users}u_{args.rounds}r'
    )

    global_protos = []
    train_loss = []
    m = max(1, int(args.frac * args.num_users))

    if use_dp:
        accountant = MomentsAccountant(delta=args.dp_delta)
        per_round_noise = compute_noise_multiplier_from_epsilon(
            args.dp_epsilon / args.rounds, sample_rate=1.0, delta=args.dp_delta)
        dp_mech = DPMechProto(clip_norm=args.dp_clip,
                              noise_multiplier=per_round_noise, use_dp=True)
        print(f'DP enabled: target_epsilon={args.dp_epsilon}, noise_multiplier={per_round_noise:.4f}')
    else:
        accountant = None
        dp_mech = DPMechProto(use_dp=False)

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        idxs_users = np.random.choice(args.num_users, m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w, loss, acc, protos = _update_weights_FedSeProto(
                local_model, args, idx, global_protos,
                model=copy.deepcopy(local_model_list[idx]),
                global_round=round, ld=args.ld)

            agg_protos = agg_func(protos, use_distributional=False)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos

            summary_writer.add_scalar(f'Train/Loss/user{idx+1}', loss['total'], round)
            summary_writer.add_scalar(f'Train/Loss1-CE/user{idx+1}', loss['1'], round)
            summary_writer.add_scalar(f'Train/Loss2-Proto/user{idx+1}', loss['2'], round)
            summary_writer.add_scalar(f'Train/Loss3-MI/user{idx+1}', loss['3'], round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)

        if use_dp:
            for idx in idxs_users:
                local_protos[idx] = dp_mech.clip_and_noise(local_protos[idx])

        for i_w, idx in enumerate(idxs_users):
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights[i_w], strict=True)
            local_model_list[idx] = local_model

        global_protos = proto_aggregation(local_protos, use_distributional=False)

        if use_dp and accountant is not None:
            sample_rate = dp_mech.sample_rate(len(idxs_users), args.num_users)
            rdp_eps = accountant.compute_rdp_gaussian(dp_mech.noise_multiplier, sample_rate)
            accountant.accumulate(rdp_eps)
            print(f'| Round {round+1} | DP epsilon: {accountant.get_epsilon():.4f}')

        train_loss.append(sum(local_losses) / len(local_losses))

    acc_list = eval_clients_multilabel(
        args, local_model_list, test_dataset, user_groups_lt)

    print('For all users, mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list), np.std(acc_list)))
    return acc_list


# ═══════════════════════════════════════════════════════════════════════════════
#  主入口：算法调度 + 数据加载 + 模型创建
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    start_time = time.time()

    # ── 解析命令行参数 ──
    args = args_parser()
    exp_details(args)  # 打印实验配置

    # ── 设备配置 ──
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── Non-IID 数据划分参数 ──
    # n_list: 每个客户端拥有的类别数（在 [ways-stdev, ways+stdev] 随机采样）
    # k_list: 每个客户端每类拥有的样本数
    ways_low = max(2, args.ways - args.stdev)
    ways_high = max(ways_low + 1, min(args.num_classes, args.ways + args.stdev + 1))
    n_list = np.random.randint(ways_low, ways_high, args.num_users)

    shots_low = max(1, args.shots - args.stdev + 1)
    shots_high = max(shots_low + 1, args.shots + args.stdev - 1)
    k_list = np.random.randint(shots_low, shots_high, args.num_users)

    # ── 加载数据集（ChestX-ray14）──
    train_dataset, test_dataset, user_groups, user_groups_lt, \
        classes_list, classes_list_gt = get_dataset(args, n_list, k_list)

    # ── 创建模型 ──
    # D²-FL 使用 D2FLResNet（预训练+增强头），其他算法使用 ResNet50
    local_model_list = []
    for i in range(args.num_users):
        if args.alg == "d2fl":
            # D²-FL 专属模型：ImageNet 预训练 + 增强 ProtoHead + 可选解耦头
            local_model = D2FLResNet(args=args)
            local_model.to(args.device)
            local_model.train()
            local_model_list.append(local_model)
        else:
            # 其他算法使用标准 ResNet50
            local_model = ResNet50(args=args)
            local_model.to(args.device)
            local_model.train()
            local_model_list.append(local_model)

    print(f'\n=== Running {args.alg.upper()} ===\n')

    # ── 算法调度 ──
    if args.alg == 'fedproto':
        # 基线：FedProto 点原型，不做任何增强
        args.use_distributional = False
        args.use_dp = False
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups,
                           user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'd2fl':
        # ★ 提出的方法：D²-FL，所有创新点由命令行参数控制
        D2FL_taskheter(args, train_dataset, test_dataset, user_groups,
                        user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'fedavg':
        FedAvg_taskheter(args, train_dataset, test_dataset, user_groups,
                         user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'fedgmkd':
        args.use_distributional = True
        FedGMKD_taskheter(args, train_dataset, test_dataset, user_groups,
                          user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'fedbcs':
        args.use_distributional = False
        FedBCS_taskheter(args, train_dataset, test_dataset, user_groups,
                         user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'fedseproto':
        args.use_distributional = False
        FedSeProto_taskheter(args, train_dataset, test_dataset, user_groups,
                             user_groups_lt, local_model_list, classes_list)

    print(f'\nTotal time: {time.time() - start_time:.2f}s')
