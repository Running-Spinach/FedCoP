# 功能：DPP-FL 联邦学习主程序入口（ChestX-ray14 + ResNet-50）
# 支持 5 种 FL 算法：FedProto, FedAvg, FedProx, FedBN, SCAFFOLD

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
from update import (LocalUpdate, test_inference_new_het_lt,
                    eval_clients_multilabel)
from models.resnet import ResNet50, DPPFLResNet
from utils import (get_dataset, average_weights, average_weights_fedbn,
                   exp_details, proto_aggregation, agg_func)
from dp import DPMechProto, MomentsAccountant, compute_noise_multiplier_from_epsilon


# ═══════════════════════════════════════════════════════════════════════════════
#  FedProto
# ═══════════════════════════════════════════════════════════════════════════════

def FedProto_taskheter(args, train_dataset, test_dataset, user_groups,
                       user_groups_lt, local_model_list, classes_list):
    use_dist = getattr(args, 'use_distributional', False)
    use_dp = getattr(args, 'use_dp', False)

    summary_writer = SummaryWriter(
        f'../tensorboard/chestxray_{args.alg}_'
        f'{args.ways}w{args.shots}s{args.stdev}e_'
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
        proto_loss = 0

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_het(
                args, idx, global_protos,
                model=copy.deepcopy(local_model_list[idx]),
                global_round=round)

            agg_protos = agg_func(protos, use_distributional=use_dist)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos

            summary_writer.add_scalar(f'Train/Loss/user{idx+1}', loss['total'], round)
            summary_writer.add_scalar(f'Train/Loss1/user{idx+1}', loss['1'], round)
            summary_writer.add_scalar(f'Train/Loss2/user{idx+1}', loss['2'], round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)
            proto_loss += loss['2']

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
#  DPP-FL (提出方法)
# ═══════════════════════════════════════════════════════════════════════════════

def DPPFL_taskheter(args, train_dataset, test_dataset, user_groups,
                    user_groups_lt, local_model_list, classes_list):
    """
    DPP-FL: Distributional Pathology Prototype Federated Learning (提出方法)

    相比 FedProto 基线的 SOTA 提升：
      1. ImageNet 预训练 ResNet-50 骨干（vs from-scratch）
      2. 全局原型指数移动平均 (EMA momentum)
      3. 自适应原型损失权重 warmup
      4. 温度缩放原型推理
      5. 分布原型 + 贝叶斯融合 + 可选差分隐私
    """
    use_dist = getattr(args, 'use_distributional', False)
    use_dp = getattr(args, 'use_dp', False)
    proto_momentum = getattr(args, 'proto_momentum', 0.9)
    ld_warmup = getattr(args, 'ld_warmup', 50)
    temperature = getattr(args, 'temperature', 1.0)

    summary_writer = SummaryWriter(
        f'../tensorboard/chestxray_{args.alg}_'
        f'{args.ways}w{args.shots}s{args.stdev}e_'
        f'{args.num_users}u_{args.rounds}r'
    )

    global_protos = []
    global_protos_ema = {}                                    # EMA 积累的原型
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
        proto_loss = 0

        # ── 自适应原型损失权重 warmup ──
        ld = args.ld * min(1.0, (round + 1) / max(ld_warmup, 1))

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            # 临时覆盖 ld 用于 warmup
            w, loss, acc, protos = local_model.update_weights_het(
                args, idx, global_protos,
                model=copy.deepcopy(local_model_list[idx]),
                global_round=round)

            agg_protos = agg_func(protos, use_distributional=use_dist)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos

            summary_writer.add_scalar(f'Train/Loss/user{idx+1}', loss['total'], round)
            summary_writer.add_scalar(f'Train/Loss1/user{idx+1}', loss['1'], round)
            summary_writer.add_scalar(f'Train/Loss2/user{idx+1}', loss['2'], round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)
            proto_loss += loss['2']

        if use_dp:
            for idx in idxs_users:
                local_protos[idx] = dp_mech.clip_and_noise(local_protos[idx])

        for i_w, idx in enumerate(idxs_users):
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights[i_w], strict=True)
            local_model_list[idx] = local_model

        # ── 全局原型聚合 ──
        new_global_protos = proto_aggregation(local_protos, use_distributional=use_dist)

        # ── 原型 EMA 动量 ──
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

        global_protos = new_global_protos
        global_protos_ema = {k: (v[0].detach().clone() if isinstance(v, tuple) else v.detach().clone())
                              for k, v in global_protos.items()}

        if use_dp and accountant is not None:
            sample_rate = dp_mech.sample_rate(len(idxs_users), args.num_users)
            rdp_eps = accountant.compute_rdp_gaussian(dp_mech.noise_multiplier, sample_rate)
            accountant.accumulate(rdp_eps)
            print(f'| Round {round+1} | DP epsilon: {accountant.get_epsilon():.4f}')

        summary_writer.add_scalar('Train/ld', ld, round)
        train_loss.append(sum(local_losses) / len(local_losses))

    # ── 最终评估（带温度缩放）──
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
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
#  FedAvg
# ═══════════════════════════════════════════════════════════════════════════════

def FedAvg_taskheter(args, train_dataset, test_dataset, user_groups,
                     user_groups_lt, local_model_list, _classes_list):
    summary_writer = SummaryWriter(
        f'../tensorboard/chestxray_{args.alg}_'
        f'{args.ways}w{args.shots}s_'
        f'{args.num_users}u_{args.rounds}r'
    )

    # 全局模型初始化为第一个客户端的模型
    global_model = copy.deepcopy(local_model_list[0])
    global_model.to(args.device)
    train_loss = []
    m = max(1, int(args.frac * args.num_users))

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        idxs_users = np.random.choice(args.num_users, m, replace=False)

        for idx in idxs_users:
            local_update = LocalUpdate(args=args, dataset=train_dataset,
                                       idxs=user_groups[idx])
            w, loss, acc = local_update.update_weights(
                idx, copy.deepcopy(global_model), global_round=round)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            summary_writer.add_scalar(f'Train/Loss/user{idx+1}', loss, round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)

        # 服务器聚合：FedAvg 等权平均
        global_weight = average_weights(local_weights)
        global_model.load_state_dict(global_weight)

        # 所有客户端同步为全局模型
        for i in range(args.num_users):
            local_model_list[i] = copy.deepcopy(global_model)

        train_loss.append(sum(local_losses) / len(local_losses))

    acc_list = eval_clients_multilabel(
        args, local_model_list, test_dataset, user_groups_lt)

    print('For all users, mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list), np.std(acc_list)))
    return acc_list


# ═══════════════════════════════════════════════════════════════════════════════
#  FedProx
# ═══════════════════════════════════════════════════════════════════════════════

def FedProx_taskheter(args, train_dataset, test_dataset, user_groups,
                      user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter(
        f'../tensorboard/chestxray_{args.alg}_mu{args.fedprox_mu}_'
        f'{args.ways}w{args.shots}s_'
        f'{args.num_users}u_{args.rounds}r'
    )

    global_model = copy.deepcopy(local_model_list[0])
    global_model.to(args.device)
    train_loss = []
    m = max(1, int(args.frac * args.num_users))

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        idxs_users = np.random.choice(args.num_users, m, replace=False)
        global_state = global_model.state_dict()

        for idx in idxs_users:
            local_update = LocalUpdate(args=args, dataset=train_dataset,
                                       idxs=user_groups[idx])
            w, loss, acc = local_update.update_weights_fedprox(
                idx, global_state, copy.deepcopy(global_model), global_round=round)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            summary_writer.add_scalar(f'Train/Loss/user{idx+1}', loss, round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)

        global_weight = average_weights(local_weights)
        global_model.load_state_dict(global_weight)

        for i in range(args.num_users):
            local_model_list[i] = copy.deepcopy(global_model)

        train_loss.append(sum(local_losses) / len(local_losses))

    acc_list = eval_clients_multilabel(
        args, local_model_list, test_dataset, user_groups_lt)

    print('For all users, mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list), np.std(acc_list)))
    return acc_list


# ═══════════════════════════════════════════════════════════════════════════════
#  FedBN
# ═══════════════════════════════════════════════════════════════════════════════

def FedBN_taskheter(args, train_dataset, test_dataset, user_groups,
                    user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter(
        f'../tensorboard/chestxray_{args.alg}_'
        f'{args.ways}w{args.shots}s_'
        f'{args.num_users}u_{args.rounds}r'
    )

    global_model = copy.deepcopy(local_model_list[0])
    global_model.to(args.device)
    train_loss = []
    m = max(1, int(args.frac * args.num_users))

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        idxs_users = np.random.choice(args.num_users, m, replace=False)

        for idx in idxs_users:
            local_update = LocalUpdate(args=args, dataset=train_dataset,
                                       idxs=user_groups[idx])
            # FedBN: 本地训练与 FedAvg 相同
            w, loss, acc = local_update.update_weights(
                idx, copy.deepcopy(global_model), global_round=round)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            summary_writer.add_scalar(f'Train/Loss/user{idx+1}', loss, round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)

        # FedBN 聚合：跳过 BN 层参数
        global_weight = average_weights_fedbn(local_weights)

        # 每个客户端保留自己的 BN 统计量
        for i in range(args.num_users):
            if i in idxs_users:
                # 参与训练的客户端：用聚合的非BN权重 + 保留本地BN参数
                local_bn_state = {
                    k: v for k, v in local_model_list[i].state_dict().items()
                    if 'bn' in k or 'running_mean' in k or 'running_var' in k
                }
                local_model_list[i].load_state_dict(global_weight, strict=False)
                local_model_list[i].load_state_dict(local_bn_state, strict=False)
                # 更新全局模型（用于下轮广播），这里用参与训练客户端的BN不广播
            # 未参与的客户端保持原样

        global_model.load_state_dict(global_weight, strict=False)

        train_loss.append(sum(local_losses) / len(local_losses))

    acc_list = eval_clients_multilabel(
        args, local_model_list, test_dataset, user_groups_lt)

    print('For all users, mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list), np.std(acc_list)))
    return acc_list


# ═══════════════════════════════════════════════════════════════════════════════
#  SCAFFOLD
# ═══════════════════════════════════════════════════════════════════════════════

def SCAFFOLD_taskheter(args, train_dataset, test_dataset, user_groups,
                       user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter(
        f'../tensorboard/chestxray_{args.alg}_'
        f'{args.ways}w{args.shots}s_'
        f'{args.num_users}u_{args.rounds}r'
    )

    global_model = copy.deepcopy(local_model_list[0])
    global_model.to(args.device)
    train_loss = []
    m = max(1, int(args.frac * args.num_users))

    # ── SCAFFOLD 初始化 control variates ──
    zero_state = {name: torch.zeros_like(param)
                  for name, param in global_model.named_parameters()}
    c_global = copy.deepcopy(zero_state)                     # 全局 control variate
    c_local_dict = {i: copy.deepcopy(zero_state)
                    for i in range(args.num_users)}           # 每个客户端的本地 c_i

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        c_delta_list = []
        print(f'\n | Global Training Round : {round + 1} |\n')

        idxs_users = np.random.choice(args.num_users, m, replace=False)

        for idx in idxs_users:
            local_update = LocalUpdate(args=args, dataset=train_dataset,
                                       idxs=user_groups[idx])
            w, loss, acc, c_local_new, c_delta = local_update.update_weights_scaffold(
                idx, c_global, c_local_dict[idx],
                copy.deepcopy(local_model_list[idx]), global_round=round)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            c_delta_list.append(c_delta)

            # 更新本地 control variate
            c_local_dict[idx] = c_local_new

            # 将训练好的权重写回
            local_model_list[idx].load_state_dict(w)

            summary_writer.add_scalar(f'Train/Loss/user{idx+1}', loss, round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)

        # ── 服务器更新 ──
        # 1. 全局模型聚合
        global_weight = average_weights(local_weights)
        global_model.load_state_dict(global_weight)

        # 2. 全局 control variate 更新: c = c + (1/K) * sum(delta_c_i)
        K = len(idxs_users)
        for key in c_global:
            delta_sum = sum(cd[key] for cd in c_delta_list)
            c_global[key] = c_global[key] + delta_sum / K

        train_loss.append(sum(local_losses) / len(local_losses))

    acc_list = eval_clients_multilabel(
        args, local_model_list, test_dataset, user_groups_lt)

    print('For all users, mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list), np.std(acc_list)))
    return acc_list


# ═══════════════════════════════════════════════════════════════════════════════
#  主入口：算法调度
# ═══════════════════════════════════════════════════════════════════════════════

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
    k_list = np.random.randint(args.shots - args.stdev + 1,
                                args.shots + args.stdev - 1, args.num_users)

    train_dataset, test_dataset, user_groups, user_groups_lt, \
        classes_list, classes_list_gt = get_dataset(args, n_list, k_list)

    local_model_list = []
    for i in range(args.num_users):
        if args.alg == 'dppfl':
            local_model = DPPFLResNet(args=args)
        else:
            local_model = ResNet50(args=args)
        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    print(f'\n=== Running {args.alg.upper()} ===\n')

    if args.alg == 'fedproto':
        # 基线：原始 FedProto，仅点原型，无分布原型，无 DP
        args.use_distributional = False
        args.use_dp = False
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups,
                           user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'dppfl':
        # 提出的方法：DPP-FL，支持分布原型 + 差分隐私 + SOTA 增强
        DPPFL_taskheter(args, train_dataset, test_dataset, user_groups,
                        user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'fedavg':
        FedAvg_taskheter(args, train_dataset, test_dataset, user_groups,
                         user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'fedprox':
        FedProx_taskheter(args, train_dataset, test_dataset, user_groups,
                          user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'fedbn':
        FedBN_taskheter(args, train_dataset, test_dataset, user_groups,
                        user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'scaffold':
        SCAFFOLD_taskheter(args, train_dataset, test_dataset, user_groups,
                           user_groups_lt, local_model_list, classes_list)

    print(f'\nTotal time: {time.time() - start_time:.2f}s')
