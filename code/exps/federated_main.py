# =============================================================================
# FedCoP 联邦学习主程序入口(ChestX-ray14 + 预训练 ResNet-50)
# =============================================================================
# 这个文件是实验的"总调度"。
# 一句话概括:给定配置 → 加载数据 → 创建模型 → 选算法 → 训练 → 评估。
#
# 支持的 FL 算法公平对比(全部使用预训练骨干):
#   1. FedAvg     — 经典联邦平均(McMahan et al., 2017)
#   2. FedProx    — 近端正则联邦学习(Li et al., 2020),经典强基线
#   3. FedProto   — 联邦原型学习(Tan et al., 2022),FedCoP 的直接基线
#   4. FedCoP     — 提出的方法 ★(共现感知分布原型)
#   5. FedGMKD    — GMM 原型 + 差异感知聚合(NeurIPS 2024)
#   6. FedBCS     — 频域风格重校准(AAAI 2026)
#   7. FedSeProto — 语义-域特征解耦(ECAI 2024)
#
# 运行示例:
#   # FedCoP 完整方法(默认)
#   python federated_main.py --alg fedcop
#
#   # FedProto 基线
#   python federated_main.py --alg fedproto
#
#   # 消融:关闭共现结构(R=I)
#   python federated_main.py --alg fedcop --no_cooccurrence
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
import random

from update import test_inference_FedCoP
from options import args_parser
from update import (LocalUpdate, test_inference_new_het_lt,
                    eval_clients_multilabel,
                    _update_weights_FedGMKD, _agg_func_FedGMKD,
                    _proto_aggregation_FedGMKD,
                    _update_weights_FedBCS,
                    _update_weights_FedSeProto)
from models.resnet import FedCoPResNet, ResNet50
from utils import (get_dataset, average_weights,
                   exp_details, proto_aggregation, agg_func)
from dist_proto.structured import (compute_local_cooc, fuse_cooccurrence,
                                    ema_correlation)


# ═══════════════════════════════════════════════════════════════════════════════
#  FedProto — 联邦原型学习基线
# ═══════════════════════════════════════════════════════════════════════════════

def FedProto_taskheter(args, train_dataset, test_dataset, user_groups,
                       user_groups_lt, local_model_list, classes_list):
    """FedProto 联邦原型学习基线(点原型,算术平均聚合)"""
    use_dist = getattr(args, 'use_distributional', False)

    global_protos = []
    train_loss = []
    m = max(1, int(args.frac * args.num_users))

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        idxs_users = np.random.choice(args.num_users, m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_FedP(
                args, idx, global_protos,
                model=copy.deepcopy(local_model_list[idx]),
                global_round=round, ld=args.ld)

            agg_protos = agg_func(protos, use_distributional=use_dist)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos

        for i_w, idx in enumerate(idxs_users):
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights[i_w], strict=True)
            local_model_list[idx] = local_model

        global_protos = proto_aggregation(local_protos, use_distributional=use_dist)
        train_loss.append(sum(local_losses) / len(local_losses))

    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
        args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)

    print('For all users (with protos), mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list_g), np.std(acc_list_g)))
    print('For all users (w/o protos), mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list_l), np.std(acc_list_l)))
    # 保存全局原型供 t-SNE 可视化对比
    try:
        from visualize import save_protos_npy
        save_protos_npy(global_protos, args.alg, args.num_classes,
                        proto_dir=getattr(args, 'proto_dir', None) or './protos_vis')
    except Exception as _e:
        print(f"[vis] skip save protos for {args.alg}: {_e}")
    return acc_list_l, acc_list_g


# ═══════════════════════════════════════════════════════════════════════════════
#  ★★★ FedCoP — 提出方法:共现感知分布原型 ★★★
# ═══════════════════════════════════════════════════════════════════════════════

def FedCoP_taskheter(args, train_dataset, test_dataset, user_groups,
                     user_groups_lt, local_model_list, classes_list):
    """FedCoP: Federated Co-occurrence-aware Prototypes

    三项支柱:
      1. 分布原型 + 贝叶斯融合(沿用,逐类对角高斯)
      2. 联邦共现结构:从各客户端标签统计聚合 14×14 共现相关矩阵 R̂
         —— Non-IID 下每客户端只见 ~3/14 类,只有联邦聚合能恢复全局共现
      3. 相关性感知:训练侧 L_co(原型余弦 Gram 对齐 R̂)+ 推理侧 mean-field 解码

    每轮流程:
      服务器:
        1. 采样 C×K 个客户端
        2. 下发全局原型 + 全局 R̂
        3. 收集本地原型(贝叶斯融合)+ 本地共现统计(计数聚合 → R̂)
        4. 原型 EMA + R̂ EMA 平滑
      客户端(LocalUpdate.update_weights_FedCoP):
        L = L_CE + λ·L_proto + λ_co·L_co + λ_ent·L_ent
    """
    use_dist = True                              # FedCoP 始终分布原型
    proto_momentum = getattr(args, 'proto_momentum', 0.9)
    ld_warmup = getattr(args, 'ld_warmup', 50)
    temperature = getattr(args, 'temperature', 1.0)
    shrinkage = getattr(args, 'cov_shrinkage', 0.1)
    co_rank = getattr(args, 'co_rank', 0)
    no_cooc = getattr(args, 'no_cooccurrence', False)
    local_only = getattr(args, 'local_cooc_only', False)
    co_warmup = getattr(args, 'co_warmup', 0)

    global_protos = []                           # 全局分布原型 {label:(mu,logvar)}
    global_R = None                              # 全局共现相关矩阵 R̂
    global_pi = None                             # 全局边际先验
    train_loss = []
    m = max(1, int(args.frac * args.num_users))

    # 客户端标签矩阵缓存(用于精确计算本地共现统计;各客户端不变,首用到时算)
    client_labels_cache = {}

    def _client_labels(idx):
        if idx not in client_labels_cache:
            # train_dataset.labels: (N_total, C) 多热;取该客户端的子集
            client_labels_cache[idx] = train_dataset.labels[user_groups[idx]]
        return client_labels_cache[idx]

    # ═══════════════════════════════════════════════════════════════
    #  联邦训练主循环
    # ═══════════════════════════════════════════════════════════════
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        local_cooc_stats = []                    # 本轮各参与客户端的共现统计
        print(f'\n | Global Training Round : {round + 1} |\n')

        idxs_users = np.random.choice(args.num_users, m, replace=False)

        # 自适应原型损失权重 warmup
        ld = args.ld * min(1.0, (round + 1) / max(ld_warmup, 1))

        # 训练用的 R̂:首轮 / no_cooc 消融 / warmup 未到 时为 None(L_co 退化)
        # warmup:前 co_warmup 轮 R̂ 还是噪声,不下发,等稳定后再开 L_co
        co_ready = (global_R is not None and not no_cooc and round >= co_warmup)
        R_for_train = global_R if co_ready else None

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_FedCoP(
                args, idx, global_protos, R_for_train,
                model=copy.deepcopy(local_model_list[idx]),
                global_round=round, ld=ld)

            agg_protos = agg_func(protos, use_distributional=use_dist)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos

            # 本地共现统计(纯标签,隐私安全)——代表客户端"上传"
            local_cooc_stats.append(compute_local_cooc(_client_labels(idx)))

        # 更新本地模型
        for i_w, idx in enumerate(idxs_users):
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights[i_w], strict=True)
            local_model_list[idx] = local_model

        # ── 服务器:分布原型贝叶斯融合 ──
        new_global_protos = proto_aggregation(local_protos, use_distributional=use_dist)

        # ── 服务器:共现结构聚合 → R̂ ──
        # no_cooc 消融:强制 R̂=I(共现结构关闭),不聚合
        if no_cooc:
            new_R = torch.eye(args.num_classes)
            new_pi = torch.zeros(args.num_classes)
        else:
            new_R, new_pi = fuse_cooccurrence(local_cooc_stats, args.num_classes,
                                              shrinkage=shrinkage, rank=co_rank)

        # ── EMA 平滑(原型 + R̂)──
        if proto_momentum > 0 and len(global_protos) > 0:
            # 原型 EMA(对角高斯:μ 和 logvar 线性插值)
            for label in new_global_protos:
                if label in global_protos:
                    new_mu, new_lv = new_global_protos[label]
                    old_mu, old_lv = global_protos[label]
                    mu_ema = proto_momentum * old_mu + (1 - proto_momentum) * new_mu
                    lv_ema = proto_momentum * old_lv + (1 - proto_momentum) * new_lv
                    new_global_protos[label] = (mu_ema, lv_ema)
            # R̂ EMA
            new_R = ema_correlation(global_R, new_R, proto_momentum)

        global_protos = new_global_protos
        global_R = new_R
        global_pi = new_pi

        train_loss.append(sum(local_losses) / len(local_losses))

        # 每 10 轮打印一次 R̂ 非对角平均(诊断共现结构是否建立)
        if (round + 1) % 10 == 0 and global_R is not None:
            off = global_R - torch.eye(args.num_classes)
            print(f'  [co-occurrence] mean |off-diag R| = {off.abs().mean().item():.4f}')

    # ── 最终评估(相关性感知 mean-field 解码 + 完整指标)──
    # local_cooc_only 消融:为每个客户端构造本地 R_k(不联邦聚合)
    local_R_dict = None
    if local_only:
        local_R_dict = {}
        for idx in range(args.num_users):
            stat = compute_local_cooc(_client_labels(idx))
            Rk, pik = fuse_cooccurrence([stat], args.num_classes,
                                        shrinkage=shrinkage, rank=co_rank)
            local_R_dict[idx] = (Rk, pik)

    acc_list_l, acc_list_g, loss_list, metrics_g = test_inference_FedCoP(
        args, local_model_list, test_dataset, classes_list, user_groups_lt,
        global_protos, global_R=global_R, global_pi=global_pi,
        local_R_dict=local_R_dict, temperature=temperature)

    print('For all users (with protos), mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list_g), np.std(acc_list_g)))
    print('For all users (w/o protos), mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std is {:.5f}'.format(
        np.mean(loss_list), np.std(loss_list)))
    # 保存全局原型供 t-SNE 可视化对比
    try:
        from visualize import save_protos_npy
        save_protos_npy(global_protos, args.alg, args.num_classes,
                        proto_dir=getattr(args, 'proto_dir', None) or './protos_vis')
    except Exception as _e:
        print(f"[vis] skip save protos for {args.alg}: {_e}")

    return acc_list_l, acc_list_g


# ═══════════════════════════════════════════════════════════════════════════════
#  FedAvg — 经典联邦平均
# ═══════════════════════════════════════════════════════════════════════════════

def FedAvg_taskheter(args, train_dataset, test_dataset, user_groups,
                     user_groups_lt, local_model_list, _classes_list):
    """FedAvg:本地 SGD + 服务器等权平均(无原型机制)"""
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
            w, loss, acc = local_update.update_weights(
                idx, copy.deepcopy(global_model), global_round=round)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        global_weight = average_weights(local_weights)
        global_model.load_state_dict(global_weight)
        for i in range(args.num_users):
            local_model_list[i] = copy.deepcopy(global_model)

        train_loss.append(sum(local_losses) / len(local_losses))

    acc_list = eval_clients_multilabel(args, local_model_list, test_dataset, user_groups_lt)
    # 保存全局原型供 t-SNE 可视化对比(FedAvg/FedProx 无全局原型,自动跳过)
    try:
        from visualize import save_protos_npy
        save_protos_npy(global_protos, args.alg, args.num_classes,
                        proto_dir=getattr(args, 'proto_dir', None) or './protos_vis')
    except NameError:
        pass  # 该算法无 global_protos(如 FedAvg/FedProx),跳过
    except Exception as _e:
        print(f"[vis] skip save protos for {args.alg}: {_e}")
    print('For all users, mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list), np.std(acc_list)))
    return acc_list


# ═══════════════════════════════════════════════════════════════════════════════
#  FedProx — 近端正则联邦学习(Li et al., 2020)
# ═══════════════════════════════════════════════════════════════════════════════

def FedProx_taskheter(args, train_dataset, test_dataset, user_groups,
                      user_groups_lt, local_model_list, _classes_list):
    """FedProx:FedAvg + 近端项 (μ/2)·‖w−w_global‖²,Non-IID 下更稳定"""
    global_model = copy.deepcopy(local_model_list[0])
    global_model.to(args.device)
    train_loss = []
    m = max(1, int(args.frac * args.num_users))

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round + 1} |\n')

        idxs_users = np.random.choice(args.num_users, m, replace=False)
        global_state = copy.deepcopy(global_model.state_dict())

        for idx in idxs_users:
            local_update = LocalUpdate(args=args, dataset=train_dataset,
                                       idxs=user_groups[idx])
            w, loss, acc = local_update.update_weights_fedprox(
                idx, global_state, copy.deepcopy(global_model), global_round=round)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        global_weight = average_weights(local_weights)
        global_model.load_state_dict(global_weight)
        for i in range(args.num_users):
            local_model_list[i] = copy.deepcopy(global_model)

        train_loss.append(sum(local_losses) / len(local_losses))

    acc_list = eval_clients_multilabel(args, local_model_list, test_dataset, user_groups_lt)
    # 保存全局原型供 t-SNE 可视化对比(FedAvg/FedProx 无全局原型,自动跳过)
    try:
        from visualize import save_protos_npy
        save_protos_npy(global_protos, args.alg, args.num_classes,
                        proto_dir=getattr(args, 'proto_dir', None) or './protos_vis')
    except NameError:
        pass  # 该算法无 global_protos(如 FedAvg/FedProx),跳过
    except Exception as _e:
        print(f"[vis] skip save protos for {args.alg}: {_e}")
    print('For all users, mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list), np.std(acc_list)))
    return acc_list


# ═══════════════════════════════════════════════════════════════════════════════
#  FedGMKD (NeurIPS 2024): GMM-based Prototype FL
# ═══════════════════════════════════════════════════════════════════════════════

def FedGMKD_taskheter(args, train_dataset, test_dataset, user_groups,
                       user_groups_lt, local_model_list, classes_list):
    """FedGMKD:GMM 后处理原型 + Discrepancy-Aware 聚合"""
    use_dist = getattr(args, 'use_distributional', True)
    n_gmm = getattr(args, 'gmm_components', 3)

    global_protos = []
    train_loss = []
    m = max(1, int(args.frac * args.num_users))

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

            agg_protos = _agg_func_FedGMKD(protos, n_components=n_gmm)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos

        for i_w, idx in enumerate(idxs_users):
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights[i_w], strict=True)
            local_model_list[idx] = local_model

        global_protos = _proto_aggregation_FedGMKD(local_protos)
        train_loss.append(sum(local_losses) / len(local_losses))

    acc_list = eval_clients_multilabel(args, local_model_list, test_dataset, user_groups_lt)
    # 保存全局原型供 t-SNE 可视化对比(FedAvg/FedProx 无全局原型,自动跳过)
    try:
        from visualize import save_protos_npy
        save_protos_npy(global_protos, args.alg, args.num_classes,
                        proto_dir=getattr(args, 'proto_dir', None) or './protos_vis')
    except NameError:
        pass  # 该算法无 global_protos(如 FedAvg/FedProx),跳过
    except Exception as _e:
        print(f"[vis] skip save protos for {args.alg}: {_e}")
    print('For all users, mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list), np.std(acc_list)))
    return acc_list


# ═══════════════════════════════════════════════════════════════════════════════
#  FedBCS (AAAI 2026): Frequency-Domain Style Recalibration
# ═══════════════════════════════════════════════════════════════════════════════

def FedBCS_taskheter(args, train_dataset, test_dataset, user_groups,
                      user_groups_lt, local_model_list, classes_list):
    """FedBCS:频域风格重校准 + 域不变原型"""
    use_dist = getattr(args, 'use_distributional', False)

    global_protos = []
    train_loss = []
    m = max(1, int(args.frac * args.num_users))

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

        for i_w, idx in enumerate(idxs_users):
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights[i_w], strict=True)
            local_model_list[idx] = local_model

        global_protos = proto_aggregation(local_protos, use_distributional=use_dist)
        train_loss.append(sum(local_losses) / len(local_losses))

    acc_list = eval_clients_multilabel(args, local_model_list, test_dataset, user_groups_lt)
    # 保存全局原型供 t-SNE 可视化对比(FedAvg/FedProx 无全局原型,自动跳过)
    try:
        from visualize import save_protos_npy
        save_protos_npy(global_protos, args.alg, args.num_classes,
                        proto_dir=getattr(args, 'proto_dir', None) or './protos_vis')
    except NameError:
        pass  # 该算法无 global_protos(如 FedAvg/FedProx),跳过
    except Exception as _e:
        print(f"[vis] skip save protos for {args.alg}: {_e}")
    print('For all users, mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list), np.std(acc_list)))
    return acc_list


# ═══════════════════════════════════════════════════════════════════════════════
#  FedSeProto (ECAI 2024): Semantic-Domain Feature Decoupling
# ═══════════════════════════════════════════════════════════════════════════════

def FedSeProto_taskheter(args, train_dataset, test_dataset, user_groups,
                          user_groups_lt, local_model_list, classes_list):
    """FedSeProto:语义-域特征解耦 + 仅共享语义原型"""
    global_protos = []
    train_loss = []
    m = max(1, int(args.frac * args.num_users))

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

        for i_w, idx in enumerate(idxs_users):
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights[i_w], strict=True)
            local_model_list[idx] = local_model

        global_protos = proto_aggregation(local_protos, use_distributional=False)
        train_loss.append(sum(local_losses) / len(local_losses))

    acc_list = eval_clients_multilabel(args, local_model_list, test_dataset, user_groups_lt)
    # 保存全局原型供 t-SNE 可视化对比(FedAvg/FedProx 无全局原型,自动跳过)
    try:
        from visualize import save_protos_npy
        save_protos_npy(global_protos, args.alg, args.num_classes,
                        proto_dir=getattr(args, 'proto_dir', None) or './protos_vis')
    except NameError:
        pass  # 该算法无 global_protos(如 FedAvg/FedProx),跳过
    except Exception as _e:
        print(f"[vis] skip save protos for {args.alg}: {_e}")
    print('For all users, mean of per-label acc is {:.5f}, std is {:.5f}'.format(
        np.mean(acc_list), np.std(acc_list)))
    return acc_list


# ═══════════════════════════════════════════════════════════════════════════════
#  主入口:算法调度 + 数据加载 + 模型创建
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    start_time = time.time()

    # ── 解析命令行参数 ──
    args = args_parser()
    exp_details(args)

    # ── 设备配置 ──
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(int(args.gpu))
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── GPU 确认(一眼看出是否在用 GPU,避免 CPU 上慢跑)──
    if args.device == 'cuda':
        print(f'[GPU] CUDA OK | device={torch.cuda.get_device_name(torch.cuda.current_device())} '
              f'| idx={int(args.gpu)} | PyTorch={torch.__version__}')
    else:
        print('[GPU] WARNING: CUDA 不可用,将在 CPU 上运行(速度会慢几十倍,建议用 GPU 跑)')

    # ── Non-IID 数据划分参数 ──
    # n_list:每个客户端拥有的类别数;k_list:每类样本数
    ways_low = max(2, args.ways - args.stdev)
    ways_high = max(ways_low + 1, min(args.num_classes, args.ways + args.stdev + 1))
    n_list = np.random.randint(ways_low, ways_high, args.num_users)

    shots_low = max(1, args.shots - args.stdev + 1)
    shots_high = max(shots_low + 1, args.shots + args.stdev - 1)
    k_list = np.random.randint(shots_low, shots_high, args.num_users)

    # ── 加载数据集(ChestX-ray14)──
    train_dataset, test_dataset, user_groups, user_groups_lt, \
        classes_list, classes_list_gt = get_dataset(args, n_list, k_list)

    # ── 算法特定参数(必须在模型创建前设置,ResNet50.__init__ 读 use_distributional)──
    # FedCoP / FedGMKD 始终分布原型;FedProto/FedBCS/FedSeProto 点原型
    if args.alg in ('fedcop', 'fedgmkd'):
        args.use_distributional = True
    elif args.alg in ('fedproto', 'fedbcs', 'fedseproto'):
        args.use_distributional = False

    # ── 创建模型 ──
    # FedCoP 用 FedCoPResNet(预训练+概率原型头);其他算法用 ResNet50
    local_model_list = []
    for i in range(args.num_users):
        if args.alg == "fedcop":
            local_model = FedCoPResNet(args=args)
        else:
            local_model = ResNet50(args=args)
        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    print(f'\n=== Running {args.alg.upper()} ===\n')

    # ── 算法调度 ──
    if args.alg == 'fedavg':
        FedAvg_taskheter(args, train_dataset, test_dataset, user_groups,
                         user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'fedprox':
        FedProx_taskheter(args, train_dataset, test_dataset, user_groups,
                          user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'fedproto':
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups,
                           user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'fedcop':
        FedCoP_taskheter(args, train_dataset, test_dataset, user_groups,
                         user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'fedgmkd':
        FedGMKD_taskheter(args, train_dataset, test_dataset, user_groups,
                          user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'fedbcs':
        FedBCS_taskheter(args, train_dataset, test_dataset, user_groups,
                         user_groups_lt, local_model_list, classes_list)
    elif args.alg == 'fedseproto':
        FedSeProto_taskheter(args, train_dataset, test_dataset, user_groups,
                             user_groups_lt, local_model_list, classes_list)

    print(f'\nTotal time: {time.time() - start_time:.2f}s')
