# 功能：分布原型损失函数，支持KL散度、Wasserstein距离、MSE距离
#       以及原型不确定性校准损失

import torch
import torch.nn.functional as F


def kl_divergence_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    """
    计算两个高斯分布之间的KL散度 KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2))

    KL = 0.5 * [log(sigma_p^2 / sigma_q^2)
                 + (sigma_q^2 + (mu_q - mu_p)^2) / sigma_p^2 - 1]

    参数:
        mu_q: 分布Q的均值 (batch, dim)
        logvar_q: 分布Q的对数方差 (batch, dim)
        mu_p: 分布P的均值 (batch, dim)
        logvar_p: 分布P的对数方差 (batch, dim)

    返回:
        kl: 批量平均KL散度
    """
    var_q = torch.exp(logvar_q) + 1e-8
    var_p = torch.exp(logvar_p) + 1e-8

    kl = 0.5 * (
        logvar_p - logvar_q
        + (var_q + (mu_q - mu_p) ** 2) / var_p
        - 1.0
    )
    return kl.mean()


def wasserstein2_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    """
    计算两个高斯分布之间的2-Wasserstein距离的平方

    W2^2 = ||mu_q - mu_p||^2 + ||sigma_q - sigma_p||_F^2

    参数:
        mu_q: 分布Q的均值 (batch, dim)
        logvar_q: 分布Q的对数方差 (batch, dim)
        mu_p: 分布P的均值 (batch, dim)
        logvar_p: 分布P的对数方差 (batch, dim)

    返回:
        批量平均2-Wasserstein距离
    """
    std_q = torch.exp(0.5 * logvar_q)
    std_p = torch.exp(0.5 * logvar_p)

    mean_term = ((mu_q - mu_p) ** 2).sum(dim=-1).mean()
    std_term = ((std_q - std_p) ** 2).sum(dim=-1).mean()

    return mean_term + std_term


def distributional_proto_loss(local_mu, local_logvar, global_mu, global_logvar,
                              dist_type='kl'):
    """
    计算分布原型正则化损失，衡量本地原型与全局原型之间的差异

    参数:
        local_mu: 本地原型均值 (batch, proto_dim)
        local_logvar: 本地原型对数方差 (batch, proto_dim)
        global_mu: 全局原型均值 (batch, proto_dim)
        global_logvar: 全局原型对数方差 (batch, proto_dim)
        dist_type: 距离类型，可选 'kl'、'wasserstein'、'mse'

    返回:
        计算得到的分布距离损失值
    """
    if dist_type == 'kl':
        return kl_divergence_gaussian(local_mu, local_logvar,
                                      global_mu, global_logvar)
    elif dist_type == 'wasserstein':
        return wasserstein2_gaussian(local_mu, local_logvar,
                                     global_mu, global_logvar)
    elif dist_type == 'mse':
        # 回退：仅对均值计算MSE（原始FedProto行为）
        return ((local_mu - global_mu) ** 2).mean()
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")


def prototype_calibration_loss(mu, logvar, labels, global_protos,
                               dist_type='kl', num_classes=14):
    """
    原型校准损失：鼓励 logvar 反映原型距离与实际分类误差的一致性

    核心思想——well-calibrated 原型应满足：
      - 当本地原型与全局原型距离大时 → 方差应大（高不确定性）
      - 当距离小时 → 方差应小（高置信度）
      - 即：logvar 应与 proto_distance 正相关

    实现：
      对每个样本的正标签类别，计算：
        dist_to_global = D(N(mu_i, σ²_i) || N(mu_global, σ²_global))
        calibration_err = |logvar_i.mean() - log(dist_to_global + ε)|

    参数:
        mu: 本地原型均值 (B, proto_dim)
        logvar: 本地原型对数方差 (B, proto_dim)
        labels: 多标签矩阵 (B, num_classes)
        global_protos: 全局原型字典 {label: (mu_g, logvar_g)}
        dist_type: 距离类型
        num_classes: 类别总数

    返回:
        cal_loss: 标量校准损失
    """
    if len(global_protos) == 0:
        return torch.tensor(0.0, device=mu.device)

    B = mu.size(0)
    total_cal = 0.0
    count = 0

    for i in range(B):
        for lbl in range(num_classes):
            if labels[i, lbl] > 0 and lbl in global_protos:
                g_val = global_protos[lbl]
                g_mu, g_logvar = (g_val if isinstance(g_val, tuple)
                                  else (g_val, torch.zeros_like(g_val)))

                # 计算到该类全局原型的距离
                dist = distributional_proto_loss(
                    mu[i:i+1], logvar[i:i+1],
                    g_mu.unsqueeze(0), g_logvar.unsqueeze(0),
                    dist_type=dist_type
                )

                # 对数值域对齐：logvar 应反映 log(distance)
                log_dist = torch.log(dist + 1e-8)
                log_var_mean = logvar[i].mean()

                # Huber loss 比 MSE 对离群值更鲁棒
                cal_err = F.smooth_l1_loss(log_var_mean, log_dist)
                total_cal += cal_err
                count += 1

    if count == 0:
        return torch.tensor(0.0, device=mu.device)

    return total_cal / count


def entropy_regularization(logvar):
    """
    熵正则化：惩罚过小的方差（过于自信的原型）

    在分布原型学习中，logvar → -∞ 意味着 σ² → 0（退化回点原型）。
    熵正则化阻止这种退化，保持分布原型的优势。

    L_ent = mean(-logvar) = mean(-log(σ²))

    当 σ² → 0 (logvar → -∞)，此项 → +∞，形成强惩罚。

    参数:
        logvar: 对数方差 (B, proto_dim)

    返回:
        标量熵正则损失
    """
    # H = 0.5 * log(2πeσ²) ∝ logvar
    # 最小化 -logvar = 最大化方差（鼓励保留不确定性）
    return -logvar.mean()

