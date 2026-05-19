# 功能：分布原型损失函数，支持KL散度、Wasserstein距离和MSE距离

import torch


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
