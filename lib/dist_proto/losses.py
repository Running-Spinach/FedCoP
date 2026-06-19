# =============================================================================
# FedCoP 分布原型损失函数
# =============================================================================
# 分布原型把每类原型表示为对角高斯 N(μ, σ²),本模块提供:
#   1. distributional_proto_loss — 本地原型到全局原型的距离(L_proto)
#   2. entropy_regularization    — 防止方差坍缩回点原型(L_ent)
#   3. kl_divergence_gaussian    — 两个对角高斯的 KL 散度(底层)
#   4. wasserstein2_gaussian     — 两个对角高斯的 2-Wasserstein 距离(底层)
#
# 注:跨类共现结构损失 L_co 不在此处,见 structured.py 的 cos_gram_structure_loss。
#     旧版的 prototype_calibration_loss(L_cal)已移除——它把 logvar 坍缩成标量
#     做粗对齐,是给"NN 直吐 logvar 无约束"打补丁,在 FedCoP 的精简设计里不再需要。
# =============================================================================

import torch


def kl_divergence_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    """两个对角高斯间的 KL 散度:KL( N(μ_q,σ²_q) ‖ N(μ_p,σ²_p) )

    闭式解:KL = 0.5 · [ log(σ²_p/σ²_q) + (σ²_q + (μ_q−μ_p)²)/σ²_p − 1 ]
    Q=本地原型,P=全局原型;越小本地越接近全局。

    参数:mu_q,logvar_q,mu_p,logvar_p 均 (batch, dim)
    返回:标量(对 batch 平均)
    """
    var_q = torch.exp(logvar_q) + 1e-8
    var_p = torch.exp(logvar_p) + 1e-8
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    return kl.mean()


def wasserstein2_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    """两个对角高斯间的 2-Wasserstein 距离平方

    闭式解:W₂² = ‖μ_q−μ_p‖² + ‖σ_q−σ_p‖²_F
    Wasserstein 是真度量(满足三角不等式),对"位移"更敏感(几何视角)。

    参数:mu_q,logvar_q,mu_p,logvar_p 均 (batch, dim)
    返回:标量(对 batch 平均)
    """
    std_q = torch.exp(0.5 * logvar_q)
    std_p = torch.exp(0.5 * logvar_p)
    mean_term = ((mu_q - mu_p) ** 2).sum(dim=-1).mean()
    std_term = ((std_q - std_p) ** 2).sum(dim=-1).mean()
    return mean_term + std_term


def distributional_proto_loss(local_mu, local_logvar, global_mu, global_logvar,
                              dist_type='kl'):
    """分布原型对齐损失 L_proto —— 本地原型向全局原型靠拢

    参数:
        local_mu, local_logvar:   本地原型高斯参数 (batch, proto_dim)
        global_mu, global_logvar: 全局原型高斯参数 (batch, proto_dim)
        dist_type: 'kl'(推荐,信息论)/ 'wasserstein'(几何)/ 'mse'(仅均值,退化为点原型)

    返回:标量距离
    """
    if dist_type == 'kl':
        return kl_divergence_gaussian(local_mu, local_logvar,
                                      global_mu, global_logvar)
    elif dist_type == 'wasserstein':
        return wasserstein2_gaussian(local_mu, local_logvar,
                                     global_mu, global_logvar)
    elif dist_type == 'mse':
        # 忽略方差,仅均值 MSE → 退化回 FedProto 行为
        return ((local_mu - global_mu) ** 2).mean()
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")


def entropy_regularization(logvar):
    """熵正则 L_ent —— 防止方差坍缩回点原型

    分布原型有天然退化方向:logvar→−∞ → σ²→0 → 退化为点原型,分布优势全失。
    L_ent = mean(−logvar):σ²→0 时 −logvar→+∞ 形成强力惩罚,阻止坍缩。

    参数:logvar (B, proto_dim)
    返回:标量
    """
    # 高斯熵 ∝ logvar(熵 = 0.5·log(2πeσ²));最小化 −logvar = 鼓励保留不确定性
    return -logvar.mean()
