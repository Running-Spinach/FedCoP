# 功能：高斯原型的贝叶斯融合模块，实现精度加权平均的全局原型聚合

import torch


def bayesian_fusion_single_label(proto_list):
    """
    对单个类别的多个高斯原型进行贝叶斯融合（精度加权平均）

    每个客户端贡献 N(mu_i, sigma_i^2)，融合公式为：
        mu_global      = sum(mu_i / sigma_i^2) / sum(1 / sigma_i^2)
        sigma^2_global = 1 / sum(1 / sigma_i^2)

    参数:
        proto_list: (mu, log_var) 元组列表，每个形状为 (proto_dim,)

    返回:
        (mu_fused, logvar_fused): 融合后的均值和方差，形状均为 (proto_dim,)
    """
    mus = torch.stack([p[0] for p in proto_list])            # (K, D)
    vars_ = torch.stack([torch.exp(p[1]) for p in proto_list])  # (K, D)
    precs = 1.0 / (vars_ + 1e-8)                              # (K, D)
    sum_prec = precs.sum(dim=0)                                # (D,)
    mu_fused = (mus * precs).sum(dim=0) / (sum_prec + 1e-8)   # (D,)
    var_fused = 1.0 / (sum_prec + 1e-8)                       # (D,)
    logvar_fused = torch.log(var_fused + 1e-8)
    return mu_fused, logvar_fused


def bayesian_fusion(gaussian_protos_list):
    """
    跨多个客户端对所有标签进行贝叶斯融合

    参数:
        gaussian_protos_list: 字典列表，每个字典格式为 {label: (mu, log_var)}

    返回:
        fused: 融合后的字典 {label: (mu_fused, logvar_fused)}
    """
    agg_protos = {}

    for client_protos in gaussian_protos_list:
        for label, (mu, log_var) in client_protos.items():
            if label not in agg_protos:
                agg_protos[label] = []
            agg_protos[label].append((mu.detach(), log_var.detach()))

    fused = {}
    for label, proto_list in agg_protos.items():
        if len(proto_list) == 1:
            fused[label] = proto_list[0]
        else:
            fused[label] = bayesian_fusion_single_label(proto_list)

    return fused
