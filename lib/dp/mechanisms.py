# 功能：差分隐私机制模块，提供高斯噪声添加和隐私预算追踪功能

import torch
import numpy as np
import math


class MomentsAccountant:
    """
    基于Renyi DP（矩会计）的隐私预算追踪器
    用于跨轮次追踪 (epsilon, delta)-DP 保证
    """

    def __init__(self, delta=1e-5, orders=None):
        """
        初始化隐私会计

        参数:
            delta: 目标delta值
            orders: Renyi阶数列表
        """
        self.delta = delta
        if orders is None:
            orders = list(np.linspace(1.1, 10.9, 99))
        self.orders_tensor = torch.tensor(orders, dtype=torch.float64)
        self.log_moments = torch.zeros_like(self.orders_tensor)

    def compute_rdp_gaussian(self, noise_multiplier, sample_rate=None):
        """
        计算单轮高斯机制的RDP隐私消耗

        参数:
            noise_multiplier: 噪声乘数 sigma
            sample_rate: 采样率（用于子采样放大）

        返回:
            rdp_eps: 各阶数的RDP epsilon值
        """
        sigma_eff = noise_multiplier

        if sample_rate is None or sample_rate >= 1.0:
            rdp_eps = self.orders_tensor / (2.0 * sigma_eff ** 2)
        else:
            q = sample_rate
            rdp_eps = torch.zeros_like(self.orders_tensor)
            for i, lam in enumerate(self.orders):
                rdp_eps[i] = (q ** 2) * lam / (2.0 * sigma_eff ** 2)
                rdp_eps[i] = min(rdp_eps[i], lam * 100)

        return rdp_eps

    def accumulate(self, rdp_eps):
        """
        累加每轮的RDP隐私消耗
        """
        self.log_moments += rdp_eps.to(torch.float64)

    def get_epsilon(self):
        """
        将累积的RDP转换为 (epsilon, delta)-DP

        返回:
            epsilon: 当前的隐私预算消耗
        """
        if self.log_moments.max() == 0:
            return 0.0

        eps = self.log_moments - math.log(self.delta) / (self.orders_tensor - 1.0)
        valid = self.orders_tensor > 1.0
        if valid.any():
            eps = eps[valid].min().item()
        else:
            eps = float('inf')
        return max(0.0, eps)


class DPMechProto:
    """
    针对原型上传的差分隐私机制
    对每个客户端的 (mu, logvar) 对进行L2裁剪+高斯噪声处理
    """

    def __init__(self, clip_norm=1.0, noise_multiplier=1.0, use_dp=False):
        """
        初始化差分隐私机制

        参数:
            clip_norm: L2裁剪范数
            noise_multiplier: 噪声乘数
            use_dp: 是否启用差分隐私
        """
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.use_dp = use_dp

    def clip_and_noise(self, proto_dict):
        """
        对单个客户端的原型字典进行L2裁剪并添加高斯噪声

        参数:
            proto_dict: {label: (mu, logvar)} 单个客户端的原型字典
                mu: 均值张量 (proto_dim,)
                logvar: 对数方差张量 (proto_dim,)

        返回:
            perturbed: 扰动后的 {label: (mu_noisy, logvar_noisy)}
        """
        if not self.use_dp:
            return proto_dict

        sigma = self.clip_norm * self.noise_multiplier
        perturbed = {}

        for label, entry in proto_dict.items():
            if isinstance(entry, tuple):
                mu, logvar = entry
            else:
                # 点原型回退：当作零方差处理
                mu = entry
                logvar = torch.zeros_like(mu) - 5.0

            dim = mu.numel()

            # 对 (mu || logvar) 进行联合L2裁剪
            if isinstance(entry, tuple):
                full_vec = torch.cat([mu.flatten(), logvar.flatten()]).detach()
            else:
                full_vec = mu.flatten().detach()

            norm = torch.norm(full_vec, p=2)
            scale = min(1.0, self.clip_norm / (norm + 1e-8))
            full_vec_clipped = full_vec * scale

            mu_clipped = full_vec_clipped[:dim].view_as(mu)
            logvar_clipped = full_vec_clipped[dim:].view_as(logvar)

            # 添加高斯噪声
            noise_mu = torch.normal(mean=0.0, std=sigma, size=mu.shape, device=mu.device)
            noise_logvar = torch.normal(mean=0.0, std=sigma, size=logvar.shape, device=logvar.device)

            mu_perturbed = mu_clipped + noise_mu
            logvar_perturbed = logvar_clipped + noise_logvar

            perturbed[label] = (mu_perturbed, logvar_perturbed)

        return perturbed

    @staticmethod
    def sample_rate(num_participating, total_clients):
        """
        计算每轮参与采样的比例
        """
        return num_participating / max(total_clients, 1)


class DPMechWeight:
    """
    针对模型权重上传的差分隐私机制 (DP-FedAvg 风格)

    对每个客户端的权重 delta (w_local - w_global) 进行 L2 裁剪 + 高斯噪声处理
    """

    def __init__(self, clip_norm=1.0, noise_multiplier=1.0, use_dp=False):
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.use_dp = use_dp

    def clip_and_noise(self, local_state_dict, global_state_dict):
        """
        对权重 delta 进行 L2 裁剪并添加高斯噪声

        参数:
            local_state_dict: 本地模型 state_dict
            global_state_dict: 全局模型 state_dict (作为 delta 参考)

        返回:
            perturbed: 扰动后的 state_dict
        """
        if not self.use_dp:
            return local_state_dict

        sigma = self.clip_norm * self.noise_multiplier
        perturbed = {}

        for key in local_state_dict:
            if key not in global_state_dict:
                perturbed[key] = local_state_dict[key]
                continue
            delta = local_state_dict[key] - global_state_dict[key].to(local_state_dict[key].device)
            delta_flat = delta.view(-1)
            norm = torch.norm(delta_flat, p=2)
            scale = min(1.0, self.clip_norm / (norm + 1e-8))
            delta_clipped = delta_flat * scale

            noise = torch.normal(mean=0.0, std=sigma, size=delta_clipped.shape,
                                device=delta.device)
            delta_perturbed = delta_clipped + noise

            perturbed[key] = (global_state_dict[key].to(delta.device)
                              + delta_perturbed.view_as(delta))

        return perturbed

    @staticmethod
    def sample_rate(num_participating, total_clients):
        return num_participating / max(total_clients, 1)


def compute_noise_multiplier_from_epsilon(target_eps, sample_rate, delta,
                                          orders=None, max_iters=100):
    """
    二分搜索找到满足目标epsilon的噪声乘数sigma

    参数:
        target_eps: 目标epsilon值（单轮）
        sample_rate: 采样率
        delta: 目标delta值
        orders: Renyi阶数列表
        max_iters: 最大迭代次数

    返回:
        sigma: 满足目标epsilon的噪声乘数
    """
    if orders is None:
        orders = list(np.linspace(1.1, 10.9, 99))

    orders_tensor = torch.tensor(orders, dtype=torch.float64)
    lo, hi = 0.01, 100.0

    for _ in range(max_iters):
        mid = (lo + hi) / 2
        sigma_eff = mid

        rdp_eps = torch.zeros(len(orders))
        for i, lam in enumerate(orders):
            rdp_eps[i] = (sample_rate ** 2) * lam / (2.0 * sigma_eff ** 2)

        eps = rdp_eps - math.log(delta) / (orders_tensor - 1.0)
        valid = orders_tensor > 1.0
        if valid.any():
            eps_val = eps[valid].min().item()
        else:
            eps_val = float('inf')

        if eps_val > target_eps:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2
