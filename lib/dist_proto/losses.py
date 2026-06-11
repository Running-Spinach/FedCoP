# =============================================================================
# 功能：D²-FL 分布原型损失函数集合
# =============================================================================
# 本模块提供了 D²-FL 算法中所有与"分布原型"相关的损失函数。
# 在 D²-FL 中，原型不再是单个向量（点原型），而是一个高斯分布 N(μ, σ²)，
# 这样可以同时编码"该类特征长什么样"（μ）和"有多确定"（σ²）。
#
# 包含四个核心函数：
#   1. distributional_proto_loss  — 计算本地原型和全局原型之间的"距离"
#   2. prototype_calibration_loss  — 确保方差真实反映不确定性（校准）
#   3. entropy_regularization      — 防止方差退化回点原型（熵正则）
#   4. kl_divergence_gaussian      — KL 散度底层实现
#   5. wasserstein2_gaussian       — Wasserstein 距离底层实现
#
# 通俗理解：
#   - 如果 FedProto 的点原型是"记住一个标准答案"，
#     那么 D²-FL 的分布原型就是"记住一个答案 + 一个容错范围"。
#   - μ 告诉你"中心在哪"，σ² 告诉你"允许偏多远"。
#   - 方差越大 → 越不确定 → 聚合时权重越小（精度加权）。
# =============================================================================

import torch
import torch.nn.functional as F


def kl_divergence_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    """
    计算两个高斯分布之间的 KL 散度：KL( N(μ_q, σ²_q) || N(μ_p, σ²_p) )

    KL 散度衡量的是"用分布 P 来近似分布 Q 时，会损失多少信息"。
    这里 Q = 本地原型，P = 全局原型，数值越小说明本地越接近全局。

    数学公式（两个对角高斯之间的闭式解）：
        KL = 0.5 * [ log(σ²_p / σ²_q) + (σ²_q + (μ_q - μ_p)²) / σ²_p - 1 ]

    逐项解读：
        - log(σ²_p / σ²_q)：方差比。如果全局方差 > 本地方差，此项 > 0，惩罚"过于自信"
        - (σ²_q + (μ_q - μ_p)²) / σ²_p：均值和方差差异在全局尺度下的归一化距离
        - -1：常数偏移，保证当两个分布完全相同时 KL = 0

    参数:
        mu_q:     本地原型的均值，shape (batch, dim)
        logvar_q: 本地原型的对数方差，shape (batch, dim)
                  （用 logvar 而不是 var，是因为 logvar 可以取负数，数值范围更友好）
        mu_p:     全局原型的均值，shape (batch, dim)
        logvar_p: 全局原型的对数方差，shape (batch, dim)

    返回:
        kl: 批量平均 KL 散度，一个标量
    """
    # 将对数方差还原为方差，+1e-8 防止除零
    var_q = torch.exp(logvar_q) + 1e-8
    var_p = torch.exp(logvar_p) + 1e-8

    kl = 0.5 * (
        logvar_p - logvar_q                              # 方差比项
        + (var_q + (mu_q - mu_p) ** 2) / var_p           # 归一化距离项
        - 1.0                                             # 常数偏移
    )
    return kl.mean()  # 对 batch 取平均，得到一个标量损失


def wasserstein2_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    """
    计算两个高斯分布之间的 2-Wasserstein 距离的平方

    和 KL 散度不同，Wasserstein 距离是真正的"距离"度量（满足三角不等式），
    而且对模式的"位移"更敏感。直观理解：
    - KL 散度：关心"用 P 解释 Q 需要多少额外信息"（信息论视角）
    - Wasserstein：关心"把一个分布'搬运'成另一个分布需要多少功"（几何视角）

    数学公式（两个对角高斯之间的闭式解）：
        W₂² = ||μ_q - μ_p||² + ||σ_q - σ_p||²_F
            = 均值差的平方 + 标准差差的 Frobenius 范数

    逐项解读：
        - ||μ_q - μ_p||²：均值向量之间的欧氏距离，衡量"中心位置差多远"
        - ||σ_q - σ_p||²_F：标准差矩阵差异的 Frobenius 范数，衡量"形状差多远"

    参数:
        mu_q, logvar_q: 本地原型的高斯参数
        mu_p, logvar_p: 全局原型的高斯参数

    返回:
        批量平均 2-Wasserstein 距离，一个标量
    """
    # 将对数方差转换为标准差（σ = exp(0.5 * logvar)）
    std_q = torch.exp(0.5 * logvar_q)
    std_p = torch.exp(0.5 * logvar_p)

    # 均值差：对所有维度求和，然后对 batch 取平均
    mean_term = ((mu_q - mu_p) ** 2).sum(dim=-1).mean()

    # 标准差差：Frobenius 范数（逐元素差的平方和）
    std_term = ((std_q - std_p) ** 2).sum(dim=-1).mean()

    return mean_term + std_term


def distributional_proto_loss(local_mu, local_logvar, global_mu, global_logvar,
                              dist_type='kl'):
    """
    分布原型正则化损失 — D²-FL 的核心损失之一（L_proto）

    这是本地训练时"拉近本地原型和全局原型"的损失。
    根据配置选择不同的距离度量方式。

    直观理解：
        每个客户端在本地训练时，不仅要做好分类（CE loss），
        还要让自己的原型不要跑得离全局原型太远（proto loss）。
        这就像每个医院的医生在诊断时，既要看好自己的病人，
        也要确保自己对疾病的理解和大伙一致。

    参数:
        local_mu:      本地原型均值，shape (batch, proto_dim)
        local_logvar:  本地原型对数方差，shape (batch, proto_dim)
        global_mu:     全局原型均值，shape (batch, proto_dim)
        global_logvar: 全局原型对数方差，shape (batch, proto_dim)
        dist_type:     距离类型，可选：
                       'kl'         — KL 散度（推荐，信息论最优）
                       'wasserstein'— Wasserstein 距离（几何直观）
                       'mse'        — 仅对均值算 MSE（退化回 FedProto）

    返回:
        计算得到的分布距离损失值，一个标量
    """
    if dist_type == 'kl':
        return kl_divergence_gaussian(local_mu, local_logvar,
                                      global_mu, global_logvar)
    elif dist_type == 'wasserstein':
        return wasserstein2_gaussian(local_mu, local_logvar,
                                     global_mu, global_logvar)
    elif dist_type == 'mse':
        # 退化模式：忽略方差，仅对均值算 MSE
        # 此时等价于原始 FedProto 的原型损失
        return ((local_mu - global_mu) ** 2).mean()
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")


def prototype_calibration_loss(mu, logvar, labels, global_protos,
                               dist_type='kl', num_classes=14):
    """
    原型校准损失（L_cal）— 确保方差"说真话"

    核心问题：
        神经网络输出的 logvar 可能不可靠。比如模型可能对某个完全不确定的
        预测也输出很小的方差（假装很自信），或者对确定的预测输出很大的方差
        （假装很谦虚）。校准损失就是要纠正这种"口是心非"。

    核心思想：
        一个"校准好"的原型应该满足——
        - 当本地原型离全局原型很远时 → 方差应该大（诚实地说"我不太确定"）
        - 当本地原型离全局原型很近时 → 方差应该小（有资格说"我比较确定"）
        即：logvar 和 log(distance) 应该正相关。

    实现方式：
        对每个样本的每个正标签类别：
        1. 计算本地高斯到该类全局高斯的距离 d
        2. 计算本地 logvar 的均值（代表模型自己声称的不确定性）
        3. 用 Huber Loss（smooth_l1）让 logvar_mean ≈ log(d + ε)

        为什么用 Huber Loss 而不用 MSE？
        — Huber Loss 对异常值不敏感。偶尔某个样本距离特别大时，
          MSE 会给出巨大的梯度，导致训练不稳定。Huber 会自动"限幅"。

    参数:
        mu:           本地原型均值，shape (B, proto_dim)
        logvar:       本地原型对数方差，shape (B, proto_dim)
        labels:       多标签矩阵，shape (B, num_classes)，值 ∈ {0, 1}
        global_protos:全局原型字典，{label: (mu_g, logvar_g)} 或 {label: mu_g}
        dist_type:    距离类型（传给 distributional_proto_loss）
        num_classes:  类别总数（ChestX-ray14 = 14）

    返回:
        cal_loss: 标量校准损失，值越小说明方差越"诚实"
    """
    # 第一轮全局原型还没建立，跳过
    if len(global_protos) == 0:
        return torch.tensor(0.0, device=mu.device)

    B = mu.size(0)        # batch 大小
    total_cal = 0.0       # 累计校准误差
    count = 0             # 有效样本-类别对计数

    for i in range(B):
        for lbl in range(num_classes):
            # 只看该样本的正标签类别（疾病实际存在）
            if labels[i, lbl] > 0 and lbl in global_protos:
                g_val = global_protos[lbl]
                # 兼容点原型：如果全局原型没有方差，补零（视为方差=1）
                g_mu, g_logvar = (g_val if isinstance(g_val, tuple)
                                  else (g_val, torch.zeros_like(g_val)))

                # 步骤1：计算该样本到该类全局原型的距离
                dist = distributional_proto_loss(
                    mu[i:i+1], logvar[i:i+1],              # 单个样本
                    g_mu.unsqueeze(0), g_logvar.unsqueeze(0),  # 该类全局原型
                    dist_type=dist_type
                )

                # 步骤2：取对数空间对齐
                # log_dist：实际距离的对数（"真实的不确定性"）
                # log_var_mean：模型声称的不确定性的对数（"声称的不确定性"）
                log_dist = torch.log(dist + 1e-8)
                log_var_mean = logvar[i].mean()

                # 步骤3：Huber Loss 让二者对齐
                # Huber = 小误差用 MSE（精确），大误差用 MAE（鲁棒）
                cal_err = F.smooth_l1_loss(log_var_mean, log_dist)
                total_cal += cal_err
                count += 1

    if count == 0:
        return torch.tensor(0.0, device=mu.device)

    return total_cal / count  # 平均校准损失


def entropy_regularization(logvar):
    """
    熵正则化（L_ent）— 防止方差坍缩回点原型

    关键问题：为什么需要这个损失？
        在分布原型学习中，有一个天然的退化方向：
        logvar → -∞  →  σ² → 0  →  高斯原型退化成点原型

        这样一来，分布原型的优势（编码不确定性、贝叶斯融合）全没了。
        熵正则化就是给这个退化方向加一个"惩罚"，阻止方差变得太小。

    数学公式：
        L_ent = mean(-logvar) = mean(-log(σ²))

        当 σ² → 0 时（logvar → -∞），-logvar → +∞，形成强力惩罚。
        当 σ² 合理大时，惩罚很小。

    直观理解：
        就像老师批改作业，不仅要看学生答得对不对，
        还要看学生对自己的答案有没有"合理的不确定性"。
        如果一个学生每题都写"我100%确定"，那反而不真实。

    参数:
        logvar: 对数方差，shape (B, proto_dim)

    返回:
        标量熵正则损失。正值说明方差总体偏小（需要惩罚），
        接近0说明方差在合理范围。
    """
    # 高斯分布的熵 ∝ logvar（熵 = 0.5 * log(2πeσ²) = 0.5 * (log(2πe) + logvar)）
    # 最小化 -logvar = 最大化方差 = 鼓励保留不确定性
    return -logvar.mean()
