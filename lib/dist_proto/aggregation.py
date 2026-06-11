# =============================================================================
# 功能：高斯原型的贝叶斯融合 — 精度加权全局聚合
# =============================================================================
# D²-FL 使用贝叶斯融合替代简单算术平均，这是分布原型相比点原型的关键优势。
#
# 为什么不能用算术平均？
#   点原型（FedProto）：所有客户端上传的向量直接取平均，公平但幼稚。
#   分布原型（D²-FL）：每个客户端上传的是 N(μ, σ²)，包含了"我有多少数据/
#   我多确定"的信息。σ² 大的客户端（数据噪声大/样本少）不应该有同等投票权。
#
# 贝叶斯融合的原理：
#   把每个客户端的高斯分布看作"对真实原型的独立测量"。
#   根据贝叶斯定理，精度（1/σ²）更高的测量应该有更大的权重。
#
#   融合公式（精度加权平均）：
#     μ_global = Σ(μ_i / σ²_i) / Σ(1 / σ²_i)
#                       ↑                  ↑
#                 精度加权的均值      总精度（归一化因子）
#
#     σ²_global = 1 / Σ(1 / σ²_i)
#                        ↑
#               总精度取倒数 = 融合后方差
#
#   关键性质：
#     - σ²_global ≤ min(σ²_i)：融合后的不确定性 ≤ 任何单个客户端的
#       不确定性（信息越多越确定，符合直觉）
#     - 如果某个客户端 σ² 很大（很不确定），它对 μ_global 的贡献就很小
#     - 如果所有客户端 σ² 都很小（都很确定），融合结果也会很确定
#
# 通俗理解：
#   就像多个医生给出诊断意见，每个医生还会说自己有多大把握。
#   贝叶斯融合就是"更有把握的医生意见权重更大"。
#   最简单的情况：两个医生，一个说"我90%确定是肺炎"，另一个说"我50%确定是肺炎"，
#   最终判断应该更接近90%确定的那个医生的意见。
# =============================================================================

import torch


def bayesian_fusion_single_label(proto_list):
    """对单个类别的多个高斯原型进行贝叶斯融合（精度加权平均）

    输入：K 个客户端对同一类别的原型估计
    - 客户端1: N(μ₁, σ²₁)，代表"我认为这类的中心在 μ₁，方差 σ²₁"
    - 客户端2: N(μ₂, σ²₂)
    - ...
    - 客户端K: N(μ_K, σ²_K)

    融合：
    - μ_global      = Σ(μ_i / σ²_i) / Σ(1 / σ²_i)    ← 精度加权平均
    - σ²_global     = 1 / Σ(1 / σ²_i)                ← 总精度的倒数
    - logvar_global = log(σ²_global)                  ← 保持 logvar 格式

    为什么精度加权而不是方差加权？
    - 方差 σ² 大 → 不确定 → 权重小（精度 1/σ² 小）
    - 方差 σ² 小 → 确定   → 权重大（精度 1/σ² 大）
    - 精度加权自动实现"确定性越高，投票权越大"

    参数:
        proto_list: (mu, logvar) 元组列表，每个 mu 和 logvar 的 shape 为 (proto_dim,)

    返回:
        (mu_fused, logvar_fused): 融合后的均值和 logvar，shape 均为 (proto_dim,)
    """
    # 步骤1：将各客户端参数收集为矩阵
    mus = torch.stack([p[0] for p in proto_list])            # (K, D)  K个客户端的D维均值
    vars_ = torch.stack([torch.exp(p[1]) for p in proto_list])  # (K, D)  logvar → var

    # 步骤2：计算精度 = 1/方差（方差越小 → 精度越高 → 权重越大）
    precs = 1.0 / (vars_ + 1e-8)                              # (K, D)  防止除零

    # 步骤3：总精度 = 所有客户端精度之和
    sum_prec = precs.sum(dim=0)                                # (D,)  Σ(1/σ²_i)

    # 步骤4：精度加权均值 = Σ(μ_i × 精度_i) / Σ(精度_i)
    mu_fused = (mus * precs).sum(dim=0) / (sum_prec + 1e-8)   # (D,)

    # 步骤5：融合后方差 = 1 / 总精度
    var_fused = 1.0 / (sum_prec + 1e-8)                       # (D,)

    # 步骤6：转回 logvar 格式（与输入格式保持一致）
    logvar_fused = torch.log(var_fused + 1e-8)

    return mu_fused, logvar_fused


def bayesian_fusion(gaussian_protos_list):
    """跨多个客户端对所有标签进行贝叶斯融合

    这个函数是全局聚合的"调度器"：
    1. 遍历所有客户端的上传原型
    2. 按标签分组（同一标签的原型放一起）
    3. 对每个标签调用 bayesian_fusion_single_label 进行精度加权融合

    参数:
        gaussian_protos_list: 字典列表，每个字典格式为 {label: (mu, logvar)}
                             每个字典对应一个客户端的上传原型

    返回:
        fused: 融合后的全局原型字典，格式为 {label: (mu_fused, logvar_fused)}

    示例：
        输入 = [
            {0: (μ₀_客户端1, lv₀_客户端1), 1: (μ₁_客户端1, lv₁_客户端1)},  # 客户端1
            {0: (μ₀_客户端2, lv₀_客户端2), 1: (μ₁_客户端2, lv₁_客户端2)},  # 客户端2
        ]
        输出 = {
            0: (μ₀_融合, lv₀_融合),  # 客户端1和2对类别0的融合结果
            1: (μ₁_融合, lv₁_融合),  # 客户端1和2对类别1的融合结果
        }
    """
    # 步骤1：按标签收集所有客户端的原型
    agg_protos = {}
    for client_protos in gaussian_protos_list:
        for label, (mu, log_var) in client_protos.items():
            if label not in agg_protos:
                agg_protos[label] = []
            # detach() 切断梯度图 — 聚合是纯数值操作，不需反向传播
            agg_protos[label].append((mu.detach(), log_var.detach()))

    # 步骤2：对每个标签进行精度加权融合
    fused = {}
    for label, proto_list in agg_protos.items():
        if len(proto_list) == 1:
            # 只有一个客户端有该标签 → 直接使用，无需融合
            fused[label] = proto_list[0]
        else:
            # 多个客户端有同标签 → 贝叶斯精度加权融合
            fused[label] = bayesian_fusion_single_label(proto_list)

    return fused
