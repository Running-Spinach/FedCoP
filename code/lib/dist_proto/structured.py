# =============================================================================
# FedCoP 结构化共现模块 — 方向一的核心创新落点
# =============================================================================
# 本模块实现 FedCoP(Federated Co-occurrence-aware Prototypes)的三项核心机制:
#
#   1. 联邦共现结构估计
#      - 客户端只上传标签的充分统计量 (m_k, M_k, n_k),不含任何特征/图像,
#        隐私安全且通信廉价(14 + 196 + 1 个整数)。
#      - 服务器聚合后估计 14×14 的 phi 相关矩阵 R̂(病理共现结构)。
#      - 关键 FL 论点:Non-IID 划分下每个客户端只见到 ~3/14 类,
#        没有任何单个客户端能观测到完整共现结构,只有联邦聚合才能恢复它。
#
#   2. 共现结构对齐损失 L_co(训练侧)
#      - 把"各类原型的相互几何"约束到共现结构上:
#        共现的疾病 → 原型方向相近;互斥的疾病 → 原型方向正交/远离。
#      - 用原型均值的余弦 Gram 矩阵对齐 R̂,替代被砍掉的对比/对抗损失。
#
#   3. 相关性感知 mean-field 解码(推理侧)
#      - 用 R̂ 在类间传播证据:某类证据强且与它共现的类,后验被抬高。
#      - 临床语义:"看到胸腔积液 → 提升肺不障概率"(共病诊断)。
#      - 可微、O(C²)、可解释,是独立 sigmoid 在相关标签下的严格改进。
#
# 设计原则:每个函数单一职责、有闭式或线性复杂度、详细中文注释。
# =============================================================================

import torch
import torch.nn.functional as F


# =============================================================================
#  1. 联邦共现结构估计
# =============================================================================

def compute_local_cooc(labels):
    """计算单个客户端的标签共现充分统计量

    共现结构只依赖标签(多热向量),与特征/模型无关,因此可以在客户端
    本地计算后只上传这几个整数统计量,既隐私安全又通信廉价。

    充分统计量(足以恢复任意阶标签矩):
        m_k ∈ R^C   : 每类的边际计数(marginal count),m_k[c] = 该客户端有多少样本含第 c 类
        M_k ∈ R^{C×C}: 共现计数矩阵,M_k[i,j] = 同时含第 i 类和第 j 类的样本数
                       (注意:多热标签下 M_k = Y_kᵀ Y_k,对角线 M_k[c,c] = m_k[c])
        n_k         : 该客户端样本总数

    参数:
        labels: 多热标签,(n, C) 的 torch.Tensor 或 numpy 数组,值 ∈ {0, 1}

    返回:
        dict {'m': (C,), 'M': (C, C), 'n': 标量},均为 torch.float 张量
    """
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32)
    labels = labels.float()

    m = labels.sum(dim=0)              # (C,)  每类边际计数
    M = labels.t() @ labels            # (C, C)  共现计数矩阵
    n = labels.size(0)                 # 样本总数
    return {'m': m, 'M': M, 'n': float(n)}


def fuse_cooccurrence(stats_list, num_classes, shrinkage=0.1, rank=0, eps=1e-8):
    """聚合多客户端的共现统计量,估计全局病理共现相关矩阵 R̂

    流程:
        1. 计数聚合:  M = Σ M_k,  m = Σ m_k,  N = Σ n_k  (计数加权 = 无偏)
        2. 概率:      p_i = m_i / N,  p_ij = M_ij / N
        3. phi 相关:  R_ij = (p_ij - p_i p_j) / sqrt(p_i(1-p_i) p_j(1-p_j))
                      —— 二值变量的 Pearson 相关,衡量"共同出现程度超过独立预期的多少"
        4. 收缩:      R̂ = (1-η) R + η I   —— 拉向单位阵,保证正定 + 小样本稳定
        5. [可选]低秩: 保留前 rank 个特征值,降噪 + 压缩(默认 rank=0 即全秩)

    为什么用 phi 相关而不是直接用共现概率 p_ij?
        - p_ij 受各类边际频率影响(罕见病 p_ij 天然小),不能跨类比较。
        - phi 相关去除了边际频率的影响,纯粹反映"共现强度",可跨类比较,
          且取值 [-1, 1],天然适合作为相关矩阵。

    为什么收缩到单位阵?
        - 样本少时 R 的非对角元噪声大,可能不正定。
        - 收缩到 I = "假设独立",η 越大越保守。η=1 退化回独立(共现结构关闭)。
        - 凸组合保持正定性(两个正定矩阵的凸组合仍正定)。

    参数:
        stats_list:  各客户端 compute_local_cooc 返回的 dict 列表
        num_classes: 类别数 C(ChestX-ray14 = 14)
        shrinkage:   收缩系数 η ∈ [0, 1],默认 0.1
        rank:        低秩近似的秩。0 = 全秩(默认,14×14 很小无需降秩)
        eps:         数值稳定小量

    返回:
        R_hat: (C, C) 全局共现相关矩阵,正定,对角线为 1
        pi:    (C,)   全局边际先验 p_d(推理时作为各类基准概率)
    """
    device = stats_list[0]['m'].device

    # ── 步骤1: 计数聚合 ──
    M = torch.zeros(num_classes, num_classes, device=device)
    m = torch.zeros(num_classes, device=device)
    N = 0.0
    for s in stats_list:
        M = M + s['M']
        m = m + s['m']
        N += s['n']
    N = max(N, 1.0)

    # ── 步骤2: 边际与联合概率 ──
    p_i = m / N                              # (C,)  P(类 c 为正)
    p_ij = M / N                             # (C, C)  P(类 i,j 同时为正)

    # ── 步骤3: phi 相关(二值变量的 Pearson 相关)──
    var_i = p_i * (1.0 - p_i)                # (C,)  Bernoulli 方差
    denom = torch.sqrt(var_i[:, None] * var_i[None, :] + eps)  # (C, C)
    R = (p_ij - p_i[:, None] * p_i[None, :]) / denom
    R = R.clamp(-1.0, 1.0)                   # 数值裁剪到合法相关范围

    # 从未见过的类(边际方差≈0)无法估计相关:置为独立(对角 1,其余 0)
    invalid = var_i < eps
    if invalid.any():
        R[invalid, :] = 0.0
        R[:, invalid] = 0.0
        R[invalid, invalid] = 1.0

    # ── 步骤4: 收缩到单位阵(正定 + 小样本稳定)──
    eye = torch.eye(num_classes, device=device)
    R_hat = (1.0 - shrinkage) * R + shrinkage * eye
    R_hat = 0.5 * (R_hat + R_hat.t())        # 强制对称(消除数值非对称)

    # ── 步骤5: 可选低秩近似(默认跳过)──
    if rank > 0 and rank < num_classes:
        # 对称特征分解,保留前 rank 个主分量,重构秩-rank 相关矩阵
        eigval, eigvec = torch.linalg.eigh(R_hat)
        top = torch.topk(eigval, rank)
        R_hat = (eigvec[:, top.indices] * top.values) @ eigvec[:, top.indices].t()
        R_hat = 0.5 * (R_hat + R_hat.t())    # 重构后再次对称化

    pi = p_i.detach()
    return R_hat, pi


def ema_correlation(old_R, new_R, momentum):
    """跨轮 EMA 平滑全局共现相关矩阵

    每轮参与客户端不同(随机采样),直接替换 R̂ 会震荡。
    EMA: R̂_t = momentum · R̂_{t-1} + (1-momentum) · R̂_new
    momentum=0.9 表示"90% 保留旧结构,10% 注入新观测"。

    合法性:两个正定相关矩阵的凸组合仍是正定的,因此 EMA 保持 R̂ 正定。

    参数:
        old_R:    (C, C) 上一轮的 R̂
        new_R:    (C, C) 本轮新估计的 R̂
        momentum: EMA 动量系数 ∈ [0, 1)

    返回:
        (C, C) 平滑后的 R̂
    """
    if old_R is None or momentum <= 0:
        return new_R
    return momentum * old_R + (1.0 - momentum) * new_R


# =============================================================================
#  2. 共现结构对齐损失 L_co(训练侧)
# =============================================================================

def cos_gram_structure_loss(mu, labels, R_hat, num_classes, eps=1e-8):
    """共现结构对齐损失 L_co —— 把原型相互几何约束到共现结构上

    核心思想:
        当前 batch 内出现的各类,各自取样本均值得到"类原型估计" P ∈ R^{C'×D}。
        归一化后,其余弦 Gram 矩阵 G = P̂ P̂ᵀ ∈ [-1,1]^{C'×C'} 描述了
        "这些类的原型方向有多接近"。

        我们要求 G 与共现相关矩阵 R̂ 的对应子块一致:
            - 共现的疾病(R̂_ij 大)→ 原型方向相近(G_ij 大)
            - 互斥的疾病(R̂_ij 小/负)→ 原型方向远离(G_ij 小/负)

        L_co = ‖G − R̂_sub‖_F²

    为什么用余弦 Gram 而非欧氏距离?
        - 余弦相似度去除了原型模长的影响,只约束"方向"。
        - 与 R̂(相关,取值 [-1,1])在同一尺度,可直接对齐。

    为什么这能替代被砍的对比/对抗损失?
        - 对比损失(InfoNCE)用启发式 Jaccard 阈值定义正负样本对,且只看同类聚集。
        - L_co 直接用联邦估计的共现结构 R̂ 约束原型布局,有明确的标签统计依据,
          且同时建模"共现拉近"和"互斥推远",信息更丰富、动机更干净。

    参数:
        mu:          (B, D) 本 batch 样本的分布原型均值
        labels:      (B, C) 多热标签
        R_hat:       (C, C) 全局共现相关矩阵
        num_classes: 类别数 C
        eps:         归一化数值稳定

    返回:
        标量损失。若 batch 内出现类少于 2(无法构成 Gram),返回 0。
    """
    # 找出本 batch 内出现的类(至少有一个正样本)
    present = labels.sum(dim=0) > 0          # (C,) bool
    present_idx = torch.nonzero(present, as_tuple=False).squeeze(-1)
    if present_idx.numel() < 2:
        # 少于 2 个类无法构成有意义的相互几何约束
        return mu.new_zeros(())

    # 每个出现类:取该类正样本的均值作为"类原型估计"
    proto_list = []
    for c in present_idx:
        mask = labels[:, c] > 0
        proto_list.append(mu[mask].mean(dim=0))   # (D,)
    P = torch.stack(proto_list, dim=0)            # (C', D)

    # 余弦 Gram 矩阵:归一化后内积
    P_norm = F.normalize(P, p=2, dim=1, eps=eps)  # (C', D)
    G = P_norm @ P_norm.t()                        # (C', C')  余弦相似度

    # R̂ 的对应子块
    R_sub = R_hat[present_idx][:, present_idx]    # (C', C')

    # Frobenius 范数²(按元素数归一化,使损失尺度与类数无关)
    loss = ((G - R_sub) ** 2).sum() / (present_idx.numel() ** 2)
    return loss


# =============================================================================
#  3. 相关性感知 mean-field 解码(推理侧)
# =============================================================================

def mean_field_decode(s, R_hat, pi, beta=1.0, steps=2):
    """相关性感知 mean-field 结构化解码

    把独立逐类 logits s ∈ R^{B×C} 解码为考虑共现结构的多标签概率 q。

    模型:把多标签联合分布建模为带成对耦合的 Bernoulli(类似 Ising / 完全可见
    玻尔兹曼机),用变分 mean-field 近似做后验推断:
        q_c ← σ( s_c + β · Σ_{d≠c} R̂_cd · (q_d − π_d) )

    直觉:
        - s_c:第 c 类的独立证据(原型马氏距离转 logit)。
        - π_d:第 d 类的全局边际先验(基准患病率)。
        - (q_d − π_d):第 d 类"超出基准的证据",正=比平时多见,负=比平时少见。
        - R̂_cd:第 c,d 类的共现强度。
        - 耦合项 β·Σ R̂_cd(q_d−π_d):把"共现类的额外证据"传播给第 c 类。
          → 共现且证据强的类会抬高本类概率(共病诊断)。

    与独立 sigmoid 的关系:
        - R̂ = I(无共现)时,耦合项为 0,退化为独立 sigmoid。
        - R̂ 非平凡时,在相关标签下严格改进(见论文命题1)。

    复杂度:每步 O(B·C²),C=14 极轻量,4090 上可忽略。

    参数:
        s:     (B, C) 独立 logits(原型马氏距离转 logit)
        R_hat: (C, C) 全局共现相关矩阵
        pi:    (C,)   全局边际先验
        beta:  耦合强度 β(默认 1.0)
        steps: mean-field 迭代步数(默认 2,通常 1~3 步即收敛)

    返回:
        q: (B, C) 结构化后验概率
    """
    # 对角线置零:避免自耦合(self-reinforcement)导致的不稳定
    # (R̂ 对角线本为 1,但 (q_c−π_c) 的自反馈会让 q_c 自我放大,无意义且不稳)
    C = R_hat.size(0)
    R_off = R_hat * (1.0 - torch.eye(C, device=R_hat.device))   # (C, C) 去对角

    q = torch.sigmoid(s)                                  # (B, C) 独立初始化
    for _ in range(steps):
        # 耦合:每个类吸收"共现类超出基准的证据"
        # delta[b, c] = Σ_d R_off[c, d] · (q[b, d] − π[d])
        delta = (q - pi.unsqueeze(0)) @ R_off.t()         # (B, C)
        q = torch.sigmoid(s + beta * delta)
    return q
