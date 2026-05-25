# DPP-FL 联邦原型学习 —— 理论与实现详解 (ChestX-ray14 + 预训练 ResNet-50)

> **DPP-FL** = Distributional Pathology Prototype Federated Learning
>
> 6 种算法 (FedAvg, FedProx, FedBN, SCAFFOLD, FedProto, DPP-FL) 全部使用 ImageNet 预训练骨干 + 差分隐私进行公平对比
>
> DPP-FL 在 FedProto 基础上新增：分布原型 + 贝叶斯融合 + EMA 动量 + Lambda 预热 + 温度缩放 + **原型解耦 (语义-风格分离)**

---

## 一、背景与动机

### 1.1 联邦学习 (Federated Learning)

联邦学习是一种分布式机器学习范式，核心思想是**数据不动模型动** —— 多个客户端（如手机、医院）在本地训练模型，仅将模型更新发送到中心服务器进行聚合，从而保护数据隐私。

经典的 **FedAvg** 算法流程：

```
每轮 t = 1, 2, ..., T:
  1. 服务器广播全局模型 w_t 给选中的客户端
  2. 每个客户端 k 在本地数据上训练得到 w_t^k
  3. 服务器聚合: w_{t+1} = Σ (n_k / n) * w_t^k
```

### 1.2 FedAvg 的两大局限

| 问题 | 描述 |
|------|------|
| **数据异构 (Non-IID)** | 不同客户端的数据分布差异大（有的客户端只有猫狗图片，有的只有车船），导致本地模型漂移，简单加权平均会破坏模型性能 |
| **模型异构 (Model Heterogeneity)** | 不同设备算力不同（手机 vs 服务器），无法使用相同的模型架构，FedAvg 要求所有模型结构一致才能做权重平均 |

### 1.3 FedProto 的解决思路

FedProto (AAAI 2022) 的核心创新：**不共享模型权重，而是共享每个类别的"原型"(Prototype) 向量**。

原型 = 模型倒数第二层（fc1）输出的特征向量，也就是分类决策前的样本表征。每个类别用一个原型向量来表示，不同客户端通过比对本地原型和全局原型来迁移知识 —— 这样不同模型架构也可以对齐（只要原型维度相同），从根本上解决了模型异构问题。

---

## 二、核心概念：原型 (Prototype)

### 2.1 什么是原型

在神经网络中，一个输入样本经过若干层后，倒数第二层全连接层输出的向量就是该样本的**特征表征**。FedProto 将其定义为**原型**。

以 ResNet-50 + ChestX-ray14 为例：

```
输入 (3×224×224)  灰度X光片转3通道
  → ResNet-50 Backbone (Bottleneck blocks, layers=[3,4,6,3])
  → AdaptiveAvgPool2d → Flatten (2048-dim)
  → fc1 (2048 → proto_dim) ← 原型空间 (默认 256 维)
  → [ProbabilisticProtoHead] (可选，μ/logvar)
  → fc2 (proto_dim → 14) → 14 维 logits → BCEWithLogitsLoss
```

原型向量具有以下性质：
- 它是一个**低维稠密向量**（本项目中 proto_dim=256）
- 同类样本的原型在特征空间中**彼此靠近**
- 它**不直接暴露原始数据**，保护隐私

### 2.2 点原型 vs 分布原型

| 类型 | 表示 | 开启方式 | 适用场景 |
|------|------|----------|----------|
| **点原型 (Point Proto)** | 单个向量 `p ∈ R^d` | 默认 | 基础场景 |
| **分布原型 (Distributional Proto)** | 高斯分布 `N(μ, σ²)` | `--use_distributional` | 需要建模不确定性的场景 |

---

## 三、FedProto 算法核心

### 3.1 全局训练流程

```
输入: K 个客户端，每客户端有本地模型 f_k，全局轮数 T, 采样率 frac
输出: 训练好的本地模型列表，全局原型集合 G

初始化: 全局原型 G = {}  (空)

For round t = 1 to T:
    # 客户端采样（由 --frac 控制，默认 0.04 = 每轮采样 4% 客户端）
    选中 m = max(1, int(frac * K)) 个客户端参与本轮

    For each selected client k:
        1. 本地训练
           - 用深拷贝的上一轮模型 + 全局原型 G 在本地数据上训练
           - 损失函数: L = L_BCE + λ * L_proto
           - 提取本地原型: P_k = { (label, proto_vec), ... }

        2. 客户端内原型聚合 (agg_func)
           - 同一客户端内同标签的多个原型取平均 / 方差合并
           - 得到聚合后的本地原型 P_k'

    (可选) 差分隐私: P_k' = DPMechProto.clip_and_noise(P_k')

    3. 服务器聚合 (proto_aggregation)
       - 收集所有参与客户端的 P_k'
       - 按标签合并，跨客户端取平均 / 贝叶斯融合
       - 更新全局原型 G

    4. 将本轮训练好的权重写回 local_model_list
       （保留各客户端个性化模型，不做 FedAvg 式权重聚合）

最终测试:
    - 不使用全局原型: sigmoid(logits) > 0.5 多标签分类
    - 使用全局原型: 负原型距离 → sigmoid → 二值预测
```

### 3.2 损失函数详解

FedProto 的本地训练使用组合损失：

```
L_total = L_CE + λ * L_proto
```

- **L_CE** (二值交叉熵损失): 多标签分类损失，每个类别独立判断

  ```
  L_BCE = -Σ [y_i * log(σ(logit_i)) + (1-y_i) * log(1-σ(logit_i))]
  ```
  其中 σ 为 sigmoid 函数，14 个类别各自独立计算，一个样本可同时属于多个类别。

- **L_proto** (原型正则化损失): 将本地原型拉向全局原型，实现知识迁移

  - **点原型模式** (默认): MSE 距离

    ```
    L_proto = (1/N) * Σ ||proto_i - global_proto[label_i]||²
    ```

  - **分布原型模式** (`--use_distributional`):

    | dist_type | 公式 |
    |-----------|------|
    | `kl` | KL(N(μ_loc, σ²_loc) || N(μ_gbl, σ²_gbl)) |
    | `wasserstein` | W₂² = \|\|μ_loc - μ_gbl\|\|² + \|\|σ_loc - σ_gbl\|\|²_F |
    | `mse` | \|\|μ_loc - μ_gbl\|\|² (回退到点原型) |

- **λ** (原型损失权重): 由 `--ld` 控制，默认 1.0。越大表示越强调全局知识迁移

### 3.3 多标签原型提取

ChestX-ray14 是多标签数据集（每张 X 光片可包含多种疾病）。FedProto 的多标签适配：

- **原型提取**：一张图片的特征向量会贡献给**所有正标签**对应的类别原型
  ```
  对样本 x 含正标签 {Atelectasis, Effusion}:
    proto = fc1(ResNet50(x))
    agg_protos['Atelectasis'].append(proto)
    agg_protos['Effusion'].append(proto)
  ```
- **原型损失**：对每张图的所有正标签，计算本地原型与对应全局原型的距离

### 3.4 原型聚合的两种层次

**层次一：客户端内聚合** (`agg_func` in [utils.py](lib/utils.py#L234))

同一个客户端处理一个 batch 时，同一类别可能有多张图片，因此有多个原型。先做客户端内聚合：

- 点原型: 直接求平均
  ```
  proto_label = (1/M) * Σ proto_i    (M 为该类样本数)
  ```

- 分布原型: 合并方差
  ```
  μ_avg = mean(mus)
  σ²_avg = mean(σ²_i) + Var(mus)   // E[Var] + Var[E]
  ```

**层次二：跨客户端聚合** (`proto_aggregation` in [utils.py](lib/utils.py#L146))

服务器收集所有客户端的本地原型后，按标签聚合：

- 点原型: 直接对所有客户端的该类别原型取平均
  ```
  global_proto_label = (1/K) * Σ proto_k    # 单一张量，不包装为列表
  ```

- 分布原型: **贝叶斯融合** (精度加权平均)
  ```
  precision_k = 1 / σ²_k
  μ_global = Σ (μ_k * prec_k) / Σ prec_k
  σ²_global = 1 / Σ prec_k
  ```
  这意味着方差小的客户端（更确定的估计）权重更大。

两种模式返回格式统一：点原型为单一张量，分布原型为 (mu, logvar) 元组。

### 3.5 推理/测试方式

FedProto 支持两种测试模式（多标签版本）：

**模式一：Sigmoid 阈值分类** (不使用全局原型)

```
对测试样本 x:
  logits, _ = model(x)
  pred = sigmoid(logits) > 0.5   # 14 维二值向量
```
只能分类当前客户端见过的类别。

**模式二：最近原型分类** (使用全局原型)

```
对测试样本 x:
  proto, _ = model(x)
  对每个类别 j:
    dist[j] = ||proto - global_proto[j]||²
    score[j] = -dist[j]          # 负距离 → 越近分数越高
  pred = sigmoid(score) > 0.5    # 二值多标签预测
```
可以分类任何在全局原型中存在的类别（即使客户端本地无该疾病样本）。

**评价指标**：per-label 准确率（所有 14 个标签位置的平均匹配率），而非单标签准确率。

对于分布原型，距离公式采用马氏距离形式：
```
dist = 0.5 * Σ (proto_i - μ_gbl_j)² / σ²_gbl_j
```

---

## 四、分布原型扩展

### 4.1 动机

点原型只传递了一个点估计，丢失了不确定性信息。如果某个客户端的数据很少（few-shot），它的原型估计应该伴随更大的不确定性。高斯分布原型同时传递**位置** (μ) 和**不确定度** (σ²)。

### 4.2 ProbabilisticProtoHead

[proto_head.py](lib/dist_proto/proto_head.py) 实现了一个双头线性层，将 fc1 的输出映射为分布参数：

```
μ  = W_μ * h + b_μ      (均值)
log σ² = clamp(W_σ * h + b_σ, -10, 10)     (对数方差)
```

clamp 防止方差爆炸或退化。

### 4.3 距离度量

| 度量 | 公式 | 特点 |
|------|------|------|
| **KL 散度** | KL(q\|\|p) = 0.5 * [log(σ²_p/σ²_q) + (σ²_q + (μ_q-μ_p)²)/σ²_p - 1] | 不对称，衡量 p 对 q 的覆盖 |
| **2-Wasserstein** | W₂² = \|\|μ_q-μ_p\|\|² + \|\|σ_q-σ_p\|\|²_F | 对称，考虑均值和方差的几何距离 |
| **MSE** | \|\|μ_q-μ_p\|\|² | 仅均值，回退到点原型 |

### 4.4 贝叶斯融合

[aggregation.py](lib/dist_proto/aggregation.py) 实现了精度加权的贝叶斯融合，是高斯观测的最优合并方式：

```
μ* = Σ (μ_k / σ²_k) / Σ (1 / σ²_k)
σ²* = 1 / Σ (1 / σ²_k)
```

融合后方差必然小于任一方差，体现了"多个观测降低不确定性"的贝叶斯原理。

---

## 五、原型解耦扩展 (DPP-FL 专属)

### 5.1 动机

在跨医院联邦学习中，不同客户端的 X 光片存在**域偏移 (Domain Shift)**：

| 风格来源 | 说明 |
|----------|------|
| 设备差异 | 不同厂商的 X 光机成像对比度、亮度不同 |
| 采集参数 | kVp、mAs、曝光时间等设置差异 |
| 后处理 | 各医院 PACS 系统的窗宽窗位、锐化参数 |
| 患者群体 | 不同地区人群体型、年龄分布差异 |

这些风格信息会混入原型向量中，产生两个问题：

1. **聚合噪声**：风格差异被当作语义差异跨客户端聚合，污染全局原型
2. **DP 效率低**：高斯噪声加在"语义+风格"混合信号上，有效信噪比被稀释

**核心洞察**：如果将原型分解为 **语义 (semantic)** 和 **风格 (style)** 两个独立子空间，仅共享语义部分，则：
- 语义原型更纯净 → 跨客户端聚合方差更小
- 同等 DP budget 下有效信号占比更高 → 隐私-效用 tradeoff 更优

### 5.2 实现方式

```
fc1 输出 (256-dim)
  ├── z_sem  (前 192 维) ──→ [ProtoHead] ──→ 共享到服务器
  └── z_style (后 64 维)  ──→ [ProtoHead] ──→ 保留本地

分类: logits = fc2(concat(z_sem, z_style))  ← 分类头同时使用两部分
```

**语义部分**：编码疾病判别特征（病灶形状、纹理、位置），跨客户端一致。
**风格部分**：编码医院成像特性（对比度、噪声模式），辅助本地分类但不共享。

### 5.3 独立性约束 (HSIC)

为强制两部分编码独立信息，在本地训练中加入 Hilbert-Schmidt Independence Criterion (HSIC) 损失。

使用线性核 HSIC 的简化形式 —— 交叉协方差矩阵的 Frobenius 范数：

```
L_dis = ||Cov(z_sem, z_style)||_F^2

其中 Cov(z_sem, z_style)_ij = (1/(n-1)) * Σ_k (z_sem_k_i - z_sem̄_i) * (z_style_k_j - z_stylē_j)
```

该损失惩罚语义和风格之间的所有线性相关性：

- **L_dis 大**：语义维度与风格维度高度相关 → 有风格泄露到语义中
- **L_dis 小**：两部分统计独立 → 成功解耦

辅助方差正则项防止解耦过程中特征坍缩到零。

### 5.4 训练流程（解耦版本）

```
本地训练:
  L_total = L_BCE + λ * L_proto_sem + λ_dis * L_dis

  L_BCE      ← 分类损失（使用完整 z_sem + z_style）
  L_proto_sem ← 仅对语义部分计算与全局原型的距离
  L_dis      ← 交叉协方差独立性约束（本地计算，不上传）

服务器聚合（仅语义原型）:
  P_k^sem = agg_func(z_sem)                   ← 客户端内语义原型聚合
  G_new = proto_aggregation({P_k^sem})        ← 跨客户端仅聚合语义

  z_style 永远不出客户端 ← 风格信息 100% 本地化
```

### 5.5 与 DP 的协同效应

解耦 + DP 的组合有一个独特的理论优势：

| | 不解耦 | 解耦 |
|---|---|---|
| 上传向量维度 | proto_dim (256) | sem_dim (192, 75%) |
| 向量内容 | 语义 + 风格混合 | 纯语义 |
| 风格噪声 | 跨客户端随机波动 → 等效噪声 | 零（不上传） |
| DP 噪声影响 | 同时损害语义和风格 | 仅损害语义 |
| 有效信噪比 | 低 | 高 |

在低 ε 场景下，解耦 vs 不解耦的性能差距应该更大——这是论文的核心卖点。

### 5.6 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_disentangle` | False | 启用原型解耦（仅 DPP-FL） |
| `--sem_ratio` | 0.75 | 语义维度占比 (0.75 → 192/256) |
| `--dis_lambda` | 0.05 | 独立性损失权重 |

### 5.7 理论性质

**命题 1 (聚合方差上界)**：Non-IID 程度为 η 时，解耦后的语义原型聚合方差上界降低为原始的 `sem_ratio²` 倍。

直观：风格噪声（约占 25% 维度）不再参与聚合，方差来源减少。

**命题 2 (DP 信噪比)**：在相同 ε 预算下，解耦后语义原型的有效信噪比为：

```
SNR_dis / SNR_orig = (sem_dim / proto_dim)^(-1/2) ≈ 1.15   (当 sem_ratio=0.75)
```

这是因为 DP 高斯噪声的 L2 范数期望与维度有关，更小的维度意味着同等 ε 下更少的绝对噪声量。

---

## 六、差分隐私 (DP) 扩展

### 6.1 动机

所有 6 种算法均支持差分隐私保护，通过 `--use_dp` 统一开启。原型向量或模型权重在上传前进行 L2 裁剪 + 高斯噪声扰动。

### 6.2 机制

**原型方式 (FedProto / DPP-FL)**: 使用 `DPMechProto`，对 `(μ || logvar)` 拼接向量进行 L2 裁剪 + 高斯噪声。

**权重方式 (FedAvg / FedProx / FedBN / SCAFFOLD)**: 使用 `DPMechWeight`，对权重 delta `w_local - w_global` 进行 L2 裁剪 + 高斯噪声。

```
v_clipped = v * min(1, C / ||v||₂)
v_noisy = v_clipped + N(0, σ² * C² * I)
```

其中 `σ` 由二分搜索根据目标 `(ε, δ)` 确定。

### 6.3 隐私预算追踪 (Moments Accountant)

使用 **Rényi Differential Privacy (RDP)** 进行跨轮隐私预算累积：

```
每轮消耗: ε_rdp(λ) = λ / (2σ²)
多轮累积: ε_rdp_total(λ) = Σ ε_rdp_i(λ)
转换为 (ε, δ)-DP: ε = min_λ [ε_rdp_total(λ) - log(δ)/(λ-1)]
```

### 6.4 关键参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--use_dp` | 启用差分隐私（所有算法） | False |
| `--dp_epsilon` | 目标 ε | 8.0 |
| `--dp_delta` | 目标 δ | 1e-5 |
| `--dp_clip` | L2 裁剪范数 | 1.0 |

---

## 七、任务异构

不同客户端拥有**不同的类别集合**。例如 20 个客户端，每个客户端随机分配到 3~5 个类别。

设置方式：
- `--ways`: 平均每客户端类别数 (默认 3)
- `--stdev`: 类别数随机波动的标准差 (默认 2)

每个客户端的类别数 = random(ways - stdev, ways + stdev)

---

## 八、模型架构

### 预训练 ResNet-50 Backbone (所有算法共用)

所有算法统一使用 `DPPFLResNet`：ImageNet 预训练 ResNet-50 + 原型头 + 分类头。

```
Input (3, 224, 224)  ← 灰度X光片经 Grayscale(3) 转3通道
  → stem: conv1 (7×7) → BN → ReLU → MaxPool  ← ImageNet 预训练权重
  → layer1: 3× Bottleneck(64→256)    ← 冻结
  → layer2: 4× Bottleneck(256→512)   ← 冻结
  → layer3: 6× Bottleneck(512→1024)  ← 微调
  → layer4: 3× Bottleneck(1024→2048) ← 微调
  → AdaptiveAvgPool2d(1) → Flatten → (2048-dim)
  → fc1 (2048 → proto_dim=256) → ReLU  ← 原型特征
  → [ProbabilisticProtoHead]  ← (可选) 分布原型
  → fc2 (256 → 14)  ← 多标签 logits
```

Bottleneck expansion=4，总参数量约 23M。

---

## 九、数据划分策略

[采样模块](lib/sampling.py) 支持多种 Non-IID 数据划分：

### 9.1 标签倾斜 (Label Skew) —— 默认策略

每个客户端只拥有部分疾病类别的数据。对于多标签 ChestX-ray14，图片按**首个阳性标签**归类后排序分配：

```
Client 0: 疾病 [Atelectasis, Effusion, Mass], 每类 ~100 样本
Client 1: 疾病 [Cardiomegaly, Nodule, Pneumonia], 每类 ~100 样本
...
```

"No Finding"（无疾病）样本均匀分配给所有客户端作为负样本。

### 9.2 数量倾斜 (Quantity Skew)

通过 `--unequal` 开启。不同客户端拥有不同数量的分片（shard）。

### 9.3 本地测试集 (Local Test)

`user_groups_lt`: 每个客户端的本地测试数据，仅包含该客户端训练过的疾病类别。

---

## 十、系统架构

```
exps/federated_main.py          ← 主入口
├── lib/options.py              ← 参数解析
├── lib/utils.py                ← 数据加载、权重聚合、原型聚合
│   ├── lib/sampling.py         ← IID/Non-IID 数据划分
│   └── lib/chestxray.py        ← ChestX-ray14 数据集类
├── lib/update.py               ← 本地训练、测试、多标签原型提取
├── lib/models/
│   ├── models.py               ← CNNMnist (MNIST)
│   └── resnet.py               ← ResNet50 (ChestX-ray14)
├── lib/dist_proto/             ← 分布原型子模块
│   ├── proto_head.py           ← ProbabilisticProtoHead (μ, logvar)
│   ├── losses.py               ← KL, Wasserstein, MSE 损失
│   ├── aggregation.py          ← 贝叶斯融合
│   └── disentangle.py          ← DisentangledProtoHead + HSIC损失 (DPP-FL专属)
├── lib/dp/                     ← 差分隐私子模块
│   └── mechanisms.py           ← DPMechProto, MomentsAccountant
└── lib/visualize.py            ← t-SNE 原型可视化
```

### 10.1 调用关系

```
federated_main.py
  │
  ├─ args_parser()             → 解析命令行参数
  ├─ get_dataset()             → 加载 + 划分数据
  │   ├─ ChestXray14()         → 读取 Data_Entry_2017.csv + PNG 图片
  │   └─ sampling.chestxray_noniid() → 多标签 Non-IID 划分
  │
  ├─ 构建 local_model_list[]   → 每个客户端一个 ResNet50
  │
  └─ FedProto_taskheter()      → 主训练循环
      │
      For each round:
        ├─ 采样 m = frac * K 个客户端 (idxs_users)
        For each selected client:
          ├─ LocalUpdate.update_weights_het()
          │   ├─ model(copy.deepcopy(local_model)) → (logits, protos) 或 (logits, mu, logvar)
          │   ├─ loss = BCE + λ * proto_loss   ← 多标签损失
          │   │   └─ distributional_proto_loss()  → KL / Wasserstein / MSE
          │   └─ 客户端内 agg_func()              → 多标签原型取平均/合并方差
          │
          ├─ DPMechProto.clip_and_noise() → (可选) DP 扰动
          │
        ├─ proto_aggregation()          → 跨客户端原型聚合
        │   └─ bayesian_fusion_single_label() → (可选) 分布原型贝叶斯融合
        │
        └─ 将本轮训练权重写回 local_model_list

      test_inference_new_het_lt()
        ├─ 不使用全局原型: sigmoid(logits) > 0.5  (per-label)
        └─ 使用全局原型: 负原型距离 → sigmoid → 二值预测 (per-label)
```

---

## 十一、对比算法详解

本实现支持 5 种 FL 算法对比，通过 `--alg` 参数切换。

### 11.1 FedAvg (McMahan et al., AISTATS 2017)

最基础的联邦学习算法，仅做权重聚合：

```
每轮:
  Server → Clients: 广播全局模型 w_t
  Clients: 在本地数据上多轮 SGD 训练得到 w_t^k
  Server: w_{t+1} = (1/K) * Σ w_t^k
```

- **优点**：简单、通信量可调
- **缺点**：Non-IID 数据下性能退化严重（client drift）
- **参数**：无额外参数

### 11.2 FedProx (Li et al., MLSys 2020)

在 FedAvg 基础上对本地目标函数添加近端项（proximal term），限制本地模型不要偏离全局模型太远：

```
L_local = L_BCE + (μ/2) * ||w - w_t||^2
```

- **优点**：部分缓解 Non-IID 导致的 client drift
- **缺点**：μ 需要调参；μ 太大导致收敛慢，太小等于 FedAvg
- **参数**：`--fedprox_mu` (默认 0.01)

### 11.3 FedBN (Li et al., ICLR 2021)

针对 **feature shift**（不同客户端数据分布不同导致的特征偏移）的解决方案：

- 本地训练：正常 SGD（BN 层统计量本地更新）
- 服务器聚合：**跳过所有 BN 层参数**（running_mean, running_var, weight, bias），仅聚合 conv 和 linear 层

```
aggregate: {conv, linear} ← 平均值
keep local: {bn.running_mean, bn.running_var, bn.weight, bn.bias}
```

- **优点**：适合医疗影像场景（不同医院设备扫描参数不同导致 feature shift）
- **缺点**：标签分布偏移（label skew）场景下效果有限
- **参数**：无额外参数

### 11.4 SCAFFOLD (Karimireddy et al., ICML 2020)

使用 **control variates** 纠正 client drift：

- 服务器维护全局 control variate `c`
- 每个客户端维护本地 control variate `c_i`
- 本地训练时修正梯度：`g_corrected = g - c_i + c`
- 训练后更新：`c_i = c_i - c + (w_global - w_local) / (lr * K)`
- 服务器更新：`c = c + (1/K) * Σ (c_i_new - c_i)`

- **优点**：理论上可完全消除 client drift，收敛速度快
- **缺点**：需要传输 control variate（与模型同大小），通信量翻倍；stateful（需保存每个客户端的 c_i）
- **参数**：`--scaffold_lr` (全局 LR，默认等于 `--lr`)

### 11.5 算法对比总结

所有算法均使用 ImageNet 预训练 ResNet-50 + 可选 DP。

| 算法 | 类型 | 共享内容 | 通信量 | Non-IID 处理 | DP 方式 |
|------|------|----------|--------|-------------|---------|
| FedAvg | 权重共享基线 | 模型权重 (~23M) | 高 | — | 权重 delta |
| FedProx | 权重共享基线 | 模型权重 | 高 | 近端约束 | 权重 delta |
| FedBN | 权重共享基线 | 模型权重 (skip BN) | 高 | 本地 BN | 权重 delta |
| SCAFFOLD | 权重共享基线 | 权重 + control variates | 极高 (~2x) | 梯度修正 | 权重 delta |
| **FedProto** | **原型共享基线** | 点原型 (256dx14) | **低** | 原型正则化 | 原型向量 |
| **DPP-FL** | **原型共享 (提出)** | 高斯原型 `N(mu, sigma^2)` | **低** | 分布原型 + 贝叶斯融合 | 原型向量 |

### 11.6 DPP-FL 与 FedProto 的区别

| 特性 | FedProto (基线) | DPP-FL (提出方法) |
|------|----------------|-------------------|
| 骨干网络 | 预训练 ResNet-50 | 预训练 ResNet-50 |
| 原型类型 | 点向量 `p in R^d` | 高斯分布 `N(mu, sigma^2)` |
| 聚合方式 | 简单平均 | 精度加权贝叶斯融合 |
| 不确定性 | 不建模 | Per-client variance |
| 原型解耦 | 无 | 语义-风格分离 + HSIC 独立性约束 |
| 原型动量 | 无 | EMA `G_t = β*G_{t-1} + (1-β)*G_new` |
| λ 调度 | 常数 | Warmup: `λ * min(1, round/W)` |
| 推理温度 | 1.0 | 可配置 `-dist / T` |
| 距离度量 | MSE | KL / Wasserstein / MSE |
| 隐私保护 | 可选 (ε, δ)-DP | 可选 (ε, δ)-DP |
| 适用场景 | 一般 Non-IID | 高异质性 + 隐私敏感 + 域偏移场景 |

---

## 十二、关键参数速查

### 算法选择

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--alg` | dppfl | FL 算法: fedproto / fedavg / fedprox / fedbn / scaffold / dppfl |

### 基础联邦参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_users` | 20 | 客户端数量 |
| `--frac` | 0.04 | 每轮参与训练的客户端比例 |
| `--rounds` | 100 | 全局通信轮数 |
| `--train_ep` | 1 | 每轮本地训练 epoch 数 |
| `--local_bs` | 4 | 本地批次大小 |
| `--lr` | 0.01 | 学习率 |
| `--momentum` | 0.5 | SGD 动量 |
| `--optimizer` | sgd | 优化器: sgd/adam |
| `--model` | resnet50 | 模型: resnet50 / cnn |
| `--num_classes` | 14 | 类别数 (ChestX-ray14) |
| `--num_channels` | 3 | 输入通道 (灰度转RGB) |
| `--image_size` | 224 | 输入图像尺寸 |

### 异构度参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--iid` | False | 是否 IID 划分 |
| `--ways` | 3 | 每客户端平均类别数 |
| `--shots` | 100 | 每类平均样本数 |
| `--stdev` | 2 | 类别数/样本数的随机波动标准差 |
| `--unequal` | False | 是否不等量划分 |

### 原型损失参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ld` | 1.0 | 原型损失权重 λ |
| `--use_distributional` | False | 启用高斯分布原型 |
| `--dist_type` | kl | 分布距离类型: kl/wasserstein/mse |
| `--proto_dim` | 256 | 原型向量维度 (ResNet-50 默认 256) |

### 差分隐私参数 (所有算法共用)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_dp` | False | 启用差分隐私（权重/原型上传） |
| `--dp_epsilon` | 8.0 | 目标 ε |
| `--dp_delta` | 1e-5 | 目标 δ |
| `--dp_clip` | 1.0 | L2 裁剪范数 |

### DPP-FL 专属参数 (提出方法)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--proto_momentum` | 0.9 | 全局原型 EMA 动量系数 |
| `--ld_warmup` | 50 | 原型损失权重 warmup 轮数 |
| `--temperature` | 1.0 | 原型推理温度系数 |
| `--pretrained` | True | ImageNet 预训练 (所有算法默认开启) |

### 原型解耦参数 (DPP-FL 专属)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_disentangle` | False | 启用原型语义-风格解耦 |
| `--sem_ratio` | 0.75 | 语义维度占比 |
| `--dis_lambda` | 0.05 | 解耦独立性损失 (HSIC) 权重 |

---

## 十三、运行示例

```bash
# === 所有算法均使用预训练 ResNet-50，通过 --use_dp 开启 DP ===

# FedProto (原始点原型)
python exps/federated_main.py --alg fedproto \
    --ways 5 --shots 100 --num_users 20 --rounds 200 --ld 1.0

# FedProto + DP
python exps/federated_main.py --alg fedproto \
    --use_dp --dp_epsilon 8.0 --dp_clip 1.0 \
    --ways 5 --num_users 20 --rounds 30

# FedAvg + DP
python exps/federated_main.py --alg fedavg \
    --use_dp --dp_epsilon 8.0 --dp_clip 1.0 \
    --ways 5 --num_users 20 --rounds 30 --frac 1.0

# FedProx + DP
python exps/federated_main.py --alg fedprox \
    --use_dp --dp_epsilon 8.0 --dp_clip 1.0 \
    --fedprox_mu 0.01 --ways 5 --num_users 20 --rounds 30

# FedBN + DP
python exps/federated_main.py --alg fedbn \
    --use_dp --dp_epsilon 8.0 --dp_clip 1.0 \
    --ways 5 --num_users 20 --rounds 30 --frac 1.0

# SCAFFOLD + DP
python exps/federated_main.py --alg scaffold \
    --use_dp --dp_epsilon 8.0 --dp_clip 1.0 \
    --ways 5 --shots 100 --num_users 20 --rounds 30

# === 提出方法: DPP-FL ===
# 点原型
python exps/federated_main.py --alg dppfl \
    --ways 5 --shots 100 --num_users 20 --rounds 200 --ld 1.0

# 分布原型 (KL 散度) + DP
python exps/federated_main.py --alg dppfl \
    --use_distributional --dist_type kl \
    --use_dp --dp_epsilon 8.0 --dp_clip 1.0 \
    --ways 5 --rounds 30

# 完整 DPP-FL (所有增强)
python exps/federated_main.py --alg dppfl \
    --use_distributional --dist_type kl \
    --use_dp --dp_epsilon 8.0 --dp_clip 1.0 \
    --proto_momentum 0.9 --temperature 0.5 --ld_warmup 50 \
    --ways 5 --num_users 20 --rounds 100

# === 原型解耦 (DPP-FL 专属) ===
# 点原型 + 解耦
python exps/federated_main.py --alg dppfl \
    --use_disentangle --sem_ratio 0.75 --dis_lambda 0.05 \
    --ways 5 --shots 100 --num_users 20 --rounds 200 --ld 1.0

# 分布原型 + 解耦 + DP (完整 SOTA 配方)
python exps/federated_main.py --alg dppfl \
    --use_distributional --dist_type kl \
    --use_disentangle --sem_ratio 0.75 --dis_lambda 0.05 \
    --use_dp --dp_epsilon 8.0 --dp_clip 1.0 \
    --proto_momentum 0.9 --temperature 0.5 --ld_warmup 50 \
    --ways 5 --num_users 20 --rounds 100

# 快速测试 (MNIST)
python exps/federated_main.py --alg fedavg \
    --model cnn --num_classes 10 --iid 1 --rounds 50 --num_users 10
```

---

## 十四、数学符号汇总

| 符号 | 含义 |
|------|------|
| K | 客户端总数 |
| N_k | 客户端 k 的样本数 |
| C | 总类别数 (ChestX-ray14: 14) |
| c_k | 客户端 k 拥有的类别集合 |
| f_k | 客户端 k 的本地模型 (ResNet-50) |
| p_k^(i) | 客户端 k 中样本 i 的原型向量 |
| P_k^(j) | 客户端 k 中类别 j 的聚合原型 |
| G^(j) | 类别 j 的全局原型 |
| L_BCE | 二值交叉熵多标签分类损失 |
| L_proto | 原型距离损失 |
| λ | 原型损失权重 (`--ld`) |
| μ, σ² | 高斯分布原型的均值和方差 |
| ε, δ | 差分隐私参数 |
| z_sem | 语义原型向量 (共享) |
| z_style | 风格原型向量 (本地) |
| L_dis | 解耦独立性损失 (HSIC) |
| λ_dis | 解耦损失权重 (`--dis_lambda`) |
| y^(i) | 样本 i 的 14 维多标签二值向量 |

---

## 十五、实现细节与注意事项

### 15.1 原型格式统一

全局原型字典 `global_protos` 的 value 格式在两种模式下保持一致：

- **点原型**: 单一张量 `tensor(shape=[proto_dim])`，不再包装为列表
- **分布原型**: 二元组 `(mu: tensor, logvar: tensor)`，各自 shape=[proto_dim]

调用方不需要根据模式做不同的索引处理（已修复原有的 `[0]` 包装不对称问题）。

### 15.2 客户端采样

`--frac` 参数控制每轮参与训练的客户端比例。每轮随机采样 `m = max(1, int(frac * K))` 个客户端：

- 降低通信开销
- 增加随机性，有助于泛化
- 未参与轮的客户端保留上一轮模型，在后续轮次可被选中继续训练

### 15.3 多标签原型损失计算

原型正则化损失 `L_proto` 的计算方式（以点原型为例）：

```
对 batch 中每张图 i:
  对每个正标签 j (labels[i, j] == 1):
    loss2 += MSE(proto_i, global_protos[j])
loss2 = loss2 / count  # 除以所有正标签总数
```

即平均到每个正标签上，而非每张图。这意味着有多个疾病的 X 光片对原型损失的贡献更大。

### 15.4 "No Finding" 负样本处理

在 Non-IID 划分时，"No Finding"（标签全为 0）的样本不按疾病标签排序，而是均匀分配给所有客户端（每个客户端至少 10 张），作为负样本参与训练。这确保了每个客户端都能学到"正常"的表示。

### 15.5 分布原型的数值稳定性

- `logvar` 被 clamp 到 [-10, 10] 范围（`ProbabilisticProtoHead`）
- `var = exp(logvar)`，对应方差范围约 [4.5e-5, 2.2e4]
- `agg_func` 中 `logvar_avg = log(avg_var + 1e-8)` 防止 log(0)
- 推理时 `g_var + 1e-8` 防止除零

## 参考文献

- Tan, Y., et al. "FedProto: Federated Prototype Learning across Heterogeneous Clients." *AAAI 2022*. [arXiv:2105.00243](https://arxiv.org/abs/2105.00243)
- McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS 2017*. (FedAvg)
- Li, T., et al. "Federated Optimization in Heterogeneous Networks." *MLSys 2020*. (FedProx)
- Mironov, I. "Rényi Differential Privacy." *CSF 2017*. (RDP / Moments Accountant)
