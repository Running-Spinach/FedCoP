# FedProto 联邦原型学习 —— 理论与实现详解 (ChestX-ray14 + ResNet-50)

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
输入: K 个客户端，每客户端有本地模型 f_k，全局轮数 T
输出: 训练好的本地模型列表，全局原型集合 G

初始化: 全局原型 G = {}  (空)

For round t = 1 to T:
    For each client k = 1 to K (并行):
        1. 本地训练
           - 使用本地数据 + 全局原型 G 训练模型 f_k
           - 损失函数: L = L_CE + λ * L_proto
           - 提取本地原型: P_k = { (label, proto_vec), ... }

        2. 客户端内原型聚合
           - 同一客户端内同标签的多个原型取平均
           - 得到聚合后的本地原型 P_k'

    (可选) 差分隐私: P_k' = P_k' + GaussianNoise

    3. 服务器聚合
       - 收集所有客户端的 P_k'
       - 按标签合并，跨客户端取平均
       - 更新全局原型 G

最终测试:
    - 不使用全局原型: 各客户端直接用本地模型 softmax 分类
    - 使用全局原型: 基于最近原型距离分类 (1-NN with prototypes)
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

**层次二：跨客户端聚合** (`proto_aggregation` in [utils.py](lib/utils.py#L270))

服务器收集所有客户端的本地原型后，按标签聚合：

- 点原型: 直接对所有客户端的该类别原型取平均
  ```
  global_proto_label = (1/K) * Σ proto_k
  ```

- 分布原型: **贝叶斯融合** (精度加权平均)
  ```
  precision_k = 1 / σ²_k
  μ_global = Σ (μ_k * prec_k) / Σ prec_k
  σ²_global = 1 / Σ prec_k
  ```
  这意味着方差小的客户端（更确定的估计）权重更大。

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

## 五、差分隐私 (DP) 扩展

### 5.1 动机

原型向量虽然不直接暴露原始数据，但仍可能泄露统计信息。差分隐私提供形式化的隐私保证。

### 5.2 机制: 高斯噪声扰动

[mechanisms.py](lib/dp/mechanisms.py) 实现了针对原型上传的 DP 保护：

1. **L2 裁剪**: 将 `(μ || logvar)` 拼接向量的 L2 范数裁剪到 `clip_norm` 以内

   ```
   v_clipped = v * min(1, C / ||v||₂)
   ```

2. **高斯噪声**: 添加零均值高斯噪声

   ```
   v_noisy = v_clipped + N(0, σ² * C² * I)
   ```

   其中 `σ` 由二分搜索根据目标 `(ε, δ)` 确定。

### 5.3 隐私预算追踪 (Moments Accountant)

使用 **Rényi Differential Privacy (RDP)** 进行跨轮隐私预算累积：

```
每轮消耗: ε_rdp(λ) = λ / (2σ²)
多轮累积: ε_rdp_total(λ) = Σ ε_rdp_i(λ)
转换为 (ε, δ)-DP: ε = min_λ [ε_rdp_total(λ) - log(δ)/(λ-1)]
```

### 5.4 关键参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--dp_epsilon` | 目标 ε | 8.0 |
| `--dp_delta` | 目标 δ | 1e-5 |
| `--dp_clip` | L2 裁剪范数 | 1.0 |

注意：ε=8 是相对宽松的隐私预算，适合研究场景。生产环境通常需要 ε < 1。

---

## 六、任务异构

不同客户端拥有**不同的类别集合**。例如 20 个客户端，每个客户端随机分配到 3~5 个类别。

设置方式：
- `--ways`: 平均每客户端类别数 (默认 3)
- `--stdev`: 类别数随机波动的标准差 (默认 2)

每个客户端的类别数 = random(ways - stdev, ways + stdev)

---

## 七、模型架构

### ResNet-50 Backbone

```
Input (3, 224, 224)  ← 灰度X光片经 Grayscale(3) 转3通道
  → conv1 (7×7, stride=2) → BN → ReLU → MaxPool(3×3, stride=2)
  → layer1: 3× Bottleneck(64→256)   输出 (256, 56, 56)
  → layer2: 4× Bottleneck(256→512)  输出 (512, 28, 28)
  → layer3: 6× Bottleneck(512→1024) 输出 (1024, 14, 14)
  → layer4: 3× Bottleneck(1024→2048) 输出 (2048, 7, 7)
  → AdaptiveAvgPool2d(1) → Flatten → (2048-dim)
  → fc1 (2048 → proto_dim=256) → ReLU  ← 原型特征
  → [ProbabilisticProtoHead]  ← (可选) 分布原型
  → fc2 (256 → 14)  ← 多标签 logits
```

Bottleneck expansion=4，总参数量约 23M。

---

## 八、数据划分策略

[采样模块](lib/sampling.py) 支持多种 Non-IID 数据划分：

### 8.1 标签倾斜 (Label Skew) —— 默认策略

每个客户端只拥有部分疾病类别的数据。对于多标签 ChestX-ray14，图片按**首个阳性标签**归类后排序分配：

```
Client 0: 疾病 [Atelectasis, Effusion, Mass], 每类 ~100 样本
Client 1: 疾病 [Cardiomegaly, Nodule, Pneumonia], 每类 ~100 样本
...
```

"No Finding"（无疾病）样本均匀分配给所有客户端作为负样本。

### 8.2 数量倾斜 (Quantity Skew)

通过 `--unequal` 开启。不同客户端拥有不同数量的分片（shard）。

### 8.3 本地测试集 (Local Test)

`user_groups_lt`: 每个客户端的本地测试数据，仅包含该客户端训练过的疾病类别。

---

## 九、系统架构

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
│   └── aggregation.py          ← 贝叶斯融合
├── lib/dp/                     ← 差分隐私子模块
│   └── mechanisms.py           ← DPMechProto, MomentsAccountant
└── lib/visualize.py            ← t-SNE 原型可视化
```

### 9.1 调用关系

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
        For each client:
          ├─ LocalUpdate.update_weights_het()
          │   ├─ model.forward()          → (logits, protos) 或 (logits, mu, logvar)
          │   ├─ loss = BCE + λ * proto_loss   ← 多标签损失
          │   │   └─ distributional_proto_loss()  → KL / Wasserstein / MSE
          │   └─ agg_func()              → 客户端内多标签原型聚合
          │
          ├─ DPMechProto.clip_and_noise() → (可选) DP 扰动
          └─ proto_aggregation()          → 跨客户端原型聚合
              └─ bayesian_fusion_single_label() → (可选) 贝叶斯融合

      test_inference_new_het_lt()
        ├─ 不使用全局原型: sigmoid(logits) > 0.5
        └─ 使用全局原型: 负原型距离 → sigmoid → 二值预测
```

---

## 十、关键参数速查

### 基础联邦参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_users` | 20 | 客户端数量 |
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

### 差分隐私参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_dp` | False | 启用差分隐私 |
| `--dp_epsilon` | 8.0 | 目标 ε |
| `--dp_delta` | 1e-5 | 目标 δ |
| `--dp_clip` | 1.0 | L2 裁剪范数 |

---

## 十一、运行示例

```bash
# 基础 FedProto (ChestX-ray14, ResNet-50, task heterogeneous, 5-way)
python exps/federated_main.py \
    --ways 5 --shots 100 --num_users 20 --rounds 200 --ld 1.0

# 分布原型模式 (KL 散度)
python exps/federated_main.py \
    --use_distributional --dist_type kl --ways 5 --rounds 200

# 差分隐私 + 分布原型
python exps/federated_main.py \
    --use_distributional --dist_type wasserstein \
    --use_dp --dp_epsilon 8.0 --dp_clip 1.0 --rounds 30

# 调整原型维度
python exps/federated_main.py \
    --proto_dim 128 --rounds 100
```

---

## 十二、数学符号汇总

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
| y^(i) | 样本 i 的 14 维多标签二值向量 |

---

## 参考文献

- Tan, Y., et al. "FedProto: Federated Prototype Learning across Heterogeneous Clients." *AAAI 2022*. [arXiv:2105.00243](https://arxiv.org/abs/2105.00243)
- McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS 2017*. (FedAvg)
- Li, T., et al. "Federated Optimization in Heterogeneous Networks." *MLSys 2020*. (FedProx)
- Mironov, I. "Rényi Differential Privacy." *CSF 2017*. (RDP / Moments Accountant)
