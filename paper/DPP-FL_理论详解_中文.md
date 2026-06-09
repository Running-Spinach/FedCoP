# DPP-FL 理论详解（中文版）

> 目标：用最直白的语言，把 DPP-FL 这篇论文的每个公式、每个设计选择都说清楚。
> 适合读者：已经知道 FL 基本概念，想深入理解 DPP-FL 为什么有效。

---

## 目录

1. [一句话概括](#1-一句话概括)
2. [背景：联邦学习到底难在哪](#2-背景联邦学习到底难在哪)
3. [从 FedAvg 到 FedProto 的进化](#3-从-fedavg-到-fedproto-的进化)
4. [DPP-FL 做了什么（先看全景）](#4-dpp-fl-做了什么先看全景)
5. [创新一：分布原型 —— 给原型加上"置信度"](#5-创新一分布原型--给原型加上置信度)
6. [创新二：贝叶斯融合 —— 谁的判断更靠谱就听谁的](#6-创新二贝叶斯融合--谁的判断更靠谱就听谁的)
7. [创新三：原型解耦 —— 把"病"和"机器"分开](#7-创新三原型解耦--把病和机器分开)
8. [创新四：EMA 动量 —— 让全局原型别"一惊一乍"](#8-创新四ema-动量--让全局原型别一惊一乍)
9. [创新五：λ 预热 + 温度缩放](#9-创新五λ-预热--温度缩放)
10. [差分隐私是怎么加进去的](#10-差分隐私是怎么加进去的)
11. [六个算法一张表对比](#11-六个算法一张表对比)
12. [完整训练流程（伪代码）](#12-完整训练流程伪代码)
13. [公式速查卡](#13-公式速查卡)
14. [常见问题 FAQ](#14-常见问题-faq)

---

## 1. 一句话概括

**DPP-FL = 让联邦学习中的"全局知识"（原型）变成带有不确定性估计的高斯分布，同时把跟疾病无关的"风格信息"去掉，只共享真正有用的"语义信息"。**

你把这个思路理解透了，所有的公式都是围绕它展开的。

---

## 2. 背景：联邦学习到底难在哪

### 2.1 联邦学习是什么

想象一个场景：10 家医院各自有 X 光片数据，但因为隐私法规，数据不能出医院。现在想联合训练一个 AI 诊断模型。

联邦学习的思路是：**数据不动，模型动**。每家医院在本地训练模型，只把模型参数发给中心服务器，服务器聚合后再发回去。

```
循环 N 轮：
  服务器 → 所有医院：这是最新的全局模型，拿回去用
  每家医院 → 在自己数据上训练，得到本地模型
  服务器 → 收到所有本地模型，取平均，得到新的全局模型
```

### 2.2 三个核心难题

| 难题 | 大白话解释 | 医疗场景例子 |
|------|-----------|------------|
| **数据异构** | 不同医院有的病多有的病少 | A 医院 90% 是肺炎，B 医院 80% 是心脏肥大 |
| **域偏移** | 不同机器拍出来的片子"风格"不一样 | 西门子 CT vs GE CT，对比度、亮度完全不同 |
| **隐私 vs 效用** | 加了隐私保护（加噪声）后模型变差 | 差分隐私加的噪声让有用信号被淹没 |

DPP-FL 的设计动机就是**同时解决这三个问题**。

---

## 3. 从 FedAvg 到 FedProto 的进化

### 3.1 FedAvg（2017）：最朴素的想法

**做法**：每家医院训练后，把模型权重取平均。

```
第 t 轮聚合：
  新全局模型 = (模型1 + 模型2 + ... + 模型N) / N
```

**问题**：医院 A 的模型擅长肺炎，医院 B 的模型擅长心脏肥大，直接平均 = 两个都不擅长了。这叫"客户端漂移"（client drift）。

### 3.2 FedProto（2022）：换个思路，不共享权重，共享"知识"

**关键洞察**：神经网络倒数第二层的输出向量（fc1 之后）就是样本的"特征表征"，称之为**原型**（Prototype）。

对于 ResNet-50：
```
X光图片 → ResNet 卷积层 → 2048维向量 → fc1 → 256维原型向量 → fc2 → 14个疾病的概率
                                                    ↑
                                            这就是"原型"
```

**FedProto 的做法**：
- 不传模型权重（2300 万个参数）
- 只传原型（256 维 × 14 类 = 3584 个浮点数）
- 通信量减少 **6000 倍**

**损失函数**：

$$\mathcal{L}_{\text{总}} = \underbrace{\mathcal{L}_{\text{分类}}}_{\text{做好诊断}} + \lambda \cdot \underbrace{\|\text{本地原型} - \text{全局原型}\|^2}_{\text{向全局知识靠拢}}$$

**FedProto 的局限**：

1. **点原型丢失了不确定性**：一个只有 10 张数据的小医院和一个有 10000 张数据的大医院，它们的原型"可信度"完全不同，但 FedProto 把它们同等对待
2. **风格污染**：不同 CT 机的成像特性会混进原型里，跨医院聚合时这些风格差异被误认为语义差异
3. **训练不稳定**：早期全局原型是噪声，却强行拉本地原型去对齐

DPP-FL 就是针对这三个局限逐一改进的。

---

## 4. DPP-FL 做了什么（先看全景）

```
DPP-FL = FedProto 基线
       + 创新1: 分布原型（点 → 高斯分布）
       + 创新2: 贝叶斯融合（简单平均 → 精度加权）
       + 创新3: 原型解耦（混在一起 → 语义/风格分离）
       + 创新4: EMA 动量（骤变 → 平滑更新）
       + 创新5: λ预热 + 温度缩放（常数 → 自适应）
       + 可选:  差分隐私保护
```

下面逐一展开。

---

## 5. 创新一：分布原型 —— 给原型加上"置信度"

### 5.1 直觉

**FedProto（点原型）**：原型 = 一个 256 维向量 `[0.3, -0.1, 0.8, ...]`

这相当于只说"这个类别的特征中心大概在这里"，没有说"我有多确定"。

**DPP-FL（分布原型）**：原型 = 一个 256 维的高斯分布，每个维度都有均值和方差：

```
维度 0: 均值 = 0.3,  方差 = 0.01  → "我很确定在 0.3 附近"
维度 1: 均值 = -0.1, 方差 = 0.50  → "大概在 -0.1 附近，但不是特别确定"
```

**物理含义**：
- **均值 $\mu$**：这个类别特征的"中心位置"（和点原型同含义）
- **方差 $\sigma^2$**：数据少 → 方差大（不确定），数据多 → 方差小（确定）

### 5.2 怎么得到均值和方差

在 fc1 之后加了两个并行的小网络（`ProbabilisticProtoHead`）：

```
fc1 输出 (256维)
    ├── Linear(256→256) → 均值 μ
    └── Linear(256→256) → clamp(-10, 10) → 对数方差 log(σ²)
```

公式：

$$\begin{aligned}
\boldsymbol{\mu} &= \mathbf{W}_\mu \cdot \mathbf{h} + \mathbf{b}_\mu \\[4pt]
\log \boldsymbol{\sigma}^2 &= \operatorname{clamp}_{[-10,10]}(\mathbf{W}_\sigma \cdot \mathbf{h} + \mathbf{b}_\sigma)
\end{aligned}$$

`clamp` 的作用：防止方差爆炸（$\sigma^2$ 可以到 $e^{10} \approx 22000$）或退化到 0（$e^{-10} \approx 0.000045$）。

### 5.3 分布之间的距离怎么算

两个高斯分布之间怎么算"距离"？三种方式（由 `--dist_type` 控制）：

#### (a) KL 散度（默认，推荐）

**公式**：

$$\text{KL}(q\|p) = \frac{1}{2}\left[\log\frac{\sigma_p^2}{\sigma_q^2} + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{\sigma_p^2} - 1\right]$$

**大白话**：$p$ 是全局原型（当作"标准答案"），$q$ 是本地原型。KL 衡量"用 $p$ 来近似 $q$ 会损失多少信息"。

**直觉**：
- 如果 $\sigma_q^2$ 很大（本地不确定），KL 会变小 → 不强制拉开距离的一方去对齐
- 如果 $\sigma_p^2$ 很小（全局很确定），KL 会变大 → 必须认真对齐

这就是 **"不确定的时候别勉强，确定的时候要认真"**。

#### (b) 2-Wasserstein 距离

**公式**：

$$W_2^2(q, p) = \|\boldsymbol{\mu}_q - \boldsymbol{\mu}_p\|^2 + \|\boldsymbol{\sigma}_q - \boldsymbol{\sigma}_p\|_F^2$$

**大白话**：把均值和标准差分别算欧几里得距离，再加起来。对称的，双向的。比 KL 更直观，但没有 KL 那种"自适应权重"的特性。

#### (c) MSE（退化模式）

**公式**：只算均值的距离 $\|\boldsymbol{\mu}_q - \boldsymbol{\mu}_p\|^2$

**白话**：退化为点原型模式，等于 FedProto。

---

## 6. 创新二：贝叶斯融合 —— 谁的判断更靠谱就听谁的

### 6.1 问题

FedProto 的聚合：所有客户端对该类别的原型取平均。

```
全局原型[肺炎] = (医院A原型 + 医院B原型 + 医院C原型) / 3
```

问题：医院 A 有 10000 张肺炎片子，医院 B 只有 10 张。直接平均 = 医院 B 的噪声和医院 A 的精确估计被同等对待。

### 6.2 精度加权平均

**核心公式**（贝叶斯融合）：

$$\boxed{\boldsymbol{\mu}^* = \frac{\sum_k \frac{\boldsymbol{\mu}_k}{\sigma_k^2}}{\sum_k \frac{1}{\sigma_k^2}}} \qquad \boxed{{\sigma^2}^* = \frac{1}{\sum_k \frac{1}{\sigma_k^2}}}$$

**大白话**：
- **均值融合**：每个客户端的均值 $\mu_k$ 被其精度 $1/\sigma_k^2$ 加权。方差小（精度高）的客户端权重更大。
- **方差融合**：融合后的方差 = 所有精度之和的倒数。**融合后方差一定小于任一个单独方差**。

**直觉验证**：
- 如果所有客户端方差一样：退化为简单平均
- 如果某个客户端方差极小：它的均值几乎直接成为全局均值
- 融合后方差变小：体现"多看几次就更确定"的直觉

### 6.3 两层聚合

**第一层：客户端内部聚合**

同一个客户端里，同一类别的多张图片产生多个原型。先做客户端内聚合，用**总方差定律**：

$$\begin{aligned}
\boldsymbol{\mu}_{\text{avg}} &= \frac{1}{M} \sum_{i=1}^{M} \boldsymbol{\mu}_i \\[4pt]
\boldsymbol{\sigma}^2_{\text{avg}} &= \underbrace{\frac{1}{M} \sum \boldsymbol{\sigma}_i^2}_{\text{平均方差（组内不确定性）}} + \underbrace{\operatorname{Var}(\{\boldsymbol{\mu}_i\})}_{\text{均值差异（组间离散度）}}
\end{aligned}$$

这符合直觉：同一个客户端内多张图的方差 = 每张图自身的不确定性 + 不同图之间的差异。

**第二层：跨客户端聚合**

用上面 6.2 的贝叶斯融合公式。

---

## 7. 创新三：原型解耦 —— 把"病"和"机器"分开

这是 DPP-FL 最核心、理论上最漂亮的创新。

### 7.1 问题：风格污染

不同医院用不同设备拍 X 光片，会产生**域偏移**：

| 差异来源 | 例子 |
|---------|------|
| 设备 | 西门子 vs GE vs 飞利浦 X 光机 |
| 参数 | kVp（管电压）、mAs（管电流）、曝光时间 |
| 后处理 | 各医院 PACS 系统的窗宽窗位、锐化设置 |
| 人群 | 不同地区患者体型、年龄分布 |

这些差异会混进原型向量里。当原型跨医院聚合时，来自 GE 机器的"风格"会被当成疾病特征，污染全局原型。

### 7.2 核心思路

**把原型拆成两个独立的部分**：

```
原原型 (256维)
  ├── 语义部分 (前 192维 = 75%)：编码疾病特征 → 共享到服务器
  └── 风格部分 (后 64维  = 25%)：编码机器特征 → 留在本地
```

- **语义**：病灶的形状、纹理、位置 → 不管在哪台机器上拍，肺炎就是肺炎
- **风格**：对比度、亮度、噪声模式 → GE 和西门子不一样，但跟疾病无关

**分类时两部分都用**（`logits = fc2([语义, 风格])`），但**上传时只传语义**。

### 7.3 怎么保证真的"解耦"了

光切分维度不够，因为网络可能把风格信息藏进语义维度里。需要**强制两部分统计独立**。

**HSIC（Hilbert-Schmidt Independence Criterion）**，用线性核的简化版：

$$\boxed{\mathcal{L}_{\text{dis}} = \|\operatorname{Cov}(\mathbf{z}_{\text{sem}}, \mathbf{z}_{\text{style}})\|_F^2}$$

**逐行解释**：

1. 对一批数据，计算语义特征 $\mathbf{z}_{\text{sem}}$ 和风格特征 $\mathbf{z}_{\text{style}}$
2. 计算它们之间的**交叉协方差矩阵** $\operatorname{Cov}$（每个元素 = 一个语义维度和一个风格维度之间的相关性）
3. 取 Frobenius 范数的平方（= 所有矩阵元素的平方和）

**结果**：
- 如果语义和风格完全独立 → 交叉协方差 ≈ 零矩阵 → $\mathcal{L}_{\text{dis}} \approx 0$
- 如果存在相关性 → $\mathcal{L}_{\text{dis}} > 0$ → 梯度会惩罚这种相关性

**辅助正则**：防止网络偷懒把所有特征都设成 0

$$\mathcal{L}_{\text{var\_reg}} = \operatorname{ReLU}(0.01 - \operatorname{Var}(\mathbf{z}_{\text{sem}})) + \operatorname{ReLU}(0.01 - \operatorname{Var}(\mathbf{z}_{\text{style}}))$$

**总解耦损失**：

$$\mathcal{L}_{\text{解耦}} = \|\operatorname{Cov}(\mathbf{z}_{\text{sem}}, \mathbf{z}_{\text{style}})\|_F^2 + 0.1 \cdot \mathcal{L}_{\text{var\_reg}}$$

### 7.4 完整训练损失（解耦 + 分布原型模式）

$$\boxed{\mathcal{L}_{\text{总}} = \underbrace{\mathcal{L}_{\text{BCE}}}_{\text{诊断分类}} + \lambda \cdot \underbrace{\mathcal{L}_{\text{proto}}^{\text{sem}}}_{\text{语义原型对齐}} + \lambda_{\text{dis}} \cdot \underbrace{\mathcal{L}_{\text{解耦}}}_{\text{语义/风格独立}}}$$

**逐个解释**：
- **$\mathcal{L}_{\text{BCE}}$**：多标签分类损失，用完整的 [语义, 风格] → 分类头需要两种信息
- **$\mathcal{L}_{\text{proto}}^{\text{sem}}$**：只对**语义部分**计算与全局原型的距离 → 风格不参与全局对齐
- **$\mathcal{L}_{\text{解耦}}$**：HSIC 独立性约束 → 强制两部分编码不同的信息

### 7.5 解耦 + DP 的协同效应（论文核心卖点）

| | 不解耦 | 解耦 |
|---|---|---|
| 上传向量维度 | 256 | 192（75%） |
| 内容 | 语义 + 风格混合 | 纯语义 |
| 风格噪声 | 跨客户端随机波动 → 污染全局 | 零（风格不上传） |
| DP 噪声打在 | 混合信号上 | 纯语义信号上 |

**命题 1（聚合方差上界）**：Non-IID 程度为 $\eta$ 时，解耦后语义原型的聚合方差上界降至原始的 $\alpha^2$ 倍。

$\alpha = 0.75$ → 方差上界降至原来的 56%。

**命题 2（DP 信噪比）**：相同 $\varepsilon$ 预算下，信噪比提升：

$$\boxed{\frac{\text{SNR}_{\text{解耦}}}{\text{SNR}_{\text{原始}}} = \left(\frac{d_{\text{sem}}}{d_{\text{proto}}}\right)^{-1/2} = \left(\frac{192}{256}\right)^{-1/2} \approx 1.15}$$

**为什么**：DP 高斯噪声 $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I}_d)$ 在 $d$ 维空间中的期望 L2 范数 $\propto \sqrt{d}$。维度从 256 降到 192，噪声减少 $\sqrt{192/256} \approx 0.866$，等效信噪比提升 $1/0.866 \approx 1.15$ 倍。

**实际含义**：在 $\varepsilon = 1$ 这种极低隐私预算下，解耦带来的信噪比提升可能决定模型是可用的还是随机的。

---

## 8. 创新四：EMA 动量 —— 让全局原型别"一惊一乍"

### 8.1 问题

每轮只有部分客户端参与（`--frac` 默认 0.04，即 20 个客户端中每轮只采样 1 个）。如果某轮恰好抽到一个数据质量差的客户端，全局原型可能剧烈波动。

### 8.2 公式

$$\boxed{\mathbf{G}_t = \beta \cdot \mathbf{G}_{t-1} + (1 - \beta) \cdot \mathbf{G}_t^{\text{new}}}$$

$\beta = 0.9$ （`--proto_momentum`）→ 新原型只占 10%，旧原型占 90%。

**效果**：全局原型像加了惯性，不会因为一轮的坏数据就剧烈跳变。

**分布原型模式**：$\mu$ 和 $\log \sigma^2$ 各自独立做 EMA。

---

## 9. 创新五：λ 预热 + 温度缩放

### 9.1 λ 预热：别一开始就强制对齐

**问题**：第 1 轮的全局原型是随机初始化的（噪声），如果从第 1 轮就用很大的 $\lambda$ 强制本地原型去对齐，等于把本地特征往随机方向拉。

**方案**：$\lambda$ 从 0 开始线性增大：

$$\boxed{\lambda_{\text{eff}}(t) = \lambda \cdot \min\left(1, \frac{t + 1}{W}\right)}$$

$W = 50$（`--ld_warmup`）→ 前 50 轮 $\lambda$ 从 0 线性增至 $\lambda$。

### 9.2 温度缩放：控制预测的"自信程度"

推理时，用原型距离作为分类分数：

$$\boxed{\text{logit}_j(\mathbf{x}) = -\frac{\operatorname{dist}(\mathbf{p}_{\mathbf{x}}, \mathbf{G}[j])}{T}}$$

| $T$ | 效果 | 什么时候用 |
|-----|------|-----------|
| $T < 1$ | 距离被放大 → 更自信 | 类间差异大、分布原型方差小时 |
| $T = 1$ | 不做缩放 | 一般情况 |
| $T > 1$ | 距离被压缩 → 更平滑 | 原型噪声大、想更保守时 |

**分布原型的距离**（马氏距离形式）：

$$\operatorname{dist}(\mathbf{p}, \mathcal{N}(\boldsymbol{\mu}_g, \boldsymbol{\sigma}_g^2)) = \frac{1}{2} \sum_{j=1}^{d} \frac{(p_j - \mu_{g,j})^2}{\sigma_{g,j}^2}$$

**直觉**：方差大的维度，距离被分母缩小 → 不确定的维度对分类影响小。方差小的维度，距离被放大 → 确定的维度对分类影响大。

---

## 10. 差分隐私是怎么加进去的

### 10.1 核心机制

无论是原型还是权重，在上传前都经过两步：

**第一步：L2 裁剪**

$$\boxed{\mathbf{v}_{\text{裁剪}} = \mathbf{v} \cdot \min\left(1, \frac{C}{\|\mathbf{v}\|_2}\right)}$$

作用：限制每个客户端对全局模型的**最大影响**（敏感度控制）。$C$ 是裁剪阈值（`--dp_clip`，默认 1.0）。

**第二步：加高斯噪声**

$$\boxed{\mathbf{v}_{\text{加噪}} = \mathbf{v}_{\text{裁剪}} + \mathcal{N}(\mathbf{0}, \sigma^2 C^2 \mathbf{I})}$$

### 10.2 噪声强度 $\sigma$ 怎么定

通过**二分搜索**找到满足 $(\varepsilon, \delta)$ 目标的 $\sigma$：

$$\sigma^* = \underset{\sigma}{\operatorname{argmin}} \; |\varepsilon_{\text{实际}}(\sigma) - \varepsilon_{\text{目标}}|$$

$\varepsilon$ 越小 → $\sigma$ 越大 → 噪声越大 → 隐私越强但效用越低。

### 10.3 跨轮隐私预算追踪（RDP Moments Accountant）

**问题**：每轮都加噪声，隐私预算会累积。$T$ 轮后总隐私损失是多少？

**解法**：用 **Rényi 差分隐私（RDP）**做会计。

**单轮消耗**：

$$\varepsilon_{\text{RDP}}(\lambda) = \frac{\lambda}{2\sigma^2}$$

**$T$ 轮累积**：

$$\varepsilon_{\text{RDP}}^{\text{total}}(\lambda) = \sum_{i=1}^{T} \varepsilon_{\text{RDP}}^{(i)}(\lambda) = T \cdot \frac{\lambda}{2\sigma^2}$$

**RDP → $(\varepsilon, \delta)$-DP 转换**：

$$\boxed{\varepsilon = \min_{\lambda > 1} \left[ \varepsilon_{\text{RDP}}^{\text{total}}(\lambda) - \frac{\log \delta}{\lambda - 1} \right]}$$

$\lambda$ 在 1.1 到 10.9 之间搜索 99 个离散值，找到使 $\varepsilon$ 最小的 $\lambda$。

### 10.4 两种 DP 模式

| 模式 | 类 | 对什么加噪 | 适用算法 |
|------|-----|----------|---------|
| 原型模式 | `DPMechProto` | $(\mu \| \log \sigma^2)$ 拼接向量 | FedProto, DPP-FL |
| 权重模式 | `DPMechWeight` | $\mathbf{w}_{\text{本地}} - \mathbf{w}_{\text{全局}}$ 差值 | FedAvg, FedProx, FedBN, SCAFFOLD |

---

## 11. 六个算法一张表对比

| 算法 | 共享什么 | 一次传多少 | Non-IID 怎么处理 | DP 对什么加噪 |
|------|---------|-----------|-----------------|-------------|
| **FedAvg** (2017) | 模型权重 | ~23M 参数 | 无 | 权重差 |
| **FedProx** (2020) | 模型权重 | ~23M 参数 | 近端约束（不让本地跑太远） | 权重差 |
| **FedBN** (2021) | 权重（跳过 BN 层） | ~23M 参数 | 本地 BN 统计量 | 权重差 |
| **SCAFFOLD** (2020) | 权重 + 控制变量 | ~46M 参数 | 梯度修正 | 权重差 |
| **FedProto** (2022) | 点原型 256d×14 | ~3.6K 浮点 | 原型正则化 | 原型向量 |
| **DPP-FL** | 高斯原型 $\mathcal{N}(\mu,\sigma^2)$ | ~7.2K 浮点 | 分布原型 + 贝叶斯 + 解耦 | 原型向量 |

**通信量比较**：
- 权重共享方法 ≈ 23M × 4 字节 = **92 MB/轮/客户端**
- DPP-FL ≈ 256 × 14 × 2 × 4 字节 = **28 KB/轮/客户端**
- 差距：约 **3000 倍**

---

## 12. 完整训练流程（伪代码）

```
初始化：
  为每个客户端创建预训练 ResNet-50 模型
  全局原型 G = 空字典  （第一轮没有全局原型，不加原型损失）
  EMA 原型 G_ema = 空字典

循环 第 t = 0 到 T-1 轮：
    采样 m = max(1, frac × K) 个客户端参与本轮

    λ_有效 = λ × min(1, (t+1) / warmup_轮数)   ← 前 W 轮 λ 线性增长

    对每个参与客户端 k：
        1. 用深拷贝的上一轮模型 + 全局原型 G 在本地数据上训练：

           对每个 batch：
              前向传播 → 得到 logits + 原型向量
              损失 = BCE(分类) + λ_有效 × 原型损失 + λ_dis × 解耦损失

              原型损失计算（多标签）：
                  对每张图的所有正标签，计算本地原型和对应全局原型的距离
                  （分布原型用 KL/Wasserstein，点原型用 MSE）

              解耦损失（如启用）：
                  HSIC = 交叉协方差矩阵的 F 范数平方 + 方差正则

              反向传播 → 更新模型参数

              按标签收集本地原型（解耦模式只收集语义部分）

        2. 客户端内原型聚合：
              同一客户端同标签多个原型 → agg_func()
              点原型：直接平均
              分布原型：均值平均 + 总方差定律合并方差

        3. (可选) DP 加噪：L2 裁剪 + 高斯噪声

        4. 保存训练后的模型权重

    服务器跨客户端原型聚合：
        点原型：直接对每个标签的所有客户端原型取平均
        分布原型：贝叶斯融合（精度加权平均）

    EMA 平滑：
        如果 G_ema 非空：
            对每个标签：G[标签] = β × G_ema[标签] + (1-β) × G_新[标签]
        G_ema = detach(G)

    (可选) 打印当前隐私预算消耗

最终测试（每个客户端独立评估）：
    测试A（不用全局原型）：
        sigmoid(模型logits) > 0.5 → 14维二值预测
    测试B（用全局原型）：
        提取测试样本原型 → 计算到每个全局原型的距离 → 负距离/温度 → sigmoid → 预测
```

---

## 13. 公式速查卡

### 损失函数

| 名称 | 公式 | 用途 |
|------|------|------|
| 多标签 BCE | $-\sum_c [y_c\log\sigma(f_c) + (1-y_c)\log(1-\sigma(f_c))]$ | 分类 |
| 点原型损失 | $\|\mathbf{p} - \mathbf{G}[c]\|^2$ | 知识迁移 |
| KL 散度 | $\frac{1}{2}[\log\frac{\sigma_p^2}{\sigma_q^2} + \frac{\sigma_q^2+(\mu_q-\mu_p)^2}{\sigma_p^2} - 1]$ | 分布距离 |
| W2 距离 | $\|\mu_q-\mu_p\|^2 + \|\sigma_q-\sigma_p\|_F^2$ | 分布距离 |
| 解耦损失 | $\|\operatorname{Cov}(\mathbf{z}_{\text{sem}}, \mathbf{z}_{\text{style}})\|_F^2 + 0.1 \cdot \text{var\_reg}$ | 独立性 |

### 聚合公式

| 名称 | 公式 |
|------|------|
| 客户端内（点原型） | $\mathbf{P} = \frac{1}{M}\sum \mathbf{p}_i$ |
| 客户端内（分布） | $\mu = \text{mean}(\mu_i),\ \sigma^2 = \text{mean}(\sigma_i^2) + \text{Var}(\mu_i)$ |
| 跨客户端（点原型） | $\mathbf{G} = \frac{1}{K}\sum \mathbf{P}_k$ |
| 贝叶斯融合 | $\mu^* = \frac{\sum \mu_k/\sigma_k^2}{\sum 1/\sigma_k^2},\ {\sigma^2}^* = \frac{1}{\sum 1/\sigma_k^2}$ |
| EMA 动量 | $\mathbf{G}_t = \beta\mathbf{G}_{t-1} + (1-\beta)\mathbf{G}_t^{\text{new}}$ |

### DP 公式

| 名称 | 公式 |
|------|------|
| L2 裁剪 | $\mathbf{v} \cdot \min(1, C/\|\mathbf{v}\|_2)$ |
| 高斯加噪 | $\mathbf{v} + \mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$ |
| 单轮 RDP | $\varepsilon_{\text{RDP}}(\lambda) = \lambda/(2\sigma^2)$ |
| RDP→DP 转换 | $\varepsilon = \min_\lambda [T\cdot\frac{\lambda}{2\sigma^2} - \frac{\log\delta}{\lambda-1}]$ |

### 推理公式

| 名称 | 公式 |
|------|------|
| 原型距离（点） | $\|\mathbf{p} - \mathbf{G}[c]\|^2$ |
| 原型距离（分布） | $\frac{1}{2}\sum_j (p_j - \mu_{g,j})^2 / \sigma_{g,j}^2$ |
| 温度缩放 | $\text{logit} = -\text{dist} / T$ |
| 预测 | $\sigma(\text{logit}) > 0.5$ |

---

## 14. 常见问题 FAQ

### Q1: 为什么用 KL 散度而不用 JS 散度？

KL 散度是**不对称**的：$\text{KL}(q\|p) \neq \text{KL}(p\|q)$。我们计算的是 $\text{KL}(\text{本地} \| \text{全局})$。

- 当本地方差大（不确定）时，KL 自动变小 → 不强制对齐
- 当全局方差小（确定）时，KL 自动变大 → 必须认真对齐

这种**自适应加权**是 JS 散度做不到的。

### Q2: 解耦为什么是 75%/25% 的维度比？

`sem_ratio=0.75` 是一个经验值。理论上：
- **太高**（如 0.9）→ 风格维度太少，可能不能有效编码风格信息 → 信息泄露到语义
- **太低**（如 0.5）→ 语义维度太少，可能不能有效编码疾病特征 → 分类性能下降

75% 在实验中取得了分类性能和隐私效用的最佳平衡。这可以作为一个超参数在实验中调优。

### Q3: 分布原型会不会让通信量翻倍？

分布原型需要传 $\mu$ 和 $\log\sigma^2$ 两个向量，确实是点原型的 2 倍。但基数太小了：

- 点原型：256 × 14 = 3,584 个浮点数
- 分布原型：256 × 14 × 2 = 7,168 个浮点数

相比模型权重的 2300 万参数，7 千个浮点数仍然可以忽略不计。

### Q4: 联邦学习和普通分布式训练有什么区别？

| | 普通分布式训练 | 联邦学习 |
|---|---|---|
| 数据分布 | IID（打乱均匀分配） | Non-IID（天然分布不均） |
| 通信 | 每步都同步梯度 | 多步本地训练后才通信 |
| 隐私 | 不保护 | 核心设计目标 |
| 客户端 | 同质服务器 | 异质设备/机构 |

### Q5: 这个框架能用于其他任务吗？

理论上可以。只要是分类任务 + Non-IID 数据分布 + 需要隐私保护的场景都可以用。关键是选择合适的主干网络和合理的原型维度。

### Q6: 为什么分类损失中的"补全编码损失"不存在？

在多标签分类中，每个标签是独立的二分类，没有"互斥"关系。所以用的是 sigmoid + BCE（每个标签独立），而不是 softmax + CE（标签互斥）。

### Q7: 如果某个全局原型只被 1 个客户端更新过怎么办？

在 `bayesian_fusion()` 中（`aggregation.py` 第 50-51 行），如果某个标签只有 1 个客户端贡献了原型，不做贝叶斯融合，直接使用该客户端的原型。这避免了精度为零的退化情况。

### Q8: "No Finding" 样本怎么处理？

"No Finding"（14 个标签全为 0 的健康胸片）均匀分配给所有客户端（每个客户端至少 10 张）。它们作为负样本参与训练，确保每个客户端都见过"正常"的样子。

---

## 附录：关键参数速查

### 算法选择
| 参数 | 默认 | 说明 |
|------|------|------|
| `--alg` | `dppfl` | 算法：`fedavg/fedprox/fedbn/scaffold/fedproto/dppfl` |

### DPP-FL 核心参数
| 参数 | 默认 | 作用 |
|------|------|------|
| `--use_distributional` | False | 用高斯分布原型（否则点原型） |
| `--dist_type` | `kl` | 分布距离：`kl/wasserstein/mse` |
| `--use_disentangle` | False | 开启语义/风格解耦 |
| `--sem_ratio` | 0.75 | 语义维度占比 |
| `--dis_lambda` | 0.05 | 解耦损失权重 |
| `--proto_momentum` | 0.9 | EMA 动量系数 |
| `--ld` | 1.0 | 原型损失权重 λ |
| `--ld_warmup` | 50 | λ 预热轮数 |
| `--temperature` | 1.0 | 推理温度系数 |

### 差分隐私
| 参数 | 默认 | 作用 |
|------|------|------|
| `--use_dp` | False | 启用 DP |
| `--dp_epsilon` | 8.0 | 目标 ε |
| `--dp_delta` | 1e-5 | 目标 δ |
| `--dp_clip` | 1.0 | L2 裁剪阈值 |

### 联邦学习基础
| 参数 | 默认 | 作用 |
|------|------|------|
| `--num_users` | 20 | 客户端数 |
| `--frac` | 0.04 | 每轮参与比例 |
| `--rounds` | 100 | 全局通信轮数 |
| `--train_ep` | 1 | 每轮本地 epoch |
| `--ways` | 3 | 每客户端平均类别数 |
| `--shots` | 100 | 每类平均样本数 |
| `--proto_dim` | 256 | 原型向量维度 |

---

## 15. 模型架构

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

Bottleneck expansion=4，总参数量约 23M。冻结浅层 (layer1-2)、微调深层 (layer3-4) 的策略兼顾了预训练知识保留和医疗影像域适应。

---

## 16. 系统架构

```
DPP-FL/
├── exps/
│   └── federated_main.py          ← 主入口
├── lib/
│   ├── options.py                 ← 参数解析
│   ├── utils.py                   ← 数据加载、权重聚合、原型聚合
│   ├── update.py                  ← 本地训练、测试、多标签原型提取
│   ├── sampling.py                ← IID/Non-IID 数据划分
│   ├── chestxray.py               ← ChestX-ray14 数据集类
│   ├── visualize.py               ← t-SNE 原型可视化
│   ├── models/
│   │   └── resnet.py              ← DPPFLResNet / ResNet50 backbone
│   ├── dist_proto/                ← 分布原型子模块
│   │   ├── proto_head.py          ← ProbabilisticProtoHead (μ, logvar)
│   │   ├── losses.py              ← KL, Wasserstein, MSE 损失
│   │   ├── aggregation.py         ← 贝叶斯融合
│   │   └── disentangle.py         ← 解耦原型头 + HSIC 损失
│   └── dp/                        ← 差分隐私子模块
│       └── mechanisms.py          ← DPMechProto, MomentsAccountant
├── figures/                       ← 架构图
├── paper/                         ← 论文与理论文档
├── scripts/
│   └── run.sh                     ← 启动脚本
└── requirements.txt
```

### 主训练循环调用关系

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
  └─ FedProto_taskheter() / DPPFL_taskheter()  → 主训练循环
      │
      For each round:
        ├─ 采样 m = frac * K 个客户端
        For each selected client:
          ├─ LocalUpdate.update_weights_het()
          │   ├─ model → (logits, protos) 或 (logits, mu, logvar)
          │   ├─ loss = BCE + λ * proto_loss
          │   └─ agg_func() → 多标签原型取平均/合并方差
          │
          ├─ DPMechProto.clip_and_noise() → (可选) DP 扰动
          │
        ├─ proto_aggregation() → 跨客户端原型聚合
        │   └─ bayesian_fusion_single_label() → (可选) 分布原型贝叶斯融合
        │
        └─ 将本轮训练权重写回 local_model_list

      test_inference_new_het_lt_DPPFL()
        ├─ 不使用全局原型: sigmoid(logits) > 0.5  (per-label)
        └─ 使用全局原型: 负原型距离 → sigmoid → 二值预测 (per-label)
```

---

## 17. 实现细节与注意事项

### 17.1 原型格式统一

全局原型字典 `global_protos` 的 value 格式：
- **点原型**: 单一张量 `tensor(shape=[proto_dim])`
- **分布原型**: 二元组 `(mu: tensor, logvar: tensor)`，各自 shape=[proto_dim]

### 17.2 客户端采样

`--frac` 参数控制每轮参与训练的客户端比例。每轮随机采样 `m = max(1, int(frac * K))` 个客户端。未参与轮的客户端保留上一轮模型，在后续轮次可被选中继续训练。

### 17.3 多标签原型损失计算

原型正则化损失 `L_proto` 的计算方式（以点原型为例）：

```
对 batch 中每张图 i:
  对每个正标签 j (labels[i, j] == 1):
    loss2 += MSE(proto_i, global_protos[j])
loss2 = loss2 / count  # 除以所有正标签总数
```

即平均到每个正标签上，而非每张图。这意味着有多个疾病的 X 光片对原型损失的贡献更大。

### 17.4 "No Finding" 负样本处理

"No Finding"（标签全为 0）的样本在 Non-IID 划分时均匀分配给所有客户端（每个客户端至少 10 张），作为负样本参与训练，确保每个客户端都能学到"正常"的表示。

### 17.5 分布原型的数值稳定性

- `logvar` 被 clamp 到 [-10, 10] 范围（`ProbabilisticProtoHead`）
- `var = exp(logvar)`，对应方差范围约 [4.5e-5, 2.2e4]
- `agg_func` 中 `logvar_avg = log(avg_var + 1e-8)` 防止 log(0)
- 推理时 `g_var + 1e-8` 防止除零

---

## 附录 B：数学符号汇总

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
