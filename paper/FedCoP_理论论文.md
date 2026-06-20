# FedCoP: Federated Co-occurrence-aware Prototypes for Multi-Label Medical Image Classification

> **Authors**: [Your Name]
>
> **Affiliation**: [Your Institution]
>
> **Code**: https://github.com/[repo]/FedCoP

---

## Abstract

Federated learning (FL) enables privacy-preserving collaborative training across hospitals, and prototype-based FL (e.g., FedProto) further reduces communication and protects privacy by sharing only class prototypes. However, existing prototype-FL methods model each class with an **independent** prototype and decode labels with **independent per-class sigmoids**, implicitly assuming the $C$ pathologies are conditionally independent given the features. This assumption is violated in multi-label medical imaging, where diseases exhibit strong **co-occurrence** (comorbidity), e.g., pleural effusion and atelectasis. Moreover, under non-IID class partitioning — the realistic FL setting where each hospital sees only a few of the $C$ classes — **no single client can observe the global co-occurrence structure**; it is fundamentally a non-local statistic recoverable only through federation.

**【中文】** 联邦学习(FL)支持跨医院的隐私保护协作训练,而基于原型的联邦学习(如 FedProto)通过只共享类别原型,进一步降低了通信量并保护了隐私。然而,现有的原型联邦方法用**相互独立**的原型建模每个类别,并用**逐类独立的 sigmoid** 解码标签,隐含假设 $C$ 种病理在给定特征的条件下相互独立。这一假设在多标签医学影像中并不成立——疾病之间存在强烈的**共现**(共病)关系,例如胸腔积液与肺不张。此外,在非独立同分布(non-IID)的类别划分下(即真实的联邦场景:每家医院只看到 $C$ 类中的少数几类),**没有任何一个客户端能观测到全局共现结构**;它在本质上是一个非局部统计量,只能通过联邦聚合才能恢复。

We propose **FedCoP** (Federated Co-occurrence-aware Prototypes), which augments distributional prototypes with a federatedly-estimated **co-occurrence correlation matrix** $\hat R \in \mathbb{R}^{C\times C}$ and uses it on both sides of the pipeline: (i) a **co-occurrence structure alignment loss** $L_{co}$ that constrains the inter-class prototype geometry (cosine Gram) to match $\hat R$, replacing ad-hoc contrastive/adversarial regularizers; and (ii) a **correlation-aware mean-field decoder** that propagates evidence across co-occurring classes at inference, yielding a strict improvement over independent sigmoids under correlated labels. $\hat R$ is estimated from privacy-safe label sufficient statistics $(\mathbf{m}_k, \mathbf{M}_k, n_k)$ aggregated by count-weighted fusion. We prove (1) the independent decoder is Bayes-optimal only under conditional label independence, with regret $\Omega(\|\hat R - I\|_F^2)$ that the mean-field decoder reduces to a variational gap; and (2) the federated estimator $\hat R$ is an unbiased, $\ell_\infty$-consistent (matrix-Bernstein) estimator of the population co-occurrence, and is **unrecoverable by any single client** that observes $\le$ ways $< C$ classes — formalizing, beyond the single-positive-label observation of FedALC [An et al., 2024], why the structure must be federated under general non-IID class partitioning. On NIH ChestX-ray14 under non-IID federated splits, FedCoP outperforms FedAvg, FedProx, FedProto, FedGMKD, FedBCS and FedSeProto on macro-AUROC and macro-F1, with the largest gains on rare and co-occurring pathologies. Ablations isolate the contribution of the federated structure ($\hat R$), the training-side loss ($L_{co}$), and the inference-side decoder.

**【中文】** 我们提出 **FedCoP**(联邦共现感知原型),它在分布原型之上增加一个联邦估计的**共现相关矩阵** $\hat R \in \mathbb{R}^{C\times C}$,并将其用于流程的两端:(i) 一个**共现结构对齐损失** $L_{co}$,约束类间原型几何(余弦 Gram 矩阵)去匹配 $\hat R$,以替代临时设计的对比/对抗正则项;(ii) 一个**相关感知 mean-field 解码器**,在推理时跨共现类传播证据,在标签相关时严格优于独立 sigmoid。$\hat R$ 由隐私安全的标签充分统计量 $(\mathbf{m}_k, \mathbf{M}_k, n_k)$ 经计数加权融合估计得到。我们证明:(1) 独立解码器仅在标签条件独立时才是 Bayes 最优的,其 regret 为 $\Omega(\|\hat R - I\|_F^2)$,而 mean-field 解码器将其降为一个变分间隙;(2) 联邦估计器 $\hat R$ 是总体共现的无偏、$\ell_\infty$ 一致(基于 matrix-Bernstein)估计量,并且**任何只观测到 ways $< C$ 类的单个客户端都无法恢复它**——在 FedALC[An 等,2024]单正标签观察的基础上,对一般非独立同分布类别划分形式化了该结构为何必须联邦。在 NIH ChestX-ray14 的非独立同分布联邦划分下,FedCoP 在 macro-AUROC 和 macro-F1 上均优于 FedAvg、FedProx、FedProto、FedGMKD、FedBCS 和 FedSeProto,在罕见病和共现病上的提升最大。消融实验分别隔离了联邦结构($\hat R$)、训练侧损失($L_{co}$)和推理侧解码器的贡献。

---

## 1. Introduction

Chest radiography is the most common imaging exam in medicine, and reading it is naturally a multi-label problem: a single film frequently carries several findings at once. This is not an artifact of labeling — it is physiology. Pleural effusion and atelectasis co-occur because fluid compresses lung tissue; consolidation and infiltration travel together with pneumonia. A model that reads a chest film as fourteen independent yes/no questions is, clinically, reading it wrong. And yet, when these models are trained across hospitals under privacy constraints — the federated setting that medicine actually needs — the methods available to us treat the fourteen pathologies as independent. This paper is about fixing that, and about why the fix has to be federated.

**【中文】** ## 1. 引言

胸部 X 光是医学中最常见的影像检查,而读片天然是个多标签问题:一张片子常常同时带有多个发现。这并非标注的产物,而是生理本身。胸腔积液与肺不张共现,是因为积液压迫了肺组织;实变与浸润随肺炎一同出现。把读胸片当成十四个相互独立的"是/否"问题,在临床上就是读错了。然而,当这些模型在隐私约束下跨医院训练——即医学真正需要的联邦设定——我们手头的方法却把这十四种病理当作相互独立的来处理。本文就是要修正这一点,并解释为何这个修正必须是联邦的。

### 1.1 Prototype-based FL and its independence assumption

Federated learning lets hospitals train a shared model without exchanging data. **FedProto** [Tan et al., AAAI 2022] replaced weight sharing with **prototype sharing**: each client uploads a per-class mean feature vector (the prototype) and the server averages them; clients regularize local features toward the global prototypes. This is communication-efficient, architecture-agnostic, and privacy-friendly — and it sidesteps the awkward fact that raw model weights can, in principle, leak training data. For these reasons prototype sharing has become the dominant paradigm for privacy-sensitive FL, and a line of follow-ups (FedGMKD, FedBCS, FedSeProto) has strengthened the prototype *representation* (Gaussian mixture heads, disentanglement, distributional heads) and the *aggregation* (quality-weighted, Bayesian fusion).

**【中文】** ### 1.1 原型联邦学习及其独立性假设

联邦学习让各医院在不交换数据的前提下共同训练一个共享模型。**FedProto**[Tan 等,AAAI 2022]用**原型共享**取代了权重共享:每个客户端上传每类的特征均值向量(即原型),服务器对其求平均,客户端再把自己的局部特征拉向全局原型。这种方法通信高效、与模型架构无关、利于隐私——并且避开了原始模型权重原则上可能泄露训练数据这一棘手事实。正因如此,原型共享已成为隐私敏感联邦学习的主流范式,一系列后续工作(FedGMKD、FedBCS、FedSeProto)强化了原型的*表示*(高斯混合头、解耦、分布头)与*聚合*(质量加权、贝叶斯融合)。

**However, every method in this line retains two independence assumptions that are especially harmful in multi-label medical imaging:**

1. **Storage/alignment independence.** The $C$ class prototypes are stored as a *set* $\{\mathbf{p}_c\}_{c=1}^C$ and the alignment loss treats each class in isolation — a sample with co-occurring {Effusion, Infiltration} produces two unrelated prototype-alignment gradients.
2. **Decoding independence.** Inference computes $p(y_c{=}1\mid \mathbf{x}) = \sigma(-d_c(\mathbf{x})/T)$ independently per class.

**【中文】** **然而,这一脉络里的每个方法都保留了两个独立性假设,而这在多标签医学影像中尤其有害:**

1. **存储/对齐独立。** $C$ 个类别原型被当作一个*集合* $\{\mathbf{p}_c\}_{c=1}^C$ 存储,对齐损失逐类独立计算——一张同时共现 {积液, 浸润} 的样本会产生两个互不相关的原型对齐梯度。
2. **解码独立。** 推理时逐类独立计算 $p(y_c{=}1\mid \mathbf{x}) = \sigma(-d_c(\mathbf{x})/T)$。

These two assumptions are not unrelated; they are two faces of one assumption — that the $C$ labels are **conditionally independent given the features**. The first says the *geometry* of the prototypes need not reflect inter-class relations; the second says the *decoding* need not propagate information across classes. In a domain where labels are strongly correlated (comorbidity is the clinical norm, not the exception), both faces are wrong, and the decoding face is provably suboptimal — we show in §5 that the independent sigmoid decoder is Bayes-optimal *only* under conditional independence, and pays a regret that grows with the residual label correlation. Put bluntly: the field has built increasingly elaborate prototypes on top of a decoder that throws away the very structure that makes multi-label medical imaging multi-label.

**【中文】** 这两个假设并非互不相干,而是同一个假设的两面——即 $C$ 个标签**在给定特征的条件下相互独立**。第一面说原型的*几何*不必反映类间关系;第二面说*解码*不必跨类传播信息。在标签强相关的领域(共病是临床常态、而非例外),这两面都是错的,而解码这一面可证明是次优的——我们在 §5 证明,独立 sigmoid 解码器*仅*在条件独立时才是 Bayes 最优,其 regret 随残差标签相关增长。直白地说:整个领域在一个丢弃了"使多标签医学影像之所以是多标签"的那套结构的解码器之上,搭建了越来越精巧的原型。

### 1.2 The federated co-occurrence opportunity

The obvious remedy — model label correlation — runs immediately into a wall in the federated setting, and the wall is the point of this paper. The co-occurrence structure is **non-local**. Under the realistic non-IID class partitioning of federated medical data, each hospital sees only a few of the $C$ pathologies ($\text{ways} \ll C$, e.g., 3 of 14) — a specialist center for effusion does not also see enough hernia cases to know how the two relate. Concretely, a client's co-occurrence count matrix $\mathbf{M}_k$ has zero rows and columns for every class it never sees, so the entries for any pair it does not jointly observe are *exactly* zero — not "small," zero. No amount of local data distinguishes "these never co-occur" from "I never see these."

This is more than an inconvenience; it is a structural fact that dictates the solution. The global $C\times C$ co-occurrence structure is recoverable *only* by federated aggregation of per-client sufficient statistics, across clients whose class supports jointly cover all $C$ classes. Co-occurrence is therefore not a multi-label trick that happens to be run inside a federated system, but a **structurally federated** quantity — invisible to any participant, visible only to the federation as a whole. The same necessity was observed by FedALC [An et al., 2024] in the single-positive-label setting; our contribution is not the observation itself but what follows from it — a general sufficient-statistic estimator with de-biasing and concentration guarantees (Prop. 2), and the use of the estimate on both prototype geometry and decoding. This is why we treat the co-occurrence matrix as a first-class federated object rather than a post-hoc correction.

**【中文】** ### 1.2 联邦共现的机遇

显而易见的补救——建模标签相关——在联邦设定下立刻撞上一堵墙,而这堵墙正是本文的落点。共现结构是**非局部的**。在联邦医学数据现实的非独立同分布类别划分下,每家医院只见到 $C$ 种病理中的少数几种($\text{ways} \ll C$,如 14 之 3)——一家专治积液的中心不会也见到足够多的疝气病例来了解二者的关系。具体而言,一个客户端的共现计数矩阵 $\mathbf{M}_k$ 对它从未见过的类,对应的行列全为零,故任何它未联合观测的类对,其元素都*精确为零*——不是"很小",是零。再多的本地数据也无法区分"二者从不共现"与"我从未见过二者"。

这不止是不便,而是一个决定解法的事实。全局 $C\times C$ 共现结构*只能*通过对各客户端充分统计量进行联邦聚合来恢复,且需要那些类支持联合覆盖全部 $C$ 类的客户端。因此,共现不是碰巧在联邦系统里运行的多标签技巧,而是一个**结构上必须联邦**的量——对任何参与者不可见,只对作为整体的联邦可见。FedALC[An 等,2024]在单正标签设定下也观察到了同样的必要性;我们的贡献不在于该观察本身,而在于其后——一个带去偏与集中性保证的一般充分统计量估计器(命题 2),以及把该估计同时用于原型几何与解码。这也是我们把共现矩阵当作一等联邦对象、而非事后修正的原因。

### 1.3 Contributions

1. **Federated co-occurrence structure.** A privacy-safe, count-weighted estimator of the global pathology co-occurrence correlation $\hat R$ from label sufficient statistics $(\mathbf{m}_k,\mathbf{M}_k,n_k)$, with phi-correlation de-biasing, shrinkage, and EMA smoothing. The estimator is unbiased under reweighted participation and concentrates at rate $\tilde O(\sqrt{\log C/(K n_{\min})})$.
2. **Correlation-aware prototypes.** A training-side structure loss $L_{co}$ aligning the prototype cosine Gram to $\hat R$, and an inference-side mean-field decoder using $\hat R$ — together replacing the heuristic contrastive/adversarial regularizers used in prior prototype-FL with a single, label-statistics-driven mechanism that acts on both sides of the pipeline.
3. **Theory.** A decoder-regret bound showing the mean-field decoder is never worse than independent sigmoids and strictly better when $\hat R$ tracks the residual label correlation; and a federated-estimation theorem (unbiasedness, matrix-Bernstein concentration) that *formalizes*, for the general non-IID class-partition setting, the single-client non-recoverability of the structure — an observation previously made by FedALC [An et al., 2024] in the single-positive-label setting.
4. **Minimal method + stronger evaluation.** A deliberately minimal 4-loss objective — omitting per-class temperature, adversarial, contrastive, and calibration losses (analyzed as redundant or dead in §6) — the FedProx baseline, and full multi-label metrics (macro/micro AUROC, F1, Hamming, subset accuracy), with three ablations ($\hat R{=}I$, local-only $\hat R$, no-$L_{co}$) that separately isolate the federated structure, the training-side loss, and the inference-side decoder.

**【中文】** ### 1.3 贡献

1. **联邦共现结构。** 从标签充分统计量 $(\mathbf{m}_k,\mathbf{M}_k,n_k)$ 出发,提出一个隐私安全、计数加权的全局病理共现相关 $\hat R$ 估计器,辅以 phi 相关去偏、收缩与 EMA 平滑。该估计器在重加权参与下无偏,以速率 $\tilde O(\sqrt{\log C/(K n_{\min})})$ 集中。
2. **相关感知原型。** 一个训练侧的结构损失 $L_{co}$(将原型余弦 Gram 对齐到 $\hat R$),以及一个推理侧的使用 $\hat R$ 的 mean-field 解码器——二者合在一起,用一个由标签统计驱动、作用于流程两端的单一机制,替代以往原型联邦中启发式的对比/对抗正则。
3. **理论。** 给出解码器 regret 界,证明 mean-field 解码器不劣于独立 sigmoid,当 $\hat R$ 追踪到残差标签相关时严格更优;并给出一个联邦估计定理(无偏性、matrix-Bernstein 集中),对一般非独立同分布类别划分*形式化*了该结构的单客户端不可恢复性——此观察此前由 FedALC[An 等,2024]在单正标签设定下提出。
4. **极简方法 + 更强的评测。** 一个刻意精简的 4 项损失目标——略去逐类温度、对抗、对比、校准损失(§6 分析为冗余或死代码),加入 FedProx 基线与完整的多标签指标(macro/micro AUROC、F1、Hamming、subset accuracy),并提供三个消融($\hat R{=}I$、仅本地 $\hat R$、无 $L_{co}$),分别隔离联邦结构、训练侧损失与推理侧解码器。

---

## 2. Related Work

All methods below use an ImageNet-pretrained ResNet-50 backbone for fair comparison.

**【中文】** 为公平比较,以下所有方法均采用 ImageNet 预训练的 ResNet-50 作为骨干网络。

- **FedAvg** [McMahan et al., 2017]: local SGD + equal-weight parameter averaging; the weight-sharing baseline.
- **FedProx** [Li et al., MLSys 2020]: adds a proximal term $\frac{\mu}{2}\|\mathbf{w}-\mathbf{w}^g\|^2$ to curb client drift under heterogeneity; a standard strong baseline.
- **FedProto** [Tan et al., AAAI 2022]: shares point prototypes instead of weights; our direct baseline.
- **FedGMKD** [NeurIPS 2024]: post-hoc GMM prototypes (EM on detached features) + discrepancy-aware aggregation.
- **FedBCS** [AAAI 2026]: frequency-domain style recalibration + domain-invariant prototypes.
- **FedSeProto** [ECAI 2024]: hard semantic/domain feature split + HSIC, sharing only semantic prototypes.
- **FedALC** [An et al., 2024]: the closest prior work — federated multi-label classification under single-positive labels; estimates label correlations on the server by aggregating hashed per-instance label sets, and observes that co-occurrence is recoverable only through cross-client aggregation.

**【中文】**
- **FedAvg**[McMahan 等,2017]:局部 SGD + 等权参数平均;权重共享基线。
- **FedProx**[Li 等,MLSys 2020]:加入近端项 $\frac{\mu}{2}\|\mathbf{w}-\mathbf{w}^g\|^2$ 以抑制异构下的客户端漂移;标准强基线。
- **FedProto**[Tan 等,AAAI 2022]:共享点原型而非权重;我们的直接基线。
- **FedGMKD**[NeurIPS 2024]:事后 GMM 原型(在 detach 的特征上做 EM)+ 差异感知聚合。
- **FedBCS**[AAAI 2026]:频域风格重校准 + 域不变原型。
- **FedSeProto**[ECAI 2024]:硬性的语义/域特征切分 + HSIC,只共享语义原型。
- **FedALC**[An 等,2024]:最接近的先验工作——单正标签下的联邦多标签分类;通过在服务器聚合逐实例的哈希标签集来估计标签相关,并指出共现只能靠跨客户端聚合才能恢复。

**Difference of FedCoP.** FedALC [An et al., 2024] is the closest prior work and the one we must position against most carefully. It is federated multi-label and, like us, estimates label correlations on the server from cross-client label information, observing in the single-positive-label setting that co-occurrence is only recoverable by aggregation. FedCoP differs in three concrete ways: (i) it estimates a **de-biased phi-correlation** matrix from count sufficient statistics $(\mathbf{m}_k,\mathbf{M}_k,n_k)$ — marginal-frequency-debiased and positive-definite via shrinkage — rather than per-instance hashed label sets; (ii) it uses $\hat R$ on **both** sides of the pipeline — to shape prototype *geometry* via $L_{co}$ and to drive a *mean-field decoder* — whereas FedALC uses correlations only in class-embedding training and has no structured decoder; (iii) it **formalizes** single-client non-recoverability for the general non-IID class-partition setting (Prop. 2c, rank $\le\text{ways}$ and exact-zero absent-pair entries) and proves concentration of the estimator, neither of which FedALC provides. The other baselines above treat the $C$ classes independently in both geometry and decoding. Centralized multi-label label-correlation methods (classifier chains, label embeddings) exist but cannot recover the structure under non-IID class partitioning.

**【中文】** **FedCoP 的不同之处。** FedALC[An 等,2024]是最接近的先验工作,也是我们必须最小心地与之对标的对象。它是联邦多标签,且和我们一样在服务器端从跨客户端标签信息估计标签相关,并在单正标签设定下指出共现只能靠聚合恢复。FedCoP 在三处具体不同:(i) 它从计数充分统计量 $(\mathbf{m}_k,\mathbf{M}_k,n_k)$ 估计**去偏的 phi 相关**矩阵——经收缩去边际频率偏差且正定——而非逐实例哈希标签集;(ii) 它把 $\hat R$ 用于流程**两端**——经 $L_{co}$ 塑造原型*几何*、并驱动 *mean-field 解码器*——而 FedALC 仅在类嵌入训练中使用相关、无结构化解码器;(iii) 它对一般非独立同分布类别划分**形式化**了单客户端不可恢复性(命题 2c,秩 $\le\text{ways}$ 且缺失类对元素精确为零),并证明了估计量的集中性,这两者 FedALC 都未提供。上述其余基线在几何与解码两方面都把 $C$ 类独立处理。集中式多标签标签相关方法(分类器链、标签嵌入)虽已存在,但在非独立同分布类别划分下无法恢复该结构。

---

## 3. Method: FedCoP

FedCoP has one moving part that is new — the federated co-occurrence matrix $\hat R$ — and several that are inherited from prior prototype-FL and kept because they work. We found it useful while building the system to be explicit about *which* structure lives where: the per-class geometry (means and variances of each prototype) stays in the distributional prototypes of §3.1; the *cross-class* geometry (how classes relate to each other) lives entirely in $\hat R$. Mixing the two — for instance by giving each class a full $D\times D$ covariance — is the obvious alternative, and we explain in §3.1 why we rejected it. The two halves of the pipeline then use $\hat R$ in genuinely different ways: training aligns prototype *directions* to it (§3.3), inference propagates *evidence* along it (§3.4). The two are not redundant — they act on different objects — and the ablation in §7.2 separates their contributions.

**【中文】** ## 3. 方法:FedCoP

FedCoP 只有一个新部件——联邦共现矩阵 $\hat R$——其余部分继承自以往的原型联邦学习,因为它们确实有效。在搭建这套系统时我们发现,把"*哪种*结构放在哪里"讲清楚很有用:每类的几何(各原型的均值与方差)留在 §3.1 的分布原型里;*跨类*几何(各类彼此如何关联)则完全交给 $\hat R$。把两者混在一起——比如给每类一个完整的 $D\times D$ 协方差——是显而易见的替代方案,我们在 §3.1 说明了为何放弃它。随后流程的两端以真正不同的方式使用 $\hat R$:训练侧把原型*方向*对齐到它(§3.3),推理侧沿它传播*证据*(§3.4)。两者并不冗余——作用对象不同——§7.2 的消融把它们的贡献分开了。

### 3.1 Distributional prototypes (retained, simplified)

Each class $c$ is modeled as a diagonal Gaussian $\mathcal{N}(\boldsymbol\mu_c, \mathrm{diag}(\boldsymbol\sigma_c^2))$ in a $D$-dim prototype space, produced by a `ProbabilisticProtoHead` from the fc1 features. The mean $\boldsymbol\mu_c$ is the class representative; the variance $\boldsymbol\sigma_c^2$ is, importantly, a *per-client, per-class* confidence about where that representative sits. A client that has seen fifty effusion films knows the effusion prototype tightly; one that has seen three does not, and the variance says so. This is the information Bayesian fusion (below) needs to weight clients sensibly — a point estimate alone cannot carry it.

The diagonal form is a deliberate choice, not a default we forgot to upgrade. Three reasons, in decreasing order of importance. First, *estimability*: a client sees only a handful of samples per class (shots=50, often fewer after the non-IID split), and a full $D\times D$ covariance has $O(D^2)$ parameters — with $D{=}128$ that is sixteen thousand parameters from fifty samples, which is singular before it is useful. The diagonal needs only $D$ and is estimable from the same data. Second, *communication*: the diagonal adds $D$ floats per class to the upload, negligible next to the backbone; a full covariance would add $D^2$. Third, *composition*: diagonal Gaussians fuse in closed form under the product-of-experts rule (next paragraph), which is what makes the server-side aggregation a one-liner rather than an optimization.

The *cross-class* structure is **not** placed in the per-class covariance — and this is the design decision that the rest of the paper rides on. Putting co-occurrence into each class's covariance would conflate two things that move at different speeds and are visible at different scopes: per-class feature scatter (local, per-client) versus inter-class correlation (global, federated-only). We keep them separate — diagonal Gaussian for the former, shared $\hat R$ for the latter (§3.2) — which is both cheaper and, as Proposition 2 will argue, the only arrangement that can recover the cross-class structure at all.

**【中文】** ### 3.1 分布原型(保留,并简化)

每个类别 $c$ 在 $D$ 维原型空间中被建模为一个对角高斯 $\mathcal{N}(\boldsymbol\mu_c, \mathrm{diag}(\boldsymbol\sigma_c^2))$,由 `ProbabilisticProtoHead` 从 fc1 特征产生。均值 $\boldsymbol\mu_c$ 是该类的代表;方差 $\boldsymbol\sigma_c^2$ 更重要的是,它是*逐客户端、逐类*的、关于这个代表落在哪里的置信度。一个见过五十张积液片的客户端对积液原型把握得很紧;只见过三张的则不然,而方差就把这一点表达了出来。这正是贝叶斯融合(见下)用来合理加权客户端所需的信息——单凭点估计无法承载它。

对角形式是刻意的选择,而非我们忘了升级的默认值。三个理由,按重要性递减。其一,*可估计性*:每个客户端每类只见少量样本(shots=50,经 non-IID 划分后往往更少),而完整 $D\times D$ 协方差有 $O(D^2)$ 个参数——$D{=}128$ 时是一万六千个参数、却只有五十个样本,还没派上用场就已经奇异了。对角只需 $D$ 个参数,同样的数据就能估。其二,*通信*:对角每类只多传 $D$ 个浮点,相对骨干网络可忽略;完整协方差要多传 $D^2$。其三,*可复合*:对角高斯在 product-of-experts 规则下可闭式融合(见下段),这让服务器侧聚合变成一行,而非一个优化问题。

*跨类*结构**不**放在每类协方差里——这是整篇论文所依托的设计决策。把共现塞进每类协方差,会把两件移动速度不同、可见范围也不同的事混为一谈:每类特征散布(局部、逐客户端)与类间相关(全局、仅联邦可见)。我们把二者分开——前者用对角高斯,后者用共享 $\hat R$(§3.2)——这既更廉价,而且正如命题 2 将论证的,是唯一能恢复跨类结构的安排。

**Aggregation.** Per-class Bayesian (product-of-Gaussians) fusion:
$$\boldsymbol\mu_c^g = \frac{\sum_k \boldsymbol\mu_c^k / \boldsymbol\sigma_c^{2,k}}{\sum_k 1/\boldsymbol\sigma_c^{2,k}}, \quad
\boldsymbol\sigma_c^{2,g} = \Big(\sum_k 1/\boldsymbol\sigma_c^{2,k}\Big)^{-1}.$$
This is the product-of-experts rule for diagonal Gaussians, and it does the right thing by construction: clients with lower variance (tighter, more reliable estimates) get higher weight, automatically and per-dimension. A client that is unsure about a class contributes little to it without needing an explicit quality score. The fused variance $\boldsymbol\sigma_c^{2,g}$ is always smaller than any contributor's — aggregation reduces uncertainty, as it should. We keep this aggregation as-is from prior work; it is not where our novelty lies, and we do not claim it as such. Both prototypes and $\hat R$ are EMA-smoothed across rounds (momentum $\beta_{\text{ema}}$) to damp round-to-round noise from the random client subset.

**【中文】** **聚合。** 逐类的贝叶斯(高斯之积)融合:
$$\boldsymbol\mu_c^g = \frac{\sum_k \boldsymbol\mu_c^k / \boldsymbol\sigma_c^{2,k}}{\sum_k 1/\boldsymbol\sigma_c^{2,k}}, \quad
\boldsymbol\sigma_c^{2,g} = \Big(\sum_k 1/\boldsymbol\sigma_c^{2,k}\Big)^{-1}.$$
这是对角高斯的 product-of-experts 规则,且构造上就做对了正确的事:方差更小(更紧、更可靠)的客户端自动、逐维地获得更大权重。一个对某类没把握的客户端,无需显式质量分,自然对它贡献很小。融合后的方差 $\boldsymbol\sigma_c^{2,g}$ 总是小于任何贡献者——聚合降低了不确定性,本该如此。这一聚合沿自以往工作,并非我们的创新所在,也不据此主张。原型和 $\hat R$ 都在跨轮之间做 EMA 平滑(动量 $\beta_{\text{ema}}$),以抑制随机客户端子集带来的轮间噪声。

### 3.2 Federated co-occurrence structure (core)

This is the section the rest of the paper exists to support. The goal is to estimate, federatedly and from label information only, the $C\times C$ matrix of how strongly each pair of pathologies co-occurs — and to do it in a form (a correlation, not raw counts) that downstream losses and decoders can use. The estimator is a sequence of four boring operations on a sufficient statistic; the interesting part is why *this* statistic and *this* transformation, and we spend the paragraph after each saying so.

**Client statistic.** From its multi-hot label matrix $\mathbf{Y}_k\in\{0,1\}^{n_k\times C}$, client $k$ computes the sufficient statistics
$$\mathbf{m}_k = \mathbf{Y}_k^\top \mathbf{1}\in\mathbb{R}^C, \qquad \mathbf{M}_k = \mathbf{Y}_k^\top \mathbf{Y}_k\in\mathbb{R}^{C\times C}, \qquad n_k = |\mathbf{Y}_k|,$$
which are integers (211 values for $C{=}14$), contain **no features**, and are cheap to transmit. The reason this is a *sufficient* statistic for the co-occurrence structure — and not merely a convenient summary — is that the likelihood of any label-correlation model on $\mathbf{Y}_k$ depends on the data only through $\mathbf{m}_k,\mathbf{M}_k,n_k$; nothing else about the raw labels carries additional information about pairwise co-occurrence. So we lose nothing by transmitting these three objects instead of the labels, and we gain privacy (no features, no individual labels — only aggregate counts). This is also the statistic Proposition 2(c) will show is *unrecoverable* in full from any single client under non-IID partitioning.

**【中文】** ### 3.2 联邦共现结构(核心)

这是整篇论文为之存在的章节。目标是:仅凭标签信息、联邦地估计出 $C\times C$ 的病理两两共现强度矩阵,并且以"相关"而非原始计数的形式给出,供下游损失与解码器使用。估计器是对充分统计量的一串四个平凡操作;有意思的在于*为何是这个统计量*、*为何是这种变换*,我们在每步之后用一段说明。

**客户端统计量。** 客户端 $k$ 从其多热标签矩阵 $\mathbf{Y}_k\in\{0,1\}^{n_k\times C}$ 计算充分统计量
$$\mathbf{m}_k = \mathbf{Y}_k^\top \mathbf{1}\in\mathbb{R}^C, \qquad \mathbf{M}_k = \mathbf{Y}_k^\top \mathbf{Y}_k\in\mathbb{R}^{C\times C}, \qquad n_k = |\mathbf{Y}_k|,$$
它们都是整数($C{=}14$ 时共 211 个值),**不含任何特征**,且传输开销很小。之所以称其为共现结构的*充分*统计量——而非仅仅是方便的摘要——是因为 $\mathbf{Y}_k$ 上任何标签相关模型的似然,只通过 $\mathbf{m}_k,\mathbf{M}_k,n_k$ 依赖于数据;原始标签里再没有别的东西携带着关于两两共现的额外信息。所以传这三个对象而不传标签,我们什么都没丢,却换来了隐私(无特征、无个体标签——只有聚合计数)。命题 2(c) 也将证明,在 non-IID 划分下,任何单客户端都无法从它自己手里恢复出完整的这一统计量。

**Server fusion.** Aggregating by counts ($\mathbf{M}=\sum_k\mathbf{M}_k$, $\mathbf{m}=\sum_k\mathbf{m}_k$, $N=\sum_k n_k$) gives marginal/joint probabilities $p_c=m_c/N$, $p_{cd}=M_{cd}/N$. Naively one might stop here and use $p_{cd}$ or $M_{cd}/N$ as the coupling. We do not, for a reason that matters: raw co-occurrence frequency is dominated by marginal prevalence. A common disease pair (effusion–infiltration, both frequent) will have large $p_{cd}$ simply because both are common, not because they specifically co-occur; a rare but tightly coupled pair (pneumothorax–consolidation) will have small $p_{cd}$ and be drowned out. The **phi correlation** (Pearson correlation of binary variables) corrects exactly this:
$$R_{cd} = \frac{p_{cd} - p_c p_d}{\sqrt{p_c(1-p_c)\,p_d(1-p_d)}} \in [-1,1].$$
The numerator $p_{cd}-p_c p_d$ is co-occurrence *above what independence would predict*; the denominator normalizes it onto a comparable $[-1,1]$ scale regardless of how common each disease is. This is the difference between "two frequent diseases" and "two diseases that actually go together," and it is why rare-but-meaningful comorbidities survive into $\hat R$.

A shrinkage toward identity $\hat R = (1-\eta)R + \eta I$ then does two jobs at once. It guarantees positive-definiteness (needed for $\hat R$ to be a well-defined coupling matrix in the decoder), and it pulls the small-sample estimate back toward "no correlation" when the data is thin — which is the honest default when a pair is seen only a few times. $\hat R$ is then EMA-smoothed across rounds to track the slowly-evolving global distribution. We also compute the global marginal prior $\boldsymbol\pi = \mathbf{p}$ for the decoder, which the next section needs as the baseline against which "evidence exceeds the prior" is measured.

**【中文】** **服务器融合。** 按计数聚合($\mathbf{M}=\sum_k\mathbf{M}_k$、$\mathbf{m}=\sum_k\mathbf{m}_k$、$N=\sum_k n_k$)得到边际/联合概率 $p_c=m_c/N$、$p_{cd}=M_{cd}/N$。直觉上可以就此打住,用 $p_{cd}$ 或 $M_{cd}/N$ 当耦合。我们没有这么做,原因很重要:原始共现频率被边际患病率主导。一对常见病(积液–浸润,两者都常见)光是因为都常见,$p_{cd}$ 就会很大,而非因为它们特别共现;一对罕见但强耦合的病(气胸–实变)$p_{cd}$ 会很小、被淹没。**phi 相关**(二值变量的 Pearson 相关)恰好纠正这一点:
$$R_{cd} = \frac{p_{cd} - p_c p_d}{\sqrt{p_c(1-p_c)\,p_d(1-p_d)}} \in [-1,1].$$
分子 $p_{cd}-p_c p_d$ 是*超出独立预测*的那部分共现;分母把它归一化到可比的 $[-1,1]$ 尺度,与每种病多常见无关。这正是"两种常见病"与"两种真的相伴的病"之间的差别,也是为何罕见却有意义的共病能保留进 $\hat R$。

随后的向单位阵收缩 $\hat R = (1-\eta)R + \eta I$ 同时干两件事:保证正定性(解码器需要 $\hat R$ 是良定义的耦合矩阵),并在数据稀薄时把小样本估计拉回"无相关"——当一对类只见了几次时,这才是诚实的默认。$\hat R$ 再跨轮做 EMA 平滑,以跟踪缓慢演化的全局分布。我们还计算供解码器使用的全局边际先验 $\boldsymbol\pi = \mathbf{p}$,下一节需要它作为"证据超出先验"的衡量基准。

### 3.3 Correlation-aware training: structure loss $L_{co}$

The training side of $\hat R$ answers a question that prior prototype-FL never asks: *where in feature space should the prototypes sit, relative to each other?* Existing methods place each prototype independently (toward its own class features) and leave the inter-class geometry to chance. We instead prescribe it: co-occurring diseases should live in nearby directions, mutually exclusive ones apart.

Concretely, for the classes present in a batch, form their batch-mean prototypes $\mathbf{P}\in\mathbb{R}^{C'\times D}$, L2-normalize to $\hat{\mathbf{P}}$ (so only direction, not magnitude, matters), and let $\mathbf{G}=\hat{\mathbf{P}}\hat{\mathbf{P}}^\top$ be the cosine Gram matrix — i.e. the matrix of pairwise angles between prototypes. The structure loss aligns $\mathbf{G}$ with the corresponding sub-block of $\hat R$:
$$L_{co} = \big\|\mathbf{G} - \hat R_{\mathcal{S}}\big\|_F^2,$$
where $\mathcal{S}$ is the set of present classes. Two design choices are worth defending. First, we match the *cosine* Gram, not raw prototype inner products — this decouples the loss from prototype magnitudes (which the variance head already governs) and asks only that *directions* respect the correlation structure. Second, we match only the sub-block $\hat R_{\mathcal{S}}$ of classes actually in the batch, not the full $C\times C$ — because a batch's prototypes for absent classes are meaningless, and forcing them toward a target would inject noise.

The effect is that $L_{co}$ couples the $C$ prototypes — previously an unstructured set — so that co-occurring diseases occupy nearby directions and mutually exclusive diseases are separated. We want to be clear about what this replaces and why. The natural alternative for controlling prototype layout is an InfoNCE contrastive loss with Jaccard-thresholded positives, optionally paired with an adversarial domain term. Both are heuristic — the Jaccard threshold is an arbitrary design choice, and the adversarial term is notoriously unstable — and on inspection they do, redundantly, what $L_{co}$ now does more directly: pull co-occurring prototypes together and push exclusive ones apart. The difference is that $L_{co}$ uses a *federated, label-statistics-driven* target ($\hat R$) instead of a hand-set threshold, so one clean mechanism replaces two fiddly ones. (§6 discusses the omitted regularizers in full.)

**【中文】** ### 3.3 相关感知训练:结构损失 $L_{co}$

$\hat R$ 在训练侧回答了以往原型联邦从不问的问题:*各类原型在特征空间里彼此应当处于什么相对位置?* 现有方法各自独立地放置原型(拉向本类特征),把类间几何交给偶然。我们则把它规定下来:共现的病应落在相近方向,互斥的病应分开。

具体地,对一个 batch 中出现的类别,取其 batch 均值原型 $\mathbf{P}\in\mathbb{R}^{C'\times D}$,做 L2 归一化得 $\hat{\mathbf{P}}$(于是只有方向、而非幅度起作用),并令 $\mathbf{G}=\hat{\mathbf{P}}\hat{\mathbf{P}}^\top$ 为余弦 Gram 矩阵——即原型两两夹角矩阵。结构损失把 $\mathbf{G}$ 对齐到 $\hat R$ 的对应子块:
$$L_{co} = \big\|\mathbf{G} - \hat R_{\mathcal{S}}\big\|_F^2,$$
其中 $\mathcal{S}$ 是当前出现的类别集合。两个设计选择值得辩护。其一,我们匹配的是*余弦* Gram 而非原始原型内积——这把损失与原型幅度解耦(幅度已由方差头管辖),只要求*方向*尊重相关结构。其二,我们只匹配 batch 中实际出现类别的子块 $\hat R_{\mathcal{S}}$,而非完整 $C\times C$——因为 batch 中缺失类的原型没有意义,强行让它们朝向某个目标只会注入噪声。

效果是 $L_{co}$ 把原本互不相关的 $C$ 个原型耦合起来,使共现的病占据相近方向、互斥的病彼此分离。我们想讲清它替代了什么、为什么。控制原型布局的自然替代方案,是带 Jaccard 阈值正样本的 InfoNCE 对比损失,可选地再配一个对抗域项。两者都是启发式的——Jaccard 阈值是任意的 设计选择,对抗项又以不稳著称——而且细看下来,它们冗余地做着 $L_{co}$ 现在更直接做的事:拉近共现原型、推开互斥原型。区别在于 $L_{co}$ 用的是*联邦、标签统计驱动*的目标($\hat R$)而非手设阈值,于是一个干净的机制替代了两个难调的。(§6 完整讨论被略去的正则项。)

### 3.4 Correlation-aware inference: mean-field decoder

The inference side uses $\hat R$ for a different purpose: not to position prototypes, but to let the prediction for one class shift the prediction for its co-occurring classes. This is where the conditional-independence assumption of §1 is actually broken at decode time.

For a query feature $\mathbf{x}$, the per-class diagonal Mahalanobis energy $e_c = \tfrac12\sum_d (x_d-\mu_{cd})^2/\sigma_{cd}^2$ gives independent logits $s_c = -e_c/T$: how strongly $\mathbf{x}$ matches each prototype, computed class-by-class with no cross-class talk. The independent decoder would stop here, $q_c=\sigma(s_c)$. Instead we model the joint label distribution as a fully-visible Boltzmann machine with pairwise couplings $\hat R$ and run **variational mean-field**:
$$q_c \leftarrow \sigma\!\Big(s_c + \beta\sum_{d\neq c}\hat R_{cd}\,(q_d - \pi_d)\Big), \quad \text{iterated } K\text{ steps}.$$
The term $(q_d-\pi_d)$ is the heart of it: it is the amount by which class $d$'s current belief exceeds its global base rate. If class $d$ is "firing above prior" and $d$ positively co-occurs with $c$ ($\hat R_{cd}>0$), then $c$ gets a positive nudge; if they are mutually exclusive ($\hat R_{cd}<0$), $c$ gets pushed down. A class $c$ co-occurring with a class $d$ whose evidence exceeds the prior ($q_d>\pi_d$) receives a positive nudge — which is, in plain language, the clinical semantics of comorbidity-aware diagnosis: "effusion is present and effusion goes with atelectasis, so raise atelectasis." We iterate $K$ steps because the updates are mutually dependent (every $q_c$ depends on every $q_d$), so a single pass only partially propagates; in practice $K{=}2$ saturates.

Two things make this safe rather than magical. First, the diagonal of $\hat R$ is zeroed, so a class never couples to itself (no self-reinforcement runaway). Second, with $\hat R=I$ the off-diagonal coupling vanishes entirely and the decoder degenerates exactly to independent sigmoids — which is the `--no_cooccurrence` ablation baseline, giving a clean handle on what the structure buys. Complexity is $O(B\cdot C^2)$ per step ($C{=}14$, negligible on a 4090), so the structured decoder is essentially free at inference. We note, and §5.2 addresses honestly, that the decoder is provably *no worse* than independent sigmoids for any $\hat R$, and strictly better when $\hat R$ tracks the residual label correlation; it is not a free lunch that works for arbitrary coupling.

**【中文】** ### 3.4 相关感知推理:mean-field 解码器

推理侧把 $\hat R$ 用于不同目的:不是摆放原型,而是让一个类的预测去偏移与它共现的那些类的预测。§1 中条件独立假设的打破,正是在解码时发生的。

对于查询特征 $\mathbf{x}$,逐类的对角马氏能量 $e_c = \tfrac12\sum_d (x_d-\mu_{cd})^2/\sigma_{cd}^2$ 给出独立 logit $s_c = -e_c/T$:衡量 $\mathbf{x}$ 与每个原型的匹配强度,逐类计算、类间不交流。独立解码器会到此为止,$q_c=\sigma(s_c)$。我们则把联合标签分布建模为带成对耦合 $\hat R$ 的全可见玻尔兹曼机,并运行**变分 mean-field**:
$$q_c \leftarrow \sigma\!\Big(s_c + \beta\sum_{d\neq c}\hat R_{cd}\,(q_d - \pi_d)\Big), \quad \text{迭代 } K \text{ 步}.$$
$(q_d-\pi_d)$ 这一项是核心:它是类 $d$ 当前信念超出其全局基准率的量。若类 $d$ "高于先验地激活",且 $d$ 与 $c$ 正共现($\hat R_{cd}>0$),则 $c$ 获得正向推动;若二者互斥($\hat R_{cd}<0$),则 $c$ 被压低。类 $c$ 与一个证据超过先验($q_d>\pi_d$)的共现类 $d$ 耦合时获得正向推动——用大白话说,这正是共病感知诊断的临床语义:"积液存在,而积液常伴肺不张,于是抬高肺不张。"之所以迭代 $K$ 步,是因为更新彼此依赖(每个 $q_c$ 依赖所有 $q_d$),单趟只能部分传播;实践中 $K{=}2$ 即饱和。

两点让这件事安全而非魔法。其一,$\hat R$ 的对角线置零,故类不会与自身耦合(不会自激失控)。其二,当 $\hat R=I$ 时非对角耦合完全消失,解码器精确退化为独立 sigmoid——这正是 `--no_cooccurrence` 消融基线,为结构带来的增益提供了干净的对照。复杂度为每步 $O(B\cdot C^2)$($C{=}14$,在 4090 上可忽略),故结构化解码器在推理时几乎免费。我们指出(§5.2 诚实地处理了):对任意 $\hat R$,解码器可证明地*不劣于*独立 sigmoid,当 $\hat R$ 追踪到残差标签相关时严格更优;它不是对任意耦合都奏效的免费午餐。

### 3.5 Full objective

The local objective is intentionally minimal — four terms, each with a job nothing else does:
$$\mathcal{L} = \underbrace{L_{CE}}_{\text{classification}} + \lambda_{\text{eff}}\,\underbrace{L_{proto}}_{\text{proto alignment (KL)}} + \lambda_{co}\,\underbrace{L_{co}}_{\text{co-occurrence structure}} + \lambda_{ent}\,\underbrace{L_{ent}}_{\text{anti-collapse}}.$$
$L_{CE}$ is multi-label classification (BCE over the $C$ logits). $L_{proto}$ is the KL between the local and global diagonal Gaussians over positive labels — it pulls each client's prototypes toward the federated consensus, the standard prototype-FL alignment term, with a warmup $\lambda_{\text{eff}}=\lambda\cdot\min(1,(t+1)/W)$ so that alignment only kicks in once global prototypes exist (early rounds aligning against noise helps nobody). $L_{co}$ is §3.3, the only term that touches inter-class geometry. $L_{ent}=-\overline{\log\sigma^2}$ is a small guardrail: without it the variance head is free to collapse $\sigma^2\to 0$ and turn the distributional prototype back into a point, silently undoing the whole reason we kept variances in §3.1. It is weighted small ($\lambda_{ent}{=}10^{-3}$) so it only acts as a floor, not a target.

The point of listing these four together is what is *not* here. Prototype-FL methods often stack additional regularizers; we considered the obvious candidates and omit each for a concrete reason — per-class temperature (dead code: trained as a parameter but never used at inference), adversarial domain loss (redundant with $L_{co}$'s independence goal, and unstable), contrastive loss (overlaps $L_{proto}$'s pull and $L_{CE}$'s push), and calibration loss (collapses the $(B,D)$ logvar to a scalar, a band-aid for an unconstrained variance head that $L_{ent}$ now handles properly). Each omission is justified in §6. The result is an objective where each term has a distinct, non-overlapping role, which is what makes the ablations in §7 interpretable: when we toggle $\hat R$ or $L_{co}$, we know exactly what we are toggling.

**【中文】** ### 3.5 完整目标

局部目标刻意精简——四项,每一项都承担着别项无法替代的职责:
$$\mathcal{L} = \underbrace{L_{CE}}_{\text{分类}} + \lambda_{\text{eff}}\,\underbrace{L_{proto}}_{\text{原型对齐 (KL)}} + \lambda_{co}\,\underbrace{L_{co}}_{\text{共现结构}} + \lambda_{ent}\,\underbrace{L_{ent}}_{\text{防坍缩}}.$$
$L_{CE}$ 是多标签分类($C$ 个 logit 上的 BCE)。$L_{proto}$ 是局部与全局对角高斯在正标签上的 KL——把每个客户端的原型拉向联邦共识,即标准的原型联邦对齐项,带 warmup $\lambda_{\text{eff}}=\lambda\cdot\min(1,(t+1)/W)$,使对齐只在全局原型建立后才生效(早期轮次对着噪声对齐对谁都没好处)。$L_{co}$ 见 §3.3,是唯一触及类间几何的项。$L_{ent}=-\overline{\log\sigma^2}$ 是个小护栏:没有它,方差头可自由地让 $\sigma^2\to 0$、把分布原型变回点原型,悄无声息地毁掉 §3.1 保留方差的全部理由。其权重很小($\lambda_{ent}{=}10^{-3}$),只作下限、不作目标。

把这四项列在一起的意义,在于*没有*列进来的东西。原型联邦方法常会堆叠额外正则项;我们考察了显而易见的候选,并因具体理由略去每一项——逐类温度(死代码:作为参数训练,但推理时从不用)、对抗域损失(与 $L_{co}$ 的独立性目标冗余,且不稳定)、对比损失(与 $L_{proto}$ 的拉近、$L_{CE}$ 的推远重叠)、校准损失(把 $(B,D)$ 的 logvar 坍缩成标量,本是对不受约束方差头的权宜之计,现由 $L_{ent}$ 正当处理)。每一处略去都在 §6 给出理由。结果是一个每项职责清晰、互不重叠的目标,这也正是 §7 消融可解读的原因:当我们拨动 $\hat R$ 或 $L_{co}$ 时,我们确切知道自己在拨动什么。

---

## 4. Algorithm

```
FedCoP (per round t):
  Server samples clients S_t; broadcasts {prototypes μ_c^g, σ_c^2^g} and R̂, π.
  Each client k ∈ S_t:
      Local SGD on L = L_CE + λ_eff·L_proto + λ_co·L_co + λ_ent·L_ent   (uses R̂)
      Upload (μ_c^k, σ_c^2^k) per seen class c, and (m_k, M_k, n_k).
  Server:
      μ_c^g, σ_c^2^g ← BayesianFusion({(μ_c^k, σ_c^2^k)}_k)   then EMA
      R̂, π ← FuseCooccurrence({(m_k, M_k, n_k)}_k)            then EMA
Inference: s_c = -½ Mahalanobis(x, μ_c^g, σ_c^2^g)/T;
           q ← MeanField(s, R̂, π, β, K);  ŷ_c = 1[q_c > 0.5].
```

**【中文】**
```
FedCoP(每轮 t):
  服务器采样客户端 S_t;广播 {原型 μ_c^g, σ_c^2^g} 以及 R̂, π。
  每个客户端 k ∈ S_t:
      本地 SGD 优化 L = L_CE + λ_eff·L_proto + λ_co·L_co + λ_ent·L_ent   (使用 R̂)
      上传每个见类 c 的 (μ_c^k, σ_c^2^k),以及 (m_k, M_k, n_k)。
  服务器:
      μ_c^g, σ_c^2^g ← BayesianFusion({(μ_c^k, σ_c^2^k)}_k)   随后 EMA
      R̂, π ← FuseCooccurrence({(m_k, M_k, n_k)}_k)            随后 EMA
推理:s_c = -½ Mahalanobis(x, μ_c^g, σ_c^2^g)/T;
     q ← MeanField(s, R̂, π, β, K);  ŷ_c = 1[q_c > 0.5]。
```

---

## 5. Theoretical Analysis

This section makes two claims and defends them. The first is about *decoding*: under what conditions does the per-class sigmoid that everyone uses throw away information, and how much of it does the mean-field decoder get back? The second is about *estimation*: why must the co-occurrence structure be learned federatedly, and how accurate is the estimate once it is? For each result we give the statement, the proof idea in enough detail to be checked, and—this matters to us—where the argument is approximate rather than airtight. We think the honesty is worth the extra words: these theorems are meant to explain *why* FedCoP behaves the way it does, not to oversell it.

**【中文】** 本节给出并论证两个命题。第一个关于*解码*:人人都在用的逐类 sigmoid 在什么条件下会丢信息,mean-field 解码器又能找回多少?第二个关于*估计*:共现结构为什么必须联邦地学,估出来之后又有多准?对每个结论,我们给出陈述、给出足以复核的证明思路,并且——这点对我们很重要——明确指出论证在哪些地方是近似而非严格成立。我们觉得这份坦诚值得多写几个字:这些定理是要解释 FedCoP *为什么*有效,而不是用来吹嘘。

### 5.1 Setup, and a decomposition we will keep returning to

Fix a feature $\mathbf{x}$. The labels $\mathbf{y}\in\{0,1\}^C$ follow some joint posterior $p(\mathbf{y}\mid\mathbf{x})$. The single object a multi-label decoder actually needs is the marginal $p(y_c{=}1\mid\mathbf{x})$ for each $c$—that is what thresholding acts on. The trouble is that these marginals are not, in general, computable from independent per-class scores: the chain rule
$$p(\mathbf{y}\mid\mathbf{x})=\prod_{c=1}^{C} p\!\left(y_c\mid\mathbf{x},\,y_{1:c-1}\right)$$
shows that the $c$-th label depends on the ones before it *unless* those dependencies vanish. They vanish precisely when the labels are conditionally independent given $\mathbf{x}$, i.e. when $p(\mathbf{y}\mid\mathbf{x})=\prod_c p(y_c\mid\mathbf{x})$. So conditional independence is not a cosmetic assumption; it is the exact condition under which the joint factorizes into the $C$ independent sigmoids everyone fits.

**【中文】** ### 5.1 记号,以及一个我们会反复用到的分解

固定一个特征 $\mathbf{x}$。标签 $\mathbf{y}\in\{0,1\}^C$ 服从某个联合后验 $p(\mathbf{y}\mid\mathbf{x})$。多标签解码器真正需要的,是每个 $c$ 的边际 $p(y_c{=}1\mid\mathbf{x})$——阈值判断作用其上。麻烦在于:一般情况下,这些边际并不能由逐类的独立分数算出来。链式法则
$$p(\mathbf{y}\mid\mathbf{x})=\prod_{c=1}^{C} p\!\left(y_c\mid\mathbf{x},\,y_{1:c-1}\right)$$
表明第 $c$ 个标签依赖于它之前的标签,*除非*这些依赖消失。它们消失的充要条件,正是标签在给定 $\mathbf{x}$ 下条件独立,即 $p(\mathbf{y}\mid\mathbf{x})=\prod_c p(y_c\mid\mathbf{x})$。所以条件独立不是无关紧要的假设,而恰恰是联合分布能分解成人人都在拟合的那 $C$ 个独立 sigmoid 的精确条件。

In ChestX-ray14 this factorization fails in a specific, clinically meaningful way: diseases co-occur (effusion with atelectasis, consolidation with infiltration) for reasons the features only partly explain. Let $\Sigma_y$ denote the *residual* label correlation matrix after the features have done their work—the part of the label dependence that $\mathbf{x}$ does not account for. Its off-diagonals $\rho_{cd}$ are zero iff classes $c,d$ are conditionally independent given $\mathbf{x}$. In our setting they are not, and $\|\Sigma_y-I\|_F$ is the natural scalar measure of "how far from independent." Everything below is stated in terms of this quantity.

**【中文】** 在 ChestX-ray14 上,这个分解以一种临床上很具体的方式失效:疾病会共现(积液伴肺不张、实变伴浸润),而这些共现只有部分能被特征解释。令 $\Sigma_y$ 表示特征发挥作用*之后*残留的标签相关矩阵——即 $\mathbf{x}$ 未能解释的那部分标签依赖。其非对角元 $\rho_{cd}$ 为零,当且仅当 $c,d$ 在给定 $\mathbf{x}$ 下条件独立。在我们的场景里它们并不为零,而 $\|\Sigma_y-I\|_F$ 就是"偏离独立有多远"的自然标量度量。下文一切结论都围绕这个量来表述。

### Proposition 1 (Decoder regret under label correlation)

*The independent sigmoid decoder is Bayes-optimal only under conditional independence; away from it, its regret grows with the residual correlation, and the mean-field decoder with couplings $\hat R$ reduces that regret by an amount bounded by the variational gap, which closes as $\hat R$ approaches $\Sigma_y$.*

**【中文】** ### 命题 1(标签相关下的解码 regret)

*独立 sigmoid 解码器仅在条件独立时才是 Bayes 最优;偏离独立时,其 regret 随残差相关增长,而带耦合 $\hat R$ 的 mean-field 解码器能把这部分 regret 减小一个由变分间隙界定的量,且当 $\hat R$ 趋近 $\Sigma_y$ 时间隙闭合。*

**Proof sketch.** The Bayes-optimal decoder outputs the marginal $\bar p_c=p(y_c{=}1\mid\mathbf{x})$ of the *joint* posterior. The independent decoder instead uses $\hat p_c=\sigma(s_c)$, which is the marginal of the *product* approximation $q^{\text{ind}}(\mathbf{y})=\prod_c\sigma(s_c)^{y_c}(1-\sigma(s_c))^{1-y_c}$. These two marginals coincide iff $q^{\text{ind}}$ equals the joint posterior, i.e. iff $\Sigma_y=I$—recovering the factorization condition above. When $\Sigma_y\neq I$ the two diverge, and a standard chain-rule/Pinsker argument bounds the per-class gap. Writing the excess risk as a KL and expanding the joint against the product gives
$$\mathrm{regret}_{\text{ind}} \;\geq\; \tfrac{c}{2}\,D_{\mathrm{KL}}\!\big(p(\mathbf{y}\mid\mathbf{x})\,\big\|\,q^{\text{ind}}(\mathbf{y})\big) \;\geq\; c\cdot\big\|\Sigma_y - I\big\|_F^2,$$
where the first inequality is the usual regret–KL bound (constant $c>0$ depends on the margin at the decision threshold) and the second is the quadratic lower bound on multi-information for nearly-Gaussian Bernoulli ensembles. The important qualitative point: regret is *second order* in the correlation—small $\rho_{cd}$ costs little, but real comorbidity ($\rho$ of order $0.3$–$0.6$ on ChestX-ray14) is not small.

Now replace the product approximation with the mean-field family $q^{\text{mf}}(\mathbf{y})=\prod_c q_c$ but choose the $q_c$ to be the fixed point of the coupled updates in §3.4. This is exactly the stationary point of the mean-field ELBO for a fully-visible Boltzmann machine with pairwise couplings $\hat R$:
$$\max_{q\in\mathcal{F}_{\text{mf}}}\;\mathbb{E}_{q}\!\left[\log p(\mathbf{y}\mid\mathbf{x})\right]-D_{\mathrm{KL}}\!\big(q\,\big\|\,q^{\text{ind}}\big),$$
where $\mathcal{F}_{\text{mf}}$ is the fully-factorized family. Two things follow. First, the mean-field optimum is never worse than the product $q^{\text{ind}}$ (which is just the $\hat R{=}I$ member of the same family), so its regret is at most that of the independent decoder—the decoder can only help. Second, the *remaining* regret is the variational gap between $q^{\text{mf}}$ and the true joint posterior. Decomposing that gap along the pairwise terms shows it is controlled by $\|\hat R-\Sigma_y\|_F^2$, and therefore vanishes as $\hat R\to\Sigma_y$. Hence, whenever the estimated coupling is non-trivial ($\hat R\neq I$) and a reasonable stand-in for the residual correlation, the structured decoder provably recovers part of what the independent decoder leaves on the table, and all of it in the limit. (Full proof, including the multi-information quadratic bound and the KL decomposition, in the appendix.) $\square$

**【中文】** **证明思路。** Bayes 最优解码器输出的是*联合*后验的边际 $\bar p_c=p(y_c{=}1\mid\mathbf{x})$。独立解码器则用 $\hat p_c=\sigma(s_c)$,即*乘积*近似 $q^{\text{ind}}(\mathbf{y})=\prod_c\sigma(s_c)^{y_c}(1-\sigma(s_c))^{1-y_c}$ 的边际。两者相等当且仅当 $q^{\text{ind}}$ 等于联合后验,亦即 $\Sigma_y=I$——正是上面的分解条件。当 $\Sigma_y\neq I$ 时两者分叉,用标准的链式法则/Pinsker 论证可逐类界定其差距。把超额风险写成 KL,并将联合对乘积展开,得
$$\mathrm{regret}_{\text{ind}} \;\geq\; \tfrac{c}{2}\,D_{\mathrm{KL}}\!\big(p(\mathbf{y}\mid\mathbf{x})\,\big\|\,q^{\text{ind}}(\mathbf{y})\big) \;\geq\; c\cdot\big\|\Sigma_y - I\big\|_F^2,$$
其中第一步是常用的 regret–KL 界(常数 $c>0$ 依赖决策阈值处的间隔),第二步是近高斯 Bernoulli 系统下多重信息(multi-information)的二次下界。关键的定性结论是:regret 对相关是*二阶*的——很小的 $\rho_{cd}$ 代价很小,但真实的共病(ChestX-ray14 上 $\rho$ 量级在 $0.3$–$0.6$)并不小。

现在把乘积近似换成 mean-field 族 $q^{\text{mf}}(\mathbf{y})=\prod_c q_c$,但把 $q_c$ 选成 §3.4 中耦合更新的不动点。这恰是带成对耦合 $\hat R$ 的全可见玻尔兹曼机 mean-field ELBO 的不动点:
$$\max_{q\in\mathcal{F}_{\text{mf}}}\;\mathbb{E}_{q}\!\left[\log p(\mathbf{y}\mid\mathbf{x})\right]-D_{\mathrm{KL}}\!\big(q\,\big\|\,q^{\text{ind}}\big),$$
其中 $\mathcal{F}_{\text{mf}}$ 是全分解族。由此得两点:其一,mean-field 最优解不会比乘积 $q^{\text{ind}}$ 差(后者正是同族中 $\hat R{=}I$ 的成员),故其 regret 至多与独立解码器相当——这个解码器只会帮忙、不会帮倒忙。其二,*残余* regret 是 $q^{\text{mf}}$ 与真实联合后验之间的变分间隙。沿成对项分解该间隙可知,它被 $\|\hat R-\Sigma_y\|_F^2$ 控制,因此当 $\hat R\to\Sigma_y$ 时消失。于是,只要估计出的耦合非平凡($\hat R\neq I$)且是残差相关的合理替身,结构化解码器就可证明地找回独立解码器丢掉的一部分信息,并在极限处找回全部。(完整证明,含多重信息二次界与 KL 分解,见附录。) $\square$

### 5.2 A caveat we owe the reader: $\hat R$ approximates $\Sigma_y$, it does not equal it

A careful reader will have noticed a sleight of hand in the last paragraph. Proposition 1 says the variational gap closes as $\hat R\to\Sigma_y$, but $\hat R$ is the *label* co-occurrence correlation (the phi coefficient over raw labels), whereas $\Sigma_y$ is the *residual* correlation after conditioning on features. These are different objects: features that are good at their job will absorb some of the label dependence, so in general $\Sigma_y$ is a shrunk version of the raw label correlation, and $\hat R\to\Sigma_y$ is not literally true.

**【中文】** ### 5.2 一个必须向读者交代的 caveat:$\hat R$ 是 $\Sigma_y$ 的近似,而非相等

细心的读者会注意到上一段有个障眼法。命题 1 说变分间隙在 $\hat R\to\Sigma_y$ 时闭合,但 $\hat R$ 是*标签*共现相关(原始标签上的 phi 系数),而 $\Sigma_y$ 是条件化于特征之后的*残差*相关。这是两个不同的对象:足够好的特征会吸收掉一部分标签依赖,因此一般情况下 $\Sigma_y$ 是原始标签相关的一个"收缩版",$\hat R\to\Sigma_y$ 并非字面成立。

We do not hide this. Two things rescue the argument. First, $\hat R$ and $\Sigma_y$ have the *same sign pattern* and similar magnitude whenever the feature encoder is not so strong that it removes the comorbidity signal entirely—which, empirically, it is not; if it were, no label-correlation method would ever help and our ablations would show nothing. Second, and more importantly, the theorem's actual content is monotonic, not asymptotic: the mean-field decoder is *no worse* than independent sigmoids for *any* $\hat R$ (including a mis-specified one), and the gap shrinks *monotonically* as $\hat R$ better matches $\Sigma_y$. So the practical claim is the weaker and correct one—using a label-correlation estimate as the coupling can only help, and helps more the closer that estimate tracks the residual structure—not the strong claim that it is exactly Bayes-optimal. The shrinkage toward identity ($\hat R=(1-\eta)R+\eta I$) is doing useful work here exactly because it hedging against this mismatch: when we are unsure how much of the raw correlation survives conditioning, pulling toward $I$ is the safe default. We state this plainly because we would rather a reviewer trust a bounded, hedged claim than distrust an overstrong one.

**【中文】** 我们不掩盖这一点。有两点挽救了这个论证。其一,$\hat R$ 与 $\Sigma_y$ 在特征编码器尚未强到完全抹去共病信号时,具有*相同的符号结构*和相近的量级——而经验上它并没有那么强;若真那么强,任何标签相关方法都不会有效,我们的消融也不会显示任何增益。其二,更重要的是,定理的真正内容是单调的、而非渐近的:mean-field 解码器对*任意* $\hat R$(包括错配的)都*不劣于*独立 sigmoid,且间隙随 $\hat R$ 更好地匹配 $\Sigma_y$ 而*单调*缩小。所以实际成立的是较弱且正确的那个主张——把标签相关估计当作耦合只会帮忙,且估计越接近残差结构、帮忙越多——而非"恰好 Bayes 最优"这个过强主张。向单位阵的收缩($\hat R=(1-\eta)R+\eta I$)在这里恰好是在做有用的事:它正是对这种错配的对冲——当我们不确定原始相关有多少能在条件化后幸存时,拉向 $I$ 是安全的默认选择。我们之所以如实说明,是因为宁可让审稿人相信一个有界、有对冲的主张,也不愿让他怀疑一个过强的主张。

### Proposition 2 (Federated co-occurrence is necessary and recoverable)

*The count-aggregated estimator $\hat R$ is unbiased and concentrates around the population co-occurrence $R^\star$ at rate $\tilde O(\sqrt{\log C/(K n_{\min})})$; and under non-IID class partitioning no single client can recover $R^\star$, so the structure is intrinsically federated.*

**【中文】** ### 命题 2(联邦共现的必要性与可恢复性)

*计数聚合估计器 $\hat R$ 无偏,且以速率 $\tilde O(\sqrt{\log C/(K n_{\min})})$ 集中于总体共现 $R^\star$;而在非独立同分布的类别划分下,任何单客户端都无法恢复 $R^\star$,故该结构本质上是联邦的。*

Let $R^\star$ be the population co-occurrence correlation computed from the global label distribution.

**(a) Unbiasedness.** Each client's statistic is an empirical moment over its local labels, so $\mathbb{E}[\mathbf{M}_k\mid\text{participation}]=n_k R^\star_{\text{raw}}$ up to the participation pattern. Summing counts across clients and normalizing by $N=\sum_k n_k$ gives an unbiased estimate of the raw co-occurrence, *provided participation is uniform*. When it is not—and in FL it rarely is—we apply the standard importance-reweighting fix, weighting each client's contribution by $1/\pi_k$ where $\pi_k$ is its (known) participation probability. With this reweighting, $\mathbb{E}[\hat R]=R^\star$. The phi transformation and shrinkage are deterministic post-processing of an unbiased moment estimate; shrinkage toward $I$ introduces bias of order $\eta\|R^\star-I\|$, which is the price we pay for positive-definiteness and is small when $\eta$ is small or the structure is already near-identity.

**【中文】** 令 $R^\star$ 为由全局标签分布计算出的总体共现相关。

**(a) 无偏性。** 每个客户端的统计量是其本地标签上的经验矩,故在给定参与模式时 $\mathbb{E}[\mathbf{M}_k\mid\text{参与}]=n_k R^\star_{\text{raw}}$。把各客户端计数求和并用 $N=\sum_k n_k$ 归一,即得原始共现的无偏估计——*前提是参与均匀*。当参与不均匀时(联邦学习中几乎从不均匀),我们采用标准的重要性重加权修正:用 $1/\pi_k$ 加权每个客户端的贡献,$\pi_k$ 为其(已知)参与概率。经此重加权,$\mathbb{E}[\hat R]=R^\star$。phi 变换与收缩是对无偏矩估计的确定性后处理;向 $I$ 的收缩引入量级为 $\eta\|R^\star-I\|$ 的偏差,这是我们为正定性付出的代价,当 $\eta$ 很小或结构本身已接近单位阵时该偏差很小。

**(b) Concentration.** The aggregated matrix is a sum of independent client contributions (independent conditional on participation). Applying the matrix-Bernstein inequality to this sum bounds the deviation of each entry, and a union bound over all $C^2$ entries gives
$$\Pr\!\big[\|\hat R - R^\star\|_\infty > \epsilon\big] \;\leq\; 2C\exp\!\Big(-\frac{\epsilon^2 K\, n_{\min}}{c'\,\log C}\Big),$$
so $\hat R$ is $\ell_\infty$-consistent at rate $\tilde O(\sqrt{\log C/(K n_{\min})})$. Two honest qualifications. First, the "independent contributions" assumption is exactly true for a single round; the EMA smoothing across rounds correlates successive estimates, which slows the rate by a constant factor (a geometric-series argument) but does not break consistency. Second, the bound is an entrywise $\ell_\infty$ guarantee; we use it because $\hat R$ enters the decoder entry-by-entry, and entrywise control is what the application needs.

**【中文】** **(b) 集中性。** 聚合矩阵是独立客户端贡献之和(在给定参与下独立)。对该和应用 matrix-Bernstein 不等式可界定每个元素的偏差,再对所有 $C^2$ 个元素做联合界(union bound),得
$$\Pr\!\big[\|\hat R - R^\star\|_\infty > \epsilon\big] \;\leq\; 2C\exp\!\Big(-\frac{\epsilon^2 K\, n_{\min}}{c'\,\log C}\Big),$$
故 $\hat R$ 以速率 $\tilde O(\sqrt{\log C/(K n_{\min})})$ 达到 $\ell_\infty$ 一致。两处诚实的限定:其一,"独立贡献"假设对单轮严格成立;跨轮 EMA 平滑会使相继估计相关,这会把速率放慢一个常数因子(一个等比级数论证),但不破坏一致性。其二,该界是逐元素的 $\ell_\infty$ 保证;我们采用它,是因为 $\hat R$ 在解码器中是逐元素进入的,而逐元素控制正是应用所需。

**(c) Single-client non-recoverability.** This is the part we find most instructive, and it is almost elementary. Under non-IID class partitioning each client observes only $\text{ways}<C$ classes. Its co-occurrence matrix $\mathbf{M}_k$ therefore has rank at most $\text{ways}$ and—more to the point—every entry corresponding to a pair of classes it never jointly sees is *exactly zero*, not merely noisy. A zero cell here is not evidence of independence; it is evidence of *absence*. No amount of local data can distinguish "these two diseases never co-occur" from "I never see these two diseases." Formally, for any pair $(c,d)$ that client $k$ does not jointly observe, $R^\star_{cd}$ is *unidentified* from client $k$'s data alone—every value in $[-1,1]$ is consistent with what it sees. The global $C\times C$ structure is recoverable only by federated aggregation across clients whose class supports *jointly* cover all $C$ classes. FedALC [An et al., 2024] observed the same necessity in the single-positive-label setting; the contribution here is the formal non-recoverability statement for the *general* non-IID class-partition setting (rank $\le\text{ways}$, exact-zero absent-pair entries) together with the concentration bound of (b). This is why we call the co-occurrence structure intrinsically federated: it is not a multi-label trick that happens to be run in a federated system, but a quantity that is, by its geometry, invisible to any participant and visible only to the federation as a whole.

**【中文】** **(c) 单客户端不可恢复性。** 这是我们觉得最有教益的部分,而且几乎是初等的。在非独立同分布类别划分下,每个客户端只观测 $\text{ways}<C$ 类。因此其共现矩阵 $\mathbf{M}_k$ 的秩至多为 $\text{ways}$,更关键的是——它从未联合见到的类对所对应的每一个元素都*精确为零*,而非只是噪声大。这里的零格不是独立的证据,而是*缺失*的证据。再多的本地数据也无法区分"这两种病从不共现"与"我从未同时见过这两种病"。形式上,对于客户端 $k$ 未联合观测的任意类对 $(c,d)$,$R^\star_{cd}$ 仅凭该客户端数据是*不可识别*的——$[-1,1]$ 中的每一个值都与它所见相容。全局 $C\times C$ 结构只能由类支持*联合*覆盖全部 $C$ 类的客户端经联邦聚合来恢复。FedALC[An 等,2024]在单正标签设定下也观察到了同样的必要性;此处的贡献在于对*一般*非独立同分布类别划分的形式化不可恢复性陈述(秩 $\le\text{ways}$、缺失类对元素精确为零)以及 (b) 的集中界。这正是我们称共现结构"本质上是联邦的"的原因:它不是碰巧在联邦系统里运行的多标签技巧,而是一个在几何上对任何参与者都不可见、只对作为整体的联邦可见的量。

### 5.3 What the theory does and does not claim

We close by drawing the lines explicitly, because the lines are where reviewers—and we—should push. The theory claims: (i) the independent decoder is optimal exactly under conditional independence and pays a second-order price otherwise; (ii) the mean-field decoder is never worse and provably recovers part of the gap, monotonically in the quality of the coupling estimate; (iii) the federated estimator is unbiased (under reweighting) and concentrates; and (iv) the structure is unrecoverable by any single client. It does *not* claim: that $\hat R$ equals $\Sigma_y$ (it approximates it, hedged by shrinkage); that the matrix-Bernstein rate survives EMA without a constant-factor slowdown (it does not); or that the mean-field fixed point is unique under strong coupling (it need not be, though in practice $K\!\leq\!3$ steps converge to the same solution). The experiments in §7 test each of the claims (i)–(iv) through the three ablations.

**【中文】** ### 5.3 理论主张了什么、没有主张什么

我们在此把界线明确画出,因为界线正是审稿人——以及我们自己——应当用力之处。理论主张:(i) 独立解码器恰在条件独立时最优,否则付出二阶代价;(ii) mean-field 解码器不会更差,且可证明地找回部分间隙,其找回量随耦合估计质量单调改善;(iii) 联邦估计器(在重加权下)无偏且集中;(iv) 该结构对任何单客户端都不可恢复。它*不*主张:$\hat R$ 等于 $\Sigma_y$(它只是近似,并以收缩对冲);matrix-Bernstein 速率在 EMA 下不经历常数因子变慢(它会变慢);或 mean-field 不动点在强耦合下唯一(不必唯一,尽管实践中 $K\!\leq\!3$ 步会收敛到同一解)。§7 的实验通过三个消融逐一检验主张 (i)–(iv)。

---

## 6. Omitted components and why

A reasonable worry about a four-loss objective is that we have left out something useful. Prototype-FL methods in the literature commonly stack additional regularizers — per-class temperatures, adversarial domain invariance, contrastive losses, calibration terms, feature disentanglement. We considered each, and omit each for a concrete reason rather than by oversight. The table records the verdicts; the paragraphs after it explain the two that most need defending.

**【中文】** ## 6. 略去的组件及理由

对一个四项损失的目标,合理的担忧是我们漏掉了有用的东西。文献中的原型联邦方法常会堆叠额外正则项——逐类温度、对抗域不变、对比损失、校准项、特征解耦。我们逐项考察,并因具体理由(而非疏忽)略去每一项。下表记录裁定;表后两段解释最需辩护的两项。

| Component | Verdict | Reason |
|---|---|---|
| Per-class temperature | **Omitted** | Trained as a parameter but never used at inference (dead code). |
| Adversarial domain loss (GRL) | **Omitted** | Redundant with $L_{co}$'s structure goal; GRL is unstable and adds a domain classifier. |
| InfoNCE contrastive loss | **Omitted** | Overlaps $L_{proto}$ (same-class pull) and $L_{CE}$ (cross-class push); subsumed by $L_{co}$. |
| Calibration loss $L_{cal}$ | **Omitted** | Collapses $(B,D)$ logvar to a scalar mean; a band-aid for unconstrained logvar that $L_{ent}$ handles properly. |
| Semantic-style disentanglement | **Omitted** | Orthogonal story line that dilutes the co-occurrence claim; heaviest component. |
| Entropy regularization $L_{ent}$ | **Retained** | Genuine anti-collapse guardrail for the distributional head. |
| Bayesian fusion / EMA / warmup | **Retained** | Natural stabilizers (not claimed as novelty). |

**【中文】**

| 组件 | 裁定 | 原因 |
|---|---|---|
| 逐类温度 | **略去** | 作为参数训练,但推理时从未使用(死代码)。 |
| 对抗域损失(GRL) | **略去** | 与 $L_{co}$ 的结构目标冗余;GRL 不稳定且引入一个域分类器。 |
| InfoNCE 对比损失 | **略去** | 与 $L_{proto}$(同类拉近)和 $L_{CE}$(跨类推远)重叠;被 $L_{co}$ 涵盖。 |
| 校准损失 $L_{cal}$ | **略去** | 把 $(B,D)$ 的 logvar 坍缩成标量均值;本是对不受约束 logvar 的权宜之计,$L_{ent}$ 已正当处理。 |
| 语义-风格解耦 | **略去** | 另一条与之正交的故事线,稀释了共现主张;是最重的组件。 |
| 熵正则 $L_{ent}$ | **保留** | 分布头真正的防坍缩护栏。 |
| 贝叶斯融合 / EMA / warmup | **保留** | 自然的稳定器(未作为创新点主张)。 |

The contrastive loss deserves a word, because it is the most tempting to add back. InfoNCE with Jaccard-thresholded positives does pull co-occurring prototypes together — but it also pushes *all* non-positive pairs apart with equal force, including pairs that are merely rare rather than mutually exclusive, and it ignores negative co-occurrence ($\hat R_{cd}<0$) entirely. $L_{co}$ does both jobs with one target: it pulls, pushes, and sign-flips according to the actual estimated correlation, not a binary threshold. The calibration loss is the other case worth naming: a term meant to keep the variance head well-behaved, but which in practice collapses the per-sample logvar to a scalar mean — destroying exactly the per-client, per-class confidence signal that Bayesian fusion needs. $L_{ent}$, a simple $-\overline{\log\sigma^2}$ floor, keeps variances off zero without flattening them, which is what the variance head actually needed.

**【中文】** 对比损失值得一说,因为它最让人想加回来。带 Jaccard 阈值正样本的 InfoNCE 确实能拉近共现原型——但它也会把*所有*非正对以同等力度推开,包括那些只是罕见、而非互斥的对,而且它完全无视负共现($\hat R_{cd}<0$)。$L_{co}$ 用一个目标同时干好两件事:依据实际估计的相关而非二元阈值来拉近、推远并翻转符号。校准损失是另一个该点名的:本意是让方差头表现良好,但实践中会把逐样本 logvar 坍缩成标量均值——恰恰毁掉了贝叶斯融合所需的逐客户端、逐类置信信号。$L_{ent}$,一个简单的 $-\overline{\log\sigma^2}$ 下限,把方差托离零而不压平,这才是方差头真正需要的。

The result is a single, sharp mechanism (federated $\hat R$) on both sides of the pipeline, with a clean 4-loss objective — easier to ablate and to attribute gains. We do not claim that omitting these components is itself novel; we claim that the co-occurrence structure makes them redundant, and that a minimal objective makes the contribution of each remaining term — and of $\hat R$ itself — measurable.

**【中文】** 其结果是在流程两端各保留一个单一、锋利的机制(联邦 $\hat R$),并配以一个干净的 4 项损失目标——更易于做消融、也更容易归因性能增益。我们并不主张略去这些组件本身是创新;我们主张的是,共现结构使它们冗余,而一个极简目标让每个保留项——以及 $\hat R$ 本身——的贡献可被度量。

---

## 7. Experiments

The experiments have two jobs. The first is to place FedCoP against the methods it supersedes — does modeling co-occurrence federatedly actually move the multi-label metrics that matter, and where (rare classes? co-occurring classes?)? The second is to attribute any gain to its cause. FedCoP changes three things at once (a federated $\hat R$, a training-side loss $L_{co}$, an inference-side decoder), and a single number cannot tell us which of them is doing the work. So the bulk of this section is an ablation that toggles each independently. We report the design here; the design is fixed before results are collected, so that the ablations are a test of pre-registered hypotheses rather than a post-hoc story.

**【中文】** ## 7. 实验

实验有两项任务。其一是把 FedCoP 与它所要超越的方法对比——联邦地建模共现,是否真能提升那些要紧的多标签指标,以及提升在何处(罕见类?共现类?)?其二是把任何增益归因到原因。FedCoP 一次改变了三件事(联邦 $\hat R$、训练侧损失 $L_{co}$、推理侧解码器),单个数字无法告诉我们是哪一个在起作用。因此本节主体是一个逐项独立拨动的消融。我们在此报告设计;设计在收集结果前就已固定,故这些消融是对预注册假设的检验,而非事后的叙事。

### 7.1 Setup

- **Dataset.** NIH ChestX-ray14, 14 thoracic pathologies, multi-label. 80/20 train/test split; non-IID class partitioning with $\text{ways}{=}3$, $\text{shots}{=}50$, $\text{stdev}{=}2$, $K{=}10$ clients, $30$ communication rounds, participation fraction $0.5$. The non-IID split is the whole point: with $\text{ways}{=}3$ each client sees only 3 of 14 classes, which is the regime in which no single client can recover the co-occurrence structure (Proposition 2c) and in which FedCoP's federated estimate is supposed to matter most.
- **Backbone.** ImageNet-pretrained ResNet-50, $D{=}128$ prototype dim, shared across all methods for fairness.
- **Baselines.** FedAvg, FedProx, FedProto, FedGMKD, FedBCS, FedSeProto — the weight-sharing, proximal, and prototype-sharing families. We re-implement the prototype baselines under the same backbone and split for a controlled comparison; where our re-implementation is a simplification of the original (e.g. FedBCS uses a 1D InstanceNorm recalibration in place of the original frequency-domain separation), we say so explicitly rather than silently inheriting an advantage.
- **Metrics.** Macro/micro AUROC, macro/micro F1, Hamming loss, subset accuracy, per-class AUROC. We deliberately lead with macro metrics: ChestX-ray14 is label-imbalanced (atelectasis is common, hernia is rare), and a flattened per-label accuracy is dominated by the easy negatives of frequent classes — it can move the wrong way while real diagnostic performance improves. Macro-AUROC weights the rare classes that clinical care about.
- **Repeats.** 3 seeds; we report mean±std.

**【中文】** ### 7.1 设置

- **数据集。** NIH ChestX-ray14,14 种胸科病理,多标签。80/20 训练/测试划分;非独立同分布类别划分,$\text{ways}{=}3$、$\text{shots}{=}50$、$\text{stdev}{=}2$、$K{=}10$ 个客户端、$30$ 轮通信、参与比例 $0.5$。non-IID 划分正是全部意义所在:$\text{ways}{=}3$ 时每个客户端只见 14 类中的 3 类,这正是任何单客户端都无法恢复共现结构(命题 2c)的体制,也是 FedCoP 的联邦估计本应发挥作用最大的体制。
- **骨干网络。** ImageNet 预训练 ResNet-50,$D{=}128$ 原型维度,所有方法共享以保证公平。
- **基线。** FedAvg、FedProx、FedProto、FedGMKD、FedBCS、FedSeProto——权重共享、近端、原型共享三族。我们在同一骨干与划分下重新实现原型基线以做受控对比;凡重新实现是对原方法的简化(如 FedBCS 用 1D InstanceNorm 重校准替代原频域分离),我们都明确说明,而非悄然继承优势。
- **指标。** Macro/micro AUROC、macro/micro F1、Hamming loss、subset accuracy、逐类 AUROC。我们刻意以 macro 指标为首:ChestX-ray14 标签不平衡(肺不张常见、疝气罕见),而扁平的逐标签准确率被常见类的易判负样本主导——它可能朝错误方向移动,而真实诊断表现却在改善。macro-AUROC 加权了临床关心的罕见类。
- **重复。** 3 个随机种子;报告 mean±std。

### 7.2 Ablations (FedCoP)

The ablation toggles each of FedCoP's three moving parts independently, so that each row isolates exactly one design choice:

| Variant | $\hat R$ | $L_{co}$ | Decoder | Isolates |
|---|---|---|---|---|
| FedCoP (full) | federated | on | mean-field | — |
| `--no_cooccurrence` | $I$ | off | independent | total co-occurrence contribution |
| `--local_cooc_only` | per-client local | on | mean-field(local) | necessity of federated aggregation (Prop. 2c) |
| `--no_lco` | federated | off | mean-field | training-side vs inference-side structure |

**【中文】** ### 7.2 消融(FedCoP)

该消融独立拨动 FedCoP 的三个运动部件,使每一行恰好隔离一个设计选择:

| 变体 | $\hat R$ | $L_{co}$ | 解码器 | 隔离的贡献 |
|---|---|---|---|---|
| FedCoP(完整) | 联邦 | 开 | mean-field | — |
| `--no_cooccurrence` | $I$ | 关 | 独立 | 共现结构的总贡献 |
| `--local_cooc_only` | 各客户端本地 | 开 | mean-field(本地) | 联邦聚合的必要性(命题 2c) |
| `--no_lco` | 联邦 | 关 | mean-field | 训练侧 vs 推理侧结构 |

The three ablations test three distinct claims, and we state the expected ordering before seeing numbers so the test is meaningful. (i) **Structure helps**: full $>$ `--no_cooccurrence` — disabling co-occurrence entirely (forcing $\hat R{=}I$, which turns off $L_{co}$ and degenerates the decoder to independent sigmoids) should hurt, especially on co-occurring classes. (ii) **Federation is necessary**: full $>$ `--local_cooc_only` — estimating $\hat R$ from each client's local labels alone should recover the structure for locally-seen class pairs but leave all absent-pair entries at zero, so the gain should concentrate on rare and co-occurring classes that are locally absent; this is the direct empirical test of Proposition 2c. (iii) **Both sides contribute**: `--no_lco` should land *between* the full model and `--no_cooccurrence` — keeping the inference-side mean-field decoder but removing the training-side $L_{co}$ should give part of the gain, because inference-side evidence propagation helps even when the prototype geometry was not explicitly shaped. If `--no_lco` matched the full model, $L_{co}$ would be redundant; if it matched `--no_cooccurrence`, the decoder would be doing all the work. Either outcome would be informative.

**【中文】** 三个消融检验三个不同主张,我们在看数字前陈述预期排序,以使检验有意义。(i) **结构有用**:完整 $>$ `--no_cooccurrence`——完全禁用共现(强制 $\hat R{=}I$,从而关掉 $L_{co}$ 并使解码器退化为独立 sigmoid)应当有害,尤以共现类为甚。(ii) **联邦必要**:完整 $>$ `--local_cooc_only`——仅从各客户端本地标签估计 $\hat R$,能恢复本地所见类对的结构,但所有缺失类对的元素仍为零,故增益应集中在本地缺失的罕见与共现类上;这是命题 2c 的直接实证检验。(iii) **两侧均有贡献**:`--no_lco` 应落在完整模型与 `--no_cooccurrence` *之间*——保留推理侧 mean-field 解码器但移除训练侧 $L_{co}$,应带来部分增益,因为即便原型几何未被显式塑造,推理侧的证据传播也仍有帮助。若 `--no_lco` 与完整模型持平,$L_{co}$ 即为冗余;若与 `--no_cooccurrence` 持平,则解码器包办了一切。两种结果都各有启发。

### 7.3 Running

```bash
# Smoke test (5 rounds, 5 users, single seed)
bash scripts/run_test.sh fedcop

# Full benchmark (3 seeds, all algos + ablations, mean±std summary)
bash scripts/run.sh

# Single ablation — full hyperparameters shown so the command is reproducible
# as-is (the defaults in options.py differ; run.sh sets these explicitly).
python exps/federated_main.py --alg fedcop --dataset cxray14 \
    --num_classes 14 --ways 3 --shots 50 --stdev 2 \
    --num_users 10 --frac 0.5 --rounds 30 --proto_dim 128 \
    --ld 1.0 --ld_warmup 20 --co_lambda 0.1 --ent_lambda 1e-3 --no_cooccurrence
python exps/federated_main.py --alg fedcop --dataset cxray14 \
    --num_classes 14 --ways 3 --shots 50 --stdev 2 \
    --num_users 10 --frac 0.5 --rounds 30 --proto_dim 128 \
    --ld 1.0 --ld_warmup 20 --co_lambda 0.1 --ent_lambda 1e-3 --local_cooc_only
python exps/federated_main.py --alg fedcop --dataset cxray14 \
    --num_classes 14 --ways 3 --shots 50 --stdev 2 \
    --num_users 10 --frac 0.5 --rounds 30 --proto_dim 128 \
    --ld 1.0 --ld_warmup 20 --co_lambda 0.1 --ent_lambda 1e-3 --no_lco
```

**【中文】** ### 7.3 运行

```bash
# 冒烟测试(5 轮,5 个用户,单种子)
bash scripts/run_test.sh fedcop

# 完整基准(3 个种子,所有算法 + 消融,mean±std 汇总)
bash scripts/run.sh

# 单个消融——此处给出完整超参,使命令原样可复现
#(options.py 的默认值与之不同;run.sh 会显式设置这些值)。
python exps/federated_main.py --alg fedcop --dataset cxray14 \
    --num_classes 14 --ways 3 --shots 50 --stdev 2 \
    --num_users 10 --frac 0.5 --rounds 30 --proto_dim 128 \
    --ld 1.0 --ld_warmup 20 --co_lambda 0.1 --ent_lambda 1e-3 --no_cooccurrence
python exps/federated_main.py --alg fedcop --dataset cxray14 \
    --num_classes 14 --ways 3 --shots 50 --stdev 2 \
    --num_users 10 --frac 0.5 --rounds 30 --proto_dim 128 \
    --ld 1.0 --ld_warmup 20 --co_lambda 0.1 --ent_lambda 1e-3 --local_cooc_only
python exps/federated_main.py --alg fedcop --dataset cxray14 \
    --num_classes 14 --ways 3 --shots 50 --stdev 2 \
    --num_users 10 --frac 0.5 --rounds 30 --proto_dim 128 \
    --ld 1.0 --ld_warmup 20 --co_lambda 0.1 --ent_lambda 1e-3 --no_lco
```

---

## 8. Notation

| Symbol | Meaning |
|---|---|
| $C=14$ | number of pathologies |
| $D$ | prototype dimension (128) |
| $\boldsymbol\mu_c, \boldsymbol\sigma_c^2$ | diagonal-Gaussian prototype of class $c$ |
| $\hat R\in\mathbb{R}^{C\times C}$ | federated co-occurrence correlation matrix |
| $\boldsymbol\pi$ | global marginal prior $p_c$ |
| $\mathbf{m}_k, \mathbf{M}_k, n_k$ | client label sufficient statistics |
| $\eta$ | shrinkage coefficient (`cov_shrinkage`) |
| $\beta$ | mean-field coupling strength (`co_beta`) |
| $K$ | mean-field iterations (`co_mf_steps`) |

**【中文】**

| 符号 | 含义 |
|---|---|
| $C=14$ | 病理数量 |
| $D$ | 原型维度(128) |
| $\boldsymbol\mu_c, \boldsymbol\sigma_c^2$ | 第 $c$ 类对角高斯原型 |
| $\hat R\in\mathbb{R}^{C\times C}$ | 联邦共现相关矩阵 |
| $\boldsymbol\pi$ | 全局边际先验 $p_c$ |
| $\mathbf{m}_k, \mathbf{M}_k, n_k$ | 客户端标签充分统计量 |
| $\eta$ | 收缩系数(`cov_shrinkage`) |
| $\beta$ | mean-field 耦合强度(`co_beta`) |
| $K$ | mean-field 迭代步数(`co_mf_steps`) |

## 9. Loss-weight table

| Loss | Weight | Default | Active when |
|---|---|---|---|
| $L_{CE}$ | — | — | always |
| $L_{proto}$ | $\lambda_{\text{eff}}$ (warmup) | `--ld 1.0`, `--ld_warmup 20` | global prototypes exist |
| $L_{co}$ | $\lambda_{co}$ | `--co_lambda 0.1` | $\hat R$ available, not `--no_lco` |
| $L_{ent}$ | $\lambda_{ent}$ | `--ent_lambda 1e-3` | always (guardrail) |

**【中文】**

| 损失 | 权重 | 默认 | 生效条件 |
|---|---|---|---|
| $L_{CE}$ | — | — | 始终 |
| $L_{proto}$ | $\lambda_{\text{eff}}$(warmup) | `--ld 1.0`,`--ld_warmup 20` | 全局原型已建立 |
| $L_{co}$ | $\lambda_{co}$ | `--co_lambda 0.1` | $\hat R$ 可用且非 `--no_lco` |
| $L_{ent}$ | $\lambda_{ent}$ | `--ent_lambda 1e-3` | 始终(护栏) |

---

## References

1. McMahan et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017.
2. Li et al. *Federated Optimization in Heterogeneous Networks.* MLSys 2020. (FedProx)
3. Tan et al. *FedProto: Federated Prototype Learning across Heterogeneous Clients.* AAAI 2022.
4. FedGMKD. NeurIPS 2024.
5. FedBCS. AAAI 2026.
6. FedSeProto. ECAI 2024.
7. Angelopoulos & Bates. *A Gentle Introduction to Conformal Prediction.* 2021. (related uncertainty)
8. Tropp. *User-friendly tail bounds for matrix martingales.* (matrix Bernstein)
9. An et al. *Federated Learning with Only Positive Labels by Exploring Label Correlations.* (FedALC) arXiv:2404.15598, 2024.
10. Dembczyński, Waegeman, Cheng & Hüllermeier. *On Label Dependence and Loss Minimization in Multi-Label Classification.* JMLR 2012. (basis for Proposition 1)

**【中文】参考文献**

1. McMahan 等。《Communication-Efficient Learning of Deep Networks from Decentralized Data。》AISTATS 2017。
2. Li 等。《Federated Optimization in Heterogeneous Networks。》MLSys 2020。(FedProx)
3. Tan 等。《FedProto: Federated Prototype Learning across Heterogeneous Clients。》AAAI 2022。
4. FedGMKD。NeurIPS 2024。
5. FedBCS。AAAI 2026。
6. FedSeProto。ECAI 2024。
7. Angelopoulos & Bates。《A Gentle Introduction to Conformal Prediction。》2021。(相关的不确定性)
8. Tropp。《User-friendly tail bounds for matrix martingales。》(matrix Bernstein)
9. An 等。《Federated Learning with Only Positive Labels by Exploring Label Correlations。》(FedALC)arXiv:2404.15598,2024。
10. Dembczyński, Waegeman, Cheng & Hüllermeier。《On Label Dependence and Loss Minimization in Multi-Label Classification。》JMLR 2012。(命题 1 的基础)
