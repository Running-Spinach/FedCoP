# =============================================================================
# 功能:FedCoP 本地更新模块
# =============================================================================
# 这个文件是联邦学习中"客户端本地训练"的核心,包含:
#   1. DatasetSplit — 数据集分割工具
#   2. LocalUpdate — 客户端本地训练类(含所有算法的训练函数)
#   3. 评估函数 — 多标签测试推理
#   4. 两个基线算法 — FedGMKD, FedSeProto
#
# 最重要的函数是 update_weights_FedCoP,FedCoP 的核心训练逻辑。
# 建议阅读顺序:
#   先看 DatasetSplit → LocalUpdate.__init__ → update_weights_FedCoP
#   → test_inference_FedCoP
# =============================================================================

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
import sys
from pathlib import Path
lib_dir = (Path(__file__).parent).resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from dist_proto.losses import (distributional_proto_loss,
                                 entropy_regularization)
from dist_proto.structured import (cos_gram_structure_loss,
                                     mean_field_decode)
from metrics import compute_multilabel_metrics, format_metrics


# =============================================================================
#  DatasetSplit — 数据集子集包装器
# =============================================================================

class DatasetSplit(Dataset):
    """数据集分割类：根据给定索引从完整数据集中提取子集

    用途：
        联邦学习中每个客户端只拥有一部分数据。这个类根据索引列表
        从完整数据集中"切出"属于某个客户端的那部分数据。

    和 PyTorch 自带的 Subset 的区别：
        Subset 共享底层 dataset 的 transform，所有子集用同一个预处理。
        DatasetSplit 允许每个子集独立设置 transform（如果需要的话）。
    """

    def __init__(self, dataset, idxs):
        """
        参数:
            dataset: 完整数据集（如 ChestXray14 的 train/test 集）
            idxs:    要提取的样本索引列表（np.array 或 list）
        """
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        """返回子数据集样本总数"""
        return len(self.idxs)

    def __getitem__(self, item):
        """获取指定索引的数据样本

        参数:
            item: 子数据集中的索引（0 到 len-1）

        返回:
            (image, label): 图像和标签。确保都是 torch.Tensor 格式。
        """
        image, label = self.dataset[self.idxs[item]]
        # 统一转成 torch.Tensor（以防原始数据是 PIL 或 numpy 格式）
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        return image, label


# =============================================================================
#  LocalUpdate — 客户端本地训练类
# =============================================================================

class LocalUpdate(object):
    """本地更新类：封装单个客户端的本地训练逻辑

    这个类代表联邦学习中的一个"客户端"（医院）。
    它持有：
    - 本地数据加载器（trainloader）
    - 损失函数（BCEWithLogitsLoss，适配多标签分类）
    - 多种算法的本地训练函数

    支持的训练函数：
    - update_weights           → FedAvg / FedBN
    - update_weights_FedP      → FedProto（点原型基线）
    - update_weights_fedprox   → FedProx
    - update_weights_scaffold  → SCAFFOLD
    - update_weights_FedCoP    → FedCoP(提出方法)★ 核心
    - _update_weights_FedGMKD  → FedGMKD（GMM 基线）
    - _update_weights_FedSeProto → FedSeProto（语义-域解耦基线）
    """

    def __init__(self, args, dataset, idxs):
        """
        参数:
            args:    全局配置参数
            dataset: 训练数据集
            idxs:    该客户端对应的数据索引
        """
        self.args = args
        # 构建本客户端的训练数据加载器
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        # BCEWithLogitsLoss = sigmoid + BCE，比手动 sigmoid 再 BCE 更数值稳定
        # 适配 ChestX-ray14 的多标签场景（每张图同时可能有多种疾病）
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """根据数据集和索引构建训练数据加载器

        注意：这里用 100% 的数据做训练（idxs[:int(1 * len(idxs))]），
        没有划分验证集。在联邦学习场景中，验证通常在服务器端用全局测试集进行。

        参数:
            dataset: 数据集
            idxs:    数据索引列表

        返回:
            DataLoader: 训练数据加载器，drop_last=True 保证 batch 大小一致
        """
        idxs_train = idxs[:int(1 * len(idxs))]
        num_workers = int(getattr(self.args, 'num_workers', 0))
        pin_memory = bool(getattr(self.args, 'pin_memory', 0)) and (self.device == 'cuda')
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory,
                                 persistent_workers=(num_workers > 0))
        return trainloader

    def _get_optimizer(self, model):
        """根据配置返回 SGD 或 Adam 优化器

        SGD:  简单、稳定、联邦学习常用（FedAvg 原论文推荐）
        Adam: 自适应学习率、收敛快、但可能在 Non-IID 数据上不稳定
        """
        if self.args.optimizer == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            return torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

    # ═══════════════════════════════════════════════════════════════════════════
    #  FedAvg 标准本地训练
    # ═══════════════════════════════════════════════════════════════════════════

    def update_weights(self, idx, model, global_round):
        """FedAvg / FedBN 本地训练：纯交叉熵 + SGD 更新

        这是最基础的联邦学习训练方式：本地做几个 epoch 的 SGD，
        然后把模型参数上传给服务器做平均。

        流程：
        for each local epoch:
            for each batch:
                1. 前向传播 → logits
                2. 计算 BCE loss
                3. 反向传播 + 参数更新

        参数:
            idx:          客户端索引（用于日志打印）
            model:        当前模型
            global_round: 全局训练轮次（用于日志打印）

        返回:
            state_dict: 更新后的模型参数
            avg_loss:   平均损失值
            acc_val:    最后一个 batch 的 per-label 准确率
        """
        model.train()
        epoch_loss = []
        optimizer = self._get_optimizer(model)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                output = model(images)
                # 兼容多输出格式（tuple 的第一个元素始终是 logits）
                logits = output[0] if isinstance(output, tuple) else output
                loss = self.criterion(logits, labels)

                loss.backward()
                optimizer.step()

                # 多标签 per-label 准确率：sigmoid(logits) > 0.5 → 预测为正类
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc_val = (preds == labels).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    # ═══════════════════════════════════════════════════════════════════════════
    #  FedProto 本地训练（点原型基线）
    # ═══════════════════════════════════════════════════════════════════════════

    def update_weights_FedP(self, args, idx, global_protos, model, global_round=round, ld=None):
        """FedProto 本地训练：BCE 分类损失 + 原型距离正则化

        损失 = L_CE + ld × L_proto（MSE 原型距离）

        这是 FedCoP 的对比基线。和 FedCoP 的核心区别：
        - 点原型（MSE 距离）vs 分布原型（KL/Wasserstein 距离）
        - 无解耦 vs 语义-风格解耦
        - 无 EMA 动量 vs 原型动量
        - 无温度缩放 vs 每类可学习温度

        参数:
            args:          配置参数
            idx:           客户端索引
            global_protos: 全局原型字典（服务器下发）
            model:         当前模型
            global_round:  全局训练轮次
            ld:            原型损失权重。None 时使用 args.ld

        返回:
            state_dict:       更新后的模型参数
            epoch_loss:       损失字典 {total, 1(CE), 2(proto)}
            acc_val:          最后一个 batch 的准确率
            agg_protos_label: 聚合后的本地原型字典（准备上传）
        """
        if ld is None:
            ld = args.ld
        model.train()
        epoch_loss = {'total': [], '1': [], '2': [], '3': []}

        optimizer = self._get_optimizer(model)

        for iter in range(self.args.train_ep):
            batch_loss = {'total': [], '1': [], '2': [], '3': []}
            agg_protos_label = {}  # 本轮收集的本地原型 {label: [proto_vec, ...]}

            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)

                # 前向传播 → 分类 logits + 原型特征
                model.zero_grad()
                logits, protos = model(images)

                # L1: 交叉熵分类损失
                loss1 = self.criterion(logits, labels)

                # L2: 原型距离损失（MSE）
                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    # 第一轮还没有全局原型，跳过 L2
                    loss2 = 0 * loss1
                else:
                    # 多标签原型损失：对每个样本的所有正标签计算 MSE
                    # 正标签 = 该疾病确实存在（label[i, lbl] == 1）
                    loss2 = 0 * loss1
                    count = 0
                    for i_lbl in range(len(labels)):
                        proto_i = protos[i_lbl]  # 第 i 个样本的原型
                        for lbl_idx in range(args.num_classes):
                            if labels[i_lbl, lbl_idx] > 0 and lbl_idx in global_protos:
                                loss2 += loss_mse(proto_i, global_protos[lbl_idx])
                                count += 1
                    loss2 = loss2 / max(count, 1)  # 防止除零

                # 总损失 = CE + λ × Proto
                loss = loss1 + loss2 * ld
                loss.backward()
                optimizer.step()

                # 收集本地原型（按标签分组）
                for i_lbl in range(len(labels)):
                    for lbl_idx in range(args.num_classes):
                        if label_g[i_lbl, lbl_idx] > 0:
                            proto_val = protos[i_lbl, :].detach()
                            if lbl_idx in agg_protos_label:
                                agg_protos_label[lbl_idx].append(proto_val)
                            else:
                                agg_protos_label[lbl_idx] = [proto_val]

                # 多标签 per-label 准确率
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc_val = (preds == labels).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())

            # Epoch 级别平均
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        # 全局平均
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

    # ═══════════════════════════════════════════════════════════════════════════
    #  FedProx 本地训练
    # ═══════════════════════════════════════════════════════════════════════════

    def update_weights_fedprox(self, idx, global_model_state, model, global_round):
        """FedProx 本地训练：BCE + 近端项惩罚

        损失 = L_CE + (μ/2) × ||w - w_global||²

        近端项的作用：
        - 限制本地模型不要偏离全局模型太远
        - 在 Non-IID 数据上比 FedAvg 更稳定
        - μ 越大，本地模型越"听话"（越接近全局模型）

        参数:
            idx:               客户端索引
            global_model_state: 全局模型的 state_dict
            model:             当前本地模型
            global_round:      全局训练轮次

        返回:
            state_dict, avg_loss, acc_val
        """
        model.train()
        epoch_loss = []
        mu = self.args.fedprox_mu  # 近端项系数
        optimizer = self._get_optimizer(model)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                output = model(images)
                logits = output[0] if isinstance(output, tuple) else output
                loss_ce = self.criterion(logits, labels)

                # FedProx 近端项：对所有可训练参数计算 L2 距离
                # 注意：只计算 requires_grad=True 的参数（跳过冻结层）
                prox_term = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad and name in global_model_state:
                        prox_term += torch.sum(
                            (param - global_model_state[name].to(self.device)) ** 2)
                loss = loss_ce + (mu / 2.0) * prox_term

                loss.backward()
                optimizer.step()

                preds = (torch.sigmoid(logits) > 0.5).float()
                acc_val = (preds == labels).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(),
                        acc_val.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    # ═══════════════════════════════════════════════════════════════════════════
    #  SCAFFOLD 本地训练
    # ═══════════════════════════════════════════════════════════════════════════

    def update_weights_scaffold(self, idx, c_global, c_local, model, global_round):
        """SCAFFOLD 本地训练：使用 control variate 修正梯度

        SCAFFOLD 的核心思想：
        在 Non-IID 数据上，FedAvg 的"本地 SGD → 平均参数"会导致
        各客户端模型漂向不同方向（client drift）。SCAFFOLD 用 control variate
        来修正本地梯度：

        g_corrected = g_raw - c_local + c_global

        其中：
        - c_local：本地 control variate（记录"本地喜欢往哪偏"）
        - c_global：全局 control variate（记录"全局平均往哪偏"）
        - 修正后的梯度 = 本地梯度 - 本地偏差 + 全局偏差
          → 相当于把本地梯度"拉回"全局方向

        参数:
            idx:      客户端索引
            c_global: 全局 control variate（服务器下发，state_dict 格式）
            c_local:  本地 control variate（state_dict 格式）
            model:    当前本地模型
            global_round: 全局训练轮次

        返回:
            state_dict, avg_loss, acc_val, c_local_new, c_delta
        """
        model.train()
        epoch_loss = []
        lr = self.args.lr
        optimizer = self._get_optimizer(model)
        total_steps = 0

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                output = model(images)
                logits = output[0] if isinstance(output, tuple) else output
                loss = self.criterion(logits, labels)
                loss.backward()

                # SCAFFOLD 梯度修正（在 optimizer.step 之前修改 grad）
                for name, param in model.named_parameters():
                    if param.grad is not None and name in c_global and name in c_local:
                        param.grad = (param.grad
                                      - c_local[name].to(self.device)
                                      + c_global[name].to(self.device))

                optimizer.step()
                total_steps += 1

                preds = (torch.sigmoid(logits) > 0.5).float()
                acc_val = (preds == labels).float().mean()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        avg_loss = sum(epoch_loss) / len(epoch_loss)

        # 更新本地 control variate
        # 公式：c_i_new = c_i - c_global + (w_init - w_final) / (lr × K)
        # K = 总训练步数，w_init 是全局模型（本地训练的起点），w_final 是本地训练后的模型
        K = max(total_steps, 1)
        c_local_new = {}
        c_delta = {}
        for name, param in model.named_parameters():
            cl = c_local[name]
            cg = c_global[name]
            cl_new = cl - cg + (cg - param.detach().cpu()) / (lr * K)
            c_local_new[name] = cl_new
            c_delta[name] = cl_new - cl

        return model.state_dict(), avg_loss, acc_val.item(), c_local_new, c_delta

    # ═══════════════════════════════════════════════════════════════════════════
    #  FedCoP 本地训练 ★★★ 核心函数 ★★★
    # ═══════════════════════════════════════════════════════════════════════════

    def update_weights_FedCoP(self, args, idx, global_protos, global_R,
                              model, global_round=round, ld=None):
        """FedCoP 本地训练:分布原型对齐 + 共现结构对齐(4 项损失,精简版)

        FedCoP 相比旧版(FedProto + 分布原型)砍掉了解耦/对抗/对比/校准/每类温度等冗余 trick,
        只保留 4 项各有明确职责的损失:

            L = L_CE + λ_eff·L_proto + λ_co·L_co + λ_ent·L_ent

            1. L_CE   : 多标签分类损失(BCE),做好本职分类
            2. L_proto: 分布原型对齐(本地对角高斯 → 全局对角高斯,KL),让本地
                        原型不跑离全局共识;λ_eff 由 warmup 控制
            3. L_co   : 共现结构对齐(NEW,核心)——把本 batch 各类原型均值的余弦
                        Gram 对齐到联邦共现相关矩阵 R̂ 的子块。共现类原型方向相近,
                        互斥类远离。替代了被砍的对比/对抗损失,动机更干净
            4. L_ent  : 熵正则,防止方差坍缩回点原型(小权重 guardrail)

        上传:每见类的 (μ_c, logvar_c)(供服务器贝叶斯融合)。
        共现统计 (m_k, M_k, n_k) 只依赖标签,在服务器端 taskheter 里从各客户端
        标签矩阵直接计算(等价于客户端上传,且更精确)。

        参数:
            args:          全局配置
            idx:           客户端索引(日志)
            global_protos: 全局原型字典 {label: (mu, logvar)}
            global_R:      全局共现相关矩阵 R̂ (C, C);首轮/无结构消融时为 None
            model:         当前 FedCoPResNet
            global_round:  全局轮次
            ld:            原型损失权重(已含 warmup)。None 用 args.ld

        返回:
            state_dict, epoch_loss{total,1,2,co,ent}, acc_val, agg_protos_label
        """
        if ld is None:
            ld = args.ld
        model.train()

        # ── 损失记录 ──
        epoch_loss = {'total': [], '1': [], '2': [], 'co': [], 'ent': []}

        # ── 配置开关 ──
        dist_type = getattr(args, 'dist_type', 'kl')
        co_lambda = getattr(args, 'co_lambda', 0.1)
        ent_lambda = getattr(args, 'ent_lambda', 1e-3)
        no_lco = getattr(args, 'no_lco', False)          # 消融:关闭 L_co
        loss_mse = nn.MSELoss()

        # R̂ 移到当前设备(首轮 None 时跳过)
        R_dev = global_R.to(self.device) if global_R is not None else None

        optimizer = self._get_optimizer(model)

        # ═══════════════════════════════════════════════════════════════
        #  本地训练循环
        # ═══════════════════════════════════════════════════════════════
        for iter in range(self.args.train_ep):
            batch_loss = {'total': [], '1': [], '2': [], 'co': [], 'ent': []}
            agg_protos_label = {}

            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)
                model.zero_grad()

                # 单一前向模式:始终 (logits, mu, logvar)
                logits, mu, logvar = model(images)

                # ── L1: 多标签分类损失 ──
                loss1 = self.criterion(logits, labels)

                # ── L2: 分布原型对齐(KL/Wasserstein)──
                # 逐样本、逐正标签,把本地高斯拉向全局高斯
                if len(global_protos) == 0:
                    loss2 = 0.0 * loss1
                else:
                    loss2 = 0.0
                    count = 0
                    for i_lbl in range(len(labels)):
                        for lbl_idx in range(args.num_classes):
                            if labels[i_lbl, lbl_idx] > 0 and lbl_idx in global_protos:
                                g_val = global_protos[lbl_idx]
                                g_mu, g_logvar = (g_val if isinstance(g_val, tuple)
                                                  else (g_val, torch.zeros_like(g_val)))
                                l2 = distributional_proto_loss(
                                    mu[i_lbl:i_lbl + 1], logvar[i_lbl:i_lbl + 1],
                                    g_mu.unsqueeze(0), g_logvar.unsqueeze(0),
                                    dist_type=dist_type)
                                loss2 = loss2 + l2
                                count += 1
                    loss2 = loss2 / max(count, 1)

                # ── L_co: 共现结构对齐(核心创新)──
                # 用本 batch 各类原型均值的余弦 Gram 对齐 R̂ 子块
                if (not no_lco) and R_dev is not None and co_lambda > 0:
                    loss_co = cos_gram_structure_loss(
                        mu, labels, R_dev, args.num_classes)
                else:
                    loss_co = 0.0 * loss1

                # ── L_ent: 熵正则(防方差坍缩)──
                if ent_lambda > 0:
                    loss_ent = entropy_regularization(logvar)
                else:
                    loss_ent = 0.0 * loss1

                # ── 总损失(4 项)──
                loss = (loss1
                        + loss2 * ld
                        + loss_co * co_lambda
                        + loss_ent * ent_lambda)

                loss.backward()
                optimizer.step()

                # ── 收集本地原型(按正标签分组,准备上传)──
                for i_lbl in range(len(labels)):
                    for lbl_idx in range(args.num_classes):
                        if label_g[i_lbl, lbl_idx] > 0:
                            proto_val = (mu[i_lbl, :].detach(),
                                         logvar[i_lbl, :].detach())
                            if lbl_idx in agg_protos_label:
                                agg_protos_label[lbl_idx].append(proto_val)
                            else:
                                agg_protos_label[lbl_idx] = [proto_val]

                # ── per-label 准确率 ──
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc_val = (preds == labels).float().mean()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Acc: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item(), acc_val.item()))

                # ── 记录损失分量 ──
                def _val(v):
                    return v.item() if isinstance(v, torch.Tensor) else v
                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(_val(loss2))
                batch_loss['co'].append(_val(loss_co))
                batch_loss['ent'].append(_val(loss_ent))

            # ── Epoch 级平均 ──
            for key in epoch_loss:
                if len(batch_loss[key]) > 0:
                    epoch_loss[key].append(sum(batch_loss[key]) / len(batch_loss[key]))

        # ── 全局平均(所有 epoch)──
        for key in epoch_loss:
            if len(epoch_loss[key]) > 0:
                epoch_loss[key] = sum(epoch_loss[key]) / len(epoch_loss[key])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

    # ═══════════════════════════════════════════════════════════════════════════
    #  本地推理
    # ═══════════════════════════════════════════════════════════════════════════

    def inference(self, model):
        """在测试集上推理，返回 per-label 准确率和损失

        参数:
            model: 待评估的模型

        返回:
            accuracy: per-label 准确率
            loss: 损失值
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            output = model(images)
            outputs = output[0] if isinstance(output, tuple) else output

            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # 多标签 per-label 准确率
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).float().sum().item()
            total += labels.numel()

        accuracy = correct / total
        return accuracy, loss


# =============================================================================
#  多标签联邦学习统一评估（FedAvg / FedProx / FedBN / SCAFFOLD 共用）
# =============================================================================

def eval_clients_multilabel(args, local_model_list, test_dataset, user_groups_gt):
    """多标签联邦学习统一评估

    每个客户端用 sigmoid(logits) > 0.5 对本地测试集做 per-label 准确率,
    并跨客户端累积概率/标签,统一计算 macro/micro AUROC、F1 等完整指标
    (与 FedCoP 的 test_inference_FedCoP 用同一套 compute_multilabel_metrics,
    保证基线与 FedCoP 在同一指标下可比)。

    参数:
        args:             全局配置
        local_model_list: 所有客户端的模型列表
        test_dataset:     测试数据集
        user_groups_gt:   测试数据的客户端划分（IID 时为 None）

    返回:
        acc_list: 各客户端 per-label 准确率列表
    """
    device = args.device
    acc_list = []
    all_probs, all_labels = [], []          # 跨客户端累积,用于整体指标

    # IID 模式或无本地测试划分时，所有客户端共享同一个测试集
    if user_groups_gt is None:
        n_test = len(test_dataset)
        user_groups_gt = [np.arange(n_test) for _ in range(args.num_users)]

    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(device)
        model.eval()
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]),
                                batch_size=64, shuffle=False)

        total_val, correct_val = 0.0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                outputs = output[0] if isinstance(output, tuple) else output

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct_val += (preds == labels).float().sum().item()
                total_val += labels.numel()

                all_probs.append(probs.detach().cpu())
                all_labels.append(labels.detach().cpu())

        acc = correct_val / total_val
        print('| User: {} | Test Acc (per-label): {:.4f}'.format(idx, acc))
        acc_list.append(acc)

    # ── 跨客户端整体多标签指标(AUROC/F1/Hamming/subset)──
    # 打印 AUROC(macro/micro)=... 行,供 run.sh 的 grep 抓取,与 FedCoP 对齐
    if len(all_probs) > 0:
        probs_all = torch.cat(all_probs, dim=0).numpy()
        labels_all = torch.cat(all_labels, dim=0).numpy()
        metrics = compute_multilabel_metrics(probs_all, labels_all,
                                             num_classes=args.num_classes)
        print('[baseline metrics] ' + format_metrics(metrics))

    return acc_list


# =============================================================================
#  FedProto 测试推理（原版 — 单标签 NLLLoss 方式）
# =============================================================================

def test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[]):
    """FedProto 测试推理（多标签适配版）

    两条评估路径:
      - w/o protos:纯分类器 sigmoid(logits) > 0.5
      - w/ protos :纯原型预测(与官方 FedProto 一致)——对每类 c 计算
        d_c = ‖emb − proto_c‖² (MSE),s_c = −d_c/temperature,
        pred_c = sigmoid(s_c) > 0.5。无全局原型的类预测为负。

    返回:
        acc_list_l: 各客户端模型自身分类器的 per-label 准确率
        acc_list_g: 各客户端使用全局原型预测的 per-label 准确率
        loss_list:  各客户端正样本到全局原型的平均 MSE 距离(诊断)
    """
    device = args.device

    acc_list_g = []
    acc_list_l = []
    loss_list = []
    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=False)

        # ── 不使用全局原型的分类测试 ──
        model.eval()
        correct_l, total_l = 0.0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                logits = output[0] if isinstance(output, tuple) else output

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct_l += (preds == labels).float().sum().item()
                total_l += labels.numel()

        acc_l = correct_l / total_l if total_l > 0 else 0.0
        print('| User: {} | Test Acc w/o protos: {:.4f}'.format(idx, acc_l))
        acc_list_l.append(acc_l)

        # ── 使用全局原型的原型预测(与官方 FedProto 一致)──
        # 对每类 c:d_c = ‖emb − proto_c‖² (MSE,与官方距离度量一致)
        #              s_c = −d_c / temperature → 越近 logit 越高
        #              pred_c = sigmoid(s_c) > 0.5
        # 无全局原型的类给极小 logit → sigmoid≈0 → 预测负
        if global_protos:
            temperature = getattr(args, 'temperature', 1.0)
            correct_g, total_g = 0.0, 0.0
            proto_loss_sum = 0.0
            proto_count = 0
            _diag_printed = False
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    output = model(images)

                    if isinstance(output, tuple):
                        protos = output[1] if len(output) >= 2 else output[0]
                    else:
                        protos = output

                    B = images.shape[0]
                    s = torch.full((B, args.num_classes), -1e9, device=device)
                    for c in range(args.num_classes):
                        if c in global_protos:
                            gproto = global_protos[c]
                            gvec = gproto[0] if isinstance(gproto, tuple) else gproto
                            d_c = ((protos - gvec) ** 2).sum(dim=1)   # (B,) MSE 距离
                            s[:, c] = -d_c / temperature
                            # 诊断:正样本到全局原型的距离
                            pos_mask = labels[:, c] > 0
                            if pos_mask.any():
                                proto_loss_sum += d_c[pos_mask].sum().item()
                                proto_count += int(pos_mask.sum().item())

                    # 首客户端首批:打印 σ(s_proto) 量级,诊断 temperature 是否饱和
                    if not _diag_printed and idx == 0:
                        print(f'  [proto-decode-diag] mean σ(s_proto)='
                              f'{torch.sigmoid(s).mean().item():.4f} '
                              f'(若 ≈0 或 ≈1 → 原型 logit 饱和,需调大 --temperature)')
                        _diag_printed = True

                    preds = (torch.sigmoid(s) > 0.5).float()
                    correct_g += (preds == labels).float().sum().item()
                    total_g += labels.numel()

            acc_g = correct_g / total_g if total_g > 0 else 0.0
            proto_loss_val = proto_loss_sum / max(proto_count, 1)
            print('| User: {} | Test Acc with protos: {:.4f}'.format(idx, acc_g))
            acc_list_g.append(acc_g)
            loss_list.append(proto_loss_val)

    return acc_list_l, acc_list_g, loss_list


# =============================================================================
#  FedCoP 多标签测试推理 ★★★ 核心评估函数 ★★★
# =============================================================================

def test_inference_FedCoP(args, local_model_list, test_dataset, classes_list,
                          user_groups_gt, global_protos=[], global_R=None,
                          global_pi=None, local_R_dict=None, temperature=None):
    """FedCoP 测试推理:分类器 + 原型融合 + 相关性感知解码 + 完整多标签指标

    两条评估路径(同一前向,共享特征):
      1. w/o protos:纯分类器 sigmoid(logits) > 0.5
      2. w/ protos :分类器 logit 与原型马氏 logit 融合 → mean-field 解码
           - 每类用对角马氏距离到全局原型 → 独立原型 logit s_c(÷ temperature)
           - fused = α·logit_cls + (1−α)·s_proto(α=args.fuse_alpha)
           - 用共现相关矩阵 R̂ 对 fused 做 mean-field 迭代,在共现类间传播证据
           - R̂=I 或 --no_cooccurrence 时退化为独立 sigmoid(消融基线)
    并在融合解码概率上计算 AUROC/F1/Hamming/subset 等完整指标。

    参数:
        args:           全局配置
        local_model_list: 各客户端模型
        test_dataset:    测试集
        classes_list:    每客户端类别(过滤用)
        user_groups_gt:  测试数据客户端划分
        global_protos:   全局原型 {label: (mu, logvar)}
        global_R:        全局共现相关矩阵 (C, C)
        global_pi:       全局边际先验 (C,)
        local_R_dict:    {idx: (R_k, pi_k)} —— 仅 --local_cooc_only 消融时用
        temperature:     马氏距离→logit 的温度。None 用 args.temperature

    返回:
        acc_list_l:  各客户端模型分类器 per-label 准确率
        acc_list_g:  各客户端原型解码 per-label 准确率
        loss_list:   各客户端原型诊断距离
        metrics_g:   原型解码路径的完整多标签指标 dict
    """
    loss_mse = nn.MSELoss()
    device = args.device
    if temperature is None:
        temperature = getattr(args, 'temperature', 1.0)
    # 共现结构开关
    no_cooc = getattr(args, 'no_cooccurrence', False)
    local_only = getattr(args, 'local_cooc_only', False)
    beta = getattr(args, 'co_beta', 1.0)
    mf_steps = getattr(args, 'co_mf_steps', 2)
    fuse_alpha = getattr(args, 'fuse_alpha', 0.5)   # 分类器 logit 与原型 logit 融合权重

    C = args.num_classes
    eye = torch.eye(C, device=device)
    # 全局 R̂ / pi(本地消融时按客户端覆盖)
    R_glb = global_R.to(device) if global_R is not None else eye
    pi_glb = global_pi.to(device) if global_pi is not None else torch.zeros(C, device=device)

    acc_list_g, acc_list_l, loss_list = [], [], []
    # 跨客户端累积概率/标签,用于整体指标
    all_probs_g, all_labels_g = [], []

    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]),
                                batch_size=64, shuffle=False)

        # 选定本客户端使用的 R̂ / pi(本地共现消融 vs 全局)
        if local_only and local_R_dict is not None and idx in local_R_dict:
            R_use, pi_use = local_R_dict[idx]
            R_use = R_use.to(device)
            pi_use = pi_use.to(device)
        elif no_cooc:
            R_use, pi_use = eye, pi_glb            # 无结构:独立解码
        else:
            R_use, pi_use = R_glb, pi_glb

        # ════════ 单次前向,同时取分类器 logits 与原型特征 ════════
        # 方式1(w/o protos):纯分类器 sigmoid(logits)
        # 方式2(w/ protos) :分类器 logit + 原型马氏 logit 融合 → mean-field 解码
        model.eval()
        total_l, total_g = 0.0, 0.0
        correct_l, correct_g = 0.0, 0.0
        diag_loss = 0.0
        _diag_printed = False
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            logits, mu, logvar = output[0], output[1], output[2]
            proto_feats = mu                       # 用分布均值作为查询特征
            B = images.shape[0]

            # 逐样本逐类:对角马氏距离 → 独立原型 logit s_c
            s = torch.zeros(B, C, device=device)
            for j in range(C):
                if j in global_protos:
                    g_mu, g_logvar = global_protos[j]
                    g_var = torch.exp(g_logvar) + 1e-8
                    # e_j = 0.5 * Σ_d (x_d − μ_jd)² / σ²_jd  (B,)
                    e_j = 0.5 * (((proto_feats - g_mu) ** 2) / g_var).sum(dim=1)
                    s[:, j] = -e_j / temperature
                    diag_loss += e_j.mean().item()
                else:
                    # 该类无全局原型(首尾轮/未见类),回退用分类器 logit
                    s[:, j] = logits[:, j]

            # 首客户端首批:打印 s 量级,诊断 temperature 是否使原型 logit 饱和
            if not _diag_printed and idx == 0:
                with torch.no_grad():
                    _sig_s = torch.sigmoid(s).mean().item()
                    _sig_l = torch.sigmoid(logits).mean().item()
                print(f'  [decode-diag] mean σ(s_proto)={_sig_s:.4f} '
                      f'mean σ(logit_cls)={_sig_l:.4f} '
                      f'(若 σ(s)≈0或1 → 原型 logit 饱和,需调大 temperature)')
                _diag_printed = True

            # ★ 融合:fused = α·logit_cls + (1−α)·s_proto
            fused = fuse_alpha * logits + (1.0 - fuse_alpha) * s

            # 相关性感知 mean-field 解码(R̂=I 时退化为独立 sigmoid)
            q = mean_field_decode(fused, R_use, pi_use, beta=beta, steps=mf_steps)

            # 方式1:纯分类器
            preds_l = (torch.sigmoid(logits) > 0.5).float()
            correct_l += (preds_l == labels).float().sum().item()
            total_l += labels.numel()

            # 方式2:融合 + mean-field
            preds_g = (q > 0.5).float()
            correct_g += (preds_g == labels).float().sum().item()
            total_g += labels.numel()

            all_probs_g.append(q.detach().cpu())
            all_labels_g.append(labels.detach().cpu())

        acc_list_l.append(correct_l / max(total_l, 1))
        acc_list_g.append(correct_g / max(total_g, 1))
        loss_list.append(diag_loss / max(len(testloader), 1))

        print('| User: {} | Test Acc w/o protos (per-label): {:.4f} | '
              'w/ protos(fused+structured): {:.4f}'.format(
                  idx, acc_list_l[-1], acc_list_g[-1]))

    # ── 原型解码路径的完整多标签指标 ──
    probs_all = torch.cat(all_probs_g, dim=0).numpy()
    labels_all = torch.cat(all_labels_g, dim=0).numpy()
    metrics_g = compute_multilabel_metrics(probs_all, labels_all, num_classes=C)
    print('[FedCoP prototype-decode metrics] ' + format_metrics(metrics_g))

    return acc_list_l, acc_list_g, loss_list, metrics_g

# ═══════════════════════════════════════════════════════════════════════════════
#  FedGMKD (NeurIPS 2024): GMM-based Prototype Federated Learning
#  对比基线，核心区别于 FedCoP:
#    - GMM 后处理拟合（EM 算法在 detach 特征上运行），而非端到端 NN 输出
#    - 多分量高斯原型（每类 K 个高斯分量）vs 单高斯 + 校准
#    - Discrepancy-Aware Aggregation (质量+数量联合加权) vs Bayesian Fusion
# ═══════════════════════════════════════════════════════════════════════════════

def _fit_gmm_pytorch(features, n_components=3, n_iters=10):
    """简易 EM 算法拟合对角高斯混合模型（纯 PyTorch 实现，无需 sklearn）

    这是 FedGMKD 的核心后处理步骤：在 detach 的特征上做 GMM 拟合，
    得到多峰高斯原型 (weights, means, logvars)。

    和 FedCoP 的端到端方式的本质区别：
    - FedGMKD: 先提取特征（detach）→ 后处理 GMM → 上传 GMM 参数
    - FedCoP: 网络直接输出 (μ, σ²) → 端到端反向传播

    参数:
        features:      (N, D) 特征矩阵
        n_components:  高斯分量数。GMM 可以捕捉类内的多峰分布。
        n_iters:       EM 迭代次数

    返回:
        weights:  (K,) 混合权重
        means:    (K, D) 各分量均值
        logvars:  (K, D) 各分量对数方差（对角协方差）
    """
    N, D = features.shape
    K = min(n_components, N)  # 分量数不超过样本数
    device = features.device

    # 随机选取 K 个样本作为初始聚类中心
    idxs = torch.randperm(N)[:K]
    means = features[idxs].clone()
    logvars = torch.zeros(K, D, device=device)
    weights = torch.ones(K, device=device) / K

    for _ in range(n_iters):
        # E 步：计算每个样本属于每个分量的后验概率（责任度）
        vars_ = torch.exp(logvars) + 1e-8
        diff = features.unsqueeze(0) - means.unsqueeze(1)     # (K, N, D)
        log_prob = -0.5 * (torch.log(2 * torch.pi * vars_).sum(dim=1, keepdim=True)
                           + (diff ** 2 / vars_.unsqueeze(1)).sum(dim=2))
        log_prob = log_prob + torch.log(weights.unsqueeze(1))  # 加入先验
        # 数值稳定：log-sum-exp 技巧
        log_prob_max = log_prob.max(dim=0, keepdim=True)[0]
        log_sum = log_prob_max + torch.log(
            torch.exp(log_prob - log_prob_max).sum(dim=0, keepdim=True) + 1e-8)
        responsibilities = torch.exp(log_prob - log_sum)       # (K, N)

        # M 步：更新 GMM 参数
        nk = responsibilities.sum(dim=1) + 1e-8               # (K,) 每个分量的有效样本数
        weights = nk / N                                        # 混合权重
        means = (responsibilities.unsqueeze(2) * features.unsqueeze(0)).sum(dim=1) / nk.unsqueeze(1)
        diff_new = features.unsqueeze(0) - means.unsqueeze(1)
        vars_new = (responsibilities.unsqueeze(2) * (diff_new ** 2)).sum(dim=1) / nk.unsqueeze(1)
        logvars = torch.log(vars_new + 1e-8)

    return weights, means, logvars


def _update_weights_FedGMKD(self, args, idx, global_protos, model, global_round=0, ld=None):
    """FedGMKD 本地训练：GMM 后处理原型 + 质量感知聚合

    损失 = L_CE + ld × L_gmm_proto
    和 FedCoP 的关键区别：GMM 在 detach 特征上用 EM 拟合，不可端到端学习。
    """
    if ld is None:
        ld = args.ld
    model.train()
    epoch_loss = {'total': [], '1': [], '2': []}
    use_dist = getattr(args, 'use_distributional', True)

    if self.args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
    elif self.args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

    for iter in range(self.args.train_ep):
        batch_loss = {'total': [], '1': [], '2': []}
        agg_protos_label = {}

        for batch_idx, (images, label_g) in enumerate(self.trainloader):
            images, labels = images.to(self.device), label_g.to(self.device)
            model.zero_grad()
            output = model(images)

            if use_dist:
                logits, mu, logvar = output
                proto_feats = mu
            else:
                logits, protos = output
                proto_feats = protos

            loss1 = self.criterion(logits, labels)

            loss_mse = nn.MSELoss()
            if len(global_protos) == 0:
                loss2 = 0 * loss1
            else:
                loss2 = 0.0
                count = 0
                for i_lbl in range(len(labels)):
                    for lbl_idx in range(args.num_classes):
                        if labels[i_lbl, lbl_idx] > 0 and lbl_idx in global_protos:
                            g_val = global_protos[lbl_idx]
                            if isinstance(g_val, tuple) and len(g_val) == 3:
                                # GMM 原型：取最近分量的距离
                                _gw, g_means, _glv = g_val
                                l_feat = proto_feats[i_lbl:i_lbl + 1]
                                dists = ((l_feat - g_means) ** 2).sum(dim=1)
                                loss2 += dists.min()
                            elif isinstance(g_val, tuple):
                                g_mu, _glv = g_val
                                loss2 += loss_mse(proto_feats[i_lbl:i_lbl + 1],
                                                  g_mu.unsqueeze(0))
                            else:
                                loss2 += loss_mse(proto_feats[i_lbl:i_lbl + 1],
                                                  g_val.unsqueeze(0))
                            count += 1
                loss2 = loss2 / max(count, 1)

            loss = loss1 + loss2 * ld
            loss.backward()
            optimizer.step()

            for i_lbl in range(len(labels)):
                for lbl_idx in range(args.num_classes):
                    if label_g[i_lbl, lbl_idx] > 0:
                        if use_dist:
                            proto_val = (proto_feats[i_lbl, :].detach(),
                                         logvar[i_lbl, :].detach())
                        else:
                            proto_val = proto_feats[i_lbl, :].detach()
                        if lbl_idx in agg_protos_label:
                            agg_protos_label[lbl_idx].append(proto_val)
                        else:
                            agg_protos_label[lbl_idx] = [proto_val]

            preds = (torch.sigmoid(logits) > 0.5).float()
            acc_val = (preds == labels).float().mean()

            batch_loss['total'].append(loss.item())
            batch_loss['1'].append(loss1.item())
            batch_loss['2'].append(loss2.item() if isinstance(loss2, torch.Tensor) else loss2)

        for key in epoch_loss:
            if len(batch_loss[key]) > 0:
                epoch_loss[key].append(sum(batch_loss[key]) / len(batch_loss[key]))

    for key in epoch_loss:
        if len(epoch_loss[key]) > 0:
            epoch_loss[key] = sum(epoch_loss[key]) / len(epoch_loss[key])

    return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label


def _agg_func_FedGMKD(protos, n_components=3):
    """FedGMKD 本地聚合：对每类特征拟合 GMM → (weights, means, logvars)

    这是 FedGMKD 区别于 FedCoP 的核心后处理步骤。
    """
    agg = {}
    for label, proto_list in protos.items():
        if isinstance(proto_list[0], tuple):
            feats = torch.stack([p[0] for p in proto_list])
        else:
            feats = torch.stack(proto_list)

        device = feats.device
        if feats.shape[0] >= n_components:
            weights, means, logvars = _fit_gmm_pytorch(
                feats, n_components=n_components, n_iters=10)
            agg[label] = (weights.detach(), means.detach(), logvars.detach())
        else:
            # 样本不足时退化为单高斯
            mu_avg = feats.mean(dim=0)
            logvar_avg = torch.log(feats.var(dim=0, unbiased=False) + 1e-8)
            agg[label] = (torch.ones(1, device=device),
                          mu_avg.unsqueeze(0), logvar_avg.unsqueeze(0))
    return agg


def _proto_aggregation_FedGMKD(local_protos_list):
    """FedGMKD 全局聚合：Discrepancy-Aware Aggregation

    质量分数 quality_k = 1 / mean(variance)（方差越小 → 质量越高）
    然后按质量 + 数量联合加权融合。
    """
    agg_pool = {}
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label, entry in local_protos.items():
            if label not in agg_pool:
                agg_pool[label] = []
            agg_pool[label].append(entry)

    global_protos = {}
    for label, proto_list in agg_pool.items():
        if len(proto_list) == 1:
            global_protos[label] = proto_list[0]
        else:
            all_w, all_m, all_lv, qualities = [], [], [], []
            for entry in proto_list:
                w, m, lv = entry
                all_w.append(w); all_m.append(m); all_lv.append(lv)
                q = 1.0 / (torch.exp(lv).mean() + 1e-8)  # 质量 = 1/平均方差
                qualities.append(q)
            device = all_m[0].device

            q_t = torch.stack(qualities)
            q_sum = q_t.sum() + 1e-8

            # 各客户端 GMM 分量数 K 可能不同(样本不足者退化为单高斯 K=1)。
            # 按 max K 零填充后再 stack,避免形状不匹配;零权分量对加权和无影响。
            k_max = max(w.shape[0] for w in all_w)
            def _pad_k(t):
                if t.shape[0] == k_max:
                    return t
                pad_shape = list(t.shape)
                pad_shape[0] = k_max - t.shape[0]
                return torch.cat([t, torch.zeros(pad_shape, device=t.device, dtype=t.dtype)], dim=0)
            all_w = [_pad_k(w) for w in all_w]
            all_m = [_pad_k(m) for m in all_m]
            all_lv = [_pad_k(lv) for lv in all_lv]

            # 质量加权融合
            fused_w = torch.stack([w * q for w, q in zip(all_w, qualities)]).sum(dim=0) / q_sum
            fused_m = torch.stack([m * q for m, q in zip(all_m, qualities)]).sum(dim=0) / q_sum
            fused_lv = torch.stack([lv * q for lv, q in zip(all_lv, qualities)]).sum(dim=0) / q_sum
            global_protos[label] = (fused_w, fused_m, fused_lv)

    return global_protos



# ═══════════════════════════════════════════════════════════════════════════════
#  FedSeProto (ECAI 2024): Semantic-Domain Feature Decoupling
#  对比基线，核心区别于 FedCoP:
#    - 硬分割（独立 MLP 编码器，前 N 维语义 + 后 M 维域）vs 软门控
#    - 仅 HSIC 互信息最小化 vs HSIC + 门控熵 + 正交 + 对抗 + 对比
#    - 点原型 vs 分布原型
# ═══════════════════════════════════════════════════════════════════════════════

class FedSeProtoHeads(nn.Module):
    """FedSeProto 语义-域解耦双头(硬分割方式)

    FedSeProto 基线自带组件:两个独立 MLP '硬'分成语义和域 → sem_dim 固定。
    (FedCoP 不使用解耦,其跨类结构由共现相关矩阵 R̂ 建模。)
    """

    def __init__(self, proto_dim=256, sem_ratio=0.75):
        super().__init__()
        sem_dim = int(proto_dim * sem_ratio)  # 语义维度（固定）
        dom_dim = proto_dim - sem_dim          # 域维度（固定）
        self.semantic_head = nn.Sequential(
            nn.Linear(proto_dim, proto_dim), nn.ReLU(inplace=True),
            nn.Linear(proto_dim, sem_dim))
        self.domain_head = nn.Sequential(
            nn.Linear(proto_dim, proto_dim), nn.ReLU(inplace=True),
            nn.Linear(proto_dim, dom_dim))
        self.sem_dim = sem_dim
        self.dom_dim = dom_dim

    def forward(self, x):
        return self.semantic_head(x), self.domain_head(x)


def _mi_minimization_loss(z_sem, z_dom):
    """HSIC 互信息最小化 — FedSeProto 的解耦损失

    FedSeProto 基线自带:仅 HSIC(交叉协方差 Frobenius 范数)。
    """
    n = z_sem.size(0)
    if n < 2:
        return torch.tensor(0.0, device=z_sem.device)
    z_sem_c = z_sem - z_sem.mean(dim=0, keepdim=True)
    z_dom_c = z_dom - z_dom.mean(dim=0, keepdim=True)
    cross_cov = torch.mm(z_sem_c.T, z_dom_c) / (n - 1)
    mi_loss = torch.sum(cross_cov ** 2)
    var_reg = F.relu(0.01 - z_sem_c.var(dim=0).mean()) + F.relu(0.01 - z_dom_c.var(dim=0).mean())
    return mi_loss + 0.1 * var_reg


def _update_weights_FedSeProto(self, args, idx, global_protos, model, global_round=0, ld=None):
    """FedSeProto 本地训练：语义-域解耦 + 仅共享语义原型

    损失 = L_CE + ld × L_proto(semantic only) + mi_lambda × L_MI(HSIC)

    关键区别于 FedCoP：
    - 仅共享语义原型（点原型），域特征完全本地
    - 仅 HSIC 互信息最小化（无门控熵、无正交、无对抗、无对比）
    """
    if ld is None:
        ld = args.ld
    model.train()
    epoch_loss = {'total': [], '1': [], '2': [], '3': []}
    mi_lambda = getattr(args, 'mi_lambda', 0.05)
    proto_dim = getattr(args, 'proto_dim', 256) or 256
    sem_ratio = getattr(args, 'sem_ratio', 0.75)

    if not hasattr(self, 'seproto_heads'):
        self.seproto_heads = FedSeProtoHeads(proto_dim, sem_ratio).to(self.device)

    if self.args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(self.seproto_heads.parameters()),
            lr=self.args.lr, momentum=0.5)
    elif self.args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.seproto_heads.parameters()),
            lr=self.args.lr, weight_decay=1e-4)

    for iter in range(self.args.train_ep):
        batch_loss = {'total': [], '1': [], '2': [], '3': []}
        agg_protos_label = {}

        for batch_idx, (images, label_g) in enumerate(self.trainloader):
            images, labels = images.to(self.device), label_g.to(self.device)
            model.zero_grad()
            output = model(images)

            logits, proto_features = output
            z_sem, z_dom = self.seproto_heads(proto_features)  # 硬分割

            loss1 = self.criterion(logits, labels)
            loss3 = _mi_minimization_loss(z_sem, z_dom)  # HSIC 互信息最小化

            loss_mse = nn.MSELoss()
            if len(global_protos) == 0:
                loss2 = 0 * loss1
            else:
                loss2 = 0.0
                count = 0
                for i_lbl in range(len(labels)):
                    for lbl_idx in range(args.num_classes):
                        if labels[i_lbl, lbl_idx] > 0 and lbl_idx in global_protos:
                            g_val = global_protos[lbl_idx]
                            if isinstance(g_val, tuple):
                                g_val = g_val[0]
                            loss2 += loss_mse(z_sem[i_lbl:i_lbl + 1], g_val.unsqueeze(0))
                            count += 1
                loss2 = loss2 / max(count, 1)

            loss = loss1 + loss2 * ld + loss3 * mi_lambda
            loss.backward()
            optimizer.step()

            # 仅上传语义原型（域特征保留本地）
            for i_lbl in range(len(labels)):
                for lbl_idx in range(args.num_classes):
                    if label_g[i_lbl, lbl_idx] > 0:
                        if lbl_idx in agg_protos_label:
                            agg_protos_label[lbl_idx].append(z_sem[i_lbl, :].detach())
                        else:
                            agg_protos_label[lbl_idx] = [z_sem[i_lbl, :].detach()]

            preds = (torch.sigmoid(logits) > 0.5).float()
            acc_val = (preds == labels).float().mean()

            batch_loss['total'].append(loss.item())
            batch_loss['1'].append(loss1.item())
            batch_loss['2'].append(loss2.item() if isinstance(loss2, torch.Tensor) else loss2)
            batch_loss['3'].append(loss3.item() if isinstance(loss3, torch.Tensor) else loss3)

        for key in epoch_loss:
            if len(batch_loss[key]) > 0:
                epoch_loss[key].append(sum(batch_loss[key]) / len(batch_loss[key]))

    for key in epoch_loss:
        if len(epoch_loss[key]) > 0:
            epoch_loss[key] = sum(epoch_loss[key]) / len(epoch_loss[key])

    return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label
