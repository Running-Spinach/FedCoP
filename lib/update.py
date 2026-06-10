# 功能：本地更新模块，实现FedProto联邦学习中的本地训练、测试和原型提取

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
                                 prototype_calibration_loss,
                                 entropy_regularization)
from dist_proto.disentangle import (disentanglement_loss,
                                     contrastive_semantic_loss,
                                     adversarial_disentanglement_loss)


class DatasetSplit(Dataset):
    """
    数据集分割类，根据给定索引从完整数据集中提取子集
    """

    def __init__(self, dataset, idxs):
        """
        初始化数据集分割对象

        参数:
            dataset: 完整数据集
            idxs: 要提取的索引列表
        """
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        """返回子数据集样本总数"""
        return len(self.idxs)

    def __getitem__(self, item):
        """
        获取指定索引的数据样本

        参数:
            item: 子数据集中的索引

        返回:
            (image, label): 图像和标签的张量元组
        """
        image, label = self.dataset[self.idxs[item]]
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        return image, label


class LocalUpdate(object):
    """
    本地更新类，封装单个客户端的本地训练逻辑
    """

    def __init__(self, args, dataset, idxs):
        """
        初始化本地更新对象

        参数:
            args: 配置参数
            dataset: 训练数据集
            idxs: 该客户端对应的数据索引
        """
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        根据数据集和索引构建训练数据加载器

        参数:
            dataset: 数据集
            idxs: 数据索引列表

        返回:
            trainloader: 训练数据加载器
        """
        idxs_train = idxs[:int(1 * len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True)

        return trainloader

    def _get_optimizer(self, model):
        """根据配置返回 SGD 或 Adam 优化器"""
        if self.args.optimizer == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            return torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

    def update_weights(self, idx, model, global_round):
        """
        标准本地模型权重更新（FedAvg方式 / FedBN方式）

        参数:
            idx: 客户端索引
            model: 当前模型
            global_round: 全局训练轮次

        返回:
            model.state_dict(): 更新后的模型状态字典
            平均损失值
            per-label 准确率
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
                logits = output[0] if isinstance(output, tuple) else output
                loss = self.criterion(logits, labels)

                loss.backward()
                optimizer.step()

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
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), acc_val.item()

    def update_weights_FedP(self, args, idx, global_protos, model, global_round=round, ld=None):
        if ld is None:
            ld = args.ld
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[], '3':[]}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'total':[],'1':[], '2':[], '3':[]}
            agg_protos_label = {}#
            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)

                # loss1: cross-entrophy loss, loss2: proto distance loss
                model.zero_grad()
                logits, protos = model(images)
                loss1 = self.criterion(logits, labels)

                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = 0*loss1
                else:
                    # 多标签原型损失：对每个样本的所有正标签计算 MSE
                    loss2 = 0*loss1
                    count = 0
                    for i_lbl in range(len(labels)):
                        proto_i = protos[i_lbl]
                        for lbl_idx in range(args.num_classes):
                            if labels[i_lbl, lbl_idx] > 0 and lbl_idx in global_protos:
                                loss2 += loss_mse(proto_i, global_protos[lbl_idx])
                                count += 1
                    loss2 = loss2 / max(count, 1)

                loss = loss1 + loss2 * ld
                loss.backward()
                optimizer.step()

                # 多标签原型聚合：每个样本的原型归属到其所有正标签
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
            epoch_loss['total'].append(sum(batch_loss['total'])/len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

    def update_weights_fedprox(self, idx, global_model_state, model, global_round):
        """
        FedProx 本地训练：L = L_CE + (mu/2) * ||w - w_global||^2

        参数:
            idx: 客户端索引
            global_model_state: 全局模型 state_dict
            model: 当前本地模型
            global_round: 全局训练轮次

        返回:
            model.state_dict(), avg_loss, acc_val
        """
        model.train()
        epoch_loss = []
        mu = self.args.fedprox_mu
        optimizer = self._get_optimizer(model)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels_g.to(self.device)

                model.zero_grad()
                output = model(images)
                logits = output[0] if isinstance(output, tuple) else output
                loss_ce = self.criterion(logits, labels)

                # FedProx 近端项：L2 距离（仅对可训练参数）
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

    def update_weights_scaffold(self, idx, c_global, c_local, model, global_round):
        """
        SCAFFOLD 本地训练：梯度修正 g_corr = g - c_local + c_global

        参数:
            idx: 客户端索引
            c_global: 全局 control variate (state_dict格式)
            c_local: 本地 control variate (state_dict格式)
            model: 当前本地模型
            global_round: 全局训练轮次

        返回:
            model.state_dict(), avg_loss, acc_val, c_local_new, c_delta
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

                # SCAFFOLD 梯度修正 + 参数更新
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
        # c_i_new = c_i - c + (w_global_init - w_local_final) / (lr * K)
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

    def update_weights_D2FL(self, args, idx, global_protos, model, global_round=round, ld=None):
        if ld is None:
            ld = args.ld
        """
        D²-FL 增强版本地训练：端到端分布原型 + 语义-风格解耦

        损失函数（7项）：
          L_total = L_CE            (分类)
                  + λ    * L_proto  (分布原型对齐)
                  + λ_dis * L_dis   (解耦独立性: HSIC + 门控熵 + 正交)
                  + λ_cal * L_cal   (原型校准: logvar ≅ log(distance))
                  + λ_ctr * L_contra (对比语义对齐: 同类拉近/异类推远)
                  + λ_adv * L_adv   (对抗域不变: 语义不应含域信息)
                  + λ_ent * L_ent   (熵正则: 防止方差坍缩)

        参数:
            args: 配置参数
            idx: 客户端索引
            global_protos: 全局原型字典
            model: 当前模型 (D2FLResNet)
            global_round: 全局训练轮次

        返回:
            model.state_dict(): 更新后的模型状态字典
            epoch_loss: 包含total/1/2/3/cal/contra/adv/ent的损失字典
            acc_val.item(): 准确率
            agg_protos_label: 聚合后的本地原型字典
        """
        model.train()
        epoch_loss = {'total': [], '1': [], '2': [], '3': [],
                       'cal': [], 'contra': [], 'adv': [], 'ent': []}

        use_dist = getattr(args, 'use_distributional', False)
        use_dis = getattr(args, 'use_disentangle', False)
        dis_lambda = getattr(args, 'dis_lambda', 0.05)
        cal_lambda = getattr(args, 'cal_lambda', 0.01)
        contra_lambda = getattr(args, 'contra_lambda', 0.05)
        adv_lambda = getattr(args, 'adv_lambda', 0.01)
        ent_lambda = getattr(args, 'ent_lambda', 0.001)

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'total': [], '1': [], '2': [], '3': [],
                           'cal': [], 'contra': [], 'adv': [], 'ent': []}
            agg_protos_label = {}

            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)

                model.zero_grad()
                # 解耦模式下请求门控值，用于门控熵正则
                output = model(images, return_gate=use_dis)

                # ── 解析模型输出 ──
                gate = None
                if use_dis and use_dist:
                    # 增强解耦 + 分布: (logits, mu_full, logvar_full,
                    #                     mu_sem, logvar_sem, mu_style, logvar_style, gate)
                    (logits, _mu_full, _logvar_full,
                     mu_sem, logvar_sem, mu_style, logvar_style, gate) = output
                    proto_for_share = (mu_sem, logvar_sem)
                    proto_for_style = (mu_style, logvar_style)
                elif use_dis:
                    # 增强解耦 + 点原型: (logits, z_full, z_sem, z_style, gate)
                    logits, _z_full, z_sem, z_style, gate = output
                    proto_for_share = z_sem
                    proto_for_style = z_style
                elif use_dist:
                    logits, mu, logvar = output
                    proto_for_share = (mu, logvar)
                else:
                    logits, protos = output
                    proto_for_share = protos

                loss1 = self.criterion(logits, labels)

                # ── 核心创新 1: 增强解耦损失 L_dis ──
                # HSIC 独立性 + 门控熵（可学习门控） + 正交约束
                if use_dis:
                    if use_dist:
                        loss3 = disentanglement_loss(
                            mu_sem, mu_style,
                            gate=gate,
                            proto_dim=getattr(args, 'proto_dim', 256) or 256
                        )
                    else:
                        loss3 = disentanglement_loss(
                            z_sem, z_style,
                            gate=gate,
                            proto_dim=getattr(args, 'proto_dim', 256) or 256
                        )
                else:
                    loss3 = 0.0 * loss1

                # ── 分布原型正则化损失 L_proto ──
                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = 0 * loss1
                else:
                    if use_dist and use_dis:
                        loss2 = 0.0
                        count = 0
                        for i_lbl in range(len(labels)):
                            for lbl_idx in range(args.num_classes):
                                if labels[i_lbl, lbl_idx] > 0 and lbl_idx in global_protos:
                                    g_val = global_protos[lbl_idx]
                                    g_mu, g_logvar = (g_val if isinstance(g_val, tuple)
                                                      else (g_val, torch.zeros_like(g_val)))
                                    l2 = distributional_proto_loss(
                                        mu_sem[i_lbl:i_lbl + 1], logvar_sem[i_lbl:i_lbl + 1],
                                        g_mu.unsqueeze(0), g_logvar.unsqueeze(0),
                                        dist_type=args.dist_type
                                    )
                                    loss2 += l2
                                    count += 1
                        loss2 = loss2 / max(count, 1)
                    elif use_dis:
                        loss2 = 0.0
                        count = 0
                        for i_lbl in range(len(labels)):
                            proto_i = z_sem[i_lbl]
                            for lbl_idx in range(args.num_classes):
                                if labels[i_lbl, lbl_idx] > 0 and lbl_idx in global_protos:
                                    loss2 += loss_mse(proto_i, global_protos[lbl_idx])
                                    count += 1
                        loss2 = loss2 / max(count, 1)
                    elif use_dist:
                        loss2 = 0.0
                        count = 0
                        for i_lbl in range(len(labels)):
                            for lbl_idx in range(args.num_classes):
                                if labels[i_lbl, lbl_idx] > 0 and lbl_idx in global_protos:
                                    g_mu, g_logvar = global_protos[lbl_idx]
                                    l2 = distributional_proto_loss(
                                        mu[i_lbl:i_lbl + 1], logvar[i_lbl:i_lbl + 1],
                                        g_mu.unsqueeze(0), g_logvar.unsqueeze(0),
                                        dist_type=args.dist_type
                                    )
                                    loss2 += l2
                                    count += 1
                        loss2 = loss2 / max(count, 1)
                    else:
                        loss2 = 0.0
                        count = 0
                        for i_lbl in range(len(labels)):
                            proto_i = protos[i_lbl]
                            for lbl_idx in range(args.num_classes):
                                if labels[i_lbl, lbl_idx] > 0 and lbl_idx in global_protos:
                                    loss2 += loss_mse(proto_i, global_protos[lbl_idx])
                                    count += 1
                        loss2 = loss2 / max(count, 1)

                # ── 核心创新 1+: 原型校准损失 L_cal ──
                # 鼓励 logvar 反映与全局原型的实际距离
                if use_dist and cal_lambda > 0 and len(global_protos) > 0:
                    if use_dis:
                        loss_cal = prototype_calibration_loss(
                            mu_sem, logvar_sem, labels, global_protos,
                            dist_type=args.dist_type, num_classes=args.num_classes
                        )
                    else:
                        loss_cal = prototype_calibration_loss(
                            mu, logvar, labels, global_protos,
                            dist_type=args.dist_type, num_classes=args.num_classes
                        )
                else:
                    loss_cal = 0.0 * loss1

                # ── 核心创新 2+: 对比语义对齐损失 L_contra ──
                # 同类语义特征在语义空间中聚集，异类分散
                if use_dis and contra_lambda > 0:
                    if use_dist:
                        loss_contra = contrastive_semantic_loss(mu_sem, labels)
                    else:
                        loss_contra = contrastive_semantic_loss(z_sem, labels)
                else:
                    loss_contra = 0.0 * loss1

                # ── 核心创新 2++: 对抗域不变损失 L_adv ──
                # 语义特征经过梯度反转后不应被域分类器识别
                if use_dis and adv_lambda > 0:
                    if use_dist:
                        domain_logits = model.forward_adversarial(
                            mu_sem, grad_reverse_lambda=1.0)
                    else:
                        domain_logits = model.forward_adversarial(
                            z_sem, grad_reverse_lambda=1.0)
                    if domain_logits is not None:
                        loss_adv = adversarial_disentanglement_loss(domain_logits)
                    else:
                        loss_adv = 0.0 * loss1
                else:
                    loss_adv = 0.0 * loss1

                # ── 熵正则 L_ent：防止方差坍缩回点原型 ──
                if use_dist and ent_lambda > 0:
                    if use_dis:
                        loss_ent = entropy_regularization(logvar_sem)
                    else:
                        loss_ent = entropy_regularization(logvar)
                else:
                    loss_ent = 0.0 * loss1

                # ── 总损失 ──
                loss = (loss1
                        + loss2 * ld
                        + loss3 * dis_lambda
                        + loss_cal * cal_lambda
                        + loss_contra * contra_lambda
                        + loss_adv * adv_lambda
                        + loss_ent * ent_lambda)

                loss.backward()
                optimizer.step()

                # ── 按标签聚合本地原型（解耦模式仅聚合语义原型）──
                if use_dis:
                    if use_dist:
                        sem_feat = mu_sem
                    else:
                        sem_feat = z_sem
                    for i_lbl in range(len(labels)):
                        for lbl_idx in range(args.num_classes):
                            if label_g[i_lbl, lbl_idx] > 0:
                                if use_dist:
                                    proto_val = (sem_feat[i_lbl, :].detach(),
                                                 logvar_sem[i_lbl, :].detach())
                                else:
                                    proto_val = sem_feat[i_lbl, :].detach()
                                if lbl_idx in agg_protos_label:
                                    agg_protos_label[lbl_idx].append(proto_val)
                                else:
                                    agg_protos_label[lbl_idx] = [proto_val]
                else:
                    for i_lbl in range(len(labels)):
                        for lbl_idx in range(args.num_classes):
                            if label_g[i_lbl, lbl_idx] > 0:
                                if use_dist:
                                    proto_val = (mu[i_lbl, :].detach(),
                                                 logvar[i_lbl, :].detach())
                                else:
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

                # 记录各损失分量
                def _val(v):
                    return v.item() if isinstance(v, torch.Tensor) else v

                batch_loss['total'].append(loss.item())
                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(_val(loss2))
                batch_loss['3'].append(_val(loss3))
                batch_loss['cal'].append(_val(loss_cal))
                batch_loss['contra'].append(_val(loss_contra))
                batch_loss['adv'].append(_val(loss_adv))
                batch_loss['ent'].append(_val(loss_ent))

            # ── Epoch 级别损失平均 ──
            for key in epoch_loss:
                if len(batch_loss[key]) > 0:
                    epoch_loss[key].append(
                        sum(batch_loss[key]) / len(batch_loss[key]))

        # ── 全局平均 ──
        for key in epoch_loss:
            if len(epoch_loss[key]) > 0:
                epoch_loss[key] = sum(epoch_loss[key]) / len(epoch_loss[key])

        return model.state_dict(), epoch_loss, acc_val.item(), agg_protos_label

    def inference(self, model):
        """
        在测试集上推理，返回准确率和损失

        参数:
            model: 待评估的模型

        返回:
            accuracy: 准确率
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

def eval_clients_multilabel(args, local_model_list, test_dataset, user_groups_gt):
    """
    多标签联邦学习统一评估（FedAvg / FedProx / FedBN / SCAFFOLD 共用）

    每个客户端用 sigmoid(logits) > 0.5 对本地测试集做 per-label 准确率

    返回:
        acc_list: 各客户端 per-label 准确率列表
    """
    device = args.device
    acc_list = []

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

                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (preds == labels).float().sum().item()
                total_val += labels.numel()

        acc = correct_val / total_val
        print('| User: {} | Test Acc (per-label): {:.4f}'.format(idx, acc))
        acc_list.append(acc)

    return acc_list

def test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[]):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0
    loss_mse = nn.MSELoss()

    device = args.device
    criterion = nn.NLLLoss().to(device)

    acc_list_g = []
    acc_list_l = []
    loss_list = []
    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(args.device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]), batch_size=64, shuffle=True)

        # test (local model)
        model.eval()
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs, protos = model(images)

            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        print('| User: {} | Global Test Acc w/o protos: {:.3f}'.format(idx, acc))
        acc_list_l.append(acc)

        # test (use global proto)
        if global_protos!=[]:
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                outputs, protos = model(images)

                # compute the dist between protos and global_protos
                a_large_num = 100
                dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
                for i in range(images.shape[0]):
                    for j in range(args.num_classes):
                        if j in global_protos.keys() and j in classes_list[idx]:
                            d = loss_mse(protos[i, :], global_protos[j][0])
                            dist[i, j] = d

                # prediction
                _, pred_labels = torch.min(dist, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                # compute loss
                proto_new = copy.deepcopy(protos.data)
                i = 0
                for label in labels:
                    if label.item() in global_protos.keys():
                        proto_new[i, :] = global_protos[label.item()][0].data
                    i += 1
                loss2 = loss_mse(proto_new, protos)
                if args.device == 'cuda':
                    loss2 = loss2.cpu().detach().numpy()
                else:
                    loss2 = loss2.detach().numpy()

            acc = correct / total
            print('| User: {} | Global Test Acc with protos: {:.5f}'.format(idx, acc))
            acc_list_g.append(acc)
            loss_list.append(loss2)

    return acc_list_l, acc_list_g, loss_list

def test_inference_new_het_lt_D2FL(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[], temperature=None):
    """多标签联邦学习测试：分别评估模型自身分类和原型最近邻分类的效果

    参数:
        temperature: 原型推理温度系数。None 时自动从 args 读取，默认 1.0
            - T > 1: 软化概率（更平滑）
            - T < 1: 锐化概率（更确信）
            - 仅 FedProto / D²-FL 有效

    返回:
        acc_list_l: 各客户端用自身模型（sigmoid阈值）的 per-label 准确率
        acc_list_g: 各客户端用全局原型距离分类的 per-label 准确率
        loss_list: 各客户端的原型损失
    """
    loss_mse = nn.MSELoss()
    use_dist = getattr(args, 'use_distributional', False)
    use_dis = getattr(args, 'use_disentangle', False)
    if temperature is None:
        temperature = getattr(args, 'temperature', 1.0)
    device = args.device

    def _extract_proto_feat(output):
        """从模型输出中提取用于原型比对的语义特征向量

        增强版 D2FLResNet 输出格式（return_gate=False）:
          解耦+分布: (logits, mu_full, logvar_full, mu_sem, logvar_sem, mu_style, logvar_style)
          解耦+点:   (logits, z_full, z_sem, z_style)
          分布:      (logits, mu, logvar)
          点:        (logits, proto_features)
        """
        if use_dis and use_dist:
            # 7-tuple: output[3] = mu_sem (语义均值 → 用于原型比对)
            return output[3]
        elif use_dis:
            # 4-tuple: output[2] = z_sem (语义特征 → 用于原型比对)
            return output[2]
        elif use_dist and len(output) >= 3:
            # 3-tuple: output[1] = mu
            return output[1]
        else:
            # 2-tuple: output[1] = proto_features
            return output[1]

    acc_list_g = []
    acc_list_l = []
    loss_list = []

    for idx in range(args.num_users):
        model = local_model_list[idx]
        model.to(device)
        testloader = DataLoader(DatasetSplit(test_dataset, user_groups_gt[idx]),
                                batch_size=64, shuffle=True)

        # ── 不使用全局原型的分类测试：sigmoid(logits) > 0.5 ──
        model.eval()
        total_val, correct_val = 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            output = model(images)
            outputs = output[0] if isinstance(output, tuple) else output

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_val += (preds == labels).float().sum().item()
            total_val += labels.numel()

        acc = correct_val / total_val
        print('| User: {} | Test Acc w/o protos (per-label): {:.4f}'.format(idx, acc))
        acc_list_l.append(acc)

        # ── 使用全局原型的最近原型分类测试 ──
        model.eval()
        total_val, correct_val = 0.0, 0.0
        loss2 = torch.tensor(0.0)
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            output = model(images)

            outputs = output[0]
            proto_feats = _extract_proto_feat(output)

            # 计算到每个全局原型的距离 → 负距离/T 作为 logit → sigmoid → 二值预测
            proto_logits = torch.zeros(images.shape[0], args.num_classes, device=device)
            for i in range(images.shape[0]):
                for j in range(args.num_classes):
                    if j in global_protos:
                        if use_dist:
                            g_mu, g_logvar = global_protos[j]
                            g_var = torch.exp(g_logvar) + 1e-8
                            dist = 0.5 * (((proto_feats[i, :] - g_mu) ** 2) / g_var).sum()
                        else:
                            dist = loss_mse(proto_feats[i, :], global_protos[j])
                        proto_logits[i, j] = -dist / temperature  # 温度缩放

            preds = (torch.sigmoid(proto_logits) > 0.5).float()
            correct_val += (preds == labels).float().sum().item()
            total_val += labels.numel()

            # 记录平均原型损失（对所有全局类别取平均）
            # 解耦模式下仅使用语义特征
            if len(global_protos) > 0:
                loss2 = 0.0
                count = 0
                for lbl_idx in global_protos:
                    if use_dist:
                        g_mu, g_logvar = global_protos[lbl_idx]
                        g_var = torch.exp(g_logvar) + 1e-8
                        loss2 += 0.5 * (((proto_feats - g_mu.unsqueeze(0)) ** 2) / (g_var.unsqueeze(0) + 1e-8)).mean().item()
                    else:
                        loss2 += loss_mse(proto_feats, global_protos[lbl_idx]).item()
                    count += 1
                loss2 = loss2 / max(count, 1)

        acc = correct_val / total_val
        print('| User: {} | Test Acc with protos (per-label): {:.4f}'.format(idx, acc))
        acc_list_g.append(acc)
        loss_list.append(loss2)

    return acc_list_l, acc_list_g, loss_list


# ═══════════════════════════════════════════════════════════════════════════════
#  FedGMKD (NeurIPS 2024): GMM-based Prototype Federated Learning
#  核心区别 vs D²-FL:
#    - GMM 后处理拟合（EM算法），非端到端 NN 输出
#    - 多分量高斯原型 vs 单高斯
#    - Discrepancy-Aware Aggregation (质量+数量加权) vs Bayesian Fusion
# ═══════════════════════════════════════════════════════════════════════════════

def _fit_gmm_pytorch(features, n_components=3, n_iters=10):
    """
    简易 EM 算法拟合对角高斯混合模型（纯 PyTorch，无需 sklearn）

    参数:
        features: (N, D) 特征矩阵
        n_components: 高斯分量数
        n_iters: EM 迭代次数

    返回:
        weights: (K,) 混合权重
        means: (K, D) 各分量均值
        logvars: (K, D) 各分量对数方差（对角协方差）
    """
    N, D = features.shape
    K = min(n_components, N)
    device = features.device

    idxs = torch.randperm(N)[:K]
    means = features[idxs].clone()
    logvars = torch.zeros(K, D, device=device)
    weights = torch.ones(K, device=device) / K

    for _ in range(n_iters):
        vars_ = torch.exp(logvars) + 1e-8
        diff = features.unsqueeze(0) - means.unsqueeze(1)
        log_prob = -0.5 * (torch.log(2 * torch.pi * vars_).sum(dim=1, keepdim=True)
                           + (diff ** 2 / vars_.unsqueeze(1)).sum(dim=2))
        log_prob = log_prob + torch.log(weights.unsqueeze(1))
        log_prob_max = log_prob.max(dim=0, keepdim=True)[0]
        log_sum = log_prob_max + torch.log(
            torch.exp(log_prob - log_prob_max).sum(dim=0, keepdim=True) + 1e-8)
        responsibilities = torch.exp(log_prob - log_sum)

        nk = responsibilities.sum(dim=1) + 1e-8
        weights = nk / N
        means = (responsibilities.unsqueeze(2) * features.unsqueeze(0)).sum(dim=1) / nk.unsqueeze(1)
        diff_new = features.unsqueeze(0) - means.unsqueeze(1)
        vars_new = (responsibilities.unsqueeze(2) * (diff_new ** 2)).sum(dim=1) / nk.unsqueeze(1)
        logvars = torch.log(vars_new + 1e-8)

    return weights, means, logvars


def _update_weights_FedGMKD(self, args, idx, global_protos, model, global_round=0, ld=None):
    """
    FedGMKD 本地训练：GMM 后处理原型 + 质量感知聚合

    L = L_CE + ld * L_gmm_proto
    关键区别: GMM 在 detach 特征上用 EM 拟合（不可端到端学习）
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
    """
    FedGMKD 本地聚合：对每类特征拟合 GMM → (weights, means, logvars)
    """
    agg = {}
    for label, proto_list in protos.items():
        if isinstance(proto_list[0], tuple):
            feats = torch.stack([p[0] for p in proto_list])
        else:
            feats = torch.stack(proto_list)

        if feats.shape[0] >= n_components:
            weights, means, logvars = _fit_gmm_pytorch(
                feats, n_components=n_components, n_iters=10)
            agg[label] = (weights.detach(), means.detach(), logvars.detach())
        else:
            mu_avg = feats.mean(dim=0)
            logvar_avg = torch.log(feats.var(dim=0, unbiased=False) + 1e-8)
            agg[label] = (torch.ones(1), mu_avg.unsqueeze(0), logvar_avg.unsqueeze(0))
    return agg


def _proto_aggregation_FedGMKD(local_protos_list):
    """
    FedGMKD 全局聚合：Discrepancy-Aware Aggregation
    quality_k = 1 / mean(variance)
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
                q = 1.0 / (torch.exp(lv).mean() + 1e-8)
                qualities.append(q)

            q_t = torch.tensor(qualities, device=all_m[0].device)
            q_sum = q_t.sum() + 1e-8

            fused_w = torch.stack([w * q for w, q in zip(all_w, qualities)]).sum(dim=0) / q_sum
            fused_m = torch.stack([m * q for m, q in zip(all_m, qualities)]).sum(dim=0) / q_sum
            fused_lv = torch.stack([lv * q for lv, q in zip(all_lv, qualities)]).sum(dim=0) / q_sum
            global_protos[label] = (fused_w, fused_m, fused_lv)

    return global_protos


# ═══════════════════════════════════════════════════════════════════════════════
#  FedBCS (AAAI 2026): Frequency-Domain Style Recalibration
#  1D 适配版：用 InstanceNorm + 可学习仿射替代 FFT 振幅/相位分离
# ═══════════════════════════════════════════════════════════════════════════════

class StyleRecalibration1D(nn.Module):
    """FedBCS 风格重校准模块（1D特征适配版）

    原始: FFT → 振幅(风格) / 相位(内容) → 可学习重校准
    适配: InstanceNorm → 去风格 + 可学习仿射 → 重校准
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-8
        x_norm = (x - mean) / std
        return self.gamma * x_norm + self.beta


def _update_weights_FedBCS(self, args, idx, global_protos, model, global_round=0, ld=None):
    """
    FedBCS 本地训练：风格重校准 + 原型正则化

    L = L_CE + ld * ||recalibrated_proto - global_proto||²
    """
    if ld is None:
        ld = args.ld
    model.train()
    epoch_loss = {'total': [], '1': [], '2': []}
    use_dist = getattr(args, 'use_distributional', False)
    proto_dim = getattr(args, 'proto_dim', 256) or 256

    if not hasattr(self, 'style_recal'):
        self.style_recal = StyleRecalibration1D(proto_dim).to(self.device)

    if self.args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(self.style_recal.parameters()),
            lr=self.args.lr, momentum=0.5)
    elif self.args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.style_recal.parameters()),
            lr=self.args.lr, weight_decay=1e-4)

    for iter in range(self.args.train_ep):
        batch_loss = {'total': [], '1': [], '2': []}
        agg_protos_label = {}

        for batch_idx, (images, label_g) in enumerate(self.trainloader):
            images, labels = images.to(self.device), label_g.to(self.device)
            model.zero_grad()
            output = model(images)

            if use_dist:
                logits, mu, logvar = output
                proto_raw = mu
            else:
                logits, protos = output
                proto_raw = protos

            loss1 = self.criterion(logits, labels)
            proto_recal = self.style_recal(proto_raw)

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
                                g_mu = g_val[0] if isinstance(g_val, tuple) else g_val
                                loss2 += loss_mse(proto_recal[i_lbl:i_lbl + 1], g_mu.unsqueeze(0))
                            else:
                                loss2 += loss_mse(proto_recal[i_lbl:i_lbl + 1], g_val.unsqueeze(0))
                            count += 1
                loss2 = loss2 / max(count, 1)

            loss = loss1 + loss2 * ld
            loss.backward()
            optimizer.step()

            for i_lbl in range(len(labels)):
                for lbl_idx in range(args.num_classes):
                    if label_g[i_lbl, lbl_idx] > 0:
                        if use_dist:
                            proto_val = (proto_recal[i_lbl, :].detach(), logvar[i_lbl, :].detach())
                        else:
                            proto_val = proto_recal[i_lbl, :].detach()
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


# ═══════════════════════════════════════════════════════════════════════════════
#  FedSeProto (ECAI 2024): Semantic-Domain Feature Decoupling
#  关键区别于 D²-FL:
#    - 硬分割（独立编码器）vs 软门控
#    - HSIC 互信息最小化 vs HSIC+对抗+对比
#    - 点原型 vs 分布原型
# ═══════════════════════════════════════════════════════════════════════════════

class FedSeProtoHeads(nn.Module):
    """FedSeProto 语义-域解耦头"""

    def __init__(self, proto_dim=256, sem_ratio=0.75):
        super().__init__()
        sem_dim = int(proto_dim * sem_ratio)
        dom_dim = proto_dim - sem_dim
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
    """HSIC 互信息最小化"""
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
    """
    FedSeProto 本地训练：语义-域解耦 + 仅共享语义原型

    L = L_CE + ld * L_proto(semantic only) + mi_lambda * L_MI
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
            z_sem, z_dom = self.seproto_heads(proto_features)

            loss1 = self.criterion(logits, labels)
            loss3 = _mi_minimization_loss(z_sem, z_dom)

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

