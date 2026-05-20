# 功能：本地更新模块，实现FedProto联邦学习中的本地训练、测试和原型提取

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
import sys
from pathlib import Path
lib_dir = (Path(__file__).parent).resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from dist_proto.losses import distributional_proto_loss


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
                log_probs = output[0] if isinstance(output, tuple) else output
                loss = self.criterion(log_probs, labels)

                loss.backward()
                optimizer.step()

                # 多标签 per-label 准确率
                preds = (torch.sigmoid(log_probs) > 0.5).float()
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
                log_probs = output[0] if isinstance(output, tuple) else output
                loss_ce = self.criterion(log_probs, labels)

                # FedProx 近端项：L2 距离（仅对可训练参数）
                prox_term = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad and name in global_model_state:
                        prox_term += torch.sum(
                            (param - global_model_state[name].to(self.device)) ** 2)
                loss = loss_ce + (mu / 2.0) * prox_term

                loss.backward()
                optimizer.step()

                preds = (torch.sigmoid(log_probs) > 0.5).float()
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
                log_probs = output[0] if isinstance(output, tuple) else output
                loss = self.criterion(log_probs, labels)
                loss.backward()

                # SCAFFOLD 梯度修正 + 参数更新
                for name, param in model.named_parameters():
                    if param.grad is not None and name in c_global and name in c_local:
                        param.grad = (param.grad
                                      - c_local[name].to(self.device)
                                      + c_global[name].to(self.device))

                optimizer.step()
                total_steps += 1

                preds = (torch.sigmoid(log_probs) > 0.5).float()
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

    def update_weights_het(self, args, idx, global_protos, model, global_round=round):
        """
        FedProto异构联邦学习的权重更新方法，结合分类损失和原型距离损失

        参数:
            args: 配置参数
            idx: 客户端索引
            global_protos: 全局原型字典
            model: 当前模型
            global_round: 全局训练轮次

        返回:
            model.state_dict(): 更新后的模型状态字典
            epoch_loss: 包含total/1/2三类损失的字典
            acc_val.item(): 准确率
            agg_protos_label: 聚合后的本地原型字典
        """
        model.train()
        epoch_loss = {'total': [], '1': [], '2': [], '3': []}

        use_dist = getattr(args, 'use_distributional', False)

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'total': [], '1': [], '2': [], '3': []}
            agg_protos_label = {}

            for batch_idx, (images, label_g) in enumerate(self.trainloader):
                images, labels = images.to(self.device), label_g.to(self.device)

                model.zero_grad()
                output = model(images)

                if use_dist:
                    log_probs, mu, logvar = output
                else:
                    log_probs, protos = output

                loss1 = self.criterion(log_probs, labels)

                loss_mse = nn.MSELoss()
                if len(global_protos) == 0:
                    loss2 = 0 * loss1
                else:
                    if use_dist:
                        # 分布原型损失：对每张图的所有正标签计算 KL/Wasserstein
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
                        # 点原型损失：MSE 距离（多标签：遍历所有正标签）
                        loss2 = 0.0
                        count = 0
                        for i_lbl in range(len(labels)):
                            proto_i = protos[i_lbl]
                            for lbl_idx in range(args.num_classes):
                                if labels[i_lbl, lbl_idx] > 0 and lbl_idx in global_protos:
                                    loss2 += loss_mse(proto_i, global_protos[lbl_idx])
                                    count += 1
                        loss2 = loss2 / max(count, 1)

                loss = loss1 + loss2 * args.ld
                loss.backward()
                optimizer.step()

                # 按标签聚合本地原型（多标签：一张图贡献给所有正标签）
                for i_lbl in range(len(labels)):
                    for lbl_idx in range(args.num_classes):
                        if label_g[i_lbl, lbl_idx] > 0:
                            if use_dist:
                                proto_val = (mu[i_lbl, :].detach(), logvar[i_lbl, :].detach())
                            else:
                                proto_val = protos[i_lbl, :].detach()
                            if lbl_idx in agg_protos_label:
                                agg_protos_label[lbl_idx].append(proto_val)
                            else:
                                agg_protos_label[lbl_idx] = [proto_val]

                # 多标签 per-label 准确率（所有标签位置的平均匹配率）
                preds = (torch.sigmoid(log_probs) > 0.5).float()
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

            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))
            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))

        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])
        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])

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

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss


class LocalTest(object):
    """
    本地测试类，用于执行单个客户端的模型测试和微调
    """

    def __init__(self, args, dataset, idxs):
        """
        初始化本地测试对象

        参数:
            args: 配置参数
            dataset: 测试数据集
            idxs: 该客户端的数据索引
        """
        self.args = args
        self.testloader = self.test_split(dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.BCEWithLogitsLoss().to(args.device)

    def test_split(self, dataset, idxs):
        """
        根据索引构建测试数据加载器

        参数:
            dataset: 数据集
            idxs: 数据索引列表

        返回:
            testloader: 测试数据加载器
        """
        idxs_test = idxs[:int(1 * len(idxs))]

        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=64, shuffle=False)
        return testloader

    def get_result(self, args, idx, classes_list, model):
        """
        获取本地测试结果

        参数:
            args: 配置参数
            idx: 客户端索引
            classes_list: 类别列表
            model: 待评估模型

        返回:
            loss: 损失值
            acc: 准确率
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            model.zero_grad()
            output = model(images)
            if isinstance(output, tuple):
                outputs = output[0]
            else:
                outputs = output
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            outputs = outputs[:, 0: args.num_classes]
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        return loss, acc

    def fine_tune(self, args, dataset, idxs, model):
        """
        在本地测试数据上微调模型

        参数:
            args: 配置参数
            dataset: 数据集
            idxs: 数据索引
            model: 待微调模型

        返回:
            model: 微调后的模型
        """
        trainloader = self.test_split(dataset, list(idxs))
        model.train()
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

        for iter in range(10):
            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                output = model(images)
                if isinstance(output, tuple):
                    outputs = output[0]
                else:
                    outputs = output
                outputs = outputs[:, 0:args.num_classes]
                loss = self.criterion(outputs, labels)

                loss.backward()
                optimizer.step()

        return model


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


def test_inference_new_het(args, local_model_list, test_dataset, global_protos=[], temperature=None):
    """全局测试推理：基于原型最近邻距离的多标签分类

    返回:
        acc: per-label 准确率
    """
    loss_mse = nn.MSELoss()
    use_dist = getattr(args, 'use_distributional', False)
    if temperature is None:
        temperature = getattr(args, 'temperature', 1.0)
    device = args.device
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    total_val, correct_val = 0.0, 0.0

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        protos_list = []
        for idx in range(args.num_users):
            model = local_model_list[idx]
            output = model(images)
            if use_dist and len(output) == 3:
                _, mu, _ = output
                protos_list.append(mu)
            else:
                _, protos = output
                protos_list.append(protos)

        # 集成所有客户端的原型
        ensem_proto = torch.zeros_like(protos_list[0])
        for p in protos_list:
            ensem_proto += p
        ensem_proto /= len(protos_list)

        # 计算到每个全局原型的距离 → 负距离作为 logit
        proto_logits = torch.zeros(images.shape[0], args.num_classes, device=device)
        for i in range(images.shape[0]):
            for j in range(args.num_classes):
                if j in global_protos:
                    if use_dist:
                        g_mu, g_logvar = global_protos[j]
                        g_var = torch.exp(g_logvar) + 1e-8
                        dist = 0.5 * (((ensem_proto[i, :] - g_mu) ** 2) / g_var).sum()
                    else:
                        dist = loss_mse(ensem_proto[i, :], global_protos[j])
                    proto_logits[i, j] = -dist / temperature

        preds = (torch.sigmoid(proto_logits) > 0.5).float()
        correct_val += (preds == labels).float().sum().item()
        total_val += labels.numel()

    return correct_val / total_val


def test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_gt, global_protos=[], temperature=None):
    """多标签联邦学习测试：分别评估模型自身分类和原型最近邻分类的效果

    参数:
        temperature: 原型推理温度系数。None 时自动从 args 读取，默认 1.0
            - T > 1: 软化概率（更平滑）
            - T < 1: 锐化概率（更确信）
            - 仅 FedProto / DPP-FL 有效

    返回:
        acc_list_l: 各客户端用自身模型（sigmoid阈值）的 per-label 准确率
        acc_list_g: 各客户端用全局原型距离分类的 per-label 准确率
        loss_list: 各客户端的原型损失
    """
    loss_mse = nn.MSELoss()
    use_dist = getattr(args, 'use_distributional', False)
    if temperature is None:
        temperature = getattr(args, 'temperature', 1.0)
    device = args.device

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

            if use_dist and len(output) == 3:
                outputs, mu, logvar = output
            else:
                outputs, protos = output

            # 计算到每个全局原型的距离 → 负距离/T 作为 logit → sigmoid → 二值预测
            proto_feats = mu if (use_dist and len(output) == 3) else protos
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
            if len(global_protos) > 0:
                loss2 = 0.0
                count = 0
                for lbl_idx in global_protos:
                    if use_dist:
                        g_mu, g_logvar = global_protos[lbl_idx]
                        g_var = torch.exp(g_logvar) + 1e-8
                        loss2 += 0.5 * (((mu - g_mu.unsqueeze(0)) ** 2) / (g_var.unsqueeze(0) + 1e-8)).mean().item()
                    else:
                        loss2 += loss_mse(protos, global_protos[lbl_idx]).item()
                    count += 1
                loss2 = loss2 / max(count, 1)

        acc = correct_val / total_val
        print('| User: {} | Test Acc with protos (per-label): {:.4f}'.format(idx, acc))
        acc_list_g.append(acc)
        loss_list.append(loss2)

    return acc_list_l, acc_list_g, loss_list


def get_proto_vec(args, local_model_list, dataset, user_groups_gt):
    """
    提取所有客户端的原型向量（用于可视化分析）

    参数:
        args: 配置参数
        local_model_list: 本地模型列表
        dataset: 数据集
        user_groups_gt: 用户测试数据分组

    返回:
        x: 原型向量数组 (n_samples, proto_dim)
        y: 标签数组 (n_samples,)
        d: 客户端索引数组 (n_samples,)
    """
    x = []
    y = []
    d = []

    for i in range(args.num_users):
        model = local_model_list[i]
        model.eval()
        protos = []
        labels = []
        for batch_idx, (images, label_g) in enumerate(DataLoader(DatasetSplit(dataset, user_groups_gt[i]), batch_size=64, shuffle=False)):
            images_l, labels_l = images.to(args.device), label_g.to(args.device)
            output = model(images_l)
            if len(output) == 3:
                _, protos_tmp, _ = output
            else:
                _, protos_tmp = output
            for proto in protos_tmp:
                protos.append(proto)
            for label in labels_l:
                labels.append(label)

        for idx in range(len(labels)):
            if args.device == 'cuda':
                tmp = protos[idx].cpu().detach().numpy()
            else:
                tmp = protos[idx].detach().numpy()
            x.append(tmp)
            y.append(labels[idx].item())
            d.append(i)

    x = np.array(x)
    y = np.array(y)
    d = np.array(d)

    return x, y, d


def get_proto_vec_lt(args, local_model_list, dataset, user_groups_gt):
    """
    提取并保存按标签聚合后的原型向量（用于后续可视化）

    参数:
        args: 配置参数
        local_model_list: 本地模型列表
        dataset: 数据集
        user_groups_gt: 用户测试数据分组
    """
    x = []
    y = []
    d = []

    for i in range(args.num_users):
        model = local_model_list[i]
        model.eval()
        agg_protos_label = {}
        for batch_idx, (images, label_g) in enumerate(DataLoader(DatasetSplit(dataset, user_groups_gt[i]), batch_size=64, shuffle=False)):
            images_l, labels_l = images.to(args.device), label_g.to(args.device)
            output = model(images_l)
            if len(output) == 3:
                _, protos, _ = output
            else:
                _, protos = output

            for k in range(len(labels_l)):
                lbl = label_g[k].item() if isinstance(label_g[k], torch.Tensor) else label_g[k]
                if lbl in agg_protos_label:
                    agg_protos_label[lbl].append(protos[k, :])
                else:
                    agg_protos_label[lbl] = [protos[k, :]]

        for label, proto_list in agg_protos_label.items():
            for proto in proto_list:
                if args.device == 'cuda':
                    tmp = proto.cpu().detach().numpy()
                else:
                    tmp = proto.detach().numpy()
                x.append(tmp)
                y.append(label)
                d.append(i)

    x = np.array(x)
    y = np.array(y)
    d = np.array(d)
    np.save('./' + args.alg + '_protos.npy', x)
    np.save('./' + args.alg + '_labels.npy', y)
    np.save('./' + args.alg + '_idx.npy', d)

    print("Save protos and labels successfully.")
