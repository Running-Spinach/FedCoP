# 功能：联邦学习数据采样模块，支持IID/Non-IID数据划分

import numpy as np
import random


def chestxray_noniid(args, dataset, num_users, n_list, k_list):
    """对 ChestX-ray14 进行 Non-IID 划分，按主标签（首个阳性疾病）分组

    每张图片可能包含多个疾病标签。划分策略：
      1. 将图片按首个阳性标签归类（"No Finding"单独归为一类）
      2. 每个客户端随机分配 n 个疾病标签 + 对应的图片
      3. 每类取 k 张图片，标签向量保持完整 14 维

    返回:
        dict_users: {client_idx: np.array of data indices}
        classes_list: [client_classes]
    """
    labels_all = dataset.labels  # (N, 14)
    num_imgs = len(dataset)

    # 计算每张图片的主标签：第一个阳性标签，若全零则为 -1 (No Finding)
    primary_labels = np.full(num_imgs, -1, dtype=int)
    for i in range(num_imgs):
        positives = np.where(labels_all[i] > 0)[0]
        primary_labels[i] = positives[0] if len(positives) > 0 else -1

    # 按主标签排序
    sorted_idxs = np.argsort(primary_labels)
    primary_labels_sorted = primary_labels[sorted_idxs]

    # 记录每个标签对应的起始和结束索引
    label_ranges = {}
    for lbl in range(-1, args.num_classes):
        positions = np.where(primary_labels_sorted == lbl)[0]
        if len(positions) > 0:
            label_ranges[lbl] = (positions[0], positions[-1] + 1)

    classes_list = []
    dict_users = {}

    for i in range(num_users):
        n = min(n_list[i], args.num_classes)
        k = k_list[i]
        classes = random.sample(range(0, args.num_classes), n)
        classes = np.sort(classes)
        print(f"user {i + 1}: {n}-way {k}-shot, classes: {classes}")

        user_data = []
        for each_class in classes:
            if each_class in label_ranges:
                start, end = label_ranges[each_class]
                available = sorted_idxs[start:end]
                if len(available) > k:
                    chosen = np.random.choice(available, k, replace=False)
                else:
                    chosen = available
                user_data.extend(chosen.tolist())

        # "No Finding" 样本均匀分配给所有客户端
        if -1 in label_ranges:
            start, end = label_ranges[-1]
            nf_samples = sorted_idxs[start:end]
            nf_per_client = max(10, len(nf_samples) // num_users)
            nf_chosen = np.random.choice(nf_samples, nf_per_client, replace=False)
            user_data.extend(nf_chosen.tolist())

        dict_users[i] = np.array(list(set(user_data)))#去重打乱
        classes_list.append(classes)

    return dict_users, classes_list


def chestxray_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):
    """对 ChestX-ray14 测试集进行 Non-IID 本地测试数据采样

    每个客户端的测试数据仅来自其训练过的疾病类别。
    """
    labels_all = test_dataset.labels
    num_imgs = len(test_dataset)

    primary_labels = np.full(num_imgs, -1, dtype=int)
    for i in range(num_imgs):
        positives = np.where(labels_all[i] > 0)[0]
        primary_labels[i] = positives[0] if len(positives) > 0 else -1

    sorted_idxs = np.argsort(primary_labels)
    primary_labels_sorted = primary_labels[sorted_idxs]

    label_ranges = {}
    for lbl in range(-1, args.num_classes):
        positions = np.where(primary_labels_sorted == lbl)[0]
        if len(positions) > 0:
            label_ranges[lbl] = (positions[0], positions[-1] + 1)

    dict_users = {}
    for i in range(num_users):
        classes = classes_list[i]
        k_test = 20
        user_data = []
        for each_class in classes:
            if each_class in label_ranges:
                start, end = label_ranges[each_class]
                available = sorted_idxs[start:end]
                n_take = min(k_test, len(available))
                chosen = np.random.choice(available, n_take, replace=False)
                user_data.extend(chosen.tolist())
        dict_users[i] = np.array(user_data)

    return dict_users


def chestxray_iid(dataset, num_users):
    """对 ChestX-ray14 进行 IID 随机均匀划分

    参数:
        dataset: ChestX-ray14 数据集对象
        num_users: 客户端数量

    返回:
        dict_users: 字典 {client_idx: set of data indices}
    """
    num_items = int(len(dataset) / num_users)
    dict_users = {}
    all_idxs = list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
