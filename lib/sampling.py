# 功能：联邦学习数据采样模块，支持IID/Non-IID数据划分

import numpy as np
from torchvision import datasets, transforms
import random


def mnist_iid(dataset, num_users):
    """
    对MNIST数据集进行IID（独立同分布）采样

    参数:
        dataset: MNIST数据集对象
        num_users: 客户端数量

    返回:
        dict_users: 字典，键为用户索引，值为该用户分配的数据索引集合
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(args, dataset, num_users, n_list, k_list):
    """
    对MNIST数据集进行Non-IID（非独立同分布）采样，每个客户端分配不同类别和数量的数据

    参数:
        args: 配置参数对象
        dataset: MNIST数据集对象
        num_users: 客户端数量
        n_list: 每个客户端拥有的类别数量列表
        k_list: 每个客户端每类拥有的样本数量列表

    返回:
        dict_users: 字典，键为用户索引，值为该用户的数据索引数组
        classes_list: 每个用户拥有的类别列表
    """
    num_shards, num_imgs = 10, 6000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # 按标签排序
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}  # 记录每个类别的起始索引
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    classes_list = []
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        k_len = args.train_shots_max
        classes = random.sample(range(0, args.num_classes), n)  # 随机选择该用户拥有的类别
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)

    return dict_users, classes_list


def mnist_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):
    """
    对MNIST测试集进行Non-IID本地测试数据采样

    参数:
        args: 配置参数对象
        test_dataset: 测试数据集
        num_users: 客户端数量
        n_list: 每个客户端的类别数量列表
        k_list: 每个客户端每类的样本数量列表
        classes_list: 每个客户端训练时的类别列表

    返回:
        dict_users: 字典，键为用户索引，值为该用户的本地测试数据索引数组
    """
    num_shards, num_imgs = 10, 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards * num_imgs)
    labels = test_dataset.train_labels.numpy()

    # 按标签排序
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    for i in range(num_users):
        k = 40  # 每个类别选择的测试样本数
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * 40 + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        dict_users[i] = user_data

    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    对MNIST数据集进行Non-IID不等量划分，各客户端数据量不均衡

    参数:
        dataset: MNIST数据集对象
        num_users: 客户端数量

    返回:
        dict_users: 字典，键为用户索引，值为该用户的数据索引数组
    """
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # 按标签排序
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # 每个客户端分配的数据分片数范围
    min_shard = 1
    max_shard = 30

    # 随机分配不等量的分片给每个客户端
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    if sum(random_shard_size) > num_shards:
        for i in range(num_users):
            # 先为每个客户端分配1个分片，确保每个客户端至少有数据
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # 再随机分配剩余分片
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:
        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # 将剩余分片分配给数据量最少的客户端
            shard_size = len(idx_shard)
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

    return dict_users


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

        dict_users[i] = np.array(list(set(user_data)))
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
    """对 ChestX-ray14 进行 IID 随机均匀划分"""
    num_items = int(len(dataset) / num_users)
    dict_users = {}
    all_idxs = list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
