# 功能：使用t-SNE降维可视化联邦学习中的原型向量分布

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from options import args_parser
import torch
import random


def visualize(args, x, y, save_path=None):
    """
    使用t-SNE降维后的数据绘制原型向量散点图并保存为PDF

    参数:
        args: 配置参数对象，包含算法名称、类别数等信息
        x: 降维后的二维坐标数组，形状为 (n_samples, 2)
        y: 标签数组，形状为 (n_samples, 1)
        save_path: 图片保存路径，默认为 ./protos_{alg}.pdf
    """
    fig = plt.figure(figsize=(4, 3))
    print("Begin visualization ...")
    colors = ['#000000', 'peru', '#FF8C00', 'gold', 'lightseagreen', 'royalblue',
              'darkseagreen', 'violet', 'palevioletred', 'g', 'crimson', 'teal',
              'indigo', 'salmon']

    S_data = np.hstack((x, y))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

    for class_index in range(args.num_classes):
        X = S_data.loc[S_data['label'] == class_index]['x']
        Y = S_data.loc[S_data['label'] == class_index]['y']
        plt.scatter(X, Y, marker='.', color=colors[class_index % len(colors)], alpha=0.08)

    plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.1, hspace=0.15)
    out_path = save_path or f'./protos_{args.alg}.pdf'
    plt.savefig(out_path, format='pdf', dpi=600)
    print(f"Saved visualization to {out_path}")


if __name__ == '__main__':
    args = args_parser()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 加载原型数据和标签（默认从当前目录读取）
    import os
    proto_dir = getattr(args, 'proto_dir', None) or os.getcwd()
    x = np.load(os.path.join(proto_dir, args.alg + '_protos.npy'), allow_pickle=True)
    y = np.load(os.path.join(proto_dir, args.alg + '_labels.npy'), allow_pickle=True)

    # t-SNE降维到2维
    tsne = TSNE()
    x = tsne.fit_transform(x)

    y = y.reshape((-1, 1))
    visualize(args, x, y)
