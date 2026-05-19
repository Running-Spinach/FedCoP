# 功能：使用t-SNE降维可视化联邦学习中的原型向量分布

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from options import args_parser
import torch
import random

def visualize(args, x, y):
    """
    使用t-SNE降维后的数据绘制原型向量散点图并保存为PDF

    参数:
        args: 配置参数对象，包含算法名称、类别数等信息
        x: 降维后的二维坐标数组，形状为 (n_samples, 2)
        y: 标签数组，形状为 (n_samples, 1)
    """
    fig = plt.figure(figsize=(4, 3))
    print("Begin visualization ...")
    markers = ['.', 'o', 'v', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
    colors = ['#000000', 'peru', '#FF8C00', 'gold', 'lightseagreen', 'royalblue', 'darkseagreen', 'violet', 'palevioletred', 'g']

    Label_Com = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 14
             }

    S_data = np.hstack((x, y))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

    # 按类别分别绘制散点
    for class_index in range(args.num_classes):
        X = S_data.loc[S_data['label'] == class_index]['x']
        Y = S_data.loc[S_data['label'] == class_index]['y']
        plt.scatter(X, Y, marker='.', color=colors[class_index], alpha=0.08)

    plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.1, hspace=0.15)
    plt.savefig("./protos_"+args.alg+".pdf", format='pdf', dpi=600)

args = args_parser()
args.alg = 'fedper'

# 设置设备与随机种子
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.device == 'cuda':
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# 加载原型数据和标签
x = np.load('/Users/tanyue/Desktop/saved/protos/' + args.alg + '_protos.npy', allow_pickle=True)
y = np.load('/Users/tanyue/Desktop/saved/protos/' + args.alg + '_labels.npy', allow_pickle=True)

# t-SNE降维到2维
tsne = TSNE()
x = tsne.fit_transform(x)

y = y.reshape((-1, 1))
visualize(args, x, y)
