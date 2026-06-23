# =============================================================================
# 功能:t-SNE 原型可视化 —— 支持单算法 + 多算法并排对比
# =============================================================================
# 两种用法:
#   1) 训练流程内调用 save_protos_npy(global_protos, alg, ...) 保存各算法原型
#   2) 独立运行:
#        # 多算法对比(并排子图,同类同色)
#        python lib/visualize.py --algs fedproto fedgmkd fedbcs fedseproto fedcop \
#                                 --num_classes 14 --proto_dir ./protos_vis
#        # 单算法
#        python lib/visualize.py --alg fedcop --num_classes 14 --proto_dir ./protos_vis
#
# 原型来源:各算法训练结束后由 federated_main 调用 save_protos_npy 生成
#   {alg}_protos.npy [C', D] + {alg}_labels.npy [C']。
# =============================================================================

import os
import numpy as np
import random

# headless 安全(训练流程 import 本模块时不报显示错误)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# sklearn 的 TSNE 改为 lazy import(_tsne_2d 内),避免 save_protos_npy
# 这类只存 .npy 的流程在未装 sklearn 时也报 ImportError。

# 20 种颜色(覆盖 ChestX-ray14 的 14 类与 MuReD 的 20 类)
COLORS = ['#000000', 'peru', '#FF8C00', 'gold', 'lightseagreen', 'royalblue',
          'darkseagreen', 'violet', 'palevioletred', 'g', 'crimson', 'teal',
          'indigo', 'salmon', 'olive', 'cyan', 'pink', 'brown', 'gray', 'purple']


def extract_proto_vector(entry):
    """从全局原型的单类条目提取 1D 向量 [D]。

    处理三种结构(各算法全局原型不同):
      - 点原型:tensor [D]                     (FedProto / FedBCS / FedSeProto)
      - 分布原型 (mu, logvar):取 mu           (FedCoP)
      - GMM 原型 (weights, means, logvars):取 means 的质量加权平均  (FedGMKD)
    """
    if isinstance(entry, (tuple, list)):
        if len(entry) == 2:  # (mu, logvar)
            mu = entry[0]
            return mu.detach().cpu().reshape(-1).numpy()
        elif len(entry) == 3:  # (weights, means, logvars)
            w, m, _ = entry
            w = w.detach().cpu().reshape(-1)
            m = m.detach().cpu()                      # [n_comp, D]
            if w.numel() == m.shape[0]:
                w = w / (w.sum() + 1e-8)
                return (m * w.unsqueeze(1)).sum(0).numpy()
            return m.mean(0).numpy()
    # 点原型 tensor [D]
    return entry.detach().cpu().reshape(-1).numpy()


def save_protos_npy(global_protos, alg, num_classes, proto_dir='./protos_vis'):
    """提取全局原型并保存 {alg}_protos.npy / {alg}_labels.npy,供 t-SNE 可视化。

    参数:
        global_protos: 各算法的全局原型 dict {label: proto_entry}
        alg:           算法名(用作文件前缀)
        num_classes:   类别数(仅用于日志)
        proto_dir:     保存目录
    """
    labels, protos = [], []
    for label in sorted(global_protos.keys(), key=lambda x: int(x)):
        try:
            proto = extract_proto_vector(global_protos[label])
        except Exception as e:
            print(f"[vis] {alg} label {label} extract failed: {e}")
            continue
        if proto.shape[0] == 0:
            continue
        labels.append(int(label))
        protos.append(proto)

    if not protos:
        print(f"[vis] {alg}: no protos to save, skip")
        return

    P = np.stack(protos)            # [C', D]
    L = np.array(labels, dtype=np.int64)
    os.makedirs(proto_dir, exist_ok=True)
    np.save(os.path.join(proto_dir, f'{alg}_protos.npy'), P)
    np.save(os.path.join(proto_dir, f'{alg}_labels.npy'), L)
    print(f"[vis] saved {alg} protos {P.shape} ({len(labels)}/{num_classes} classes) -> {proto_dir}/{alg}_protos.npy")


def _tsne_2d(x, seed=1234):
    """t-SNE 降到 2 维。原型点数少(C=14/20),perplexity 自适应。"""
    from sklearn.manifold import TSNE  # lazy import: 仅画 t-SNE 图时需要
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    n = x.shape[0]
    perp = max(2.0, min(5.0, n - 1.0))
    tsne = TSNE(n_components=2, perplexity=perp, init='pca',
                learning_rate='auto', max_iter=1000, random_state=seed)
    return tsne.fit_transform(x)


def _scatter(ax, x2d, labels, num_classes, title):
    """单子图:按类别上色 + 在点旁标类别号。"""
    for c in range(num_classes):
        m = labels == c
        if m.any():
            ax.scatter(x2d[m, 0], x2d[m, 1], marker='o', s=80,
                       color=COLORS[c % len(COLORS)],
                       edgecolor='k', linewidth=0.4, zorder=2)
    # 点旁标类别号(原型少,标号可读)
    for i, c in enumerate(labels):
        ax.annotate(str(int(c)), (x2d[i, 0], x2d[i, 1]),
                    fontsize=7, ha='center', va='bottom',
                    xytext=(0, 3), textcoords='offset points', zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_compare(algs, proto_dir='./protos_vis', num_classes=14,
                      save_path=None, seed=1234):
    """多算法原型 t-SNE 并排对比:每个算法独立 t-SNE,同一类别跨子图同色。

    用途:直观对比 FedCoP 与基线的原型几何 —— FedCoP 的 L_co 应使共现类
    靠近、互斥类远离,而基线原型布局无此结构约束。
    """
    n = len(algs)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, alg in enumerate(algs):
        ax = axes[i]
        ppath = os.path.join(proto_dir, f'{alg}_protos.npy')
        if not os.path.exists(ppath):
            ax.text(0.5, 0.5, f'{alg}\n(no npy)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(alg, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        P = np.load(ppath, allow_pickle=True)
        L = np.load(os.path.join(proto_dir, f'{alg}_labels.npy'), allow_pickle=True)
        if np.isnan(P).any():
            print(f"[vis] WARN {alg} protos contain {int(np.isnan(P).sum())} NaN "
                  f"(zeroed; check training stability / proto_dim)")
        # 全 NaN / 全零 → 训练数值不稳,跳过该子图而非崩溃
        P_clean = np.nan_to_num(P, nan=0.0)
        if np.isnan(P).all() or np.all(P_clean == 0):
            ax.text(0.5, 0.5, f'{alg}\n(unstable protos\n— increase rounds/proto_dim)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title(alg, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
            continue
        x2d = _tsne_2d(P, seed=seed)
        _scatter(ax, x2d, L, num_classes, alg)

    for j in range(len(algs), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    out = save_path or os.path.join(proto_dir, f'tsne_compare_{num_classes}c.pdf')
    plt.savefig(out, format='pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"[vis] saved comparison -> {out}")


def visualize(args, x, y, save_path=None):
    """单算法 t-SNE 散点(输入 x 已降维到 2D)。保留以兼容旧调用。"""
    fig = plt.figure(figsize=(4, 3))
    for c in range(args.num_classes):
        m = (y.reshape(-1) == c)
        if m.any():
            plt.scatter(x[m, 0], x[m, 1], marker='.',
                        color=COLORS[c % len(COLORS)], alpha=0.6, label=str(c))
    plt.subplots_adjust(bottom=0.15)
    out = save_path or f'./protos_{args.alg}.pdf'
    plt.savefig(out, format='pdf', dpi=600)
    plt.close(fig)
    print(f"Saved visualization to {out}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='FedCoP t-SNE prototype visualization')
    p.add_argument('--algs', nargs='+', default=None,
                   help='多算法对比(并排子图),如:fedproto fedgmkd fedbcs fedseproto fedcop')
    p.add_argument('--alg', default='fedcop', help='单算法名(--algs 优先)')
    p.add_argument('--num_classes', type=int, default=14)
    p.add_argument('--proto_dir', default='./protos_vis', help='原型 npy 目录')
    p.add_argument('--save_path', default=None)
    p.add_argument('--seed', type=int, default=1234)
    a = p.parse_args()

    np.random.seed(a.seed)
    random.seed(a.seed)

    if a.algs:
        visualize_compare(a.algs, proto_dir=a.proto_dir, num_classes=a.num_classes,
                          save_path=a.save_path, seed=a.seed)
    else:
        ppath = os.path.join(a.proto_dir, f'{a.alg}_protos.npy')
        P = np.load(ppath, allow_pickle=True)
        L = np.load(os.path.join(a.proto_dir, f'{a.alg}_labels.npy'), allow_pickle=True)
        x2d = _tsne_2d(P, seed=a.seed)

        class _Args:
            pass
        _a = _Args()
        _a.alg = a.alg
        _a.num_classes = a.num_classes
        visualize(_a, x2d, L.reshape(-1, 1), save_path=a.save_path)
