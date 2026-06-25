# =============================================================================
# 多标签分类评估指标
# =============================================================================
# 原代码只报告"逐标签二值准确率"(flatten 后 correct/total),在稀疏的 14 类
# 多标签场景下被大量负样本主导,无法区分方法优劣。本模块补齐顶会必需的
# 多标签指标,供 FedCoP 及所有基线统一使用:
#   - macro / micro AUROC  : 区分能力(跨类平均 / 整体)
#   - macro / micro F1     : 阈值化后的分类质量
#   - Hamming loss         : 误标比例(越低越好)
#   - subset accuracy      : 全部标签完全匹配的样本比例(最严格)
#   - per-class AUROC      : 每类单独的区分能力(便于看共现/罕见类提升)
# =============================================================================

import numpy as np


def _to_numpy(x):
    """把 torch.Tensor / numpy / list 统一转成 numpy float 数组"""
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def compute_multilabel_metrics(probs, labels, num_classes=None):
    """计算多标签分类的完整指标集

    参数:
        probs:       (N, C) 预测概率 ∈ [0, 1](sigmoid 后,未阈值化)
        labels:      (N, C) 真实多热标签 ∈ {0, 1}
        num_classes: 类别数 C。None 时从 probs 形状推断

    返回:
        dict,包含:
            'auroc_macro', 'auroc_micro', 'f1_macro', 'f1_micro',
            'hamming_loss', 'subset_acc',
            'auroc_per_class' (list[float],长度 C,无法计算的类为 nan)
    """
    from sklearn.metrics import (roc_auc_score, f1_score,
                                  hamming_loss as sk_hamming,
                                  accuracy_score)

    probs = _to_numpy(probs)
    labels = _to_numpy(labels)
    if num_classes is None:
        num_classes = probs.shape[1]

    # ── 阈值化预测(0.5)用于 F1 / Hamming / subset ──
    preds = (probs >= 0.5).astype(np.float64)

    metrics = {}

    # ── AUROC(需要概率;某些类若只有一个标签值则无法计算)──
    auroc_per_class = []
    valid_auroc = []
    for c in range(num_classes):
        y = labels[:, c]
        # 该类必须同时含正负样本,AUROC 才有定义
        if len(np.unique(y)) < 2:
            auroc_per_class.append(float('nan'))
            continue
        try:
            a = roc_auc_score(y, probs[:, c])
            auroc_per_class.append(float(a))
            valid_auroc.append(float(a))
        except ValueError:
            auroc_per_class.append(float('nan'))
    metrics['auroc_per_class'] = auroc_per_class
    metrics['auroc_macro'] = float(np.mean(valid_auroc)) if valid_auroc else float('nan')

    # micro AUROC:把所有 (样本,类) 展平后整体算
    try:
        # 需要至少两个不同标签值
        if len(np.unique(labels)) >= 2:
            metrics['auroc_micro'] = float(
                roc_auc_score(labels.ravel(), probs.ravel()))
        else:
            metrics['auroc_micro'] = float('nan')
    except ValueError:
        metrics['auroc_micro'] = float('nan')

    # ── F1(macro / micro)──
    metrics['f1_macro'] = float(f1_score(labels, preds, average='macro', zero_division=0))
    metrics['f1_micro'] = float(f1_score(labels, preds, average='micro', zero_division=0))

    # ── Hamming loss(越低越好)──
    metrics['hamming_loss'] = float(sk_hamming(labels, preds))

    # ── subset accuracy(全部标签完全匹配,最严格)──
    metrics['subset_acc'] = float(accuracy_score(labels, preds))

    return metrics


def format_metrics(metrics):
    """把指标 dict 格式化成一行可打印字符串 (NaN 安全)"""
    def _fmt(key):
        v = metrics.get(key, float('nan'))
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 'nan'
            return f'{float(v):.4f}'
        except (ValueError, TypeError):
            return 'nan'
    return (
        f"AUROC(macro/micro)={_fmt('auroc_macro')}/{_fmt('auroc_micro')} | "
        f"F1(macro/micro)={_fmt('f1_macro')}/{_fmt('f1_micro')} | "
        f"Hamming={_fmt('hamming_loss')} | SubsetAcc={_fmt('subset_acc')}"
    )
