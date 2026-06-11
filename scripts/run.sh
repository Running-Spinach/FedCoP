#!/bin/bash
# =============================================================================
# D²-FL 全算法对比运行脚本
# =============================================================================
# 用法:
#   bash ./scripts/run.sh              # 运行所有 6 种算法（顺序执行）
#   bash ./scripts/run.sh d2fl         # 仅运行指定算法
#   bash ./scripts/run.sh --dry-run    # 仅打印命令，不实际执行
#
# 算法列表:
#   1. FedAvg      — 经典联邦平均（McMahan et al., 2017）
#   2. FedProto    — 联邦原型学习（Tan et al., AAAI 2022）
#   3. FedGMKD     — GMM 原型 + 差异感知聚合（NeurIPS 2024）
#   4. FedBCS      — 频域风格重校准（AAAI 2026）
#   5. FedSeProto  — 语义-域特征解耦（ECAI 2024）
#   6. D²-FL       — ★ 提出方法（分布原型 + 解耦 + 贝叶斯融合 + EMA + 温度缩放）
# =============================================================================

set -e  # 任一算法出错则停止

# ── 时间估算 ──
# 每轮约 60-120s（取决于算法复杂度），200 轮 × 6 算法 ≈ 20-40 GPU 小时
# 建议在 screen/tmux 中运行：screen -S d2fl_bench && bash ./scripts/run.sh

DRY_RUN=false
if [ "${1}" = "--dry-run" ]; then
    DRY_RUN=true
    shift
fi

# ── 共享参数 ──
ROUNDS=200
NUM_USERS=20
WAYS=5
SHOTS=100
STDEV=2
LD=1.0
FRAC=0.25

BASE_ARGS="--num_classes 14 --num_users ${NUM_USERS} --ways ${WAYS} --shots ${SHOTS} --stdev ${STDEV} --rounds ${ROUNDS} --frac ${FRAC} --ld ${LD}"

# ── 输出目录 ──
LOG_DIR="./logs/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo " D²-FL Benchmark: 6 算法对比"
echo " 日志目录: ${LOG_DIR}"
echo " 开始时间: $(date)"
echo "=============================================="
echo ""
echo " 共享配置: rounds=${ROUNDS} users=${NUM_USERS} ways=${WAYS} shots=${SHOTS} frac=${FRAC}"
echo ""
echo " 预估总耗时: 20-40 GPU 小时（单张 RTX 3080/4070 级别）"
echo "           : 建议在 screen/tmux 后台运行"
echo ""

# ── 辅助函数 ──
run_algo() {
    local name="$1"
    local args="$2"
    local log_file="${LOG_DIR}/${name}.log"

    echo "┌──────────────────────────────────────────────"
    echo "│ [$(date +%H:%M:%S)] 开始: ${name}"
    echo "├──────────────────────────────────────────────"
    echo "│ 命令: python exps/federated_main.py ${args}"
    echo "└──────────────────────────────────────────────"

    if [ "${DRY_RUN}" = true ]; then
        echo "  [DRY-RUN] 跳过执行"
        echo ""
        return
    fi

    START_EPOCH=$(date +%s)
    python exps/federated_main.py ${args} 2>&1 | tee "${log_file}"
    END_EPOCH=$(date +%s)
    ELAPSED=$(( (END_EPOCH - START_EPOCH) / 60 ))

    echo ""
    echo "  ✓ ${name} 完成，耗时: ${ELAPSED} 分钟"
    echo "${name}: ${ELAPSED} min" >> "${LOG_DIR}/_summary.txt"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════
#  算法执行列表
# ═══════════════════════════════════════════════════════════════════

ALL_ALGOS=(
    "fedavg:${BASE_ARGS} --alg fedavg"
    "fedproto:${BASE_ARGS} --alg fedproto"
    "fedgmkd:${BASE_ARGS} --alg fedgmkd --gmm_components 3"
    "fedbcs:${BASE_ARGS} --alg fedbcs"
    "fedseproto:${BASE_ARGS} --alg fedseproto --mi_lambda 0.05"
    "d2fl:${BASE_ARGS} --alg d2fl --use_distributional --dist_type kl --use_disentangle --dis_lambda 0.05 --cal_lambda 0.01 --contra_lambda 0.05 --adv_lambda 0.01 --ent_lambda 0.001 --proto_momentum 0.9 --temperature 1.0 --ld_warmup 50"
)

# ── 如果指定了单个算法，只运行那一个 ──
TARGET="${1:-all}"

if [ "${TARGET}" != "all" ]; then
    echo ">>> 单算法模式: ${TARGET}"
    FOUND=false
    for entry in "${ALL_ALGOS[@]}"; do
        ALGO_NAME="${entry%%:*}"
        ALGO_ARGS="${entry#*:}"
        if [ "${ALGO_NAME}" = "${TARGET}" ]; then
            FOUND=true
            run_algo "${ALGO_NAME}" "${ALGO_ARGS}"
        fi
    done
    if [ "${FOUND}" = false ]; then
        echo "错误: 未知算法 '${TARGET}'"
        echo "可用: fedavg fedproto fedgmkd fedbcs fedseproto d2fl"
        exit 1
    fi
else
    echo ">>> 全算法对比模式（顺序执行）"
    echo ""
    for entry in "${ALL_ALGOS[@]}"; do
        ALGO_NAME="${entry%%:*}"
        ALGO_ARGS="${entry#*:}"
        run_algo "${ALGO_NAME}" "${ALGO_ARGS}"
    done
fi

# ═══════════════════════════════════════════════════════════════════
#  完成摘要
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "=============================================="
echo " Benchmark 完成"
echo " 结束时间: $(date)"
echo " 日志目录: ${LOG_DIR}"
echo "=============================================="

if [ -f "${LOG_DIR}/_summary.txt" ]; then
    echo ""
    echo "各算法耗时汇总:"
    cat "${LOG_DIR}/_summary.txt"
fi
