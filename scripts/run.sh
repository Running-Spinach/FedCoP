#!/bin/bash
# =============================================================================
# D²-FL 全算法对比运行脚本 — 精简输出
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
#   6. D²-FL       — ★ 提出方法
# =============================================================================

set -e

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

LOG_DIR="./logs/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# ═══════════════════════════════════════════════════════════════════
#  表头
# ═══════════════════════════════════════════════════════════════════

print_header() {
    printf "\n"
    printf "==============================================\n"
    printf " D²-FL Benchmark: 6 算法对比\n"
    printf " rounds=%-3d  users=%-2d  ways=%-2d  shots=%-3d\n" ${ROUNDS} ${NUM_USERS} ${WAYS} ${SHOTS}
    printf " 开始: %s\n" "$(date)"
    printf "==============================================\n"
    printf " %-12s | %-14s | %-12s | %s\n" "算法" "Acc (proto)" "Acc (model)" "耗时"
    printf " %-12s-+-%-14s-+-%-12s-|-%s\n" "------------" "--------------" "------------" "--------"
}

# ═══════════════════════════════════════════════════════════════════
#  运行单个算法
# ═══════════════════════════════════════════════════════════════════

run_algo() {
    local name="$1"
    local args="$2"
    local log_file="${LOG_DIR}/${name}.log"

    printf " %-12s | ..." "${name}"

    if [ "${DRY_RUN}" = true ]; then
        printf "\r %-12s | %-14s | %-12s | %s\n" "${name}" "[dry-run]" "-" "-"
        printf "  命令: python exps/federated_main.py %s\n" "${args}"
        return
    fi

    SECONDS=0
    _exit=0
    python exps/federated_main.py ${args} > "${log_file}" 2>&1 || _exit=$?
    ELAPSED=$(awk "BEGIN {printf \"%.1f\", ${SECONDS} / 60}")

    # 提取结果
    ACC_PROTO=$(grep -oP 'with protos.*mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null || echo "")
    ACC_MODEL=$(grep -oP 'w/o protos.*mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null || echo "")
    ACC_SINGLE=$(grep -oP 'For all users, mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null || echo "")

    if [ -z "${ACC_PROTO}" ] && [ -z "${ACC_MODEL}" ] && [ -n "${ACC_SINGLE}" ]; then
        ACC_PROTO="${ACC_SINGLE}"
        ACC_MODEL="-"
    fi

    if [ -z "${ACC_PROTO}" ]; then
        # 提取失败，显示 Python 退出码和最后 3 行 stderr
        _tail=$(tail -3 "${log_file}" 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g')
        ACC_PROTO="ERR($_exit)"
        ACC_MODEL="${_tail:-(empty log)}"
    fi

    printf "\r %-12s | %-14s | %-12s | %s min\n" \
        "${name}" "${ACC_PROTO}" "${ACC_MODEL}" "${ELAPSED}"

    printf "%-12s  proto_acc=%-8s  model_acc=%-8s  time=%s min\n" \
        "${name}" "${ACC_PROTO}" "${ACC_MODEL}" "${ELAPSED}" >> "${LOG_DIR}/_summary.txt"
}

# ═══════════════════════════════════════════════════════════════════
#  算法列表
# ═══════════════════════════════════════════════════════════════════

ALL_ALGOS=(
    "fedavg:${BASE_ARGS} --alg fedavg"
    "fedproto:${BASE_ARGS} --alg fedproto"
    "fedgmkd:${BASE_ARGS} --alg fedgmkd --gmm_components 3"
    "fedbcs:${BASE_ARGS} --alg fedbcs"
    "fedseproto:${BASE_ARGS} --alg fedseproto --mi_lambda 0.05"
    "d2fl:${BASE_ARGS} --alg d2fl --use_distributional --dist_type kl --use_disentangle --dis_lambda 0.05 --cal_lambda 0.01 --contra_lambda 0.05 --adv_lambda 0.01 --ent_lambda 0.001 --proto_momentum 0.9 --temperature 1.0 --ld_warmup 50"
)

# ═══════════════════════════════════════════════════════════════════
#  执行
# ═══════════════════════════════════════════════════════════════════

TARGET="${1:-all}"

if [ "${TARGET}" != "all" ]; then
    echo ">>> 单算法: ${TARGET}"
    print_header
    FOUND=false
    for entry in "${ALL_ALGOS[@]}"; do
        ALGO_NAME="${entry%%:*}"
        if [ "${ALGO_NAME}" = "${TARGET}" ]; then
            FOUND=true
            run_algo "${ALGO_NAME}" "${entry#*:}"
        fi
    done
    if [ "${FOUND}" = false ]; then
        echo "错误: 未知算法 '${TARGET}'"
        echo "可用: fedavg fedproto fedgmkd fedbcs fedseproto d2fl"
        exit 1
    fi
else
    print_header
    for entry in "${ALL_ALGOS[@]}"; do
        ALGO_NAME="${entry%%:*}"
        run_algo "${ALGO_NAME}" "${entry#*:}"
    done
fi

printf " %-12s-+-%-14s-+-%-12s-|-%s\n" "------------" "--------------" "------------" "--------"
printf "\n 完成: %s\n" "$(date)"
printf " 详细日志: ${LOG_DIR}/\n\n"

if [ -f "${LOG_DIR}/_summary.txt" ]; then
    echo "各算法耗时汇总:"
    cat "${LOG_DIR}/_summary.txt"
fi
