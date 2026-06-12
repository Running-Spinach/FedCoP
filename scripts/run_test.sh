#!/bin/bash
# =============================================================================
# D²-FL 快速测试脚本 — 精简输出，方便对比各算法结果
# =============================================================================
# 用法:
#   bash ./scripts/run_test.sh           # 运行全部 6 种算法
#   bash ./scripts/run_test.sh d2fl      # 仅运行指定算法
# =============================================================================

set -e

# ── 快速 Bug 检查参数（4090 上 ~30-50 min 跑完 6 算法）──
# 核心思路：砍 rounds（多轮逻辑 5 轮足够验证），降 shots（省数据加载），
# 缩 proto_dim（减计算量），其他保持能触发非 IID + 多客户端路径即可。
ROUNDS=5
NUM_USERS=5
WAYS=3
SHOTS=20
STDEV=1
LD=1.0
FRAC=0.4
PROTO_DIM=32

BASE_ARGS="--num_classes 14 --num_users ${NUM_USERS} --ways ${WAYS} --shots ${SHOTS} --stdev ${STDEV} --rounds ${ROUNDS} --frac ${FRAC} --ld ${LD} --proto_dim ${PROTO_DIM}"

LOG_DIR="./logs/test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# ═══════════════════════════════════════════════════════════════════
#  表头
# ═══════════════════════════════════════════════════════════════════

print_header() {
    printf "\n"
    printf "==============================================\n"
    printf " D²-FL 快速测试\n"
    printf " rounds=%-2d  users=%-2d  ways=%-2d  shots=%-2d\n" ${ROUNDS} ${NUM_USERS} ${WAYS} ${SHOTS}
    printf "==============================================\n"
    printf " %-12s | %-14s | %-12s | %s\n" "算法" "Acc (proto)" "Acc (model)" "耗时"
    printf " %-12s-+-%-14s-+-%-12s-|-%s\n" "------------" "--------------" "------------" "--------"
}

# ═══════════════════════════════════════════════════════════════════
#  运行单个算法并提取关键结果
# ═══════════════════════════════════════════════════════════════════

run_algo() {
    local name="$1"
    local args="$2"
    local log_file="${LOG_DIR}/${name}.log"

    printf " %-12s | ..." "${name}"

    # 静默运行，全部输出写入日志文件
    SECONDS=0
    _exit=0
    python exps/federated_main.py ${args} > "${log_file}" 2>&1 || _exit=$?
    ELAPSED=$(awk "BEGIN {printf \"%.1f\", ${SECONDS} / 60}")

    # 从日志提取结果
    # 格式1 (FedProto/D2FL): "For all users (with protos), mean of per-label acc is X, std is Y"
    # 格式2 (FedAvg/FedGMKD/FedBCS/FedSeProto): "For all users, mean of per-label acc is X, std is Y"
    # 格式3 (FedProto/D2FL): "For all users (w/o protos), mean of per-label acc is X, std is Y"

    ACC_PROTO=$(grep -oP 'with protos.*mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null || echo "")
    ACC_MODEL=$(grep -oP 'w/o protos.*mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null || echo "")
    ACC_SINGLE=$(grep -oP 'For all users, mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null || echo "")

    # FedAvg/FedGMKD/FedBCS/FedSeProto 只有单一 acc
    if [ -z "${ACC_PROTO}" ] && [ -z "${ACC_MODEL}" ] && [ -n "${ACC_SINGLE}" ]; then
        ACC_PROTO="${ACC_SINGLE}"
        ACC_MODEL="-"
    fi

    # 如果都没提取到，标记失败并显示原因
    if [ -z "${ACC_PROTO}" ]; then
        _tail=$(tail -3 "${log_file}" 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g')
        ACC_PROTO="ERR($_exit)"
        ACC_MODEL="${_tail:-(empty log)}"
    fi

    printf "\r %-12s | %-14s | %-12s | %s min\n" \
        "${name}" "${ACC_PROTO}" "${ACC_MODEL}" "${ELAPSED}"

    # 同时写入汇总文件
    printf "%-12s  proto_acc=%-8s  model_acc=%-8s  time=%s min\n" \
        "${name}" "${ACC_PROTO}" "${ACC_MODEL}" "${ELAPSED}" >> "${LOG_DIR}/_summary.txt"
}

# ═══════════════════════════════════════════════════════════════════
#  算法列表
# ═══════════════════════════════════════════════════════════════════

ALL_ALGOS=(
    "fedavg:${BASE_ARGS} --alg fedavg"
    "fedproto:${BASE_ARGS} --alg fedproto"
    "fedgmkd:${BASE_ARGS} --alg fedgmkd --gmm_components 2"
    "fedbcs:${BASE_ARGS} --alg fedbcs"
    "fedseproto:${BASE_ARGS} --alg fedseproto --mi_lambda 0.05"
    "d2fl:${BASE_ARGS} --alg d2fl --use_distributional --dist_type kl --use_disentangle --dis_lambda 0.05 --cal_lambda 0.01 --contra_lambda 0.05 --adv_lambda 0.01 --ent_lambda 0.001 --proto_momentum 0.9 --ld_warmup 3"
)

# ═══════════════════════════════════════════════════════════════════
#  执行
# ═══════════════════════════════════════════════════════════════════

TARGET="${1:-all}"

print_header

if [ "${TARGET}" != "all" ]; then
    for entry in "${ALL_ALGOS[@]}"; do
        ALGO_NAME="${entry%%:*}"
        if [ "${ALGO_NAME}" = "${TARGET}" ]; then
            run_algo "${ALGO_NAME}" "${entry#*:}"
        fi
    done
else
    for entry in "${ALL_ALGOS[@]}"; do
        ALGO_NAME="${entry%%:*}"
        run_algo "${ALGO_NAME}" "${entry#*:}"
    done
fi

printf " %-12s-+-%-14s-+-%-12s-|-%s\n" "------------" "--------------" "------------" "--------"
printf "\n 详细日志: ${LOG_DIR}/\n\n"
