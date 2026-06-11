#!/bin/bash
# =============================================================================
# D²-FL 快速测试脚本 — 验证所有算法能正常运行
# =============================================================================
# 用法:
#   bash ./scripts/run_test.sh           # 运行全部 6 种算法
#   bash ./scripts/run_test.sh d2fl      # 仅运行指定算法
# =============================================================================

set -e

# ── 快速测试参数（大幅缩减） ──
ROUNDS=10
NUM_USERS=5
WAYS=3
SHOTS=30
STDEV=1
LD=1.0
FRAC=0.4
PROTO_DIM=64        # 缩小原型维度加速

BASE_ARGS="--num_classes 14 --num_users ${NUM_USERS} --ways ${WAYS} --shots ${SHOTS} --stdev ${STDEV} --rounds ${ROUNDS} --frac ${FRAC} --ld ${LD} --proto_dim ${PROTO_DIM}"

LOG_DIR="./logs/test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo " D²-FL 快速测试"
echo " rounds=${ROUNDS} users=${NUM_USERS} ways=${WAYS} shots=${SHOTS}"
echo " 日志: ${LOG_DIR}"
echo " 预估: 每算法约 2-5 min，总计 15-30 min"
echo "=============================================="

run_algo() {
    local name="$1"
    local args="$2"
    local log_file="${LOG_DIR}/${name}.log"

    echo ""
    echo ">>> [$(date +%H:%M:%S)] ${name} 开始..."
    echo "    python exps/federated_main.py ${args}"

    START=$(date +%s)
    python exps/federated_main.py ${args} 2>&1 | tee "${log_file}"
    ELAPSED=$(( ($(date +%s) - START) / 60 ))

    echo "  ✓ ${name} 完成 (${ELAPSED} min)"
}

ALL_ALGOS=(
    "fedavg:${BASE_ARGS} --alg fedavg"
    "fedproto:${BASE_ARGS} --alg fedproto"
    "fedgmkd:${BASE_ARGS} --alg fedgmkd --gmm_components 2"
    "fedbcs:${BASE_ARGS} --alg fedbcs"
    "fedseproto:${BASE_ARGS} --alg fedseproto --mi_lambda 0.05"
    "d2fl:${BASE_ARGS} --alg d2fl --use_distributional --dist_type kl --use_disentangle --dis_lambda 0.05 --cal_lambda 0.01 --contra_lambda 0.05 --adv_lambda 0.01 --ent_lambda 0.001 --proto_momentum 0.9 --ld_warmup 5"
)

TARGET="${1:-all}"

if [ "${TARGET}" != "all" ]; then
    echo ">>> 单算法: ${TARGET}"
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

echo ""
echo "=============================================="
echo " 测试完成 — $(date)"
echo " 日志: ${LOG_DIR}"
echo "=============================================="
