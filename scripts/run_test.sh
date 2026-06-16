#!/bin/bash
# =============================================================================
# D²-FL 快速测试脚本 — 精简但报错清晰
# =============================================================================
# 用法:
#   bash ./scripts/run_test.sh           # 运行全部 6 种算法
#   bash ./scripts/run_test.sh d2fl      # 仅运行指定算法
# =============================================================================

set -e

# ── 颜色 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ── 快速 Bug 检查参数（4090 上 ~30-50 min 跑完 6 算法）──
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

# 全局追踪
PASS_COUNT=0
FAIL_COUNT=0
declare -a FAIL_NAMES
declare -a FAIL_MSGS

# ═══════════════════════════════════════════════════════════════════
#  表头
# ═══════════════════════════════════════════════════════════════════

print_header() {
    echo ""
    echo "=============================================="
    echo " D²-FL 快速测试"
    echo " rounds=${ROUNDS}  users=${NUM_USERS}  ways=${WAYS}  shots=${SHOTS}  proto_dim=${PROTO_DIM}"
    echo " 日志: ${LOG_DIR}"
    echo "=============================================="
    printf " %-12s | %-10s | %-8s | %-14s | %s\n" "算法" "状态" "Acc" "耗时" "备注"
    printf " %-12s-+-%-10s-+-%-8s-+-%-14s-|-%s\n" "------------" "----------" "--------" "--------------" "----------"
}

# ═══════════════════════════════════════════════════════════════════
#  运行单个算法并提取关键结果
# ═══════════════════════════════════════════════════════════════════

run_algo() {
    local name="$1"
    local args="$2"
    local log_file="${LOG_DIR}/${name}.log"

    # 启动提示
    printf " %-12s | ${CYAN}%-10s${NC} | %-8s | ...          |" "${name}" "running..." "..."

    # 运行
    SECONDS=0
    _exit=0
    python exps/federated_main.py ${args} > "${log_file}" 2>&1 || _exit=$?
    local elapsed_min=$(awk "BEGIN {printf \"%.1f\", ${SECONDS} / 60}")

    # ── 提取准确率 ──
    local acc_proto=$(grep -oP 'with protos.*mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null || echo "")
    local acc_model=$(grep -oP 'w/o protos.*mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null || echo "")
    local acc_single=$(grep -oP 'For all users, mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null || echo "")

    # ── 判断成功/失败 ──
    if [ "${_exit}" -ne 0 ]; then
        # Python 非零退出
        local err_tail=$(tail -5 "${log_file}" 2>/dev/null)
        # 提取最后一行 Traceback 中实际有用的错误
        local err_line=$(grep -E '^[A-Za-z]+Error:|^ModuleNotFoundError:|^NameError:|^ValueError:|^TypeError:|^AttributeError:|^FileNotFoundError:' "${log_file}" 2>/dev/null | tail -1 || echo "exit code ${_exit}")

        printf "\r %-12s | ${RED}%-10s${NC} | %-8s | %-14s | ${RED}%s${NC}\n" \
            "${name}" "FAIL" "-" "${elapsed_min} min" "${err_line:-(see log)}"

        FAIL_NAMES+=("${name}")
        FAIL_MSGS+=("${err_line:-(see log)}")
        ((FAIL_COUNT++))

        # 打印错误详情
        echo ""
        echo -e "  ${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "  ${RED}[ERROR] ${name} 失败 (exit ${_exit}, ${elapsed_min} min)${NC}"
        echo -e "  ${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${err_tail}" | while IFS= read -r line; do
            echo -e "  ${RED}|${NC} ${line}"
        done
        echo -e "  ${YELLOW}完整日志: ${log_file}${NC}"
        echo ""
        return
    fi

    if [ -z "${acc_proto}" ] && [ -z "${acc_model}" ] && [ -z "${acc_single}" ]; then
        # Python 正常退出但没提取到准确率
        local err_tail=$(tail -5 "${log_file}" 2>/dev/null)

        printf "\r %-12s | ${YELLOW}%-10s${NC} | %-8s | %-14s | no acc output\n" \
            "${name}" "WARN" "-" "${elapsed_min} min"

        FAIL_NAMES+=("${name}")
        FAIL_MSGS+=("no accuracy output — possible silent failure")
        ((FAIL_COUNT++))

        echo ""
        echo -e "  ${YELLOW}[WARN] ${name} 运行完成但未输出准确率 (${elapsed_min} min)${NC}"
        echo -e "  ${YELLOW}完整日志: ${log_file}${NC}"
        echo ""
        return
    fi

    # ── 成功 ──
    local disp_acc="${acc_proto}"
    local disp_note=""
    if [ -z "${disp_acc}" ]; then
        disp_acc="${acc_single}"
    fi
    if [ -n "${acc_model}" ] && [ "${acc_model}" != "" ]; then
        disp_note="w/o proto: ${acc_model}"
    fi

    printf "\r %-12s | ${GREEN}%-10s${NC} | %-8s | %-14s | %s\n" \
        "${name}" "OK" "${disp_acc}" "${elapsed_min} min" "${disp_note}"

    ((PASS_COUNT++))
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

# ═══════════════════════════════════════════════════════════════════
#  汇总
# ═══════════════════════════════════════════════════════════════════

printf " %-12s-+-%-10s-+-%-8s-+-%-14s-|-%s\n" "------------" "----------" "--------" "--------------" "----------"
echo ""
echo "=============================================="
echo " 结果: ${GREEN}${PASS_COUNT} 通过${NC}  ${RED}${FAIL_COUNT} 失败${NC}  (共 $((PASS_COUNT + FAIL_COUNT)) 个算法)"
echo " 日志: ${LOG_DIR}"
echo "=============================================="

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo ""
    echo -e " ${RED}失败汇总:${NC}"
    for i in $(seq 0 $((FAIL_COUNT - 1))); do
        echo -e "   ${RED}✗${NC} ${FAIL_NAMES[$i]} — ${FAIL_MSGS[$i]}"
    done
    echo ""
fi
