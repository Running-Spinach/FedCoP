#!/bin/bash
# =============================================================================
# FedCoP 快速烟测脚本 — 精简但报错清晰(单 seed,验证通路)
# =============================================================================
# 用法:
#   bash code/scripts/run_test.sh           # 两个数据集 × 全部算法(含 FedCoP 消融)
#   bash code/scripts/run_test.sh fedcop    # 两个数据集 × 仅指定算法
#   DATASETS=mured bash code/scripts/run_test.sh           # 仅指定数据集(子集)
#   DATASETS="chestxray14 mured" bash code/scripts/run_test.sh
#
# 目的:快速 Bug 检查(4090 上数十分钟跑完全部),非最终结果。
# 最终结果请用 scripts/run.sh(多 seed)。
# =============================================================================
set -e

# 切到项目根,使所有相对路径(./data, ./logs, ./protos_vis, code/exps)统一
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── 数据集列表(默认两个数据集都跑,可用 DATASETS 覆盖)──
DATASETS=${DATASETS:-"chestxray14 mured"}

# ── 颜色 ──
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
CYAN=$'\033[0;36m'
NC=$'\033[0m'

# ── 烟测参数(极小规模,只为验证不崩)──
# WAYS 不在此固定:按数据集取总类别数 50%(见逐数据集循环)
ROUNDS=3
NUM_USERS=5
SHOTS=20
STDEV=1
LD=1.0
FRAC=0.4
PROTO_DIM=32
SEED=1234

# ── 数据集 → 类别数映射(在循环内按数据集取值)──
num_classes_for() {
    case "$1" in
        chestxray14) echo 14 ;;
        mured)       echo 20 ;;
        *) echo "" ;;
    esac
}

# BASE_ARGS 在循环内按数据集拼装(含 --dataset / --num_classes)
make_base_args() {
    local ds="$1"
    local nc
    nc=$(num_classes_for "${ds}")
    [ -z "${nc}" ] && { echo "未知 DATASET=${ds}(支持: chestxray14 / mured)"; return 1; }
    echo "--dataset ${ds} --num_classes ${nc} --num_users ${NUM_USERS} --ways ${WAYS} --shots ${SHOTS} \
--stdev ${STDEV} --rounds ${ROUNDS} --frac ${FRAC} --ld ${LD} --proto_dim ${PROTO_DIM} \
--train_ep 5 --seed ${SEED}"
}

FEDCOP_FLAGS="--co_lambda 0.1 --cov_shrinkage 0.1 --co_beta 1.0 --co_mf_steps 2 \
--ent_lambda 1e-3 --proto_momentum 0.9 --temperature 1.0 --ld_warmup 2 \
--co_warmup 0 --fuse_alpha 0.5"

LOG_ROOT="./logs/test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_ROOT}"

PASS_COUNT=0
FAIL_COUNT=0
declare -a FAIL_NAMES FAIL_MSGS

print_header() {
    echo ""
    echo "=============================================="
    echo " FedCoP 快速烟测 [${DATASET}(${NUM_CLASSES}类)]"
    echo " rounds=${ROUNDS} users=${NUM_USERS} ways=${WAYS} shots=${SHOTS} proto_dim=${PROTO_DIM} seed=${SEED}"
    echo " 日志: ${LOG_DIR}"
    echo "=============================================="
    printf " %-16s | %-8s | %-8s | %-12s | %s\n" "算法" "状态" "Acc" "耗时" "备注"
    printf " %-16s-+-%-8s-+-%-8s-+-%-12s-|-%s\n" "----------------" "--------" "--------" "------------" "----------"
}

run_algo() {
    local name="$1"
    local args="$2"
    local log_file="${LOG_DIR}/${name}.log"

    printf " %-16s | ${CYAN}%-8s${NC} | %-8s | ...         |" "${name}" "running" "..."

    SECONDS=0
    _exit=0
    python code/exps/federated_main.py ${args} > "${log_file}" 2>&1 || _exit=$?
    local elapsed_min=$(awk "BEGIN {printf \"%.1f\", ${SECONDS} / 60}")

    local acc_proto=$(grep -oP 'with protos.*mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null | tail -1 || echo "")
    local acc_model=$(grep -oP 'w/o protos.*mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null | tail -1 || echo "")
    local acc_single=$(grep -oP 'For all users, mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null | tail -1 || echo "")
    local auroc=$(grep -oP 'AUROC\(macro/micro\)=\K[0-9.]+' "${log_file}" 2>/dev/null | tail -1 || echo "")

    if [ "${_exit}" -ne 0 ]; then
        local err_line=$(grep -E 'Error:|Exception' "${log_file}" 2>/dev/null | tail -1 || echo "exit ${_exit}")
        printf "\r %-16s | ${RED}%-8s${NC} | %-8s | %-12s | ${RED}%s${NC}\n" \
            "${name}" "FAIL" "-" "${elapsed_min}min" "${err_line:0:40}"
        FAIL_NAMES+=("${name}"); FAIL_MSGS+=("${err_line}")
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo -e "  ${YELLOW}日志: ${log_file}${NC}"
        return
    fi

    local disp_acc="${acc_proto:-${acc_single}}"
    if [ -z "${disp_acc}" ]; then
        printf "\r %-16s | ${YELLOW}%-8s${NC} | %-8s | %-12s | no acc\n" "${name}" "WARN" "-" "${elapsed_min}min"
        FAIL_NAMES+=("${name}"); FAIL_MSGS+=("no acc output")
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return
    fi

    local note=""
    [ -n "${auroc}" ] && note="auroc=${auroc}"
    [ -n "${acc_model}" ] && note="${note} w/o=${acc_model}"
    printf "\r %-16s | ${GREEN}%-8s${NC} | %-8s | %-12s | %s\n" "${name}" "OK" "${disp_acc}" "${elapsed_min}min" "${note}"
    PASS_COUNT=$((PASS_COUNT + 1))
}

# ═══════════════════════════════════════════════════════════════════
#  算法列表(只存算法专属 flag;BASE_ARGS 按数据集在运行时拼装)
# ═══════════════════════════════════════════════════════════════════
ALL_ALGOS=(
    "fedavg:--alg fedavg"
    "fedprox:--alg fedprox --fedprox_mu 0.01"
    "fedproto:--alg fedproto"
    "fedgmkd:--alg fedgmkd --gmm_components 2"
    "fedseproto:--alg fedseproto --mi_lambda 0.05"
    "fedcop:--alg fedcop ${FEDCOP_FLAGS}"
    "fedcop_nocoo:--alg fedcop ${FEDCOP_FLAGS} --no_cooccurrence"
    "fedcop_local:--alg fedcop ${FEDCOP_FLAGS} --local_cooc_only"
    "fedcop_nolco:--alg fedcop ${FEDCOP_FLAGS} --no_lco"
)

TARGET="${1:-all}"

# ═══════════════════════════════════════════════════════════════════
#  逐数据集执行
# ═══════════════════════════════════════════════════════════════════
for DATASET in ${DATASETS}; do
    NUM_CLASSES=$(num_classes_for "${DATASET}")
    if [ -z "${NUM_CLASSES}" ]; then
        echo "${RED}未知 DATASET=${DATASET}(支持: chestxray14 / mured),跳过${NC}"
        continue
    fi
    WAYS=$((NUM_CLASSES / 2))          # 每客户端类别数 = 总类别数 50%(chestxray→7, mured→10)

    BASE_ARGS=$(make_base_args "${DATASET}") || exit 1
    LOG_DIR="${LOG_ROOT}/${DATASET}"
    mkdir -p "${LOG_DIR}"

    echo ""
    echo "${CYAN}########## 数据集: ${DATASET} (${NUM_CLASSES} 类) ##########${NC}"
    print_header

    if [ "${TARGET}" != "all" ]; then
        for entry in "${ALL_ALGOS[@]}"; do
            ALGO_NAME="${entry%%:*}"
            [ "${ALGO_NAME}" = "${TARGET}" ] && run_algo "${ALGO_NAME}" "${BASE_ARGS} ${entry#*:}"
        done
    else
        for entry in "${ALL_ALGOS[@]}"; do
            run_algo "${entry%%:*}" "${BASE_ARGS} ${entry#*:}"
        done
    fi

    printf " %-16s-+-%-8s-+-%-8s-+-%-12s-|-%s\n" "----------------" "--------" "--------" "------------" "----------"
done

# ═══════════════════════════════════════════════════════════════════
#  汇总(跨两个数据集)
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "=============================================="
echo " 结果: ${GREEN}${PASS_COUNT} 通过${NC}  ${RED}${FAIL_COUNT} 失败${NC}  (共 $((PASS_COUNT + FAIL_COUNT)) 个)"
echo " 数据集: ${DATASETS}"
echo " 日志根目录: ${LOG_ROOT}"
echo "=============================================="

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo ""
    echo -e " ${RED}失败汇总:${NC}"
    for i in $(seq 0 $((FAIL_COUNT - 1))); do
        echo -e "   ${RED}✗${NC} ${FAIL_NAMES[$i]} — ${FAIL_MSGS[$i]}"
    done
    echo ""
fi
