#!/bin/bash
# =============================================================================
# FedCoP 全算法对比运行脚本 — 带详细汇总表(多 seed 平均)
# =============================================================================
# 用法:
#   bash code/scripts/run.sh                # 运行所有算法 × 所有 seed
#   bash code/scripts/run.sh fedcop         # 仅运行指定算法(所有 seed)
#   bash code/scripts/run.sh --dry-run      # 仅打印命令,不实际执行
#
# 环境假设:Linux 服务器 + NVIDIA GPU(如 4090)。直接用 python 调用。
# 多 seed:CCF-A 要求多次重复,默认 3 个 seed,结果取 mean±std。
# =============================================================================
set -e

# 切到项目根,使所有相对路径(./data, ./logs, ./protos_vis, code/exps)统一
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DRY_RUN=false
if [ "${1}" = "--dry-run" ]; then
    DRY_RUN=true
    shift
fi

# ── 颜色 ──
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
CYAN=$'\033[0;36m'
BOLD=$'\033[1m'
NC=$'\033[0m'

# ── 共享参数(4090 友好;100 轮约数小时跑完全部)──
ROUNDS=30
NUM_USERS=10
WAYS=3
SHOTS=50
STDEV=2
LD=1.0
FRAC=0.5
PROTO_DIM=128
SEEDS=(1234 2024 42)                       # 3 seed,CCF-A 标准

# ── 数据集选择(环境变量,默认 chestxray14)──
# 用法: DATASET=mured bash code/scripts/run.sh fedcop
DATASET=${DATASET:-chestxray14}
case ${DATASET} in
    chestxray14) NUM_CLASSES=14 ;;
    mured)       NUM_CLASSES=20 ;;
    *) echo "未知 DATASET=${DATASET}(支持: chestxray14 / mured)"; exit 1 ;;
esac

BASE_ARGS="--dataset ${DATASET} --num_classes ${NUM_CLASSES} --num_users ${NUM_USERS} --ways ${WAYS} --shots ${SHOTS} \
--stdev ${STDEV} --rounds ${ROUNDS} --frac ${FRAC} --ld ${LD} --proto_dim ${PROTO_DIM} \
--local_bs 16 --train_ep 5"

# FedCoP 专属默认 flag(完整方法)
FEDCOP_FLAGS="--co_lambda 0.1 --cov_shrinkage 0.1 --co_beta 1.0 --co_mf_steps 2 \
--ent_lambda 1e-3 --proto_momentum 0.9 --temperature 1.0 --ld_warmup 20"

LOG_DIR="./logs/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
START_TS=$(date)
RESULTS_TSV="${LOG_DIR}/_results.tsv"
printf "algo\tseed\tacc_proto\tacc_model\tauroc_macro\tstatus\n" > "${RESULTS_TSV}"

# ═══════════════════════════════════════════════════════════════════
#  运行单个 (算法, seed)
# ═══════════════════════════════════════════════════════════════════
run_one() {
    local name="$1"
    local seed="$2"
    local args="$3"
    local log_file="${LOG_DIR}/${name}_seed${seed}.log"

    printf "  %-16s seed=%-5s ... " "${name}" "${seed}"

    if [ "${DRY_RUN}" = true ]; then
        printf "${YELLOW}[dry-run]${NC}\n"
        return
    fi

    SECONDS=0
    local _exit=0
    python code/exps/federated_main.py ${args} --seed ${seed} > "${log_file}" 2>&1 || _exit=$?
    local elapsed_min=$(awk "BEGIN {printf \"%.1f\", ${SECONDS} / 60}")

    # ── 提取指标 ──
    local acc_proto=$(grep -oP 'with protos.*mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null | tail -1 || echo "")
    local acc_model=$(grep -oP 'w/o protos.*mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null | tail -1 || echo "")
    local acc_single=$(grep -oP 'For all users, mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null | tail -1 || echo "")
    local auroc=$(grep -oP 'AUROC\(macro/micro\)=\K[0-9.]+' "${log_file}" 2>/dev/null | tail -1 || echo "")
    # 无原型算法(fedavg/fedprox/fedgmkd/fedbcs/fedseproto)只有 acc_single
    local use_proto="${acc_proto:-${acc_single}}"
    local use_model="${acc_model:--}"

    local status
    if [ "${_exit}" -ne 0 ]; then
        status="FAIL"
        local err=$(grep -E 'Error:|Exception' "${log_file}" 2>/dev/null | tail -1 || echo "exit ${_exit}")
        printf "${RED}FAIL${NC} (%s) %s\n" "${elapsed_min}min" "${err:0:50}"
    elif [ -z "${use_proto}" ]; then
        status="WARN"
        printf "${YELLOW}WARN${NC} (no acc) %s\n" "${elapsed_min}min"
    else
        status="OK"
        printf "${GREEN}OK${NC} acc=%s auroc=%s (%smin)\n" "${use_proto}" "${auroc:--}" "${elapsed_min}"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${name}" "${seed}" "${use_proto:--}" "${use_model:--}" "${auroc:--}" "${status}" \
        >> "${RESULTS_TSV}"
}

# ═══════════════════════════════════════════════════════════════════
#  算法列表(name:args)
# ═══════════════════════════════════════════════════════════════════
declare -A ALGO_ARGS
ALGO_ARGS[fedavg]="${BASE_ARGS} --alg fedavg"
ALGO_ARGS[fedprox]="${BASE_ARGS} --alg fedprox --fedprox_mu 0.01"
ALGO_ARGS[fedproto]="${BASE_ARGS} --alg fedproto"
ALGO_ARGS[fedgmkd]="${BASE_ARGS} --alg fedgmkd --gmm_components 3"
ALGO_ARGS[fedbcs]="${BASE_ARGS} --alg fedbcs"
ALGO_ARGS[fedseproto]="${BASE_ARGS} --alg fedseproto --mi_lambda 0.05"
ALGO_ARGS[fedcop]="${BASE_ARGS} --alg fedcop ${FEDCOP_FLAGS}"
ALGO_ARGS[fedcop_nocoo]="${BASE_ARGS} --alg fedcop ${FEDCOP_FLAGS} --no_cooccurrence"
ALGO_ARGS[fedcop_local]="${BASE_ARGS} --alg fedcop ${FEDCOP_FLAGS} --local_cooc_only"
ALGO_ARGS[fedcop_nolco]="${BASE_ARGS} --alg fedcop ${FEDCOP_FLAGS} --no_lco"

# 默认顺序:基线 → 提出方法 → 消融
ALGO_ORDER=(fedavg fedprox fedproto fedgmkd fedbcs fedseproto fedcop fedcop_nocoo fedcop_local fedcop_nolco)

# ═══════════════════════════════════════════════════════════════════
#  执行
# ═══════════════════════════════════════════════════════════════════
TARGET="${1:-all}"

printf "\n===========================================================\n"
printf " %sFedCoP Benchmark%s  rounds=%d users=%d ways=%d shots=%d proto_dim=%d seeds={%s} dataset=%s(%d类)\n" \
    "${BOLD}" "${NC}" ${ROUNDS} ${NUM_USERS} ${WAYS} ${SHOTS} ${PROTO_DIM} "${SEEDS[*]}" "${DATASET}" ${NUM_CLASSES}
printf " 开始: %s\n" "${START_TS}"
printf "===========================================================\n"

if [ "${TARGET}" != "all" ]; then
    echo ">>> 单算法: ${TARGET} × ${#SEEDS[@]} seed"
    for seed in "${SEEDS[@]}"; do
        run_one "${TARGET}" "${seed}" "${ALGO_ARGS[${TARGET}]}"
    done
else
    for name in "${ALGO_ORDER[@]}"; do
        echo ">>> ${name}"
        for seed in "${SEEDS[@]}"; do
            run_one "${name}" "${seed}" "${ALGO_ARGS[${name}]}"
        done
    done
fi

# ═══════════════════════════════════════════════════════════════════
#  汇总表(跨 seed 平均:mean±std)
# ═══════════════════════════════════════════════════════════════════
printf "\n===========================================================\n"
printf " %s跨 seed 汇总(mean±std,仅 OK 运行)%s\n" "${BOLD}" "${NC}"
printf " %-16s | %-18s | %-18s | %-10s\n" "算法" "Acc(proto)" "AUROC(macro)" "状态"
printf " %-16s-+-%-18s-+-%-18s-|-%s\n" "----------------" "------------------" "------------------" "--------"

awk -F'\t' '
NR>1 {
    if ($6 != "OK") { fail[$1]=1; next }
    n[$1]++
    if ($3 != "-") { asum[$1]+=$3; asq[$1]+=$3*$3; aok[$1]=1 }
    if ($5 != "-") { usum[$1]+=$5; usq[$1]+=$5*$5; uok[$1]=1 }
}
END {
    for (a in n) {
        amean = (aok[a]) ? asum[a]/n[a] : -1
        astd  = (aok[a] && n[a]>1) ? sqrt((asq[a]/n[a]-amean*amean)*n[a]/(n[a]-1)) : 0
        umean = (uok[a]) ? usum[a]/n[a] : -1
        ustd  = (uok[a] && n[a]>1) ? sqrt((usq[a]/n[a]-umean*umean)*n[a]/(n[a]-1)) : 0
        astr = (amean>=0) ? sprintf("%.4f±%.4f", amean, astd) : "-"
        ustr = (umean>=0) ? sprintf("%.4f±%.4f", umean, ustd) : "-"
        st = (fail[a]) ? "HAS-FAIL" : "ok"
        printf " %-16s | %-18s | %-18s | %s\n", a, astr, ustr, st
    }
}' "${RESULTS_TSV}" | sort

echo ""
echo "-----------------------------------------------------------"
printf " 日志目录: ${CYAN}%s${NC}\n" "${LOG_DIR}"
printf " 原始结果: ${CYAN}%s${NC}\n" "${RESULTS_TSV}"
printf " 完成: %s\n" "$(date)"
echo "==========================================================="
