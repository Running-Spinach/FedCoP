#!/bin/bash
# =============================================================================
# FedCoP 消融实验脚本 — 逐组件拆解,验证每个损失项 / 解码模块的贡献
# =============================================================================
# 用法:
#   bash code/scripts/ablation.sh                # 跑全部消融 × 全部 seed
#   bash code/scripts/ablation.sh A1             # 仅跑指定消融(所有 seed)
#   bash code/scripts/ablation.sh --dry-run      # 只打印命令,不执行
#   DATASET=mured bash code/scripts/ablation.sh  # 换数据集
#
# 设计:固定 BASE + FEDCOP_FLAGS(完整方法 A0),每次只改一个变量。
# 与 run.sh 共享同一套参数规模,保证可与基线表对齐。
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DRY_RUN=false
if [ "${1}" = "--dry-run" ]; then DRY_RUN=true; shift; fi

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'
CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; NC=$'\033[0m'

# ── 共享参数(与 run.sh 一致)──
# WAYS 不在此固定:按数据集取总类别数 50%(见下方 case 后)
ROUNDS=20; NUM_USERS=10; SHOTS=50; STDEV=2
LD=1.0; FRAC=0.5; PROTO_DIM=128
SEEDS=(1234)                                   # 调参阶段单 seed;定稿改 (1234 5678 9012)

DATASET=${DATASET:-chestxray14}
case ${DATASET} in
    chestxray14) NUM_CLASSES=14 ;;
    mured)       NUM_CLASSES=20 ;;
    *) echo "未知 DATASET=${DATASET}"; exit 1 ;;
esac
WAYS=$((NUM_CLASSES / 2))              # 每客户端类别数 = 总类别数 50%(chestxray→7, mured→10)

BASE_ARGS="--dataset ${DATASET} --num_classes ${NUM_CLASSES} --num_users ${NUM_USERS} \
--ways ${WAYS} --shots ${SHOTS} --stdev ${STDEV} --rounds ${ROUNDS} --frac ${FRAC} \
--ld ${LD} --proto_dim ${PROTO_DIM} --local_bs 16 --train_ep 5"

# 完整方法(A0)的 FedCoP flag
FC="--co_lambda 0.1 --cov_shrinkage 0.1 --co_beta 1.0 --co_mf_steps 2 \
--ent_lambda 1e-3 --proto_momentum 0.9 --temperature 1.0 --ld_warmup 10 \
--co_warmup 5 --fuse_alpha 0.5"

LOG_DIR="./logs/ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
RESULTS_TSV="${LOG_DIR}/_ablation.tsv"
printf "variant\tseed\tacc_proto\tacc_model\tauroc_macro\tstatus\n" > "${RESULTS_TSV}"

# ═══════════════════════════════════════════════════════════════════
#  消融变体:每次只改一个变量。格式 "id|desc|extra_flags"
# ═══════════════════════════════════════════════════════════════════
VARIANTS=(
    "A0|full                              |"
    "A1|no L_co(训练侧共现对齐)          |--no_lco"
    "A2|no structure decode(R=I 独立sig)  |--no_cooccurrence"
    "A3|point proto(dist=mse,无 ent)     |--dist_type mse --ent_lambda 0"
    "A4|no L_ent(熵正则)                 |--ent_lambda 0"
    "A5|decode=纯分类器(fuse_alpha=1)    |--fuse_alpha 1.0"
    "A6|decode=纯原型(fuse_alpha=0)      |--fuse_alpha 0.0"
    "A7|local cooc(非联邦 R)             |--local_cooc_only"
    "A8|W2 距离(替代 KL)                |--dist_type wasserstein"
    "A9|temperature=10(诊断 logit 饱和)  |--temperature 10"
)

run_one() {
    local id="$1" desc="$2" extra="$3" seed="$4"
    local log_file="${LOG_DIR}/${id}_seed${seed}.log"
    printf "  %-4s %-38s seed=%-5s ... " "${id}" "${desc}" "${seed}"
    if [ "${DRY_RUN}" = true ]; then printf "${YELLOW}[dry-run]${NC}\n"; return; fi

    SECONDS=0; local _exit=0
    python code/exps/federated_main.py ${BASE_ARGS} ${FC} --alg fedcop ${extra} \
        --seed ${seed} > "${log_file}" 2>&1 || _exit=$?
    local elapsed_min=$(awk "BEGIN {printf \"%.1f\", ${SECONDS} / 60}")

    local acc_proto=$(grep -oP 'with protos.*mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null | tail -1 || echo "")
    local acc_model=$(grep -oP 'w/o protos.*mean of per-label acc is \K[0-9.]+' "${log_file}" 2>/dev/null | tail -1 || echo "")
    local auroc=$(grep -oP 'AUROC\(macro/micro\)=\K[0-9.na]+' "${log_file}" 2>/dev/null | tail -1 || echo "")
    local use_proto="${acc_proto:--}"

    local status
    if [ "${_exit}" -ne 0 ]; then
        status="FAIL"; local err=$(grep -E 'Error:|Exception' "${log_file}" 2>/dev/null | tail -1 || echo "exit ${_exit}")
        printf "${RED}FAIL${NC} (%smin) %s\n" "${elapsed_min}" "${err:0:50}"
    elif [ -z "${acc_proto}" ]; then
        status="WARN"; printf "${YELLOW}WARN${NC} (no acc) %smin\n" "${elapsed_min}"
    else
        status="OK"; printf "${GREEN}OK${NC} acc=%s auroc=%s (%smin)\n" "${use_proto}" "${auroc:--}" "${elapsed_min}"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${id}" "${seed}" "${use_proto}" "${acc_model:--}" "${auroc:--}" "${status}" >> "${RESULTS_TSV}"
}

TARGET="${1:-all}"

printf "\n===========================================================\n"
printf " %sFedCoP Ablation%s  rounds=%d users=%d proto_dim=%d seeds={%s} dataset=%s(%d类)\n" \
    "${BOLD}" "${NC}" ${ROUNDS} ${NUM_USERS} ${PROTO_DIM} "${SEEDS[*]}" "${DATASET}" ${NUM_CLASSES}
printf " 开始: %s\n" "$(date)"
printf "===========================================================\n"

for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r id desc extra <<< "${entry}"
    echo ">>> ${id} ${desc}"
    if [ "${TARGET}" != "all" ] && [ "${TARGET}" != "${id}" ]; then continue; fi
    for seed in "${SEEDS[@]}"; do
        run_one "${id}" "${desc}" "${extra}" "${seed}"
    done
done

# ── 汇总(mean±std,仅 OK)──
printf "\n===========================================================\n"
printf " %s消融汇总(mean±std)%s\n" "${BOLD}" "${NC}"
printf " %-5s | %-38s | %-18s | %-14s\n" "ID" "变体" "Acc(proto)" "AUROC(macro)"
printf " %-5s-+-%-38s-+-%-18s-+-%s\n" "-----" "--------------------------------------" "------------------" "--------------"
awk -F'\t' '
NR>1 {
    if ($6 != "OK") { fail[$1]=1; next }
    n[$1]++; d[$1]=$2
    if ($3 != "-") { asum[$1]+=$3; asq[$1]+=$3*$3; aok[$1]=1 }
    if ($5 != "-") { usum[$1]+=$5; usq[$1]+=$5*$5; uok[$1]=1 }
}
END {
    for (a in n) {
        amean=(aok[a])?asum[a]/n[a]:-1; astd=(aok[a]&&n[a]>1)?sqrt((asq[a]/n[a]-amean*amean)*n[a]/(n[a]-1)):0
        umean=(uok[a])?usum[a]/n[a]:-1; ustd=(uok[a]&&n[a]>1)?sqrt((usq[a]/n[a]-umean*umean)*n[a]/(n[a]-1)):0
        astr=(amean>=0)?sprintf("%.4f±%.4f",amean,astd):"-"
        ustr=(umean>=0)?sprintf("%.4f±%.4f",umean,ustd):"-"
        printf " %-5s | %-38s | %-18s | %-14s\n", a, d[a], astr, ustr
    }
}' "${RESULTS_TSV}" | sort
echo ""
printf " 日志目录: ${CYAN}%s${NC}\n" "${LOG_DIR}"
printf " 原始结果: ${CYAN}%s${NC}\n" "${RESULTS_TSV}"
printf " 完成: %s\n" "$(date)"
echo "==========================================================="
