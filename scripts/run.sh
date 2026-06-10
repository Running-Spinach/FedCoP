#!/bin/bash
# D²-FL 运行脚本：5 种基线 + 提出方法 D²-FL
# 用法: bash ./scripts/run.sh [算法名]
# 算法: d2fl (默认), fedproto, fedavg, fedgmkd, fedbcs, fedseproto
echo "Script: $0"
echo "Algorithm: ${1:-d2fl}"

ALG="${1:-d2fl}"
BASE_ARGS="--num_classes 14 --num_users 20 --ways 5 --stdev 2 --rounds 200"

case $ALG in
  # 提出方法
  d2fl)
    python exps/federated_main.py --alg d2fl ${BASE_ARGS} --ld 1.0 \
        --use_distributional --dist_type kl --use_disentangle
    ;;
  # 基线算法
  fedproto)
    python exps/federated_main.py --alg fedproto ${BASE_ARGS} --ld 1.0
    ;;
  fedavg)
    python exps/federated_main.py --alg fedavg ${BASE_ARGS} --frac 1.0
    ;;
  fedgmkd)
    python exps/federated_main.py --alg fedgmkd ${BASE_ARGS} --ld 1.0 --gmm_components 3
    ;;
  fedbcs)
    python exps/federated_main.py --alg fedbcs ${BASE_ARGS} --ld 1.0
    ;;
  fedseproto)
    python exps/federated_main.py --alg fedseproto ${BASE_ARGS} --ld 1.0 --mi_lambda 0.05
    ;;
  *)
    echo "Unknown algorithm: $ALG"
    echo "Usage: bash ./scripts/run.sh [d2fl|fedproto|fedavg|fedgmkd|fedbcs|fedseproto]"
    exit 1
    ;;
esac
