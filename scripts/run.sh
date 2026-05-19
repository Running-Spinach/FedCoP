#!/bin/bash
# DPP-FL 运行脚本：5 种基线 + 提出方法 DPP-FL
# 用法: bash ./scripts/run.sh [算法名]
# 算法: dppfl (默认), fedproto, fedavg, fedprox, fedbn, scaffold
echo "Script: $0"
echo "Algorithm: ${1:-dppfl}"

ALG="${1:-dppfl}"
BASE_ARGS="--num_classes 14 --num_users 20 --ways 5 --stdev 2 --rounds 200"

case $ALG in
  # 提出方法
  dppfl)
    python exps/federated_main.py --alg dppfl ${BASE_ARGS} --ld 1.0
    ;;
  # 基线算法
  fedproto)
    python exps/federated_main.py --alg fedproto ${BASE_ARGS} --ld 1.0
    ;;
  fedavg)
    python exps/federated_main.py --alg fedavg ${BASE_ARGS} --frac 1.0
    ;;
  fedprox)
    python exps/federated_main.py --alg fedprox ${BASE_ARGS} --fedprox_mu 0.01
    ;;
  fedbn)
    python exps/federated_main.py --alg fedbn ${BASE_ARGS} --frac 1.0
    ;;
  scaffold)
    python exps/federated_main.py --alg scaffold ${BASE_ARGS}
    ;;
  *)
    echo "Unknown algorithm: $ALG"
    echo "Usage: bash ./scripts/run.sh [dppfl|fedproto|fedavg|fedprox|fedbn|scaffold]"
    exit 1
    ;;
esac
