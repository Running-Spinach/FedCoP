#!/bin/bash
# bash ./scripts/run.sh
echo script name: $0

python exps/federated_main.py --num_classes 14 --num_users 20 --ways 5 --stdev 2 --rounds 200
