#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 trainBundle machine gpuid" 1>&2
  exit 1
fi
trainBundle="$1"
host="$2"
gpuid="$3"
desc="DocumentQA, Convert to CPU, ${trainBundle}"
cl work "$(cat cl_worksheet.txt)"
cl run traindir:${trainBundle} :docqa 'export PYTHONPATH=${PYTHONPATH}:`pwd`; export CUDA_VISIBLE_DEVICES='"${gpuid}"'; ln -s traindir/model* .; python3 docqa/scripts/convert_to_cpu.py model* cpu_model'  --request-docker-image robinjia/tf-1.3.0-py3:1.0 -n "docqa-cpu" -d "${desc}" --request-queue host=${host}
