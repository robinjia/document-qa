#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 bundleName mode train.json machine gpuid [train flags]" 1>&2
  exit 1
fi
name="$1"
mode="$2"
train_file="$3"
host="$4"
gpuid="$5"
shift
shift
shift
shift
shift
flags="$@"
desc="DocQA ${mode}, train on ${train_file}"
if [ -n "${flags}" ]; then
  desc="${desc}, ${flags}"
fi
cl work "$(cat cl_worksheet.txt)"
cl run train-v1.1.json:"${train_file}" :dev-v1.1.json :docqa :glove nltk_data:nltk_data_docqa :eval_squad.py 'export PYTHONPATH=${PYTHONPATH}:`pwd`; export CUDA_VISIBLE_DEVICES='"${gpuid}"'; python3 docqa/squad/build_squad_dataset.py --train_file train-v1.1.json --dev_file dev-v1.1.json; python3 docqa/scripts/ablate_squad.py '"${mode} ${flags}"' model; python3 docqa/eval/squad_eval.py -o pred.json -c dev model*; python eval_squad.py dev-v1.1.json pred.json > eval.json'  --request-docker-image robinjia/tf-1.3.0-py3:1.0 -n "${name}" -d "${desc}" --request-queue host=${host}
