#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 trainBundle dev.json machine gpuid [--elmo]" 1>&2
  exit 1
fi
train_bundle="$1"
dev_file="$2"
host="$3"
gpuid="$4"
shift
shift
shift
shift
flags="$@"
desc="DocumentQA, test ${train_bundle} on ${dev_file}"
if [ -n "${flags}" ]; then
  desc="${desc}, ${flags}"
fi
cl work "$(cat cl_worksheet.txt)"
cl run train_bundle:"${train_bundle}" dev.json:"${dev_file}" :evaluate-v2.0.py :docqa :glove :elmo-params nltk_data:nltk_data_docqa :eval_squad.py 'export PYTHONPATH=${PYTHONPATH}:`pwd`; export CUDA_VISIBLE_DEVICES='"${gpuid}"'; mkdir -p data/lm; cd data/lm; ln -s ../../elmo-params squad-context-concat-skip; cd -; python3 docqa/run/run_json.py train_bundle/model* dev.json pred.json --na-prob-file na_prob.json --always-answer-file pred_alwaysAnswer.json '"${flags}"'; python3 evaluate-v2.0.py dev.json pred.json -o eval.json; python3 evaluate-v2.0.py dev.json pred_alwaysAnswer.json -o eval_pr.json -n na_prob.json -p plots'  --request-docker-image robinjia/tf-1.3.0-py3:1.0.1 -n "docqa-test" -d "${desc}" --request-queue host=${host}
