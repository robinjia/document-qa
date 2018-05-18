#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 model_dir input.tsv output.jsonl [--no-vec] [--reuse]" 1>&2
  echo "/out/0x9bea85-cpu is the original docqa-confidence model." 1>&2
  exit 1
fi
dir="$(dirname "$0")"
PYTHONPATH="$dir":$PYTHONPATH python3 "$dir"/docqa/run/run_batch.py $@
