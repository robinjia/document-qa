#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 model_dir" 1>&2
  echo "/out/0x9bea85-cpu is the original docqa-confidence model." 1>&2
  exit 1
fi
PYTHONPATH=.:$PYTHONPATH  python3 docqa/run/demo_server.py "$1" -d
