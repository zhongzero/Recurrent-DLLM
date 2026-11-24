#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
launcher=("${PYTHON_BIN:-python}")
if [[ "${NPROC_PER_NODE:-1}" != 1 ]]; then
    launcher+=(-m torch.distributed.run --standalone "--nproc_per_node=$NPROC_PER_NODE")
fi
exec "${launcher[@]}" "$root_dir/src/eval_model.py" "$@"
