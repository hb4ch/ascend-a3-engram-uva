#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${ASCEND_HOME_PATH:=/usr/local/Ascend/ascend-toolkit/latest}"
if ! command -v bisheng >/dev/null; then
  echo 'Please source the CANN set_env.sh first.' >&2
  exit 1
fi
bisheng -shared engram.asc -o libengram_uva.so -fPIC --npu-arch=dav-2201
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 test_engram.py
