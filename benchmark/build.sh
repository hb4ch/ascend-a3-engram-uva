#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if ! command -v bisheng >/dev/null; then
    echo 'Please source /usr/local/Ascend/ascend-toolkit/set_env.sh first.' >&2
    exit 1
fi

# Rebuild every library loaded by bench_support and the benchmark entry point.
bisheng -shared gather.asc -o libgather.so -fPIC --npu-arch=dav-2201
bisheng -shared gather_pipeline.asc -o libgather_pipeline.so -fPIC --npu-arch=dav-2201
bisheng -shared hash_e2e.asc -o libhash_e2e.so -fPIC --npu-arch=dav-2201

# Keep the historical measured-run provenance unchanged. Record this build separately.
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

sources = [
    'gather.asc', 'gather_pipeline.asc', 'hash_e2e.asc',
    'bench_pipeline_e2e.py', 'bench_support.py', 'test_pipeline_fresh_stream.py',
]
libraries = ['libgather.so', 'libgather_pipeline.so', 'libhash_e2e.so']
record = {
    'scope': 'Local rebuild identity; this is not a correctness or performance result.',
    'sha256': {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in sources + libraries
    },
}
Path('build-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
PY
