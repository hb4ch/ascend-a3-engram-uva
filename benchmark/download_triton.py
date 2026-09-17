"""Download the pinned upstream Triton source and extract unchanged kernels."""
import hashlib
from pathlib import Path
import runpy
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
REV = '258a5bdf65b5b026d5e1e9a46e0d6396454272a7'
URL = f'https://raw.githubusercontent.com/vllm-project/vllm-ascend/{REV}/vllm_ascend/ops/triton/engram_int8.py'
EXPECTED = '93aba18767b4e2701d42f15eb46946e7fb5a351fe46d1365e717d0bf7684f48a'
target = ROOT / 'upstream_engram_int8.py'
data = target.read_bytes() if target.exists() else urlopen(URL, timeout=120).read()
if hashlib.sha256(data).hexdigest() != EXPECTED:
    raise RuntimeError('Upstream source SHA256 mismatch')
target.write_bytes(data)
runpy.run_path(str(ROOT / 'extract.py'))
print('Verified upstream source and extracted original Triton kernels')
