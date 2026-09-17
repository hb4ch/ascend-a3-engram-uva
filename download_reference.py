"""Fetch pinned official reference files; verify their SHA256 before use."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
REVISION = 'dba1be0a40aa45a94ad051997016db3960a90277'
BASE = f'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/{REVISION}'
checksums = json.loads((ROOT/'reference-checksums.json').read_text(encoding='utf-8-sig'))
for name, expected in checksums.items():
    target = ROOT/'reference'/name
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        print('Already verified:', name)
        continue
    source = 'inference/engram.py' if name == 'engram.py' else name
    data = urllib.request.urlopen(f'{BASE}/{source}', timeout=120).read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f'{name}: SHA256 mismatch: {actual}')
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix(target.suffix+'.part')
    temporary.write_bytes(data)
    temporary.replace(target)
    print('Downloaded and verified:', name)
