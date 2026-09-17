import ast
from pathlib import Path
p=Path(__file__).parent
src=(p/'upstream_engram_int8.py').read_text()
tree=ast.parse(src)
# Preserve the two upstream kernel function bodies byte-for-byte; replace package-only imports.
parts=['import triton\nimport triton.language as tl\n']
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name.startswith('_engram_'):
  start=min([node.lineno]+[x.lineno for x in node.decorator_list])-1
  parts.append('\n'.join(src.splitlines()[start:node.end_lineno]))
(p/'extracted_triton.py').write_text('\n\n'.join(parts)+'\n')
