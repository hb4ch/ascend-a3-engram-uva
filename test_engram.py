"""Official DeepSeek hash oracle vs A3 hash + mapped DDR int8 gather."""
import ctypes as C
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reference'))
from engram import EngramLayout,NgramHashState

cfg=json.loads((ROOT/'reference/config.json').read_text())['text_config']
args=SimpleNamespace(**cfg)
args.engram_pad_id=cfg['engram_pad_token_id'];args.max_batch_size=2;args.max_seq_len=16
tok=PreTrainedTokenizerFast(tokenizer_file=str(ROOT/'reference/tokenizer.json'))
layout=EngramLayout.from_args(args);oracle=NgramHashState(args,layout,tok)
tokens=np.array([[0,17,250,250,70001,129279,1,2],[2,999,45000,7,7,123456,128799,17]],dtype=np.int64)
mask=np.ones_like(tokens,dtype=np.int32);mask[1,3]=0
expected=oracle(torch.from_numpy(tokens),0,torch.from_numpy(mask.astype(bool))).numpy()
split_oracle=NgramHashState(args,layout,tok)
split=torch.cat([split_oracle(torch.from_numpy(tokens[:,s:e]),s,torch.from_numpy(mask[:,s:e].astype(bool))) for s,e in [(0,3),(3,7),(7,8)]],dim=1).numpy()
assert np.array_equal(expected,split)
keys=np.unique(expected+np.array([0,1000000000],dtype=np.int64)[None,None,:,None])
cols=np.arange(256,dtype=np.int64)[None,:]
table=((keys[:,None]*17+cols*13)%255-127).astype(np.int8)
scales=((keys%31+1)/128).astype(np.float32)

acl=C.CDLL('libascendcl.so');kernel=C.CDLL(str(ROOT/'libengram_uva.so'))
P=C.c_void_p;U=C.c_uint64;I=C.c_int
def bind(name,types):
 f=getattr(acl,name);f.argtypes=types;f.restype=I
 def call(*a):
  rc=f(*a)
  if rc:raise RuntimeError(f'{name}: {rc}')
 return call
init=bind('aclInit',[P]);setdev=bind('aclrtSetDevice',[I]);create=bind('aclrtCreateStream',[C.POINTER(P)])
malloc=bind('aclrtMalloc',[C.POINTER(P),U,I]);hostalloc=bind('aclrtMallocHost',[C.POINTER(P),U])
reg=bind('aclrtHostRegisterV2',[P,U,C.c_uint32]);alias=bind('aclrtHostGetDevicePointer',[P,C.POINTER(P),C.c_uint32])
copy=bind('aclrtMemcpy',[P,U,P,U,I]);sync=bind('aclrtSynchronizeStream',[P]);free=bind('aclrtFree',[P])
allocs=[];stream=P();host=P();mapped=P();registered=False
def hbm(a):
 a=np.ascontiguousarray(a);p=P();malloc(C.byref(p),a.nbytes,2);allocs.append(p)
 copy(p,a.nbytes,P(a.ctypes.data),a.nbytes,1);return p
def fetch(p,shape,dtype):
 a=np.empty(shape,dtype=dtype);copy(P(a.ctypes.data),a.nbytes,p,a.nbytes,2);return a
init(None);setdev(0);create(C.byref(stream))
try:
 hostalloc(C.byref(host),table.nbytes);C.memmove(host,table.ctypes.data,table.nbytes)
 reg(host,table.nbytes,2);registered=True;alias(host,C.byref(mapped),0)
 inputs=[hbm(tokens),hbm(mask),hbm(oracle.token_map.numpy()),hbm(oracle.multipliers.numpy()),hbm(oracle.primes.numpy().reshape(2,24)),hbm(oracle.offsets.numpy()),hbm(keys),hbm(scales),mapped]
 launch=kernel.launch_engram;launch.argtypes=[C.POINTER(P),C.c_uint32,C.c_uint32,C.c_uint32,C.c_uint32,C.c_uint32,C.c_int64,P];launch.restype=None
 results=[]
 for pass_id in range(2):
  if pass_id:
   table=(-table.astype(np.int16)).astype(np.int8);C.memmove(host,table.ctypes.data,table.nbytes)
  for start,count in [(0,8),(0,3),(3,4),(7,1)]:
   shape=(2,count,2,24);outHash=hbm(np.full(shape,-1,dtype=np.int64))
   outRaw=hbm(np.zeros((*shape,256),dtype=np.int8));outVal=hbm(np.zeros((*shape,256),dtype=np.float32))
   ptrs=(P*12)(*(inputs+[outHash,outRaw,outVal]))
   launch(ptrs,2,8,start,count,len(keys),oracle.pad_id,stream);sync(stream)
   hashes=fetch(outHash,shape,np.int64);raw=fetch(outRaw,(*shape,256),np.int8);values=fetch(outVal,(*shape,256),np.float32)
   ref=expected[:,start:start+count];slots=np.searchsorted(keys,ref+np.array([0,1000000000])[None,None,:,None])
   refRaw=table[slots];refValues=refRaw.astype(np.float32)*scales[slots,None]
   row=dict(pass_id=pass_id,start=start,count=count,hash_mismatches=int(np.count_nonzero(hashes!=ref)),int8_mismatches=int(np.count_nonzero(raw!=refRaw)),dequant_mismatches=int(np.count_nonzero(values!=refValues)),max_abs_error=float(np.max(np.abs(values-refValues))))
   print(json.dumps(row),flush=True);results.append(row)
   assert not any(row[k] for k in ['hash_mismatches','int8_mismatches','dequant_mismatches'])
   for p in [outHash,outRaw,outVal]:free(p);allocs.remove(p)
 report=dict(config={k:v for k,v in cfg.items() if k.startswith('engram')},tokens=tokens.tolist(),mask=mask.tolist(),host_pointer=host.value,device_alias=mapped.value,fixture_rows=len(keys),ddr_bytes=table.nbytes,scale_hbm_bytes=scales.nbytes,scale_format='synthetic FP32 per row; dequant=int8*scale',fixture='sparse physical rows keyed by exact official logical hash; not checkpoint weights',official_chunk_consistency=True,reference_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'reference').glob('*') if p.is_file()},sample_hashes=expected[0,0].tolist(),results=results)
 (ROOT/'result.json').write_text(json.dumps(report,indent=2));print('ALL PASS',flush=True)
finally:
 sync(stream)
 for p in allocs:free(p)
 if registered:bind('aclrtHostUnregister',[P])(host)
 if host.value:bind('aclrtFreeHost',[P])(host)
 bind('aclrtDestroyStream',[P])(stream) # Only release this process resources; no device reset.
