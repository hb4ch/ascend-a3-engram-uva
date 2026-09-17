import ctypes as C, gc, json, struct, time, hashlib, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch,torch_npu
from transformers import PreTrainedTokenizerFast
from extracted_triton import _engram_int8_gather_dequant_kernel as tri_gather, _engram_int8_dequant_kernel as tri_dequant
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent/'reference'))
from engram import EngramLayout,NgramHashState
MODEL=Path('/model');torch.set_num_threads(8);torch.npu.set_device(0)
P=C.c_void_p;U=C.c_uint64
acl=C.CDLL('libascendcl.so')
def bind(name,types):
 f=getattr(acl,name);f.argtypes=types;f.restype=C.c_int
 def call(*args):
  rc=f(*args)
  if rc:raise RuntimeError(f'{name}: {rc}')
 return call
ha=bind('aclrtMallocHost',[C.POINTER(P),U]);reg=bind('aclrtHostRegisterV2',[P,U,C.c_uint32]);alias=bind('aclrtHostGetDevicePointer',[P,C.POINTER(P),C.c_uint32]);unreg=bind('aclrtHostUnregister',[P]);hf=bind('aclrtFreeHost',[P])
lib=C.CDLL(str(ROOT/'libgather.so'));gather=lib.launch_gather;gather.argtypes=[P,P,P,P,C.c_uint32,P]
stream=torch.npu.current_stream().npu_stream

def header(path):
 with open(path,'rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];d=json.loads(f.read(n))
 k=next(k for k in d if k!='__metadata__');return k,d[k],8+n
