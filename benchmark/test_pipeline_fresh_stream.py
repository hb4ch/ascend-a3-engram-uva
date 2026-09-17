from bench_support import *
libp=C.CDLL(str(ROOT/'libgather_pipeline.so'));config=libp.launch_gather_config;config.argtypes=[P,P,P,P,C.c_uint32,C.c_uint32,C.c_uint32,C.c_uint32,P]
def launch(w,s,ids,out,n,stream):config(w,s,ids,out,n,tile,48,mode,torch.npu.current_stream().npu_stream)
rng=np.random.default_rng(101);count=65537;raw=rng.integers(-128,128,(count,256),dtype=np.int8);scales=(rng.random((count,8),dtype=np.float32)*.1);scales[0]=0
sg=torch.from_numpy(scales).npu();host=P();mapped=P();ha(C.byref(host),raw.nbytes);C.memmove(host,raw.ctypes.data,raw.nbytes);reg(host,raw.nbytes,2);alias(host,C.byref(mapped),0)
checks=[]
try:
 for turn,(tile,mode) in enumerate([(1,0),(1,1),(8,0),(8,1),(16,0),(16,1)]):
  if turn==1:
   raw=np.bitwise_not(raw);C.memmove(host,raw.ctypes.data,raw.nbytes)
  for n in [0,1,3,7,8,9,23,24,31,39,40,41,319,320,321,767,768,12288,98304]:
   ix=rng.integers(0,count,n,dtype=np.int64)
   if n:ix[:min(n,3)]=[0,count-1,0][:min(n,3)]
   ids=torch.from_numpy(ix).npu();storage=torch.full((n+4,256),99,dtype=torch.bfloat16,device='npu');out=storage[2:2+n]
   launch(mapped,sg.data_ptr(),ids.data_ptr(),out.data_ptr(),n,stream);torch.npu.synchronize()
   got=out.cpu();expect=torch.from_numpy((raw[ix].astype(np.float32).reshape(n,8,32)*scales[ix,:,None]).reshape(n,256)).bfloat16()
   bad=int((got!=expect).sum());guard=bool((storage[:2].cpu()==99).all() and (storage[-2:].cpu()==99).all());print(turn,n,bad,guard,flush=True);assert bad==0 and guard
   checks.append(dict(turn=turn,tile=tile,pipelined=mode,rows=n,mismatches=bad,guard_ok=guard))
 (ROOT/'pipeline-variants-fresh-stream.json').write_text(json.dumps(dict(status='pass',checks=checks),indent=2));print('PIPELINE SMOKE PASS',flush=True)
finally:
 torch.npu.synchronize();unreg(host);hf(host)
