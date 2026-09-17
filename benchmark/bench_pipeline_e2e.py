from bench_support import *
from concurrent.futures import ThreadPoolExecutor
import os,random
SHAPES=[(1,1),(8,1),(32,1),(128,1),(512,1),(1,16),(8,16),(1,128),(8,128),(1,512),(8,512)]
PATHS=['pipeline_uva','serial_tiled_uva','pipeline16_uva','triton_uva','ascendc_uva','cpu_stage','cpu_dequant']
libp=C.CDLL(str(ROOT/'libgather_pipeline.so'));gather_config=libp.launch_gather_config;gather_config.argtypes=[P,P,P,P,C.c_uint32,C.c_uint32,C.c_uint32,C.c_uint32,P]
vector_cores=torch.npu.get_device_properties(0).vector_core_num
cfg=json.loads((MODEL/'config.json').read_text())['text_config'];args=SimpleNamespace(**cfg);args.engram_pad_id=cfg['engram_pad_token_id'];args.max_batch_size=512;args.max_seq_len=515
print('build token map',flush=True)
tok=PreTrainedTokenizerFast(tokenizer_file=str(MODEL/'tokenizer.json'));oracle=NgramHashState(args,EngramLayout.from_args(args),tok)
const=[x.npu() for x in [oracle.token_map,oracle.multipliers,oracle.primes,oracle.offsets]]
hlib=C.CDLL(str(ROOT/'libhash_e2e.so'));hash_kernel=hlib.launch_hash;hash_kernel.argtypes=[C.POINTER(P),C.c_uint32,C.c_uint32,C.c_int64,P]
report=dict(status='running',version=3,launch_boundary='Acquire current npu_stream before each raw ACL launch to flush preceding framework host tasks; task queue enabled',vector_cores=vector_cores,pipeline_description='8-row tiles, 2 input buffers, 2 output buffers, one-tile lookahead, vector scale broadcast; serial ablation identical with end-of-tile barrier',checks_after_every_e2e_sample=True,created=time.strftime('%Y-%m-%d %H:%M:%S'),host='7.208.149.157',layout='Both complete INT8 tables concurrently in pinned local DDR; both complete FP32 scales in HBM. CPU dequant baseline also reads host scale mapping.',boundary='Pinned host token IDs plus 3-token history and masks -> two layers BF16 gathered embeddings ready in HBM. Excludes projection/gate/convolution/model forward.',shapes=SHAPES,layers=[],cases=[],unit='us',cpu_threads=8,affinity=list(os.sched_getaffinity(0)),weights_path=str(MODEL),limitations=['Fixed seeded queries repeated after warmup; not random cold-table traffic.','No vLLM service, no network or RPC. Centralized single-host single-device table access.','Trace uses stream events with no intermediate synchronization; intervals include queue/launch gaps. Fused UVA read/dequant is not artificially split.','Uninstrumented E2E and instrumented breakdown are separate runs. Each trace decomposes its own directly measured total; means may be added, percentiles may not.','CPU gather is a straightforward NumPy reference, not an optimized upstream CPU kernel.','CPU dequant path uses host scales as well as the resident HBM scales kept for identical capacity.'])
tables=[]
def save(): (ROOT/'pipeline-e2e-results.json').write_text(json.dumps(report,indent=2))
class Pointer:
 dtype=torch.int8
 def __init__(self,value):self.value=value
 def data_ptr(self):return self.value
try:
 for layer in [1,14]:
  wp=MODEL/f'engram_embed_weight_l{layer}.safetensors';sp=MODEL/f'engram_embed_scale_l{layer}.safetensors';_,wh,wo=header(wp);_,sh,so=header(sp);rows=wh['shape'][0];nbytes=rows*256
  host=P();mapped=P();print('alloc full layer',layer,flush=True);ha(C.byref(host),nbytes)
  table=dict(host=host,registered=False);tables.append(table)
  arr=np.ctypeslib.as_array((C.c_int8*nbytes).from_address(host.value)).reshape(rows,256);t=time.perf_counter()
  def load_piece(worker):
   with open(wp,'rb',buffering=0) as f:
    for start in range(worker*64*1024**2,nbytes,8*64*1024**2):
     f.seek(wo+start);view=memoryview(arr.reshape(-1)[start:min(start+64*1024**2,nbytes)]).cast('B');off=0
     while off<len(view):
      got=f.readinto(view[off:]);assert got;off+=got
     if worker==0 and start%(16*1024**3)==0:print('load',layer,start//1024**3,'GiB',flush=True)
  with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(load_piece,range(8)))
  reg(host,nbytes,2);table['registered']=True;alias(host,C.byref(mapped),0)
  sm=np.memmap(sp,dtype=np.float32,mode='r',offset=so,shape=(rows,8))
  # Populate pinned scale storage concurrently; do not serialize mmap page faults in H2D.
  scale_host=P();ha(C.byref(scale_host),rows*32)
  try:
   scale_arr=np.ctypeslib.as_array((C.c_float*(rows*8)).from_address(scale_host.value)).reshape(rows,8)
   def read_scale(worker):
    with open(sp,'rb',buffering=0) as f:
     total=rows*32
     for start in range(worker*64*1024**2,total,8*64*1024**2):
      f.seek(so+start);view=memoryview(scale_arr).cast('B')[start:min(start+64*1024**2,total)];off=0
      while off<len(view):
       got=f.readinto(view[off:]);assert got;off+=got
   with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(read_scale,range(8)))
   sg=torch.from_numpy(scale_arr).to('npu');torch.npu.synchronize()
  finally:hf(scale_host)
  table.update(arr=arr,sm=sm,sg=sg,mapped=mapped,ptr=Pointer(mapped.value))
  report['layers'].append(dict(layer=layer,weight_bytes=nbytes,scale_bytes=rows*32,load_seconds=time.perf_counter()-t));save();print('full layer ready',layer,flush=True)
 rng=np.random.default_rng(20260918);order=random.Random(49)
 for b,s in SHAPES:
  print('SHAPE',b,s,flush=True);n=b*s*24
  tokens=torch.empty((b,s+3),dtype=torch.int64,pin_memory=True);tokens.copy_(torch.from_numpy(rng.integers(0,len(tok),size=(b,s+3),dtype=np.int64)))
  mask=torch.ones((b,s+3),dtype=torch.int32,pin_memory=True)
  if b>1:mask[1,1]=0
  if s>8:mask[0,6]=0
  mask_bool=mask.bool();oracle(tokens[:,:3],0,mask_bool[:,:3]);expected_ids=oracle(tokens[:,3:],3,mask_bool[:,3:]).permute(2,0,1,3).contiguous().reshape(2,n)
  td=torch.empty_like(tokens,device='npu');md=torch.empty_like(mask,device='npu');ids=torch.empty((2,n),dtype=torch.int64,device='npu');out=torch.empty((2,n,256),dtype=torch.bfloat16,device='npu')
  cid=torch.empty((2,n),dtype=torch.int64,pin_memory=True);codes=torch.empty((2,n,256),dtype=torch.int8,pin_memory=True);scales=torch.empty((2,n,8),dtype=torch.float32,pin_memory=True);bf=torch.empty((2,n,256),dtype=torch.bfloat16,pin_memory=True)
  cg=torch.empty_like(codes,device='npu');sgather=torch.empty_like(scales,device='npu');ids_np=cid.numpy();codes_np=codes.numpy();scales_np=scales.numpy()
  ptrs=(P*7)(*[x.data_ptr() for x in [td,md,*const,ids]])
  # Cache views outside timing, consistently for every path.
  idviews=[ids[i] for i in range(2)];outviews=[out[i] for i in range(2)];cgviews=[cg[i] for i in range(2)];sgviews=[sgather[i] for i in range(2)]
  def run(path,trace=False):
   cpu={};events=[]
   def cpu_phase(name,fn):
    if trace:
     st=time.perf_counter_ns();fn();cpu[name]=(time.perf_counter_ns()-st)/1000
    else:fn()
   def mark(name):
    if trace:
     ev=torch.npu.Event(enable_timing=True);ev.record();events.append((name,ev))
   start=time.perf_counter_ns()
   if path in ['ascendc_uva','triton_uva','pipeline_uva','serial_tiled_uva','pipeline16_uva']:
    mark('start');td.copy_(tokens,non_blocking=True);md.copy_(mask,non_blocking=True);mark('input_h2d')
    hash_kernel(ptrs,b,s,oracle.pad_id,torch.npu.current_stream().npu_stream);mark('device_hash')
    for li,table in enumerate(tables):
     if path=='ascendc_uva':gather(table['mapped'],table['sg'].data_ptr(),idviews[li].data_ptr(),outviews[li].data_ptr(),n,torch.npu.current_stream().npu_stream)
     elif path in ['pipeline_uva','serial_tiled_uva','pipeline16_uva']:
      tile=1 if n<320 else (16 if path=='pipeline16_uva' else 8)
      gather_config(table['mapped'],table['sg'].data_ptr(),idviews[li].data_ptr(),outviews[li].data_ptr(),n,tile,vector_cores,int(path!='serial_tiled_uva'),torch.npu.current_stream().npu_stream)
     else:tri_gather[(n,)](table['ptr'],table['sg'],idviews[li],outviews[li],n,WIDTH=256,num_warps=4)
     mark('uva_read_dequant_l'+str([1,14][li]))
   else:
    cpu_phase('cpu_hash',lambda:cid.copy_(oracle(tokens[:,3:],3,mask_bool[:,3:]).permute(2,0,1,3).reshape(2,n)))
    def collect():
     for li,table in enumerate(tables):
      np.take(table['arr'],ids_np[li],axis=0,out=codes_np[li])
      if path=='cpu_dequant':np.take(table['sm'],ids_np[li],axis=0,out=scales_np[li])
    cpu_phase('cpu_gather',collect)
    if path=='cpu_stage':
     mark('start');cg.copy_(codes,non_blocking=True);ids.copy_(cid,non_blocking=True);mark('codes_ids_h2d')
     for li,table in enumerate(tables):torch.index_select(table['sg'],0,idviews[li],out=sgviews[li])
     mark('hbm_scale_gather')
     for li in range(2):tri_dequant[(n,)](cgviews[li],sgviews[li],outviews[li],n,WIDTH=256,num_warps=4)
     mark('device_dequant')
    else:
     cpu_phase('cpu_dequant',lambda:bf.copy_((codes.float().reshape(2,n,8,32)*scales[:,:,:,None]).reshape(2,n,256)))
     mark('start');out.copy_(bf,non_blocking=True);mark('bf16_h2d')
   torch.npu.synchronize();total=(time.perf_counter_ns()-start)/1000
   if not trace:return total
   components=dict(cpu)
   for i in range(1,len(events)):components[events[i][0]]=events[i-1][1].elapsed_time(events[i][1])*1000
   components['boundary_overhead']=total-sum(components.values())
   assert components['boundary_overhead']>=-2,(path,total,components)
   return dict(total=total,components=components)
  refs=[]
  for li,table in enumerate(tables):
   ix=expected_ids[li].numpy();raw=table['arr'][ix].astype(np.float32);sc=table['sm'][ix];refs.append(torch.from_numpy((raw.reshape(n,8,32)*sc[:,:,None]).reshape(n,256)).bfloat16())
  reference=torch.stack(refs);checks={}
  for path in PATHS:
   run(path);got=out.cpu();checks[path]=dict(mismatches=int((got!=reference).sum()),max_abs_error=float((got.float()-reference.float()).abs().max()))
   if path in ['ascendc_uva','triton_uva','pipeline_uva','serial_tiled_uva','pipeline16_uva']:checks[path]['hash_mismatches']=int((ids.cpu()!=expected_ids).sum())
   if not torch.equal(got,reference):
    bad=(got!=reference).any(dim=2).nonzero();checks[path]['bad_rows']=bad.tolist()
    print('CHECK FAILED',b,s,path,checks[path],flush=True)
    torch.save(dict(shape=(b,s),path=path,ids=ids.cpu(),expected_ids=expected_ids,bad_rows=bad,got=got[bad[:,0],bad[:,1]],expected=reference[bad[:,0],bad[:,1]],tokens=tokens,mask=mask),ROOT/f'e2e-failure-{b}-{s}-{path}.pt')
    for attempt in range(3):
     run(path);torch.npu.synchronize();print('RETRY',attempt,path,int((out.cpu()!=reference).sum()),flush=True)
   else:print('CHECK OK',b,s,path,flush=True)
   if path in ['ascendc_uva','triton_uva','pipeline_uva','serial_tiled_uva','pipeline16_uva']:assert torch.equal(ids.cpu(),expected_ids)
  if any(v['mismatches'] for v in checks.values()):
   (ROOT/'e2e-failure.json').write_text(json.dumps(dict(batch=b,sequence=s,checks=checks),indent=2));raise RuntimeError('correctness failed; diagnostics preserved')
  samples={p:[] for p in PATHS};traces={p:[] for p in PATHS}
  for _ in range(5):
   for p in PATHS:run(p)
  for _ in range(50):
   ps=PATHS.copy();order.shuffle(ps)
   for p in ps:
    samples[p].append(run(p))
    assert torch.equal(out.cpu(),reference),('sample correctness',b,s,p,len(samples[p]))
  for _ in range(20):
   ps=PATHS.copy();order.shuffle(ps)
   for p in ps:
    traces[p].append(run(p,True))
    assert torch.equal(out.cpu(),reference),('trace correctness',b,s,p,len(traces[p]))
  case=dict(batch=b,sequence=s,rows_per_layer=n,output_bytes=2*n*256*2,checks=checks,paths={})
  for p in PATHS:
   v=samples[p];case['paths'][p]=dict(samples=v,mean=float(np.mean(v)),p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95)),p99=float(np.percentile(v,99)),max=float(max(v)),traces=traces[p]);print(b,s,p,'E2E p50 us',case['paths'][p]['p50'],flush=True)
  report['cases'].append(case);save()
 report['status']='complete';save();print('E2E ALL COMPLETE',flush=True)
finally:
 torch.npu.synchronize()
 for t in tables:
  if t['registered']:unreg(t['host'])
  hf(t['host'])
