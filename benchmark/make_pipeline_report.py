from pathlib import Path
import json,re
r=Path(__file__).resolve().parent
D=json.loads((r/'pipeline-e2e-results.json').read_text())
assert D['status']=='complete' and len(D['cases'])==11
for c in D['cases']:
 assert all(x['mismatches']==0 and x.get('hash_mismatches',0)==0 for x in c['checks'].values())
 for p in c['paths'].values():
  assert len(p['samples'])==50 and len(p['traces'])==20
  assert all(abs(t['total']-sum(t['components'].values()))<.01 for t in p['traces'])
s=(r/'e2e-report-template.html').read_text(encoding='utf-8-sig')
s=s.replace("const keys=['ascendc_uva','triton_uva','cpu_stage','cpu_dequant'];", "const keys=['pipeline_uva','serial_tiled_uva','pipeline16_uva','triton_uva','ascendc_uva','cpu_stage','cpu_dequant'];")
s=re.sub(r'const labels=.*?;',"const labels=['③ UVA 流水8','③ UVA 串行8','③ UVA 流水16','③ Triton UVA','③ 原始 Ascend C','① CPU查表·NPU反量化','② CPU查表·CPU反量化'];",s)
s=re.sub(r'const colors=.*?;',"const colors=['#007e87','#a67c00','#6454b9','#df5656','#8c99a6','#68a33d','#bd6c99'];",s)
s=s.replace('D.cases.length*4','D.cases.length*keys.length')
s=re.sub(r'<p class="note warning">.*?</p>', '<p class="note warning"><strong>调用顺序已修正：</strong>旧 ctypes 调用缓存 ACL stream，绕过了 PyTorch 尚未提交的主机任务。本次每次 raw kernel 发射前重新取得当前 stream，保留任务队列；114 个边界组合通过，完整真实权重每个计时样本之后逐元素校验。此前 757 个元素异常尚不能单凭此认定为同一根因。旧版报告的计时不与本轮混用。</p>',s,count=1)
s=s.replace('不同 shape，直接比较整条 Engram 查表链路','流水化 Ascend C：整条查表链路与消融对比')
s=s.replace('(w-l-r)/4','(w-l-r)/keys.length')
s=s.replace('四条路径','七条路径').replace('e2e-results.json','pipeline-e2e-results.json').replace('bench_e2e.py','bench_pipeline_e2e.py')
s=s.replace('<tr><td>Ascend C UVA</td>', '<tr><td>流水 8 行</td><td>NPU</td><td>双缓冲、下一 tile 预取、向量 scale 广播；少于 320 行用 tile=1</td></tr><tr><td>分块串行 8 行</td><td>NPU</td><td>同样分块与向量计算，每 tile 加全流水屏障，关闭预取</td></tr><tr><td>流水 16 行</td><td>NPU</td><td>双缓冲 tile=16；少于 320 行用 tile=1</td></tr><tr><td>原始 Ascend C UVA</td>')
s=s.replace('所有性能测试前进行两层 BF16 逐元素正确性检查','每个无打点和带打点样本之后，计时区间外进行两层 BF16 逐元素正确性检查')
s=s.replace('__DATA__',json.dumps(D,ensure_ascii=False,separators=(',',':')))
(r/'report-pipeline.html').write_text(s,encoding='utf-8')
print('report bytes',len(s.encode()))
for c in D['cases']:
 print((c['batch'],c['sequence']), {k:round(v['p50']/1000,4) for k,v in c['paths'].items()})
