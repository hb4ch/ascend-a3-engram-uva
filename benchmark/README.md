# 完整真实权重 Engram 查表 E2E benchmark

先阅读 [当日总结](../SUMMARY-2026-09-17.md) 与 [实现说明](PIPELINE.md)。本目录为独立测试，不启动推理服务。结果仅是 token 输入到 BF16 embedding 就绪，不含后续投影、门控、卷积或整个模型。

## 复现

需要 Ascend A3 / CANN 9.1.0、配套 torch_npu 2.12.0、Triton Ascend 3.2，以及 numpy、transformers。完整真实权重只读挂载到 `/model`，包含 config、tokenizer、两层 engram_embed_weight_l{1,14}.safetensors 和 engram_embed_scale_l{1,14}.safetensors。预留超过 210 GiB 主存及超过 23 GiB HBM 和中间输出空间。

从仓库根目录：

```bash
python3 download_reference.py
cd benchmark
python3 download_triton.py
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bisheng -shared gather.asc -o libgather.so -fPIC --npu-arch=dav-2201
bisheng -shared gather_pipeline.asc -o libgather_pipeline.so -fPIC --npu-arch=dav-2201
bisheng -shared hash_e2e.asc -o libhash_e2e.so -fPIC --npu-arch=dav-2201
TASK_QUEUE_ENABLE=1 python3 test_pipeline_fresh_stream.py
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TASK_QUEUE_ENABLE=1 python3 -u bench_pipeline_e2e.py
python3 make_pipeline_report.py
```

已提交的完整结果可以直接生成报告，无需 NPU：`python3 make_pipeline_report.py`。HTML 自包含，约 385 KB，没有外部图表依赖。

第三方参考代码由固定 revision 下载并校验 SHA256，不将模型权重或 tokenizer 入库。`provenance.json` 记录本次自有实现身份，`weight-provenance.json` 标识被测权重。`bench_support.py` 仅将官方参考导入路径调整为本仓库根目录的 reference。

## 计时与检查

11 个 [B,S] shape，7 条路径，每组 5 次预热、50 次无阶段打点、20 次带事件分解，随机交错；每次计时后校验。阶段事件含发射间隙，不能当作 profiler 的引擎忙碌时间；不相加分位数。CPU baseline、UVA 与中转路径的详细分工在 HTML 中列出。

原始 Ascend C 是串行对照；流水化版本仍然在大 shape 落后于 Triton，不应以已完成生产优化的名义使用。直接 ctypes 发射必须每次取得当前 stream，不能使用 bench_support 中初始化时的缓存 stream 启动依赖 PyTorch producer 的 kernel；当前发布的两个可执行测试均使用每次获取。
