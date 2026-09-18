# 完整真实权重 Engram 查表 E2E benchmark

先阅读 [当日总结](../SUMMARY-2026-09-17.md) 与 [实现说明](PIPELINE.md)。本目录为独立测试，不启动推理服务。结果仅是 token 输入到 BF16 embedding 就绪，不含后续投影、门控、卷积或整个模型。

三种场景的算子职责、计算单元、启动数量和完整 Mermaid 流程图见 [根目录 README](../README.md)。其中 block、Triton program、CPU 线程配置是不同口径，不能互换。

## 复现

需要 Ascend A3 / CANN 9.1.0、配套 torch_npu 2.12.0、Triton Ascend 3.2，以及 numpy、transformers。完整真实权重只读挂载到 `/model`，包含 config、tokenizer、两层 engram_embed_weight_l{1,14}.safetensors 和 engram_embed_scale_l{1,14}.safetensors。预留超过 210 GiB 主存及超过 23 GiB HBM 和中间输出空间。

从仓库根目录：

```bash
python3 download_reference.py
cd benchmark
python3 download_triton.py
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh
TASK_QUEUE_ENABLE=1 python3 test_pipeline_fresh_stream.py
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 TASK_QUEUE_ENABLE=1 python3 -u bench_pipeline_e2e.py
python3 make_pipeline_report.py
```

已提交的完整结果可以直接生成报告，无需 NPU：`python3 make_pipeline_report.py`。HTML 自包含，约 385 KB，没有外部图表依赖。

第三方参考代码由固定 revision 下载并校验 SHA256，不将模型权重或 tokenizer 入库。`provenance.json` 记录本次自有实现身份，`weight-provenance.json` 标识被测权重。`bench_support.py` 仅将官方参考导入路径调整为本仓库根目录的 reference。

## 计时与检查

11 个 [B,S] shape，7 条路径，每组 5 次预热、50 次无阶段打点、20 次带事件分解，随机交错；每次计时后校验。阶段事件含发射间隙，不能当作 profiler 的引擎忙碌时间；不相加分位数。CPU baseline、UVA 与中转路径的详细分工在 HTML 中列出。

原始 Ascend C 是串行对照；流水化版本仍然在大 shape 落后于 Triton，不应以已完成生产优化的名义使用。直接 ctypes 发射必须每次取得当前 stream，不能使用 bench_support 中初始化时的缓存 stream 启动依赖 PyTorch producer 的 kernel；当前发布的两个可执行测试均使用每次获取。

## 源码与复现结果的对应关系

2026-09-18 对四个 AscendC 文件做了可读性整理，增加 `.clang-format`，拆开声明、补齐花括号并补充注释。本次未重跑 NPU，仓库中的 JSON/HTML 仍是此前实测结果，不能将它们标成此次源码重编译后的验证结果。

- `../engram.asc`：早期功能 demo，设备实际计算 hash，但通过二分映射访问 720 行合成测试表，使用每行一个 scale，输出 FP32。运行入口是根目录 `bash run.sh`。
- `hash_e2e.asc`：真实权重 benchmark 的设备 hash。输入包含当前 token 和前 3 个 token 的历史，输出两层各 24 个行号。
- `gather.asc`：真实完整表的串行 AscendC 对照。
- `gather_pipeline.asc`：真实完整表的分块串行/流水版本，按行号直接取 INT8 行，每 32 个元素使用一个 FP32 scale，输出 BF16，没有 demo 的二分映射。

更新源码后必须执行 `bash build.sh`，避免 Python 继续加载旧 `.so`。即使只运行 `test_pipeline_fresh_stream.py`，也需要 `libgather.so`，因为 `bench_support.py` 会加载它。`build-manifest.json` 记录当前源码和编译产物的 SHA256；`provenance.json` 保留历史版本身份，不随重新格式化而覆盖。

测试会覆盖当前目录的 `pipeline-variants-fresh-stream.json`、`pipeline-e2e-results.json` 和生成的 `report-pipeline.html`。复测前请备份旧文件，完成后将本轮 `build-manifest.json` 与新结果一起保存。功能检查失败时不要继续性能测试。

格式检查（需要 clang-format 21）：

```bash
# 仓库根目录
clang-format --dry-run --Werror engram.asc benchmark/*.asc
git diff --check
```
