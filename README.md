# Ascend A3 Engram UVA demo

在 Ascend A3 上验证：Device 计算 DeepSeek-V4.1-Flash Engram hash，直接读取映射的主机 DDR int8 表，将选中行 gather 到 HBM，并使用 HBM 中的 scale 反量化。

这是已运行验证的功能原型，不是完整 Engram 模块，也不是优化后的性能实现。

## 数据路径

```text
HBM: token IDs、mask、token 压缩映射、hash 常数
  → A3 Device: 计算 2/3/4-gram hash
  → 根据真实逻辑行号查找测试表中的物理行
  → pinned Host DDR: 读取 int8 行（256 bytes/row）到 UB
  → HBM: 输出 gather 后的 int8 行
  → 使用 HBM 中的 FP32 scale，输出反量化 FP32 行
```

Host 使用 `aclrtMallocHost` 分配 pinned memory，再调用 `aclrtHostRegisterV2(..., ACL_HOST_REG_MAPPED)` 和 `aclrtHostGetDevicePointer`。Kernel 收到的是映射后的 Device 地址。CPU 地址与 Device 地址不要求相同。全表不做 H2D 拷贝，只有选中的行进入算子的片上 UB 和输出 HBM。

## Hash 与测试数据

参考 [DeepSeek 官方 engram.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/engram.py)。下载脚本固定模型 revision，并逐文件验证 SHA256；第三方源文件和 tokenizer 不随本仓库分发。

- 使用真实 tokenizer，归一化压缩词表大小为 99092。
- 层号 1、14；2/3/4-gram 各 8 个 head；每层每 token 查询 24 行。
- Host 根据官方实现生成压缩映射、乘数、质数和偏移；Device 执行实际 token hash。
- padding、mask 引起的历史截断，以及 lookback 顺序遵循官方实现。

测试只物理存储假 token 命中的 **720 行**，按 `(layer, logical_hash)` 排序。Device 二分查找该稀疏测试表；没有二次取模，也没有更改官方 hash 行号。生产版完整表或分片表需替换这层测试映射。当前输入是两个各 8 token 的序列。

int8 数据为确定性合成数据，不是 checkpoint 权重。量化约定为每行一个 FP32 scale：`value = int8 * scale`，不代表原模型 checkpoint 的量化布局。此 demo 不包含后续 projection、gating、卷积或完整模型精度验证。

## 运行

已验证环境：Linux aarch64、Ascend910_9392（A3）、CANN 9.1.0、PyTorch 2.12.0。需要 CANN 自带 bisheng、Ascend C 头文件和 AscendCL 动态库。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 使用已有 CANN/PyTorch 环境；requirements.txt 列出 Python 依赖。
python3 download_reference.py
bash run.sh
```

默认使用逻辑 Device 0。Kernel 构建目标为 `dav-2201`。测试只释放自己的内存和 stream，不调用设备 reset。输出 `ALL PASS` 表示逐项比对成功；详细结果写入 `result.json`。

## 已验证结果

[数值证据](results/a3-functional.json)：8 组执行全部通过。

| 检查 | 结果 |
|---|---|
| Device hash 与官方 CPU oracle | 0 mismatch |
| DDR int8 → HBM gather | 0 mismatch |
| HBM scale 反量化 | 最大绝对误差 0 |
| 整段与 3+4+1 分段输入 | 一致 |
| 序列起点、重复 token、mask 边界 | 通过 |
| 同步后 CPU 修改 DDR，再运行 | 读到更新值 |

共校验 3072 个 hash，int8 与 FP32 输出各校验 786432 个元素。表占主存 184320 bytes，scale 占 HBM 2880 bytes。原型含标量 hash/反量化和顺序小块 gather，尚未报告带宽或时延。

测试严格在 stream 完成后才修改或释放 Host 表，不能由此推断 CPU/NPU 无同步并发更新安全。分段测试提供完整 token 历史，kernel 只读取当前及以前的位置；生产请求历史状态管理不在本 demo 范围内。

## 文件

- `engram.asc`：Device hash、DDR gather、反量化。
- `test_engram.py`：官方 oracle、内存映射、测试与清理。
- `download_reference.py` / `reference-checksums.json`：可追溯的官方参考下载。
- `run.sh`：编译与运行。

## 2026-09-17：真实权重 E2E 与流水化实验

新增 [当日工作总结](SUMMARY-2026-09-17.md)、[完整性能报告](benchmark/report-pipeline.html) 和 [复现步骤](benchmark/README.md)。包含两层完整真实表、CPU/NPU 分工、七条路径的 shape 对比与阶段拆分、双缓冲消融，以及 raw ACL 发射顺序问题的修正。流水版大 shape 仍落后于 Triton，具体限制见总结。

## 更新源码后的复现

根目录 `bash run.sh` 会重新编译 `engram.asc` 再执行合成小表功能验证。真实权重性能测试请使用 [benchmark 复现步骤](benchmark/README.md#复现)，先执行 `bash build.sh` 重建三个动态库，不能使用根目录小表 demo 代替。

2026-09-18 的修改仅整理 AscendC 可读性及复现入口，尚未进行新的设备运行。历史结果与 provenance 保持原样，重新构建的源码及动态库身份单独记录在 `benchmark/build-manifest.json`。
