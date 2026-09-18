# Ascend C Engram UVA 流水化实验

## 数据与范围

完整两层 INT8 表驻留 pinned DDR，完整 FP32 scale 驻留 HBM；输出 BF16 embedding 在 HBM。测试包含 CPU token 输入、设备 hash、两层 gather/dequant 和终点同步，不包含 projection、gate、卷积、完整模型或网络服务。真实权重总计约 183.11 GiB，scale 约 22.89 GiB。

## 实现

`gather_pipeline.asc` 将多行组成 tile，并使用两套输入队列和输出队列。提前提交下一 tile 的 DDR/HBM 读取，使搬运与当前 tile 的向量计算、结果写回具有重叠机会。实际重叠程度由硬件执行决定，不仅凭双缓冲声明推断收益。

- tile 支持 1、8、16 行；不足 tile 的尾部精确处理。
- 每行一次读取 256 字节 INT8 和 8 个 FP32 scale，使用向量 broadcast 替代逐 scale 标量计算。
- IDs 用 MTE2 搬入 UB 后才读；队列事件保护输入读取、输出写回和缓冲复用。
- 移除流水版每 tile 的全流水屏障，保留必要向量依赖和队列同步。
- A3 实测 48 个 Vector Core，按实际 tile 数限制启动核数。
- 保留同样分块、同样向量计算的串行版本，在每 tile 后加全流水屏障，用于区分分块收益与流水收益。

## PyTorch 与 raw ACL 调用顺序

不要长期缓存 `torch.npu.current_stream().npu_stream` 后直接通过 ctypes 发射。PyTorch 的主机任务队列可能还没有向设备 stream 提交之前的 copy/fill；raw kernel 会越过这些操作。

实验中原调用方式出现整个输出被初始化值覆盖；关闭主机任务队列后 114 个边界组合通过。保持任务队列开启、在每次 raw kernel 发射前重新获取当前 stream 后，同样 114 个组合通过。真实权重版本也复现了首次 hash 读取未就绪输入、随后重试恢复的现象。

本次 benchmark 对 hash、原始 gather、流水 gather 都采用同样的每次获取 stream 的边界。它可能增加主机提交开销，已经计入 E2E。后续集成生产算子宜使用 PyTorch NPU 正式算子任务队列封装。不要用关闭整个任务队列的方法掩盖顺序问题。

这为先前 757 个元素差异提供了线索，但不能证明那次历史异常也完全由此引起。

## 验证与公平性

`test_pipeline_fresh_stream.py`：6 个 tile/串行流水组合 × 19 个行数，共 114 个；随机 ID、重复 ID、首末行、尾块、零行、DDR 更新、输出前后 guard。

`bench_pipeline_e2e.py`：11 个 [B,S] shape × 7 条路径；每组 5 次预热、50 次直接 E2E、20 次阶段计时，交错随机执行。每个计时样本之后在计时区间外与真实权重 CPU BF16 参考逐元素比较。阶段图的 event 区间包含主机提交间隙，不应解读为纯硬件引擎忙碌时间。

CPU baseline 是官方 hash 加 NumPy gather/CPU 反量化参考，不能宣称为优化后的上游 CPU 算子。查询是固定 seed、预热后重复输入，不代表全随机冷页业务。主机同时进行权重复制任务，CPU/磁盘/内存带宽可能有背景干扰。

运行结果见 `pipeline-e2e-results.json`，图表见 `report-pipeline.html`。只有 status=complete 且校验全部通过的轮次才用于结论；cached-stream-invalid 文件仅用于故障证据。

## 复现命令（容器内）

以 [README 的完整步骤](README.md#复现) 为准。不要沿用旧实验工作目录 `/work/engram-real-benchmark` 中的 `.so`。从当前仓库根目录下载参考代码后，进入 `benchmark`，下载 Triton 参考并执行 `bash build.sh`，重新编译全部三个动态库，再依次运行边界检查、端到端 benchmark、报告生成。

`/model` 必须指向完整真实权重目录。当前结果保留的是此前已完成的实测；2026-09-18 的格式化没有触发设备复测。报告生成只依赖 Python 标准库及现有模板，可单独在本地执行。
