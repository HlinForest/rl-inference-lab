# RL + Inference Systems Lab

用 6 周、42 个学习日，把 GPU 基础、训练、推理和 RL 更新串成一个能解释、能测试、能测量的小系统。每天约 2–3 小时；对已有 Python / PyTorch / 概率基础的学习者，这是第一轮学习范围，不是精通分布式 RL 的承诺。

**Credit:** 阅读清单来自 [David Bellamy (@DavidRBellamy) 的原帖](https://x.com/DavidRBellamy/status/2110839750060065216)及[博客补充帖](https://x.com/DavidRBellamy/status/2110839752396337192)。42 天安排、中文评审与教学实验是本仓库的独立改编，不代表原作者参与或背书。见 [CREDITS](CREDITS.md)。

## 从这里开始

1. 看 [课程评审](docs/review.md)：主线通过，已修正资料版本、评估和纠偏边界。
2. 打开 [D01：tensor 存储与权重快照](docs/day01.md)，先预测，再运行。
3. 按 [42 天计划](docs/curriculum.md) 推进，用 [进度表](PROGRESS.md) 记录真正完成的内容。

**当前状态：** D01 参考实现可运行，8 项 CPU 测试通过。D02–D42 尚未实现；个人学习验收尚未勾选。CUDA / vLLM / 多卡性能均待测。

## 快速运行

已安装兼容 PyTorch 时，在仓库根目录运行：

```sh
python -m kernels.tensor_layout --output reports/day01.json
python -m unittest discover -s tests -v
```

新环境推荐 Python 3.11。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m kernels.tensor_layout --output reports/day01.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux（CPU 入门实验）：

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python -m kernels.tensor_layout --output reports/day01.json
.venv/bin/python -m unittest discover -s tests -v
```

CPU 依赖锁定 PyTorch 2.13.0；不要把这个 CPU 安装文件用于将来的 GPU 环境。进入 D02/D15 前，根据目标硬件另建 CUDA / vLLM 环境。安装耗时取决于下载；不要求第一天安装 Ray 或完整 RL 框架。

## 六周路线

| 周 | 问题 | 可验收产物 |
|---|---|---|
| 1 | 数据怎么存、线程怎么访问？ | tensor 断言、CUDA 基础与正确计时 |
| 2 | 并行训练切什么、通信什么？ | 显存记录、梯度等价性与通信语义 |
| 3 | 推理时间耗在哪里？ | KV 预算、请求路径与单卡 benchmark |
| 4 | 奖励怎样变成参数更新？ | 有精确梯度参照的 bandit 同步闭环 |
| 5 | 异步样本旧了多少？ | 有界队列、lag、超时去重与版本日志 |
| 6 | 训练与采样不一致怎么办？ | IS/TIS 对照、完整权重发布与最终报告 |

## 实验目录

```text
kernels/     tensor、CUDA 和数值实验（D01 已实现）
trainer/     显存、梯度累积、并行语义（待实现）
serving/     KV cache、调度、推理测量（待实现）
rl/          loss、advantage、纠偏（待实现）
runtime/     sampler、buffer、权重版本（待实现）
tests/       可执行的行为验证
reports/     原始结果、环境和学习记录
docs/        评审、课程、资料表、第一天讲义
```

## 验收约定

- 阅读完成、代码通过、真实性能验证分开记录。
- CPU 模拟可以验证算法和协议，不能证明 GPU、NCCL、RDMA 的性能。
- 训练 reward、独立任务评估、单位样本收益、单位时间收益分别报告。
- bandit 的无偏 IS 条件不能直接外推为 LLM 整条轨迹的无偏纠偏；TIS 截断会引入偏差。
- 复用已有等价课程作业时附代码、测试与结果，不重复建设。
- 每周第 7 天复盘；第 6 周留 D41–D42 做整体验证与交付。42 日共约 84–126 小时，可按掌握程度顺延。

资料入口与版本见 [resources](docs/resources.md)。本仓库只链接外部教材，不收录整本书或付费内容。
