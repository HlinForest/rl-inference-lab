# 课程评审 · 2026-10-16

**结论：通过，修订后开学。** 四个主材料覆盖 kernel → trainer → inference → RL 更新，补充文章连接通信、编排、权重同步和训练—采样偏差。这是一条范围合理的 LLM RL systems 入门线；本评审认为它不等同于完整经典 RL、生产运维或 AI safety 课程。时间估算是教学判断，不是原作者的保证。

## 保留什么

保留 6 周顺序、每天 2–3 小时、按章节选读、每周复盘、小型可枚举 bandit 和单卡推理双实验线。原计划主动区分 CPU 机制模拟与真实 GPU 性能，也没有要求从零复刻 vLLM/FSDP，范围控制合理。42 日约 84–126 小时；已有 PyTorch 基础可按此推进，入门者应允许延长。

## 修订什么

| 问题 | 已采用的修订 |
|---|---|
| PMPP 原链接指向另一个版次，章节编号有混淆风险 | 明确采用第 4 版的第 2–6、10 章；持有其他版次时按章节名称对照。 |
| RLHF 网页“构建于 2026-10-14”没有核实依据 | 删除该日期断言，锚定 `book/v0.12`；网页版用于便捷阅读，章节变动时回到固定源码。 |
| 只看训练 reward 容易高估效果 | D27 加第 16 章 Evaluation、独立任务、固定评估策略、reference retention；D41 保留样本与时间两类预算。 |
| PPO ratio 与 mismatch ratio 容易混用 | D22 起定义三种 logprob，D37 分开验证两个 ratio、支持域与截断偏差。 |
| 四动作任务不能覆盖真实序列 GRPO | 明确叫 bandit 上的 GRPO-style 实验，写清 advantage 标准差、KL、mask 与 loss 分母。 |
| 原生 Windows 与本机 CPU 不适合所有实验 | 第一日用 CPU；CUDA/vLLM 单独规划 Linux/WSL2 环境，真实性能项待测。 |
| 只有最新案例容易形成“新方法必胜”的印象 | NIXL/Modal 当作案例分析，不要求复现集群，也不预设 TIS 必须使学习更快。 |

PMPP 章节依据 [第 4 版目录](https://www.oreilly.com/library/view/programming-massively-parallel/9780323984638/xhtml/Title_page.xhtml)。RLHF 版本依据 [官方 v0.12 release](https://github.com/natolambert/rlhf-book/releases/tag/book/v0.12)；已经核对该版本的第 6 章包含 TIS。独立评估依据 [第 16 章](https://rlhfbook.com/c/16-evaluation)。

## 最容易学错的边界

**“同一权重”不等于“同一行为分布”。** kernel、精度、batch 组织和采样变换都可能改变概率。固定 seed 解决不了所有 batch 变化问题。D36 必须记录实际 dtype/shape/版本；没有观察到差异就报告未观察到。[Thinking Machines 原文](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

记 `q` 为实际 sampler 分布、`p_old` 为固定的旧 learner 分布、`p_theta` 为当前 learner 分布。策略更新比率是 `p_theta/p_old`，训练—采样比率是 `p_old/q`。概率对象一致、支持域充分且未截断时，两者相乘为 `p_theta/q`。PPO clipping、TIS 截断和序列状态分布会破坏“只是乘法就无偏”的简单结论。[RLHF Ch6 固定版本](https://github.com/natolambert/rlhf-book/blob/book/v0.12/book/chapters/06-policy-gradients.md)

D37 先用全支持域的单步 categorical bandit 验证：`q(a)>0` 对所有 `p(a)>0` 的动作成立时，枚举期望能检验普通 IS。TIS 是有偏的方差控制方法，应测偏差、方差、有效样本量与截断比例。top-k/top-p 使部分动作概率为零时，简单重放 mask 也不能恢复原本无采样支持的完整目标分布；它改变/对齐的是所定义的截断目标，要另说清楚。[Feng Yao 等人的分析](https://fengyao.notion.site/off-policy-rl)、[Modal 案例](https://modal.com/blog/toward-rl-on-mixed-precision-mixed-accelerator-samplers)

**组内标准化不能和所有任务混用。** 同一 context 的 4 个样本只是 advantage 实验；全等 reward 需返回有限的零 advantage。要声明 std 使用总体还是样本公式，以及 KL 是 reward shaping 还是单独 loss 项。[RLHF Ch6](https://rlhfbook.com/c/06-policy-gradients)

## 环境与工作量

实测环境：Windows、Python 3.11.7、PyTorch 2.13.0+cpu，`torch.cuda.is_available()` 为 false。这说明当前解释器没有可用 CUDA，不据此推断整台电脑绝无 GPU。

官方 [vLLM 安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)要求 Linux，并给 Windows 用户指出 WSL 路线；[Ray 安装文档](https://docs.ray.io/en/latest/ray-overview/installation.html)将 Windows 列为 Beta。WSL 的 GPU 能力还受 [NVIDIA WSL 限制](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)影响。因此现阶段先完成 CPU 实验；多 GPU / NCCL / RDMA 留给明确兼容的 Linux 集群。

D18 的长短负载/前缀缓存实验、D31 的异步接线和 D39 的故障发布最可能超时，可顺延到复盘日。若 CPU-only 学习先行，GPU 项必须继续保持待测，不能据此宣布整个课程完成。

## 最终验收

1. 讲清楚 sampler → reward → buffer → learner → weights 的数据流和版本流。
2. 对至少一个错误 loss、旧样本或中途权重更新，拿出能失败的测试。
3. 固定任务、资源、种子与采样策略，分别在相同更新/样本预算、相同时间预算下比较同步与异步配置；报告多 seed 的分散程度，不拿 3 个 seed 做强统计保证。
4. 用 held-out contexts 检查泛化时须采用共享参数和可泛化任务；若是每个 context 一组独立 logits，只能报告已知 contexts 的精确期望，不能宣称 held-out generalization。
5. 如实列出已验证、仅逻辑模拟与仍待测项目。

详细资料核对见 [resources](resources.md)。
