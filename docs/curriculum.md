# 42 天 RL + Inference Systems 学习计划

修订于 2026-10-16。阅读清单 credit to [David Bellamy](https://x.com/DavidRBellamy/status/2110839750060065216)；详细来源见 [CREDITS](../CREDITS.md)。本日程是独立改编。

**6 周、42 个学习日：前 5 周每周 6 天新内容、1 天复盘；第 6 周 5 天新内容、2 天整体验证。**按每天 **2–3 小时**设计，目标不是“读完四本书”，而是能够解释、实现并测量一个小型的 **RL 训练—推理闭环**。

如果你同时学习 **UCSD CSE291（Hao Zhang）**，已完成且有测试证据的 CUDA、并行训练、serving 作业可以抵扣对应项目。不要仅因课程名称相同就判为完成。

配套入口：[课程评审](review.md) · [可勾选进度](../PROGRESS.md) · [资料与版本](resources.md) · [D01 讲义](day01.md)。每天约 2–3 小时，共约 84–126 小时；未完成可顺延。

## 一、先锁定范围：不是四本书全部读完

| 主材料 | 本轮重点 | 暂时不展开 |
|---|---|---|
| **PMPP 第 4 版** | 第 2–6 章的线程组织、内存层级、tiling、性能基础；补第 10 章 reduction。按章节名称定位即可。 | 各种应用案例、高级矩阵乘优化，不要求手写超过库实现的 kernel。[第 4 版目录](https://www.oreilly.com/library/view/programming-massively-parallel/9780323984638/xhtml/Title_page.xhtml) |
| **HF Ultra-Scale Playbook** | 单卡显存、重计算、梯度累积、DP、ZeRO、TP、PP、计算与通信重叠。 | 不逐个复现大型集群实验，不从头再学一套 trainer 框架。[Nanotron Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/) |
| **Baseten Inference Engineering** | 第 2–3 章按需补基础；重点读 4.3 inference engines、4.5 benchmark，以及第 5 章推理优化。 | 多模态与完整生产平台部署先不作为必修。[Baseten](https://www.baseten.co/inference-engineering/book/) |
| **Lambert RLHF Book** | 第 3 章训练流程、第 6 章 RL、第 7 章 reasoning/RLVR、第 15 章 regularization；SFT、reward model、DPO 先理解位置。 | 不先训练大型 reward model，也不先实现完整 PPO critic 系统。[RLHF Book](https://rlhfbook.com/c/03-training-overview) |

**版本约定：PMPP 采用第 4 版；RLHF Book 锚定 [book/v0.12](https://github.com/natolambert/rlhf-book/releases/tag/book/v0.12)。** 正文保留网页版阅读入口，章节变更时以 [资料登记](resources.md) 的固定源码为准。HF Playbook 按 section 名称选读。

### 每天的固定节奏

**40–50 分钟阅读 → 60–90 分钟实现或测量 → 15–20 分钟记录。**

所有内容放进同一个仓库：

```text
rl-inference-lab/
├── kernels/       # GPU基础与数值实验
├── trainer/       # 显存、梯度累积、并行语义
├── serving/       # 推理基线与benchmark
├── rl/            # loss、advantage、纠偏
├── runtime/       # sampler、buffer、版本与同步
├── tests/
└── reports/
```

这里设计两个验证层：**单卡推理实验负责真实 GPU 性能；小型异步 RL 闭环负责算法、调度和一致性。**不要求你在六周内从零重写 vLLM、FSDP 或完整的分布式 RL 框架。

---

## 二、第 1 周：PyTorch 与 GPU 基础

**本周目标：看到一个算子，能说明张量如何存储、线程如何访问、哪里发生数据搬运。**

资料：PMPP 基础章节，以及你列出的 **PyTorch internals**。后者重点看 tensor、storage、stride、view 等结构；不要把 2019 年的源码位置当作当前版本导航。[blog.ezyang.com](https://blog.ezyang.com/2019/05/pytorch-internals/)

| 天 | 当天阅读范围 | 动手任务与完成标准 |
|---|---|---|
| **D1** | PyTorch internals：tensor、storage、stride、view。 | 建仓库；写 `transpose`、切片、`reshape` 小实验。**至少 3 个断言，能区分共享底层存储与新分配。**另画出 sampler→reward→buffer→learner→weights 的流程。 |
| **D2** | PMPP 第 2 章：host/device、kernel launch。 | 实现或补完 vector add；测试长度不是 block 整数倍的情况。**结果与 CPU 参考一致，边界检查正确。** |
| **D3** | PMPP 第 3 章二维索引；第 4 章 warp、block、SM 的基本关系。 | 在朴素 matmul 上补齐索引与边界处理。**非方阵、非整块尺寸通过测试；能手算一个线程负责哪个输出。** |
| **D4** | PMPP 第 5 章：memory hierarchy、data locality、tiling。 | 在 D3 上加入 shared-memory tiling。**两版结果一致，并解释每个 tile 复用了什么数据。** |
| **D5** | PMPP 第 6 章：coalescing、资源限制、性能考虑。 | 改两种 block/tile 配置，充分预热并正确计时。**保存 shape、dtype、时延；能解释变快或变慢，而不是必须“优化成功”。** |
| **D6** | PMPP 第 10 章：基本 reduction。 | 实现简单 row-sum，或复用 CSE291 等价作业。**不同长度与尾块通过测试，记录浮点误差。** |
| **D7** | 不读新材料。 | 补测试和测量，写一页总结：**数据在哪里、怎样访问、瓶颈是什么。** |

**这一周不追求 FlashAttention，也不追求击败 cuBLAS。**先让最基本的实验可信、可复现。

---

## 三、第 2 周：Trainer、显存与通信

**本周目标：不是记住并行缩写，而是理解“切什么、通信什么、节省什么、付出什么”。**

资料：Ultra-Scale Playbook，以及 **Inside TPU and GPU Clusters: The Anatomy of Collective Communication**。通信文章优先读 All-Gather、Reduce-Scatter/All-Reduce、All-to-All 和 GPU 拓扑，TPU 细节后置。[Nanotron Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/)

| 天 | 当天阅读范围 | 动手任务与完成标准 |
|---|---|---|
| **D8** | Ultra-Scale：单卡训练、显存组成、mixed precision 基本概念。 | 复用一个现有小模型，跑 forward/backward/optimizer。**列出参数、梯度、优化器状态与激活；区分估算值与实测显存。** |
| **D9** | Gradient accumulation、activation recomputation。 | 比较整 batch 与 microbatch 累积的梯度；再打开现成 checkpoint 功能测量。**控制 dropout、正确缩放 loss 后梯度在容差内一致；记录显存和时间交换。** |
| **D10** | 通信文章：All-Gather、Reduce-Scatter、All-Reduce、All-to-All。 | 用 4 个小数组模拟 4 个 rank。**能手算每个 rank 最后得到什么，验证归约与拼接的组合关系。** |
| **D11** | Ultra-Scale：Data Parallelism、ZeRO。 | 写一个 4-rank 内存分配表或估算器。**准确标出 ZeRO-1/2/3 分别切分哪些状态、增加哪些通信。** |
| **D12** | Tensor Parallelism。 | 用 CPU 张量把一个线性层切到两个逻辑 rank，再拼接或求和恢复。**与原线性层输出一致，能标出通信位置。** |
| **D13** | Pipeline Parallelism、计算与通信重叠。 | 为 2-stage、4-microbatch 画时间线或写小模拟。**标出 bubble、等待和可重叠区间，说明增加 microbatch 的限制。** |
| **D14** | 不读新材料。 | 做一页 DP/ZeRO/TP/PP 对照。**每种方法都回答：切什么、通信什么、解决什么、代价是什么。** |

没有多卡不妨碍这一周理解机制，但 **CPU 上的逻辑 rank 模拟不能当作真实 NCCL 性能实验**。

---

## 四、第 3 周：Inference engine 与性能测量

**本周目标：从“能启动 vLLM”推进到“能解释请求路径，并用数据判断优化是否有效”。**

资料：Baseten 的模型、硬件、benchmark 和优化章节，加上 **Inside vLLM: Anatomy of a High-Throughput LLM Inference System**。这篇文章覆盖 engine core、调度、缓存、chunked prefill、speculation 和 serving；它发布于 2025 年，阅读当前源码时按职责寻找对应模块，不硬找旧路径。[vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)

| 天 | 当天阅读范围 | 动手任务与完成标准 |
|---|---|---|
| **D15** | Baseten 第 2–3 章中的 prefill/decode、compute/memory；4.3 引擎定位。 | 固定一个设备能容纳的小模型，建立单请求基线。**记录模型 revision、引擎版本、dtype、输入和输出长度。** |
| **D16** | Baseten 5.3 Caching；vLLM 的 KV manager、分页相关内容。 | 写 KV cache 预算器和简化 block 分配模拟。**请求结束能回收 block；区分 query heads 与 KV heads。** |
| **D17** | vLLM：Engine Core、Generate、Scheduler、forward。 | 追踪一条 request 的代码路径；用长度 8/16/64 的假请求模拟 continuous batching。**解释短请求结束后槽位如何复用。** |
| **D18** | vLLM：Chunked prefill、Prefix Caching。 | 分别设计长短请求混合负载、重复前缀负载。**与基线比较，每次只改变一个开关；记录 TTFT 和吞吐。** |
| **D19** | Baseten 5.1 Quantization、5.2 Speculative Decoding。 | 做位宽与存储量、draft 与 verification 成本的小算例。**给出 speculation 反而变慢的条件；本轮不实现完整 speculative pipeline。** |
| **D20** | Baseten 4.5 Benchmarking；5.5 P/D 分离，5.4 并行知识复用上周。 | 固定工作负载，测并发 1/4/8。**输出吞吐、TTFT、TPOT 的统计结果，保留样本数、错误率与原始日志。** |
| **D21** | 不读新材料。 | 写推理实验报告。**说明瓶颈、改动效果、失败案例，以及结论适用于什么负载。** |

这里的 **TTFT 是首 token 延迟，TPOT 是后续 token 的平均生成耗时**。实验不仅要固定并发，还应控制输入输出长度、请求内容和采样参数；否则可能是在比较不同工作负载，而不是比较优化。[Baseten](https://www.baseten.co/inference-engineering/book/04-software/performance-benchmarking-and-load-testing/)

---

## 五、第 4 周：RL 算法与同步闭环

**本周目标：真正写出并测试 loss，而不是停留在“看过 PPO/GRPO 公式”。**

资料：RLHF Book 第 3、6、7、15 章。第 6 章已经包括 REINFORCE、PPO、GRPO、loss aggregation、异步系统和 TIS，不需要再平行开一门完整 RL 课程。[RLHF Book](https://rlhfbook.com/c/06-policy-gradients)

建议先设计一个**可枚举的 contextual bandit**：给定一个 context，从 4 个 action 中选一个获得奖励。因为动作很少，你可以直接枚举计算精确梯度，后面判断纠偏是否正确就有了参照。

| 天 | 当天阅读范围 | 动手任务与完成标准 |
|---|---|---|
| **D22** | 第 3 章 Training Overview；第 4/5 章只理解 SFT、reward model 的位置。 | 建立小任务和样本结构。**样本包含 sample_id、context/action/reward、实际采样 logprob、采样配置与 sampler version；区分固定的 old learner logprob、当前 learner logprob、行为 sampler logprob。** |
| **D23** | 第 6 章：Policy gradient、REINFORCE。 | 实现 REINFORCE；比较枚举精确梯度与采样梯度。**loss 符号正确，固定奖励/基线不误入求导，保留训练曲线。** |
| **D24** | PPO objective；value function、GAE 先理解作用。 | 用固定 logprob 和正负 advantage 测试 PPO clipped objective。**old logprob 固定，current logprob 有梯度；正负 advantage 的行为都通过测试。** |
| **D25** | GRPO；RLOO 只比较 baseline 构造。 | 同一 context 采 4 个 action，计算组内 advantage，接上已有 learner。**全同 reward 组给出有限零 advantage；声明 std 的定义、epsilon、KL 位置与 loss 分母，明确这是 bandit 上的 GRPO-style 简化。** |
| **D26** | 第 15 章 KL；第 6 章 Loss Aggregation。 | 用两条长短不同的序列测试 mask、token 平均与 sequence 平均。**明确 loss 分母，mask 位置不贡献对应 loss，区分 KL 正则与纠偏。** |
| **D27** | 第 7 章 RLVR；第 16 章 Evaluation 选读；第 8 章 DPO 只作定位。 | 在同一小任务上比较 REINFORCE 与 GRPO-style，各跑 3 个 seed。**记录 reward、entropy、梯度范数与更新次数；另用固定评估策略与独立 contexts 检查任务成功率和 reference retention。若为独立 logits 表则只报告已知 contexts 的精确期望，不声称泛化。** |
| **D28** | 不读新材料。 | 整理同步 `sample → reward → train → publish` 入口。**一个命令能跑通，每批样本与权重版本可追溯。** |

**这一周暂不要求完整 PPO 训练。**先把 policy-gradient、advantage、mask、old logprob 和 reference 的关系做对，比同时实现多个完整算法更符合这轮目标。

---

## 六、第 5 周：Ray、异步调度与样本过期

**本周目标：让 sampler 和 learner 并行运行，同时知道数据究竟“旧了多少”。**

资料：Ray 2.0 架构白皮书及当前 Ray Core 概念文档；HF 的 **Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries**。后者发表于 **2026 年 3 月 10 日**，重点学习 buffer、权重同步、staleness、partial rollout 等设计维度，不背诵框架功能表。[Ray](https://docs.ray.io/en/latest/ray-contribute/whitepaper.html)

| 天 | 当天阅读范围 | 动手任务与完成标准 |
|---|---|---|
| **D29** | Ray：Task、Actor、Object；Placement Groups 先定位。 | 本机启动两个 Sampler actor，维护各自状态。**能解释 task 与 actor、对象与引用的区别；正常清理资源。** |
| **D30** | HF Async 第 1 节；第 3 节 Axis 1–2。 | 接入有界 rollout buffer 和最大在途任务数。**生产者过快时不会无限占内存，队列长度可观察。** |
| **D31** | Colocated/disaggregated 与并发编排。 | 把 D28 learner 接到两个 sampler。**日志显示采样与训练交错；每次采样使用完整的权重快照，保存实际采样 logprob。** |
| **D32** | Axis 4：Staleness；RLHF 第 6 章 async 部分。 | 记录 `learner_version - sampler_version`，改变同步间隔并设置最大 lag。**同时记录过期样本丢弃率、等待时间、有效样本率。** |
| **D33** | Axis 5：Partial Rollout Handling。 | 给 worker 注入延迟或异常，增加超时和 `sample_id` 去重。**collector 不会永久等待，重复结果不重复入队；暂不实现完整 partial rollout。** |
| **D34** | Axis 3：Weight Synchronisation。 | 实现最简单的完整快照更新：load 完成后才发布 version；模拟缓存按版本失效。**不会读到半更新权重，也不会错误复用旧版本缓存。** |
| **D35** | 不读新材料。 | 比较同步、异步无 lag 上限、异步有界 lag。**同时报告 reward/update、reward/wall-clock、有效样本率和丢弃率。** |

**这一周不要只优化“每秒生成多少样本”。**你的验收要同时回答：生成的样本有多少最终被使用？使用时落后多少版本？更快采样是否真的让学习更快？

---

## 七、第 6 周：数值一致性、纠偏与权重同步

**本周目标：把清单最后几篇文章读成同一个问题，而不是几个孤立技巧。**

你列的最后四组资料分别对应：

| 资料 | 这一轮要抓住的问题 |
|---|---|
| **Thinking Machines：Defeating Nondeterminism** | 同样的权重和输入，为什么不同 batch 组织可能产生数值差异？固定 seed 为什么不能代替 batch invariance？[Thinking Machines Lab](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) |
| **Feng Yao：Off-policy RL / mismatch** | sampler 与 trainer 即使持有相同权重，也可能给出不同概率；如何在更新时处理这种偏差？[Notion](https://fengyao.notion.site/off-policy-rl) |
| **Prime Intellect：NIXL + ModelExpress** | 权重同步不仅是传字节，还涉及分片、布局、转换和接收端重建。[Prime Intellect](https://www.primeintellect.ai/blog/nixl-modelexpress-weight-transfer) |
| **Modal：Toward RL on mixed-precision, mixed-accelerator samplers** | 如何将异步、异构采样和纠偏组合起来？文章使用的 MIS、score centering、top-p mask replay 各在解决什么？[Modal](https://modal.com/blog/toward-rl-on-mixed-precision-mixed-accelerator-samplers) |

| 天 | 当天阅读范围 | 动手任务与完成标准 |
|---|---|---|
| **D36** | Thinking Machines：浮点非结合性、batch invariance、规约策略。 | 对同一输入、同一权重比较单独执行与不同 batch 下的结果。**记录误差、dtype、shape、版本；没有观察到差异也如实记录。** |
| **D37** | Feng Yao mismatch 文章；RLHF 第 6 章 TIS 对照。 | 在可枚举 bandit 上构造不同采样分布与目标策略，比较精确梯度、未纠偏、IS、TIS。**先使用全支持域 categorical 分布，检查实际采样概率、两个 ratio、detach；测梯度偏差、方差、ESS 与截断比例，不预设 TIS 必胜。** |
| **D38** | NIXL：RDMA primer、传输下界、Weight format hell。 | 手算传输下界；对两块小权重实现 pack/unpack，保存 shape、dtype、offset。**重建权重和前向结果通过对照，不把本机搬运叫作 RDMA benchmark。** |
| **D39** | NIXL 布局与转换；回看 D34。 | 加入 staging→完整性检查→切换 version，故意制造中途失败。**失败时不发布半更新版本；缓存与权重版本一致。** |
| **D40** | Modal **2026-10-13** 综合文章：设定、稳定化方法、实验与局限。 | 写一页案例分析，将每种差异对应到纠偏方法。**解释为什么不是“乘一个 ratio 就解决全部问题”；不复现大集群。** |
| **D41** | 不读新材料。 | 固定小任务、资源与预算，对比同步 baseline、异步未纠偏、异步＋TIS，每组 3 个 seed。**在相同样本/更新预算与相同时间预算下分别比较，报告独立评估、时间、lag、有效样本率、丢弃率与 3 个 seed 的分散程度。** |
| **D42** | 不读新材料。 | 补 README、依赖、配置、测试、原始日志，从干净环境复跑。**5 分钟讲清闭环、一个关键 trade-off，以及哪些结果仍未验证。** |

D37 的重要边界是：**先在单步 bandit 中验证重要性采样，再讨论序列任务。**实际 LLM 中的 token 级纠偏还涉及前缀分布、截断、mask 和概率记录，不能因为小实验成立就称它为“整条轨迹的无偏修复”。RLHF Book 的 TIS 实现也明确区分实际采样 logprob、旧 trainer logprob、当前更新 logprob，并把纠偏系数作为固定权重使用。[RLHF Book](https://rlhfbook.com/c/06-policy-gradients)

---

## 八、推进时守住三个边界

### 1. 用已有作业抵扣，不重复建设

如果已有 CSE291 等价作业，直接把允许分享的代码、测试和结果接进这个仓库。**同一份 matmul、同一份并行实验、同一份 serving benchmark，不要为了换资料再重做。**

一天没完成就顺延；复盘日用来补测，而不是继续塞新论文。

### 2. 把“逻辑通过”和“性能验证”分开

| 资源 | 本计划中可以完成的内容 | 不能据此证明 |
|---|---|---|
| **CPU** | 梯度、loss、collective 语义、Ray 本机编排、buffer、版本协议、故障测试 | GPU 性能、真实集群通信效率、LLM 收敛效果 |
| **单张 GPU** | CUDA 基础、单卡 trainer profiling、小模型 vLLM benchmark | 多卡通信效率、跨节点同步性能 |
| **多 GPU / RDMA 集群** | 后续真实并行与权重传输验证 | 本轮不要求提前获得这些资源 |

没有 GPU，可以继续推进 CPU 部分，但 GPU 测量项保留为“待测”，不要直接打勾。

### 3. 每天留下证据，不只留下阅读时长

每天用这个模板即可：

```text
今天的问题：
一句话解释：
改了哪段代码 / 做了哪个分析：
测试与原始日志：
硬件、版本、shape、dtype、配置：
结果支持什么结论：
还不能证明什么：
明天第一步：
```

资料可访问性与版本以 [资料登记](resources.md) 为准。Google Docs / Notion 是动态网页，展开失败时分别用 Ray 官方概念文档和固定版本 RLHF Book 交叉核对，不把未读部分标为已读。

**今天就做 D1：建仓库，读 tensor/storage/stride，完成三个断言，再画出训练—推理闭环。先不要安装整套分布式 RL 框架，也不要同时从第一页开始读四本书。**

## 九、修订后的实验约定

### 概率对象从 D22 起就要明确

记 q 为实际 sampler 的采样分布，p_old 为固定的旧 learner 分布，p_theta 为正在求导的 learner 分布。

- 策略更新比率：p_theta / p_old。
- sampler mismatch 比率：p_old / q，作为固定纠偏系数；TIS 对它截断时会引入偏差。
- 只有概率定义一致、支持域充分、未经 clipping/truncation 时，两者才直接相乘为 p_theta / q。
- 普通 IS 要求目标非零处行为概率也非零。top-p/top-k mask replay 能对齐所定义的截断目标，但不能恢复未采到的动作或自动消除完整目标的支持域缺失。
- token 条件概率修正不自动修复前缀/状态分布；不要把 bandit 结论直接移植到整条 LLM 轨迹。

参照 [RLHF Ch6 固定版本](https://github.com/natolambert/rlhf-book/blob/book/v0.12/book/chapters/06-policy-gradients.md)。

### GPU 和评估不能用模拟抵扣

当前 starter 在 Windows + PyTorch CPU 上验证。D02–D06 的 CUDA 和 D15–D21 的真实性能部分需要兼容 GPU；vLLM 使用 Linux/WSL 环境。CPU 可以提前做 D08–D14 的机制和 D22 以后的算法/协议实验，GPU 项仍记待测。

D20 增加 TTFT/TPOT 的 p50/p95、错误率、输出 token 数及暖机策略；固定并发的闭环测量与固定到达率的开环测量分开报告。只有一个输出 token 时 TPOT 分母为零，应标记不适用。三次种子是起点，不是统计显著性保证。

D27、D41 的 reward 上升必须同时接受独立 verifier/held-out 任务检验。若任务设计没有泛化空间，如每个 context 使用独立参数，则报告精确期望和重复采样误差，不包装成泛化评测。

### 扩展阅读上限

HF Async 的 Axis 6（LoRA）、Axis 7（后端/并行）和 MoE routing / sampling-mask mismatch 本轮只读概念，暂不加新实现。NIXL 与 Modal 保持案例分析，防止第 6 周膨胀成集群复现项目。
