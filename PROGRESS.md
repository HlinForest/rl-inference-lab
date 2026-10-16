# 学习进度

D01 参考代码与 8 项测试已通过；学习者自测尚未验收。以下复选框只在本人完成对应解释、实验与证据记录后勾选。

GPU 项缺少硬件时保留待测。详细要求见 [42 天计划](docs/curriculum.md)。

## 第 1 周

- [ ] **D01** — PyTorch internals：tensor、storage、stride、view。 产物：建仓库；写 `transpose`、切片、`reshape` 小实验。**至少 3 个断言，能区分共享底层存储与新分配。**另画出 sampler→reward→buffer→learner→weights 的流程。
- [ ] **D02** — PMPP 第 2 章：host/device、kernel launch。 产物：实现或补完 vector add；测试长度不是 block 整数倍的情况。**结果与 CPU 参考一致，边界检查正确。**
- [ ] **D03** — PMPP 第 3 章二维索引；第 4 章 warp、block、SM 的基本关系。 产物：在朴素 matmul 上补齐索引与边界处理。**非方阵、非整块尺寸通过测试；能手算一个线程负责哪个输出。**
- [ ] **D04** — PMPP 第 5 章：memory hierarchy、data locality、tiling。 产物：在 D3 上加入 shared-memory tiling。**两版结果一致，并解释每个 tile 复用了什么数据。**
- [ ] **D05** — PMPP 第 6 章：coalescing、资源限制、性能考虑。 产物：改两种 block/tile 配置，充分预热并正确计时。**保存 shape、dtype、时延；能解释变快或变慢，而不是必须“优化成功”。**
- [ ] **D06** — PMPP 第 10 章：基本 reduction。 产物：实现简单 row-sum，或复用 CSE291 等价作业。**不同长度与尾块通过测试，记录浮点误差。**
- [ ] **D07** — 不读新材料。 产物：补测试和测量，写一页总结：**数据在哪里、怎样访问、瓶颈是什么。**

## 第 2 周

- [ ] **D08** — Ultra-Scale：单卡训练、显存组成、mixed precision 基本概念。 产物：复用一个现有小模型，跑 forward/backward/optimizer。**列出参数、梯度、优化器状态与激活；区分估算值与实测显存。**
- [ ] **D09** — Gradient accumulation、activation recomputation。 产物：比较整 batch 与 microbatch 累积的梯度；再打开现成 checkpoint 功能测量。**控制 dropout、正确缩放 loss 后梯度在容差内一致；记录显存和时间交换。**
- [ ] **D10** — 通信文章：All-Gather、Reduce-Scatter、All-Reduce、All-to-All。 产物：用 4 个小数组模拟 4 个 rank。**能手算每个 rank 最后得到什么，验证归约与拼接的组合关系。**
- [ ] **D11** — Ultra-Scale：Data Parallelism、ZeRO。 产物：写一个 4-rank 内存分配表或估算器。**准确标出 ZeRO-1/2/3 分别切分哪些状态、增加哪些通信。**
- [ ] **D12** — Tensor Parallelism。 产物：用 CPU 张量把一个线性层切到两个逻辑 rank，再拼接或求和恢复。**与原线性层输出一致，能标出通信位置。**
- [ ] **D13** — Pipeline Parallelism、计算与通信重叠。 产物：为 2-stage、4-microbatch 画时间线或写小模拟。**标出 bubble、等待和可重叠区间，说明增加 microbatch 的限制。**
- [ ] **D14** — 不读新材料。 产物：做一页 DP/ZeRO/TP/PP 对照。**每种方法都回答：切什么、通信什么、解决什么、代价是什么。**

## 第 3 周

- [ ] **D15** — Baseten 第 2–3 章中的 prefill/decode、compute/memory；4.3 引擎定位。 产物：固定一个设备能容纳的小模型，建立单请求基线。**记录模型 revision、引擎版本、dtype、输入和输出长度。**
- [ ] **D16** — Baseten 5.3 Caching；vLLM 的 KV manager、分页相关内容。 产物：写 KV cache 预算器和简化 block 分配模拟。**请求结束能回收 block；区分 query heads 与 KV heads。**
- [ ] **D17** — vLLM：Engine Core、Generate、Scheduler、forward。 产物：追踪一条 request 的代码路径；用长度 8/16/64 的假请求模拟 continuous batching。**解释短请求结束后槽位如何复用。**
- [ ] **D18** — vLLM：Chunked prefill、Prefix Caching。 产物：分别设计长短请求混合负载、重复前缀负载。**与基线比较，每次只改变一个开关；记录 TTFT 和吞吐。**
- [ ] **D19** — Baseten 5.1 Quantization、5.2 Speculative Decoding。 产物：做位宽与存储量、draft 与 verification 成本的小算例。**给出 speculation 反而变慢的条件；本轮不实现完整 speculative pipeline。**
- [ ] **D20** — Baseten 4.5 Benchmarking；5.5 P/D 分离，5.4 并行知识复用上周。 产物：固定工作负载，测并发 1/4/8。**输出吞吐、TTFT、TPOT 的统计结果，保留样本数、错误率与原始日志。**
- [ ] **D21** — 不读新材料。 产物：写推理实验报告。**说明瓶颈、改动效果、失败案例，以及结论适用于什么负载。**

## 第 4 周

- [ ] **D22** — 第 3 章 Training Overview；第 4/5 章只理解 SFT、reward model 的位置。 产物：建立小任务和样本结构。**样本包含 sample_id、context/action/reward、实际采样 logprob、采样配置与 sampler version；区分固定的 old learner logprob、当前 learner logprob、行为 sampler logprob。**
- [ ] **D23** — 第 6 章：Policy gradient、REINFORCE。 产物：实现 REINFORCE；比较枚举精确梯度与采样梯度。**loss 符号正确，固定奖励/基线不误入求导，保留训练曲线。**
- [ ] **D24** — PPO objective；value function、GAE 先理解作用。 产物：用固定 logprob 和正负 advantage 测试 PPO clipped objective。**old logprob 固定，current logprob 有梯度；正负 advantage 的行为都通过测试。**
- [ ] **D25** — GRPO；RLOO 只比较 baseline 构造。 产物：同一 context 采 4 个 action，计算组内 advantage，接上已有 learner。**全同 reward 组给出有限零 advantage；声明 std 的定义、epsilon、KL 位置与 loss 分母，明确这是 bandit 上的 GRPO-style 简化。**
- [ ] **D26** — 第 15 章 KL；第 6 章 Loss Aggregation。 产物：用两条长短不同的序列测试 mask、token 平均与 sequence 平均。**明确 loss 分母，mask 位置不贡献对应 loss，区分 KL 正则与纠偏。**
- [ ] **D27** — 第 7 章 RLVR；第 16 章 Evaluation 选读；第 8 章 DPO 只作定位。 产物：在同一小任务上比较 REINFORCE 与 GRPO-style，各跑 3 个 seed。**记录 reward、entropy、梯度范数与更新次数；另用固定评估策略与独立 contexts 检查任务成功率和 reference retention。若为独立 logits 表则只报告已知 contexts 的精确期望，不声称泛化。**
- [ ] **D28** — 不读新材料。 产物：整理同步 `sample → reward → train → publish` 入口。**一个命令能跑通，每批样本与权重版本可追溯。**

## 第 5 周

- [ ] **D29** — Ray：Task、Actor、Object；Placement Groups 先定位。 产物：本机启动两个 Sampler actor，维护各自状态。**能解释 task 与 actor、对象与引用的区别；正常清理资源。**
- [ ] **D30** — HF Async 第 1 节；第 3 节 Axis 1–2。 产物：接入有界 rollout buffer 和最大在途任务数。**生产者过快时不会无限占内存，队列长度可观察。**
- [ ] **D31** — Colocated/disaggregated 与并发编排。 产物：把 D28 learner 接到两个 sampler。**日志显示采样与训练交错；每次采样使用完整的权重快照，保存实际采样 logprob。**
- [ ] **D32** — Axis 4：Staleness；RLHF 第 6 章 async 部分。 产物：记录 `learner_version - sampler_version`，改变同步间隔并设置最大 lag。**同时记录过期样本丢弃率、等待时间、有效样本率。**
- [ ] **D33** — Axis 5：Partial Rollout Handling。 产物：给 worker 注入延迟或异常，增加超时和 `sample_id` 去重。**collector 不会永久等待，重复结果不重复入队；暂不实现完整 partial rollout。**
- [ ] **D34** — Axis 3：Weight Synchronisation。 产物：实现最简单的完整快照更新：load 完成后才发布 version；模拟缓存按版本失效。**不会读到半更新权重，也不会错误复用旧版本缓存。**
- [ ] **D35** — 不读新材料。 产物：比较同步、异步无 lag 上限、异步有界 lag。**同时报告 reward/update、reward/wall-clock、有效样本率和丢弃率。**

## 第 6 周

- [ ] **D36** — Thinking Machines：浮点非结合性、batch invariance、规约策略。 产物：对同一输入、同一权重比较单独执行与不同 batch 下的结果。**记录误差、dtype、shape、版本；没有观察到差异也如实记录。**
- [ ] **D37** — Feng Yao mismatch 文章；RLHF 第 6 章 TIS 对照。 产物：在可枚举 bandit 上构造不同采样分布与目标策略，比较精确梯度、未纠偏、IS、TIS。**先使用全支持域 categorical 分布，检查实际采样概率、两个 ratio、detach；测梯度偏差、方差、ESS 与截断比例，不预设 TIS 必胜。**
- [ ] **D38** — NIXL：RDMA primer、传输下界、Weight format hell。 产物：手算传输下界；对两块小权重实现 pack/unpack，保存 shape、dtype、offset。**重建权重和前向结果通过对照，不把本机搬运叫作 RDMA benchmark。**
- [ ] **D39** — NIXL 布局与转换；回看 D34。 产物：加入 staging→完整性检查→切换 version，故意制造中途失败。**失败时不发布半更新版本；缓存与权重版本一致。**
- [ ] **D40** — Modal **2026-10-13** 综合文章：设定、稳定化方法、实验与局限。 产物：写一页案例分析，将每种差异对应到纠偏方法。**解释为什么不是“乘一个 ratio 就解决全部问题”；不复现大集群。**
- [ ] **D41** — 不读新材料。 产物：固定小任务、资源与预算，对比同步 baseline、异步未纠偏、异步＋TIS，每组 3 个 seed。**在相同样本/更新预算与相同时间预算下分别比较，报告独立评估、时间、lag、有效样本率、丢弃率与 3 个 seed 的分散程度。**
- [ ] **D42** — 不读新材料。 产物：补 README、依赖、配置、测试、原始日志，从干净环境复跑。**5 分钟讲清闭环、一个关键 trade-off，以及哪些结果仍未验证。**
