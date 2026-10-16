# 资料与版本登记

核对日期：**2026-10-16（Asia/Shanghai）**。发布日期来自原站或官方仓库；“访问日期”不等于文章发布日期。书籍按下表版次/版本引用，动态文档的运行环境版本另外记录。

阅读清单策展人：[David Bellamy](https://x.com/DavidRBellamy/status/2110839750060065216)。下面的选读顺序和实验范围由本仓库补充。

## 四个主材料

| ID | 原始资料与作者 | 版本 / 日期 | 本轮范围与硬件 |
|---|---|---|---|
| R01 | [Ultra-Scale Playbook · HF/Nanotron](https://nanotron-ultrascale-playbook.static.hf.space/) / [源码 Space](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | 发布 2025-02-19；本次核对 revision 前缀 `c93890d` | 按 section 读显存、累积、重计算、DP/ZeRO/TP/PP、重叠。CPU 验证语义；GPU 才能验证性能。 |
| R02 | [Inference Engineering · Philip Kiely/Baseten](https://www.baseten.co/inference-engineering/book/) | 官网说明完成于 2026-01；动态网页，无固定 release tag | Ch2–3 补基础，4.3 引擎、4.5 测量、Ch5 优化；单卡性能实验另配兼容环境。 |
| R03 | [RLHF Book · Nathan Lambert](https://rlhfbook.com/) / [官方源码 release](https://github.com/natolambert/rlhf-book/releases/tag/book/v0.12) | 锚定 `book/v0.12`，发布 2026-09-11，commit 前缀 `d80e85d` | Ch3/6/7/15 为主，补 Ch16 评估；Ch4/5/8 定位。CPU bandit 可完成算法部分。 |
| R04 | [Programming Massively Parallel Processors · 第4版目录](https://www.oreilly.com/library/view/programming-massively-parallel/9780323984638/xhtml/Title_page.xhtml) | 4th edition，2022；Hwu / Kirk / El Hajj | Ch2–6、Ch10 Reduction；CUDA 实现需要兼容 GPU。持其他版次按章节名称对照。 |

RLHF 固定版本章节入口：[Ch3](https://github.com/natolambert/rlhf-book/blob/book/v0.12/book/chapters/03-training-overview.md)、[Ch6](https://github.com/natolambert/rlhf-book/blob/book/v0.12/book/chapters/06-policy-gradients.md)、[Ch7](https://github.com/natolambert/rlhf-book/blob/book/v0.12/book/chapters/07-reasoning.md)、[Ch15](https://github.com/natolambert/rlhf-book/blob/book/v0.12/book/chapters/15-regularization.md)、[Ch16](https://github.com/natolambert/rlhf-book/blob/book/v0.12/book/chapters/16-evaluation.md)。

Baseten 快捷入口：[4.3 Inference Engines](https://www.baseten.co/inference-engineering/book/04-software/inference-engines/)、[4.5 Benchmarking](https://www.baseten.co/inference-engineering/book/04-software/performance-benchmarking-and-load-testing/)。

## 九篇补充材料

| ID | 原始资料 | 日期 / 版本与阅读边界 |
|---|---|---|
| R05 | [PyTorch internals · Edward Z. Yang](https://blog.ezyang.com/2019/05/pytorch-internals/) | 2019-05-16；学 tensor/storage/stride 的概念，不把旧源码位置当现行 API。 |
| R06 | [Inside vLLM: Anatomy of a High-Throughput LLM Inference System · Aleksa Gordić](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) | 2025-09-05；基于 V1 commit `42172ad`（2025-08-09），追源码时另固定实际引擎 commit。 |
| R07 | [Ray Architecture 官方入口](https://docs.ray.io/en/latest/ray-contribute/whitepaper.html) / [Ray 2.0 Architecture 原文](https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview) | Ray Team，2022-10；架构基线。Task/Actor/Object API 以安装版本文档为准。 |
| R08 | [Inside TPU and GPU Clusters: The Anatomy of Collective Communication · Aleksa Gordić](https://www.aleksagordic.com/blog/collective-operations) | 2026-07-14；优先 All-Gather / Reduce-Scatter / All-Reduce / All-to-All 和 GPU 拓扑。 |
| R09 | [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries · HF](https://huggingface.co/blog/async-rl-training-landscape) | 2026-03-10；读 7 个设计轴，前 5 个做实践，LoRA/后端作为扩展，不背框架功能表。 |
| R10 | [Defeating Nondeterminism in LLM Inference · Thinking Machines Lab](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) | 2025-09-10；固定运行的确定性与 batch invariance 不同，CPU 实验不能代替 GPU 证据。 |
| R11 | [Off-policy RL / trainer–inference mismatch · Feng Yao 等](https://fengyao.notion.site/off-policy-rl) | 原文标记首发 2025-08-05、更新 2025-10-13、Work in Progress；阅读时与 R03 TIS 交叉核对。Notion 动态页可能展开失败。 |
| R12 | [NIXL + ModelExpress weight transfer · Prime Intellect](https://www.primeintellect.ai/blog/nixl-modelexpress-weight-transfer) | 2026-09-03；学格式/分片/转换/传输与发布协议；本机 pack/unpack 不是 RDMA benchmark。 |
| R13 | [Toward RL on mixed-precision, mixed-accelerator samplers · Nan Jiang/Modal](https://modal.com/blog/toward-rl-on-mixed-precision-mixed-accelerator-samplers) | 2026-10-13；MIS、score centering、top-p mask replay 的案例分析；不作为异构采样普遍有效的保证。 |

## 实验所需的现行文档

- [PyTorch Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)：D01 的行为参考。测试记录的 PyTorch 版本为 2.13.0+cpu。
- [PyTorch detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)：返回值仍共享存储。
- [Ray Core](https://docs.ray.io/en/latest/ray-core/walkthrough.html)与[安装限制](https://docs.ray.io/en/latest/ray-overview/installation.html)：首次引入时单独锁依赖；本仓库现阶段尚未安装或测试 Ray。
- [vLLM GPU 安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)与[NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)：先确认 Linux/WSL、驱动、GPU 与引擎兼容矩阵，再安装。

## 如何保持可复现

每个实测报告记录模型名称与 revision、代码 commit、Python/torch/engine 版本、驱动与硬件、dtype、shape、seed、tokenizer/chat template、temperature/top-p/top-k、输入/输出限制、并发或到达率、暖机、样本数、失败率。未使用的项目写“不适用”。

本文保存引用与核对记录，不镜像教材正文。网页后续变化时修订本登记表；原有实验仍保留原版本信息。
