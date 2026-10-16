# D01 · 一个 tensor 不等于一份独立的数据

目标：解释 storage、shape、stride、offset；判断转置、切片和 reshape 是否分配新存储；把这一点连接到 sampler 的权重快照。

今天只用 CPU，预计 2–3 小时。仓库中提供参考实现；读者自己的解释与复现仍需完成。

## 1. 先预测，再运行（15 分钟）

```python
x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
a = x.t()
b = x[1:, 1:]
c = x.reshape(-1)
d = a.reshape(-1)
e = x[:, ::2].reshape(-1)
```

先写下 a–e 的 shape、stride，以及是否与 x 共享 storage。再回答：修改 a[2, 1] 会不会改变 x？b.data_ptr() 与 x 不同能不能证明 b 是一份拷贝？

## 2. 读到哪里就停（40 分钟）

- [Edward Z. Yang · PyTorch internals](https://blog.ezyang.com/2019/05/pytorch-internals/)：只读 Tensor、Storage、Stride 与 view 的概念段落。2019 年的 dispatch 和目录介绍不作为当前源码地图。
- [PyTorch · Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)：核对 basic indexing、advanced indexing、reshape 和 contiguous 的语义。

可把普通稠密 tensor 看成“存储 + 访问规则”。对本实验的二维 tensor，元素对应的存储索引为：

```text
storage_offset + i × stride[0] + j × stride[1]
```

stride 和 offset 的单位是元素；换算字节地址还要乘 element_size。转置只需换访问规则。某些 reshape 能沿已有规则读取，另一些必须复制。非连续本身不足以判断 reshape 必然复制。

## 3. 运行与改动（45–60 分钟）

在仓库根目录运行：

```sh
python -m kernels.tensor_layout --output reports/day01.json
python -m unittest discover -s tests -v
```

对照自己的预测与 JSON。测试包含 8 个行为验证；阅读每个断言，并把 x 改成 4×6 后重新做预测。不要只修改测试让它通过，先用索引公式解释结果。

注意：`data_ptr()` 是第一个逻辑元素的地址，带偏移切片可以共享 storage，却有不同的 `data_ptr()`。本实验比较的是同时存活的非空 CPU tensor 的 `untyped_storage().data_ptr()`；共享一块存储也不一定意味着两段切片访问相同元素。

## 4. 把 tensor 放回 RL 系统（20 分钟）

```mermaid
flowchart LR
    P[发布完整权重快照与版本 v] --> S[Sampler 依据版本 v 采样]
    S --> R[Reward / Verifier]
    R --> B[有界 Buffer: 样本、采样 logprob、版本]
    B --> L[Learner: 计算 loss 与更新]
    L --> P
```

`detach()` 解除梯度关系，但仍共享存储。如果 learner 更新这块存储，sampler 手里的 detached tensor 也会变。`detach().clone()` 在没有并发写入时能做独立拷贝。多参数模型需要额外的快照协议：暂停更新或加锁，复制到 staging，检查完整性，再整体发布版本；单个 clone 不是原子发布协议。

今天只理解这个连接。真正的同步闭环安排在 D28，异步与完整发布安排在 D31–D39。

## 5. 自测与记录（20 分钟）

- [ ] 不看代码，解释 transpose 为什么通常不搬数据。
- [ ] 用 offset 与 stride 手算 `x[1:, 1:][0, 0]` 的存储索引。
- [ ] 给出一个 reshape 复制和一个非连续 reshape 不复制的例子。
- [ ] 解释 detach 与独立权重快照的区别。
- [ ] 在 `reports/` 新建自己的学习记录，记录预测错在哪里。

完成这些项目后再勾选 [学习进度](../PROGRESS.md) 的 D01。参考代码跑通不自动代表你已掌握。

<details>
<summary>参考答案：先预测再展开</summary>

对于题目中的 3×4 输入：a 的 shape=(4,3)、stride=(1,4)，共享存储；b 的 shape=(2,3)、stride=(4,1)、offset=5，共享存储；c 的 stride=(1,)，共享存储；d 按转置后的逻辑顺序展平，需要新存储；e 的 stride=(2,)，仍共享原存储。修改 a[2,1] 会改变 x[1,2]。

</details>
