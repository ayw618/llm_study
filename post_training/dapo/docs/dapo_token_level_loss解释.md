你的观察非常敏锐，直击了深度学习损失函数设计的核心。你说得对，**无论哪种损失，最终 `loss.backward()` 时得到的都是一个标量**。

但是，**“Token-Level”和“Sample-Level”的区别不在于最终输出是不是标量，而在于这个标量在计算**梯度**时，分配给每个 Token 的权重是否相同**。

以下是你代码中的 Token-Level 损失与 GRPO 原始 Sample-Level 损失的**数学本质区别**：

### 1. 数学公式的对比（权重分配）

假设一个 Batch 里有 2 个样本，样本 A 长度为 10，样本 B 长度为 2。

- **GRPO (Sample-Level)**（也是你之前 `get_log_probs` 取平均时的做法）：
  损失计算为：先对每个样本内的 Token 取平均，再对样本间取平均。
  $$
  \text{Loss} = \frac{1}{2} \left( \frac{\sum_{t=1}^{10} l_{A,t}}{10} + \frac{\sum_{t=1}^{2} l_{B,t}}{2} \right)
  $$
  **结果**：样本 A 的**每个 Token** 对梯度的贡献是 `1/20`，样本 B 的**每个 Token** 对梯度的贡献是 `1/4`（B 的 Token 重要性是 A 的 5 倍）。

- **DAPO (Token-Level)**（你现在的代码）：
  损失计算为：所有样本的所有 Token 损失相加，再除以总 Token 数。
  $$
  \text{Loss} = \frac{\sum_{t=1}^{10} l_{A,t} + \sum_{t=1}^{2} l_{B,t}}{10 + 2}
  $$
  **结果**：样本 A 的**每个 Token** 对梯度的贡献是 `1/12`，样本 B 的**每个 Token** 对梯度的贡献也是 `1/12`（**所有 Token 完全平等**）。

---

### 2. 为什么你的代码确实算 Token-Level？

你在代码中写的是：
```python
loss = - (surrogate_loss * token_mask).sum() / (token_mask.sum() + 1e-8)
```
这里有两个关键操作，决定了它是 Token-Level：

1. **`.sum()`**：没有指定 `dim`，意味着它对所有样本、所有位置（`[B*G, L-1]`）的损失进行**全局求和**。这保留了长序列应有的“大数值权重”。
2. **`/ token_mask.sum()`**：除以全局所有样本的**总有效 Token 数**，而非除以样本数 `B*G`。

**对比**：如果你想把它改成 Sample-Level，代码应该是：
```python
# 这是错误的 Sample-Level 写法（会导致长样本被稀释）
loss = - (surrogate_loss * token_mask).sum(dim=-1).mean()  # 先对每个样本求和，再对样本求平均
```

---

### 3. 反向传播（梯度）的差异

虽然 `loss` 是一个数，但 `loss.backward()` 计算梯度时，会使用链式法则：

- 在你的代码中，梯度传到 `surrogate_loss` 时，乘的系数是 **`1 / 总Token数`**（全局常数）。
- 这意味着：**长序列由于包含更多的 Token，它们在 `surrogate_loss` 的总和中贡献了更多数值，因此在反向传播时，长序列的总梯度绝对值会更大**。
- 而在 Sample-Level 中，长序列因为先除以了自身长度，再平均，所以即使它很长，它对总损失的贡献也不会超过短序列。

论文（Section 3.3）指出，Sample-Level 会导致长样本的 Token 贡献被“稀释”，使得模型不重视长样本中的推理模式，甚至无法有效惩罚长样本中的乱码（因为乱码的 Token 被平均掉了）。而你的代码通过**全局求和并除以总 Token 数**，完美解决了这个问题。

---

### 4. 总结

你代码里的 `loss` 虽然是一个数，但它是 **“所有 Token 损失的总和 / Token 总数”**，这正是 DAPO 论文定义的 Token-Level Policy Gradient Loss。  
你之前 `get_log_probs` 返回平均值（返回 `[B]`）的做法才是 Sample-Level；现在返回 `[B, L-1]`，且 Loss 对全局求和取平均，便是名副其实的 Token-Level。