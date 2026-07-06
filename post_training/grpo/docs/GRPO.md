# GRPO

本文是本人对GRPO算法的理解，从全流程介绍，到最后遇到的一些问题的解决。结合代码实现，详细解释每一步。

## 全流程概述

GRPO（Group Relative Policy Optimization）是一种强化学习算法，用于优化语言模型的策略。与传统PPO不同，GRPO通过组内相对优势来计算奖励，使得训练更加稳定。

总体流程：

```
输入数据 → 编码 → 采样(Rollout) → 奖励计算 → 优势计算 → 损失计算 → 反向传播 → 重复
```

以10以内加减法为例，完整流程如下：

1. **输入数据**：准备prompts（如"1+1=?"）和对应的标准答案ground_truths（如"2"）
2. **编码**：将字符串转化为token序列
3. **采样**：使用旧策略(old_policy)生成多个响应
4. **奖励计算**：根据规则验证回答是否正确
5. **优势计算**：组内归一化计算优势值
6. **损失计算**：计算剪裁目标和KL散度惩罚
7. **更新策略**：反向传播优化策略网络

---

## 1. 输入数据

构建训练集数据，需要两个部分：

- **prompts**：问题列表，每个问题是一个字符串
- **ground_truths**：对应的标准答案列表

```python
train_prompts = ["1+1=?", "2+3=?", "3+4=?", "4+5=?"]
train_ground_truths = ["2", "5", "7", "9"]
```

每个prompt对应一个ground_truth，索引一一对应。

---

## 2. 分词器与词汇表

### 2.1 词汇表构建

词汇表（vocab）是字符到整数索引的映射。代码中定义的词汇表如下：

| 字符 | 索引 | 字符 | 索引 |
|------|------|------|------|
| "0" | 0 | "?" | 15 |
| "1" | 1 | "(" | 16 |
| "2" | 2 | ")" | 17 |
| "3" | 3 | " " | 18 |
| "4" | 4 | "\n" | 19 |
| "5" | 5 | "<PAD>" | 20 |
| "6" | 6 | "<BOS>" | 21 |
| "7" | 7 | "<EOS>" | 22 |
| "8" | 8 | | |
| "9" | 9 | | |
| "+" | 10 | | |
| "-" | 11 | | |
| "=" | 12 | | |
| "<" | 13 | | |
| ">" | 14 | | |

特殊token说明：

- **\<PAD>**（索引20）：填充token，用于补齐不足最大长度的序列
- **\<BOS>**（索引21）：Begin of Sequence，序列开始标记
- **\<EOS>**（索引22）：End of Sequence，序列结束标记

### 2.2 编码（encode）

编码过程将字符串转化为token序列，主要步骤：

1. 根据词汇表获取每个字符的索引
2. 在序列开头添加`<BOS>`，结尾添加`<EOS>`
3. 如果长度不足max_len，用`<PAD>`填充

以"1+1=?"为例，max_len=12：

| 位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|------|---|---|---|---|---|---|---|---|---|---|----|----|
| 字符 | \<BOS> | 1 | + | 1 | = | ? | \<EOS> | \<PAD> | \<PAD> | \<PAD> | \<PAD> | \<PAD> |
| 索引 | 21 | 1 | 10 | 1 | 12 | 15 | 22 | 20 | 20 | 20 | 20 | 20 |

代码实现（[simple_dum_grpo.py#L20-L39]）：

```python
def encode(self, text: str, max_len: int = 12) -> torch.Tensor:
    # 步骤1：根据词汇表获取token编码
    ids = [self.vocab.get(ch, self.pad_token_id) for ch in text[:max_len]]
    
    # 步骤2：添加<BOS>和<EOS>
    ids = [self.bos_token_id] + ids + [self.eos_token_id]
    
    # 步骤3：填充到max_len
    if len(ids) < max_len:
        ids += [self.pad_token_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    
    # 返回形状 [1, L] 的张量（batch_size=1）
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
```

### 2.3 解码（decode）

解码过程是编码的逆操作，将token序列转化为字符串：

```python
def decode(self, ids: torch.Tensor) -> str:
    return ''.join([self.inv_vocab.get(int(i), '') for i in ids 
                    if int(i) not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]])
```

解码时自动忽略`<PAD>`、`<BOS>`、`<EOS>`三个特殊token。

---

## 3. RoPE旋转位置编码

### 3.1 原理

RoPE（Rotary Position Embedding）是一种位置编码方法，通过旋转query和key向量来引入位置信息。核心思想是：

1. 将向量沿最后一维分成两半：`x = [x1, x2]`
2. 对后半部分取负并交换：`rotate_half(x) = [-x2, x1]`
3. 使用三角函数进行旋转：`q_rot = q * cos + rotate_half(q) * sin`

### 3.2 代码实现

旋转函数（[simple_dum_grpo.py#L49-L51]）：

```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
```

应用RoPE（[simple_dum_grpo.py#L53-L62]）：

```python
def apply_rope(q, k, seq_len, head_dim):
    # 计算频率：inv_freq[i] = 1 / 10000^(2i / head_dim)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim))
    
    # 生成位置索引：t = [0, 1, 2, ..., seq_len-1]
    t = torch.arange(seq_len, device=q.device).float()
    
    # 计算频率矩阵：freqs[i, j] = t[i] * inv_freq[j]
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    
    # 拼接得到完整的位置编码
    emb = torch.cat((freqs, freqs), dim=-1)
    
    # 计算cos和sin，形状 [1, 1, L, head_dim]
    cos = emb.cos().unsqueeze(0).unsqueeze(1)
    sin = emb.sin().unsqueeze(0).unsqueeze(1)
    
    # 应用旋转
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    
    return q_rot, k_rot
```

### 3.3 维度变化

假设输入q的形状为`[B, num_heads, L, head_dim]`：

1. `inv_freq`形状：`[head_dim/2]`
2. `t`形状：`[L]`
3. `freqs`形状：`[L, head_dim/2]`（通过einsum外积）
4. `emb`形状：`[L, head_dim]`（拼接两次freqs）
5. `cos/sin`形状：`[1, 1, L, head_dim]`（添加两个维度适配batch和heads）
6. 输出`q_rot, k_rot`形状：`[B, num_heads, L, head_dim]`（与输入相同）

---

## 4. Transformer层实现

### 4.1 多头注意力机制

多头注意力（MultiHeadAttention）将输入映射到多个头，每个头独立计算注意力，最后拼接输出。

代码实现（[simple_dum_grpo.py#L65-L93]）：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0  # d_model必须能被num_heads整除
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 每个头的维度
        
        # 定义投影矩阵
        self.q_proj = nn.Linear(d_model, d_model, bias=False)  # Q投影
        self.k_proj = nn.Linear(d_model, d_model, bias=False)  # K投影
        self.v_proj = nn.Linear(d_model, d_model, bias=False)  # V投影
        self.o_proj = nn.Linear(d_model, d_model, bias=False)  # 输出投影
    
    def forward(self, x, mask=None):
        B, L, _ = x.shape
        
        # 步骤1：线性投影，形状 [B, L, d_model] → [B, L, num_heads, head_dim] → [B, num_heads, L, head_dim]
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 步骤2：应用RoPE位置编码
        q, k = apply_rope(q, k, L, self.head_dim)
        
        # 步骤3：计算注意力分数，形状 [B, num_heads, L, L]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 步骤4：应用mask（如果有）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 步骤5：softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 步骤6：加权求和，形状 [B, num_heads, L, head_dim]
        attn_out = torch.matmul(attn_weights, v)
        
        # 步骤7：拼接多头输出，形状 [B, num_heads, L, head_dim] → [B, L, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        
        # 步骤8：输出投影
        return self.o_proj(attn_out)
```

### 4.2 Transformer Block

Transformer Block包含注意力层和前馈网络（FFN），并使用残差连接和层归一化。

代码实现（[simple_dum_grpo.py#L95-L110]）：

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=32):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)  # 注意力层前的层归一化
        self.ln2 = nn.LayerNorm(d_model)  # FFN前的层归一化
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),    # 升维
            nn.GELU(),                   # 激活函数
            nn.Linear(d_ff, d_model),    # 降维
        )
    
    def forward(self, x, mask=None):
        # 残差连接：x = x + attn(ln1(x))
        x = x + self.attn(self.ln1(x), mask)
        
        # 残差连接：x = x + ffn(ln2(x))
        x = x + self.ffn(self.ln2(x))
        
        return x
```

**注意**：代码使用的是Pre-LN架构，即层归一化在注意力/FFN之前。

### 4.3 MiniTransformer完整模型

MiniTransformer是完整的语言模型，包含嵌入层、多层Transformer Block、最终层归一化和语言模型头。

代码实现（[simple_dum_grpo.py#L112-L154]）：

```python
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, pad_token_id, d_model=16, num_heads=2, num_layers=2):
        super().__init__()
        self.pad_token_id = pad_token_id
        
        # 嵌入层：将token映射到d_model维向量
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Transformer层
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        
        # 最终层归一化
        self.ln_final = nn.LayerNorm(d_model)
        
        # 语言模型头：将d_model维向量映射到vocab_size维logits
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, tokens, mask=None):
        # 步骤1：嵌入，形状 [B, L] → [B, L, d_model]
        x = self.embed(tokens)
        
        # 步骤2：逐层Transformer
        for layer in self.layers:
            x = layer(x, mask)
        
        # 步骤3：最终层归一化
        x = self.ln_final(x)
        
        # 步骤4：输出logits，形状 [B, L, vocab_size]
        logits = self.lm_head(x)
        
        return logits
```

### 4.4 嵌入层详解

嵌入层（Embedding）的作用是将离散的token索引转化为连续的向量表示。

对于输入tokens形状`[B, L]`（B是batch_size，L是序列长度）：
- 每个token索引（0到vocab_size-1）被映射为一个d_model维的向量
- 输出形状为`[B, L, d_model]`

每个位置的输出向量表示当前token的语义信息，后续的Transformer层会利用这些向量进行上下文建模。

### 4.5 语言模型头（lm_head）

语言模型头是一个线性层，将d_model维的隐藏状态映射到vocab_size维的logits。

- 输入：`[B, L, d_model]`
- 输出：`[B, L, vocab_size]`

每个位置的logits表示当前位置预测下一个token时，每个词的原始分数（未归一化的概率）。通过softmax可以得到概率分布：

```python
probs = F.softmax(logits, dim=-1)  # 形状 [B, L, vocab_size]
```

### 4.6 get_log_probs方法

该方法计算整个序列的自回归平均log概率，用于策略优化。

代码实现（[simple_dum_grpo.py#L129-L154]）：

```python
def get_log_probs(self, tokens: torch.Tensor):
    """
    计算整个序列的自回归平均 log 概率。
    忽略 PAD token，只对有效 token 计算。
    """
    B, L = tokens.shape
    if L <= 1:
        return torch.zeros(B, device=tokens.device)
    
    # 步骤1：前向传播得到logits
    logits = self.forward(tokens)  # [B, L, V]
    
    # 步骤2：自回归预测
    shift_logits = logits[:, :-1, :]  # [B, L-1, V]  去掉最后一个位置的logits
    shift_tokens = tokens[:, 1:]      # [B, L-1]     去掉第一个token
    
    # 步骤3：计算log概率
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, L-1, V]
    token_log_probs = log_probs.gather(dim=-1, index=shift_tokens.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    
    # 步骤4：忽略PAD token
    mask = (shift_tokens != self.pad_token_id).float()  # [B, L-1]
    log_prob_sum = (token_log_probs * mask).sum(dim=-1)  # [B]
    valid_count = mask.sum(dim=-1).clamp(min=1)          # [B]
    avg_log_prob = log_prob_sum / valid_count            # [B]
    
    return avg_log_prob
```

**自回归解释**：
- 输入token序列：`[t0, t1, t2, ..., tL-1]`
- 使用`t0`预测`t1`，使用`t0,t1`预测`t2`，依此类推
- `shift_logits[:, i, :]`表示在看到`t0..ti`后对`t_i+1`的预测logits
- `shift_tokens[:, i]`表示实际的下一个token`t_i+1`

---

## 5. 生成回答（Rollout）

Rollout是使用策略网络生成响应的过程。代码中使用核采样（top-p sampling）来增加多样性。

代码实现（[simple_dum_grpo.py#L157-L200]）：

```python
@torch.no_grad()
def generate_response(
    model: MiniTransformer,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 1,       # 生成新token的最大数量
    temperature: float = 1.0,      # 温度，控制随机性
    do_sample: bool = True,        # 是否采样
    top_p: float = 0.9,            # 核采样参数
):
    model.eval()
    
    # 步骤1：编码prompt
    input_ids = tokenizer.encode(prompt, max_len=12)  # [1, L]
    generated = input_ids.squeeze(0).tolist()
    
    # 步骤2：逐token生成
    for _ in range(max_new_tokens):
        # 构造输入张量
        tokens = torch.tensor([generated], device=next(model.parameters()).device)
        
        # 前向传播
        logits = model(tokens)  # [1, seq_len, vocab]
        
        # 取最后一个位置的logits，并进行温度缩放
        next_token_logits = logits[0, -1, :] / temperature
        
        # 步骤3：采样策略
        if do_sample:
            # 核采样：只保留累积概率 <= top_p 的token
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cumsum_probs > top_p] = -float('Inf')
                probs = F.softmax(sorted_logits, dim=-1)
            else:
                probs = F.softmax(next_token_logits, dim=-1)
            
            # 从概率分布中采样
            next_token = torch.multinomial(probs, 1).item()
        else:
            # 贪婪解码：取概率最大的token
            next_token = torch.argmax(next_token_logits).item()
        
        # 步骤4：检查是否结束
        if next_token == tokenizer.eos_token_id:
            break
        
        # 步骤5：添加到生成序列
        generated.append(next_token)
    
    return torch.tensor(generated, device=next(model.parameters()).device).unsqueeze(0)
```

### 5.1 温度（Temperature）

温度参数控制采样的随机性：

- `temperature > 1`：增加随机性，生成更多样化的文本
- `temperature = 1`：原始分布
- `temperature < 1`：降低随机性，生成更确定的文本

### 5.2 核采样（Top-p Sampling）

核采样只保留累积概率不超过top_p的token：

1. 将logits按降序排序
2. 计算累积概率
3. 将累积概率超过top_p的token的logits设为负无穷
4. 重新归一化并采样

这样可以避免低概率token被采样到，同时保持一定的多样性。

### 5.3 贪婪解码 vs 采样

- **贪婪解码**（`do_sample=False`）：每次选择概率最大的token，结果确定
- **采样**（`do_sample=True`）：根据概率分布随机选择token，结果多样

---

## 6. 奖励函数

奖励函数用于评估生成响应的质量。代码中使用规则验证奖励函数。

代码实现（[simple_dum_grpo.py#L202-L265]）：

```python
def math_reward_fn(
    response_tokens: torch.Tensor,
    tokenizer: SimpleTokenizer,
    ground_truths: List[str],
    group_size: int,
    verbose: bool = False
) -> torch.Tensor:
    """
    规则验证奖励函数：检查每个生成的响应文本是否包含对应的正确答案。
    
    工作原理：
        - response_tokens 的总行数 = len(ground_truths) * group_size
        - 顺序是：先对第1个问题生成 group_size 个响应，再对第2个问题生成 group_size 个响应，依此类推。
        - 因此，第 i 个响应对应的问题索引为 i // group_size。
        - 对每个响应，解码为文本，检查对应问题的标准答案是否出现在文本中。
        - 包含则奖励 1.0，否则 0.0。
    
    参数:
        response_tokens: torch.Tensor，形状 [total_samples, seq_len]
        tokenizer: SimpleTokenizer，用于将 token 序列解码为可读字符串
        ground_truths: List[str]，每个问题对应的标准答案（与问题顺序一致）
        group_size: int，每个问题采样的响应数量
        verbose: bool，是否打印解码文本（便于调试）
    
    返回:
        torch.Tensor: 形状 [total_samples]，每个样本的奖励值（0.0 或 1.0）
    """
    total_samples = response_tokens.shape[0]
    expected_total = len(ground_truths) * group_size
    
    # 验证输入维度
    if total_samples != expected_total:
        raise ValueError(
            f"响应总数 ({total_samples}) 与期望 ({expected_total}) 不匹配。"
        )
    
    rewards = []
    for i, tokens in enumerate(response_tokens):
        # 步骤1：确定对应的问题索引
        question_idx = i // group_size
        
        # 步骤2：获取标准答案
        correct_answer = ground_truths[question_idx]
        
        # 步骤3：解码为文本
        text = tokenizer.decode(tokens)
        
        # 步骤4：规则验证
        if correct_answer in text:
            reward = 1.0   # 答对，正奖励
        else:
            reward = 0.0   # 答错，零奖励
        
        rewards.append(reward)
    
    return torch.tensor(rewards, device=response_tokens.device)
```

### 6.1 奖励计算示例

假设：
- `ground_truths = ["2", "5"]`（两个问题的标准答案）
- `group_size = 2`（每个问题生成2个响应）

响应顺序：

| 索引i | question_idx | 响应文本 | 标准答案 | 奖励 |
|-------|-------------|---------|---------|------|
| 0 | 0 | "1+1=2" | "2" | 1.0 |
| 1 | 0 | "1+1=3" | "2" | 0.0 |
| 2 | 1 | "2+3=5" | "5" | 1.0 |
| 3 | 1 | "2+3=8" | "5" | 0.0 |

---

## 7. GRPO训练器

GRPO训练器是核心部分，负责执行完整的训练循环。

### 7.1 初始化

代码实现（[simple_dum_grpo.py#L267-L281]）：

```python
class GRPOTrainer:
    def __init__(self, policy, ref, tokenizer, config):
        self.policy = policy          # 当前策略（待优化）
        self.ref = ref                # 参考策略（冻结，用于KL散度计算）
        self.tokenizer = tokenizer    # 分词器
        self.config = config          # 配置参数
        
        # 优化器
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)
        
        # 冻结参考策略
        for p in ref.parameters():
            p.requires_grad = False
        
        # 创建并冻结old_policy（用于采样）
        self.old_policy = copy.deepcopy(policy)
        for p in self.old_policy.parameters():
            p.requires_grad = False
```

三个模型的作用：

| 模型 | 作用 | 是否可训练 |
|------|------|-----------|
| `policy` | 当前策略，待优化 | 是 |
| `ref` | 参考策略，用于KL散度惩罚 | 否（冻结） |
| `old_policy` | 旧策略，用于采样 | 否（每步更新后复制policy） |

### 7.2 train_step完整流程

代码实现（[simple_dum_grpo.py#L283-L359]）：

```python
def train_step(self, prompts, ground_truths):
    B = len(prompts)           # batch_size
    G = self.config.group_size # 每个问题采样的响应数
    device = next(self.policy.parameters()).device
    
    # ===== 步骤1：同步old_policy =====
    # 将当前策略复制给旧策略（作为采样策略）
    self.old_policy.load_state_dict(self.policy.state_dict())
    
    # ===== 步骤2：Rollout采样 =====
    # 使用old_policy为每个prompt生成G个响应
    all_responses = []
    for prompt in prompts:
        for _ in range(G):
            resp = generate_response(model=self.old_policy, tokenizer=self.tokenizer, prompt=prompt)
            all_responses.append(resp)
    
    # 填充到相同长度
    max_len = max(r.size(1) for r in all_responses)
    padded_responses = []
    for r in all_responses:
        pad_len = max_len - r.size(1)
        if pad_len > 0:
            pad = torch.full((1, pad_len), self.tokenizer.pad_token_id, device=device)
            r_pad = torch.cat([r, pad], dim=1)
        else:
            r_pad = r
        padded_responses.append(r_pad)
    
    responses = torch.cat(padded_responses, dim=0)  # 形状 [B*G, max_len]
    
    # ===== 步骤3：计算old_log_probs =====
    # 使用old_policy计算响应的log概率（无梯度）
    with torch.no_grad():
        old_log_probs = self.old_policy.get_log_probs(responses)  # [B*G]
    
    # ===== 步骤4：计算奖励 =====
    rewards = math_reward_fn(responses, self.tokenizer, ground_truths, G)  # [B*G]
    rewards = rewards.view(B, G)  # 形状 [B, G]
    
    # ===== 步骤5：计算优势 =====
    # GRPO的核心：组内归一化优势
    mean_r = rewards.mean(dim=1, keepdim=True)  # 形状 [B, 1]，每个问题的平均奖励
    std_r = rewards.std(dim=1, keepdim=True)    # 形状 [B, 1]，每个问题的奖励标准差
    advantages = (rewards - mean_r) / (std_r + 1e-8)  # 形状 [B, G]，组内归一化优势
    advantages_flat = advantages.view(-1)  # 形状 [B*G]
    
    # ===== 步骤6：计算参考策略log_prob =====
    # 使用ref策略计算响应的log概率（无梯度）
    with torch.no_grad():
        log_probs_ref = self.ref.get_log_probs(responses)  # [B*G]
    
    # ===== 步骤7：PPO epochs梯度更新 =====
    EPS = 0.2        # 剪裁参数
    BETA = 0.0001    # KL惩罚系数
    
    for _ in range(self.config.ppo_epochs):
        # 重新计算当前策略的log概率（有梯度）
        log_probs_theta = self.policy.get_log_probs(responses)  # [B*G]
        
        # 计算比率：ratio = exp(log_prob_theta - log_prob_old)
        ratio = torch.exp(log_probs_theta - old_log_probs)
        
        # PPO剪裁
        clipped_ratio = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS)
        
        # 剪裁目标
        surr1 = ratio * advantages_flat
        surr2 = clipped_ratio * advantages_flat
        surrogate_loss = torch.min(surr1, surr2)
        
        # KL散度惩罚（使用参考策略）
        log_diff = log_probs_ref - log_probs_theta
        kl_div = torch.exp(log_diff) - log_diff - 1  # KL散度的一种计算方式
        kl_loss = BETA * kl_div.mean()
        
        # 总损失
        loss = -surrogate_loss.mean() + kl_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)  # 梯度裁剪
        self.optimizer.step()
    
    return {
        'loss': loss.item(),
        'kl_loss': kl_loss.item(),
        'mean_ratio': ratio.mean().item(),
    }
```

### 7.3 优势计算详解

GRPO的核心创新在于**组内归一化优势**：

```python
mean_r = rewards.mean(dim=1, keepdim=True)  # 形状 [B, 1]
std_r = rewards.std(dim=1, keepdim=True)    # 形状 [B, 1]
advantages = (rewards - mean_r) / (std_r + 1e-8)  # 形状 [B, G]
```

假设有2个问题（B=2），每个问题采样4个响应（G=4）：

```
rewards = [[1.0, 1.0, 0.0, 0.0],  # 问题0的4个响应奖励
           [1.0, 0.0, 0.0, 0.0]]  # 问题1的4个响应奖励

mean_r = [[0.5],   # 问题0的平均奖励
          [0.25]]  # 问题1的平均奖励

std_r = [[0.5],    # 问题0的标准差
         [0.433]]  # 问题1的标准差

advantages = [[1.0, 1.0, -1.0, -1.0],  # 问题0的优势
              [1.732, -0.577, -0.577, -0.577]]  # 问题1的优势
```

**为什么使用组内归一化？**

- 不同问题的难度不同，奖励分布也不同
- 组内归一化使得每个问题的优势在相同尺度上比较
- 避免某个简单问题的大量正奖励主导整个训练

### 7.4 PPO剪裁目标

PPO使用剪裁目标来限制策略更新的幅度：

```python
ratio = torch.exp(log_probs_theta - old_log_probs)
clipped_ratio = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS)
surr1 = ratio * advantages_flat
surr2 = clipped_ratio * advantages_flat
surrogate_loss = torch.min(surr1, surr2)
```

- **ratio**：当前策略与旧策略的概率比值
- **clipped_ratio**：将ratio限制在[0.8, 1.2]范围内
- **surrogate_loss**：取surr1和surr2的最小值，形成剪裁目标

**直觉理解**：

- 如果`ratio > 1.2`（当前策略概率远大于旧策略），且优势为正，则只按`1.2 * advantage`更新，避免过度更新
- 如果`ratio < 0.8`（当前策略概率远小于旧策略），且优势为负，则只按`0.8 * advantage`更新

### 7.5 KL散度惩罚

为了防止策略偏离参考策略太远，引入KL散度惩罚：

```python
log_diff = log_probs_ref - log_probs_theta
kl_div = torch.exp(log_diff) - log_diff - 1
kl_loss = BETA * kl_div.mean()
```

使用的是KL散度的一种等价形式：`KL(P_ref || P_theta) = E_ref[log(P_ref/P_theta)] = E_ref[log_diff]`

这里使用的是采样估计：`kl_div = exp(log_diff) - log_diff - 1`

**为什么需要KL惩罚？**

- 防止策略在优化过程中变得过于激进
- 保持策略的稳定性
- BETA控制惩罚强度

### 7.6 总损失函数

```python
loss = -surrogate_loss.mean() + kl_loss
```

总损失由两部分组成：

1. **负的剪裁目标**：最大化奖励（因为loss越小越好，所以取负）
2. **KL散度惩罚**：限制策略与参考策略的偏离

---

## 8. 评估函数

评估函数用于计算模型在测试集上的准确率。

代码实现（[simple_dum_grpo.py#L362-L382]）：

```python
def evaluate_model(model, tokenizer, test_prompts, test_ground_truths, max_new_tokens=1):
    """
    评估模型在测试集上的准确率（贪婪解码）。
    """
    model.eval()
    correct = 0
    total = len(test_prompts)
    
    with torch.no_grad():
        for prompt, gt in zip(test_prompts, test_ground_truths):
            # 使用贪婪解码（确定性）
            resp_tokens = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # 不采样，取概率最大的token
                temperature=1.0
            )
            
            # 解码并检查答案
            text = tokenizer.decode(resp_tokens.squeeze(0))
            if gt in text:
                correct += 1
    
    return correct / total
```

评估使用贪婪解码（`do_sample=False`），确保结果可重复。

---

## 9. 主程序

主程序负责初始化模型、配置参数、执行训练和评估。

代码实现（[simple_dum_grpo.py#L385-L428]）：

```python
if __name__ == "__main__":
    # 步骤1：设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 步骤2：初始化分词器
    tokenizer = SimpleTokenizer()
    vocab_size = tokenizer.vocab_size
    
    # 步骤3：初始化策略和参考模型
    policy = MiniTransformer(vocab_size, pad_token_id=tokenizer.pad_token_id, 
                            d_model=16, num_heads=2, num_layers=2).to(device)
    ref = MiniTransformer(vocab_size, pad_token_id=tokenizer.pad_token_id, 
                          d_model=16, num_heads=2, num_layers=2).to(device)
    ref.load_state_dict(policy.state_dict())  # 参考模型初始化为策略模型的副本
    
    # 步骤4：配置参数
    class Config:
        group_size = 8    # 每个问题采样8个响应
        lr = 1e-2         # 学习率
        ppo_epochs = 2    # 每步对同一批数据重复更新的次数
    config = Config()
    
    # 步骤5：初始化训练器
    trainer = GRPOTrainer(policy, ref, tokenizer, config)
    
    # 步骤6：准备数据
    train_prompts = ["1+1=?", "2+3=?", "3+4=?", "4+5=?"]
    train_ground_truths = ["2", "5", "7", "9"]
    
    test_prompts = ["3+4=?", "4+5=?"]
    test_ground_truths = ["7", "9"]
    
    # 步骤7：训练循环
    steps_num = 10
    eval_interval = 20
    
    for step in range(steps_num):
        metrics = trainer.train_step(train_prompts, train_ground_truths)
        print(f"Step {step}: Loss={metrics['loss']:.4f}, KL={metrics['kl_loss']:.4f}, Ratio={metrics['mean_ratio']:.4f}")
        
        # 定期评估（可选）
        if to_test and (step + 1) % eval_interval == 0:
            acc = evaluate_model(policy, tokenizer, test_prompts, test_ground_truths)
            print(f"  >>> Test Accuracy: {acc:.2f}")
    
    print("训练完成！")
```

---

## 10. 常见问题与解决

### 10.1 梯度消失/爆炸

**问题**：训练过程中loss变为NaN或Inf。

**解决**：

1. 使用梯度裁剪：`torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)`
2. 降低学习率：`lr = 1e-3` 或更小
3. 检查数值稳定性：在除法中添加epsilon（如`1e-8`）

### 10.2 奖励稀疏

**问题**：大部分响应的奖励都是0，导致训练困难。

**解决**：

1. 增加group_size，使得每个组内有更多样本，优势估计更准确
2. 调整奖励函数，给予部分正确的响应一定的奖励

### 10.3 KL散度惩罚过大

**问题**：KL损失占主导，策略更新缓慢。

**解决**：

1. 减小BETA值
2. 使用自适应KL惩罚（根据KL散度动态调整BETA）

### 10.4 训练不稳定

**问题**：loss波动较大，策略性能不稳定。

**解决**：

1. 增加ppo_epochs，对同一批数据进行多次更新
2. 使用更小的剪裁参数EPS
3. 确保old_policy在每步训练开始时正确同步

---

## 11. GRPO与PPO的区别

| 方面 | PPO | GRPO |
|------|-----|------|
| 优势计算 | 基于累计回报 | 组内归一化优势 |
| 采样方式 | 单个样本 | 每组多个样本 |
| 稳定性 | 中等 | 更高（组内归一化） |
| 适用场景 | 连续控制、语言模型 | 语言模型、需要稳定训练的场景 |

**GRPO的核心改进**：通过组内归一化，使得不同问题的优势在相同尺度上比较，避免某个问题主导训练，从而提高训练稳定性。

---

## 12. 完整数据流示例（以"1+1=?"为例）

本节以具体的运行输出为例，详细展示从输入"1+1=?"到完成一次训练更新的完整数据流过程。

### 12.1 配置参数

```python
batch_size (B) = 1          # 一个问题
group_size (G) = 3          # 每个问题生成3个响应
总样本数 = B * G = 3
prompts = ["1+1=?"]
ground_truths = ["2"]
```

### 12.2 步骤1：编码过程

输入：`"1+1=?"`

**步骤1.1：字符转token**

| 字符 | '1' | '+' | '1' | '=' | '?' |
|------|-----|-----|-----|-----|-----|
| token | 1 | 10 | 1 | 12 | 15 |

**步骤1.2：添加<BOS>和<EOS>**

```
[21] + [1, 10, 1, 12, 15] + [22] = [21, 1, 10, 1, 12, 15, 22]
```

**步骤1.3：补充<PAD>到max_len=12**

```
[21, 1, 10, 1, 12, 15, 22, 20, 20, 20, 20, 20]
```

**最终token序列（长度12）**

| 位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|------|---|---|---|---|---|---|---|---|---|---|----|----|
| token | 21 | 1 | 10 | 1 | 12 | 15 | 22 | 20 | 20 | 20 | 20 | 20 |
| 含义 | <BOS> | '1' | '+' | '1' | '=' | '?' | <EOS> | <PAD> | <PAD> | <PAD> | <PAD> | <PAD> |

### 12.3 步骤2：嵌入层（Embedding）

将token序列转化为向量表示：

**输入**：`[21, 1, 10, 1, 12, 15, 22, 20, 20, 20, 20, 20]`（形状 `[1, 12]`）

**输出**：形状 `[1, 12, 16]`，每个token被映射为16维向量

**实际数据（前3个token的完整向量）**：

```python
token[0] = 21 ('<BOS>'): 
  [0.4253, 1.4747, 0.6494, 1.3766, 1.0390, 0.4811, -0.2884, -0.6953, 
   0.1394, 1.3048, 0.4172, 0.0877, 1.0554, -1.6589, -0.2349, 0.0685]

token[1] = 1 ('1'): 
  [-1.0192, 0.2344, 0.3224, -0.0478, 0.0560, -1.1232, 0.0589, -0.0202, 
   -0.1050, -1.2400, 0.3455, 1.5442, -0.6208, -1.8473, -0.7452, -1.0137]

token[2] = 10 ('+'): 
  [0.0477, -0.0422, 0.2855, -0.6764, 1.8822, 1.4232, -1.0232, 0.7809, 
   -0.2504, -1.1280, 0.4082, -1.3760, -0.1036, -2.5896, 0.4694, -0.5112]
```

**含义**：每个token通过嵌入层被转化为一个16维的向量，这些向量包含了token的语义信息。不同的token有不同的向量表示，例如`<BOS>`的向量和`1`的向量完全不同。

### 12.4 步骤3：RoPE位置编码

RoPE（Rotary Position Embedding）通过旋转query和key向量来引入位置信息。

**参数**：`seq_len=12`, `head_dim=8`

**步骤3.1：计算频率**

```python
# 频率倒数公式: inv_freq[i] = 1 / 10000^(2i / head_dim)
inv_freq = [1.0, 0.1, 0.01, 0.001]

# 位置索引
t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

# 频率矩阵: freqs[i, j] = t[i] * inv_freq[j]
freqs[0] = [0.0, 0.0, 0.0, 0.0]      # 位置0
freqs[1] = [1.0, 0.1, 0.01, 0.001]   # 位置1
freqs[2] = [2.0, 0.2, 0.02, 0.002]   # 位置2
...
```

**步骤3.2：计算cos和sin**

```python
# 拼接频率矩阵，使前半和后半相同
emb形状: [12, 8]

# 计算余弦和正弦值
cos[0,0,0] (位置0): [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
sin[0,0,0] (位置0): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

cos[0,0,1] (位置1): [0.5403, 0.9950, 0.9999, 1.0000, 0.5403, 0.9950, 0.9999, 1.0000]
sin[0,0,1] (位置1): [0.8415, 0.0998, 0.0100, 0.0010, 0.8415, 0.0998, 0.0100, 0.0010]
```

**步骤3.3：应用旋转**

旋转公式：
```
rotate_half([x1, x2]) = [-x2, x1]
q_rot = q * cos + rotate_half(q) * sin
k_rot = k * cos + rotate_half(k) * sin
```

**实际数据对比**：

```python
# 位置0（cos=1, sin=0，旋转后不变）
q[0,0,0] (旋转前): [0.2981, -0.2399, -0.1217, 0.4511, -0.4076, -0.6983, 0.4583, 0.2658]
q_rot[0,0,0] (旋转后): [0.2981, -0.2399, -0.1217, 0.4511, -0.4076, -0.6983, 0.4583, 0.2658]

# 位置1（cos≠1, sin≠0，旋转后变化）
q[0,0,1] (旋转前): [0.5683, -0.8895, 0.5188, 1.6410, ...]
q_rot[0,0,1] (旋转后): [0.1282, -0.8314, 0.5214, 1.6412, ...]
```

**含义**：位置编码使模型能够区分不同位置的token。位置0的向量不旋转（cos=1, sin=0），位置越靠后，旋转角度越大。

### 12.5 步骤4：多头注意力机制

**参数**：`B=1`, `L=12`, `d_model=16`, `num_heads=2`, `head_dim=8`

**步骤4.1：QKV投影**

```python
# 投影后形状: [1, 2, 12, 8]
Q[0,0,0] (第1个头，位置0): [0.2981, -0.2399, -0.1217, 0.4511, -0.4076, -0.6983, 0.4583, 0.2658]
K[0,0,0] (第1个头，位置0): [0.4202, -0.0088, -0.0339, 0.6711, -0.1311, 0.1543, 0.3513, -0.2554]
V[0,0,0] (第1个头，位置0): [0.3432, -0.3038, 0.3742, 0.7984, 0.9402, 0.2229, -0.3999, -0.3320]
```

**步骤4.2：计算注意力分数**

```python
# 注意力分数形状: [1, 2, 12, 12]
# attn_scores[i,j] = Q[i] · K[j] / sqrt(head_dim)

attn_scores[0,0] (第1个头):
  位置0 -> [0.1672, 0.0070, -0.0907, -0.2315, 0.0168, ...]
  位置1 -> [0.3261, 0.0170, -0.3590, -0.0072, 0.6373, ...]
  位置2 -> [0.0532, 0.0792, -0.3044, 0.1065, 0.1664, ...]
  ...
```

**步骤4.3：Softmax归一化**

```python
# 每行的和为1.0
attn_weights[0,0] (第1个头):
  位置0注意力: [0.0919, 0.0783, 0.0710, 0.0616, 0.0790, ...]  (和=1.0000)
  位置1注意力: [0.1225, 0.0900, 0.0618, 0.0878, 0.1673, ...]  (和=1.0000)
  位置2注意力: [0.0732, 0.0752, 0.0512, 0.0772, 0.0820, ...]  (和=1.0000)
  ...
```

**步骤4.4：加权求和**

```python
# attn_out = attn_weights × V
attn_out[0,0,0]: [0.3170, 0.0164, 0.4377, 0.6254, -0.4418, 0.1350, 0.0976, -0.4371]
```

**步骤4.5：拼接多头并输出投影**

```python
# 拼接后形状: [1, 12, 16]
# 输出投影后形状: [1, 12, 16]
out[0,0]: [-0.3210, 0.1176, -0.0257, -0.0879, -0.0922, 0.2348, 0.1637, -0.1244, 
           -0.1028, 0.0065, 0.0912, 0.0349, 0.1612, 0.0181, 0.0160, 0.2365]
```

**含义**：多头注意力让模型能够同时关注序列中不同位置的信息。每个头学习不同的注意力模式，最终拼接所有头的输出。

### 12.6 步骤5：Transformer Block与残差连接

**输入**：形状 `[1, 12, 16]`

**步骤5.1：第一个残差连接（注意力层）**

```python
# ln1(x)[0,0] (层归一化后): 
#   [-0.3274, 1.4858, 0.3055, 1.1159, 0.6992, ...]

# attn_out[0,0] (注意力输出): 
#   [-0.3210, 0.1176, -0.0257, -0.0879, -0.0922, ...]

# 残差连接: x = x + attn(ln1(x))
# x[0,0] (残差后): 
#   [0.1043, 1.5923, 0.6237, 1.2887, 0.9467, ...]
```

**步骤5.2：第二个残差连接（前馈网络）**

```python
# ln2(x)[0,0] (层归一化后): 
#   [-0.3274, 1.4858, 0.3055, 1.1159, 0.6992, ...]

# ffn_out[0,0] (前馈网络输出): 
#   [0.0310, -0.0612, -0.1332, -0.0366, -0.1330, ...]

# 残差连接: x = x + ffn(ln2(x))
# x[0,0] (残差后): 
#   [0.1353, 1.5311, 0.4905, 1.2520, 0.8137, ...]
```

**含义**：残差连接允许梯度直接传播，缓解了深层网络的梯度消失问题。Pre-LN架构（层归一化在注意力/FFN之前）使训练更稳定。

### 12.7 步骤6：语言模型头与logits

**输入**：形状 `[1, 12, 16]`（经过两层Transformer Block后）

**步骤6.1：最终层归一化**

```python
ln_final输出形状: [1, 12, 16]
x[0,0]: [0.4351, -1.4146, 1.2700, -0.3746, -1.3350, ...]
```

**步骤6.2：语言模型头**

```python
# lm_head: Linear(16, 23)
# 输出形状: [1, 12, 23]

# 最后一个位置的logits（预测下一个token）:
logits[0, -1, :] = [
   -0.5011,  # '0'
    0.1731,  # '1'
    0.0018,  # '2' ← 正确答案，但logit很低
    0.3869,  # '3'
    0.8057,  # '4'
   -1.2490,  # '5'
    0.3014,  # '6'
   -0.4267,  # '7'
   -0.3972,  # '8'
   -0.1056,  # '9'
    0.6591,  # '+'
    0.6557,  # '-'
    0.5885,  # '='
   -0.8925,  # '<'
    0.9140,  # '>'
   -0.1275,  # '?'
    0.3783,  # '('
   -0.4899,  # ')'
   -0.1322,  # ' '
    0.8354,  # '\n'
    1.2815,  # '<PAD>' ← logit最高（模型未训练）
    0.8879,  # '<BOS>'
   -0.5021,  # '<EOS>'
]
```

**含义**：logits是未归一化的分数，表示模型认为每个词汇是"下一个token"的可能性。logit越大，概率越高。

**预测概率最高的3个token**：

| 排名 | token | 字符 | logit | 概率 |
|------|-------|------|-------|------|
| 1 | 20 | '<PAD>' | 1.2815 | 11.36% |
| 2 | 14 | '>' | 0.9140 | 7.87% |
| 3 | 21 | '<BOS>' | 0.8879 | 7.66% |

### 12.8 步骤7：生成响应（Rollout）

使用old_policy生成3个响应：

**采样机制说明**：代码使用`torch.multinomial`进行随机采样（`do_sample=True`），不是取概率最大的token。因此即使某些token概率较低，仍然有机会被选中。

**采样1：**
```
生成前: [21, 1, 10, 1, 12, 15, 22, 20, 20, 20, 20, 20]
预测概率最高的3个: '<PAD>'(11.36%), '>'(7.87%), '<BOS>'(7.66%)
采样选择: token=14 -> '>'
生成后: [21, 1, 10, 1, 12, 15, 22, 20, 20, 20, 20, 20, 14]
解码结果: '1+1=?>'
```

**采样2：**
```
生成后: [21, 1, 10, 1, 12, 15, 22, 20, 20, 20, 20, 20, 4]
解码结果: '1+1=?4'
```

**采样3：**
```
生成后: [21, 1, 10, 1, 12, 15, 22, 20, 20, 20, 20, 20, 14]
解码结果: '1+1=?>'
```

**所有响应填充后（形状 [3, 13]）：**

| 索引 | token序列 | 解码结果 |
|------|----------|---------|
| 0 | [21, 1, 10, 1, 12, 15, 22, 20, 20, 20, 20, 20, 14] | '1+1=?>' |
| 1 | [21, 1, 10, 1, 12, 15, 22, 20, 20, 20, 20, 20, 4] | '1+1=?4' |
| 2 | [21, 1, 10, 1, 12, 15, 22, 20, 20, 20, 20, 20, 14] | '1+1=?>' |

### 12.9 步骤8：计算old_log_probs

使用old_policy计算每个响应的自回归平均log概率：

```python
old_log_probs = [-3.4260, -3.4338, -3.4260]
```

**含义**：每个值表示old_policy生成该响应序列的对数概率。值越大（越接近0），表示策略越"确定"地生成这个序列。

### 12.10 步骤9：计算奖励

奖励函数检查响应中是否包含标准答案"2"：

```python
rewards = [[0., 0., 0.]]  # 形状 [1, 3]
```

| 样本 | 响应 | 标准答案 | 奖励 |
|------|------|---------|------|
| 0 | '1+1=?>' | '2' | 0.0 |
| 1 | '1+1=?0' | '2' | 0.0 |
| 2 | '1+1=?<' | '2' | 0.0 |

**结果**：三个响应都没有包含"2"，所以奖励都是0。这是因为模型还未训练，随机采样的结果。

### 12.11 步骤10：计算优势（GRPO核心）

GRPO使用组内归一化优势：

```python
mean_r = rewards.mean(dim=1, keepdim=True)  # [0.]
std_r = rewards.std(dim=1, keepdim=True)    # [0.]
advantages = (rewards - mean_r) / (std_r + 1e-8)  # [[0., 0., 0.]]
advantages_flat = [0.0, 0.0, 0.0]
```

**含义**：
- 由于所有奖励都是0，组内均值也是0，标准差也是0
- 因此所有响应的优势都是0，表示它们的表现与组内平均水平相同
- 在这种情况下，PPO剪裁目标不会产生任何梯度信号，loss=0

### 12.12 步骤11：计算参考策略log_probs

```python
log_probs_ref = [-3.4260, -3.4338, -3.4260]
```

由于ref策略初始化为policy的副本，且policy尚未更新，所以log_probs_ref与old_log_probs相同。

### 12.13 步骤12：PPO梯度更新

**Epoch 1/1：**

```python
log_probs_theta = [-3.4260, -3.4338, -3.4260]  # 当前策略log概率

ratio = exp(log_probs_theta - old_log_probs) = [1.0, 1.0, 1.0]
clipped_ratio = clamp(ratio, 0.8, 1.2) = [1.0, 1.0, 1.0]

surr1 = ratio * advantages_flat = [0.0, 0.0, 0.0]
surr2 = clipped_ratio * advantages_flat = [0.0, 0.0, 0.0]
surrogate_loss = min(surr1, surr2) = [0.0, 0.0, 0.0]

log_diff = log_probs_ref - log_probs_theta = [0.0, 0.0, 0.0]
kl_div = exp(log_diff) - log_diff - 1 = [0.0, 0.0, 0.0]
kl_loss = 0.0001 * kl_div.mean() = 0.0

loss = -surrogate_loss.mean() + kl_loss = 0.0
```

**结果**：第一次训练时，由于所有响应的奖励都是0，优势都是0，loss=0，模型没有任何更新。

### 12.14 多次训练后的预期变化

经过多次训练后，模型会逐渐学会生成正确的答案"2"：

**假设训练后采样结果：**

| 样本 | 响应 | 标准答案 | 奖励 |
|------|------|---------|------|
| 0 | '1+1=?2' | '2' | 1.0 |
| 1 | '1+1=?5' | '2' | 0.0 |
| 2 | '1+1=?2' | '2' | 1.0 |

**优势计算：**

```python
rewards = [[1.0, 0.0, 1.0]]
mean_r = [0.6667]
std_r = [0.4714]

advantages = (rewards - 0.6667) / 0.4714
           = [[0.7071, -1.4142, 0.7071]]
```

**含义**：
- 样本0：优势=0.7071（比平均好）→ 策略会倾向于增加生成"1+1=?2"的概率
- 样本1：优势=-1.4142（比平均差）→ 策略会倾向于降低生成"1+1=?5"的概率
- 样本2：优势=0.7071（比平均好）→ 策略会倾向于增加生成"1+1=?2"的概率

### 12.15 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      GRPO 训练数据流（1+1=? 示例）                        │
└─────────────────────────────────────────────────────────────────────────┘

输入: "1+1=?"
    │
    ▼
┌─────────────────┐
│  编码过程        │  "1+1=?" → [21,1,10,1,12,15,22,20,20,20,20,20]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  嵌入层(Embedding)│ token序列 → [1,12,16]向量矩阵
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RoPE位置编码    │ 旋转Q/K向量，引入位置信息
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  多头注意力      │ 计算QKV投影、注意力分数、加权求和
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  残差连接+FFN    │ 层归一化→注意力→残差→层归一化→FFN→残差
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LM头+logits     │ [1,12,16] → [1,12,23]词汇概率
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  生成响应(Rollout)│ 使用old_policy生成3个响应
└────────┬────────┘
         │ 输出: [[21,1,10,1,12,15,22,20,20,20,20,20,14],  # '1+1=?>'
         │        [21,1,10,1,12,15,22,20,20,20,20,20,4],   # '1+1=?4'
         │        [21,1,10,1,12,15,22,20,20,20,20,20,14]]  # '1+1=?>'
         ▼
┌─────────────────┐
│  计算old_log_probs │ [-3.4260, -3.4338, -3.4260]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  计算奖励        │ [[0., 0., 0.]]  (都没有包含"2")
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  计算优势(GRPO)  │ [[0., 0., 0.]]  (组内归一化后)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  计算ref_log_probs │ [-3.0968, -3.1153, -3.0832]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PPO梯度更新     │ loss = 0.0 (第一次训练无更新)
└─────────────────┘
```

### 12.16 关键概念总结

| 步骤 | 输入 | 输出 | 核心作用 |
|------|------|------|---------|
| 编码 | 字符串"1+1=?" | token序列 [21,1,10,1,12,15,22,...] | 将自然语言转化为模型可处理的数值序列 |
| 嵌入层 | token序列 | [B,L,d_model]向量矩阵 | 将离散token转化为连续向量表示 |
| RoPE | Q/K向量 | 旋转后的Q/K向量 | 引入位置信息，使模型感知顺序 |
| 多头注意力 | Q/K/V向量 | 注意力输出 | 让模型关注序列中不同位置的信息 |
| 残差连接 | 输入+子层输出 | 残差后输出 | 缓解梯度消失，稳定训练 |
| LM头 | 隐藏状态 | [B,L,vocab_size]logits | 预测下一个token的概率分布 |
| Rollout | prompt + old_policy | 多个响应序列 | 生成训练数据 |
| old_log_probs | 响应序列 + old_policy | log概率值 | 计算概率比值的基准 |
| 奖励计算 | 响应序列 + 标准答案 | 奖励值(0或1) | 评估响应质量 |
| 优势计算 | 奖励值 | 归一化优势 | GRPO核心，组内相对评估 |
| ref_log_probs | 响应序列 + ref策略 | log概率值 | KL散度惩罚的基准 |
| PPO更新 | 优势 + ratio + KL | 梯度更新 | 优化策略模型 |

---

## 13. 代码调试指南

### 13.1 添加调试输出

代码中已添加`verbose=True`参数，可以在以下函数中启用详细输出：

- `SimpleTokenizer.encode()`：查看编码过程
- `MiniTransformer.forward()`：查看前向传播各层输出
- `generate_response()`：查看生成过程和预测概率
- `GRPOTrainer.train_step()`：查看完整训练流程

### 13.2 运行调试模式

```python
# 修改主程序配置
class Config:
    group_size = 3    # 减少采样数量便于观察
    lr = 1e-2
    ppo_epochs = 1    # 减少epoch便于观察

train_prompts = ["1+1=?"]
train_ground_truths = ["2"]

# 启用verbose模式
metrics = trainer.train_step(train_prompts, train_ground_truths, verbose=True)
```

### 13.3 关键观察点

1. **编码过程**：确认token序列是否正确生成
2. **生成响应**：观察预测概率最高的token，确认模型是否学到正确模式
3. **奖励计算**：确认奖励函数是否正确评估响应
4. **优势计算**：确认组内归一化是否正常工作
5. **loss计算**：确认梯度是否正常更新