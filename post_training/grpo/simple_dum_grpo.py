import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
import copy
# ==================== 1. 分词器（字符级） ====================
class SimpleTokenizer:
    def __init__(self):
        chars = "0123456789+-=<>?() \n"
        self.vocab = {ch: idx for idx, ch in enumerate(chars)}
        self.vocab['<PAD>'] = len(self.vocab)
        self.vocab['<BOS>'] = len(self.vocab)
        self.vocab['<EOS>'] = len(self.vocab)
        self.inv_vocab = {idx: ch for ch, idx in self.vocab.items()}
        self.pad_token_id = self.vocab['<PAD>']
        self.bos_token_id = self.vocab['<BOS>']
        self.eos_token_id = self.vocab['<EOS>']

    def encode(self, text: str, max_len: int = 12) -> torch.Tensor:
        # 根据词汇表vocab获取token编码
        ids = [self.vocab.get(ch, self.pad_token_id) for ch in text[:max_len]]
        # 为token序列添加起始token bos 和终止token eos
        ids = [self.bos_token_id] + ids + [self.eos_token_id]

        # 如果达不到max_len，则补充pad token
        # 如 对于 1+1=? 的token序列对应如下：
        # <BOS>, 1, +, 1, =, ?, <EOS>, <PAD>, <PAD>, <PAD>, <PAD>, <PAD>
        # 21, 1, 10, 1, 12, 15, 22, 20, 20, 20, 20, 20
        if len(ids) < max_len:
            ids += [self.pad_token_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]

        # print("token_ids: ",ids)
        # unsqueeze(0) 的作用非常直接：在索引 0 的位置插入一个大小为 1 的新维度。
        # 操作前：torch.tensor(ids) 生成的张量形状是 [L]（比如 [12]），它仅仅是一个普通的序列（向量）。
        # 操作后：.unsqueeze(0) 后，形状变成 [1, L]（比如 [1, 12]），它变成了一个“只有 1 行”的矩阵。
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # [1, L]

    def decode(self, ids: torch.Tensor) -> str:
        return ''.join([self.inv_vocab.get(int(i), '') for i in ids if int(i) not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]])

    @property
    def vocab_size(self):
        return len(self.vocab)

# ==================== 2. RoPE 旋转位置编码 ====================
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, seq_len, head_dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim))
    t = torch.arange(seq_len, device=q.device).float()
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(1)  # [1, 1, L, head_dim]
    sin = emb.sin().unsqueeze(0).unsqueeze(1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

# ==================== 3. Transformer 层（含 RoPE 多头注意力） ====================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, L, self.head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(attn_out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=32):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, pad_token_id, d_model=16, num_heads=2, num_layers=2):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, mask=None):
        x = self.embed(tokens)  # [B, L, d_model]
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

    def get_log_probs(self, tokens: torch.Tensor):
        """
        计算整个序列的自回归平均 log 概率。
        忽略 PAD token，只对有效 token 计算。
        """
        B, L = tokens.shape
        if L <= 1:
            return torch.zeros(B, device=tokens.device)

        # 前向传播得到 logits
        logits = self.forward(tokens)  # [B, L, V]
        
        # 自回归：输入 token[:L-1] 预测 token[1:]
        shift_logits = logits[:, :-1, :]  # [B, L-1, V]
        shift_tokens = tokens[:, 1:]      # [B, L-1]

        # 计算每个位置预测正确 token 的 log 概率
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shift_tokens.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

        # 忽略 PAD token (shift_tokens 中 PAD 不参与平均)
        mask = (shift_tokens != self.pad_token_id).float()
        log_prob_sum = (token_log_probs * mask).sum(dim=-1)
        valid_count = mask.sum(dim=-1).clamp(min=1)
        avg_log_prob = log_prob_sum / valid_count   # [B]
        return avg_log_prob
# ==================== 4. 生成回答（Rollout） ====================
@torch.no_grad()  # 装饰器：禁用梯度计算，节省显存并加速推理（因为生成过程无需反向传播）
def generate_response(model: MiniTransformer, tokenizer: SimpleTokenizer, prompt: str, max_new_tokens=2):
    """
    使用给定的 Transformer 模型自回归地生成对 prompt 的回答。
    
    参数:
        model: 当前策略模型（或任何 MiniTransformer 实例）
        tokenizer: 简易字符级分词器
        prompt: 输入的问题文本（如 "1+1=?"）
        max_new_tokens: 最多生成的新 token 数量（不含 prompt）

    返回:
        torch.Tensor: 形状为 [1, L_gen]，其中 L_gen 是最终生成的完整序列长度（prompt + 生成的回答）
    """
    # 1. 将模型切换为评估模式（影响 Dropout/BatchNorm 等层的行为，此处没有这些层，但习惯性设置）
    model.eval()

    # 2. 编码 prompt 文本
    #    tokenizer.encode 返回形状 [1, L] 的张量（因为 encode 内部加了 unsqueeze(0)）
    input_ids = tokenizer.encode(prompt, max_len=12)  # 例如: tensor([[BOS, 1, +, 1, =, ?, EOS, PAD, ...]])

    # 3. 去掉最外层的 batch 维度，得到一维列表，便于后续逐个添加 token
    #    squeeze(0) 将 [1, L] -> [L]；tolist() 将张量转为 Python 列表
    # print("input_ids:", input_ids)
    generated = input_ids.squeeze(0).tolist()  # 例如: [BOS, 1, +, 1, =, ?, EOS, PAD, ...]

    # 4. 自回归生成循环（最多生成 max_new_tokens 个新 token）
    for _ in range(max_new_tokens):
        # 4.1 截取最近的一小段上下文（防止序列过长导致计算量爆炸）
        #     这里设定最大上下文长度为 16（包含 prompt 和已生成的 token）
        max_len = 16
        if len(generated) > max_len:
            context = generated[-max_len:]   # 只保留最后 max_len 个 token
        else:
            context = generated              # 序列还不长，保留全部

        # 4.2 将上下文列表重新包装成模型需要的输入格式
        #     [context] 是一个嵌套列表，例如 [[BOS, 1, +, 1, =, ?, EOS]]
        #     torch.tensor 将其转为形状 [1, len(context)] 的张量（即 batch_size=1）
        tokens = torch.tensor([context], device=next(model.parameters()).device)

        # 4.3 模型前向传播，得到 logits（每个位置的原始分数）
        logits = model(tokens)  # 形状 [1, 当前序列长度, 词汇表大小]

        # 4.4 取出最后一个位置的 logits（即预测下一个 token 的分数）
        next_token_logits = logits[0, -1, :]  # 形状 [词汇表大小]

        # 4.5 使用 argmax 选择概率最大的 token（确定性采样，非随机）
        # next_token = torch.argmax(next_token_logits).item()  # 得到整数 token ID
        # 4.5 将生成策略从 argmax 改为 采样（Sampling）（引入随机性）
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        # 4.6 如果生成了结束符，停止生成
        if next_token == tokenizer.eos_token_id:
            break

        # 4.7 将新 token 追加到生成序列末尾
        generated.append(next_token)
    # print("generate_tokens:",generated)
    # 5. 将最终的生成序列（列表）转换回张量，并恢复 batch 维度
    #    返回形状 [1, L_gen]，其中 L_gen 是 prompt + 所有生成 token 的总长度
    return torch.tensor(generated, device=next(model.parameters()).device).unsqueeze(0)

# ==================== 5. 奖励函数 ====================
def math_reward_fn(
    response_tokens: torch.Tensor,
    tokenizer: SimpleTokenizer,
    ground_truths: List[str],
    group_size: int,
    verbose: bool = True
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
        verbose: bool，是否打印解码文本（便于调试，生产环境可设为 False）

    返回:
        torch.Tensor: 形状 [total_samples]，每个样本的奖励值（0.0 或 1.0）
    """
    # 验证输入维度是否正确
    total_samples = response_tokens.shape[0]
    expected_total = len(ground_truths) * group_size
    if total_samples != expected_total:
        raise ValueError(
            f"响应总数 ({total_samples}) 与期望 ({expected_total}) 不匹配。"
            f"请确保 len(ground_truths) * group_size == response_tokens.shape[0]。"
        )

    rewards = []
    # 逐条处理响应
    for i, tokens in enumerate(response_tokens):
        # ---- 1. 确定当前样本对应的问题索引 ----
        # 因为前 group_size 个是问题0，接下来的 group_size 个是问题1，依此类推
        question_idx = i // group_size

        # ---- 2. 获取该问题的标准答案 ----
        correct_answer = ground_truths[question_idx]

        # ---- 3. 将 token 序列解码为文本（自动忽略 <PAD>、<BOS>、<EOS>） ----
        text = tokenizer.decode(tokens)


        # ---- 5. 规则验证：正确答案是否出现在文本中（简单子串匹配） ----
        reward = 0
        if correct_answer in text:
            reward = 1.0   # 答对，正奖励
        else:
            reward = 0.0   # 答错，零奖励（不扣分）
        rewards.append(reward)
        # ---- 4. 调试输出（可选） ----
        if verbose:
            print(f"样本 {i:2d} (问题 {question_idx}, 期望 '{correct_answer}', 奖励 '{reward}'): {text}")

    # 返回奖励张量，设备与输入一致
    return torch.tensor(rewards, device=response_tokens.device)

# ==================== 6. GRPO 训练器 ====================
class GRPOTrainer:
    def __init__(self, policy, ref, tokenizer, config):
        self.policy = policy
        self.ref = ref
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)
        for p in ref.parameters():
            p.requires_grad = False

        # 新增：创建并冻结 old_policy
        self.old_policy = copy.deepcopy(policy)
        for p in self.old_policy.parameters():
            p.requires_grad = False

    def train_step(self, prompts, ground_truths):
        B = len(prompts)
        G = self.config.group_size
        device = next(self.policy.parameters()).device

        # <<< 1. 将当前策略复制给旧策略（作为采样策略）
        self.old_policy.load_state_dict(self.policy.state_dict())

        # <<< 2. Rollout：使用 old_policy 采样
        all_responses = []
        for prompt in prompts:
            for _ in range(G):
                resp = generate_response(self.old_policy, self.tokenizer, prompt)  # 使用 old_policy
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
        responses = torch.cat(padded_responses, dim=0)  # [B*G, max_len]

        # <<< 3. 计算 old_log_probs（来自 old_policy，无梯度）
        with torch.no_grad():
            old_log_probs = self.old_policy.get_log_probs(responses)  # 使用 old_policy

        # 4. 奖励
        rewards = math_reward_fn(responses, self.tokenizer, ground_truths, G)
        rewards = rewards.view(B, G)
        print("rewards:\n",rewards)
        # 5. 优势
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True)
        advantages = (rewards - mean_r) / (std_r + 1e-8)
        advantages_flat = advantages.view(-1)

        # 6. 参考策略 log probs（只需计算一次，ref 冻结）
        with torch.no_grad():
            log_probs_ref = self.ref.get_log_probs(responses)  # [B*G]

        # <<< 7. 对同一批数据进行多次梯度更新（PPO epochs）
        EPS = 0.2
        BETA = 0.001
        for _ in range(self.config.ppo_epochs):
            # 每次更新后重新计算当前策略的 log 概率
            log_probs_theta = self.policy.get_log_probs(responses)  # 有梯度

            ratio = torch.exp(log_probs_theta - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS)
            surr1 = ratio * advantages_flat
            surr2 = clipped_ratio * advantages_flat
            surrogate_loss = torch.min(surr1, surr2)

            log_diff = log_probs_ref - log_probs_theta
            kl_div = torch.exp(log_diff) - log_diff - 1
            kl_loss = BETA * kl_div.mean()

            loss = -surrogate_loss.mean() + kl_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        # 返回监控指标（取最后一次的 ratio 和 loss）
        return {
            'loss': loss.item(),
            'kl_loss': kl_loss.item(),
            'mean_ratio': ratio.mean().item(),
        }
    
# ==================== 7. 主程序 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleTokenizer()
    print(tokenizer.vocab)
    vocab_size = tokenizer.vocab_size

    policy = MiniTransformer(vocab_size,pad_token_id=tokenizer.pad_token_id, d_model=16, num_heads=2, num_layers=2).to(device)
    ref = MiniTransformer(vocab_size,pad_token_id=tokenizer.pad_token_id, d_model=16, num_heads=2, num_layers=2).to(device)
    ref.load_state_dict(policy.state_dict())

    class Config:
        group_size = 8
        lr = 1e-3
        ppo_epochs = 2   # 对同一批数据重复更新的次数
    config = Config()

    trainer = GRPOTrainer(policy, ref, tokenizer, config)

    prompts = ["1+1=?", "2+3=?"]
    ground_truths = ["2", "5"]

    for step in range(20):
        metrics = trainer.train_step(prompts, ground_truths)
        print(f"Step {step}: Loss={metrics['loss']:.4f}, KL={metrics['kl_loss']:.4f}, Ratio={metrics['mean_ratio']:.4f}")

    print("训练完成！")