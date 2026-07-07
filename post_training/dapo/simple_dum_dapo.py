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
        ids = [self.vocab.get(ch, self.pad_token_id) for ch in text[:max_len]]
        ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if len(ids) < max_len:
            ids += [self.pad_token_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

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
    cos = emb.cos().unsqueeze(0).unsqueeze(1)
    sin = emb.sin().unsqueeze(0).unsqueeze(1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

# ==================== 3. Transformer 层 ====================
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
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

# ==================== 修改 MiniTransformer.get_log_probs ====================
    def get_log_probs(self, tokens: torch.Tensor):
        """
        返回每个 token 的 log 概率（形状 [B, L-1]），用于 token-level 损失。
        忽略 PAD token（对应位置设为 0，但梯度不参与）。
        """
        B, L = tokens.shape
        if L <= 1:
            return torch.zeros(B, L-1, device=tokens.device)
        logits = self.forward(tokens)                     # [B, L, V]
        shift_logits = logits[:, :-1, :]                  # [B, L-1, V]
        shift_tokens = tokens[:, 1:]                      # [B, L-1]
        log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, L-1, V]
        # 提取每个位置预测正确 token 的 log 概率
        token_log_probs = log_probs.gather(dim=-1, index=shift_tokens.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        return token_log_probs  # 不取平均，直接返回每个 token 的 log 概率

# ==================== 4. 生成回答（Rollout） ====================
@torch.no_grad()
def generate_response(
    model: MiniTransformer,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 1,
    temperature: float = 1.0,
    do_sample: bool = True,
    top_p: float = 0.9,
):
    model.eval()
    input_ids = tokenizer.encode(prompt, max_len=12)
    generated = input_ids.squeeze(0).tolist()

    for _ in range(max_new_tokens):
        tokens = torch.tensor([generated], device=next(model.parameters()).device)
        logits = model(tokens)
        next_token_logits = logits[0, -1, :] / temperature

        if do_sample:
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits / temperature, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                # 保留累积概率 <= top_p 的部分
                mask = cumsum_probs <= top_p
                # 如果第一个 token 的概率已经超过 top_p，则至少保留第一个
                if not mask.any():
                    mask[0] = True
                sorted_probs[~mask] = 0.0
                # 重新归一化，并添加小 epsilon 防止除零
                prob_sum = sorted_probs.sum()
                if prob_sum == 0:
                    sorted_probs = torch.ones_like(sorted_probs) / len(sorted_probs)  # 退化为均匀分布
                else:
                    sorted_probs = sorted_probs / prob_sum
                # 采样并映射回原始索引
                sampled_sorted_idx = torch.multinomial(sorted_probs, 1).item()
                next_token = sorted_indices[sampled_sorted_idx].item()
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
        else:
            next_token = torch.argmax(next_token_logits).item()

        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)

    return torch.tensor(generated, device=next(model.parameters()).device).unsqueeze(0)

# ==================== 5. 奖励函数 ====================
def math_reward_fn(
    response_tokens: torch.Tensor,
    tokenizer: SimpleTokenizer,
    ground_truths: List[str],
    group_size: int,
    verbose: bool = False
) -> torch.Tensor:
    total_samples = response_tokens.shape[0]
    expected_total = len(ground_truths) * group_size
    if total_samples != expected_total:
        raise ValueError(f"响应总数 ({total_samples}) 与期望 ({expected_total}) 不匹配。")

    rewards = []
    for i, tokens in enumerate(response_tokens):
        question_idx = i // group_size
        correct_answer = ground_truths[question_idx]
        text = tokenizer.decode(tokens)
        reward = 1.0 if correct_answer in text else 0.0
        rewards.append(reward)
        if verbose:
            print(f"样本 {i:2d} (问题 {question_idx}, 期望 '{correct_answer}', 奖励 '{reward}'): {text}")
    return torch.tensor(rewards, device=response_tokens.device)

# ==================== 6. DAPO 训练器（继承GRPO，增强动态采样与裁剪） ====================
class GRPOTrainer:
    def __init__(self, policy, ref, tokenizer, config):
        self.policy = policy
        self.ref = ref
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)
        for p in ref.parameters():
            p.requires_grad = False
        self.old_policy = copy.deepcopy(policy)
        for p in self.old_policy.parameters():
            p.requires_grad = False

    def train_step(self, prompts, ground_truths, max_retries=100):
        B = len(prompts)
        G = self.config.group_size
        device = next(self.policy.parameters()).device

        self.old_policy.load_state_dict(self.policy.state_dict())

        # ---------- DAPO 创新点 2: Dynamic Sampling ----------
        # 论文要求：过滤掉全0或全1的组（即组内奖励标准差为0），
        # 持续采样直到所有组都有正有负，确保每个组都能提供有效梯度。
        try_count = 0
        while try_count < max_retries:
            try_count += 1
            all_responses = []
            for prompt in prompts:
                for _ in range(G):
                    resp = generate_response(self.old_policy, self.tokenizer, prompt)
                    all_responses.append(resp)

            def pad_responses(responses_list):
                max_len = max(r.size(1) for r in responses_list)
                padded = []
                for r in responses_list:
                    pad_len = max_len - r.size(1)
                    if pad_len > 0:
                        pad = torch.full((1, pad_len), self.tokenizer.pad_token_id, device=device)
                        r_pad = torch.cat([r, pad], dim=1)
                    else:
                        r_pad = r
                    padded.append(r_pad)
                return torch.cat(padded, dim=0)

            responses = pad_responses(all_responses)
            rewards = math_reward_fn(responses, self.tokenizer, ground_truths, G)
            rewards = rewards.view(B, G)

            # 检查组内标准差是否为0（全0或全1）
            std_r = rewards.std(dim=1)
            valid_mask = std_r > 1e-8
            if valid_mask.all():
                break  # 全部有效，退出循环

            # 对无效组重新采样（仅替换这些组）
            invalid_indices = (~valid_mask).nonzero(as_tuple=True)[0].tolist()
            for idx in invalid_indices:
                prompt = prompts[idx]
                for g in range(G):
                    new_resp = generate_response(self.old_policy, self.tokenizer, prompt)
                    all_responses[idx * G + g] = new_resp
            # 继续循环，重新检查

        # ---------- 计算 old_log_probs (token-level) ----------
        # 注意：get_log_probs 返回每个 token 的 log 概率，形状 [B*G, L-1]
        with torch.no_grad():
            old_log_probs = self.old_policy.get_log_probs(responses)  # [B*G, L-1]

        # ---------- 计算优势 (组内标准化) ----------
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True)
        advantages = (rewards - mean_r) / (std_r + 1e-8)  # [B, G]
        # 将标量优势广播到每个 token（所有 token 共享同一组优势）
        adv_flat = advantages.view(-1)  # [B*G]
        adv_expanded = adv_flat.unsqueeze(1).expand(-1, old_log_probs.size(1))  # [B*G, L-1]

        # ---------- 参考策略 log probs (token-level) ----------
        # 在DAPO里没有用，因为DAPO不计算KL Loss
        # with torch.no_grad():
        #     log_probs_ref = self.ref.get_log_probs(responses)  # [B*G, L-1]

        # ---------- DAPO 创新点 1: Clip-Higher (非对称裁剪) ----------
        # 论文设置 ε_low=0.2, ε_high=0.28，提高上界以鼓励探索
        EPS_LOW = 0.2
        EPS_HIGH = 0.28

        for _ in range(self.config.ppo_epochs):
            log_probs_theta = self.policy.get_log_probs(responses)  # [B*G, L-1]
            ratio = torch.exp(log_probs_theta - old_log_probs)      # [B*G, L-1]
            clipped_ratio = torch.clamp(ratio, 1.0 - EPS_LOW, 1.0 + EPS_HIGH)
            surr1 = ratio * adv_expanded
            surr2 = clipped_ratio * adv_expanded
            surrogate_loss = torch.min(surr1, surr2)  # [B*G, L-1]

            # ---------- DAPO 创新点 3: Token-Level Policy Gradient Loss ----------
            # 对所有有效 token 取平均，每个 token 贡献相同，避免长样本被稀释
            shift_tokens = responses[:, 1:]
            token_mask = (shift_tokens != self.tokenizer.pad_token_id).float()  # [B*G, L-1]
            loss = - (surrogate_loss * token_mask).sum() / (token_mask.sum() + 1e-8)

            # ---------- DAPO 创新点 4: 移除 KL 散度 ----------
            # 论文 Section 2.3 明确排除 KL 惩罚，长CoT场景下约束不必要

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        return {
            'loss': loss.item(),
            'mean_ratio': ratio.mean().item(),
        }
        
# ==================== 7. 评估函数 ====================
def evaluate_model(model, tokenizer, test_prompts, test_ground_truths, max_new_tokens=1):
    model.eval()
    correct = 0
    total = len(test_prompts)
    with torch.no_grad():
        for prompt, gt in zip(test_prompts, test_ground_truths):
            resp_tokens = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0
            )
            text = tokenizer.decode(resp_tokens.squeeze(0))
            if gt in text:
                correct += 1
    return correct / total

# ==================== 8. 主程序 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleTokenizer()
    print(tokenizer.vocab)
    vocab_size = tokenizer.vocab_size

    policy = MiniTransformer(vocab_size, pad_token_id=tokenizer.pad_token_id, d_model=16, num_heads=2, num_layers=2).to(device)
    ref = MiniTransformer(vocab_size, pad_token_id=tokenizer.pad_token_id, d_model=16, num_heads=2, num_layers=2).to(device)
    ref.load_state_dict(policy.state_dict())

    class Config:
        group_size = 8          # 每组的rollout数量
        lr = 1e-3              # 学习率
        ppo_epochs = 2          # 同一批数据的更新轮数
    config = Config()

    trainer = GRPOTrainer(policy, ref, tokenizer, config)

    # 训练数据（4个加法题）
    train_prompts = ["1+1=?", "2+3=?", "3+4=?", "4+5=?"]
    train_ground_truths = ["2", "5", "7", "9"]

    # 测试数据（用于评估泛化）
    test_prompts = ["2+4=?", "4+3=?", "1+4=?", "3+1=?"]
    test_ground_truths = ["6", "7", "5","4"]

    steps_num = 100
    eval_interval = 100
    to_test = False   # 改为True开启评估

    for step in range(steps_num):
        metrics = trainer.train_step(prompts=train_prompts, ground_truths=train_ground_truths, max_retries=100)
        # 修改打印语句，移除 KL
        print(f"Step {step}: Loss={metrics['loss']:.4f}, Ratio={metrics['mean_ratio']:.4f}")

        if to_test and (step + 1) % eval_interval == 0:
            acc = evaluate_model(policy, tokenizer, test_prompts, test_ground_truths)
            print(f"  >>> Test Accuracy: {acc:.2f}")

    print("训练完成！")