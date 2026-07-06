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

    def get_log_probs(self, tokens: torch.Tensor):
        B, L = tokens.shape
        if L <= 1:
            return torch.zeros(B, device=tokens.device)
        logits = self.forward(tokens)
        shift_logits = logits[:, :-1, :]
        shift_tokens = tokens[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shift_tokens.unsqueeze(-1)).squeeze(-1)
        mask = (shift_tokens != self.pad_token_id).float()
        log_prob_sum = (token_log_probs * mask).sum(dim=-1)
        valid_count = mask.sum(dim=-1).clamp(min=1)
        avg_log_prob = log_prob_sum / valid_count
        return avg_log_prob

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

    def train_step(self, prompts, ground_truths, max_retries=5):
        B = len(prompts)
        G = self.config.group_size
        device = next(self.policy.parameters()).device

        # 复制当前策略作为旧策略（用于采样）
        self.old_policy.load_state_dict(self.policy.state_dict())

        # ---------- 初始化采样 ----------
        all_responses = []
        for prompt in prompts:
            for _ in range(G):
                resp = generate_response(self.old_policy, self.tokenizer, prompt)
                all_responses.append(resp)

        # 填充到相同长度
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

        # 计算奖励
        rewards = math_reward_fn(responses, self.tokenizer, ground_truths, G)
        rewards = rewards.view(B, G)

        # ---------- 只对全0组进行重试，最多 max_retries 次 ----------
        retry_count = 0
        while retry_count < max_retries:
            # 找出全0的组（每行所有元素都为0）
            zero_mask = (rewards == 0).all(dim=1)  # [B]
            if not zero_mask.any():
                break  # 没有全0组，退出

            # 对每个全0组重新采样
            invalid_indices = zero_mask.nonzero(as_tuple=True)[0].tolist()
            for idx in invalid_indices:
                prompt = prompts[idx]
                for g in range(G):
                    new_resp = generate_response(self.old_policy, self.tokenizer, prompt)
                    all_responses[idx * G + g] = new_resp

            # 重新填充并计算奖励
            responses = pad_responses(all_responses)
            rewards = math_reward_fn(responses, self.tokenizer, ground_truths, G)
            rewards = rewards.view(B, G)

            # print("rewards:\n",rewards)
            retry_count += 1
            # print(f"重试第 {retry_count} 次，全0组索引: {invalid_indices}")

        # 如果重试后仍然存在全0组，保留最后一次的结果（不再重试）
        # 此时优势计算中全0组的标准差为0，优势为0，不会产生梯度，但至少不会死循环

        # ---------- 后续计算（与原代码一致） ----------
        with torch.no_grad():
            old_log_probs = self.old_policy.get_log_probs(responses)

        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True)
        advantages = (rewards - mean_r) / (std_r + 1e-8)
        advantages_flat = advantages.view(-1)

        with torch.no_grad():
            log_probs_ref = self.ref.get_log_probs(responses)

        EPS_LOW = 0.2
        EPS_HIGH = 0.28
        BETA = 0.0001

        for _ in range(self.config.ppo_epochs):
            log_probs_theta = self.policy.get_log_probs(responses)
            ratio = torch.exp(log_probs_theta - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - EPS_LOW, 1.0 + EPS_HIGH)
            surr1 = ratio * advantages_flat
            surr2 = clipped_ratio * advantages_flat
            surrogate_loss = torch.min(surr1, surr2)

            log_diff = log_probs_ref - log_probs_theta
            kl_div = torch.exp(log_diff) - log_diff - 1
            kl_loss = BETA * kl_div.mean()

            loss = -surrogate_loss.mean() + kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        return {
            'loss': loss.item(),
            'kl_loss': kl_loss.item(),
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

    steps_num = 500
    eval_interval = 100
    to_test = False   # 改为True开启评估

    for step in range(steps_num):
        metrics = trainer.train_step(prompts=train_prompts, ground_truths=train_ground_truths,max_retries=100)
        print(f"Step {step}: Loss={metrics['loss']:.4f}, KL={metrics['kl_loss']:.4f}, Ratio={metrics['mean_ratio']:.4f}")

        if to_test and (step + 1) % eval_interval == 0:
            acc = evaluate_model(policy, tokenizer, test_prompts, test_ground_truths)
            print(f"  >>> Test Accuracy: {acc:.2f}")

    print("训练完成！")