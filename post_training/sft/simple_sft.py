import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple
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
        return ''.join([self.inv_vocab.get(int(i), '') for i in ids 
                       if int(i) not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]])

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


# ==================== 4. SFT 训练器（标准化版） ====================
class SFTTrainer:
    """
    监督微调 (Supervised Fine-Tuning) 训练器

    标准流程：
        1. 将 (prompt, response) 拼接为完整序列：<BOS> prompt <EOS> response <EOS>
        2. 构造 labels：prompt 部分设为 -100（忽略），response 部分为目标 token
        3. 使用交叉熵损失（teacher forcing）训练模型
    """
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        # 使用 -100 作为 ignore_index，因为 labels 中 prompt 部分为 -100
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def prepare_data(self, prompts: List[str], responses: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 prompts 和 responses 编码并拼接成训练样本。
        返回:
            tokens: [B, max_len] 输入 token ids
            labels: [B, max_len] 目标 token ids (prompt 部分为 -100)
        """
        batch_tokens = []
        batch_labels = []
        max_len = 0

        for p, r in zip(prompts, responses):
            # 编码 prompt 和 response（均包含 BOS 和 EOS）
            prompt_ids = self.tokenizer.encode(p, max_len=12).squeeze(0).tolist()  # [BOS, ..., EOS]
            resp_ids = self.tokenizer.encode(r, max_len=12).squeeze(0).tolist()     # [BOS, ..., EOS]
            
            # 去掉 response 的 BOS，因为我们要用 prompt 的 BOS 作为整个序列的起始
            # 最终序列：<BOS> prompt <EOS> response <EOS>
            # 但 prompt_ids 已经包含 <BOS> 和 <EOS>，resp_ids 也包含 <BOS> 和 <EOS>
            # 为了不出现两个 BOS，我们去掉 resp_ids 的第一个 token（BOS）
            resp_tokens = resp_ids[1:]  # 去掉 BOS，保留内容 + EOS
            
            seq = prompt_ids + resp_tokens
            # labels: prompt 部分为 -100，response 部分为对应的 token（包含 EOS）
            labels = [-100] * len(prompt_ids) + resp_tokens

            batch_tokens.append(seq)
            batch_labels.append(labels)
            max_len = max(max_len, len(seq))

        # Padding
        padded_tokens = []
        padded_labels = []
        for seq, lab in zip(batch_tokens, batch_labels):
            pad_len = max_len - len(seq)
            padded_tokens.append(seq + [self.tokenizer.pad_token_id] * pad_len)
            padded_labels.append(lab + [-100] * pad_len)  # 对 PAD 也忽略

        tokens = torch.tensor(padded_tokens, dtype=torch.long)
        labels = torch.tensor(padded_labels, dtype=torch.long)
        return tokens, labels

    def train_step(self, prompts: List[str], responses: List[str], verbose: bool = False):
        """
        单步训练：对所有样本计算损失并更新参数。
        """
        if verbose:
            print("\n=== SFT 训练步骤 ===")
            print(f"  prompts: {prompts}")
            print(f"  responses: {responses}")

        tokens, labels = self.prepare_data(prompts, responses)
        tokens = tokens.to(next(self.model.parameters()).device)
        labels = labels.to(tokens.device)

        if verbose:
            print(f"  tokens shape: {tokens.shape}")
            print(f"  labels shape: {labels.shape}")

        # 前向传播
        logits = self.model(tokens)  # [B, L, vocab_size]

        # Shift: 输入 t 预测 t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        if verbose:
            print(f"  loss: {loss.item():.4f}")

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {'loss': loss.item()}


# ==================== 5. 生成响应（用于评估） ====================
@torch.no_grad()
def generate_response(
    model: MiniTransformer,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 1,
    temperature: float = 1.0,
    do_sample: bool = False,
):
    """自回归生成（用于评估）"""
    model.eval()
    input_ids = tokenizer.encode(prompt, max_len=12)
    generated = input_ids.squeeze(0).tolist()

    for _ in range(max_new_tokens):
        tokens = torch.tensor([generated], device=next(model.parameters()).device)
        logits = model(tokens)
        next_token_logits = logits[0, -1, :] / temperature
        if do_sample:
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        else:
            next_token = torch.argmax(next_token_logits).item()
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)

    return torch.tensor(generated, device=next(model.parameters()).device).unsqueeze(0)


def evaluate_model(model, tokenizer, test_prompts, test_ground_truths, max_new_tokens=1):
    """贪婪解码评估准确率"""
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


# ==================== 6. 主程序 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleTokenizer()
    print("词汇表:", tokenizer.vocab)
    vocab_size = tokenizer.vocab_size

    model = MiniTransformer(
        vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        d_model=16,
        num_heads=2,
        num_layers=2
    ).to(device)

    class Config:
        lr = 1e-2
        max_grad_norm = 1.0
    config = Config()

    trainer = SFTTrainer(model, tokenizer, config)

    # 训练数据
    train_prompts = ["1+1=?", "2+3=?", "3+4=?", "4+5=?"]
    train_responses = ["2", "5", "7", "9"]

    # 测试数据
    test_prompts = ["2+4=?", "4+3=?", "1+4=?", "3+1=?"]
    test_ground_truths = ["6", "7", "5", "4"]

    steps_num = 100
    eval_interval = 20

    for step in range(steps_num):
        metrics = trainer.train_step(train_prompts, train_responses, verbose=(step==0))
        print(f"Step {step}: Loss={metrics['loss']:.4f}")

        if (step + 1) % eval_interval == 0:
            acc = evaluate_model(model, tokenizer, test_prompts, test_ground_truths)
            print(f"  >>> Test Accuracy: {acc:.2f}")

    print("训练完成！")