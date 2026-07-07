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

    def encode(self, text: str, max_len: int = 12, verbose: bool = False) -> torch.Tensor:
        ids = [self.vocab.get(ch, self.pad_token_id) for ch in text[:max_len]]
        ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if len(ids) < max_len:
            ids += [self.pad_token_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        
        if verbose:
            print(f"\n=== 编码过程 ===\n输入文本: '{text}'")
            print(f"字符转token: {[ch for ch in text[:max_len]]} -> {ids[1:-1] if len(ids) > 2 else []}")
            print(f"添加<BOS>和<EOS>: [{self.bos_token_id}] + {ids[1:-1]} + [{self.eos_token_id}] = {ids[:len([self.bos_token_id] + [self.vocab.get(ch, self.pad_token_id) for ch in text[:max_len]] + [self.eos_token_id])]}")
            if len(ids) < max_len:
                print(f"长度不足，补充{max_len - len(ids)}个<PAD>: {ids}")
            print(f"最终token序列长度: {len(ids)}")
            print(f"token含义:")
            for i, idx in enumerate(ids):
                ch = self.inv_vocab.get(idx, '?')
                special = ""
                if idx == self.pad_token_id: special = " (PAD)"
                elif idx == self.bos_token_id: special = " (BOS)"
                elif idx == self.eos_token_id: special = " (EOS)"
                print(f"  位置{i}: {idx} -> '{ch}'{special}")
        
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

def apply_rope(q, k, seq_len, head_dim, verbose=False):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim))
    t = torch.arange(seq_len, device=q.device).float()
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(1)
    sin = emb.sin().unsqueeze(0).unsqueeze(1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    
    if verbose:
        print(f"\n=== RoPE位置编码 ===")
        print(f"  seq_len = {seq_len}, head_dim = {head_dim}")
        print(f"  inv_freq (频率倒数): {inv_freq.tolist()}")
        print(f"    公式: inv_freq[i] = 1 / 10000^(2i / head_dim)")
        print(f"  t (位置索引): {t.tolist()}")
        print(f"  freqs形状: {freqs.shape}")
        print(f"  freqs (前3个位置的频率):")
        for i in range(min(3, seq_len)):
            print(f"    位置{i}: {freqs[i].tolist()}")
        print(f"  emb形状: {emb.shape}")
        print(f"  emb含义: 每个位置的旋转角度，前半和后半相同")
        print(f"  cos形状: {cos.shape}, sin形状: {sin.shape}")
        print(f"  cos[0,0,0] (位置0的cos值): {cos[0,0,0].tolist()}")
        print(f"  sin[0,0,0] (位置0的sin值): {sin[0,0,0].tolist()}")
        print(f"  cos[0,0,1] (位置1的cos值): {cos[0,0,1].tolist()}")
        print(f"  sin[0,0,1] (位置1的sin值): {sin[0,0,1].tolist()}")
        print(f"\n  旋转前后对比 (以第1个头为例):")
        print(f"  q[0,0,0] (旋转前位置0): {q[0,0,0].tolist()[:4]}...")
        print(f"  q_rot[0,0,0] (旋转后位置0): {q_rot[0,0,0].tolist()[:4]}...")
        print(f"  q[0,0,1] (旋转前位置1): {q[0,0,1].tolist()[:4]}...")
        print(f"  q_rot[0,0,1] (旋转后位置1): {q_rot[0,0,1].tolist()[:4]}...")
        print(f"\n  旋转公式: q_rot = q * cos + rotate_half(q) * sin")
        print(f"            rotate_half([x1, x2]) = [-x2, x1]")
    
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

    def forward(self, x, mask=None, verbose=False):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        if verbose:
            print(f"\n=== 多头注意力 ===")
            print(f"  B={B}, L={L}, d_model={self.d_model}, num_heads={self.num_heads}, head_dim={self.head_dim}")
            print(f"  Q投影后形状: {q.shape}")
            print(f"  K投影后形状: {k.shape}")
            print(f"  V投影后形状: {v.shape}")
            print(f"  Q[0,0,0] (第1个头，位置0): {q[0,0,0].tolist()}")
            print(f"  K[0,0,0] (第1个头，位置0): {k[0,0,0].tolist()}")
            print(f"  V[0,0,0] (第1个头，位置0): {v[0,0,0].tolist()}")
        
        q, k = apply_rope(q, k, L, self.head_dim, verbose=verbose)
        
        if verbose:
            print(f"\n  应用RoPE后:")
            print(f"  q_rot[0,0,0] (旋转后): {q[0,0,0].tolist()}")
            print(f"  k_rot[0,0,0] (旋转后): {k[0,0,0].tolist()}")
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if verbose:
            print(f"\n  注意力分数:")
            print(f"  attn_scores形状: {attn_scores.shape}")
            print(f"  attn_scores[0,0] (第1个头的注意力分数矩阵):")
            for i in range(min(5, L)):
                print(f"    位置{i} -> {attn_scores[0,0,i].tolist()[:5]}...")
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        if verbose:
            print(f"\n  注意力权重(softmax后):")
            print(f"  attn_weights[0,0] (第1个头):")
            for i in range(min(5, L)):
                row_sum = attn_weights[0,0,i].sum().item()
                print(f"    位置{i}注意力: {attn_weights[0,0,i].tolist()[:5]}... (和={row_sum:.4f})")
        
        attn_out = torch.matmul(attn_weights, v)
        
        if verbose:
            print(f"\n  加权求和后:")
            print(f"  attn_out形状: {attn_out.shape}")
            print(f"  attn_out[0,0,0]: {attn_out[0,0,0].tolist()}")
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        
        if verbose:
            print(f"  拼接多头后形状: {attn_out.shape}")
        
        out = self.o_proj(attn_out)
        
        if verbose:
            print(f"  输出投影后形状: {out.shape}")
            print(f"  out[0,0]: {out[0,0].tolist()}")
        
        return out


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

    def forward(self, x, mask=None, verbose=False):
        if verbose:
            print(f"\n=== Transformer Block ===")
            print(f"  输入形状: {x.shape}")
            print(f"  x[0,0] (位置0输入): {x[0,0].tolist()[:5]}...")
        
        x_norm1 = self.ln1(x)
        
        if verbose:
            print(f"\n  ln1(x)[0,0] (层归一化后): {x_norm1[0,0].tolist()[:5]}...")
        
        attn_out = self.attn(x_norm1, mask, verbose=verbose)
        
        if verbose:
            print(f"\n  attn_out[0,0] (注意力输出): {attn_out[0,0].tolist()[:5]}...")
        
        x = x + attn_out
        
        if verbose:
            print(f"\n  残差连接后 x[0,0]: {x[0,0].tolist()[:5]}...")
            print(f"  残差公式: x = x + attn(ln1(x))")
        
        x_norm2 = self.ln2(x)
        
        if verbose:
            print(f"\n  ln2(x)[0,0] (层归一化后): {x_norm2[0,0].tolist()[:5]}...")
        
        ffn_out = self.ffn(x_norm2)
        
        if verbose:
            print(f"\n  ffn_out[0,0] (前馈网络输出): {ffn_out[0,0].tolist()[:5]}...")
        
        x = x + ffn_out
        
        if verbose:
            print(f"\n  残差连接后 x[0,0]: {x[0,0].tolist()[:5]}...")
            print(f"  残差公式: x = x + ffn(ln2(x))")
        
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, pad_token_id, d_model=16, num_heads=2, num_layers=2):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.tokenizer_inv = None

    def forward(self, tokens, mask=None, verbose=False):
        if verbose:
            print(f"\n=== MiniTransformer 前向传播 ===\n输入tokens形状: {tokens.shape}, 值: {tokens[0].tolist()}")
        
        x = self.embed(tokens)
        
        if verbose:
            print(f"\n--- 嵌入层 ---")
            print(f"嵌入层输出形状: {x.shape}")
            print(f"嵌入层输出示例(前3个token的完整向量):")
            for i in range(min(3, x.size(1))):
                ch = self.tokenizer_inv.get(tokens[0][i].item(), '?') if self.tokenizer_inv else '?'
                print(f"  token[{i}] = {tokens[0][i].item()} ('{ch}'): {x[0, i].tolist()}")
        
        for layer_idx, layer in enumerate(self.layers):
            if verbose:
                print(f"\n--- Transformer Block {layer_idx} ---")
            x = layer(x, mask, verbose=verbose)
            if verbose:
                print(f"\nTransformer Block {layer_idx} 输出形状: {x.shape}")
                print(f"  x[0,0] (位置0输出): {x[0,0].tolist()[:5]}...")
        
        if verbose:
            print(f"\n--- 最终层归一化 ---")
        x = self.ln_final(x)
        
        if verbose:
            print(f"ln_final输出形状: {x.shape}")
            print(f"x[0,0]: {x[0,0].tolist()[:5]}...")
        
        if verbose:
            print(f"\n--- 语言模型头(lm_head) ---")
        logits = self.lm_head(x)
        
        if verbose:
            print(f"语言模型头输出形状: {logits.shape}")
            print(f"logits含义说明:")
            print(f"  - 第一维度: batch_size = {logits.size(0)}")
            print(f"  - 第二维度: 序列长度 = {logits.size(1)}")
            print(f"  - 第三维度: vocab_size = {logits.size(2)}")
            print(f"\n最后一个位置的logits(全部23个词汇):")
            last_logits = logits[0, -1, :]
            for i in range(logits.size(2)):
                ch = self.tokenizer_inv.get(i, '?') if self.tokenizer_inv else '?'
                special = ""
                if i == self.pad_token_id: special = " (PAD)"
                elif i == 21: special = " (BOS)"
                elif i == 22: special = " (EOS)"
                print(f"  词汇{i} -> '{ch}'{special}: {last_logits[i].item():.4f}")
        
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
        
        # 设置tokenizer_inv用于打印
        model.tokenizer_inv = tokenizer.inv_vocab

    def prepare_data(self, prompts: List[str], responses: List[str], verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 prompts 和 responses 编码并拼接成训练样本。
        返回:
            tokens: [B, max_len] 输入 token ids
            labels: [B, max_len] 目标 token ids (prompt 部分为 -100)
        """
        batch_tokens = []
        batch_labels = []
        max_len = 0

        for idx, (p, r) in enumerate(zip(prompts, responses)):
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

            if verbose and idx == 0:
                print(f"\n--- 数据准备 (prompt='{p}', response='{r}') ---")
                print(f"  prompt编码: {prompt_ids} -> '{self.tokenizer.decode(torch.tensor(prompt_ids))}'")
                print(f"  response编码: {resp_ids} -> '{self.tokenizer.decode(torch.tensor(resp_ids))}'")
                print(f"  response去掉BOS: {resp_tokens}")
                print(f"  拼接后序列: {seq}")
                print(f"  labels: {labels}")
                print(f"  labels含义: [-100]*{len(prompt_ids)} (prompt忽略) + {resp_tokens} (response目标)")

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
        
        if verbose:
            print(f"\n  所有样本padding后:")
            print(f"  tokens形状: {tokens.shape}")
            print(f"  labels形状: {labels.shape}")
            print(f"  tokens[0] (第1个样本): {tokens[0].tolist()}")
            print(f"  labels[0] (第1个样本): {labels[0].tolist()}")
        
        return tokens, labels

    def train_step(self, prompts: List[str], responses: List[str], verbose: bool = False):
        """
        单步训练：对所有样本计算损失并更新参数。
        """
        if verbose:
            print("\n============================================================\n=== SFT 训练步骤 ===")
            print(f"  prompts: {prompts}")
            print(f"  responses: {responses}")

        tokens, labels = self.prepare_data(prompts, responses, verbose=verbose)
        tokens = tokens.to(next(self.model.parameters()).device)
        labels = labels.to(tokens.device)

        if verbose:
            print(f"\n  tokens shape: {tokens.shape}")
            print(f"  labels shape: {labels.shape}")

        # 前向传播
        logits = self.model(tokens, verbose=verbose)  # [B, L, vocab_size]

        # Shift: 输入 t 预测 t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        if verbose:
            print(f"\n--- Shift操作 ---")
            print(f"  logits形状: {logits.shape}")
            print(f"  shift_logits形状: {shift_logits.shape} (去掉最后一个位置)")
            print(f"  shift_labels形状: {shift_labels.shape} (去掉第一个位置)")
            print(f"  含义: 用位置0~L-2的输出预测位置1~L-1的token")

        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        if verbose:
            print(f"\n--- 损失计算 ---")
            print(f"  loss: {loss.item():.4f}")
            print(f"  损失公式: CrossEntropy(logits, labels)，其中labels中-100的位置被忽略")

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