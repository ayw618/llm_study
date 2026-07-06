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

    def encode(self, text: str, max_len: int = 12, verbose: bool = False) -> torch.Tensor:
        if verbose:
            print(f"\n=== 编码过程 ===\n输入文本: '{text}'")
        # 根据词汇表vocab获取token编码
        ids = [self.vocab.get(ch, self.pad_token_id) for ch in text[:max_len]]
        if verbose:
            print(f"字符转token: {list(text[:max_len])} -> {ids}")
        # 为token序列添加起始token bos 和终止token eos
        ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if verbose:
            print(f"添加<BOS>和<EOS>: [{self.bos_token_id}] + {ids[1:-1]} + [{self.eos_token_id}] = {ids}")

        # 如果达不到max_len，则补充pad token
        # 如 对于 1+1=? 的token序列对应如下：
        # <BOS>, 1, +, 1, =, ?, <EOS>, <PAD>, <PAD>, <PAD>, <PAD>, <PAD>
        # 21, 1, 10, 1, 12, 15, 22, 20, 20, 20, 20, 20
        if len(ids) < max_len:
            pad_count = max_len - len(ids)
            ids += [self.pad_token_id] * pad_count
            if verbose:
                print(f"长度不足，补充{pad_count}个<PAD>: {ids}")
        else:
            ids = ids[:max_len]

        if verbose:
            print(f"最终token序列长度: {len(ids)}")
            print("token含义:")
            for i, idx in enumerate(ids):
                ch = self.inv_vocab.get(idx, '?')
                special = ""
                if idx == self.pad_token_id: special = " (PAD)"
                elif idx == self.bos_token_id: special = " (BOS)"
                elif idx == self.eos_token_id: special = " (EOS)"
                print(f"  位置{i}: {idx} -> '{ch}'{special}")

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

def apply_rope(q, k, seq_len, head_dim, verbose: bool = False):
    if verbose:
        print(f"\n=== RoPE位置编码 ===")
        print(f"  seq_len = {seq_len}, head_dim = {head_dim}")
    
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim))
    
    if verbose:
        print(f"  inv_freq (频率倒数): {inv_freq.tolist()}")
        print(f"    公式: inv_freq[i] = 1 / 10000^(2i / head_dim)")
    
    t = torch.arange(seq_len, device=q.device).float()
    
    if verbose:
        print(f"  t (位置索引): {t.tolist()}")
    
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    
    if verbose:
        print(f"  freqs形状: {freqs.shape}")
        print(f"  freqs (前3个位置的频率):")
        for i in range(min(3, seq_len)):
            print(f"    位置{i}: {freqs[i].tolist()}")
    
    emb = torch.cat((freqs, freqs), dim=-1)
    
    if verbose:
        print(f"  emb形状: {emb.shape}")
        print(f"  emb含义: 每个位置的旋转角度，前半和后半相同")
    
    cos = emb.cos().unsqueeze(0).unsqueeze(1)  # [1, 1, L, head_dim]
    sin = emb.sin().unsqueeze(0).unsqueeze(1)
    
    if verbose:
        print(f"  cos形状: {cos.shape}, sin形状: {sin.shape}")
        print(f"  cos[0,0,0] (位置0的cos值): {cos[0,0,0].tolist()}")
        print(f"  sin[0,0,0] (位置0的sin值): {sin[0,0,0].tolist()}")
        print(f"  cos[0,0,1] (位置1的cos值): {cos[0,0,1].tolist()}")
        print(f"  sin[0,0,1] (位置1的sin值): {sin[0,0,1].tolist()}")
    
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    
    if verbose:
        print(f"\n 旋转前后对比 (以第1个头为例):")
        print(f"  q[0,0,0] (旋转前位置0): {q[0,0,0].tolist()[:4]}...")
        print(f"  q_rot[0,0,0] (旋转后位置0): {q_rot[0,0,0].tolist()[:4]}...")
        print(f"  q[0,0,1] (旋转前位置1): {q[0,0,1].tolist()[:4]}...")
        print(f"  q_rot[0,0,1] (旋转后位置1): {q_rot[0,0,1].tolist()[:4]}...")
        print(f"\n  旋转公式: q_rot = q * cos + rotate_half(q) * sin")
        print(f"            rotate_half([x1, x2]) = [-x2, x1]")
    
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

    def forward(self, x, mask=None, verbose: bool = False):
        B, L, _ = x.shape
        
        if verbose:
            print(f"\n=== 多头注意力 ===")
            print(f"  B={B}, L={L}, d_model={self.d_model}, num_heads={self.num_heads}, head_dim={self.head_dim}")
        
        # QKV投影
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        if verbose:
            print(f"  Q投影后形状: {q.shape}")
            print(f"  K投影后形状: {k.shape}")
            print(f"  V投影后形状: {v.shape}")
            print(f"  Q[0,0,0] (第1个头，位置0): {q[0,0,0].tolist()}")
            print(f"  K[0,0,0] (第1个头，位置0): {k[0,0,0].tolist()}")
            print(f"  V[0,0,0] (第1个头，位置0): {v[0,0,0].tolist()}")

        # 应用RoPE位置编码
        q, k = apply_rope(q, k, L, self.head_dim, verbose=verbose)
        
        if verbose:
            print(f"\n  应用RoPE后:")
            print(f"  q_rot[0,0,0] (旋转后): {q[0,0,0].tolist()}")
            print(f"  k_rot[0,0,0] (旋转后): {k[0,0,0].tolist()}")

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if verbose:
            print(f"\n  注意力分数:")
            print(f"  attn_scores形状: {attn_scores.shape}")
            print(f"  attn_scores[0,0] (第1个头的注意力分数矩阵):")
            for i in range(min(5, L)):
                row = attn_scores[0,0,i].tolist()
                print(f"    位置{i} -> {row[:5]}...")
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        if verbose:
            print(f"\n  注意力权重(softmax后):")
            print(f"  attn_weights[0,0] (第1个头):")
            for i in range(min(5, L)):
                row = attn_weights[0,0,i].tolist()
                print(f"    位置{i}注意力: {row[:5]}... (和={sum(row):.4f})")
        
        # 加权求和
        attn_out = torch.matmul(attn_weights, v)
        
        if verbose:
            print(f"\n  加权求和后:")
            print(f"  attn_out形状: {attn_out.shape}")
            print(f"  attn_out[0,0,0]: {attn_out[0,0,0].tolist()}")
        
        # 拼接多头输出
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        
        if verbose:
            print(f"\n  拼接多头后形状: {attn_out.shape}")
        
        # 输出投影
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

    def forward(self, x, mask=None, verbose: bool = False):
        if verbose:
            print(f"\n=== Transformer Block ===")
            print(f"  输入形状: {x.shape}")
            print(f"  x[0,0] (位置0输入): {x[0,0].tolist()[:5]}...")
        
        # 残差连接1: x = x + attn(ln1(x))
        x_ln1 = self.ln1(x)
        
        if verbose:
            print(f"\n  ln1(x)[0,0] (层归一化后): {x_ln1[0,0].tolist()[:5]}...")
        
        attn_out = self.attn(x_ln1, mask, verbose=verbose)
        
        if verbose:
            print(f"\n  attn_out[0,0] (注意力输出): {attn_out[0,0].tolist()[:5]}...")
        
        x = x + attn_out
        
        if verbose:
            print(f"\n  残差连接后 x[0,0]: {x[0,0].tolist()[:5]}...")
            print(f"  残差公式: x = x + attn(ln1(x))")
        
        # 残差连接2: x = x + ffn(ln2(x))
        x_ln2 = self.ln2(x)
        
        if verbose:
            print(f"\n  ln2(x)[0,0] (层归一化后): {x_ln2[0,0].tolist()[:5]}...")
        
        ffn_out = self.ffn(x_ln2)
        
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
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.tokenizer_inv = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: '+', 11: '-', 12: '=', 13: '<', 14: '>', 15: '?', 16: '(', 17: ')', 18: ' ', 19: '\n',
            20: '<PAD>', 21: '<BOS>', 22: '<EOS>'
        }

    def forward(self, tokens, mask=None, verbose: bool = False):
        if verbose:
            print(f"\n=== MiniTransformer 前向传播 ===\n输入tokens形状: {tokens.shape}, 值: {tokens[0].tolist()}")
        
        x = self.embed(tokens)  # [B, L, d_model]
        if verbose:
            print(f"\n--- 嵌入层 ---")
            print(f"嵌入层输出形状: {x.shape}")
            print(f"嵌入层输出示例(前3个token的完整向量):")
            for i in range(min(3, x.size(1))):
                ch = self.tokenizer_inv.get(tokens[0][i].item(), '?') if hasattr(self, 'tokenizer_inv') else '?'
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
                ch = self.tokenizer_inv.get(i, '?') if hasattr(self, 'tokenizer_inv') else '?'
                special = ""
                if i == 20: special = " (PAD)"
                elif i == 21: special = " (BOS)"
                elif i == 22: special = " (EOS)"
                print(f"  词汇{i} -> '{ch}'{special}: {last_logits[i].item():.4f}")
        
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
@torch.no_grad()
def generate_response(
    model: MiniTransformer,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 1,       # 只有一个数字，因此，生成一个数字即可
    temperature: float = 1.0,      # 新增：控制随机性
    do_sample: bool = True,        # 新增：是否采样
    top_p: float = 0.9,            # 新增：核采样
    verbose: bool = False,
):
    model.eval()
    input_ids = tokenizer.encode(prompt, max_len=12, verbose=verbose)  # [1, L]
    generated = input_ids.squeeze(0).tolist()

    if verbose:
        print(f"\n=== 生成响应 ===\n初始prompt: '{prompt}'")
        print(f"初始generated序列: {generated}")

    for step in range(max_new_tokens):
        # 使用完整的生成序列（去掉截断，保留全部上下文）
        tokens = torch.tensor([generated], device=next(model.parameters()).device)
        
        if verbose:
            print(f"\n--- 生成第{step+1}个token ---")
            print(f"输入tokens: {tokens[0].tolist()}")
        
        logits = model(tokens, verbose=verbose)  # [1, seq_len, vocab]
        
        next_token_logits = logits[0, -1, :] / temperature  # 温度缩放
        
        if verbose:
            print(f"\n下一步token预测:")
            print(f"  温度缩放前logits最大值: {logits[0, -1, :].max().item():.4f}")
            print(f"  温度缩放后logits最大值: {next_token_logits.max().item():.4f}")
            print(f"  预测概率最高的3个token:")
            top3 = torch.topk(next_token_logits, 3)
            for i in range(3):
                idx = top3.indices[i].item()
                prob = F.softmax(next_token_logits, dim=-1)[idx].item()
                ch = tokenizer.inv_vocab.get(idx, '?')
                special = ""
                if idx == tokenizer.pad_token_id: special = " (PAD)"
                elif idx == tokenizer.bos_token_id: special = " (BOS)"
                elif idx == tokenizer.eos_token_id: special = " (EOS)"
                print(f"    位置{i+1}: token={idx} -> '{ch}'{special}, 概率={prob*100:.2f}%")

        if do_sample:
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cumsum_probs > top_p] = -float('Inf')
                probs = F.softmax(sorted_logits, dim=-1)
                next_token = sorted_indices[torch.multinomial(probs, 1).item()].item()
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
        else:
            next_token = torch.argmax(next_token_logits).item()

        if verbose:
            ch = tokenizer.inv_vocab.get(next_token, '?')
            special = ""
            if next_token == tokenizer.pad_token_id: special = " (PAD)"
            elif next_token == tokenizer.bos_token_id: special = " (BOS)"
            elif next_token == tokenizer.eos_token_id: special = " (EOS)"
            print(f"  选择的token: {next_token} -> '{ch}'{special}")

        if next_token == tokenizer.eos_token_id:
            if verbose:
                print("  遇到<EOS>，停止生成")
            break
        generated.append(next_token)
        
        if verbose:
            print(f"  当前generated序列: {generated}")

    if verbose:
        print(f"\n最终生成结果: {generated}")
        decoded = tokenizer.decode(torch.tensor(generated))
        print(f"解码结果: '{decoded}'")

    return torch.tensor(generated, device=next(model.parameters()).device).unsqueeze(0)

# ==================== 5. 奖励函数 ====================
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

    def train_step(self, prompts, ground_truths, verbose: bool = False):
        B = len(prompts)
        G = self.config.group_size
        device = next(self.policy.parameters()).device

        if verbose:
            print(f"\n{'='*60}")
            print(f"=== GRPO 训练步骤 ===")
            print(f"  batch_size (B) = {B}")
            print(f"  group_size (G) = {G}")
            print(f"  总样本数 = {B * G}")
            print(f"  prompts = {prompts}")
            print(f"  ground_truths = {ground_truths}")

        # <<< 1. 将当前策略复制给旧策略（作为采样策略）
        if verbose:
            print(f"\n--- 步骤1: 同步old_policy ---")
            print(f"  将当前policy参数复制到old_policy")
        self.old_policy.load_state_dict(self.policy.state_dict())

        # <<< 2. Rollout：使用 old_policy 采样
        if verbose:
            print(f"\n--- 步骤2: Rollout采样 ---")
            print(f"  使用old_policy为每个prompt生成{G}个响应")
        
        all_responses = []
        for prompt_idx, prompt in enumerate(prompts):
            if verbose:
                print(f"\n  处理prompt[{prompt_idx}]: '{prompt}'")
            for g in range(G):
                resp = generate_response(model=self.old_policy, tokenizer=self.tokenizer, prompt=prompt, verbose=verbose and g==0)  # 只打印第一个采样的详细信息
                all_responses.append(resp)
                if verbose:
                    decoded = self.tokenizer.decode(resp.squeeze(0))
                    print(f"    采样{g+1}: {resp[0].tolist()} -> '{decoded}'")

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
        
        if verbose:
            print(f"\n  所有响应填充后形状: {responses.shape}")
            print(f"  响应示例(前3个):")
            for i in range(min(3, responses.size(0))):
                decoded = self.tokenizer.decode(responses[i])
                print(f"    [{i}]: {responses[i].tolist()} -> '{decoded}'")

        # <<< 3. 计算 old_log_probs（来自 old_policy，无梯度）
        if verbose:
            print(f"\n--- 步骤3: 计算old_log_probs ---")
            print(f"  使用old_policy计算响应的log概率（无梯度）")
        with torch.no_grad():
            old_log_probs = self.old_policy.get_log_probs(responses)  # 使用 old_policy
        
        if verbose:
            print(f"  old_log_probs形状: {old_log_probs.shape}")
            print(f"  old_log_probs值: {old_log_probs.tolist()}")

        # 4. 奖励
        if verbose:
            print(f"\n--- 步骤4: 计算奖励 ---")
        rewards = math_reward_fn(responses, self.tokenizer, ground_truths, G, verbose=verbose)
        rewards = rewards.view(B, G)
        
        if verbose:
            print(f"  rewards形状: {rewards.shape}")
            print(f"  rewards值:\n{rewards}")
            print(f"\n  奖励含义:")
            for i in range(B):
                print(f"    问题{i} (\"{prompts[i]}\"): 奖励分布 = {rewards[i].tolist()}, 标准答案 = \"{ground_truths[i]}\"")

        # 5. 优势
        if verbose:
            print(f"\n--- 步骤5: 计算优势(GRPO核心) ---")
            print(f"  组内归一化优势 = (奖励 - 组内均值) / (组内标准差 + 1e-8)")
        
        mean_r = rewards.mean(dim=1, keepdim=True) # 形状 [B, 1]（每行的平均值）。
        std_r = rewards.std(dim=1, keepdim=True) # 形状 [B, 1]（每行的标准差）。
        advantages = (rewards - mean_r) / (std_r + 1e-8) # 形状 [B, G]。通过组内归一化（减去平均值，除以标准差）计算出的优势值。
        advantages_flat = advantages.view(-1) # 将 advantages 展平为形状 [B * G] 的一维张量

        if verbose:
            print(f"  mean_r (组内均值):\n{mean_r}")
            print(f"  std_r (组内标准差):\n{std_r}")
            print(f"  advantages (组内归一化优势):\n{advantages}")
            print(f"  advantages_flat: {advantages_flat.tolist()}")
            print(f"\n  优势含义:")
            print(f"    正值: 该响应比同组平均水平好")
            print(f"    负值: 该响应比同组平均水平差")
            print(f"    零: 该响应等于同组平均水平")

        # 6. 参考策略 log probs（只需计算一次，ref 冻结）
        if verbose:
            print(f"\n--- 步骤6: 计算参考策略log_probs ---")
            print(f"  使用ref策略计算响应的log概率（无梯度）")
        with torch.no_grad():
            log_probs_ref = self.ref.get_log_probs(responses)  # [B*G]
        
        if verbose:
            print(f"  log_probs_ref形状: {log_probs_ref.shape}")
            print(f"  log_probs_ref值: {log_probs_ref.tolist()}")

        # <<< 7. 对同一批数据进行多次梯度更新（PPO epochs）
        if verbose:
            print(f"\n--- 步骤7: PPO梯度更新 ({self.config.ppo_epochs} epochs) ---")
            print(f"  EPS (剪裁参数) = {0.2}")
            print(f"  BETA (KL惩罚系数) = {0.0001}")
        
        EPS = 0.2
        BETA = 0.0001
        for epoch in range(self.config.ppo_epochs):
            if verbose:
                print(f"\n  === Epoch {epoch+1}/{self.config.ppo_epochs} ===")
            
            # 每次更新后重新计算当前策略的 log 概率
            log_probs_theta = self.policy.get_log_probs(responses)  # 有梯度
            
            if verbose:
                print(f"    log_probs_theta (当前策略): {log_probs_theta.tolist()}")

            ratio = torch.exp(log_probs_theta - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS)
            
            if verbose:
                print(f"    ratio (概率比值): {ratio.tolist()}")
                print(f"    clipped_ratio (剪裁后): {clipped_ratio.tolist()}")

            surr1 = ratio * advantages_flat # 逐元素乘法
            surr2 = clipped_ratio * advantages_flat
            surrogate_loss = torch.min(surr1, surr2)
            
            if verbose:
                print(f"    surr1 = ratio * advantages: {surr1.tolist()}")
                print(f"    surr2 = clipped_ratio * advantages: {surr2.tolist()}")
                print(f"    surrogate_loss (min(surr1, surr2)): {surrogate_loss.tolist()}")

            log_diff = log_probs_ref - log_probs_theta
            kl_div = torch.exp(log_diff) - log_diff - 1
            kl_loss = BETA * kl_div.mean()

            if verbose:
                print(f"    log_diff = log_probs_ref - log_probs_theta: {log_diff.tolist()}")
                print(f"    kl_div: {kl_div.tolist()}")
                print(f"    kl_loss = {BETA} * kl_div.mean() = {kl_loss.item():.6f}")

            loss = -surrogate_loss.mean() + kl_loss
            
            if verbose:
                print(f"    loss = -surrogate_loss.mean() + kl_loss")
                print(f"         = -{surrogate_loss.mean().item():.6f} + {kl_loss.item():.6f}")
                print(f"         = {loss.item():.6f}")

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            if verbose:
                print(f"    梯度更新完成")

        # 返回监控指标（取最后一次的 ratio 和 loss）
        return {
            'loss': loss.item(),
            'kl_loss': kl_loss.item(),
            'mean_ratio': ratio.mean().item(),
        }


def evaluate_model(model, tokenizer, test_prompts, test_ground_truths, max_new_tokens=1):
    """
    评估模型在测试集上的准确率（贪婪解码）。
    """
    model.eval()
    correct = 0
    total = len(test_prompts)
    with torch.no_grad():
        for prompt, gt in zip(test_prompts, test_ground_truths):
            # 使用贪婪解码（不采样）
            resp_tokens = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # 确定性的
                temperature=1.0
            )
            # 解码并检查答案
            text = tokenizer.decode(resp_tokens.squeeze(0))
            if gt in text:
                correct += 1
    return correct / total

# ==================== 7. 主程序 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleTokenizer()
    print("词汇表:")
    print(tokenizer.vocab)
    vocab_size = tokenizer.vocab_size

    policy = MiniTransformer(vocab_size,pad_token_id=tokenizer.pad_token_id, d_model=16, num_heads=2, num_layers=2).to(device)
    ref = MiniTransformer(vocab_size,pad_token_id=tokenizer.pad_token_id, d_model=16, num_heads=2, num_layers=2).to(device)
    ref.load_state_dict(policy.state_dict())

    class Config:
        group_size = 3
        lr = 1e-2
        ppo_epochs = 1
    config = Config()

    trainer = GRPOTrainer(policy, ref, tokenizer, config)
    
    train_prompts = ["1+1=?"]
    train_ground_truths = ["2"]
    
    steps_num = 1
    
    for step in range(steps_num):
        metrics = trainer.train_step(train_prompts, train_ground_truths, verbose=True)
        print(f"\nStep {step}: Loss={metrics['loss']:.4f}, KL={metrics['kl_loss']:.4f}, Ratio={metrics['mean_ratio']:.4f}")
    print("\n训练完成！")