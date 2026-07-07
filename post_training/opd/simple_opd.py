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

    def get_token_log_probs(self, tokens: torch.Tensor):
        """返回每个 token 位置的 log 概率（形状 [B, L-1]）"""
        B, L = tokens.shape
        if L <= 1:
            return torch.zeros(B, L-1, device=tokens.device)
        
        logits = self.forward(tokens)
        shift_logits = logits[:, :-1, :]
        shift_tokens = tokens[:, 1:]
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, 
            index=shift_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs


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
    verbose: bool = False,
):
    """学生模型自回归生成响应（On-policy 采样）"""
    model.eval()
    input_ids = tokenizer.encode(prompt, max_len=12, verbose=verbose)
    generated = input_ids.squeeze(0).tolist()

    if verbose:
        print(f"\n=== 生成响应 ===\n初始prompt: '{prompt}'")
        print(f"初始generated序列: {generated}")

    for step in range(max_new_tokens):
        tokens = torch.tensor([generated], device=next(model.parameters()).device)
        
        if verbose:
            print(f"\n--- 生成第{step+1}个token ---")
            print(f"输入tokens: {tokens[0].tolist()}")
        
        logits = model(tokens, verbose=verbose)
        next_token_logits = logits[0, -1, :] / temperature

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
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum_probs <= top_p
                if not mask.any():
                    mask[0] = True
                sorted_probs[~mask] = 0.0
                prob_sum = sorted_probs.sum()
                if prob_sum == 0:
                    sorted_probs = torch.ones_like(sorted_probs) / len(sorted_probs)
                else:
                    sorted_probs = sorted_probs / prob_sum
                sampled_sorted_idx = torch.multinomial(sorted_probs, 1).item()
                next_token = sorted_indices[sampled_sorted_idx].item()
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

    return torch.tensor(generated, device=next(model.parameters()).device).unsqueeze(0)


# ==================== 5. OPD 训练器（标准版） ====================
class OPDTrainer:
    """
    On-Policy Distillation (OPD) 训练器 - 标准实现
    
    参考：https://thinkingmachines.ai/blog/on-policy-distillation/
    
    标准 OPD 流程 [8†L2-L10][9†L7-L15]：
        1. 学生模型采样轨迹 (on-policy rollout)
        2. 教师模型计算每个 token 的 log 概率
        3. 计算反向 KL 散度: KL(π_student || π_teacher)
        4. 将负反向 KL 作为每个 token 的优势
        5. 使用 RL 重要性采样损失进行单次更新 [9†L37-L39][8†L27-L28]
    
    与原始代码的关键区别：
        - ❌ 移除 PPO 风格的 multiple epochs 和 clipping [2†L50-L51]
        - ✅ 使用单次更新（标准 OPD）
        - ✅ 直接使用负反向 KL 作为优势
        - ✅ 使用重要性采样损失（而非 PPO 裁剪损失）
    """
    
    def __init__(self, student, teacher, tokenizer, config):
        self.student = student
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr)
        
        # 冻结教师模型
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # 设置tokenizer_inv用于打印
        student.tokenizer_inv = tokenizer.inv_vocab
        teacher.tokenizer_inv = tokenizer.inv_vocab

    def train_step(self, prompts: List[str], verbose: bool = False):
        """
        单步 OPD 训练（标准流程）
        
        标准 OPD 使用单次更新，不进行 PPO 风格的 multiple epochs [2†L50-L51]
        """
        B = len(prompts)
        G = self.config.group_size
        device = next(self.student.parameters()).device

        if verbose:
            print(f"\n============================================================\n=== OPD 训练步骤 ===\n  batch_size (B) = {B}\n  group_size (G) = {G}\n  总样本数 = {B * G}\n  prompts = {prompts}")

        # ========== Step 1: On-policy Rollout ==========
        if verbose:
            print(f"\n--- 步骤1: On-policy Rollout ---")
            print(f"  使用学生模型为每个prompt生成{G}个响应")
        
        all_responses = []
        for idx, prompt in enumerate(prompts):
            if verbose:
                print(f"\n  处理prompt[{idx}]: '{prompt}'")
            for g in range(G):
                resp = generate_response(self.student, self.tokenizer, prompt, verbose=verbose and idx==0 and g==0)
                all_responses.append(resp)
                if verbose and idx==0:
                    decoded = self.tokenizer.decode(resp.squeeze(0))
                    print(f"    采样{g+1}: {resp[0].tolist()} -> '{decoded}'")

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

        responses = pad_responses(all_responses)  # [B*G, max_len]
        
        if verbose:
            print(f"\n  所有响应填充后形状: {responses.shape}")

        # ========== Step 2: 计算学生模型的 log 概率 ==========
        if verbose:
            print(f"\n--- 步骤2: 计算学生模型log概率 ---")
        student_log_probs = self.student.get_token_log_probs(responses)  # [B*G, L-1]
        
        if verbose:
            print(f"  student_log_probs形状: {student_log_probs.shape}")
            print(f"  student_log_probs[0] (第1个样本的token级log概率): {student_log_probs[0].tolist()[:6]}...")

        # ========== Step 3: 计算教师模型的 log 概率 ==========
        if verbose:
            print(f"\n--- 步骤3: 计算教师模型log概率 ---")
            print(f"  教师模型对同一个轨迹的每个token评分")
        with torch.no_grad():
            teacher_log_probs = self.teacher.get_token_log_probs(responses)  # [B*G, L-1]
        
        if verbose:
            print(f"  teacher_log_probs形状: {teacher_log_probs.shape}")
            print(f"  teacher_log_probs[0] (第1个样本的token级log概率): {teacher_log_probs[0].tolist()[:6]}...")

        # ========== Step 4: 计算反向 KL 散度 ==========
        if verbose:
            print(f"\n--- 步骤4: 计算反向KL散度 ---")
            print(f"  公式: KL(π_student || π_teacher) = E[log π_student - log π_teacher]")
        
        reverse_kl = student_log_probs - teacher_log_probs  # [B*G, L-1]
        
        if verbose:
            print(f"  reverse_kl形状: {reverse_kl.shape}")
            print(f"  reverse_kl[0] (第1个样本的token级KL散度): {reverse_kl[0].tolist()[:6]}...")

        # 创建有效 token 掩码
        shift_tokens = responses[:, 1:]
        token_mask = (shift_tokens != self.tokenizer.pad_token_id).float()  # [B*G, L-1]
        
        if verbose:
            print(f"  token_mask形状: {token_mask.shape}")
            print(f"  token_mask[0] (第1个样本的有效token掩码): {token_mask[0].tolist()[:6]}...")

        # ========== Step 5: 标准 OPD 更新 ==========
        if verbose:
            print(f"\n--- 步骤5: OPD损失计算与更新 ---")
            print(f"  优势 = -反向KL")
            print(f"  损失 = -E[ ratio * advantages ]")
        
        student_log_probs_current = self.student.get_token_log_probs(responses)
        ratio = torch.exp(student_log_probs_current - student_log_probs.detach())
        advantages = -reverse_kl.detach()  # [B*G, L-1]
        
        if verbose:
            print(f"  ratio形状: {ratio.shape}")
            print(f"  ratio[0] (第1个样本的重要性采样比率): {ratio[0].tolist()[:6]}...")
            print(f"  advantages形状: {advantages.shape}")
            print(f"  advantages[0] (第1个样本的token级优势): {advantages[0].tolist()[:6]}...")
        
        surrogate_loss = ratio * advantages  # [B*G, L-1]
        loss = -(surrogate_loss * token_mask).sum() / (token_mask.sum() + 1e-8)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()

        # 计算平均反向 KL 用于监控
        mean_reverse_kl = (reverse_kl * token_mask).sum() / (token_mask.sum() + 1e-8)

        return {
            'loss': loss.item(),
            'mean_reverse_kl': mean_reverse_kl.item(),
            'mean_ratio': ratio.mean().item(),
        }


# ==================== 6. 评估函数 ====================
def evaluate_model(model, tokenizer, test_prompts, test_ground_truths, max_new_tokens=1):
    """评估模型在测试集上的准确率（贪婪解码）"""
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


# ==================== 7. 主程序 ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleTokenizer()
    print("词汇表:", tokenizer.vocab)
    vocab_size = tokenizer.vocab_size

    # 学生模型（待训练）
    student = MiniTransformer(
        vocab_size, 
        pad_token_id=tokenizer.pad_token_id, 
        d_model=16, 
        num_heads=2, 
        num_layers=2
    ).to(device)

    # 教师模型（高性能模型，冻结）
    teacher = MiniTransformer(
        vocab_size, 
        pad_token_id=tokenizer.pad_token_id, 
        d_model=16, 
        num_heads=2, 
        num_layers=2
    ).to(device)
    
    # 实际应用中教师应显著强于学生
    teacher.load_state_dict(student.state_dict())

    class Config:
        group_size = 8          # 每个 prompt 采样的轨迹数
        lr = 1e-3               # 学习率
    config = Config()

    trainer = OPDTrainer(student, teacher, tokenizer, config)

    # 训练数据
    train_prompts = ["1+1=?", "2+3=?", "3+4=?", "4+5=?"]
    train_ground_truths = ["2", "5", "7", "9"]

    # 测试数据
    test_prompts = ["2+4=?", "4+3=?", "1+4=?", "3+1=?"]
    test_ground_truths = ["6", "7", "5", "4"]

    steps_num = 100
    eval_interval = 20
    to_test = False

    for step in range(steps_num):
        metrics = trainer.train_step(train_prompts, verbose=(step == 0))
        print(f"Step {step}: Loss={metrics['loss']:.4f}, "
              f"Reverse_KL={metrics['mean_reverse_kl']:.4f}, "
              f"Ratio={metrics['mean_ratio']:.4f}")

        if to_test and (step + 1) % eval_interval == 0:
            acc = evaluate_model(student, tokenizer, test_prompts, test_ground_truths)
            print(f"  >>> Test Accuracy: {acc:.2f}")

    print("训练完成！")