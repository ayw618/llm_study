"""
简化版 GRPO (Group Relative Policy Optimization) 训练实现
从 verl 框架中提取核心逻辑，去除 Ray 分布式和复杂文件结构

核心组件：
1. compute_grpo_outcome_advantage: 组内相对优势计算 (来自 core_algos.py)
2. compute_policy_loss: PPO-style 策略损失 (含 clip 和 KL 惩罚)
3. grpo_train_step: 单步训练循环 (来自 ray_trainer.py 的 fit 逻辑)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np


# ==================== 配置类 ====================
@dataclass
class GRPOConfig:
    """GRPO 训练配置"""
    # 组大小：每个 prompt 采样多少个 response
    group_size: int = 4
    # PPO clip 范围
    clip_ratio: float = 0.2
    # KL 惩罚系数
    kl_coef: float = 0.001
    # 是否用标准差归一化 advantage
    norm_adv_by_std: bool = True
    # 损失聚合模式: "token-mean" 或 "seq-mean-token-mean"
    loss_agg_mode: str = "token-mean"
    # 训练轮数 (每个 batch 重复训练的 epoch 数)
    ppo_epochs: int = 1
    # 学习率
    lr: float = 1e-6


# ==================== 核心算法：优势计算 ====================
# 来源: verl/trainer/ppo/core_algos.py - compute_grpo_outcome_advantage

def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,    # [batch_size, response_len]
    response_mask: torch.Tensor,          # [batch_size, response_len]
    index: np.ndarray,                    # [batch_size] 每个样本所属的组 ID
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO 优势计算 (Outcome Reward 版本)
    
    对于每个组 g:
        a_i = (r_i - mean(r_g)) / (std(r_g) + epsilon)
    
    然后将标量 advantage 广播到整个 response 序列 (乘以 response_mask)
    
    Args:
        token_level_rewards: 每个 token 的奖励 (实际只有最后一个 token 有值)
        response_mask: 有效 token 的 mask
        index: 每个样本对应的组 ID (0, 0, 0, 0, 1, 1, 1, 1, ...)
        norm_adv_by_std_in_grpo: 是否除以组内标准差
    
    Returns:
        advantages: [batch_size, response_len] 每个 token 的 advantage
        returns: 同 advantages (GRPO 中 return = advantage)
    """
    # 每个 response 的总奖励 (标量)
    scores = token_level_rewards.sum(dim=-1)  # [batch_size]
    
    # 按组 ID 收集分数
    id2score = defaultdict(list)
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        # 计算每组的均值和标准差
        id2mean = {}
        id2std = {}
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # 只有 1 个样本时，均值为 0，标准差为 1 (避免除零)
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            else:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
        
        # 计算每个样本的 advantage (标量)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
    
    # 将标量 advantage 广播到 token 维度
    advantages = scores.unsqueeze(-1) * response_mask  # [batch_size, response_len]
    return advantages, advantages


# ==================== 核心算法：策略损失 ====================
# 来源: verl/trainer/ppo/core_algos.py - 策略损失函数

def compute_policy_loss(
    old_log_probs: torch.Tensor,      # [batch_size, response_len]
    log_probs: torch.Tensor,          # [batch_size, response_len]
    advantages: torch.Tensor,         # [batch_size, response_len]
    response_mask: torch.Tensor,      # [batch_size, response_len]
    clip_ratio: float = 0.2,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, dict]:
    """
    计算 PPO-style 策略损失 (含 clip)
    
    Loss = - mean( min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) )
    
    Args:
        old_log_probs: 旧策略的 log 概率 (采样时保存)
        log_probs: 当前策略的 log 概率
        advantages: 优势值
        response_mask: 有效 token mask
        clip_ratio: clip 范围 ε
        loss_agg_mode: "token-mean" 或 "seq-mean-token-mean"
    
    Returns:
        loss: 标量损失
        metrics: 监控指标字典
    """
    # 计算概率比率 ratio = π_θ / π_old
    ratio = torch.exp(log_probs - old_log_probs)  # [batch_size, response_len]
    
    # 裁剪后的比率
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    
    # Surrogate 损失
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    surr_loss = torch.min(surr1, surr2)  # [batch_size, response_len]
    
    # 聚合损失
    if loss_agg_mode == "token-mean":
        # 对所有 token 取平均 (verl 默认)
        loss = - (surr_loss * response_mask).sum() / (response_mask.sum() + 1e-8)
    elif loss_agg_mode == "seq-mean-token-mean":
        # 先对每个序列取平均，再对 batch 取平均 (原始 GRPO 论文)
        seq_loss = (surr_loss * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-8)
        loss = - seq_loss.mean()
    else:
        raise ValueError(f"Unknown loss_agg_mode: {loss_agg_mode}")
    
    # 计算 approx_kl 用于监控
    with torch.no_grad():
        log_ratio = log_probs - old_log_probs
        approx_kl = ((log_ratio.exp() - 1) - log_ratio).mean()
    
    metrics = {
        "policy_loss": loss.item(),
        "approx_kl": approx_kl.item(),
        "clip_fraction": ((ratio - clipped_ratio).abs() > 1e-6).float().mean().item(),
    }
    
    return loss, metrics


# ==================== 核心算法：KL 惩罚 ====================
# 来源: verl 的 apply_kl_penalty

def compute_kl_penalty(
    log_probs: torch.Tensor,          # [batch_size, response_len]
    ref_log_probs: torch.Tensor,      # [batch_size, response_len]
    response_mask: torch.Tensor,      # [batch_size, response_len]
    kl_coef: float = 0.001,
) -> torch.Tensor:
    """
    计算 KL 散度惩罚项，直接加到损失中
    
    KL(π_θ || π_ref) = (π_ref / π_θ) - log(π_ref / π_θ) - 1
    
    来源: verl 的 apply_kl_penalty，使用 loss 模式 (而非 reward 模式)
    参考: https://github.com/verl-project/verl/issues/134
    """
    # 计算 log 概率差
    log_diff = ref_log_probs - log_probs  # log(π_ref / π_θ)
    
    # KL 散度: exp(log_diff) - log_diff - 1
    # 当 π_θ = π_ref 时，KL = 0
    kl = torch.exp(log_diff) - log_diff - 1  # [batch_size, response_len]
    
    # 只对有效 token 计算，取平均
    kl_loss = (kl * response_mask).sum() / (response_mask.sum() + 1e-8)
    
    return kl_coef * kl_loss


# ==================== 训练器：单步训练 ====================
# 来源: verl/trainer/ppo/ray_trainer.py - fit 方法中的核心循环

class GRPOTrainer:
    """
    简化版 GRPO 训练器 (单进程，无分布式)
    
    训练流程 (每步):
    1. Rollout: 用当前策略为每个 prompt 生成 group_size 个 response
    2. Reward: 计算每个 response 的奖励 (规则验证)
    3. Advantage: 组内相对优势计算 (compute_grpo_outcome_advantage)
    4. Policy Update: 计算 loss (含 clip + KL)，反向传播
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer,
        config: GRPOConfig,
        reward_fn,  # 奖励函数: (responses, prompts) -> rewards
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # 优化器
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    def _generate_responses(self, prompts: list[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        为每个 prompt 生成 group_size 个 response
        
        Returns:
            responses: [batch_size * group_size, response_len]
            response_mask: [batch_size * group_size, response_len]
            log_probs: [batch_size * group_size, response_len] 采样时的 log 概率
        """
        # 实际实现中会使用 model.generate()
        # 这里仅示意，需根据实际模型实现
        pass
    
    def train_step(self, batch_prompts: list[str]) -> dict:
        """
        单步 GRPO 训练
        
        Args:
            batch_prompts: 长度为 batch_size 的 prompt 列表
            
        Returns:
            metrics: 训练指标字典
        """
        config = self.config
        device = next(self.model.parameters()).device
        
        # ===== Step 1: Rollout - 生成 responses =====
        # 每个 prompt 采样 group_size 个 response
        # 实际使用 model.generate()，这里用占位
        # responses: [batch_size * group_size, response_len]
        # response_mask: [batch_size * group_size, response_len]
        # old_log_probs: [batch_size * group_size, response_len]
        
        # ===== Step 2: 计算奖励 =====
        # rewards: [batch_size * group_size] 标量奖励
        # 使用规则验证 (如数学题答案比对)
        # rewards = self.reward_fn(responses, prompts)
        
        # ===== Step 3: 构建组 ID =====
        # 每个 prompt 对应 group_size 个样本
        batch_size = len(batch_prompts)
        group_size = config.group_size
        index = np.repeat(np.arange(batch_size), group_size)  # [0,0,0,0, 1,1,1,1, ...]
        
        # ===== Step 4: 计算 Advantage (GRPO 核心) =====
        # token_level_rewards: [total_samples, response_len]
        # 将标量奖励广播到 token 维度 (只在最后一个有效 token 有值)
        token_level_rewards = torch.zeros_like(response_mask, dtype=torch.float32)
        # 实际实现中，将 rewards 放到每个 response 的最后一个有效 token 位置
        
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            norm_adv_by_std_in_grpo=config.norm_adv_by_std,
        )
        
        # ===== Step 5: PPO Epochs =====
        # 在 same batch 上训练多个 epoch
        total_samples = len(batch_prompts) * group_size
        
        for epoch in range(config.ppo_epochs):
            # 可选: shuffle mini-batches
            
            # 前向传播: 计算当前策略的 log 概率
            # log_probs = model(response).log_probs  # [total_samples, response_len]
            
            # 前向传播: 计算参考策略的 log 概率
            with torch.no_grad():
                # ref_log_probs = ref_model(response).log_probs
                pass
            
            # ===== Step 6: 计算策略损失 (含 clip) =====
            policy_loss, policy_metrics = compute_policy_loss(
                old_log_probs=old_log_probs,
                log_probs=log_probs,
                advantages=advantages,
                response_mask=response_mask,
                clip_ratio=config.clip_ratio,
                loss_agg_mode=config.loss_agg_mode,
            )
            
            # ===== Step 7: 计算 KL 惩罚 =====
            kl_loss = compute_kl_penalty(
                log_probs=log_probs,
                ref_log_probs=ref_log_probs,
                response_mask=response_mask,
                kl_coef=config.kl_coef,
            )
            
            # ===== Step 8: 总损失 & 反向传播 =====
            total_loss = policy_loss + kl_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            # 可选: grad norm clipping
            self.optimizer.step()
        
        # 返回监控指标
        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            **policy_metrics,
        }
        return metrics


# ==================== 使用示例 ====================

def example_training():
    """
    GRPO 训练使用示例
    注意: 这是伪代码，需要替换为实际的模型、数据集和奖励函数
    """
    # 1. 加载模型
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    # ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # 2. 定义奖励函数 (规则验证)
    def math_reward_fn(responses, prompts):
        """
        数学题奖励: 提取答案并与 ground truth 比对
        返回: [batch_size] 的奖励张量 (0 或 1)
        """
        # 实际实现: 解析 <answer> 标签，与正确答案比对
        pass
    
    # 3. 创建训练器
    # config = GRPOConfig(group_size=4, clip_ratio=0.2, kl_coef=0.001)
    # trainer = GRPOTrainer(
    #     model=model,
    #     ref_model=ref_model,
    #     tokenizer=tokenizer,
    #     config=config,
    #     reward_fn=math_reward_fn,
    # )
    
    # 4. 训练循环
    # for epoch in range(num_epochs):
    #     for batch in dataloader:
    #         metrics = trainer.train_step(batch["prompts"])
    #         print(f"Step {step}: loss={metrics['loss']:.4f}, kl={metrics['kl_loss']:.4f}")
    
    print("GRPO 训练器已初始化，请替换为实际模型和数据进行训练")


if __name__ == "__main__":
    example_training()