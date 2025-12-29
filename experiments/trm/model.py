from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.layers import (
    trunc_normal_init_, rms_norm, SwiGLU, Attention, RotaryEmbedding, 
    CosSin, CastedEmbedding, CastedLinear, CastedSparseEmbedding
)

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_L: torch.Tensor

@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]

class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "float32"
    mlp_t: bool = False 
    puzzle_emb_len: int = 0 
    no_ACT_continue: bool = True 

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pre-Norm
        normed = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=normed)
        hidden_states = hidden_states + attn_out
        
        normed_2 = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        mlp_out = self.mlp(normed_2)
        hidden_states = hidden_states + mlp_out
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale = math.sqrt(self.config.hidden_size)
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)
        self.puzzle_emb_len = 0 

        self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                          max_position_embeddings=self.config.seq_len,
                                          base=self.config.rope_theta)

        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _ in range(self.config.L_layers)]
        )

        # MATCHING CHECKPOINT: [1, SeqLen, Hidden]
        self.L_init = nn.Parameter(trunc_normal_init_(torch.empty(1, self.config.seq_len, self.config.hidden_size), std=0.02))

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-3)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.long())
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: torch.device):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_L=torch.zeros(batch_size, self.config.seq_len, self.config.hidden_size, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        B = reset_flag.shape[0]
        # Expand [1, L, D] -> [B, L, D]
        l_init_expanded = self.L_init.expand(B, -1, -1)
        mask = reset_flag.view(-1, 1, 1)
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_L=torch.where(mask, l_init_expanded, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(cos_sin=self.rotary_emb())
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        z_L = carry.z_L
        
        # Apply Recurrence
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L + input_embeddings, **seq_info)
        
        # Final pass for output
        z_L = self.L_level(z_L, **seq_info)

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_L=z_L.detach())
        output = self.lm_head(z_L)
        q_logits = self.q_head(z_L.mean(dim=1)) 
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size, device),
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),
            current_data={k: v.clone() for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]):
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        
        mask_expanded = carry.halted.view((-1, ) + (1, ) * (batch["inputs"].ndim - 1))
        new_current_data = {
            k: torch.where(mask_expanded, batch[k], carry.current_data[k]) 
            for k in carry.current_data if k in batch
        }

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        new_steps = new_steps + 1
        is_max_step = new_steps >= self.config.halt_max_steps
        
        if self.training:
            should_halt = (q_halt_logits > 0)
            rand_halt = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
            halted = should_halt ^ rand_halt
        else:
            halted = (q_halt_logits > 0)

        halted = halted | is_max_step
        
        outputs = {
            "logits": logits,
            "q_halt": q_halt_logits,
            "halted": halted
        }
        
        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs