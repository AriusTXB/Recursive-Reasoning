import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Initializers ---
def trunc_normal_init_(tensor, mean=0., std=1., a=-2., b=2.):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

# --- Normalization ---
def rms_norm(x, variance_epsilon=1e-5):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    return x.to(input_dtype)

# --- Activation Layers ---
class SwiGLU(nn.Module):
    def __init__(self, hidden_size, expansion=4.0):
        super().__init__()
        intermediate_size = int(hidden_size * expansion)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class LinearSwish(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, x):
        return F.silu(self.linear(x))

# --- Embeddings (Casted Wrappers) ---
class CastedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, init_std=0.02, cast_to=None, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias=bias)

class CastedSparseEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, batch_size=None, init_std=0.02, cast_to=None):
        super().__init__(num_embeddings, embedding_dim)

# --- Attention & Positional ---
CosSin = tuple

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x=None, seq_len=None):
        return self.cos_cached, self.sin_cached

class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, cos_sin=None):
        B, L, D = hidden_states.shape
        # Initial Projections: [B, L, H, D]
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        
        # --- FIX: Transpose to [B, H, L, D] BEFORE RoPE ---
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if cos_sin is not None:
            cos, sin = cos_sin
            # cos/sin shape is [1, 1, MaxLen, D]
            # Slicing to current length L: [1, 1, L, D]
            # Broadcasting with q [B, H, L, D] works perfectly now (1 matches H)
            cos = cos[:, :, :L, :]
            sin = sin[:, :, :L, :]
            
            def rotate_half(x):
                x1, x2 = x.chunk(2, dim=-1)
                return torch.cat((-x2, x1), dim=-1)
            
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

        # Scaled Dot Product Attention
        # PyTorch F.scaled_dot_product_attention expects [B, H, L, D]
        out = F.scaled_dot_product_attention(q, k, v)
        
        # Transpose back to [B, L, H, D] and flatten to [B, L, Hidden]
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        
        return self.o_proj(out)