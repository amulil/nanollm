import torch
import torch.nn as nn

from dataclasses import dataclass

@dataclass
class LlamaConfig:
    vocab_size: int = 128256
    hidden_size: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 8
    max_position_embeddings: int = 8192
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-05
    attention_bias: bool = False

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.gate_proj = nn.Linear(config.hidden_size, 3.5 * config.hidden_size)
        self.up_proj = nn.Linear(config.hidden_size, 3.5 * config.hidden_size)
        self.silu = nn.SiLU()
        self.down_proj = nn.Linear(3.5 * config.hidden_size, config.hidden_size)
    def forward(self, x):
        gate_x, up_x = self.gate_proj(x), self.up_proj(x)
        x = self.silu(gate_x)
        x = x * up_x
        x = self.down_proj(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LayerNorm is equivalent to T5LayerNorm ??
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        x_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(x_dtype)
    
class RoPE(nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    @torch.no_grad()
    def forward(self, x, position_ids):
        
    

    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.hidden_size // self.n_head
        
        self.q_proj = nn.Linear(config.hidden_size, config.n_head * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.n_kv_head * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.n_kv_head * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias))
        self.rotary_emb = RoPE(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
    def forward(self, x):
        B, T, _ = x.size()
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) 
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(v, position_ids)
        
        
        
        
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.self_attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, x):
        

class Llama(nn.Module):
    
    def __init__(self, config):
        super.__init__()
        # ? pad token id
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)