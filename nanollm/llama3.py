import torch
import torch.nn as nn

from dataclasses import dataclass
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@dataclass
class LlamaConfig:
    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 14336
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 8
    max_position_embeddings: int = 8192
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-05
    attention_bias: bool = False
    mlp_bias: bool = False


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.silu = nn.SiLU()
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
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
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.hidden_size // self.n_head
        self.n_kv_head = config.n_kv_head
        self.n_kv_group = self.n_head // self.n_kv_head
        self.q_proj = nn.Linear(config.hidden_size, config.n_head * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.n_kv_head * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.n_kv_head * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.rotary_emb = RoPE(
            self.head_dim,
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
        position_ids = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, T)
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        k = repeat_kv(k, self.n_kv_group)
        v = repeat_kv(v, self.n_kv_group)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.o_proj(y)
        
        return y
        

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.self_attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Llama(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ? pad token id
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # weight sharing scheme
        self.embed_tokens.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.max_position_embeddings, "Cannot forward, max position embeddings is exhausted."
        
        tok_emb = self.embed_tokens(idx)
        for block in self.layers:
            x = block(tok_emb)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'llama3-8b'}
        config_args = {
            'llama3-8b': dict(n_layer=32),
        }[model_type]
        config_args['vocab_size'] = 128256
        
        config = LlamaConfig(**config_args)
        model = Llama(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.bias')]
        
        model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
        if model_type == 'llama3-8b':
            model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
        
        model_hf = AutoModelForCausalLM.from_pretrained(model_id)
        sd_hf = model_hf.state_dict()
        
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        print(sd_keys_hf)
        print(sd_keys)
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                _k = k if k == "lm_head.weight" else k[6:]
                assert sd_hf[k].shape == sd[_k].shape
                with torch.no_grad():
                    sd[_k].copy_(sd_hf[k])

        return model
    
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
    
model = Llama.from_pretrained("llama3-8b")
model.to(device)
model.eval()

print("load successful")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

idx = tokenizer.apply_chat_template([
    {"role": "user", "content": "你是谁?"}
], tokenize=True)
idx = torch.tensor(idx).unsqueeze(0).to(device)

logits, loss = model(idx)
probs = torch.softmax(logits[0, -1], dim=0)
topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
sample_rng = torch.Generator(device=device)
ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
xcol = torch.gather(topk_indices, -1, ix)
print("decode sample:", tokenizer.decode(xcol.tolist()))

