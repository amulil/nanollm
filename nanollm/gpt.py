import math
import torch
import torch.nn as nn

from dataclasses import dataclass
from model import Block

@dataclass
class GPTConfig:
    block_size: int = 1024 # max position embedding
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_emdb),
            wpe = nn.Embdding(config.bloack_size, config.n_emdb),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.Linear(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.n_vocab_size, bias=False)
        
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        
    def forward():
        pass
    
    def crop_block_size():
        pass
    
    @classmethod
    def from_pretrained():
        pass
    
    def configure_optimizers():
        pass
    
    def estimate_mfu():
        pass
    
    @torch.no_grad()
    def generate():
        pass
        
        