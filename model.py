# model.py
from typing import Optional, Tuple

import torch
from torch import nn

from dataclasses import dataclass
from simple_parsing.helpers import Serializable

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)



class CacheView:
    # TODO: Implement cacheview

    def __init__(self, cache_k: torch.Tensor, cache_v: torch.Tensor):
        self.cache_k = cache_k
        self.cache_v = cache_v



@dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


@dataclass
class ModelArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int

    max_batch_size: int = 0

    # For rotary embeddings. It not set, will be inferred from sliding window
    rope_theta: Optional[float] = None
    # If this is set, use sliding window attention rotating cache
    sliding_window: Optional[int] = None
    # If this is set, we will use MoE layers instead of dense layers
    moe: Optional[MoeArgs] = None


class Attention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.n_kv_heads = args.n_kv_heads


        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:




class TransformerBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)



class Transformer(nn.Module):

    def __init__(self, args: ModelArgs, pipeline_rank: int = 0, num_pipeline_ranks: int =1):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        assert self.vocab_size > 0, "Vocab size must be set"


        # Initialize all layers
        layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]

