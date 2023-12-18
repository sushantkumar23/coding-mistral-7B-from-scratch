# model.py
from typing import Optional, Tuple, List

import torch
from torch import nn
from xformers import AttentionBias
from xformers.ops.fmha import memory_efficient_attention

from dataclasses import dataclass
from simple_parsing.helpers import Serializable

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)



@dataclass
class RotatingCacheInputMetadata:
    # rope absolute positions
    positions: torch.Tensor
    # which elements in the sequences need to be cached
    to_cache_mask: torch.Tensor
    # how many elements are cached per sequence
    cached_elements: torch.Tensor
    # where tokens should go in the cache
    cache_positions: torch.Tensor

    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]


def interleave_list(l1: List[torch.Tensor], l2: List[torch.Tensor]):
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


def unrotate(cache: torch.Tensor, seqlen: int) -> torch.Tensor:
    assert cache.ndim == 3 # (W, H, D)
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:
        return cache[:seqlen]
    elif position == 0:
        return cache
    else:
        return torch.cat([cache[position:], cache[:position]], dim=0)

class CacheView:

    def __init__(self, cache_k: torch.Tensor, cache_v: torch.Tensor, metadata: RotatingCacheInputMetadata, kv_seqlens: torch.Tensor):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: torch.Tensor, xv: torch.Tensor):
        """
        to_cache_mask masks the last [sliding_window] tokens in each sequence
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)

        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])

    def interleave_kv(self, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interleave the cached elements with the new elements
        """
        assert xk.ndim == xv.ndim == 3 # (seqlen, n_kv_heads, head_dim)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache yet, just return the new elements
            return xk, xv

        # Make it a list of tensors
        xk = torch.split(xk, self.metadata.seqlens)
        xv = torch.split(xv, self.metadata.seqlens)
        assert len(xk) == len(xv) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)} but got {len(xk)} and {len(xv)}"

        # Order the cached elements by unrotating
        cache_k = [unrotate(t, s) for t, s in zip(self.cache_k, self.kv_seqlens)]
        cache_v = [unrotate(t, s) for t, s in zip(self.cache_v, self.kv_seqlens)]

        interleaved_k = interleave_list(cache_k, xk)
        interleaved_v = interleave_list(cache_v, xv)

        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)


class RotatingBufferCache:
    """
    This is an example that implements a less naive rotating buffer cache, allowing for variable
    length sequences. Allocated cache is rectangular which is wasteful (see PageAttention for better mechanisms)
    """
    def __init__(self, n_layers: int, max_batch_size: int, sliding_window: int, n_kv_heads: int, head_dim: int):

        self.sliding_window = sliding_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.cache_k = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))

        self.cache_v = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        # holds the valid length for each batch element in the cache
        self.kv_seqlens = None


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



def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats, dim=dim)
    values = torch.repeate_interleave(values, repeats, dim=dim)
    return keys, values


class Attention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.n_kv_heads = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim)
            val = val.view(seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim)

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(
            xq, key, val, None if cache is None else cache.mask
        )

        return self.wo(output.view(seqlen_sum, self.n_heads * self.head_dim))

class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = nn.functional.silu(self.w1(x)) * self.w3(x)
        return self.w2(x_)


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

        self.feed_forward: nn.Module = FeedForward(args=args)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out



class Transformer(nn.Module):

    def __init__(self, args: ModelArgs, pipeline_rank: int = 0, num_pipeline_ranks: int =1):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        assert self.vocab_size > 0, "Vocab size must be set"


        # Initialize all layers
        layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]

    def forward(self, input_ids: torch.Tensor, seqlens: List[int], cache: Optional[RotatingBufferCache] = None):
        pass
