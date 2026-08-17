import torch.nn as nn
import torch
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph

allow_ops_in_compiled_graph()


class RotaryPositionalEmbeddings(nn.Module):
    # RoPE implementation sourced from https://github.com/aju22/RoPE-PyTorch

    def __init__(self, d: int, base: int = 10_000):

        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):

        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        seq_len = x.shape[0]

        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(
            x.device
        )  # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)

        seq_idx = (
            torch.arange(seq_len, device=x.device).float().to(x.device)
        )  # Position Index -> [0,1,2...seq-1]

        idx_theta = torch.einsum(
            "n,d->nd", seq_idx, theta
        )  # Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]

        idx_theta2 = torch.cat(
            [idx_theta, idx_theta], dim=1
        )  # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]

        self.cos_cached = idx_theta2.cos()[
            :, None, None, :
        ]  # Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d]
        self.sin_cached = idx_theta2.sin()[
            :, None, None, :
        ]  # cache [sinTHETA_1, sinTHETA_2...sinTHETA_d]

    def _neg_half(self, x: torch.Tensor):

        d_2 = self.d // 2  #

        return torch.cat(
            [-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1
        )  # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]

    def forward(self, x: torch.Tensor):

        self._build_cache(x)

        neg_half_x = self._neg_half(x)

        x_rope = (x * self.cos_cached[: x.shape[0]]) + (
            neg_half_x * self.sin_cached[: x.shape[0]]
        )  # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]

        return x_rope


class SelfAttention(torch.nn.Module):
    def __init__(self, dim, nheads, dropout=0.1):
        super().__init__()

        self.nheads = nheads
        self.dropout = dropout

        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.out_proj = torch.nn.Linear(dim, dim)

    def forward(self, x):
        # assume input is shape B x L x D
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.nheads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.nheads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.nheads)

        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=(self.dropout if self.training else 0.0)
        )
        output = rearrange(output, "b h l d -> b l (h d)")
        output = self.out_proj(output)

        return output


class RoPESelfAttention(torch.nn.Module):
    def __init__(self, dim, nheads, dropout=0.1):
        super().__init__()

        self.nheads = nheads
        self.dropout = dropout

        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.out_proj = torch.nn.Linear(dim, dim)

        self.rope = RotaryPositionalEmbeddings(dim // nheads)

    def forward(self, x):
        # assume input is shape B x L x D
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b l (h d) -> l b h d", h=self.nheads)
        k = rearrange(k, "b l (h d) -> l b h d", h=self.nheads)

        q = self.rope(q)
        k = self.rope(k)

        q = rearrange(q, "l b h d -> b h l d")
        k = rearrange(k, "l b h d -> b h l d")

        v = rearrange(v, "b l (h d) -> b h l d", h=self.nheads)

        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=(self.dropout if self.training else 0.0)
        )
        output = rearrange(output, "b h l d -> b l (h d)")
        output = self.out_proj(output)

        return output


class SwiGLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.silu = torch.nn.SiLU()

    def forward(self, x):
        x, v = x.chunk(2, dim=1)
        return self.silu(x) * v
