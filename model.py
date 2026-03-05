import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def assert_no_nan(x, name):
    if torch.isnan(x).any():
        raise ValueError(f"NaNs found in {name}")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, debug=False):
        # x: [B, T, D]
        B, T, D = x.shape

        qkv = self.qkv(x)  # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape into heads
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)

        out = attn @ v

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.proj(out)

        debug_dict = {}
        if debug:
            debug_dict["attn"] = attn
            debug_dict["scores"] = scores

        return out, debug_dict


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x, debug=False):
        attn_out, dbg = self.attn(self.ln1(x), debug=debug)
        x = x + attn_out

        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out

        return x, dbg


class TransformerDemo(nn.Module):
    def __init__(self, vocab_size=100, d_model=64, n_heads=4, d_ff=128, max_len=128):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.block = TransformerBlock(d_model, n_heads, d_ff)

        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids, debug=False):
        # token_ids: [B, T]
        B, T = token_ids.shape

        pos = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T)

        x = self.token_embedding(token_ids) + self.pos_embedding(pos)

        x, dbg = self.block(x, debug=debug)

        x = self.final_ln(x)
        logits = self.head(x)

        # invariant checks
        assert logits.shape == (B, T, self.vocab_size)

        if debug:
            assert_no_nan(x, "x")
            assert_no_nan(logits, "logits")

        return logits, dbg