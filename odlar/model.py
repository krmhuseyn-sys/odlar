"""Odlar modelləri: bir faylda iki quruluş.

arch="gpt2"  - Odlar-1 və Odlar-1.5-in quruluşu: LayerNorm, öyrənilən mövqe vektorları, GELU.
               Köhnə model.pt faylları bu quruluşla dəyişmədən yüklənir.
arch="llama" - Odlar-2 üçün: RMSNorm, RoPE (fırlanan mövqe kodlaması), SwiGLU, bias yoxdur.
               RoPE sayəsində kontekst sonradan uzadıla bilər (rope_scale).
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8000    # tokenizerdəki "söz parçaları"nın sayı
    block_size: int = 256     # model bir dəfəyə neçə tokenə baxır (kontekst uzunluğu)
    n_layer: int = 6          # transformer blok sayı
    n_head: int = 6           # attention "baş"larının sayı
    n_embd: int = 384         # hər tokenin vektor ölçüsü
    dropout: float = 0.1
    arch: str = "gpt2"        # "gpt2" və ya "llama"
    rope_theta: float = 10000.0
    rope_scale: float = 1.0   # >1: kontekst uzadılması (mövqelər bu qədər sıxılır)
    norm_eps: float = 1e-5

    @classmethod
    def from_dict(cls, d):
        """Köhnə və yeni config-ləri qəbul edir; tanınmayan açarları atır."""
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- köməkçi hissələr
class RMSNorm(nn.Module):
    """LayerNorm-un sadə variantı: yalnız ölçünü normallaşdırır, orta qiyməti çıxmır."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cache(head_dim, max_len, theta=10000.0, scale=1.0):
    """RoPE üçün cos/sin cədvəli. scale>1 olanda mövqelər sıxılır (position interpolation)."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_len).float() / scale
    ang = torch.outer(pos, inv)                       # (T, head_dim/2)
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    """x: (B, H, T, D). Hər cüt ədədi mövqeyə görə fırladır."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


# ---------------------------------------------------------------- bloklar
class Block(nn.Module):
    """GPT-2 bloku. Parametr adları Odlar-1/1.5 checkpoint-ləri ilə eynidir, dəyişmək olmaz."""

    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_head = c.n_head
        self.dropout = c.dropout
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(c.n_embd, 4 * c.n_embd),
            nn.GELU(),
            nn.Linear(4 * c.n_embd, c.n_embd),
            nn.Dropout(c.dropout),
        )
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, rope=None):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))
        x = x + self.mlp(self.ln2(x))
        return x


class LlamaBlock(nn.Module):
    """Llama bloku: RMSNorm + RoPE attention + SwiGLU MLP, bias yoxdur."""

    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_head = c.n_head
        self.dropout = c.dropout
        hidden = int(8 * c.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)          # GPU üçün 64-ə bölünən ölçü
        self.norm1 = RMSNorm(c.n_embd, c.norm_eps)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.norm2 = RMSNorm(c.n_embd, c.norm_eps)
        self.gate = nn.Linear(c.n_embd, hidden, bias=False)
        self.up = nn.Linear(c.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, c.n_embd, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, rope=None):
        B, T, C = x.shape
        cos, sin = rope
        q, k, v = self.qkv(self.norm1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))
        h = self.norm2(x)
        x = x + self.drop(self.down(F.silu(self.gate(h)) * self.up(h)))
        return x


# ---------------------------------------------------------------- model
class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        assert c.arch in ("gpt2", "llama"), f"naməlum arch: {c.arch}"
        assert c.n_embd % c.n_head == 0
        self.config = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        if c.arch == "gpt2":
            self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
            self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
            self.ln_f = nn.LayerNorm(c.n_embd)
        else:
            self.blocks = nn.ModuleList([LlamaBlock(c) for _ in range(c.n_layer)])
            self.ln_f = RMSNorm(c.n_embd, c.norm_eps)
            cos, sin = rope_cache(c.n_embd // c.n_head, c.block_size, c.rope_theta, c.rope_scale)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # giriş və çıxış çəkiləri ortaqdır
        self.apply(self._init)
        # Dərin şəbəkələrdə sabitlik üçün: qalıq yola yazan qatlar daha kiçik başlayır
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "down.weight", "mlp.2.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        rope = None
        if self.config.arch == "gpt2":
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        else:
            rope = (self.rope_cos[:T], self.rope_sin[:T])
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, rope)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=40, stop_id=None,
                 repetition_penalty=1.0):
        """Hər addımda növbəti tokeni ehtimala görə seçib sona əlavə edir.
        repetition_penalty > 1 artıq yazılmış tokenlərin ehtimalını azaldır."""
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.config.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :]
            if repetition_penalty != 1.0:
                seen = torch.unique(ctx[0])
                v = logits[0, seen]
                logits[0, seen] = torch.where(v > 0, v / repetition_penalty, v * repetition_penalty)
            logits = logits / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            if stop_id is not None and nxt.item() == stop_id:
                break
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def load_checkpoint(path, device="cpu", **overrides):
    """model.pt-ni yükləyir. overrides: məsələn block_size=4096, rope_scale=4 (kontekst uzatmaq üçün)."""
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = GPTConfig.from_dict({**ck["config"], **overrides})
    model = GPT(cfg)
    sd = ck["model"]
    if cfg.arch == "gpt2" and sd["pos_emb.weight"].shape[0] != cfg.block_size:
        raise ValueError("gpt2 quruluşunda kontekst uzadıla bilməz (öyrənilən mövqe vektorları)")
    model.load_state_dict(sd)
    return model.float().to(device).eval()
