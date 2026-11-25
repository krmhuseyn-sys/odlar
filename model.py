"""Kiçik GPT modeli (nanoGPT üslubunda). Həm Kaggle-da təlim, həm də lokal sayt bunu istifadə edir."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 8000   # tokenizerdəki "söz parçaları"nın sayı
    block_size: int = 256    # model bir dəfəyə neçə tokenə baxır (kontekst uzunluğu)
    n_layer: int = 6         # transformer blok sayı
    n_head: int = 6          # attention "baş"larının sayı
    n_embd: int = 384        # hər tokenin vektor ölçüsü
    dropout: float = 0.1


class Block(nn.Module):
    """Bir transformer bloku: əvvəl attention (tokenlər bir-birinə baxır), sonra MLP (hər token özü düşünür)."""

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

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # is_causal=True: hər token yalnız özündən əvvəlkilərə baxa bilir (gələcəyi görmür)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.config = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # giriş və çıxış çəkiləri ortaqdır (parametrə qənaət)
        self.apply(self._init)

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
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=40, stop_id=None,
                 repetition_penalty=1.0):
        """Hər addımda növbəti tokeni ehtimala görə seçib sona əlavə edir.
        repetition_penalty > 1 artıq yazılmış tokenlərin ehtimalını azaldır (təkrarın qarşısını alır)."""
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
