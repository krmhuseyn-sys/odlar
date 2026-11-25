"""Mətn yazdırmaq (terminaldan).

    python -m odlar.generate runs/odlar15 "Xəzər dənizi"
    python -m odlar.generate runs/odlar15 "Qədim zamanlarda" -n 3 --tokens 150 --temp 0.7
Qovluqda model.pt və tokenizer.json olmalıdır.
"""
import argparse
import os

import torch
from tokenizers import Tokenizer

from .model import load_checkpoint


def load(model_dir, device="cpu", **overrides):
    model = load_checkpoint(os.path.join(model_dir, "model.pt"), device, **overrides)
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    return model, tok


def write(model, tok, prompt, tokens=100, temp=0.8, top_k=40, rep=1.3, device="cpu"):
    eot = tok.token_to_id("<|endoftext|>")
    idx = torch.tensor([tok.encode(prompt).ids[-model.config.block_size:]], device=device)
    out = model.generate(idx, max_new_tokens=tokens, temperature=temp, top_k=top_k,
                         stop_id=eot, repetition_penalty=rep)
    return tok.decode(out[0].tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Odlar ilə mətn yazdır")
    ap.add_argument("model_dir")
    ap.add_argument("prompt")
    ap.add_argument("-n", type=int, default=1, help="neçə variant")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--rep", type=float, default=1.3, help="təkrar cəzası")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if a.seed is not None:
        torch.manual_seed(a.seed)
    m, t = load(a.model_dir)
    for i in range(a.n):
        print(write(m, t, a.prompt, a.tokens, a.temp, a.top_k, a.rep))
        if i < a.n - 1:
            print("-" * 40)
