"""Odlar-ın "beyninə" baxmaq: növbəti söz parçası üçün hansı ehtimalları verir?

İstifadə:
    python ehtimal.py "Azərbaycanın paytaxtı" Bakı
    python ehtimal.py "Nizami Gəncəvi" şair --top 20

Nəticə: terminalda cədvəl + ehtimal_<söz>.png qrafiki.
"""
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import GPT, GPTConfig


def load(model_path="model.pt", tok_path="tokenizer.json"):
    ck = torch.load(model_path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, Tokenizer.from_file(tok_path)


@torch.no_grad()
def next_probs(model, tok, prompt):
    """Prompt-dan sonra gələ biləcək hər tokenin ehtimalı (8000 ədəd, cəmi = 1)."""
    ids = tok.encode(prompt).ids[-model.config.block_size:]
    logits, _ = model(torch.tensor([ids]))
    return F.softmax(logits[0, -1], dim=-1)


def top_tokens(model, tok, prompt, k=15):
    p = next_probs(model, tok, prompt)
    vals, idx = torch.topk(p, k)
    return [(tok.decode([i]), v) for i, v in zip(idx.tolist(), vals.tolist())]


@torch.no_grad()
def word_report(model, tok, prompt, word):
    """Sözün ehtimalı: söz bir neçə parçadan ibarətdirsə, parçaların ehtimalları vurulur."""
    word_ids = tok.encode(" " + word).ids
    ctx = tok.encode(prompt).ids
    pieces, total = [], 1.0
    for j, wid in enumerate(word_ids):
        p = next_probs(model, tok, tok.decode(ctx + word_ids[:j]))
        rank = int((p > p[wid]).sum().item()) + 1
        pieces.append((tok.decode([wid]), p[wid].item(), rank))
        total *= p[wid].item()
    return pieces, total


def sample_answers(model, tok, prompt, n=5, max_new=12, temperature=0.8):
    eot = tok.token_to_id("<|endoftext|>")
    outs = []
    for _ in range(n):
        idx = torch.tensor([tok.encode(prompt).ids])
        out = model.generate(idx, max_new_tokens=max_new, temperature=temperature,
                             top_k=40, stop_id=eot, repetition_penalty=1.3)
        outs.append(tok.decode(out[0].tolist())[len(prompt):].split("\n")[0].strip())
    return outs


def chart(prompt, top, word, word_rank, word_p, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [t.strip() or repr(t) for t, _ in top][::-1]
    vals = [v * 100 for _, v in top][::-1]
    colors = ["#f47230" if l == word else "#257de0" for l in labels]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(top) + 1.6), dpi=160)
    bars = ax.barh(labels, vals, color=colors, height=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=8, color="#191919")
    ax.set_title(f'"{prompt} ..." – növbəti söz üçün Odlar-ın təxminləri',
                 fontsize=11, color="#191919", loc="left")
    note = f'"{word}": {word_rank}-ci yer, ehtimal {word_p * 100:.2f}%'
    ax.set_xlabel(f"Ehtimal (%)      |      {note}", color="#666666", fontsize=9)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#cccccc")
    ax.tick_params(colors="#666666", labelsize=9)
    ax.set_xlim(0, max(vals) * 1.15)
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("word", help="yoxlamaq istədiyin düzgün cavab, məsələn Bakı")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    model, tok = load()
    top = top_tokens(model, tok, a.prompt, a.top)
    print(f'\n"{a.prompt} ..." – Odlar növbəti parça üçün ən çox bunlara mərc edir:\n')
    for i, (t, v) in enumerate(top, 1):
        mark = "  <--" if t.strip() == a.word else ""
        print(f"{i:>3}. {t!r:<18} {v * 100:6.2f}%{mark}")

    pieces, total = word_report(model, tok, a.prompt, a.word)
    print(f'\n"{a.word}" sözü {len(pieces)} parçadan qurulur:')
    for t, p, r in pieces:
        print(f"   {t!r:<14} ehtimal {p * 100:6.2f}%   ({r}-ci yer)")
    print(f'Bütöv "{a.word}" sözünün ehtimalı: {total * 100:.3f}%')

    print(f'\nOdlar-dan 5 dəfə soruşduq, "{a.prompt}" ...')
    for s in sample_answers(model, tok, a.prompt):
        print("   ->", s)

    path = f"ehtimal_{a.word}.png"
    chart(a.prompt, top, a.word, pieces[0][2], total, path)
    print(f"\nQrafik saxlanıldı: {path}")


if __name__ == "__main__":
    main()
