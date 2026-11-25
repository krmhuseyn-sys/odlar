import sys, json, math, torch
sys.path.insert(0, '.')
from model import GPT, GPTConfig
from tokenizers import Tokenizer
def load(d):
    ck = torch.load(f"{d}/model.pt", map_location="cpu", weights_only=True)
    m = GPT(GPTConfig(**ck["config"])); m.load_state_dict(ck["model"]); m.eval()
    return m, Tokenizer.from_file(f"{d}/tokenizer.json")
@torch.no_grad()
def bpc(m, tok, text):
    ids = tok.encode(text).ids; eot = tok.token_to_id("<|endoftext|>")
    ids = [eot] + ids
    x = torch.tensor([ids[:-1]]); y = torch.tensor([ids[1:]])
    logits, _ = m(x)
    nll = torch.nn.functional.cross_entropy(logits[0], y[0], reduction="sum").item()
    return nll, len(text), len(ids) - 1
texts = json.load(open("eval_texts.json", encoding="utf-8"))
models = {"Odlar-1 (14M)": load("/home/claude/llm_zip"), "Odlar-1.5 (40M)": load(".")}
print(f"{'üslub':14}" + "".join(f"{k:>20}" for k in models))
tot = {k: [0, 0, 0] for k in models}
for style, ts in texts.items():
    row = f"{style:14}"
    for k, (m, tok) in models.items():
        n = c = t = 0
        for s in ts:
            a, b, cc = bpc(m, tok, s); n += a; c += b; t += cc
        tot[k][0] += n; tot[k][1] += c; tot[k][2] += t
        row += f"{n / c / math.log(2):>20.3f}"
    print(row)
print(f"{'CƏMİ':14}" + "".join(f"{tot[k][0] / tot[k][1] / math.log(2):>20.3f}" for k in models))
print("simvol/token:", {k: round(tot[k][1] / tot[k][2], 2) for k in models})
