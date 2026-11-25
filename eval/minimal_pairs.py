"""Qrammatika testi: hər cütdə model düzgün cümləyə səhvdən yüksək ehtimal verirmi? Təsadüfi səviyyə 50%."""
import json, torch
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bpc import load, MODELS
@torch.no_grad()
def logp(m, tok, s):
    eot = tok.token_to_id("<|endoftext|>"); ids = [eot] + tok.encode(s).ids
    logits, _ = m(torch.tensor([ids[:-1]]))
    return -torch.nn.functional.cross_entropy(logits[0], torch.tensor(ids[1:]), reduction="sum").item()
pairs = json.load(open("minimal_pairs.json", encoding="utf-8"))
M = {n: load(d) for n, (d, _) in MODELS.items()}
res = {}; fails = {n: [] for n in M}
print(f"{'kateqoriya':46}" + "".join(f"{n:>12}" for n in M))
for cat, ps in pairs.items():
    row = f"{cat:46}"
    for n, (m, tok) in M.items():
        ok = 0
        for g, b in ps:
            if logp(m, tok, g) > logp(m, tok, b): ok += 1
            else: fails[n].append(g)
        res.setdefault(n, {})[cat] = ok / len(ps); row += f"{str(ok)+'/'+str(len(ps)):>12}"
    print(row)
tot = {n: sum(v * 8 for v in res[n].values()) for n in res}
print(f"{'CƏMİ (40 cüt)':46}" + "".join(f"{str(int(tot[n]))+'/40':>12}" for n in res))
for n in fails: print(n, "səhv etdiyi:", fails[n])
json.dump({"accuracy": res, "fails": fails}, open("results_minimal_pairs.json", "w"), ensure_ascii=False, indent=1)
