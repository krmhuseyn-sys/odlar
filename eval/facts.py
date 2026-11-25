import sys, torch
sys.path.insert(0, '.')
from bpc import load
import torch.nn.functional as F
caps = [("Azərbaycanın","Bakı"),("Türkiyənin","Ankara"),("Gürcüstanın","Tiflis"),("Rusiyanın","Moskva"),("İranın","Tehran"),
("Fransanın","Paris"),("Almaniyanın","Berlin"),("İtaliyanın","Roma"),("İspaniyanın","Madrid"),("Yaponiyanın","Tokio"),
("Çinin","Pekin"),("Misirin","Qahirə"),("Ukraynanın","Kiyev"),("Qazaxıstanın","Astana"),("Özbəkistanın","Daşkənd"),
("Böyük Britaniyanın","London"),("Yunanıstanın","Afina"),("Polşanın","Varşava"),("Pakistanın","İslamabad"),("Macarıstanın","Budapeşt")]
@torch.no_grad()
def probe(m, tok, prompt, word):
    ctx = tok.encode(prompt).ids; w = tok.encode(" " + word).ids; p_tot = 1.0; rank0 = None
    for j, wid in enumerate(w):
        logits, _ = m(torch.tensor([ctx + w[:j]])); p = F.softmax(logits[0, -1], -1)
        if j == 0: rank0 = int((p > p[wid]).sum()) + 1
        p_tot *= p[wid].item()
    return rank0, p_tot
@torch.no_grad()
def greedy(m, tok, prompt, n=6):
    out = m.generate(torch.tensor([tok.encode(prompt).ids]), max_new_tokens=n, temperature=1e-5, top_k=1)
    return tok.decode(out[0].tolist())[len(prompt):].split("\n")[0].strip()
M = {"Odlar-1": load("/home/claude/llm_zip"), "Odlar-1.5": load(".")}
print(f"{'sual':28}{'cavab':10}" + "".join(f"{k:>22}" for k in M))
top1 = {k: 0 for k in M}; top10 = {k: 0 for k in M}
for c, a in caps:
    pr = f"{c} paytaxtı"; row = f"{pr:28}{a:10}"
    for k, (m, tok) in M.items():
        r, p = probe(m, tok, pr, a); top1[k] += r == 1; top10[k] += r <= 10
        row += f"{'#'+str(r)+'  '+format(p*100,'.2f')+'%':>22}"
    print(row)
print("1-ci yerdə:", top1, " ilk 10-da:", top10)
print()
for t in ["2 + 2 =", "5 + 3 =", "1 + 1 = 2, 2 + 2 = 4, 3 + 3 =", "Bir həftədə yeddi", "Azərbaycanın paytaxtı", "Sual: Azərbaycanın paytaxtı hansı şəhərdir? Cavab:"]:
    print(f"{t!r:52}" + " | ".join(f"{k}: {greedy(m, tok, t)!r}" for k, (m, tok) in M.items()))
