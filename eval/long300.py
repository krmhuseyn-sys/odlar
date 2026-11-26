"""Uzun mətn testi: hər modeldən 10 başlanğıca 300 tokenlik mətn (EOT qadağan, sabit seed).
    python eval/long300.py runs/odlar2 [runs/odlar15 ...]
Mətnlər eval/long300_<ad>.json faylına yazılır; qiymətləndirmə results/uzun_metn_300.md-dəki şkala ilə."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from odlar.generate import load

PROMPTS = ["Qış gecəsi idi", "Qədim zamanlarda", "Xəzər dənizi", "Azərbaycan mətbəxi", "Kür çayı",
           "Kəndimizin kənarında", "Nənəm mənə danışırdı ki", "Bahar gələndə", "Məktəbin ilk günü", "Dağların başında"]


@torch.no_grad()
def write300(m, t, prompt, seed, n=300, temp=0.8, top_k=40, rep=1.3):
    torch.manual_seed(seed)
    eot = t.token_to_id("<|endoftext|>")
    idx = torch.tensor([t.encode(prompt).ids])
    for _ in range(n):
        ctx = idx[:, -m.config.block_size:]
        lg = m(ctx)[0][:, -1, :].clone()
        lg[0, eot] = -float("inf")                       # model düz n token yazmalıdır
        seen = torch.unique(ctx[0]); v = lg[0, seen]
        lg[0, seen] = torch.where(v > 0, v / rep, v * rep)
        lg = lg / temp
        kv, _ = torch.topk(lg, top_k); lg[lg < kv[:, [-1]]] = -float("inf")
        idx = torch.cat([idx, torch.multinomial(torch.softmax(lg, -1), 1)], 1)
    return t.decode(idx[0].tolist())


if __name__ == "__main__":
    for d in sys.argv[1:]:
        m, t = load(d)
        name = os.path.basename(os.path.normpath(d))
        res = [{"prompt": p, "seed": 2000 + i, "text": write300(m, t, p, 2000 + i)} for i, p in enumerate(PROMPTS)]
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"long300_{name}.json")
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("yazıldı:", out)
