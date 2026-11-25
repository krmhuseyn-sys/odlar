"""Tokenizer səmərəliliyi, generasiyada təkrar (distinct-n), nümunədən öyrənmə və CPU sürəti."""
import sys, os, json, re, time, random, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bpc import load, MODELS
M = {n: load(d) for n, (d, _) in MODELS.items()}
texts = [t for v in json.load(open("eval_texts.json", encoding="utf-8")).values() for t in v]
out = {}
# 1) Tokenizer: söz başına token
words = [w for t in texts for w in re.findall(r"\w+", t)]
print("1) Tokenizer: söz başına orta token sayı (fertility)")
for n, (m, tok) in M.items():
    f = sum(len(tok.encode(" " + w).ids) for w in words) / len(words)
    ex = {w: tok.encode(" " + w).tokens for w in ["Azərbaycanın", "universitetə", "gedəcəyik", "kitablarımızdan", "proqramlaşdırma"]}
    out.setdefault(n, {})["fertility"] = f
    print(f"   {n}: {f:.2f} token/söz;", " | ".join(f"{w}={len(t)}" for w, t in ex.items()))
# 2) Təkrar: distinct-2 (unikal söz cütlərinin payı) və 4-qram təkrarı
prompts = ["Qədim zamanlarda", "Xəzər dənizi", "Azərbaycan mətbəxi", "Qış gecəsi idi", "Kür çayı", "Futbol üzrə"]
def gen(m, tok, p, rp, seed):
    torch.manual_seed(seed)
    o = m.generate(torch.tensor([tok.encode(p).ids]), max_new_tokens=100, temperature=0.8, top_k=40, repetition_penalty=rp)
    return tok.decode(o[0].tolist())[len(p):]
def distinct(ws, n): g = [tuple(ws[i:i+n]) for i in range(len(ws)-n+1)]; return len(set(g)) / max(1, len(g))
print("2) Təkrar ölçüsü: distinct-2 (yüksək = az təkrar), 6 prompt x 3 nümunə x 100 token")
for n, (m, tok) in M.items():
    for rp in (1.0, 1.3):
        d2 = []
        for p in prompts:
            for s in range(3):
                ws = re.findall(r"\w+", gen(m, tok, p, rp, s).lower()); d2.append(distinct(ws, 2))
        out[n][f"distinct2_rp{rp}"] = sum(d2) / len(d2)
        print(f"   {n}, təkrar cəzası {rp}: distinct-2 = {sum(d2)/len(d2):.3f}")
# 3) Nümunədən öyrənmə (few-shot): ölkə -> paytaxt, 0 və 3 nümunə ilə
caps = [("Türkiyə","Ankara"),("Rusiya","Moskva"),("İran","Tehran"),("Fransa","Paris"),("Yaponiya","Tokio"),("Ukrayna","Kiyev"),("Almaniya","Berlin"),("İtaliya","Roma"),("Misir","Qahirə"),("Gürcüstan","Tiflis")]
shots = "Azərbaycan - Bakı\nQazaxıstan - Astana\nÖzbəkistan - Daşkənd\n"
@torch.no_grad()
def first_ok(m, tok, prompt, ans):
    o = m.generate(torch.tensor([tok.encode(prompt).ids]), max_new_tokens=4, temperature=1e-5, top_k=1)
    return tok.decode(o[0].tolist())[len(prompt):].strip().startswith(ans)
print("3) Nümunədən öyrənmə: 'Ölkə - ' -> paytaxt, 10 ölkə (greedy)")
for n, (m, tok) in M.items():
    z = sum(first_ok(m, tok, f"{c} - ", a) for c, a in caps); f = sum(first_ok(m, tok, shots + f"{c} - ", a) for c, a in caps)
    out[n]["fewshot"] = {"0-shot": z, "3-shot": f}
    print(f"   {n}: nümunəsiz {z}/10, 3 nümunə ilə {f}/10")
# 4) CPU sürəti (bu serverdə, 1 thread, PyTorch fp32)
torch.set_num_threads(1)
print("4) CPU sürəti (1 thread, fp32, 100 token)")
for n, (m, tok) in M.items():
    t = time.time(); gen(m, tok, "Qədim zamanlarda", 1.3, 0); s = time.time() - t
    out[n]["cpu_tok_per_s"] = 100 / s; print(f"   {n}: {100/s:.1f} token/san")
json.dump(out, open("results_analysis.json", "w"), ensure_ascii=False, indent=1)
