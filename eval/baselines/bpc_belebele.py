import json, math, sys, gc, time, torch
sys.path.insert(0, "../..")
P = json.load(open("belebele_az_passages.json", encoding="utf-8"))
C = sum(map(len, P)); out = {}
def odlar(name, d):
    from odlar.generate import load
    from odlar.evaluate import nll
    m, t = load(d); eot = t.token_to_id("<|endoftext|>")
    a = sum(nll(m, [eot] + t.encode(p).ids) for p in P)
    return round(a / C / math.log(2), 4), round(m.num_params() / 1e6, 1)
def hf(name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if "1.3B" in name else torch.float32).eval()
    s = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    a = 0.0
    for p in P:
        x = torch.tensor([[s] + tok(p, add_special_tokens=False)["input_ids"]])
        with torch.no_grad():
            lg = m(x).logits[0, :-1].float()
        a += torch.nn.functional.cross_entropy(lg, x[0, 1:], reduction="sum").item()
    r = (round(a / C / math.log(2), 4), round(sum(p.numel() for p in m.parameters()) / 1e6, 1))
    del m; gc.collect(); return r
for n, d in [("Odlar-1", "../../runs/odlar1"), ("Odlar-1.5", "../../runs/odlar15")]:
    out[n] = odlar(n, d); print(n, out[n], flush=True)
for n in ["HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen3-0.6B-Base", "ai-forever/mGPT-1.3B-azerbaijan"]:
    t0 = time.time(); out[n] = hf(n); print(n, out[n], f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open("belebele_bpc.json", "w"), ensure_ascii=False, indent=1)
