import json, math, gc, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
P = json.load(open("belebele_az_passages.json", encoding="utf-8")); C = sum(map(len, P))
name = "ai-forever/mGPT-1.3B-azerbaijan"
tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).eval()
s = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
a = 0.0; t0 = time.time()
for i, p in enumerate(P):
    x = torch.tensor([[s] + tok(p, add_special_tokens=False)["input_ids"]])
    with torch.no_grad():
        lg = m(x).logits[0, :-1].float()
    a += torch.nn.functional.cross_entropy(lg, x[0, 1:], reduction="sum").item()
    if i % 100 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)
r = {"Odlar-1": [1.5295, 13.8], "Odlar-1.5": [1.3939, 40.0], "HuggingFaceTB/SmolLM2-135M": [3.4237, 134.5],
     "Qwen/Qwen3-0.6B-Base": [2.036, 596.0], name: [round(a / C / math.log(2), 4), round(sum(p.numel() for p in m.parameters()) / 1e6, 1)]}
json.dump(r, open("belebele_bpc.json", "w"), ensure_ascii=False, indent=1); print(r[name], flush=True)
