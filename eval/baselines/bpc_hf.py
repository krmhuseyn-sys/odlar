import json, math, sys, time, gc, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
texts = json.load(open("../eval_texts.json", encoding="utf-8"))
MODELS = sys.argv[1:]
out = {}
for name in MODELS:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.bfloat16 if "1.3B" in name else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).eval()
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    nparams = sum(p.numel() for p in model.parameters())
    res, A, C, W = {}, 0.0, 0, 0
    for style, ts in texts.items():
        a = c = 0
        for t in ts:
            ids = tok(t, add_special_tokens=False)["input_ids"]
            x = torch.tensor([[start] + ids])
            with torch.no_grad():
                lg = model(x).logits[0, :-1].float()
            a += torch.nn.functional.cross_entropy(lg, x[0, 1:], reduction="sum").item()
            c += len(t); W += len(ids)
        res[style] = round(a / c / math.log(2), 4); A += a; C += c
    res["cəmi"] = round(A / C / math.log(2), 4)
    words = sum(len(t.split()) for ts in texts.values() for t in ts)
    out[name] = {"params_M": round(nparams / 1e6, 1), "bpc": res, "tokens_per_word": round(W / words, 2)}
    print(name, out[name], f"{time.time()-t0:.0f}s", flush=True)
    del model; gc.collect()
json.dump(out, open("results_" + str(abs(hash(tuple(MODELS))) % 10**6) + ".json", "w"), ensure_ascii=False, indent=1)
