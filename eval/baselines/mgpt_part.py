import json, math, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
lo, hi = int(sys.argv[1]), int(sys.argv[2])
P = json.load(open("belebele_az_passages.json", encoding="utf-8"))[lo:hi]
name = "ai-forever/mGPT-1.3B-azerbaijan"
tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).eval()
s = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
a = 0.0
with torch.inference_mode():
    for p in P:
        x = torch.tensor([[s] + tok(p, add_special_tokens=False)["input_ids"]])
        lg = m(x).logits[0, :-1]
        a += torch.nn.functional.cross_entropy(lg.float(), x[0, 1:], reduction="sum").item()
        del lg
json.dump({"lo": lo, "hi": hi, "nll": a, "chars": sum(map(len, P))}, open(f"mgpt_part_{lo}.json", "w"))
print(lo, hi, "ok", flush=True)
