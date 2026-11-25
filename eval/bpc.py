"""Simvol başına bit (bits-per-character): lüğətdən asılı olmayan müqayisə. fp32 (PyTorch) və int8 (ONNX)."""
import sys, json, math, torch, numpy as np
sys.path.insert(0, '/home/claude/odlar')
from model import GPT, GPTConfig
from tokenizers import Tokenizer
MODELS = {"Odlar-1": ("/home/claude/llm_zip", "/home/claude/web/site/odlar1.onnx"),
          "Odlar-1.5": ("/home/claude/o15", "/home/claude/web/site/odlar15.onnx")}
def load(d):
    ck = torch.load(f"{d}/model.pt", map_location="cpu", weights_only=True)
    m = GPT(GPTConfig(**ck["config"])); m.load_state_dict(ck["model"]); m.eval()
    return m, Tokenizer.from_file(f"{d}/tokenizer.json")
@torch.no_grad()
def nll_torch(m, ids):
    logits, _ = m(torch.tensor([ids[:-1]]))
    return torch.nn.functional.cross_entropy(logits[0], torch.tensor(ids[1:]), reduction="sum").item()
def nll_onnx(s, ids):
    t = 0.0
    for i in range(1, len(ids)):
        l = s.run(None, {"input_ids": np.array([ids[:i]], dtype=np.int64)})[0][0].astype(np.float64)
        l -= l.max(); t -= l[ids[i]] - math.log(np.exp(l).sum())
    return t
if __name__ == "__main__":
    import onnxruntime as ort
    texts = json.load(open("eval_texts.json", encoding="utf-8"))
    res = {}
    for name, (d, onnx) in MODELS.items():
        m, tok = load(d); s = ort.InferenceSession(onnx); eot = tok.token_to_id("<|endoftext|>")
        res[name] = {}
        for style, ts in texts.items():
            a = b = c = ntok = 0
            for t in ts:
                ids = [eot] + tok.encode(t).ids
                a += nll_torch(m, ids); b += nll_onnx(s, ids); c += len(t); ntok += len(ids) - 1
            res[name][style] = {"bpc_fp32": a / c / math.log(2), "bpc_int8": b / c / math.log(2), "chars": c, "tokens": ntok}
    json.dump(res, open("results_bpc.json", "w"), ensure_ascii=False, indent=1)
    print(f"{'üslub':14}" + "".join(f"{n+' fp32':>16}{n+' int8':>16}" for n in res))
    tot = {n: [0, 0, 0, 0] for n in res}
    for st in texts:
        row = f"{st:14}"
        for n in res:
            r = res[n][st]; row += f"{r['bpc_fp32']:>16.3f}{r['bpc_int8']:>16.3f}"
            tot[n][0] += r['bpc_fp32'] * r['chars']; tot[n][1] += r['bpc_int8'] * r['chars']; tot[n][2] += r['chars']; tot[n][3] += r['tokens']
        print(row)
    print(f"{'CƏMİ':14}" + "".join(f"{tot[n][0]/tot[n][2]:>16.3f}{tot[n][1]/tot[n][2]:>16.3f}" for n in res))
    print("simvol/token:", {n: round(tot[n][2] / tot[n][3], 2) for n in res})
