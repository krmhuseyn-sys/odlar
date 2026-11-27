"""Modeli brauzer üçün hazırlayır: ONNX + int8 sıxılma + tokenizer JSON.

    python -m odlar.export runs/odlar2 --name odlar2 --to web/

Nəticə: web/odlar2.onnx (int8) və web/tok_odlar2.json. Sonra docs/models.js-ə bir sətir əlavə et.
Defolt olaraq KV-keşli model ixrac olunur (R1): girişlər input_ids və past, çıxışlar logits və
present. Hər yeni token üçün yalnız o token hesablanır, bütün kontekst yox. docs/generate.js hər iki
formatı tanıyır; --no-kv köhnə formatı (Odlar-1, 1.5) verir.
Yoxlama: int8 modelin keyfiyyəti (BPC) PyTorch-dakı ilə müqayisə olunur.
"""
import argparse
import json
import os
import tempfile

import numpy as np
import torch

from .generate import load


class _Cached(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, past):
        return self.m.forward_cached(input_ids, past)


class _Logits(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return self.m(input_ids)[0][:, -1, :]     # sayt yalnız son tokenin ehtimallarını istəyir


def tokenizer_json(tok_path, out_path):
    """tokenizers kitabxanasının faylından JS tokenizerin istifadə etdiyi sadə format: {vocab, merges}."""
    with open(tok_path, encoding="utf-8") as f:
        d = json.load(f)
    vocab_map = d["model"]["vocab"]
    vocab = [None] * len(vocab_map)
    for t, i in vocab_map.items():
        vocab[i] = t
    merges = [m if isinstance(m, str) else " ".join(m) for m in d["model"]["merges"]]
    out = {"vocab": vocab, "merges": merges}
    if "Digits" in json.dumps(d.get("pre_tokenizer")):
        out["digits"] = True                       # Odlar-2: rəqəmlər ayrı-ayrı (docs/tokenizer.js)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))


def share_embedding(path, emb="m.tok_emb.weight"):
    """R5: ortaq embedding int8 faylda iki dəfə yazılır (axtarış üçün V x D, çıxış qatı üçün D x V).
    Axtarışı çıxış qatının int8 matrisinə yönəldir (Gather axis=1 + Transpose) və artıq nüsxəni silir.
    Qaytarır: qənaət olunan bayt."""
    import onnx
    m = onnx.load(path)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    eq = inits.get(emb + "_quantized")
    if eq is None:
        return 0
    V, D = eq.dims
    head = next((n for n in g.node if n.op_type == "MatMulInteger" and n.input[1] in inits
                 and list(inits[n.input[1]].dims) == [D, V]), None)
    gather = next((n for n in g.node if n.op_type == "Gather" and n.input[0] == eq.name), None)
    if head is None or gather is None:
        return 0
    hq = head.input[1]
    base = hq[:-len("_quantized")]
    deq = next(n for n in g.node if n.op_type == "DequantizeLinear" and n.input[0] == gather.output[0])
    gather.input[0] = hq
    del gather.attribute[:]
    gather.attribute.append(onnx.helper.make_attribute("axis", 1))        # (D, 1, S)
    t_out = gather.output[0] + "_t"
    tr = onnx.helper.make_node("Transpose", [gather.output[0]], [t_out], perm=[1, 2, 0], name="emb_transpose")
    g.node.insert(list(g.node).index(gather) + 1, tr)                    # (1, S, D)
    deq.input[0], deq.input[1], deq.input[2] = t_out, base + "_scale", base + "_zero_point"
    used = {x for n in g.node for x in n.input}
    saved = 0
    for name in (emb + "_quantized", emb + "_scale", emb + "_zero_point"):
        if name in inits and name not in used:
            saved += len(inits[name].raw_data)
            g.initializer.remove(inits[name])
    for vi in list(g.value_info):                                        # köhnə forma məlumatı
        if vi.name in (gather.output[0], deq.output[0]):
            g.value_info.remove(vi)
    onnx.save(m, path)
    return saved


def past_shape(model, P=0):
    c = model.config
    return (c.n_layer, 2, 1, c.n_head, P, c.n_embd // c.n_head)


def export(model_dir, name, to, quantize=True, kv=True):
    import onnx
    import onnxruntime as ort
    model, tok = load(model_dir)
    model.eval()
    os.makedirs(to, exist_ok=True)
    final = os.path.join(to, f"{name}.onnx")
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = os.path.join(tmp, "fp32.onnx")
        dummy = torch.tensor([tok.encode("Bakı şəhəri").ids])
        if kv:
            past = torch.zeros(past_shape(model, 3))
            torch.onnx.export(_Cached(model), (dummy, past), fp32, input_names=["input_ids", "past"],
                              output_names=["logits", "present"],
                              dynamic_axes={"input_ids": {1: "seq"}, "past": {4: "past_len"},
                                            "present": {4: "total_len"}},
                              opset_version=17, dynamo=False)
        else:
            torch.onnx.export(_Logits(model), (dummy,), fp32, input_names=["input_ids"],
                              output_names=["logits"], dynamic_axes={"input_ids": {1: "seq"}},
                              opset_version=17, dynamo=False)
        onnx.checker.check_model(fp32)
        if quantize:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            from onnxruntime.quantization.shape_inference import quant_pre_process
            pre = os.path.join(tmp, "pre.onnx")
            quant_pre_process(fp32, pre, skip_symbolic_shape=True)
            quantize_dynamic(pre, final, weight_type=QuantType.QInt8)
            saved = share_embedding(final)
            if saved:
                print(f"  ortaq embedding bir dəfə yazıldı: {saved/1e6:.1f} MB qənaət (R5)")
        else:
            os.replace(fp32, final)

    # yoxlama: eyni mətndə PyTorch və ONNX-in keyfiyyəti (simvol başına bit) nə qədər fərqlənir
    text = "Azərbaycan Respublikası Cənubi Qafqazda yerləşən dövlətdir. Paytaxtı Bakı şəhəridir."
    ids = [tok.token_to_id("<|endoftext|>")] + tok.encode(text).ids
    sess = ort.InferenceSession(final)
    model.eval()        # torch.onnx.export sarğı modulun rejimini bərpa edəndə modeli train-ə qaytarır
    with torch.no_grad():
        ref = model(torch.tensor([ids[:-1]]))[0][0].double().numpy()
    if kv:      # brauzerdəki kimi: əvvəl bir token, sonra hər addımda bir token və keş
        past, rows = np.zeros(past_shape(model), dtype=np.float32), []
        for i in range(len(ids) - 1):
            lg, past = sess.run(None, {"input_ids": np.array([[ids[i]]], dtype=np.int64), "past": past})
            rows.append(lg[0])
        got = np.stack(rows).astype(np.float64)
    else:
        got = np.stack([sess.run(None, {"input_ids": np.array([ids[:i]], dtype=np.int64)})[0][0]
                        for i in range(1, len(ids))]).astype(np.float64)
    y = np.array(ids[1:])

    def bpc(lg):
        lg = lg - lg.max(-1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        return -lp[np.arange(len(y)), y].sum() / len(text) / np.log(2)
    b_ref, b_got = bpc(ref), bpc(got)
    tok_out = os.path.join(to, f"tok_{name}.json")
    tokenizer_json(os.path.join(model_dir, "tokenizer.json"), tok_out)
    mb = os.path.getsize(final) / 1e6
    print(f"{final}: {mb:.1f} MB | BPC: PyTorch {b_ref:.3f}, ONNX {b_got:.3f} (fərq {b_got-b_ref:+.3f})")
    print(f"{tok_out}: {os.path.getsize(tok_out)/1e6:.1f} MB")
    c = model.config
    kv_txt = f", kv: {{ layers: {c.n_layer}, heads: {c.n_head}, headDim: {c.n_embd // c.n_head} }}" if kv else ""
    print(f"docs/models.js üçün: blockSize: {c.block_size}{kv_txt} "
          f"({model.num_params()/1e6:.1f}M parametr, {c.arch})")
    return final, tok_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Odlar -> ONNX (brauzer üçün)")
    ap.add_argument("model_dir")
    ap.add_argument("--name", required=True, help="məs. odlar2")
    ap.add_argument("--to", default="docs")
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument("--no-kv", action="store_true", help="köhnə format: keşsiz (Odlar-1, 1.5 saytı)")
    a = ap.parse_args()
    export(a.model_dir, a.name, a.to, quantize=not a.no_quantize, kv=not a.no_kv)
