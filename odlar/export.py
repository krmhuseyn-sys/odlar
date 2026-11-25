"""Modeli brauzer üçün hazırlayır: ONNX + int8 sıxılma + tokenizer JSON.

    python -m odlar.export runs/odlar15 --name odlar15 --to docs/

Nəticə: docs/odlar15.onnx (int8) və docs/tok_odlar15.json. Sonra docs/models.js-ə bir sətir əlavə et.
Yoxlama: int8 modelin cavabı PyTorch-dakı ilə müqayisə olunur.
"""
import argparse
import json
import os
import tempfile

import numpy as np
import torch

from .generate import load


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
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "merges": merges}, f, ensure_ascii=False, separators=(",", ":"))


def export(model_dir, name, to, quantize=True):
    import onnx
    import onnxruntime as ort
    model, tok = load(model_dir)
    model.eval()
    os.makedirs(to, exist_ok=True)
    final = os.path.join(to, f"{name}.onnx")
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = os.path.join(tmp, "fp32.onnx")
        dummy = torch.tensor([tok.encode("Bakı şəhəri").ids])
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
        else:
            os.replace(fp32, final)

    # yoxlama: eyni mətndə PyTorch və ONNX-in keyfiyyəti (simvol başına bit) nə qədər fərqlənir
    text = "Azərbaycan Respublikası Cənubi Qafqazda yerləşən dövlətdir. Paytaxtı Bakı şəhəridir."
    ids = [tok.token_to_id("<|endoftext|>")] + tok.encode(text).ids
    sess = ort.InferenceSession(final)
    with torch.no_grad():
        ref = model(torch.tensor([ids[:-1]]))[0][0].double().numpy()
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
    print(f"docs/models.js üçün: blockSize {model.config.block_size}, "
          f"{model.num_params()/1e6:.1f}M parametr, {model.config.arch}")
    return final, tok_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Odlar -> ONNX (brauzer üçün)")
    ap.add_argument("model_dir")
    ap.add_argument("--name", required=True, help="məs. odlar2")
    ap.add_argument("--to", default="docs")
    ap.add_argument("--no-quantize", action="store_true")
    a = ap.parse_args()
    export(a.model_dir, a.name, a.to, quantize=not a.no_quantize)
