"""YAML config faylları.

Hər model bir YAML faylıdır (configs/ qovluğunda). Bir config digərinin üzərində qurula bilər:
    base: odlar15.yaml      # əvvəlcə bu yüklənir, sonra aşağıdakılar onun üstünə yazılır
Komanda sətrindən də dəyişmək olar:
    python -m odlar.train configs/odlar15.yaml --set train.max_steps=100 model.n_layer=4
"""
import copy
import os

import yaml

DEFAULTS = {
    "out_dir": "runs/{name}",           # {name} config-in adı ilə əvəz olunur
    "model": {
        "arch": "gpt2", "block_size": 256, "n_layer": 6, "n_head": 6, "n_embd": 384,
        "dropout": 0.1, "rope_theta": 10000.0, "rope_scale": 1.0,
    },
    "data": {
        "dir": "{out_dir}/data",        # train.npy, val.npy, tokenizer.json bura yazılır
        "tokenizer": None,              # hazır tokenizer.json yolu (boşdursa, yenisi öyrədilir)
        "vocab_size": 8000,
        "tok_train_chars": 300_000_000,
        "val_frac": 0.02,               # hər mənbədən bu qədər validasiyaya ayrılır
        "sources": [],
    },
    "train": {
        "batch_size": 32, "grad_accum": 1, "lr": 5e-4, "min_lr_frac": 0.1, "warmup": 300,
        "max_steps": 5000, "weight_decay": 0.1, "beta2": 0.95, "grad_clip": 1.0,
        "eval_every": 500, "eval_batches": 40, "sample_prompt": "Bakı şəhəri",
        "init_from": None,              # başqa model.pt-dən başlamaq (məs. kontekst uzatma)
        "compile": False, "seed": 1337,
        "hub_repo": None,               # məs. krmhus/odlar-2: checkpoint-lər HF Hub-a da yazılır
    },
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(s):
    v = yaml.safe_load(s)
    if isinstance(v, str):                # YAML "3e-4"-ü mətn sayır; rəqəmə çeviririk
        try:
            v = float(v)
        except ValueError:
            pass
    return v


def load_config(path, overrides=()):
    """YAML faylını oxuyur, base zəncirini birləşdirir, --set dəyişikliklərini tətbiq edir."""
    def read(p, seen=()):
        p = os.path.abspath(p)
        if p in seen:
            raise ValueError(f"config dövrə düşdü: {p}")
        with open(p, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        base = d.pop("base", None)
        if base:
            parent = read(os.path.join(os.path.dirname(p), base), seen + (p,))
            parent.pop("name", None)          # ad miras qalmır: hər config öz qovluğuna yazır
            return _merge(parent, d)
        return d

    cfg = _merge(DEFAULTS, read(path))
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    cfg = {"name": cfg.pop("name"), **cfg}
    for item in overrides:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(val)
    cfg["out_dir"] = cfg["out_dir"].format(name=cfg["name"])
    cfg["data"]["dir"] = cfg["data"]["dir"].format(out_dir=cfg["out_dir"], name=cfg["name"])
    return cfg


def dump(cfg):
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
