"""Təlim: bir config, bir komanda.

    python -m odlar.train configs/odlar15.yaml
    python -m odlar.train configs/odlar15.yaml --set out_dir=/content/drive/MyDrive/Odlar15

Nə edir:
  - data yoxdursa, əvvəlcə hazırlayır (odlar.data)
  - out_dir-də checkpoint.pt varsa, qaldığı yerdən davam edir
  - hər eval_every addımda: val loss, nümunə mətn, log.csv, model.pt (ən yaxşısı), checkpoint.pt
  - train.hub_repo yazılıbsa, faylları Hugging Face Hub-a da göndərir (HF_TOKEN lazımdır)
"""
import csv
import math
import os
import random
import shutil
import time

import numpy as np
import torch

from .config import load_config, dump
from .data import prepare
from .model import GPT, GPTConfig


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- Hugging Face Hub
class Hub:
    """İstəyə bağlı: checkpoint-ləri HF Hub-a yazır və oradan bərpa edir (Kaggle/Colab üçün)."""

    def __init__(self, repo, out_dir):
        self.repo, self.out_dir, self.api, self.future = repo, out_dir, None, None
        if not repo:
            return
        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                from google.colab import userdata      # Colab: Secrets bölməsindəki HF_TOKEN
                token = userdata.get("HF_TOKEN")
            except Exception:
                token = None
        if not token:
            try:
                from kaggle_secrets import UserSecretsClient   # Kaggle: Add-ons -> Secrets
                token = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                token = None
        if not token:
            log("  diqqət: HF_TOKEN tapılmadı, Hub-a yazılmayacaq")
            self.repo = None
            return
        from huggingface_hub import HfApi
        self.api = HfApi(token=token)
        self.api.create_repo(repo, private=True, exist_ok=True)
        log(f"  Hub: {repo} (gizli)")

    def restore(self, name):
        """Lokal fayl yoxdursa, Hub-dan endirir."""
        if not self.repo:
            return False
        from huggingface_hub import hf_hub_download
        try:
            p = hf_hub_download(self.repo, name, token=self.api.token)
            shutil.copy(p, os.path.join(self.out_dir, name))
            log(f"  Hub-dan bərpa olundu: {name}")
            return True
        except Exception:
            return False

    def push(self, with_checkpoint):
        if not self.repo:
            return
        if self.future is not None and not self.future.done():
            return                                      # əvvəlki yükləmə hələ bitməyib
        allow = ["model.pt", "tokenizer.json", "config.yaml", "log.csv", "samples.md"]
        if with_checkpoint:
            allow.append("checkpoint.pt")
        self.future = self.api.upload_folder(repo_id=self.repo, folder_path=self.out_dir,
                                             allow_patterns=allow, run_as_future=True,
                                             commit_message="checkpoint")

    def wait(self):
        if self.future is not None:
            self.future.result()


# ---------------------------------------------------------------- təlim
def train(cfg):
    tc, mc = cfg["train"], cfg["model"]
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(dump(cfg))

    random.seed(tc["seed"]); np.random.seed(tc["seed"]); torch.manual_seed(tc["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16 else torch.float16
    else:
        log("GPU yoxdur, CPU-da işləyir (yalnız kiçik sınaq üçün)")
        amp_dtype = None
    hub = Hub(tc.get("hub_repo"), out)

    # --- data
    P = prepare(cfg)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(P["tok"])
    shutil.copy(P["tok"], os.path.join(out, "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    train_data = np.load(P["train"], mmap_mode="r")
    val_data = np.load(P["val"], mmap_mode="r")
    log(f"Data: təlim {len(train_data)/1e6:.1f}M token, validasiya {len(val_data)/1e6:.2f}M token")

    # --- model
    mcfg = GPTConfig.from_dict({**mc, "vocab_size": tok.get_vocab_size()})
    model = GPT(mcfg)
    if tc.get("init_from"):
        ck = torch.load(tc["init_from"], map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        log(f"Başlanğıc çəkilər: {tc['init_from']} (çatışmayan {len(missing)}, artıq {len(unexpected)})")
    model.to(device)
    log(f"Model: {mcfg.arch}, {model.num_params()/1e6:.1f}M parametr, kontekst {mcfg.block_size}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=tc["lr"], betas=(0.9, tc["beta2"]),
                            fused=(device == "cuda"))
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    # --- davam
    CKPT, BEST = os.path.join(out, "checkpoint.pt"), os.path.join(out, "model.pt")
    LOG = os.path.join(out, "log.csv")
    if not os.path.exists(CKPT):
        hub.restore("checkpoint.pt") and hub.restore("model.pt") and hub.restore("log.csv")
    start, best = 0, float("inf")
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start, best = ck["step"] + 1, ck["best"]
        log(f"Qaldığı yerdən davam edir: addım {start}, ən yaxşı val {best:.3f}")
    new_log = not os.path.exists(LOG)
    logf = open(LOG, "a", newline="", encoding="utf-8")
    writer = csv.writer(logf)
    if new_log:
        writer.writerow(["step", "train_loss", "val_loss", "lr", "tokens", "session_minutes", "tok_per_sec"])

    run_model = torch.compile(model) if tc.get("compile") and device == "cuda" else model
    B, T, GA = tc["batch_size"], mcfg.block_size, tc["grad_accum"]
    max_steps = tc["max_steps"]

    def get_batch(d):
        ix = np.random.randint(0, len(d) - T - 1, B)
        x = torch.from_numpy(np.stack([d[i:i + T] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([d[i + 1:i + 1 + T] for i in ix]).astype(np.int64))
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y

    def lr_at(step):
        w, base = tc["warmup"], tc["lr"]
        if step < w:
            return base * (step + 1) / w
        p = min(1.0, (step - w) / max(1, max_steps - w))
        m = tc["min_lr_frac"]
        return base * (m + (1 - m) * 0.5 * (1 + math.cos(math.pi * p)))

    def autocast():
        return torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None)

    @torch.no_grad()
    def eval_loss(d):
        model.eval()
        s = 0.0
        for _ in range(tc["eval_batches"]):
            x, y = get_batch(d)
            with autocast():
                s += run_model(x, y)[1].item()
        model.train()
        return s / tc["eval_batches"]

    def sample(prompt, n=60):
        model.eval()
        idx = torch.tensor([tok.encode(prompt).ids], device=device)
        o = model.generate(idx, max_new_tokens=n, stop_id=eot, repetition_penalty=1.2)[0].tolist()
        model.train()
        return tok.decode(o)

    def save_ckpt(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "best": best}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)        # yazarkən kəsilsə, köhnə checkpoint xarab olmur

    model.train()
    t0, tok_seen, recent = time.time(), 0, []
    step = start
    try:
        for step in range(start, max_steps + 1):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            for _ in range(GA):
                x, y = get_batch(train_data)
                with autocast():
                    _, loss = run_model(x, y)
                scaler.scale(loss / GA).backward()
                recent.append(loss.item())
            if tc["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            tok_seen += B * T * GA

            if step % tc["eval_every"] == 0 or step == max_steps:
                vl = eval_loss(val_data)
                tl = sum(recent) / len(recent)
                recent = []
                el = (time.time() - t0) / 60
                done = step - start + 1
                tps = tok_seen / max(1e-9, time.time() - t0)
                eta = el / done * (max_steps - step) if done > 1 else 0
                log(f"addım {step:>6} | train {tl:.3f} | val {vl:.3f} | lr {lr_at(step):.2e} | "
                    f"{tps/1000:.1f}k tok/s | {el:.1f} dəq | qalıb ~{eta:.0f} dəq")
                writer.writerow([step, round(tl, 4), round(vl, 4), f"{lr_at(step):.3e}",
                                 step * B * T * GA, round(el, 2), round(tps)])
                logf.flush()
                improved = vl < best
                if improved:
                    best = vl
                    torch.save({"config": mcfg.to_dict(), "model": model.state_dict()}, BEST)
                save_ckpt(step)
                s = sample(tc["sample_prompt"]).replace("\n", " ")
                log("   ->", s[:200])
                with open(os.path.join(out, "samples.md"), "a", encoding="utf-8") as f:
                    f.write(f"**addım {step}** (val {vl:.3f}): {s}\n\n")
                hub.push(with_checkpoint=True)
    except KeyboardInterrupt:
        log("Dayandırıldı, checkpoint yazılır...")
        save_ckpt(step)
    logf.close()
    hub.wait()
    hub.push(with_checkpoint=True)
    hub.wait()
    log(f"Təlim bitdi. Ən yaxşı val loss: {best:.3f}. Model: {BEST}")
    return BEST


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Odlar təlimi")
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="məs. train.max_steps=100 out_dir=/content/run")
    a = ap.parse_args()
    train(load_config(a.config, a.set))
