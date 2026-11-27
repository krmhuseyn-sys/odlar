"""Təlim: bir config, bir komanda.

    python -m odlar.train configs/odlar15.yaml
    python -m odlar.train configs/odlar15.yaml --set out_dir=/content/drive/MyDrive/Odlar15
    torchrun --nproc_per_node=2 -m odlar.train configs/odlar2.yaml      # Kaggle T4x2

Nə edir:
  - data yoxdursa, əvvəlcə hazırlayır (odlar.data)
  - out_dir-də checkpoint.pt varsa, qaldığı yerdən davam edir (data mövqeyi də daxil)
  - hər eval_every addımda: hər mənbə üzrə val loss, nümunə mətn, log.csv, model.pt (ən yaxşısı),
    checkpoint.pt
  - train.hub_repo yazılıbsa, faylları Hugging Face Hub-a da göndərir (HF_TOKEN lazımdır)
  - loss qəfil sıçrayıb qalxırsa, son checkpoint-ə qayıdır və datada irəli keçir
  - torchrun ilə bir neçə GPU-da işləyir (DDP); qlobal batch GPU sayından asılı deyil
"""
import contextlib
import csv
import datetime
import math
import os
import random
import shutil
import time

import numpy as np
import torch

from .config import load_config, dump
from .data import prepare, load_parts
from .model import GPT, GPTConfig
from .sampler import BlockSampler, fixed_blocks

RANK, WORLD, LOCAL = 0, 1, 0


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


# ---------------------------------------------------------------- bir neçə GPU
def setup_dist():
    """torchrun ilə başladılıbsa, proses qrupunu qurur."""
    global RANK, WORLD, LOCAL
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False
    import torch.distributed as dist
    RANK, WORLD = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    LOCAL = int(os.environ.get("LOCAL_RANK", RANK))
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL)
    # Data hazırlığı saatlarla çəkə bilər: o vaxt digər GPU-lar gözləyir
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                            timeout=datetime.timedelta(hours=12))
    return True


def barrier():
    if WORLD > 1:
        import torch.distributed as dist
        dist.barrier()


def all_mean(t):
    """Bütün GPU-lar üzrə orta qiymət (bir GPU-da dəyişmir)."""
    if WORLD > 1:
        import torch.distributed as dist
        t = t.clone()
        dist.all_reduce(t)
        t /= WORLD
    return t


def all_sum(t):
    if WORLD > 1:
        import torch.distributed as dist
        t = t.clone()
        dist.all_reduce(t)
    return t


# ---------------------------------------------------------------- Hugging Face Hub
class Hub:
    """İstəyə bağlı: checkpoint-ləri HF Hub-a yazır və oradan bərpa edir (Kaggle/Colab üçün).
    Hər yükləmədən sonra repo tarixçəsi sıxılır (K7): köhnə checkpoint-lər yer tutmur."""

    def __init__(self, repo, squash=True):
        self.repo = repo if RANK == 0 else None        # Hub-a yalnız birinci proses yazır
        self.api, self.future, self.squash = None, None, squash
        if not self.repo:
            return
        from .hubutil import hf_token
        token = hf_token()
        if not token:
            log("  diqqət: HF_TOKEN tapılmadı, Hub-a yazılmayacaq")
            self.repo = None
            return
        from huggingface_hub import HfApi
        self.api = HfApi(token=token)
        self.api.create_repo(repo, private=True, exist_ok=True)
        log(f"  Hub: {repo} (gizli)")

    def restore(self, name, dest_dir):
        """Lokal fayl yoxdursa, Hub-dan endirir."""
        if not self.repo:
            return False
        from huggingface_hub import hf_hub_download
        try:
            p = hf_hub_download(self.repo, name, token=self.api.token)
            shutil.copy(p, os.path.join(dest_dir, name))
            log(f"  Hub-dan bərpa olundu: {name}")
            return True
        except Exception:
            return False

    def push(self, paths):
        if not self.repo:
            return
        if self.future is not None:
            if not self.future.done():
                return                                  # əvvəlki yükləmə hələ bitməyib
            try:
                self.future.result()
                if self.squash:
                    self.api.super_squash_history(self.repo, commit_message="son vəziyyət")
            except Exception as e:                      # şəbəkə xətası təlimi dayandırmamalıdır
                log(f"  diqqət: Hub xətası: {e}")
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=os.path.basename(p), path_or_fileobj=p)
               for p in paths if os.path.exists(p)]
        self.future = self.api.create_commit(self.repo, operations=ops, commit_message="checkpoint",
                                             run_as_future=True)

    def wait(self):
        if self.future is not None:
            try:
                self.future.result()
            except Exception as e:
                log(f"  diqqət: Hub xətası: {e}")


# ---------------------------------------------------------------- öyrənmə sürəti
def make_lr(tc):
    base, w, max_steps, m = tc["lr"], tc["warmup"], tc["max_steps"], tc["min_lr_frac"]
    sched = tc.get("schedule", "cosine")
    decay_frac = tc.get("decay_frac", 0.2)

    def lr_at(step):
        if step < w:
            return base * (step + 1) / w
        if sched == "wsd":
            # Warmup-Stable-Decay (K4): sabit sürət, son decay_frac hissədə 1-sqrt ilə enmə
            d0 = int(max_steps * (1 - decay_frac))
            if step < d0:
                return base
            p = min(1.0, (step - d0) / max(1, max_steps - d0))
            return base * (m + (1 - m) * (1 - math.sqrt(p)))
        p = min(1.0, (step - w) / max(1, max_steps - w))
        return base * (m + (1 - m) * 0.5 * (1 + math.cos(math.pi * p)))
    return lr_at


# ---------------------------------------------------------------- loss sıçrayışı
class SpikeGuard:
    """Loss uzun müddət adi səviyyədən xeyli yuxarıdırsa (və ya NaN-dırsa), geri dönüş istəyir.
    Tək-tük pis batch-lər nəzərə alınmır: yalnız `patience` addım ardıcıl pis olanda işə düşür."""

    def __init__(self, delta=0.5, patience=25, start=0):
        self.delta, self.patience, self.start = delta, patience, start
        self.ema, self.bad = None, 0

    def update(self, step, loss):
        if step < self.start:
            return False
        finite = math.isfinite(loss)
        if self.ema is None:
            if finite:
                self.ema = loss
            return False
        if finite and loss < self.ema + self.delta:
            self.ema = 0.99 * self.ema + 0.01 * loss
            self.bad = 0
        else:
            self.bad += 1
        return self.bad >= self.patience

    def reset(self):
        self.ema, self.bad = None, 0

    @property
    def calm(self):
        return self.bad == 0


# ---------------------------------------------------------------- təlim
def train(cfg):
    ddp = setup_dist()
    tc, mc = cfg["train"], cfg["model"]
    out = cfg["out_dir"]
    ck_dir = tc.get("ckpt_dir") or out          # Colab: checkpoint lokal diskdə, Drive-da yalnız model.pt (K8)
    if RANK == 0:
        os.makedirs(out, exist_ok=True)
        os.makedirs(ck_dir, exist_ok=True)
        with open(os.path.join(out, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(dump(cfg))

    seed = tc["seed"]
    random.seed(seed + RANK); np.random.seed(seed + RANK); torch.manual_seed(seed + RANK)
    if torch.cuda.is_available():
        device = f"cuda:{LOCAL}"
        log(f"GPU: {torch.cuda.get_device_name(LOCAL)} x{WORLD}")
        # bf16 yalnız aparat dəstəyi olan GPU-larda (Ampere+, sm80+). torch 2.14 T4 (sm75) üçün də
        # is_bf16_supported() = True qaytarır (emulyasiya): o zaman yaddaşa qənaət edən attention
        # işləmir və T4-də sürət ~4 dəfə düşür (27 noyabr Colab ölçüsü)
        cap = torch.cuda.get_device_capability(LOCAL)
        amp_dtype = torch.bfloat16 if cap[0] >= 8 and torch.cuda.is_bf16_supported() else torch.float16
        log(f"Dəqiqlik: {'bf16' if amp_dtype == torch.bfloat16 else 'fp16 + GradScaler'} (sm{cap[0]}{cap[1]})")
    else:
        device = "cpu"
        log(f"GPU yoxdur, CPU-da işləyir (yalnız kiçik sınaq üçün), proses sayı {WORLD}")
        amp_dtype = None
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    hub = Hub(tc.get("hub_repo"), squash=tc.get("hub_squash", True))

    # --- data (yalnız birinci proses hazırlayır, digərləri gözləyir)
    if RANK == 0:
        P = prepare(cfg)
    barrier()
    P = prepare(cfg) if RANK != 0 else P
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(P["tok"])
    if RANK == 0:
        shutil.copy(P["tok"], os.path.join(out, "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    parts, vals = load_parts(P)
    T = mc["block_size"]
    sampler = BlockSampler(parts, T, seed=seed, rank=RANK, world=WORLD)
    cov = sampler.coverage()
    total_cov = sum(cov.values())
    log(f"Data: bir epoxa {total_cov/1e6:.1f}M token, "
        + ", ".join(f"{k} {v/total_cov:.0%}" for k, v in cov.items()))

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
                            fused=(dev_type == "cuda"))
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    # --- davam
    CKPT, BEST = os.path.join(ck_dir, "checkpoint.pt"), os.path.join(out, "model.pt")
    LOG, VLOG = os.path.join(out, "log.csv"), os.path.join(out, "val_sources.csv")
    if RANK == 0 and not os.path.exists(CKPT):
        if hub.restore("checkpoint.pt", ck_dir):
            for n in ("model.pt", "log.csv", "val_sources.csv", "samples.md"):
                hub.restore(n, out)
    barrier()
    start, best, tokens, rollbacks = 0, float("inf"), 0, 0

    def load_ckpt(restore_sampler):
        ck = torch.load(CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        if restore_sampler and "sampler" in ck:
            sampler.load_state_dict(ck["sampler"])
        return ck

    if os.path.exists(CKPT):
        ck = load_ckpt(restore_sampler=True)
        start, best = ck["step"] + 1, ck["best"]
        tokens, rollbacks = ck.get("tokens", 0), ck.get("rollbacks", 0)
        log(f"Qaldığı yerdən davam edir: addım {start}, ən yaxşı val {best:.3f}, "
            f"epoxa {sampler.epoch}, mövqe {sampler.pos}")
        del ck
    if RANK == 0:
        new_log = not os.path.exists(LOG)
        logf = open(LOG, "a", newline="", encoding="utf-8")
        writer = csv.writer(logf)
        if new_log:
            writer.writerow(["step", "train_loss", "val_loss", "lr", "tokens", "session_minutes", "tok_per_sec"])
        new_vlog = not os.path.exists(VLOG)
        vlogf = open(VLOG, "a", newline="", encoding="utf-8")
        vwriter = csv.writer(vlogf)
        if new_vlog:
            vwriter.writerow(["step", "source", "val_loss"])

    run_model = model
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        run_model = DDP(model, device_ids=[LOCAL] if dev_type == "cuda" else None)
    if tc.get("compile") and dev_type == "cuda":
        run_model = torch.compile(run_model)

    B, GA_total, max_steps = tc["batch_size"], tc["grad_accum"], tc["max_steps"]
    if GA_total % WORLD:
        raise SystemExit(f"train.grad_accum ({GA_total}) GPU sayına ({WORLD}) bölünməlidir")
    GA = GA_total // WORLD                     # qlobal batch eyni qalır: B x grad_accum x T
    step_tokens = B * GA_total * T
    log(f"Batch: {B} x {GA} addım x {WORLD} GPU = {step_tokens:,} token/addım")
    doc_sep = eot if tc.get("doc_mask") else None
    if doc_sep is not None:
        log("Sənəd maskası: açıq (hər token yalnız öz sənədinə baxır)")
    lr_at = make_lr(tc)
    guard = SpikeGuard(tc.get("spike_delta", 0.5), tc.get("spike_patience", 25),
                       start=tc["warmup"] + 100) if tc.get("spike_rollback", True) else None

    def to_dev(a):
        t = torch.from_numpy(a)
        if dev_type == "cuda":
            t = t.pin_memory().to(device, non_blocking=True)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def autocast():
        return torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=amp_dtype is not None)

    # Validasiya: hər mənbədən hər dəfə eyni bloklar, GPU-lar arasında bölünür
    n_val = max(B, tc["eval_batches"] * B // max(1, len(vals)))
    val_blocks = {k: fixed_blocks(v, T, n_val, seed=0)[RANK::WORLD] for k, v in vals.items()}

    @torch.no_grad()
    def eval_loss():
        model.eval()
        res = {}
        for k, blk in val_blocks.items():
            s = torch.zeros(2, device=device)
            for i in range(0, len(blk), B):
                x, y = to_dev(blk[i:i + B])
                with autocast():
                    # Validasiya həmişə sənəd daxilində (maska ilə): saytda model yalnız bir mətnə baxır,
                    # və maskalı/maskasız öyrədilmiş qaçışlar eyni qayda ilə müqayisə olunur (K3)
                    _, l = model(x, y, eot)
                s += torch.tensor([l.float().item() * len(x), len(x)], device=device)
            s = all_sum(s)
            res[k] = (s[0] / s[1].clamp(min=1)).item()
        model.train()
        return sum(res.values()) / len(res), res

    def sample(prompt, n=60):
        model.eval()
        idx = torch.tensor([tok.encode(prompt).ids], device=device)
        o = model.generate(idx, max_new_tokens=n, stop_id=eot, repetition_penalty=1.2)[0].tolist()
        model.train()
        return tok.decode(o)

    def save_ckpt(step):
        if RANK != 0:
            return
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "best": best, "sampler": sampler.state_dict(),
                    "tokens": tokens, "rollbacks": rollbacks}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)        # yazarkən kəsilsə, köhnə checkpoint xarab olmur

    model.train()
    t0, tok_session, recent = time.time(), 0, []
    hub_every = tc.get("hub_every", tc["eval_every"])
    step = start
    try:
        while step <= max_steps:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            acc = torch.zeros((), device=device)
            for micro in range(GA):
                x, y = to_dev(sampler.next_batch(B))
                sync = contextlib.nullcontext() if (not ddp or micro == GA - 1) else run_model.no_sync()
                with sync:
                    with autocast():
                        _, loss = run_model(x, y, doc_sep)
                    scaler.scale(loss / GA).backward()
                acc += loss.detach().float()
            if tc["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step_loss = all_mean(acc / GA).item()     # bütün GPU-lar eyni qərarı versin
            recent.append(step_loss)
            tokens += step_tokens
            tok_session += step_tokens

            # --- loss sıçrayışı: son checkpoint-ə qayıt, data isə irəli getməyə davam edir
            if guard is not None and guard.update(step, step_loss):
                rollbacks += 1
                if rollbacks > tc.get("max_rollbacks", 3) or not os.path.exists(CKPT):
                    raise SystemExit(f"Loss sıçrayışı təkrarlanır (addım {step}); təlim dayandırıldı. "
                                     "Öyrənmə sürətini azaldıb yenidən başla.")
                barrier()
                ck = load_ckpt(restore_sampler=False)
                log(f"!! addım {step}: loss {step_loss:.3f} (adi ~{guard.ema:.3f}). "
                    f"Addım {ck['step']}-ə qayıdılır, data irəli keçir (geri dönüş {rollbacks})")
                step = ck["step"] + 1
                del ck
                guard.reset()
                recent = []
                continue

            if step % tc["eval_every"] == 0 or step == max_steps:
                vl, per = eval_loss()
                tl = sum(recent) / max(1, len(recent))
                recent = []
                el = (time.time() - t0) / 60
                tps = tok_session / max(1e-9, time.time() - t0)
                eta = (max_steps - step) * step_tokens / max(tps, 1) / 60
                log(f"addım {step:>6} | train {tl:.3f} | val {vl:.3f} | lr {lr_at(step):.2e} | "
                    f"{tps/1000:.1f}k tok/s | {el:.1f} dəq | qalıb ~{eta:.0f} dəq")
                if len(per) > 1:
                    log("   val: " + ", ".join(f"{k} {v:.3f}" for k, v in per.items()))
                if RANK == 0:
                    writer.writerow([step, round(tl, 4), round(vl, 4), f"{lr_at(step):.3e}",
                                     tokens, round(el, 2), round(tps)])
                    logf.flush()
                    for k, v in per.items():
                        vwriter.writerow([step, k, round(v, 4)])
                    vlogf.flush()
                    if vl < best:
                        best = vl
                        torch.save({"config": mcfg.to_dict(), "model": model.state_dict()}, BEST)
                    if guard is None or guard.calm:    # sıçrayış başlayanda checkpoint yazılmır
                        save_ckpt(step)
                    s = sample(tc["sample_prompt"]).replace("\n", " ")
                    log("   ->", s[:200])
                    with open(os.path.join(out, "samples.md"), "a", encoding="utf-8") as f:
                        f.write(f"**addım {step}** (val {vl:.3f}): {s}\n\n")
                    if step % hub_every == 0 or step == max_steps:
                        hub.push([CKPT, BEST, os.path.join(out, "tokenizer.json"),
                                  os.path.join(out, "config.yaml"), LOG, VLOG,
                                  os.path.join(out, "samples.md")])
                barrier()
            step += 1
    except KeyboardInterrupt:
        log("Dayandırıldı, checkpoint yazılır...")
        save_ckpt(step)
    if RANK == 0:
        logf.close()
        vlogf.close()
        hub.wait()
        hub.push([CKPT, BEST, os.path.join(out, "tokenizer.json"), os.path.join(out, "config.yaml"),
                  LOG, VLOG, os.path.join(out, "samples.md")])
        hub.wait()
    log(f"Təlim bitdi. Ən yaxşı val loss: {best:.3f}. Model: {BEST}")
    if ddp:
        import torch.distributed as dist
        dist.destroy_process_group()
    return BEST


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Odlar təlimi")
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="məs. train.max_steps=100 out_dir=/content/run")
    a = ap.parse_args()
    train(load_config(a.config, a.set))
