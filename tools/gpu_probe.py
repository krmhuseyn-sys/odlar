"""GPU diaqnostikası: attention yolları, dtype-lar, batch ölçüsünə görə yaddaş və sürət.

    python tools/gpu_probe.py [configs/odlar2.yaml]

Təlim config-i ilə eyni model qurulur, təsadüfi tokenlərlə bir neçə tam təlim addımı atılır
(fp16 autocast + GradScaler + AdamW), hər batch ölçüsü üçün pik yaddaş və token/san yazılır.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch                                                   # noqa: E402
import torch.nn.functional as F                                # noqa: E402
from odlar.config import load_config                           # noqa: E402
from odlar.model import GPT, GPTConfig, apply_rope            # noqa: E402

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/odlar2.yaml")
dev = "cuda"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory/2**30:.1f} GiB, sm{p.major}{p.minor}, torch {torch.__version__}")
dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# 1) attention yolları
from torch.nn.attention import sdpa_kernel, SDPBackend        # noqa: E402
q = torch.randn(2, 12, 1024, 64, device=dev, dtype=dt)
for name, b in [("flash", SDPBackend.FLASH_ATTENTION), ("mem-efficient", SDPBackend.EFFICIENT_ATTENTION),
                ("math", SDPBackend.MATH)]:
    try:
        with sdpa_kernel(b):
            F.scaled_dot_product_attention(q, q, q, is_causal=True)
        print(f"  attention {name}: işləyir")
    except Exception as e:
        print(f"  attention {name}: YOX ({str(e).splitlines()[0][:90]})")

# 2) autocast altında RoPE-dən sonra tip
cos = torch.ones(1024, 32, device=dev)
with torch.autocast("cuda", dtype=dt):
    print("  RoPE-dən sonra q tipi:", apply_rope(q, cos, cos).dtype, "(gözlənilən:", dt, ")")

# 3) batch ölçüləri
mc = GPTConfig.from_dict({**cfg["model"], "vocab_size": cfg["data"]["vocab_size"]})
T = mc.block_size
for B in [16, 12, 8, 6, 4]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = GPT(mc).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        scaler = torch.amp.GradScaler(enabled=dt == torch.float16)
        x = torch.randint(0, mc.vocab_size, (B, T + 1), device=dev)
        times = []
        for i in range(6):
            torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast("cuda", dtype=dt):
                _, loss = model(x[:, :-1], x[:, 1:])
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); times.append(time.time() - t0)
        t = sum(times[2:]) / len(times[2:])
        print(f"  batch {B:>2} x {T}: pik yaddaş {torch.cuda.max_memory_allocated()/2**30:5.1f} GiB, "
              f"{B*T/t/1000:5.1f}k token/san")
    except torch.OutOfMemoryError:
        print(f"  batch {B:>2} x {T}: YADDAŞ ÇATMIR")
    finally:
        for n in ("model", "opt", "x", "loss"):
            globals().pop(n, None)
