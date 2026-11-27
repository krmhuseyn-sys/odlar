"""Təlim nümunələrinin seçimi (K2).

Köhnə üsul hər addımda təsadüfi yerdən pəncərə götürürdü: bir epoxada datanın ~37%-i heç
görünmürdü, bəzi yerlər isə iki-üç dəfə görünürdü. Burada isə:

  - hər mənbə üst-üstə düşməyən T+1 tokenlik bloklara bölünür;
  - hər epoxada blokların başlanğıcı təsadüfi sürüşdürülür (bloklar hər dəfə başqa yerdən kəsilir);
  - mənbənin "repeat" çəkisi qədər nüsxə götürülür (1.5 = bir tam nüsxə + blokların yarısı);
  - hamısı birlikdə qarışdırılır və ardıcıl oxunur: epoxa bitənə qədər heç bir blok iki dəfə gəlmir.

Vəziyyət (epoxa, mövqe) checkpoint-də saxlanır, ona görə davam edəndə eyni yerdən gedir.
Bir neçə GPU olanda (DDP) hər GPU qlobal batch-in öz hissəsini götürür.
"""
import numpy as np


class BlockSampler:
    def __init__(self, parts, T, seed=1337, rank=0, world=1):
        """parts: [{"arr": 1D massiv (memmap ola bilər), "repeat": float, "label": str}, ...]"""
        self.parts = [p for p in parts if len(p["arr"]) > T + 1]
        if not self.parts:
            raise ValueError(f"data blok uzunluğundan ({T + 1}) qısadır")
        self.T, self.seed, self.rank, self.world = T, seed, rank, world
        self.epoch, self.pos = 0, 0
        self._plan_epoch, self._plan = None, None

    # ------------------------------------------------------------ epoxa planı
    def _make_plan(self, epoch):
        rng = np.random.default_rng([self.seed, epoch])
        T = self.T
        src, off = [], []
        for i, p in enumerate(self.parts):
            n_tok = len(p["arr"])
            rep = float(p.get("repeat", 1))
            copies = int(np.ceil(rep))
            for c in range(copies):
                phase = int(rng.integers(0, T)) if n_tok > 3 * T else 0
                n = (n_tok - 1 - phase) // T
                starts = phase + np.arange(n, dtype=np.int64) * T
                frac = rep - c
                if frac < 1:                                   # sonuncu natamam nüsxə
                    starts = rng.choice(starts, int(round(frac * n)), replace=False)
                src.append(np.full(len(starts), i, dtype=np.int32))
                off.append(starts)
        src, off = np.concatenate(src), np.concatenate(off)
        order = rng.permutation(len(src))
        return src[order], off[order]

    def plan(self, epoch):
        if self._plan_epoch != epoch:
            self._plan, self._plan_epoch = self._make_plan(epoch), epoch
        return self._plan

    def epoch_len(self, epoch=None):
        return len(self.plan(self.epoch if epoch is None else epoch)[0])

    # ------------------------------------------------------------ oxuma
    def _take(self, n):
        """Qlobal ardıcıllıqdan növbəti n bloku götürür (epoxa sərhədini keçə bilər)."""
        out = []
        while n > 0:
            src, off = self.plan(self.epoch)
            k = min(n, len(src) - self.pos)
            out.append((src[self.pos:self.pos + k], off[self.pos:self.pos + k]))
            self.pos += k
            n -= k
            if self.pos >= len(src):
                self.epoch, self.pos = self.epoch + 1, 0
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

    def next_batch(self, B):
        """Bu GPU üçün B blok: (B, T+1) int64 massivi. Bütün GPU-lar eyni qlobal sıranı görür."""
        src, off = self._take(B * self.world)
        src, off = src[self.rank * B:(self.rank + 1) * B], off[self.rank * B:(self.rank + 1) * B]
        T1 = self.T + 1
        return np.stack([np.asarray(self.parts[s]["arr"][o:o + T1]) for s, o in zip(src, off)]).astype(np.int64)

    # ------------------------------------------------------------ checkpoint
    def state_dict(self):
        return {"epoch": self.epoch, "pos": self.pos, "seed": self.seed}

    def load_state_dict(self, s):
        self.epoch, self.pos = int(s["epoch"]), int(s["pos"])
        self.seed = int(s.get("seed", self.seed))

    def coverage(self):
        """Bir epoxada hər mənbədən neçə token görünür (yoxlama üçün)."""
        src, _ = self.plan(self.epoch)
        counts = np.bincount(src, minlength=len(self.parts))
        out = {}
        for i, p in enumerate(self.parts):
            lab = p.get("label", str(i))
            out[lab] = out.get(lab, 0) + int(counts[i]) * self.T
        return out


def fixed_blocks(arr, T, n, seed=0):
    """Validasiya üçün: hər dəfə eyni n blok (müqayisə edilə bilən val loss)."""
    m = (len(arr) - 1) // T
    if m <= 0:
        return np.zeros((0, T + 1), dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(m, min(n, m), replace=False))
    return np.stack([np.asarray(arr[i * T:i * T + T + 1]) for i in idx]).astype(np.int64)
