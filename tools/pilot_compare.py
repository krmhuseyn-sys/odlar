"""Pilot qaçışlarının müqayisəsi: eyni addımlarda val loss (cəmi və mənbə üzrə).

    python tools/pilot_compare.py [runs/pilot runs/pilot_mask runs/pilot_cos]
"""
import csv
import os
import sys

runs = sys.argv[1:] or ["runs/pilot", "runs/pilot_mask", "runs/pilot_cos"]
data = {}
for r in runs:
    p = os.path.join(r, "log.csv")
    if not os.path.exists(p):
        continue
    rows = {int(x["step"]): x for x in csv.DictReader(open(p, encoding="utf-8"))}
    src = {}
    vp = os.path.join(r, "val_sources.csv")
    if os.path.exists(vp):
        for x in csv.DictReader(open(vp, encoding="utf-8")):
            src.setdefault(int(x["step"]), {})[x["source"]] = float(x["val_loss"])
    data[os.path.basename(r)] = (rows, src)
if not data:
    sys.exit("hələ nəticə yoxdur")
names = list(data)
steps = sorted(set().union(*[set(d[0]) for d in data.values()]))
print("\nval loss (cəmi)  | " + " | ".join(f"{n:>11}" for n in names))
for s in steps:
    if s % 1000 and s != steps[-1]:
        continue
    vals = [data[n][0].get(s, {}).get("val_loss", "") for n in names]
    print(f"addım {s:>6}    | " + " | ".join(f"{float(v):11.4f}" if v else f"{'':>11}" for v in vals))
last = {n: max(data[n][0]) for n in names}
print("\nson addımda, mənbə üzrə:")
for n in names:
    s = last[n]
    per = data[n][1].get(s, {})
    tps = data[n][0][s].get("tok_per_sec", "")
    print(f"  {n:<11} addım {s:>5}: " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(per.items())) +
          (f" | {int(float(tps))/1000:.1f}k tok/s" if tps else ""))
