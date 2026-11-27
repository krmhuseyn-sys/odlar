#!/usr/bin/env bash
# Pilotun 2-ci və 3-cü toxumları (K17): iki GPU-da paralel, hər GPU-da ayrıca qaçış.
# Birinci toxum (1337) tools/pilot.sh ilə ayrıca işləyir. Data eynidir (runs/pilot_data).
set -e
cd "$(dirname "$0")/.."
mkdir -p runs/logs
python -m odlar.data configs/pilot.yaml
run() {  # gpu config seed
  CUDA_VISIBLE_DEVICES=$1 python -m odlar.train configs/$2.yaml \
    --set train.seed=$3 out_dir=runs/$2_s$3 > runs/logs/$2_s$3.log 2>&1
}
for pair in "pilot:2 pilot_mask:2" "pilot_cos:2 pilot:3" "pilot_mask:3 pilot_cos:3"; do
  set -- $pair
  echo "== $(date +%H:%M) GPU0: $1, GPU1: $2"
  run 0 ${1%%:*} ${1##*:} & p0=$!
  run 1 ${2%%:*} ${2##*:} & p1=$!
  wait $p0 $p1
  python tools/pilot_compare.py runs/pilot* || true
done
echo "== $(date +%H:%M) hazırdır"
