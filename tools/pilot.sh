#!/usr/bin/env bash
# Pilot: data bir dəfə hazırlanır, sonra üç qaçış ardıcıl, sonda müqayisə cədvəli.
# Qaçış yarımçıq qalıbsa, yenidən işə salanda qaldığı yerdən davam edir.
set -e
cd "$(dirname "$0")/.."
python -m odlar.data configs/pilot.yaml
for c in pilot pilot_mask pilot_cos; do
  echo "================ $c ================"
  python -m odlar.train configs/$c.yaml
  python tools/pilot_compare.py
done
