#!/bin/bash
# mGPT-1.3B 7 GB RAM-da bir prosesdə yaddaşı doldurur: 100 paraqraflıq hissələrlə, hər hissə ayrıca prosesdə.
for lo in 0 100 200 300 400; do
  [ -f mgpt_part_$lo.json ] || python mgpt_part.py $lo $((lo+100))
done
python -c "
import json,glob,math
p=[json.load(open(f)) for f in glob.glob('mgpt_part_*.json')]
print('mGPT Belebele BPC:', round(sum(x['nll'] for x in p)/sum(x['chars'] for x in p)/math.log(2),4))"
