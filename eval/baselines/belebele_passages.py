"""Belebele azj_Latn paraqraflarını endirir (488 unikal, CC BY-SA 4.0, facebook/belebele)."""
import json, requests
rows = []
for off in range(0, 900, 100):
    r = requests.get("https://datasets-server.huggingface.co/rows",
                     params={"dataset": "facebook/belebele", "config": "azj_Latn", "split": "test",
                             "offset": off, "length": 100}, timeout=60).json()
    rows += [x["row"] for x in r["rows"]]
P = sorted(set(r["flores_passage"].strip() for r in rows))
json.dump(P, open("belebele_az_passages.json", "w"), ensure_ascii=False)
print(len(rows), "sual,", len(P), "paraqraf,", sum(map(len, P)), "simvol")
