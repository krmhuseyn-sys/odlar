"""Odlar: lokal veb sayt (Odlar-1 və Odlar-1.5).
Lokal:  python app.py   ->  brauzer özü açılır (adətən http://127.0.0.1:7860)

Lazımdır: odlar/ qovluğu (model kodu). Fayllar:
  Odlar-1:    model.pt, tokenizer.json               (bu qovluqda)
  Odlar-1.5:  odlar15/model.pt, odlar15/tokenizer.json
Hansı model tapılmasa, siyahıda görünmür.
"""
import os
import torch
import gradio as gr
from tokenizers import Tokenizer
from odlar.model import load_checkpoint

ON_SPACES = bool(os.environ.get("SPACE_ID"))
HERE = os.path.dirname(os.path.abspath(__file__))

MODELLER = {
    "Odlar-1.5 (40M)": ("odlar15/model.pt", "odlar15/tokenizer.json"),
    "Odlar-1 (14M)": ("model.pt", "tokenizer.json"),
}
MODELLER = {ad: (os.path.join(HERE, m), os.path.join(HERE, t))
            for ad, (m, t) in MODELLER.items()
            if os.path.exists(os.path.join(HERE, m)) and os.path.exists(os.path.join(HERE, t))}
if not MODELLER:
    raise SystemExit("Heç bir model tapılmadı: model.pt və tokenizer.json lazımdır.")

_yaddas = {}


def yukle(ad):
    """Modeli ilk dəfə lazım olanda yükləyir, sonra yaddaşda saxlayır."""
    if ad not in _yaddas:
        mpath, tpath = MODELLER[ad]
        model = load_checkpoint(mpath)
        tok = Tokenizer.from_file(tpath)
        _yaddas[ad] = (model, tok, tok.token_to_id("<|endoftext|>"))
        print(f"{ad} yükləndi: {model.num_params()/1e6:.1f} milyon parametr")
    return _yaddas[ad]


def yaz(ad, prompt, uzunluq, temperatur, top_k, tekrar):
    if not prompt.strip():
        return "Bir neçə söz yaz, model davam etdirsin."
    model, tok, eot = yukle(ad)
    idx = torch.tensor([tok.encode(prompt).ids[-model.config.block_size:]])
    out = model.generate(idx, max_new_tokens=int(uzunluq), temperature=float(temperatur),
                         top_k=int(top_k), stop_id=eot,
                         repetition_penalty=float(tekrar))
    return tok.decode(out[0].tolist())


HAQQINDA = """
**Odlar nədir?** Azərbaycan dilində sıfırdan öyrədilmiş kiçik GPT modelləri.
Odlar-1: 13.8 milyon parametr, yalnız Vikipediya. Odlar-1.5: 40 milyon parametr, Vikipediya və veb mətnləri.
Sual-cavab etmirlər, yazdığın cümləni davam etdirirlər.

**Diqqət:** modellər qrammatikanı yaxşı bilir, amma faktları bilmir. Yerləri, tarixləri, adları və hadisələri
tez-tez uydururlar. Real şəxslər və qurumlar haqqında yazdıqları da uydurmadır.

Kod və nəticələr: [github.com/krmhuseyn-sys/odlar](https://github.com/krmhuseyn-sys/odlar)
"""

with gr.Blocks(title="Odlar") as demo:
    gr.Markdown("# Odlar\n**Odlar yurdunun balaca dil modelləri.** Sıfırdan, Azərbaycan dilində öyrədilib. "
                "Cümləyə başla, Odlar davam etdirsin.")
    with gr.Row():
        with gr.Column():
            secim = gr.Dropdown(list(MODELLER), value=list(MODELLER)[0], label="Model")
            prompt = gr.Textbox(label="Başlanğıc mətn", value="Xəzər dənizi", lines=3)
            btn = gr.Button("Yaz", variant="primary")
            with gr.Accordion("Ayarlar", open=False):
                uzunluq = gr.Slider(10, 300, value=100, step=10, label="Neçə token yazsın")
                temperatur = gr.Slider(0.1, 1.5, value=0.8, step=0.05,
                                       label="Temperatur (aşağı = ehtiyatlı, yuxarı = yaradıcı)")
                top_k = gr.Slider(1, 200, value=40, step=1, label="Top-k")
                tekrar = gr.Slider(1.0, 2.0, value=1.3, step=0.05,
                                   label="Təkrar cəzası (1 = söndürülüb, yuxarı = daha az təkrar)")
        cavab = gr.Textbox(label="Odlar yazır", lines=14)
    gr.Examples(["Xəzər dənizi", "Qədim zamanlarda", "Azərbaycan mətbəxi", "Qış gecəsi idi",
                 "Kür çayı", "Azərbaycanın paytaxtı"], inputs=prompt, label="Nümunələr")
    gr.Markdown(HAQQINDA)
    inputs = [secim, prompt, uzunluq, temperatur, top_k, tekrar]
    btn.click(yaz, inputs, cavab)
    prompt.submit(yaz, inputs, cavab)

if __name__ == "__main__":
    yukle(list(MODELLER)[0])
    if ON_SPACES:
        demo.launch()
    else:
        # server_port=None: 7860 məşğuldursa, Gradio özü növbəti boş portu tapır
        demo.launch(server_name="127.0.0.1", server_port=None, inbrowser=True)
