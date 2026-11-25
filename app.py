"""Odlar-1: veb sayt (lokal və Hugging Face Spaces).
Lokal:  python app.py   →  brauzerdə http://localhost:7860
Bu qovluqda model.py, model.pt və tokenizer.json olmalıdır.
"""
import os
import torch
import gradio as gr
from tokenizers import Tokenizer
from model import GPT, GPTConfig

ON_SPACES = bool(os.environ.get("SPACE_ID"))

ck = torch.load("model.pt", map_location="cpu", weights_only=True)
model = GPT(GPTConfig(**ck["config"]))
model.load_state_dict(ck["model"])
model.eval()
tok = Tokenizer.from_file("tokenizer.json")
EOT = tok.token_to_id("<|endoftext|>")
print(f"Odlar-1 yükləndi: {model.num_params()/1e6:.1f} milyon parametr")


def yaz(prompt, uzunluq, temperatur, top_k, tekrar):
    if not prompt.strip():
        return "Bir neçə söz yaz, model davam etdirsin."
    idx = torch.tensor([tok.encode(prompt).ids[-model.config.block_size:]])
    out = model.generate(idx, max_new_tokens=int(uzunluq), temperature=float(temperatur),
                         top_k=int(top_k), stop_id=EOT,
                         repetition_penalty=float(tekrar))
    return tok.decode(out[0].tolist())


HAQQINDA = """
**Odlar-1 nədir?** Azərbaycan Vikipediyası üzərində sıfırdan öyrədilmiş 13.8 milyon parametrli kiçik GPT modeli.
Sual-cavab etmir, yazdığın cümləni davam etdirir.

**Diqqət:** model qrammatikanı yaxşı bilir, amma faktları bilmir. Yerləri, tarixləri və adları tez-tez uydurur.
Yazdıqlarını həqiqət kimi qəbul etmə.

Kod və təlim nəticələri: [github.com/krmhuseyn-sys/odlar](https://github.com/krmhuseyn-sys/odlar)
"""

with gr.Blocks(title="Odlar-1") as demo:
    gr.Markdown("# Odlar-1\n**Odlar yurdunun ilk balaca dil modeli.** Sıfırdan, Azərbaycan dilində öyrədilib. "
                "Cümləyə başla, Odlar davam etdirsin.")
    with gr.Row():
        with gr.Column():
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
    gr.Examples(["Xəzər dənizi", "Qədim zamanlarda", "Azərbaycan mətbəxi", "Futbol üzrə",
                 "Azərbaycanın paytaxtı"], inputs=prompt, label="Nümunələr")
    gr.Markdown(HAQQINDA)
    inputs = [prompt, uzunluq, temperatur, top_k, tekrar]
    btn.click(yaz, inputs, cavab)
    prompt.submit(yaz, inputs, cavab)

if __name__ == "__main__":
    if ON_SPACES:
        demo.launch()
    else:
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
