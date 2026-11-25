"""Odlar-1: lokal veb sayt.
İşə salmaq:  python app.py   →  brauzerdə http://localhost:7860 aç
Bu qovluqda model.py, model.pt və tokenizer.json olmalıdır.
"""
import torch
import gradio as gr
from tokenizers import Tokenizer
from model import GPT, GPTConfig

torch.set_num_threads(max(1, torch.get_num_threads()))
ck = torch.load("model.pt", map_location="cpu", weights_only=True)
model = GPT(GPTConfig(**ck["config"]))
model.load_state_dict(ck["model"])
model.eval()
tok = Tokenizer.from_file("tokenizer.json")
EOT = tok.token_to_id("<|endoftext|>")
print(f"Odlar-1 yükləndi: {model.num_params()/1e6:.1f} milyon parametr")


def yaz(prompt, uzunluq, temperatur, top_k):
    if not prompt.strip():
        return "Bir neçə söz yaz, model davam etdirsin."
    idx = torch.tensor([tok.encode(prompt).ids])
    out = model.generate(idx, max_new_tokens=int(uzunluq), temperature=float(temperatur),
                         top_k=int(top_k), stop_id=EOT)
    return tok.decode(out[0].tolist())


with gr.Blocks(title="Odlar-1") as demo:
    gr.Markdown("# Odlar-1\n**Odlar yurdunun ilk balaca dil modeli.** Sıfırdan, Azərbaycan dilində öyrədilib. Cümləyə başla, Odlar davam etdirsin.")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Başlanğıc mətn", value="Bakı şəhəri", lines=3)
            uzunluq = gr.Slider(10, 300, value=100, step=10, label="Neçə token yazsın")
            temperatur = gr.Slider(0.1, 1.5, value=0.8, step=0.05,
                                   label="Temperatur (aşağı = ehtiyatlı, yuxarı = yaradıcı)")
            top_k = gr.Slider(1, 200, value=40, step=1, label="Top-k")
            btn = gr.Button("Yaz", variant="primary")
        cavab = gr.Textbox(label="Odlar yazır", lines=14)
    btn.click(yaz, [prompt, uzunluq, temperatur, top_k], cavab)
    prompt.submit(yaz, [prompt, uzunluq, temperatur, top_k], cavab)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
