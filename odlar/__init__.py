"""Odlar: Azərbaycan dilində sıfırdan dil modelləri.

Modullar:
  odlar.model     GPT quruluşu (gpt2 və llama variantları), load_checkpoint
  odlar.config    YAML config-lər (configs/ qovluğu)
  odlar.data      data hazırlığı: mənbələr -> tokenizer -> train.npy/val.npy
  odlar.train     təlim, davam etmə, checkpoint, HF Hub
  odlar.generate  terminaldan mətn yazdırmaq
  odlar.evaluate  standart testlər (BPC, qrammatika, paytaxtlar)
  odlar.export    brauzer üçün ONNX
"""
__version__ = "0.2.0"
