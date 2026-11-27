"""HF_TOKEN-i tapır: mühit dəyişəni, Kaggle Secrets və ya Colab Secrets. Token heç yerə yazılmır.

Kaggle-ın yeni şəklində google.colab paketi də quraşdırılıb, ona görə mühit paketlə yox,
qovluqlarla müəyyən edilir.
"""
import os


def _kaggle():
    try:
        from kaggle_secrets import UserSecretsClient   # Kaggle: Add-ons -> Secrets
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def _colab():
    try:
        from google.colab import userdata          # Colab: soldakı açar işarəsi (Secrets)
        return userdata.get("HF_TOKEN")
    except Exception:
        return None


def hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    if os.path.exists("/kaggle/working"):
        return _kaggle()
    if os.path.exists("/var/colab/hostname"):
        return _colab()
    return None
