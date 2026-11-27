"""HF_TOKEN-i tapır: mühit dəyişəni, Colab Secrets və ya Kaggle Secrets. Token heç yerə yazılmır."""
import os


def hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from google.colab import userdata          # Colab: Secrets bölməsindəki HF_TOKEN
        token = userdata.get("HF_TOKEN")
    except Exception:
        token = None
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient   # Kaggle: Add-ons -> Secrets
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None
