import os
from huggingface_hub import login, snapshot_download
import torch
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

def download_models(auth_token=None):
    """
    Download all required models and weights from HuggingFace.
    Uses auth_token if provided, otherwise looks for HF_TOKEN env variable.
    """
    if auth_token:
        login(auth_token)
    elif "HF_TOKEN" in os.environ:
        login(os.environ["HF_TOKEN"])
    else:
        print("Warning: No HuggingFace token provided. Some models might not be accessible.")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download base model
    print("Downloading FLUX base model...")
    base_model_path = snapshot_download(
        "black-forest-labs/FLUX.1-dev",
        local_dir="models/flux-base",
        ignore_patterns=["*.md", "*.txt"],
    )
    
    # Download LoRA weights
    print("Downloading OminiControl LoRA weights...")
    lora_path = snapshot_download(
        "Yuanshi/OminiControl",
        local_dir="models/omini-control",
        allow_patterns=["omini/subject_512.safetensors", "omini/subject_1024_beta.safetensors"],
    )
    
    return {
        "base_model": base_model_path,
        "lora": lora_path
    }

if __name__ == "__main__":
    download_models()