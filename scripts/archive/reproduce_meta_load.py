import traceback
import torch
from transformers import AutoConfig, AutoModel

model_path = r"C:\Users\Roberto\aria\data\assets\models\acestep-v15-xl-sft"
print(f"Reproducing model load on 'meta' device from: {model_path}")

try:
    # Use meta device to avoid memory issues and just check initialization logic
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Force SDPA to see if it triggers the Flex Attention inductor import on Blackwell
    print("Loading model on 'meta' device with attn_implementation='sdpa'...")
    with torch.device("meta"):
        model = AutoModel.from_pretrained(
            model_path, 
            config=config,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16
        )
    print("SUCCESS: Model loaded on meta device (no weights).")
    
except Exception:
    print("\nREPRODUCED FAILURE:")
    traceback.print_exc()
