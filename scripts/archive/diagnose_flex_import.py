import traceback
import sys

print("Checking flex_attention imports...")
try:
    import torch
    print(f"Torch version: {torch.__version__}")
    
    print("\nAttempting: from torch.nn.attention.flex_attention import flex_attention")
    from torch.nn.attention.flex_attention import flex_attention
    print("SUCCESS: Imported flex_attention")
except Exception:
    print("FAILURE: flex_attention import failed")
    traceback.print_exc()

print("\nAttempting: import torch._inductor.kernel.flex_attention")
try:
    import torch._inductor.kernel.flex_attention
    print("SUCCESS: Imported torch._inductor.kernel.flex_attention")
except ModuleNotFoundError:
    print("FAILURE: ModuleNotFoundError for torch._inductor.kernel.flex_attention (AS EXPECTED on some builds)")
except Exception:
    print("FAILURE: Unexpected error during inductor kernel import")
    traceback.print_exc()

print("\nAttempting to load ACE-Step XL model logic (partial)...")
try:
    from transformers import AutoConfig, AutoModel
    from transformers.utils import is_torch_flex_attn_available
    print(f"is_torch_flex_attn_available(): {is_torch_flex_attn_available()}")
    
    # Simulate the failing call
    model_path = r"C:\Users\Roberto\aria\data\assets\models\acestep-v15-xl-sft"
    print(f"Loading config from {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("Config loaded.")
    
    print("Attempting AutoModel.from_pretrained with sdpa...")
    # We use a dummy or just check if it triggers the import
    # To avoid VRAM issues in a test, we could just check the class initialization
except Exception:
    traceback.print_exc()
