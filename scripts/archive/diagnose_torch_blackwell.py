import torch
import sys

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SM: {torch.cuda.get_device_capability(0)}")

print("\n--- Testing Flex Attention Modules ---")

try:
    import torch._inductor.kernel.flex_attention
    print("torch._inductor.kernel.flex_attention: LOADED")
except Exception as e:
    print(f"torch._inductor.kernel.flex_attention: FAILED ({type(e).__name__}: {e})")

try:
    from torch.nn.attention.flex_attention import flex_attention
    print("torch.nn.attention.flex_attention: API AVAILABLE")
except Exception as e:
    print(f"torch.nn.attention.flex_attention: API MISSING ({type(e).__name__}: {e})")

print("\n--- Testing Transformers Import ---")
try:
    from transformers.utils import is_torch_flex_attn_available
    print(f"Transformers is_torch_flex_attn_available(): {is_torch_flex_attn_available()}")
except Exception as e:
    print(f"Transformers import error: {e}")
