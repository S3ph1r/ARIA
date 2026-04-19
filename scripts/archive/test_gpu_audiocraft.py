import torch
import sys
import os
from pathlib import Path

# Configurazione Cache modelli (Warehouse-First)
ARIA_ROOT = Path(__file__).parent.parent
os.environ["HF_HOME"] = str(ARIA_ROOT / "data" / "assets" / "models" / "huggingface")
os.environ["AUDIOCRAFT_CACHE_DIR"] = str(ARIA_ROOT / "data" / "assets" / "models" / "audiocraft")

def test_environment():
    print("--- ARIA GEOMETRY GPU TEST ---")
    print(f"Python Version: {sys.version}")
    
    # 1. Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        device_cap = torch.cuda.get_device_capability(0)
        print(f"Device Name: {device_name}")
        print(f"CUDA Capability: {device_cap}")
        
        if device_cap == (12, 0):
            print(">>> SUCCESS: Blackwell SM_120 Architecture detected! (RTX 50-series)")
        else:
            print(f">>> WARNING: Unexpected architecture {device_cap}")
            
    # 2. Test Audiocraft (MusicGen) - Lightweight test
    try:
        from audiocraft.models import MusicGen
        print(">>> Audiocraft import: SUCCESS")
        
        # Micro-test (LARGE model - maximum quality)
        model = MusicGen.get_pretrained('facebook/musicgen-large')
        model.set_generation_params(duration=5) # 5 seconds for a proper sting
        
        descriptions = ['A bright, shimmering rise ending in a bell-like chime, cinematic revelation sting']
        wav = model.generate(descriptions)
        
        # Save output to ARIA Warehouse
        output_dir = ARIA_ROOT / "data" / "assets" / "sounds" / "stings"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "sting_revelation_test.wav"
        
        # We need a quick way to save the wav (torchaudio or soundfile)
        import soundfile as sf
        # (1, Channels, Samples) -> (Samples, Channels)
        audio_data = wav[0].cpu().numpy().T
        sf.write(str(output_path), audio_data, model.sample_rate)
        
        print(f">>> Audio Generation: SUCCESS (Saved to: {output_path})")
        
    except Exception as e:
        print(f">>> FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_environment()
