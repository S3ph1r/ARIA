import torch
import numpy as np
import scipy.io.wavfile
import os
import json
from pathlib import Path
from audiocraft.models import MusicGen
from diffusers import AudioLDM2Pipeline, StableAudioPipeline

# Configurazione Warehouse-First
ARIA_ROOT = Path(__file__).parent.parent
os.environ["HF_HOME"] = str(ARIA_ROOT / "data" / "assets" / "models" / "huggingface")
os.environ["AUDIOCRAFT_CACHE_DIR"] = str(ARIA_ROOT / "data" / "assets" / "models" / "audiocraft")

def save_asset(asset_id, category, audio, sr, description):
    asset_dir = f"data/assets/{category}/{asset_id}"
    os.makedirs(asset_dir, exist_ok=True)
    wav_path = os.path.join(asset_dir, f"{asset_id}.wav")
    
    # Normalizzazione Peak 0.98
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = (audio / max_val) * 0.98
        
    scipy.io.wavfile.write(wav_path, sr, audio.T if len(audio.shape) > 1 else audio)
    
    profile = {
        "id": asset_id,
        "category": category,
        "description": description,
        "duration": len(audio.flatten()) / sr if len(audio.shape) == 1 else audio.shape[1] / sr,
        "sample_rate": sr
    }
    with open(os.path.join(asset_dir, "profile.json"), "w") as f:
        json.dump(profile, f, indent=4)
    print(f">>> Creato asset: {asset_id} ({category})")

def run_grand_test():
    device = "cuda"
    dtype = torch.float16
    
    # 1. MUS - MusicGen Large
    print("\n--- TEST MUS (MusicGen Large) ---")
    mg = MusicGen.get_pretrained('facebook/musicgen-large')
    mg.set_generation_params(duration=30)
    wav = mg.generate(['Epic cinematic space pad, starlight strings, shimmering galaxy texture, ethereal choir in the distance'], progress=True)
    save_asset("mus_starlight_pad", "pads", wav[0, 0].cpu().numpy(), 32000, "Epic orchestral space pad")
    del mg # Libera VRAM
    torch.cuda.empty_cache()

    # 2. AMB - AudioLDM 2
    print("\n--- TEST AMB (AudioLDM 2) ---")
    aldm = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", torch_dtype=dtype).to(device)
    audio = aldm("Late night cyberpunk city, distant flying cars, neon rain hitting glass", num_inference_steps=50, audio_length_in_s=30).audios[0]
    save_asset("amb_neon_noir", "pads", audio, 16000, "High-detail cyberpunk atmosphere")
    del aldm
    torch.cuda.empty_cache()

    # 3. SFX & STING - Stable Audio Open
    print("\n--- TEST SFX & STING (Stable Audio 44.1kHz) ---")
    sa = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=dtype).to(device)
    
    # SFX
    sfx = sa("A large crystal vase shattering on cold stone floor, high-pitched glass shards scattering, sharp realistic impact", num_inference_steps=100, audio_end_in_s=3).audios[0]
    save_asset("sfx_crystal_shatter", "sfx", sfx.cpu().numpy().astype(np.float32), 44100, "High-fidelity crystal shattering impact")
    
    # STING
    sting = sa("Dark horror jump-scare accent, dissonant bowing on a metal sheet, terrifying high-pitched screech, sudden hit", num_inference_steps=100, audio_end_in_s=8).audios[0]
    save_asset("sting_horror_shriek", "stings", sting.cpu().numpy().astype(np.float32), 44100, "Piercing horror scream/sting")
    
    print("\n>>> GRAND TEST COMPLETATO!")

if __name__ == "__main__":
    run_grand_test()
