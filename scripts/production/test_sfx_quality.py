import os
import json
import torch
import numpy as np
import scipy.io.wavfile
import gc
from diffusers import StableAudioPipeline
from pathlib import Path

# Configurazione
ROOT = Path("c:/Users/Roberto/aria")
OUTPUT_DIR = ROOT / "data" / "assets" / "quality_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_cases = [
    {
        "id": "door_2s",
        "prompt": "Heavy wooden door slam",
        "duration": 2
    },
    {
        "id": "hammer_2s",
        "prompt": "Hammer striking anvil",
        "duration": 2
    },
    {
        "id": "laser_2s",
        "prompt": "Laser blast",
        "duration": 2
    }
]

def trim_silence(audio, sr, threshold_percent=0.25, padding_ms=50):
    """
    Taglia il silenzio all'inizio E alla fine (TIGHT TRIM).
    """
    if len(audio.shape) > 1:
        analysis_audio = np.max(np.abs(audio), axis=0)
    else:
        analysis_audio = np.abs(audio)
    
    global_peak = np.max(analysis_audio)
    if global_peak == 0: return audio
    
    adaptive_threshold = global_peak * threshold_percent
    
    # Trova INIZIO
    start_idx = 0
    for i in range(len(analysis_audio)):
        if analysis_audio[i] > adaptive_threshold:
            start_idx = i
            break
            
    # Trova FINE
    end_idx = len(analysis_audio) - 1
    for i in range(len(analysis_audio)-1, -1, -1):
        if analysis_audio[i] > adaptive_threshold:
            end_idx = i
            break
            
    # Aggiungi un piccolo padding (50ms) per sicurezza
    padding_samples = int(sr * (padding_ms / 1000))
    
    final_start = max(0, start_idx - padding_samples)
    final_end = min(len(analysis_audio), end_idx + padding_samples)
    
    print(f"   [Trim] Original: {len(analysis_audio)} | Cut: {final_start} to {final_end}")
    
    return audio[:, final_start:final_end] if len(audio.shape) > 1 else audio[final_start:final_end]

def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"🚀 Inizio Test Qualita SFX + Auto-Trimming su {device.upper()}")

    # Carichiamo il modello UNA VOLTA per SFX
    print("⏳ Caricamento Stable Audio Open...")
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0", 
        torch_dtype=dtype
    ).to(device)

    for case in test_cases:
        print(f"\n--- Generazione: {case['id']} ---")
        
        # Generazione
        audio = pipe(
            case["prompt"], 
            num_inference_steps=75, 
            guidance_scale=7.0,
            audio_end_in_s=case["duration"]
        ).audios[0]

        # Conversione e Trimming
        audio_np = audio.cpu().numpy().astype(np.float32)
        
        # Trimming disabilitato per vedere l'output grezzo del modello
        # audio_trimmed = trim_silence(audio_np, 44100)
        audio_trimmed = audio_np
        
        duration_real = round(audio_trimmed.shape[1] / 44100, 3)
        wav_path = OUTPUT_DIR / f"{case['id']}.wav"
        profile_path = OUTPUT_DIR / f"{case['id']}_profile.json"
        
        # Normalizzazione
        max_val = np.max(np.abs(audio_trimmed))
        if max_val > 0:
            audio_trimmed = (audio_trimmed / max_val) * 0.95
            
        # Salvataggio WAV
        scipy.io.wavfile.write(str(wav_path), 44100, audio_trimmed.T if len(audio_trimmed.shape) > 1 else audio_trimmed)
        
        # Salvataggio Profilo
        profile = {
            "id": case["id"],
            "prompt": case["prompt"],
            "duration": duration_real,
            "sample_rate": 44100,
            "trimmed": True
        }
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=4)

        print(f"✅ Salvato WAV ({duration_real}s) e Profilo JSON: {case['id']}")

    # Scaricamento finale
    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print("🧹 VRAM Pulita.")

    print("\n🎉 Test completati!")

if __name__ == "__main__":
    run_test()
