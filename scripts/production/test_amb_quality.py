import os
import torch
import numpy as np
import scipy.io.wavfile
import gc
from diffusers import AudioLDM2Pipeline
from pathlib import Path

# Configurazione
ROOT = Path("c:/Users/Roberto/aria")
OUTPUT_DIR = ROOT / "data" / "assets" / "quality_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_cases = [
    {
        "id": "test_amb_amazon_jungle",
        "prompt": "Lush primordial Amazonian jungle, dense tropical rainforest, diverse bird calls, macaw screeches, intense cicada buzzing, humid atmosphere, distant waterfall, rustling thick leaves, 8k resolution audio, immersive 3D soundscape, cinematic quality, high-fidelity.",
        "negative_prompt": "static, digital noise, white noise, muffled, low resolution, mono, low bitrate, distorted, robotic, human voices.",
        "duration": 30
    }
]

def trim_silence(audio, sr, threshold_percent=0.25, release_ms=150):
    """
    Taglia il silenzio partendo dal FONDO del file (Reverse Scan).
    Soglia adattiva al 25% del picco massimo.
    """
    if len(audio.shape) > 1:
        analysis_audio = np.max(np.abs(audio), axis=0)
    else:
        analysis_audio = np.abs(audio)
    
    global_peak = np.max(analysis_audio)
    if global_peak == 0: return audio
    
    adaptive_threshold = global_peak * threshold_percent
    
    # Scansione inversa
    found_idx = len(analysis_audio) - 1
    for i in range(len(analysis_audio)-1, -1, -1):
        if analysis_audio[i] > adaptive_threshold:
            found_idx = i
            break
            
    margin_samples = int(sr * (release_ms / 1000))
    end_sample = min(len(analysis_audio), found_idx + margin_samples)
    
    return audio[:, :end_sample] if len(audio.shape) > 1 else audio[:end_sample]

def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"🚀 Inizio Test Qualita AMB (AudioLDM 2 LARGE) su {device.upper()}")

    # Caricamento del modello LARGE
    print("⏳ Caricamento AudioLDM2-LARGE...")
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2-large", 
        torch_dtype=dtype
    ).to(device)

    for case in test_cases:
        print(f"\n--- Generazione: {case['id']} ---")
        print(f"Prompt: {case['prompt']}")
        
        # Generazione
        audio = pipe(
            case["prompt"],
            negative_prompt=case["negative_prompt"],
            num_inference_steps=100,
            audio_length_in_s=case["duration"]
        ).audios[0]

        # Trimming e Salvataggio
        audio_np = audio.astype(np.float32)
        if len(audio_np.shape) == 1:
            audio_np = audio_np[np.newaxis, :]
            
        print("✂️ Trimming silenzio in corso...")
        audio_trimmed = trim_silence(audio_np, 16000)
        
        wav_path = OUTPUT_DIR / f"{case['id']}.wav"
        
        # Normalizzazione
        max_val = np.max(np.abs(audio_trimmed))
        if max_val > 0:
            audio_trimmed = (audio_trimmed / max_val) * 0.98
            
        scipy.io.wavfile.write(str(wav_path), 16000, audio_trimmed.T)
        print(f"✅ Salvato ({round(audio_trimmed.shape[1]/16000, 2)}s): {wav_path}")

    # Scaricamento finale
    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print("🧹 VRAM Pulita e modello scaricato.")
    print("\n🎉 Test AMB completato!")

if __name__ == "__main__":
    run_test()
