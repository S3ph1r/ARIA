import torch
import numpy as np
import scipy.io.wavfile
import os
from diffusers import AudioLDM2Pipeline

def trim_silence(audio, threshold=0.01):
    # Trova il primo e l'ultimo indice sopra la soglia
    mask = np.abs(audio) > threshold
    if not np.any(mask):
        return audio
    first_idx = np.argmax(mask)
    last_idx = len(audio) - np.argmax(mask[::-1])
    # Aggiungi un piccolo margine (pad) di 0.1s se possibile
    pad = int(16000 * 0.1)
    start = max(0, first_idx - pad)
    end = min(len(audio), last_idx + pad)
    return audio[start:end]

def generate_sfx_v5():
    asset_id = "test_door_slam_v5"
    asset_dir = f"data/assets/sfx/{asset_id}"
    os.makedirs(asset_dir, exist_ok=True)
    
    print(f">>> Inizializzazione AudioLDM 2 (Excellence Tuning)...")
    model_id = "cvssp/audioldm2"
    pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    prompt = "A massive heavy wooden door slamming shut with a violent wooden thud, close-up impact, indoor room acoustics, cinematic high quality, sharp transient, 8k"
    negative_prompt = "low quality, distorted, noisy, hiss, static, music, electronic, metallic, generic"
    
    print(f">>> Generazione SFX per: '{prompt}'...")
    audio = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=100, 
        guidance_scale=4.5,
        audio_length_in_s=3.0 # Ridotto a 3s per precisione
    ).audios[0]
    
    # 1. Trimming (Rimuove silenzi iniziali/finali)
    print(">>> Auto-Trimming silenzi...")
    audio = trim_silence(audio, threshold=0.05)
    
    # 2. Normalizzazione Massima (0.98)
    print(">>> Massimizzazione volume (0.98 Peak)...")
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = (audio / max_val) * 0.98
    
    # Salvataggio
    wav_path = os.path.join(asset_dir, f"{asset_id}.wav")
    scipy.io.wavfile.write(wav_path, 16000, audio)
    
    print(f"\n>>> Asset SFX v5 creato (Trimmed & Max Volume): {wav_path}")
    print(f">>> Nuova durata: {len(audio)/16000:.2f} secondi")

if __name__ == "__main__":
    generate_sfx_v5()
