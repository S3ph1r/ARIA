import torch
import scipy.io.wavfile
import os
import json
from diffusers import AudioLDM2Pipeline

def generate_sfx_v4():
    asset_id = "test_door_slam_pro"
    asset_dir = f"data/assets/sfx/{asset_id}"
    os.makedirs(asset_dir, exist_ok=True)
    
    print(f">>> Inizializzazione AudioLDM 2 (HQ Tuning)...")
    model_id = "cvssp/audioldm2"
    pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    # Prompt ottimizzato secondo documentazione
    prompt = "A heavy solid wooden door slamming shut, powerful mechanical impact, resonant wooden thud, indoor room acoustics, high fidelity, cinematic sound effect, sharp transient"
    negative_prompt = "low quality, average quality, distorted, noisy, hiss, static, music, melody, electronic, metallic"
    
    print(f">>> Generazione HQ SFX per: '{prompt}'...")
    # Aumento passi (100) e regolazione guidance (4.5)
    audio = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=100, 
        guidance_scale=4.5,
        audio_length_in_s=4.0
    ).audios[0]
    
    # Normalizzazione a -3dB
    print(">>> Normalizzazione volume...")
    audio_tensor = torch.from_numpy(audio)
    max_val = torch.max(torch.abs(audio_tensor))
    if max_val > 0:
        target_peak = 0.707 
        audio = (audio / max_val.item()) * target_peak
    
    # Salvataggio
    wav_path = os.path.join(asset_dir, f"{asset_id}.wav")
    scipy.io.wavfile.write(wav_path, 16000, audio)
    
    print(f"\n>>> Asset HQ creato con successo: {wav_path}")

if __name__ == "__main__":
    generate_sfx_v4()
