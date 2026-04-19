import torch
import torch.nn.functional as F
import scipy.io.wavfile
import os
import json
from diffusers import AudioLDM2Pipeline

def generate_sfx_v2():
    asset_id = "test_earthquake_02"
    asset_dir = f"data/assets/sfx/{asset_id}"
    os.makedirs(asset_dir, exist_ok=True)
    
    print(f">>> Inizializzazione AudioLDM 2 (Golden Stack)...")
    model_id = "cvssp/audioldm2"
    pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    # Prompt viscerale per massimo impatto
    prompt = "Deep seismic earthquake, ground-shaking bass rumble, heavy tectonic impact, visceral near-field vibration, detailed rock shifting, cinematic low-end, no music"
    negative_prompt = "music, melody, singing, high pitch, electronic, bright"
    
    print(f">>> Generazione SFX per: '{prompt}'...")
    audio = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=50, 
        audio_length_in_s=5.0
    ).audios[0]
    
    # Normalizzazione Audio (Picco a -3dB)
    print(">>> Normalizzazione volume...")
    audio_tensor = torch.from_numpy(audio)
    max_val = torch.max(torch.abs(audio_tensor))
    if max_val > 0:
        target_peak = 0.707 # -3dB approx
        audio_norm = (audio_tensor / max_val) * target_peak
        audio = audio_norm.numpy()
    
    # Salvataggio Warehouse Compliant
    wav_path = os.path.join(asset_dir, f"{asset_id}.wav")
    scipy.io.wavfile.write(wav_path, 16000, audio)
    
    # Generazione Profile JSON (Obbligatorio per RegistryManager)
    profile = {
        "id": asset_id,
        "category": "sfx",
        "description": "Powerful seismic earthquake rumble with deep bass impact and realistic ground vibration.",
        "tags": ["earthquake", "seismic", "rumble", "vibration", "dark", "cinematic"],
        "duration": 5.0,
        "sample_rate": 16000,
        "model": "AudioLDM 2 (Golden Stack)"
    }
    
    with open(os.path.join(asset_dir, "profile.json"), "w") as f:
        json.dump(profile, f, indent=4)
        
    print(f"\n>>> Asset creato con successo:")
    print(f">>> WAV: {wav_path}")
    print(f">>> Profile: {os.path.join(asset_dir, 'profile.json')}")

if __name__ == "__main__":
    generate_sfx_v2()
