#!/usr/bin/env python3
"""
ARIA Sound Factory: Shopping List Processor
Trasforma le richieste registiche di DIAS in asset sonori reali.
"""

import os
import json
import time
import torch
import numpy as np
import scipy.io.wavfile
import argparse
from pathlib import Path

# Modelli AI
from audiocraft.models import MusicGen
from diffusers import AudioLDM2Pipeline, StableAudioPipeline

# Configurazione Ambiente
ARIA_ROOT = Path(__file__).parent.parent.parent
os.environ["HF_HOME"] = str(ARIA_ROOT / "data" / "assets" / "models" / "huggingface")
os.environ["AUDIOCRAFT_CACHE_DIR"] = str(ARIA_ROOT / "data" / "assets" / "models" / "audiocraft")

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
    
    return audio[:, final_start:final_end] if len(audio.shape) > 1 else audio[final_start:final_end]

def process_item(item, sfx_model, amb_pipe):
    category = item.get("category", "SFX")
    prompt = item.get("prompt", "")
    target_id = item.get("id", "temp_asset")
    
    # EURISTICA DURATA FINALE
    if category == "AMB":
        duration = 5.0 # Ambiente a 5s come richiesto
        print(f"Generating AMB [{target_id}] with AudioLDM2 (Duration: {duration}s)...")
        with torch.no_grad():
            audio = amb_pipe(prompt, num_inference_steps=50, audio_length_in_s=duration).audios[0]
        sr = 16000 # AudioLDM2 standard
    elif category == "MUS":
        duration = 30.0 # Musica a 30s per il loop
        print(f"Generating MUS [{target_id}] with StableAudio (Duration: {duration}s)...")
        # Nota: Usiamo StableAudio per MUS per qualità superiore su durate lunghe
        with torch.no_grad():
            audio = sfx_model.generate_diffusion_cond(
                [prompt],
                steps=50,
                cfg_scale=7.0,
                sample_size=int(duration * 44100),
                sample_rate=44100,
                device="cuda"
            )
        audio = audio[0]
        sr = 44100
    else: # SFX e STING
        duration = 2.0 # SFX a 2s come richiesto
        print(f"Generating {category} [{target_id}] with StableAudio (Duration: {duration}s)...")
        with torch.no_grad():
            audio = sfx_model.generate_diffusion_cond(
                [prompt],
                steps=50,
                cfg_scale=7.0,
                sample_size=int(duration * 44100),
                sample_rate=44100,
                device="cuda"
            )
        audio = audio[0]
        sr = 44100

    # Conversione (Trimming disabilitato)
    audio_trimmed = audio.cpu().numpy().astype(np.float32) if torch.is_tensor(audio) else audio.astype(np.float32)
    
    duration_real = round(audio_trimmed.shape[1] / sr, 3)
    
    # Save logic remains same...
    output_path = ARIA_ROOT / "data" / "assets" / "production" / f"{target_id}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    scipy.io.wavfile.write(output_path, sr, audio_trimmed.T if len(audio_trimmed.shape) > 1 else audio_trimmed)
    
    # Update profile
    profile = {
        "id": target_id,
        "prompt": prompt,
        "duration": duration_real,
        "sample_rate": sr,
        "category": category,
        "trimmed": True
    }
    
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(profile, f, indent=4)
    
    return profile

def save_asset(asset_id, category, audio, sr, description):
    """Salva l'asset nel magazzino sound_library con relativo profile.json."""
    asset_dir = ARIA_ROOT / "data" / "assets" / "sound_library" / category / asset_id
    os.makedirs(asset_dir, exist_ok=True)
    wav_path = asset_dir / f"{asset_id}.wav"
    
    # Trimming disabilitato: output grezzo del modello
    # audio = trim_silence(audio, sr)
    duration_real = round(audio.shape[1] / sr if len(audio.shape) > 1 else len(audio) / sr, 3)

    # Normalizzazione Peak 0.98
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = (audio / max_val) * 0.98
        
    scipy.io.wavfile.write(str(wav_path), sr, audio.T if len(audio.shape) > 1 else audio)
    
    # Creazione passaporto digitale con durata reale
    profile = {
        "id": asset_id,
        "category": category,
        "description": description,
        "duration": duration_real,
        "sample_rate": sr,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "registry": {
            "author": "ARIA Sound Factory (B2 Orchestrator)",
            "blackwell_native": True,
            "quality": "high-fidelity-v2"
        }
    }
    
    with open(asset_dir / "profile.json", "w", encoding='utf-8') as f:
        json.dump(profile, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Asset salvato: {asset_id} ({duration_real}s) -> sound_library/{category}/")

def process_list(json_path):
    if not os.path.exists(json_path):
        print(f"❌ File non trovato: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assets = data.get("missing_assets", [])
    project_id = data.get("project_id", "unknown")
    print(f"🚀 Avvio Sound Factory (HQ Mode) per: {project_id}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Raggruppamento per tipo
    grouped = {}
    for a in assets:
        t = a["type"]
        if t not in grouped: grouped[t] = []
        grouped[t].append(a)

    # 1. Processo MUS (MusicGen) -> PAD
    if "mus" in grouped:
        print("\n--- 🎹 Fase 1: MUSICA (MusicGen-Large) -> PAD ---")
        mg = MusicGen.get_pretrained('facebook/musicgen-large')
        for item in grouped["mus"]:
            asset_id = item["canonical_id"]
            prompt = item.get("production_prompt") or item.get("universal_prompt")
            print(f"Generating MUS: {asset_id}...")
            mg.set_generation_params(duration=30)
            wav = mg.generate([prompt], progress=True)
            save_asset(asset_id, "pad", wav[0, 0].cpu().numpy(), 32000, prompt)
        del mg
        torch.cuda.empty_cache()

    # 2. Processo AMB (AudioLDM 2 Large) -> AMB
    if "amb" in grouped:
        print("\n--- 🌌 Fase 2: AMBIENTE (AudioLDM 2 LARGE) -> AMB ---")
        # Switch a modello LARGE per AMB
        aldm = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-large", torch_dtype=dtype).to(device)
        neg_prompt = "static, digital noise, white noise, muffled, low resolution, mono, low bitrate, distorted, robotic"
        
        for item in grouped["amb"]:
            asset_id = item["canonical_id"]
            orig_prompt = item.get("production_prompt") or item.get("universal_prompt")
            # Quality Injection
            full_prompt = f"{orig_prompt}, immersive soundscape, ultra-detailed, cinematic depth, high fidelity"
            
            duration = 5.0
            print(f"Generating AMB: {asset_id} (5s)...")
            audio = aldm(
                full_prompt, 
                negative_prompt=neg_prompt,
                num_inference_steps=100, 
                audio_length_in_s=duration
            ).audios[0]
            save_asset(asset_id, "amb", audio, 16000, orig_prompt)
        del aldm
        torch.cuda.empty_cache()

    # 3. Processo SFX e STING (Stable Audio)
    if "sfx" in grouped or "sting" in grouped:
        print("\n--- 🎧 Fase 3: SFX & STING (Stable Audio Open) ---")
        sa = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=dtype).to(device)
        for item in grouped.get("sfx", []) + grouped.get("sting", []):
            asset_id = item["canonical_id"]
            orig_prompt = item.get("production_prompt") or item.get("universal_prompt")
            # Quality Injection
            full_prompt = f"{orig_prompt}, high fidelity, professional foley, 44.1kHz, clear transients"
            
            duration = 8.0
            print(f"Generating {item['type'].upper()}: {asset_id} (8s)...")
            audio = sa(
                full_prompt, 
                negative_prompt="Low quality, muffled, static, synthesized, distorted, background noise, low bit rate",
                num_inference_steps=200, 
                guidance_scale=7.0,
                audio_end_in_s=duration
            ).audios[0]
            save_asset(asset_id, item['type'], audio.cpu().numpy().astype(np.float32), 44100, orig_prompt)
        del sa
        torch.cuda.empty_cache()

    # 4. Aggiornamento Registro (Redis)
    print("\n--- 🔄 Aggiornamento Registro Sound Library ---")
    try:
        import sys
        sys.path.insert(0, str(ARIA_ROOT))
        from aria_node_controller.settings_gui import load_settings
        import redis
        
        settings = load_settings()
        r_host = settings.get("redis_host", "127.0.0.1")
        r_port = settings.get("redis_port", 6379)
        r_pass = settings.get("redis_password", None)
        node_ip = settings.get("local_ip", "127.0.0.1")
        
        r_client = redis.Redis(host=r_host, port=r_port, password=r_pass, decode_responses=True)
        r_client.ping()
        
        from aria_node_controller.core.registry_manager import AriaRegistryManager
        manager = AriaRegistryManager(ARIA_ROOT, r_client, local_ip=node_ip)
        manager.publish()
        print("✅ Registro master aggiornato su Redis.")
    except Exception as e:
        print(f"⚠️ Errore Registro: {e}")

    print("\n🎉 Produzione Master Completata!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a DIAS Master Shopping List")
    parser.add_argument("json_file", help="Path to the shopping list JSON")
    args = parser.parse_args()
    
    process_list(args.json_file)
