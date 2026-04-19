import os
import sys
import json
import argparse
import shutil
import re
import time
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
from audiocraft.models import MusicGen

# Funzione per import dinamici (evita errori se le lib non sono usate)
def get_diffusers_pipeline(model_id):
    from diffusers import AudioLDM2Pipeline, StableAudioPipeline
    if "audioldm2" in model_id.lower():
        return AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    if "stable-audio" in model_id.lower():
        return StableAudioPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    return None

# Configurazione Cache modelli (Warehouse-First)
ARIA_ROOT = Path(__file__).parent.parent
os.environ["HF_HOME"] = str(ARIA_ROOT / "data" / "assets" / "models" / "huggingface")
os.environ["AUDIOCRAFT_CACHE_DIR"] = str(ARIA_ROOT / "data" / "assets" / "models" / "audiocraft")

# Stato globale per gestione VRAM
current_model_id = None
active_pipeline = None

# Statistiche finali
STATS = {"success": 0, "failure": 0, "skipped": 0, "start_time": 0}

def slugify(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')

def get_model_id(asset_type):
    if asset_type == 'mus': return 'facebook/musicgen-large'
    if asset_type == 'amb': return 'cvssp/audioldm2'
    if asset_type in ['sfx', 'sting']: return 'stabilityai/stable-audio-open-1.0'
    return 'facebook/musicgen-large'

def get_smart_duration(asset_type, prompt):
    prompt_l = prompt.lower()
    if asset_type == 'mus': return 150
    if asset_type == 'amb': return 60
    if asset_type == 'sting': return 10
    if asset_type == 'sfx':
        if any(word in prompt_l for word in ['rumble', 'earthquake', 'vibration']): return 20
        return 8
    return 8

def ensure_model_loaded(model_id):
    global current_model_id, active_pipeline
    if current_model_id == model_id: return active_pipeline
    
    print(f"\n[VRAM] >>> Cambio Modello: {current_model_id or 'NESSUNO'} -> {model_id}")
    if active_pipeline is not None:
        print(f"[VRAM] --- Scaricamento {current_model_id} per liberare memoria...")
        del active_pipeline
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    print(f"[VRAM] +++ Caricamento {model_id} in corso...")
    if "musicgen" in model_id.lower():
        active_pipeline = MusicGen.get_pretrained(model_id)
    else:
        active_pipeline = get_diffusers_pipeline(model_id)
    
    current_model_id = model_id
    print(f"[VRAM] ✅ Modello {model_id} pronto.")
    return active_pipeline

def generate_asset(model_id, prompt, duration):
    pipeline = ensure_model_loaded(model_id)
    print(f"\n[AI] 🎬 Generazione in corso...")
    print(f"     Prompt: '{prompt[:70]}...'")
    print(f"     Durata: {duration}s | Modello: {model_id}")
    
    with torch.inference_mode():
        if "musicgen" in model_id.lower():
            pipeline.set_generation_params(duration=duration)
            wav = pipeline.generate([prompt])
            # MusicGen restituisce Tensore [B, C, T]
            return wav[0].cpu().numpy(), pipeline.sample_rate
        
        elif "audioldm2" in model_id.lower():
            output = pipeline(prompt, num_inference_steps=200, audio_length_in_s=duration)
            audio = output.audios[0]
            if torch.is_tensor(audio): audio = audio.cpu().numpy()
            return audio, 16000
            
        elif "stable-audio" in model_id.lower():
            # Stable Audio Open 1.0
            output = pipeline(prompt, num_inference_steps=200, audio_end_in_s=duration)
            audio = output.audios[0]
            # Protezione CUDA -> CPU
            if torch.is_tensor(audio): audio = audio.cpu().numpy()
            return audio, 44100
            
    return None, None


def save_asset(audio_np, sample_rate, asset_type, asset_id, prompt, model_id, tags=None):
    # Mapping tipi -> cartelle (tutto al singolare)
    mapping = {
        'mus': 'pad',
        'amb': 'amb',
        'sfx': 'sfx',
        'sting': 'sting'
    }
    folder_type = mapping.get(asset_type, 'sfx')
    asset_dir = ARIA_ROOT / "data" / "assets" / "sound_library" / folder_type / asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    output_path = asset_dir / f"{asset_id}.wav"
    
    # Conversione forzata in float32 (soundfile non supporta float16)
    import numpy as np
    audio_data = audio_np.astype(np.float32)
    
    # DIAGNOSTICA (Talking Logs)
    print(f"[DEBUG] Tentativo salvataggio {asset_id}:")
    print(f"        Shape originale: {audio_np.shape} | Dtype: {audio_np.dtype}")
    print(f"        Shape conversione: {audio_data.shape} | Dtype: {audio_data.dtype}")
    print(f"        Valori: min={np.min(audio_data):.4f}, max={np.max(audio_data):.4f}")

    # Soundfile si aspetta (samples, channels)
    if len(audio_data.shape) > 1 and audio_data.shape[0] < audio_data.shape[1]:
        audio_to_save = audio_data.T
    else:
        audio_to_save = audio_data
        
    print(f"        Final Shape per disk: {audio_to_save.shape}")

    # Salvataggio finale
    sf.write(str(output_path), audio_to_save, sample_rate)


    
    # Calcolo durata basato su NumPy
    total_samples = audio_to_save.shape[0]
    duration_actual = round(total_samples / sample_rate, 2)

    profile = {
        "id": asset_id, "category": asset_type, "name": asset_id.replace("_", " ").title(),
        "description": prompt,
        "prompt": prompt, "tags": tags if tags else [],
        "technical": {

            "duration": duration_actual,
            "sample_rate": sample_rate, "model": model_id, "vram_optimized": True
        },
        "registry": { "author": "Sound Factory Universal v1.5", "blackwell_native": True }
    }
    with open(asset_dir / "profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4)
    print(f"[DISK] 📦 Asset {asset_id} catalogato in '{folder_type}'!")


def print_summary(all_tasks):
    print("\n" + "="*80)
    print(" 🛠️  ARIA SOUND FACTORY - PIANO DI PRODUZIONE")
    print("="*80)
    print(f"{'TYPE':<6} | {'MODEL':<35} | {'ASSET ID'}")
    print("-" * 80)
    for t in all_tasks:
        print(f"{t['type']:<6} | {t['model']:<35} | {t['id']}")
    print("="*80 + "\n")

def run_batch_json(json_paths):
    STATS["start_time"] = time.time()
    all_tasks = []
    
    print(f"\n[INGESTION] >>> Analisi Shopping List in corso...")
    
    for jpath in json_paths:
        if not os.path.exists(jpath):
            print(f"[WARNING] ⚠️ File {jpath} non trovato. Lo salto.")
            continue
        with open(jpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            missing = data.get('missing_assets', [])
            print(f"[INGESTION] --- Trovati {len(missing)} asset in {os.path.basename(jpath)}")
            for item in missing:
                atype = item.get('type', 'sfx')
                prompt = item.get('universal_prompt', '')
                if not prompt: continue
                
                model_id = get_model_id(atype)
                duration = item.get('duration', get_smart_duration(atype, prompt))
                slug = slugify(prompt)
                asset_id = f"{atype}_{'_'.join(slug.split('_')[:4])}"
                
                all_tasks.append({
                    "id": asset_id, "type": atype, "prompt": prompt,
                    "duration": duration, "model": model_id, "tags": slug.split('_')
                })

    if not all_tasks:
        print("[TERMINAL] ❌ Nessun asset da produrre. Esco.")
        return

    # ORDINAMENTO VRAM
    all_tasks.sort(key=lambda x: x['model'])
    
    # RIEPILOGO VISIVO
    print_summary(all_tasks)

    for i, task in enumerate(all_tasks):
        print(f"\n[PROGRESSO] 🏭 Produzione Asset {i+1}/{len(all_tasks)}")
        
        mapping = {'mus': 'pad', 'amb': 'amb', 'sfx': 'sfx', 'sting': 'sting'}
        folder_type = mapping.get(task['type'], 'sfx')
        if (ARIA_ROOT / "data" / "assets" / "sound_library" / folder_type / task['id'] / "profile.json").exists():
            print(f"[SKIP] ⏩ Asset {task['id']} già presente in Warehouse.")
            STATS["skipped"] += 1
            continue
            
        try:
            audio_np, sr = generate_asset(task['model'], task['prompt'], task['duration'])
            if audio_np is not None:
                save_asset(audio_np, sr, task['type'], task['id'], task['prompt'], task['model'], task['tags'])
                STATS["success"] += 1
            else:
                print(f"[FAIL] ❌ Generazione fallita per {task['id']}.")
                STATS["failure"] += 1
        except Exception as e:
            print(f"[ERR] 💥 Errore critico durante {task['id']}: {e}")
            STATS["failure"] += 1

    # REPORT FINALE
    total_time = time.time() - STATS["start_time"]
    print("\n" + "!"*80)
    print(" 🏁 PRODUZIONE COMPLETATA!")
    print("!"*80)
    print(f" ✅ Successi:   {STATS['success']}")
    print(f" ❌ Fallimenti: {STATS['failure']}")
    print(f" ⏩ Saltati:    {STATS['skipped']}")
    print(f" ⏱️  Tempo Tot: {total_time:.2f} secondi")
    print("!"*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="ARIA Sound Factory - Universal NumPy-Native Executor")
    parser.add_argument("--json", nargs='+', help="Percorsi di uno o più file Shopping List JSON")
    args = parser.parse_args()
    if args.json:
        run_batch_json(args.json)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
