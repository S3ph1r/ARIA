import os
from safetensors import safe_open
import torch

def debug_safetensors():
    model_dir = r"C:\Users\Roberto\aria\data\assets\models\acestep-v15-xl-sft"
    files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    files.sort()
    
    total_keys = 0
    print(f"Inizio controllo integrità su {len(files)} file...")
    
    try:
        for filename in files:
            path = os.path.join(model_dir, filename)
            print(f"\nAnalisi {filename}...")
            with safe_open(path, framework="pt", device="cpu") as f:
                keys = f.keys()
                for i, key in enumerate(keys):
                    if i % 50 == 0:
                        print(f"  Controllati {i}/{len(keys)} pesi...")
                    # Forza il caricamento del tensore in memoria
                    tensor = f.get_tensor(key)
                    total_keys += 1
            print(f"File {filename} INTEGRO.")
            
        print(f"\n[SUCCESS] Tutti i {total_keys} pesi sono stati letti correttamente.")
    except Exception as e:
        print(f"\n[CRASH] Errore durante la lettura: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_safetensors()
