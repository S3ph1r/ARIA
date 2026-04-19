import os
import sys
from pathlib import Path

# Inseriamo il percorso del repository ACE-Step nel path di sistema
sys.path.append(os.path.join(os.getcwd(), "ACE-Step-1.5"))

from acestep.model_downloader import get_checkpoints_dir, MAIN_MODEL_COMPONENTS, _contains_model_weights

def debug_checkpoints():
    checkpoints_dir = get_checkpoints_dir()
    print(f"--- ACE-Step Checkpoint Debug ---")
    print(f"Project Root: {os.getcwd()}")
    print(f"Checkpoints Dir (detected): {checkpoints_dir}")
    print(f"Exists: {checkpoints_dir.exists()}")
    
    if checkpoints_dir.exists():
        print(f"Contents of checkpoints dir:")
        for item in checkpoints_dir.iterdir():
            print(f"  - {item.name} ({'Dir' if item.is_dir() else 'File'})")

    print("\n--- Component Check ---")
    for component in MAIN_MODEL_COMPONENTS:
        component_path = checkpoints_dir / component
        exists = component_path.exists()
        has_weights = _contains_model_weights(component_path) if exists else False
        
        status = "OK" if has_weights else ("MISSING WEIGHTS" if exists else "NOT FOUND")
        print(f"Component: {component}")
        print(f"  Path: {component_path}")
        print(f"  Status: {status}")
        
        if exists and not has_weights:
            print(f"  Files inside {component}:")
            for f in component_path.iterdir():
                print(f"    - {f.name}")

if __name__ == "__main__":
    debug_checkpoints()
