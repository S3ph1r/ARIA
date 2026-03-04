#!/usr/bin/env python3
import os
import subprocess
import sys

# IP del PC Gaming
TARGET_HOST = "192.168.1.139"
TARGET_USER = "gemini"
TARGET_PASS = "gemini!"
TARGET_DIR = "C:/Users/Roberto/aria/envs/fish-speech"

# Cartella locale su LXC 190
LOCAL_BACKEND_DIR = "/home/Projects/NH-Mini/sviluppi/ARIA/external/fish-speech"

def run_scp(local_path, remote_path):
    cmd = f"sshpass -p '{TARGET_PASS}' scp -o StrictHostKeyChecking=no -r {local_path} {TARGET_USER}@{TARGET_HOST}:{remote_path}"
    print(f"🚀 Deploying {local_path} -> {remote_path}...")
    subprocess.run(cmd, shell=True, check=True)

def main():
    if not os.path.exists(LOCAL_BACKEND_DIR):
        print(f"❌ Errore: Cartella locale {LOCAL_BACKEND_DIR} non trovata.")
        sys.exit(1)
        
    try:
        # Sincronizza fish_speech logic
        run_scp(f"{LOCAL_BACKEND_DIR}/fish_speech/*", f"{TARGET_DIR}/fish_speech/")
        
        # Sincronizza tools
        run_scp(f"{LOCAL_BACKEND_DIR}/tools/*", f"{TARGET_DIR}/tools/")
        
        # Sincronizza top-level files (requirements, etc)
        # Nota: usiamo scp individuale per evitare casini con cartelle gia esistenti
        for f in ["requirements.txt", "pyproject.toml"]:
            local_f = f"{LOCAL_BACKEND_DIR}/{f}"
            if os.path.exists(local_f):
                run_scp(local_f, f"{TARGET_DIR}/{f}")
                
        print("\n✅ Deploy completato con successo su 192.168.1.139")
    except Exception as e:
        print(f"\n❌ Errore durante il deploy: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
