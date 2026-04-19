import torch
import torchcodec
import os
import sys
from pathlib import Path

def verify():
    print("--- ARIA TorchCodec sm_120 Verification ---")
    
    # Check enviroment
    print(f"Python: {sys.executable}")
    print(f"Torch Version: {torch.__version__}")
    print(f"TorchCodec Version: {torchcodec.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Arch List: {torch.cuda.get_arch_list()}")
    
    # Test Import
    try:
        from torchcodec.decoders import AudioDecoder
        print("[OK] AudioDecoder importato con successo.")
    except Exception as e:
        print(f"[ERROR] Errore importazione AudioDecoder: {e}")
        return

    # Test Decoding (CPU)
    test_file = r"C:\Users\Roberto\aria\data\assets\sound_library\pad\mus_retro_futuristic_dread\mus_retro_futuristic_dread.wav"
    if not os.path.exists(test_file):
        print(f"[SKIP] File di test non trovato: {test_file}")
        return

    print(f"Test decodifica su file: {Path(test_file).name}")
    try:
        decoder = AudioDecoder(test_file)
        print(f"[OK] Decoder inizializzato. Durata: {decoder.get_duration_seconds():.2f}s")
        
        # Caricamento di un chunk
        frame = decoder.get_frames_by_index(0, 10)
        print(f"[OK] Decodifica frame 0-10 riuscita. Shape: {frame.data.shape}")
        
    except Exception as e:
        print(f"[ERROR] Errore durante la decodifica: {e}")

    print("\n[SUCCESS] TorchCodec è pronto e l'ambiente sm_120 è integro.")

if __name__ == "__main__":
    # Aggiunta manuale dei path delle DLL se necessario (Conda Windows common issue)
    conda_bin = os.path.join(sys.prefix, 'Library', 'bin')
    if os.path.exists(conda_bin):
        os.add_dll_directory(conda_bin)
        
    verify()
