import numpy as np
import scipy.io.wavfile
from pathlib import Path

OUTPUT_DIR = Path("c:/Users/Roberto/aria/data/assets/quality_tests")

def analyze_envelope(f_name):
    path = OUTPUT_DIR / f_name
    if not path.exists():
        print(f"File non trovato: {f_name}")
        return
    
    sr, data = scipy.io.wavfile.read(path)
    # Convertire in float e mono per l'analisi
    audio = data.astype(np.float32)
    if audio.dtype != np.float32: # Se scipy ha letto int16
        audio = audio / 32768.0
        
    if len(audio.shape) > 1:
        audio = np.max(np.abs(audio), axis=1)
    else:
        audio = np.abs(audio)
        
    max_val = np.max(audio)
    if max_val == 0:
        print(f"{f_name} è completamente silenzioso.")
        return

    win_size = int(sr * 0.01) # Finestre da 10ms (Alta Risoluzione)
    print(f"\n--- Analisi Picchi High-Res (10ms): {f_name} ---")
    print(f"Picco Massimo Assoluto: {max_val:.4f}")
    
    # Analizziamo solo dai 0.5s in poi (dove dovrebbe esserci il decadimento)
    start_frame = int(sr * 0.5)
    for i in range(start_frame, len(audio), win_size):
        window = audio[i:i+win_size]
        if len(window) == 0: continue
        peak = np.max(window)
        percent = (peak / max_val) * 100
        timestamp = i / sr
        if percent > 5: # Mostriamo solo ciò che è rilevante
            bar = "#" * int(percent / 2)
            print(f"{timestamp:.2f}s | {percent:6.2f}% {bar}")

analyze_envelope("test_heavy_door.wav")
analyze_envelope("test_laser_blast.wav")
