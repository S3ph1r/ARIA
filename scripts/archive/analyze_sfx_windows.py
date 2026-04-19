import numpy as np
import scipy.io.wavfile
from pathlib import Path

OUTPUT_DIR = Path("c:/Users/Roberto/aria/data/assets/quality_tests")

def analyze_peaks(f_name):
    path = OUTPUT_DIR / f_name
    if not path.exists():
        print(f"File non trovato: {f_name}")
        return
    
    sr, data = scipy.io.wavfile.read(path)
    # Normalizziamo a 0-1
    audio = np.abs(data.astype(np.float32) / 32768.0)
    if len(audio.shape) > 1:
        audio = np.max(audio, axis=1) # Max tra i canali per ogni campione (assumendo shape campioni, canali da scipy)
    
    win_size = int(sr * 0.1) # Finestre da 100ms
    print(f"\n--- Analisi Picchi (100ms steps): {f_name} ---")
    
    for i in range(0, len(audio), win_size):
        window = audio[i:i+win_size]
        if len(window) == 0: continue
        peak = np.max(window)
        timestamp = i / sr
        # Stampiamo solo il "profilo" per vedere quando cade
        bar = "#" * int(peak * 50)
        print(f"{timestamp:.1f}s | {peak:.4f} {bar}")

analyze_peaks("test_heavy_door.wav")
analyze_peaks("test_laser_blast.wav")
