import numpy as np
import scipy.io.wavfile
from pathlib import Path

OUTPUT_DIR = Path("c:/Users/Roberto/aria/data/assets/quality_tests")

files = ["test_heavy_door.wav", "test_laser_blast.wav"]

for f_name in files:
    path = OUTPUT_DIR / f_name
    if not path.exists():
        continue
    
    sr, data = scipy.io.wavfile.read(path)
    # Prendiamo l'ultimo secondo
    last_second = data[-sr:]
    max_amp = np.max(np.abs(last_second)) / 32768.0 # Normalizzato
    avg_amp = np.mean(np.abs(last_second)) / 32768.0
    
    print(f"File: {f_name}")
    print(f" - Ampiezza Max nell'ultimo secondo: {max_amp:.5f}")
    print(f" - Ampiezza Media nell'ultimo secondo: {avg_amp:.5f}")
    print("-" * 30)
