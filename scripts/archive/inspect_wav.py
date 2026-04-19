import scipy.io.wavfile
import numpy as np
import sys

def inspect(file_path):
    try:
        sr, data = scipy.io.wavfile.read(file_path)
        print(f">>> File: {file_path}")
        print(f">>> Sample Rate: {sr}")
        print(f">>> Shape: {data.shape}")
        print(f">>> Dtype: {data.dtype}")
        print(f">>> Min Val: {np.min(data)}")
        print(f">>> Max Val: {np.max(data)}")
        print(f">>> Mean: {np.mean(data)}")
        print(f">>> Abs Max: {np.max(np.abs(data))}")
        
        if np.max(np.abs(data)) < 1e-5:
            print("\n!!! ALERTA: Il file è praticamente muto (valori vicini allo zero).")
        else:
            print("\n>>> Il file contiene dati audio validi.")
            
    except Exception as e:
        print(f"Errore: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        inspect("data/assets/sfx/test_earthquake_02/test_earthquake_02.wav")
