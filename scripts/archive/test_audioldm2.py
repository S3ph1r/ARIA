import torch
import scipy.io.wavfile
from diffusers import AudioLDM2Pipeline
from pathlib import Path

def test_generation():
    print(">>> Inizializzazione AudioLDM 2 (Blackwell Optimized)...")
    
    # Caricamento modello in float16
    model_id = "cvssp/audioldm2"
    pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    # Prompt per un rumore sismico secco e sordo (niente musica!)
    prompt = "Low-frequency deep earthquake rumble, ground vibration, tectonic plates shifting, no music, realistic sound effect"
    negative_prompt = "music, melody, rhythm, instruments, bright, clean"
    
    print(f">>> Generazione SFX per: '{prompt}'...")
    
    # Generazione (5 secondi)
    audio = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=50, 
        audio_length_in_s=5.0
    ).audios[0]
    
    # Salvataggio
    output_path = Path("data/production/test_sfx_audioldm2.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # AudioLDM produce audio a 16kHz o 48kHz a seconda del modello, audioldm2 default 16k o 44k
    # Nota: scipy vuole l'audio normalizzato se float o int16
    scipy.io.wavfile.write(str(output_path), rate=16000, data=audio)
    
    print(f">>> SUCCESS: Asset di prova salvato in {output_path}")

if __name__ == "__main__":
    try:
        test_generation()
    except Exception as e:
        print(f"!!! ERRORE durante la generazione: {e}")
