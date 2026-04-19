import torch
import torchaudio
import time
from pathlib import Path
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import convert_audio

def run_demucs(audio_path, output_dir):
    print("--- ARIA HTDemucs Test ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device detect: {device}")
    
    print("Inizializzazione HTDemucs...")
    model = get_model(name="htdemucs")
    model.to(device)
    model.eval()

    print(f"Caricamento traccia originaria: {Path(audio_path).name}")
    
    # Integrazione DLL per Windows/Conda
    import os
    import sys
    conda_bin = os.path.join(sys.prefix, 'Library', 'bin')
    if os.path.exists(conda_bin):
        os.add_dll_directory(conda_bin)

    from torchcodec.decoders import AudioDecoder
    import soundfile as sf # Lo teniamo solo per il salvataggio finale

    decoder = AudioDecoder(audio_path)
    # in torchcodec 0.11.0, get_all_samples() restituisce un oggetto AudioSamples
    # i dati veri e propri sono nel campo .data (che è un Tensor)
    wav = decoder.get_all_samples().data.float()
    sr = decoder.metadata.sample_rate # Accesso corretto ai metadati in 0.11.0
    
    # Conversione sample rate e canali richiesti dal modello
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
    wav = wav.unsqueeze(0).to(device) # Batch dimension
    
    print("Inizio scissione acustica (Stem Separation)...")
    start_time = time.time()
    
    with torch.no_grad():
        # Applica separazione
        sources = apply_model(model, wav, shifts=1, split=True, overlap=0.25)[0]
    
    end_time = time.time()
    print(f"Scissione completata in {end_time - start_time:.2f} secondi!")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Salvataggio Stems...")
    for name, src in zip(model.sources, sources):
        src = src.cpu()
        if name == "vocals":
            print(" -> Scartato: vocals (inutile per i PAD musicali)")
            continue
        elif name == "other":
            out_name = "melody.wav"
        else:
            out_name = f"{name}.wav"
            
        out_path = output_dir / out_name
        sf.write(out_path, src.numpy().T, model.samplerate)
        print(f"[OK] Salvato stem: {out_name}")

if __name__ == "__main__":
    audio_file = r"C:\Users\Roberto\aria\data\assets\sound_library\pad\mus_retro_futuristic_dread\mus_retro_futuristic_dread.wav"
    out_dir = r"C:\Users\Roberto\aria\tmp\demucs_test"
    run_demucs(audio_file, out_dir)
