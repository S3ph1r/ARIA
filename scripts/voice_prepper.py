#!/usr/bin/env python3
"""
Voice Prepper 🎙️
Automa l'estrazione di voice samples da YouTube per ARIA Voice Library.
Scarica, taglia, e usa l'intelligenza artificiale (Gemini) per trascrivere l'esatto testo.
"""

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path

# Tentativo di importare le librerie necessarie, mostra messaggi chiari se mancano
try:
    from dotenv import load_dotenv
except ImportError:
    print("[!] Modulo 'python-dotenv' non trovato. Assicurati di essere in un virtual environment (es. DIAS .venv).")
    exit(1)

try:
    from google import genai
except ImportError:
    print("[!] Modulo 'google-genai' non trovato. Assicurati di averlo installato (pip install google-genai).")
    exit(1)

def setup_env():
    # Tenta di caricare il .env da DIAS (se lo script è eseguito in questo server)
    dias_env_path = Path("/home/Projects/NH-Mini/sviluppi/dias/.env")
    if dias_env_path.exists():
        load_dotenv(dias_env_path)
    else:
        # Fallback alla directory corrente
        load_dotenv()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[!] GOOGLE_API_KEY non trovata nelle variabili d'ambiente o in .env")
        exit(1)
    
    return api_key

def transcribe_audio(audio_path: str, api_key: str) -> str:
    print("[*] Avvio trascrizione con Gemini...")
    client = genai.Client(api_key=api_key)
    
    # Upload del file
    print("    - Upload del file audio...")
    audio_file = client.files.upload(file=audio_path)
    
    print("    - Generazione testo...")
    prompt = """
    Trascrivi ESATTAMENTE parola per parola quello che viene detto in questo audio.
    Ignora suoni di sottofondo, musica o rumori, scrivi solo il parlato.
    Inserisci la punteggiatura corretta (virgole, punti, punti di domanda) in base all'intonazione.
    Non aggiungere nessun tipo di prefisso o commento, restituisci SOLO la trascrizione nuda e cruda.
    La lingua è l'italiano.
    """
    
    try:
        # Usa il modello flash per la trascrizione (veloce e supporta audio)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[audio_file, prompt]
        )
        testo = response.text.strip()
        print("    [+] Trascrizione completata!")
        return testo
    except Exception as e:
        print(f"    [!] Errore durante la trascrizione: {e}")
        return ""
    finally:
        # Pulisci il file caricato
        try:
            client.files.delete(name=audio_file.name)
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="ARIA Voice Prepper: Estrai campioni vocali da YouTube.")
    parser.add_argument("url", help="URL del video YouTube (es. https://www.youtube.com/watch?v=XYZ)")
    parser.add_argument("voice_id", help="Nome/ID della voce (es. narratore, giulia, marco)")
    parser.add_argument("--start", help="Tempo di inizio (es. 00:10 o 10)", required=True)
    parser.add_argument("--end", help="Tempo di fine (es. 00:30 o 30)", required=True)
    
    args = parser.parse_args()
    api_key = setup_env()
    
    # Path di output in ARIA (sia su Linux che Windows, ma noi gestiamo il path dove sta girando lo script)
    # Assumiamo che la cartella ARIA sia vicina o che siamo nella root del progetto
    aria_root = Path("/home/Projects/NH-Mini/sviluppi/ARIA")
    if not aria_root.exists():
        # Fallback al path Windows standard per coerenza se eseguito di là
        aria_root = Path(os.path.expanduser("~")) / "aria"
        
    voices_dir = aria_root / "data" / "voices" / args.voice_id
    os.makedirs(voices_dir, exist_ok=True)
    
    ref_wav_dest = voices_dir / "ref.wav"
    ref_txt_dest = voices_dir / "ref.txt"
    
    print(f"\n🎙️ Voice Prepper avviato per la voce: {args.voice_id}")
    print(f"[*] Destinazione: {voices_dir}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_audio = os.path.join(tmpdir, "raw_audio.wav")
        sliced_audio = os.path.join(tmpdir, "sliced_audio.wav")
        
        # 1. Download con yt-dlp
        print(f"\n[1] Scaricamento audio da {args.url} ...")
        cmd_yt = [
            "yt-dlp",
            "-x", "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", raw_audio,
            args.url
        ]
        res = subprocess.run(cmd_yt, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print("[!] Errore durante il download con yt-dlp:")
            print(res.stderr.decode("utf-8", errors="ignore"))
            exit(1)
            
        print("    [+] Download completato.")
        
        # 2. Taglio con ffmpeg (e conversione a mono, 44100Hz per pulizia)
        print(f"\n[2] Taglio audio da {args.start} a {args.end} ...")
        cmd_ff = [
            "ffmpeg", "-y",
            "-i", raw_audio,
            "-ss", str(args.start),
            "-to", str(args.end),
            "-ac", "1",           # Mono (richiesto da molti TTS)
            "-ar", "44100",       # Sample rate standard
            sliced_audio
        ]
        res = subprocess.run(cmd_ff, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print("[!] Errore durante il taglio con ffmpeg:")
            print(res.stderr.decode("utf-8", errors="ignore"))
            exit(1)
            
        print("    [+] Taglio e conversione completati.")
        
        # 3. Trascrizione con Gemini
        print("\n[3] Trascrizione esatta in corso...")
        transcript = transcribe_audio(sliced_audio, api_key)
        
        if not transcript:
            print("[!] Trascrizione fallita. Sarà necessario scriverla a mano.")
            transcript = "--- SOSTITUIRE CON LA TRASCRIZIONE MANUALE ---"
            
        ref_padded_dest = voices_dir / "ref_padded.wav"

        # 4. Salvataggio finale
        print("\n[4] Salvataggio dei file nella Voice Library...")
        import shutil
        shutil.copy2(sliced_audio, ref_wav_dest)
        print(f"    [+] {ref_wav_dest.name} salvato (Standard).")
        
        print("    [+] Generazione ref_padded.wav (padding 0.5s per Qwen3)...")
        cmd_pad = [
            "ffmpeg", "-y",
            "-i", sliced_audio,
            "-af", "apad=pad_dur=0.5",
            str(ref_padded_dest)
        ]
        res_pad = subprocess.run(cmd_pad, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res_pad.returncode != 0:
            print("    [!] Avviso: Fallita generazione ref_padded.wav con ffmpeg.")
        else:
            print(f"    [+] {ref_padded_dest.name} salvato (Qwen3-Ready).")
        
        with open(ref_txt_dest, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"    [+] {ref_txt_dest.name} salvato.")
        
    print("\n✅ PROCESSO COMPLETATO CON SUCCESSO!")
    print(f"La voce '{args.voice_id}' è pronta in ARIA per il voice cloning.")
    print("\nTesto trascritto (controllalo!):")
    print("-" * 50)
    print(transcript)
    print("-" * 50)

if __name__ == "__main__":
    main()
