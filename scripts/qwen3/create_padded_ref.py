"""
ARIA — Creazione ref_padded.wav per Qwen3-TTS
==============================================
Fix obbligatorio per il phonetic bleeding: aggiunge 0.5s di silenzio
alla fine di ogni ref.wav nella voice library.

Usage:
    conda activate qwen3-tts
    cd C:\\Users\\Roberto\\aria
    python create_padded_ref.py [--voice VOICE_ID] [--silence 0.5]

Senza argomenti processa TUTTE le voci nella library.
"""

import os
import sys
import argparse
import soundfile as sf
import numpy as np
from pathlib import Path

def create_padded_ref(ref_wav_path: Path, silence_s: float = 0.5) -> Path:
    """
    Legge ref.wav, aggiunge silenzio finale, salva ref_padded.wav.
    Restituisce il path del file generato.
    """
    out_path = ref_wav_path.parent / "ref_padded.wav"

    audio, sr = sf.read(ref_wav_path)

    # Mono se stereo
    if audio.ndim > 1:
        audio = audio[:, 0]
        print(f"  [INFO] File stereo convertito in mono.")

    duration = len(audio) / sr
    print(f"  Durata originale: {duration:.2f}s  |  Sample rate: {sr}Hz")

    # Avviso se il sample è troppo lungo (>30s rischio loop infinito)
    if duration > 30:
        print(f"  [WARN] Sample più lungo di 30s! Considera di tagliarlo a 20s.")
        print(f"         Usa: ffmpeg -ss 0 -t 20 -i ref.wav ref.wav (backup prima)")
    elif duration < 3:
        print(f"  [WARN] Sample molto corto ({duration:.1f}s). Qualità cloning inferiore.")

    silence = np.zeros(int(sr * silence_s), dtype=audio.dtype)
    padded = np.concatenate([audio, silence])

    sf.write(out_path, padded, sr)
    print(f"  Creato: {out_path} ({len(padded)/sr:.2f}s, +{silence_s}s silenzio)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Crea ref_padded.wav per Qwen3-TTS")
    parser.add_argument("--voice", type=str, default=None,
                        help="ID voce specifica (es: narratore). Default: tutte.")
    parser.add_argument("--silence", type=float, default=0.5,
                        help="Durata silenzio finale in secondi (default: 0.5)")
    args = parser.parse_args()

    # Percorso base libreria voci
    script_dir = Path(__file__).parent
    voices_dir = Path(os.environ.get(
        "ARIA_VOICES_DIR",
        r"C:\Users\Roberto\aria\data\voices"
    ))

    if not voices_dir.exists():
        print(f"[ERRORE] Directory voci non trovata: {voices_dir}")
        print("Imposta ARIA_VOICES_DIR se il percorso è diverso.")
        sys.exit(1)

    # Raccogli voci da processare
    if args.voice:
        voice_dirs = [voices_dir / args.voice]
        if not voice_dirs[0].exists():
            print(f"[ERRORE] Voce '{args.voice}' non trovata in {voices_dir}")
            sys.exit(1)
    else:
        voice_dirs = [d for d in voices_dir.iterdir() if d.is_dir()]

    if not voice_dirs:
        print(f"[WARN] Nessuna directory voce trovata in {voices_dir}")
        sys.exit(0)

    print(f"\nProcessando {len(voice_dirs)} voce(i) in {voices_dir}\n")
    success = 0
    errors = 0

    for voice_dir in voice_dirs:
        ref_wav = voice_dir / "ref.wav"
        print(f"[{voice_dir.name}]")

        if not ref_wav.exists():
            print(f"  [SKIP] ref.wav non trovato.")
            continue

        try:
            create_padded_ref(ref_wav, args.silence)
            success += 1
        except Exception as e:
            print(f"  [ERRORE] {e}")
            errors += 1

    print(f"\nCompletato: {success} OK, {errors} errori")


if __name__ == "__main__":
    main()
