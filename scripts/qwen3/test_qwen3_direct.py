"""
ARIA — Test Qwen3-TTS Standalone (QW-0)
========================================
Verifica la qualità del modello prima di integrarlo in ARIA.
Genera due file WAV:
  - test_synth.wav    : sintesi senza voice cloning (speaker built-in)
  - test_cloned.wav   : sintesi con voice cloning dal sample del narratore

Usage:
    conda activate qwen3-tts
    cd C:\\Users\\Roberto\\aria
    python test_qwen3_direct.py [--voice narratore] [--model C:\\models\\qwen3-tts-1.7b]
"""

import argparse
import time
import sys
import os
from pathlib import Path

import torch
import soundfile as sf
import numpy as np

# ======================================================
# TESTI DI TEST (estratti da Cronache del Silicio)
# Includono parole con accento grafico per verificare
# che il modello rispetti la pronuncia italiana
# ======================================================
TEST_TEXTS = {
    "breve": (
        "La pàtina del tempo rivelava i segni di un'epoca lontana. "
        "Adso posò il futòn sul pavimento di pietra e si avvicinò alla finestra."
    ),
    "medio": (
        "Si dice che una città non sia fatta di cemento e luce, ma di storie. "
        "Neo-Kyoto era un'antologia infinita scritta in una lingua che nessuno poteva più leggere per intero. "
        "Era una città costruita su strati di altre città, un palinsesto di futuri sognati e passati dimenticati. "
        "Viveva in uno stato di presente perpetuo, illuminata da un sole artificiale."
    ),
    "lungo": (
        "Kaelen aprì gli occhi nel buio del suo appartamento al ventunesimo piano. "
        "Il vibrare del terminale lo aveva strappato da un sogno che già sbiadiva — qualcosa di verde, "
        "di organico, di impossibile in questa città di vetro e neon. "
        "Si passò una mano sul viso e rimase immobile un momento, lasciando che la realtà si ricomponesse "
        "intorno a lui come un ologramma che si ricalibra. "
        "Fuori dalla finestra, Neo-Kyoto pulsava nella notte con la sua solita intensità indifferente."
    )
}

INSTRUCT_MAP = {
    "neutral":  "Warm Italian male voice, professional audiobook narrator, calm and measured, moderate pace.",
    "suspense": "Warm Italian male voice, professional audiobook narrator, tense and restrained, hushed intensity, slow pace.",
}


def load_model(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nCaricamento modello su {device}...")
    print(f"Path: {model_path}")

    t0 = time.time()
    try:
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        mode = "flash_attention_2"
    except Exception as e:
        print(f"[WARN] Flash attention non disponibile ({type(e).__name__}: {e})")
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
        )
        mode = "standard"

    load_time = time.time() - t0
    vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
    print(f"Modello caricato in {load_time:.1f}s | VRAM: {vram:.2f} GB | Attn: {mode}")
    return model, device


def synthesize(model, text: str, instruct: str, ref_audio=None, ref_sr=None,
               ref_text=None, language="Italian") -> tuple:
    """Genera audio e restituisce (wav_array, sample_rate, inference_time_s)."""
    t0 = time.time()

    if ref_audio is None:
        raise ValueError("Base model requires ref_audio for voice cloning. "
                         "Use Qwen3-TTS-12Hz-1.7B-CustomVoice for speaker-only synthesis.")

    # generate_voice_clone accetta ref_audio come numpy array + sr come tuple
    ref_input = (ref_audio, ref_sr) if ref_sr is not None else ref_audio

    # Se non c'è testo di riferimento per il cloning (zero-shot),
    # il modello Base richiede obbligatoriamente x_vector_only_mode=True
    is_x_vector_only = (ref_text is None)

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_input,
        ref_text=ref_text,
        x_vector_only_mode=is_x_vector_only,
        non_streaming_mode=True,
        instruct=instruct,
        max_new_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    wav = wavs[0] if isinstance(wavs, list) else wavs
    inference_time = time.time() - t0
    return wav, sr, inference_time


def main():
    parser = argparse.ArgumentParser(description="Test qualità Qwen3-TTS (QW-0)")
    parser.add_argument("--model", default=r"C:\Users\Roberto\aria\data\models\qwen3-tts-1.7b",
                        help="Path locale al modello")
    parser.add_argument("--voice", default="narratore",
                        help="ID voce nella voice library (default: narratore)")
    parser.add_argument("--voices-dir",
                        default=r"C:\Users\Roberto\aria\data\voices",
                        help="Directory voice library")
    parser.add_argument("--out-dir", default=r"C:\Users\Roberto\aria\data\outputs\qwen3_test",
                        help="Directory output WAV")
    parser.add_argument("--text-key", default="breve", choices=list(TEST_TEXTS.keys()),
                        help="Testo di test da usare")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Carica modello
    model, device = load_model(args.model)

    text = TEST_TEXTS[args.text_key]
    instruct_neutral  = INSTRUCT_MAP["neutral"]
    instruct_suspense = INSTRUCT_MAP["suspense"]

    print(f"\nTesto di test ({args.text_key}): {text[:80]}...")
    print("\n[NOTE] Modello Base: richiede sempre reference audio (voice cloning).")
    print("       Per sintesi senza cloning, usa Qwen3-TTS-12Hz-1.7B-CustomVoice.\n")

    # =========================================================
    # Risolvi riferimento audio
    # =========================================================
    voice_dir   = Path(args.voices_dir) / args.voice
    ref_padded  = voice_dir / "ref_padded.wav"
    ref_plain   = voice_dir / "ref.wav"
    ref_path    = ref_padded if ref_padded.exists() else ref_plain

    if not ref_path.exists():
        print(f"[ERRORE] ref.wav non trovato in {voice_dir}. Impossibile procedere.")
        sys.exit(1)

    print(f"Reference audio: {ref_path.name} ({args.voice})")
    ref_audio, ref_sr = sf.read(str(ref_path))
    if ref_audio.ndim > 1:
        ref_audio = ref_audio[:, 0]

    ref_txt_path = voice_dir / "ref.txt"
    ref_text = None
    if ref_txt_path.exists():
        try:
            ref_text = ref_txt_path.read_text(encoding="utf-8").strip()
            print(f"ref.txt: \"{ref_text[:60]}...\"")
        except Exception:
            ref_text = None

    # =========================================================
    # TEST 1 — Voice cloning, emozione neutral
    # =========================================================
    print(f"\n[TEST 1] Voice cloning + emozione neutral...")
    wav, sr, t = synthesize(model, text, instruct_neutral, ref_audio, ref_sr, ref_text)
    duration = len(wav) / sr
    rtf = t / duration
    out1 = out_dir / f"test_cloned_{args.voice}_neutral.wav"
    sf.write(str(out1), wav, sr)
    print(f"  OK: {out1.name} | {duration:.1f}s audio in {t:.1f}s | RTF {rtf:.2f}x")

    # =========================================================
    # TEST 2 — Voice cloning, emozione suspense
    # =========================================================
    print(f"\n[TEST 2] Voice cloning + emozione suspense...")
    wav, sr, t = synthesize(model, text, instruct_suspense, ref_audio, ref_sr, ref_text)
    duration = len(wav) / sr
    rtf = t / duration
    out2 = out_dir / f"test_cloned_{args.voice}_suspense.wav"
    sf.write(str(out2), wav, sr)
    print(f"  OK: {out2.name} | {duration:.1f}s audio in {t:.1f}s | RTF {rtf:.2f}x")

    # Riepilogo VRAM
    if device == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nVRAM peak: {peak_vram:.2f} GB")

    print(f"\nFile generati in: {out_dir}")
    print("\n=== CHECKLIST QW-0 POST-ASCOLTO ===")
    print("[ ] Timbro riconoscibile come voce del narratore")
    print("[ ] 'pàtina' pronunciato con accento sulla prima a")
    print("[ ] 'futòn' pronunciato con accento sulla o")
    print("[ ] Nessun artefatto sonoro iniziale (se usato ref_padded.wav)")
    print("[ ] RTF > 1.0x  (audio più lungo del tempo di generazione)")
    print("")
    print("GO se tutti i check sono OK → procedi con QW-1 (start-qwen3-tts.bat)")


if __name__ == "__main__":
    main()
