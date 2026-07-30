"""
check_model_updates.py — Controllo MANUALE di eventuali aggiornamenti dei
modelli usati da Lifelog WhisperX (server.py), da lanciare a mano ogni tanto.

NON scarica né aggiorna nulla: interroga solo HuggingFace Hub per la
revisione (commit sha + data ultima modifica) più recente di ciascun
repository pinnato, e la stampa a schermo. La decisione di aggiornare (e
verificarne l'impatto su trascrizione/diarizzazione/identità prima di
promuoverlo in produzione) resta manuale.

Non è collegato all'avvio del backend né a nessuno scheduler — girare questo
script non ha alcun effetto sul servizio in corso. Runnalo quando vuoi
sapere se vale la pena controllare, non più spesso.

Uso:
    python check_model_updates.py
"""

import sys
from datetime import datetime, timezone

from huggingface_hub import HfApi

# Repository pinnati da questo backend (vedi server.py e
# docs/backends/lifelog-whisperx.md §7 "Modelli su disco"). L'ASR è un caso
# a parte: la cartella locale si chiama "faster-whisper-large-v3" (formato
# CTranslate2), quasi certamente convertito da un repo tipo
# Systran/faster-whisper-large-v3 piuttosto che da openai/whisper-large-v3
# (pesi PyTorch originali, formato diverso) — non verificabile da qui con
# certezza, per questo è segnalato invece di essere dato per buono.
MODELS = {
    "whisper-large-v3 (ASR)": {
        "repo_id": "Systran/faster-whisper-large-v3",
        "note": "verifica il repo esatto la prima volta: la cartella locale "
                "e' gia' in formato CTranslate2, potrebbe non venire da qui",
    },
    "wav2vec2 align ITA": {
        "repo_id": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
        "note": None,
    },
    "pyannote diarization": {
        "repo_id": "pyannote/speaker-diarization-community-1",
        "note": "repo con accesso condizionato (gated) — serve HF_TOKEN valido",
    },
    "wespeaker voiceprint (resnet293)": {
        "repo_id": "Wespeaker/wespeaker-voxceleb-resnet293-LM",
        "note": None,
    },
}


def main() -> int:
    import os
    token = os.getenv("HF_TOKEN") or None
    api = HfApi(token=token)

    print("=== Controllo revisioni disponibili su HuggingFace Hub ===")
    print("(solo lettura — nessun download, nessuna modifica locale)\n")

    any_error = False
    for label, cfg in MODELS.items():
        repo_id = cfg["repo_id"]
        try:
            info = api.model_info(repo_id)
            last_mod = info.lastModified
            if isinstance(last_mod, datetime):
                age_days = (datetime.now(timezone.utc) - last_mod.replace(tzinfo=timezone.utc)).days
                age_str = f"{age_days} giorni fa"
            else:
                age_str = "sconosciuta"
            print(f"[{label}]")
            print(f"  repo:               {repo_id}")
            print(f"  ultima revisione:   {info.sha[:12]}")
            print(f"  ultima modifica:    {last_mod} ({age_str})")
            if cfg["note"]:
                print(f"  nota:               {cfg['note']}")
        except Exception as exc:
            any_error = True
            print(f"[{label}] repo={repo_id} — ERRORE nel controllo: {exc}")
        print()

    print(
        "Nessun confronto automatico con la versione installata: le cartelle "
        "locali (MODELS_DIR) non sono gestite dalla cache standard di "
        "huggingface_hub, quindi non c'e' un hash locale affidabile da "
        "confrontare da qui. Se una revisione sopra e' molto più recente di "
        "quando l'hai scaricata (vedi docs/backends/lifelog-whisperx.md §6-7 "
        "per le date di setup), puo' valere la pena investigare — ma testa "
        "sempre su un campione prima di sostituire un modello in produzione."
    )
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
