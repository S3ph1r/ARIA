# ARIA — Music & Sound Effects Backend
## Specifiche Tecniche e Linee Guida di Implementazione

> **Stato**: 🔲 Progettazione / In Attesa di Implementazione
> **Tecnologia Principale**: Meta Audiocraft / MusicGen
> **Ambiente Consigliato**: Python 3.10+ (isolato)

---

## 🏗️ Architettura del Backend

Il backend per la musica e gli effetti sonori (SFX) funzionerà come un **External HTTP Backend** integrato nell'Orchestratore, seguendo il pattern degli altri servizi ARIA.

- **Queue**: `gpu:queue:music:{model_id}`
- **Porta Default**: `8086` (da confermare)
- **Modello di Riferimento**: `musicgen-medium` (per equilibrio qualità/VRAM) o `musicgen-small` (per velocità).

---

## 🔧 Specifiche dell'Ambiente (Anti-Nightmare)

Per evitare l'incubo delle dipendenze e del kernel CUDA su architetture **RTX 5000 (sm_120)**, l'ambiente Conda dovrà rigorosamente seguire queste specifiche:

| Componente | Versione / Specifica |
|------------|-----------------------|
| **Python** | 3.10 o 3.11 |
| **PyTorch**| `>= 2.7.0` (Stabile) |
| **CUDA**   | `12.8` |
| **Index URL**| `https://download.pytorch.org/whl/cu128` |

### ⚠️ Nota Critica sulle Dipendenze
`audiocraft` ha dipendenze strette con `ffmpeg` e librerie audio specifiche. Non tentare di installare `audiocraft` nell'ambiente di Fish-Speech: creane uno dedicato `%ARIA_ROOT%\envs\music-env\` per evitare conflitti di versione tra `transformers` e `audiocraft`.

---

## 🎵 Funzionalità Previste

### 1. Music Generation (MusicGen)
- **Modelli**: `facebook/musicgen-medium` (~3.2GB VRAM).
- **Parametri**: prompt testuale, durata (sec), fade-in/out, BPM.
- **Output**: WAV Stereo 48kHz.

### 2. Sound Effects (AudioGen) - *Futuro*
- Generazione di suoni ambientali (pioggia, passi, vento) tramite prompt.

---

## 📊 Performance e Vincoli Hardware

- **VRAM Occupata**: ~4-8 GB a seconda del modello caricato.
- **Coesistenza**: Con 16GB VRAM, il backend MusicGen **non può** coesistere con Qwen3 o Fish-Speech. L'Orchestratore deve eseguire lo scarico (`unload`) dei modelli TTS prima di caricare il backend musica.
- **Tempo di Generazione**: Stimato 1.5x rispetto alla durata dell'audio (es. 30s di audio = 45s di calcolo).

---
*ARIA Music Backend Spec — Marzo 2026*
