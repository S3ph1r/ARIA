# MISSION SUMMARY: Sound Library & Music Environment Setup

**Data**: 04 Aprile 2026  
**Stato**: Handover da LXC 190 a PC 139  

## 🎯 Obiettivo Corrente
Implementare lo **Stage B2 (Sound Director)** e configurare la **Sound Library** deterministica sul PC Worker (PC 139).

## 🔑 Credenziali e Accessi (PC 139 - Windows)
- **User**: `roberto`
- **Password**: `Farscape@666`
- **Path ARIA**: `C:\Users\roberto\aria`
- **SSH Status**: Chiavi allineate tra LXC 190 (root) e PC 139 (roberto). Accesso bidirezionale configurato.

## 🏗️ Struttura Sound Library da Creare
Il "Magazzino" deve risiedere in `aria/data/assets/sound_library/`:
- `pads/`: Tappeti sonori (Stem A).
- `stings/`: Effetti brevi 3-8s (Stem C).
- `sound_catalog.json`: Il database che l'LLM di Stage B2 dovrà interrogare.

## 🛠️ Task Immediati su PC 139
1.  **Ambiente Conda**: Creare un ambiente Python 3.10 (es. `audiocraft`) per gestire la generazione via `MusicGen`.
2.  **Dipendenze**: Installare `audiocraft`, `torch`, `torchaudio` (puntando a CUDA 12.x).
3.  **Catalog Setup**: Inizializzare `sound_catalog.json` con IDs di base (`pad_dark_tension`, `sting_bass_drop`, etc.).
4.  **Factory Mode**: Preparare lo script di generazione batch per generare i primi WAV usando la RTX 5060 Ti locale.

## 📝 Note per l'Agente su PC 139
- Il documento di riferimento principale è `docs/DIAS StageB2 e Soundlibrary.md` (già presente nel repo).
- Non utilizzare SSH per operare localmente su Windows se il workspace è già aperto lì.
- Utilizzare PowerShell per i comandi nativi di Windows/Conda.

---
**Firmato**: Antigravity (Session LXC 190)
