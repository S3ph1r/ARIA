# 🐟 Fish TTS Backend — Roadmap Operativa

> Checklist da spuntare step by step. Basata su `fish-tts-backend.md`.
> Ogni box `[ ]` diventa `[x]` quando completato.

---

## FS-0 — Preparazione Voice Samples (Automazione completata)

- [x] Sviluppato `scripts/voice_prepper.py` per automatizzare scraping e tagging
- [x] Scraping audio automatico via `yt-dlp -x --audio-format wav`
- [x] Taglio automatico via `ffmpeg` (mono, 44100 Hz)
- [x] Trascrizione automatizzata supportata dall'API `gemini-flash-lite-latest` (Gemini 3.1)
- [x] Salvataggio automatico nella Voice Library `data/assets/voices/<voice_id>/`
- [ ] Recuperare voice link di YouTube dal team e avviare il tool per generare il sample finale

---

## FS-1 — Setup Ambiente Windows (PC Gaming 192.168.1.139)

- [x] Installare Miniconda su Windows (se non presente)
- [x] Creare ambiente: `conda create -n fish-speech python=3.10`
- [x] Installare PyTorch 2.7+ cu128: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
- [x] Clonare repo: `git clone https://github.com/fishaudio/fish-speech.git C:\fish-speech`
- [x] Installare: `cd C:\fish-speech && pip install -e .`
- [x] Scaricare modello: `huggingface-cli download fishaudio/openaudio-s1-mini --local-dir C:\models\fish-s1-mini`

---

## FS-2 — Test Manuale Fish (validazione qualità prima di toccare ARIA)

- [x] Test inferenza CLI con testo italiano + emotion markers
- [x] Test con voice reference (campione da FS-0) — verifica cloning
- [x] Confronto qualità: testo grezzo vs testo con `(scared)`, `(hesitating)` etc.
- [x] Misurare RTF su RTX 5060 Ti (obiettivo: >1:3)
- [x] Avviare server API: `python -m tools.api_server --listen 0.0.0.0:8080 ...`
- [x] Test chiamata HTTP diretta (curl o Python) verso `:8080`
- [x] **GO/NO-GO**: la qualità emotiva è soddisfacente? Se no → rivalutare

---

## FS-3 — Backend `fish_tts.py` in ARIA

- [x] Creare `aria_server/backends/fish_tts.py` (classe `FishTTSBackend`)
- [x] Creare `aria_server/backends/mock_fish_tts.py` (mock per dev offline)
- [x] Aggiornare `aria_server/main.py` — registrare `FishTTSBackend` per `fish-s1-mini`
- [x] Aggiornare `config.yaml` — `fish-s1-mini: enabled`, `orpheus-3b: disabled`
- [x] Aggiornare `requirements.txt` se necessario
- [x] Test unitari con mock backend
- [x] Test integrazione: task Redis → FishTTSBackend → WAV su `/aria-shared/`

---

## FS-4 — Avvio Automatico Fish API Server su Windows

- [x] Creare `start-fish-api.bat` (conda activate + python -m tools.api_server)
- [x] Configurare Task Scheduler: trigger "At startup", ritardo 60s
- [x] Test: reboot Windows → Fish API server disponibile su `:8080` entro 3 min
- [x] Aggiornare documentazione setup

---

## FS-5 — Aggiornamento DIAS TextDirector

- [ ] Aggiornare prompt `EMOTION_TAG_INSTRUCTIONS` — tag Fish al posto di tag Orpheus
- [ ] Aggiornare mapping tag nel modulo di annotazione
- [ ] Test: capitolo di prova → copione annotato con tag Fish → WAV generato
- [ ] Confronto qualità: narrazione con tag Fish vs senza tag

---

## FS-6 — Test E2E: DIAS → ARIA → Fish → WAV

- [x] DIAS SceneDirector genera task con `model_id: fish-s1-mini`
- [x] Task arriva in `gpu:queue:tts:fish-s1-mini`
- [x] ARIA BatchOptimizer sceglie la coda corretta
- [x] FishTTSBackend genera WAV sul disco locale di Windows (`%ARIA_ROOT%\data\outputs`)
- [x] Risultato (con l'URL HTTP generata) scritto su `gpu:result:dias-minipc:{job_id}`
- [x] DIAS Watcher trova il risultato e aggiorna stato pipeline
- [x] **Primo capitolo audiolibro generato end-to-end** 🎉

---

*Creato: 2026-02-26 — Da fish-tts-backend.md*
