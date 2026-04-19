# ARIA — Universal Sound Library & Sound Factory
## Master Blueprint v2.1 — Aprile 2026 (Stabilized & Native)

> **Filosofia**: Warehouse-First (Produzione Una Tantum di Alta Qualità)
> **Hardware**: RTX 5060 Ti (16GB VRAM, Architettura Blackwell sm_120)
> **Ambiente**: Sound Factory v4.5 (`dias-sound-engine`)
> **Obiettivo**: Stabilità 100% e Fedeltà Musicale tramite ACE-Step 1.5 XL.

---

## 1. Visione Architetturale (Realignment 16/04)

Dopo la fase di stabilizzazione del 16 Aprile, l'architettura è stata riportata allo standard **"Factory-Clean"** con l'aggiunta di patch mirate per l'integrazione ARIA.

### Architettura a 2 Livelli (Standard ARIA)

```
                    ┌─────────────────────────────────┐
                    │   DIAS (Stage B2 / Sound Engine)  │
                    │   → Scrive task su Redis           │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│  LIVELLO 0 — Node Orchestrator (aria_wrapper_server.py)           │
│                                                                    │
│  Gestisce le code e invoca cli.py con i caricamenti nativi.        │
└──────────────────────────────────┬───────────────────────────────┘
                                   │ subprocess
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  LIVELLO 1 — ACE-Step Cli (Factory Standard + Aria Patch)         │
│                                                                    │
│  Offloading Nativo: Tier 6a (16GB VRAM Optimized)                  │
│  Warehouse: data/assets/models/ (Path Absoluti abilitati)          │
│  Output: Salvataggio automatico score.json nel Warehouse           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Gestione Memoria: Tier 6a (Sovereign Offloading)

Per la **RTX 5060 Ti (16GB)**, ARIA utilizza la configurazione nativa di ACE-Step senza swap manuali esterni.

- **VRAM Threshold**: Impostata a **20.0 GB** in `acestep/gpu_config.py`.
- **Comportamento**: 
  1. Caricamento LM in GPU per la fase di ragionamento (CoT).
  2. **Auto-Offload**: Il sistema sposta l'LM in RAM non appena finisce di generare i codici.
  3. Caricamento DiT in GPU per la sintesi audio.
  4. Pulizia finale della VRAM.
- **Vantaggio**: Massima stabilità e zero errori di frammentazione CUDA.

---

## 3. Protocollo di Purezza Semantica (Musicality First)

> [!WARNING]
> **REGOLA D'ORO**: È vietato iniettare metadati tecnici (come `[Target duration]`) direttamente nel prompt inviato al Text Encoder.

**Perché?** L'iniezione manuale degrada l'adesione semantica del modello, causando perdita di armonia e ritmo incoerente.
- **Corretto**: Passare il tempo tramite il parametro `--duration` di `cli.py`.
- **Effetto**: Il modello pianifica la struttura temporale internamente (CoT) basandosi sul tempo tecnico, non sul testo del prompt.

---

## 4. Warehouse Integration (Critical Patches)

Il sistema è stato "sbloccato" per operare direttamente sulla struttura `data/assets/models/` tramite tre patch core:

1.  **Abs Path Awareness**: `init_service_orchestrator.py` ora accetta percorsi assoluti senza cercare di unirli a cartelle relative errate su Windows.
2.  **Bypass Downloader**: `init_service_downloads.py` salta i controlli di download se rileva che il modello esiste già nel Warehouse.
3.  **JSON Serialization**: Corretto il caricamento del `dtype` (bfloat16) nel loader per evitare crash durante il salvataggio dei log di configurazione.

---

## 5. Performance Benchmark (5060 Ti)

Tempi medi per una generazione di **30s (HQ)**:
- **LM (Reasoning)**: ~240s (Fallback PyTorch stabile).
- **DiT (Synthesis)**: ~35s (Nativo sm_120).
- **Total E2E**: **~4.5 minuti**.

---

## 6. Risoluzione Problemi (Troubleshooting Aggiornato)

### Errore `Object of type dtype is not JSON serializable`
- **Causa**: Versione di `transformers` o del loader che tenta di serializzare l'oggetto `torch.dtype`.
- **Stato**: **RISOLTO** via Patch in `init_service_loader.py` (uso di `torch_dtype`).

### Errore `NameError: name 'os' is not defined`
- **Causa**: Import mancante nei mixin core originali di ACE-Step.
- **Stato**: **RISOLTO** in `init_service_downloads.py`.

### Musicalità degradata (Rumore o mancanza di ritmo)
- **Causa**: Iniezione di prompt non standard o saturazione VRAM.
- **Soluzione**: Verificare `llm_inference.py` per assicurarsi che sia allo stato "Factory" e che il prompt loggato sia pulito.

---

## 7. Stato dell'Implementazione (16/04/2026)

- [x] **Factory Restoration**: Core files ripristinati e stabili.
- [x] **Tier 6a Native Offloading**: Gestione memoria delegata al motore ufficiale.
- [x] **Windows Absolute Path Support**: Modelli caricati direttamente dal Warehouse.
- [x] **Aria Score Hook**: Generazione automatica di `score.json` integrata nel `cli.py`.
- [x] **Semantic Purity Audit**: Rimozione iniezioni di prompt manuali.
- [x] **Test di Validazione E2E**: Passato con successo (48kHz FLAC outputs).

---
*Documento aggiornato alla versione 2.1 — Stabilized & Native Architecture*

- Verificare che nessun altro backend (Fish, Qwen3) sia attivo in contemporanea.

---

## 11. Stato dell'Implementazione

- [x] **JIT Orchestration**: Integrazione completa nel loop dell'Orchestratore ARIA.
- [x] **API Server**: Server ufficiale ACE-Step v1.5 (`api_server.py`) configurato e testato.
- [x] **Connettore Proxy**: `aria_node_controller/backends/acestep.py` con pattern submit+poll.
- [x] **Warehouse Output**: Asset salvati direttamente in `data/assets/sound_library/<style>/<job_id>/`.
- [x] **Manifest Aggiornato**: `backends_manifest.json` con path warehouse e startup ottimizzato.
- [ ] **Test E2E Redis**: Validazione end-to-end con task iniettato da script di test.
- [ ] **profile.json Auto-Generation**: Creazione automatica del DNA dell'asset dopo la generazione.

---

*Documento aggiornato con l'Architettura JIT completa — 15/04/2026*
