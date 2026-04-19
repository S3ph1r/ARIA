# ARIA — Universal Sound Library & Sound Factory
## Master Blueprint v1.5 — Aprile 2026

> **Filosofia**: Warehouse-First (Produzione Una Tantum di Alta Qualità)
> **Hardware**: RTX 5060 Ti (16GB VRAM, Architettura Blackwell sm_120)
> **Ambiente**: Sound Factory v4.5 (`dias-sound-engine`)
> **Obiettivo**: Coprire il 90% delle esigenze narrative tramite ACE-Step 1.5.

---

## 1. Visione Architetturale

A differenza dei servizi TTS (che generano audio on-the-fly), la **Sound Library** adotta un approccio a "Magazzino":
- **Produzione**: Gli asset vengono creati in batch con la massima qualità possibile (modelli `large`).
- **Pubblicazione**: ARIA espone l'inventario tramite il registro Redis (`aria:registry:master`).
- **Consumo**: Le app (DIAS) leggono il catalogo, effettuano il "casting sonoro" e scaricano l'asset via HTTP (porta 8082).

---

## 2. Specifiche dell'Ambiente: `dias-sound-engine` (Blackwell Native)

L'ambiente di produzione audio per architettura Blackwell segue lo standard verificato Aprile 2026:

| Componente | Versione | Canale / Metodo | Note |
| :--- | :--- | :--- | :--- |
| **Python** | 3.11.15 | Conda | Stabilità migliorata su Win11 |
| **FFmpeg** | 8.0.1+ | `conda-forge` | **CRITICO**: Per TorchCodec DLLs |
| **PyTorch** | 2.11.0+cu128 | Pip (`--index-url`) | Motore nativo Blackwell sm_120 |
| **torchao** | 0.12.0 | Pip | **OBBLIGATORIO**: Per quantizzazione INT8 |
| **transformers** | 4.55.0 | Pip | Supporto ACE-Step v1.5 |
| **diffusers** | 0.37.1 | Pip | Supporto Pipeline ACE-Step |
| **triton-windows**| 3.3.1 | Pip | Backend vLLM (fallback su PT) |
| **flash_attn** | 2.8.2 | Pip | Accelerazione sm_120 |

### ⚠️ Regole d'oro per la Ricostruzione:
1.  **Conda-First**: Installare Python, FFmpeg e av insieme via Conda-Forge. Se installati via Pip, le DLL falliranno il caricamento.
2.  **Blackwell Native**: Usare sempre la rotella `cu128` per PyTorch.
3.  **Golden Stack (SFX)**: Per la coesistenza Musica/SFX, utilizzare rigorosamente le versioni 4.41.x di Transformers e 0.30.x di Diffusers. Evitare assolutamente la branch 5.x sperimentale.
4.  **No Downgrade**: Installare le librerie con il flag `--no-deps` per proteggere il motore Torch.
5.  **Cache Warehouse**: Configurare sempre `HF_HOME` e `AUDIOCRAFT_CACHE_DIR` verso `aria/data/assets/models/`.

---

## 3. Strategia di Selezione Modelli (Universal Routing)

Per una resa professionale v4.5, ARIA unifica la pipeline su **ACE-Step 1.5**, differenziando solo i pesi (XL vs Base):

| Categoria | Modello AI | Sample Rate | Punti di Forza |
| :--- | :--- | :--- | :--- |
| **MUS (Music Pads)** | `ACE-Step 1.5 XL` | **48kHz** | Fedeltà cinematografica e CoT reasoning. |
| **AMB (Ambience)** | `ACE-Step 1.5 Base` | **48kHz** | Tridimensionalità fisica e texture realistiche. |
| **SFX (High Impact)** | `ACE-Step 1.5 Base` | **48kHz** | Transienti nitidi e dinamica estesa. |
| **STING (Accents)** | `ACE-Step 1.5 Base` | **48kHz** | Precisione e coerenza timbrica con MUS. |

> [!IMPORTANT]
> **Tier 6a (16GB)**: Per caricare ACE-Step 1.5 XL è mandatorio l'uso della quantizzazione `int8_weight_only` tramite il pacchetto `torchao`. In assenza di quantizzazione, il modello XL fallirà l'inizializzazione per OOM (Out Of Memory).

---

## 4. Ricostruzione dell'Ambiente (Step-by-Step)

In caso di migrazione o crash, seguire rigorosamente questo ordine:

### A. Conda Base Layer (DLL Priority)
```bash
conda create -n audiocraft-env python=3.10 -y
conda activate audiocraft-env
conda install -c conda-forge ffmpeg av==16.0.1 -y
```

### B. Core Engines (Blackwell sm_120)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install xformers==0.0.35 --no-deps
```

### C. Golden Stack (Audio Excellence)
```bash
pip install transformers==4.41.0 huggingface-hub==0.23.0 tokenizers==0.19.1 --no-deps
pip install diffusers==0.30.0 torchsde trampoline --no-deps
pip install audiocraft --no-deps
```

## 4. Struttura del Magazzino (Assets)

Gli asset sono archiviati in modo autodescrittivo in `C:\Users\Roberto\aria\data\assets\sound_library\`:

```text
data/assets/sound_library/
├── pad/             # Tappeti musicali (ex mus/pads)
├── amb/             # Ambienti e texture (ex audioldm2/pads)
├── sting/           # Accenti brevi (ex stings)
└── sfx/             # Effetti sonori puntuali
```

### Anatomia di un Asset:
Ogni sottocartella (es. `pad/tension_dark/`) deve contenere:
- **profile.json (Il DNA)**: Contiene l'ID, la categoria, e la `description` (prompt universale) essenziale per il matching di DIAS. **Obbligatorio**.
- **Audio File**: Il file WAV principale (es. `pad_tension_dark.wav`). Il sistema effettua la Discovery dinamica del primo file WAV nella cartella.

---

## 5. Discovery & Registro Pubblico

L'Orchestratore di ARIA (`AriaRegistryManager`) scansiona le cartelle e pubblica su Redis:
- **Key**: `aria:registry:master` (JSON consolidato)
- **Asset Access**: Ogni suono è raggiungibile via URL:
  `http://{ARIA_NODE_IP}:8082/assets/sound_library/{category}/{id}/{filename}.wav` 

---

## 🏁 Roadmap Operativa: Workflow Produzione Suoni

1. **Intercettazione (B2)**: Lo Stage B2 di DIAS legge il registro Redis e identifica cosa manca, scrivendo la `master_shopping_list_*.json`.
2. **Batch Factory (ARIA)**: Su PC 139, lanciamo `process_shopping_list.py`.
3. **Blackwell Inference**: I modelli generano i file audio (VRAM RTX 5060 Ti).
4. **Publish**: Il registro viene aggiornato automaticamente o via `RegistryManager`. I suoni sono subito pronti per DIAS.

---

## 6. Risoluzione Problemi (Troubleshooting)

### Errore: `Dependency Missing (torchcodec)`
- **Causa**: `torchaudio.save` su Windows richiede `torchcodec`.
- **Soluzione**: Usare `scipy.io.wavfile` o `soundfile` per il salvataggio.

---

## 7. Stato dell'Implementazione
- [x] **Zero-Touch JSON**: Eliminato formato CSV.
- [x] **Discovery Dinamica**: Implementata discovery del primo WAV disponibile (senza `ref.wav`).
- [x] **JIT Factory**: Motore multi-modello operativo.
- [x] **Universal Naming**: Tutto al singolare (`pad`, `amb`, `sfx`, `sting`).

## 8. Strategia delle Durate (Smart Duration)

Per massimizzare l'efficienza della GPU e garantire un mixaggio naturale nello Stage E di DIAS, ARIA adotta durate differenziate per categoria:

| Categoria | Cartella | Durata (s) | Logica di Mix (Stage E) |
| :--- | :--- | :--- | :--- |
| **MUS** | `pad` | **120-180** | Loop con cross-fade lunga (10s). |
| **AMB** | `amb` | **45-60** | Loop con cross-fade media (5s). |
| **STING** | `sting` | **6-10** | One-shot. Preserva la coda del riverbero. |
| **SFX** | `sfx` | **3-5** | One-shot. Colpi secchi per cue-points. |

### Note Tecniche:
- **Looping**: Gli asset di tipo `pads` (MUS/AMB) non sono necessariamente anelli perfetti a livello di sample; lo Stage E deve gestire la transizione tramite dissolvenza incrociata.
- **Normalizzazione**: Tutti gli SFX e i Pad devono essere normalizzati con un picco a -3dB durante la generazione per garantire consistenza nel mix.
- **Trimming**: Lo Stage E può tagliare (trim) qualsiasi asset se la scena lo richiede, partendo sempre da un file che contiene l'intero sviluppo dinamico del suono.

---
*Documento aggiornato con la Strategia delle Durate — 06/04/2026*
