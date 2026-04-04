# ARIA — Universal Sound Library & Sound Factory
## Master Blueprint v1.0 — Aprile 2026

> **Filosofia**: Warehouse-First (Produzione Una Tantum di Alta Qualità)
> **Hardware**: RTX 5060 Ti (16GB VRAM, Architettura Blackwell sm_120)
> **Obiettivo**: Coprire il 90% delle esigenze narrative di DIAS e app future.

---

## 1. Visione Architetturale

A differenza dei servizi TTS (che generano audio on-the-fly), la **Sound Library** adotta un approccio a "Magazzino":
- **Produzione**: Gli asset vengono creati in batch con la massima qualità possibile (modelli `large`).
- **Pubblicazione**: ARIA espone l'inventario tramite il registro Redis (`aria:registry:master`).
- **Consumo**: Le app (DIAS) leggono il catalogo, effettuano il "casting sonoro" e scaricano l'asset via HTTP (porta 8082).

---

## 2. Specifiche dell'Ambiente: `audiocraft-env`

L'ambiente Conda JIT è configurato per massimizzare le prestazioni sulla serie RTX 5000:

| Componente | Specifica | Note |
| :--- | :--- | :--- |
| **Python** | 3.10 | Necessario per compatibilità AudioCraft |
| **PyTorch** | 2.6.0+ | Supporto nativo Blackwell sm_120 |
| **CUDA** | 12.8 | Indice: `https://download.pytorch.org/whl/cu128` |
| **FFmpeg** | Build Statica | Per elaborazione e campionamento WAV |

### Strumenti AI Inclusi:
1. **MusicGen (AudioCraft)**: Per Stings (accenti) e Leitmotif (temi melodici).
2. **Stable Audio Open**: Per Pads (tappeti atmosferici) e texture lunghe.
3. **AudioLDM 2**: Per SFX realistici (pioggia, passi, ambiente).

---

## 3. Struttura del Magazzino (Assets)

Gli asset sono archiviati in modo autodescrittivo in `C:\Users\Roberto\aria\data\assets\`:

```text
data/assets/
├── pads/            # Tappeti lunghi (> 2 min)
├── stings/          # Accenti brevi (3-10 sec)
└── sfx/             # Effetti ambientali
```

### Anatomia di un Asset:
Ogni sottocartella (es. `pads/tension_dark/`) deve contenere:
- **Audio File**: Nome parlante (es. `pad_tension_dark_synth_drone.wav`).
- **profile.json**: Metadati completi (ID, tag emotivi, descrizione, durata).
- **ref.wav**: Link simbolico o copia per compatibilità URL automatica di ARIA.

---

## 4. Discovery & Registro Pubblico

L'Orchestratore di ARIA scansiona periodicamente le cartelle e pubblica su Redis:
- **Key**: `aria:registry:master` (JSON)
- **Asset Access**: Ogni suono è raggiungibile via URL:
  `http://{ARIA_IP}:8082/assets/{type}/{id}/ref.wav`

---

## 5. Roadmap di Implementazione

1. **Fase 1**: Bootstrap dell'ambiente `audiocraft-env`.
2. **Fase 2**: Creazione del tool `scripts/sound_factory.py` (Factory Mode).
3. **Fase 3**: Produzione del "Core 90%" (Set iniziale di Pads e Stings).
4. **Fase 4**: Test di integrazione con il casting di DIAS su LXC 190.

---
*Documento creato da Antigravity per Roberto — 04/04/2026*
