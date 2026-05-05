# Analisi Comparativa: ARIA Legacy (139) vs ARIA Registry (190)

Questo documento chiarisce le differenze tra lo stato attuale in produzione e il nuovo standard implementato in sviluppo.

| Componente | PC 139 (Produzione / Legacy) | LXC 190 (Sviluppo / Registry) | Stato Logica |
| :--- | :--- | :--- | :--- |
| **Discovery** | Hardcoded (DIAS "sa" cosa c'è) | Master Registry (ARIA "dice" cosa ha) | **Avanzato su 190** |
| **Voci** | `data/voices/{id}/` (Solo audio) | `data/assets/voices/{id}/profile.json` | **Nuovo Standard su 190** |
| **Modelli** | `data/models/{id}/` (Solo file pesanti) | `data/assets/models/{type}/{id}/profile.json` | **Nuovo Standard su 190** |
| **Logic Core** | Orchestrator standard | Orchestrator + Registry Manager | **Nuovo su 190** |
| **Cloud ID** | `gemini-1.5-flash` | `gemini-flash-lite-latest` | **Allineato su 190** |
| **Dashboard** | Lista stringhe ("angelo", "luca") | Card Rich (Icone, Genere, Descrizioni) | **Aggiornato su 190** |

## "Teoria" vs "Pratica": Cosa abbiamo fatto su LXC 190?
Non abbiamo fatto solo teoria; abbiamo scritto la **Versione 1.0 del nuovo ARIA**. Ecco i file che sono "più avanti" su 190:

1.  **Nuove Funzionalità (Mancanti su 139)**:
    - `aria_node_controller/core/registry_manager.py`: Il "cervello" che scansiona gli asset.
    - `data/assets/`: La cartella dei metadati (Identity Cards degli asset).

2.  **Codice Evoluto (Da aggiornare su 139)**:
    - `aria_node_controller/core/orchestrator.py`: Ora pubblica il registro su Redis all'avvio.
    - `aria_node_controller/config/backends_manifest.json`: Ora contiene i nomi "belli" per la UI.
    - `aria_node_controller/backends/qwen3_tts.py`: Ora sa cercare le voci sia nel nuovo posto che nel vecchio.

3.  **Integrazione DIAS**:
    - `dias/src/api/main.py`: Ora legge i dati arricchiti da ARIA.
    - `dias/dashboard/...`: Tutta la UI ora mostra i profili completi (Angelo, Luca, etc.).

## Perché non c'è rischio?
Il lavoro su **LXC 190** è il "progetto" della nuova casa. Sul **PC 139** hai la "vecchia casa" con tutti i mobili pesanti (i modelli da giga).
Quando sincronizzerai il codice, la nuova logica di ARIA si installerà sul PC 139, riconoscerà i tuoi asset pesanti (grazie alla backward compatibility) e inizierà a presentarli al mondo con i nuovi "nomi belli" e i profili che ho creato.

> [!TIP]
> **Asset Pesanti**: I file da giga (.pth, .gguf, .wav) NON devono essere spostati o scaricati su LXC 190. Rimangono dove sono su PC 139. Solo i piccoli JSON dei profili (`profile.json`) viaggiano tra i due ambienti.
