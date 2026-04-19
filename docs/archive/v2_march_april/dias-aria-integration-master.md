# 🤝 DIAS & ARIA: Il Regista Universale (B2 -> E -> Sound Factory)
## Integrazione Industriale v2.0 — Aprile 2026


Questo documento definisce il "Nuovo Contratto" tra il Brain di DIAS (Regia) e i nodi ARIA (Produzione), basato sul paradigma del **Riuso Universale** e della **Separazione Netta delle Responsabilità**.

---

## 1. La Filosofia: "Command & Execute"

L'obiettivo è trasformare ARIA da un generatore di suoni "per scena" a una **Fabbrica di Archetipi**.
- **Il brain (DIAS)** analizza la narrazione e decide *cosa* serve.
- **La fabbrica (ARIA)** produce l'oggetto fisico basandosi su una richiesta tecnica pura.

---

## 2. Lo Stage B2 (The Warehouse-Aware Spotter)

Stage B2 non è più un semplice estrattore, ma un **Selettore di Magazzino**. 

### A. Interrogazione del Catalogo (Redis)
*   **Query Semantica**: Prima di ogni decisione, B2 interroga il registro master di ARIA su Redis (`aria:registry:master`).
*   **Contesto**: Gemini riceve l'elenco degli asset disponibili nei campi `{available_pads}`, `{available_stings}` e `{available_sfx}`.
*   **Matching dell'85%**: Gemini confronta l'esigenza narrativa con i `tags` e la `description` degli asset esistenti. Se il match è soddisfacente (>85%), seleziona l'ID già pronto.

### B. Dual-Output Logic
1.  **Cue Sheet (Cue Sheet per Stage E)**: Mappa ogni `SceneID` a un `AssetID` (esistente o di nuova produzione).
2.  **Shopping List (Universal Request per ARIA)**: Generata **solo** se Gemini decide che nessun asset in magazzino soddisfa la soglia dell'85%. 
    *   **Universal Prompt**: B2 genera un prompt tecnico agnostico (es: *"Deep seismic rumble, cinematic noir"*) privo di riferimenti alla trama o ai personaggi.


---

## 3. Lo Stage E: Il "Master Clock" del Mixdown

Lo Stage E è l'esecutore materiale del suono in DIAS.
1.  **Riceve la Timeline**: Carica i WAV delle voci (Stage D) e ne calcola la durata precisa al millisecondo.
2.  **Esegue il B2 Cue Sheet**: Piazza gli asset musicali e gli SFX sulla timeline basandosi sull'ID della scena.
3.  **Automazione**: Gestisce il Ducking (abbassamento volume musica) e i Build (crescendo) ordinati da B2.

---

## 4. ARIA Sound Factory: JIT Production & Hardware Tiers

La Sound Factory opera come un **Esecutore Industriale multi-modello**:
1.  **JIT Model Switching (VRAM Optimized)**: Lo script riordina internamente le richieste per modello. Per ACE-Step 1.5 XL, applica automaticamente la quantizzazione INT8 e l'offload CPU su GPU da 16GB (Tier 6a).
2.  **Blackwell Native**: Supporto sm_120 con PyTorch 2.11.0+cu128.
3.  **Processo NumPy-Native**: Tutti i dati audio sono gestiti in formato NumPy fin dalla generazione per garantire la massima compatibilità con le librerie di salvataggio (`soundfile`).
4.  **Catalogazione Warehouse**: Gli asset vengono salvati nella gerarchia `data/assets/{type}/{id}/`.
    *   **Addio ref.wav**: Il sistema utilizza la **Dynamic Discovery** basata sull'ID cartella.
    *   **profile.json (Il DNA)**: Contiene l'ID, i tag, e la `description` (prompt universale) essenziale per il futuro matching di DIAS.


---

## 5. Il Ciclo di Vita Virtuoso: Dalla Richiesta al Riuso

Il sistema è progettato per auto-alimentarsi. Ecco come una "mancanza" di oggi diventa un "successo" di domani:

### A. La Fase di Richiesta (B2 -> Shopping List)
Quando Gemini (B2) decide che serve un suono non presente, genera un **Universal Prompt tecnico**. 
> *Esempio*: "Heavy sliding metallic door, air pressure release, science fiction airlock style".

### B. La Fase di Produzione & Catalogazione (ARIA)
La Sound Factory produce l'audio e scrive il `profile.json`. 
- Il `universal_prompt` ricevuto viene salvato **identico** nel campo `description`.
- L'asset viene catalogato con un ID derivato (es: `sfx_heavy_sliding_metallic_door`).

### C. La Fase di Sincronizzazione (Registry -> Redis)
Il `RegistryManager` intercetta l'asset e pubblica su Redis il profilo. Il campo `description` diventa la **carta d'identità semantica** dell'asset nel catalogo master.

### D. Il Cerchio si Chiude (Nuovo giro B2)
Al successivo blocco narrativo o nuovo progetto, B2 riceve nuovamente il catalogo aggiornato:
1. DIAS B2 legge nel catalogo: `sfx_heavy_sliding_metallic_door` -> Description: *"Heavy sliding metallic door..."*
2. Gemini riconosce che quel suono è perfetto per la scena attuale.
3. **Match Semantico Confermato**: L'asset viene inserito nel Cue Sheet. 
4. **Risultato**: Produzione evitata, coerenza sonora garantita tra i capitoli, tempo di calcolo zero.



---

## 6. Il Contratto Tecnico JSON (Shopping List)

Ogni applicazione che richiede asset alla Sound Factory di ARIA deve produrre un file JSON conforme a questo schema. Il file deve contenere un array `missing_assets` con oggetti strutturati come segue:

### Schema dell'Oggetto Asset
```json
{
  "type": "mus | amb | sfx | sting",
  "universal_prompt": "Testo descrittivo universale",
  "duration": 60,
  "tags": ["tag1", "tag2"]
}
```

### Dettaglio Classi e Modelli AI
| Type | Nome Esteso | Modello AI (Backend) | Sample Rate | Scopo Tecnico |
| :--- | :--- | :--- | :--- | :--- |
| **`mus`** | Music Pad | **ACE-Step 1.5 XL** | 48.0 kHz | Tappeto armonico emotivo (High-Fidelity) |
| **`amb`** | Ambience | **ACE-Step 1.5 Base** | 48.0 kHz | Texture spaziale/ambientale |
| **`sfx`** | Sound Effect | **ACE-Step 1.5 Base** | 48.0 kHz | Effetto foley puntuale / One-shot |
| **`sting`** | Stinger | **ACE-Step 1.5 Base** | 48.0 kHz | Accento drammatico (One-shot) |

### Descrizione dei Campi
- **`type` (Obbligatorio)**: Determina quale modello GPU verrà caricato e i parametri di salvataggio.
- **`universal_prompt` (Obbligatorio)**: Il prompt inviato all'IA. Deve essere descrittivo e privo di riferimenti alla trama specifica dell'opera.
- **`duration` (Opzionale)**: Durata in secondi. Se assente, Sound Factory applica la "Smart Duration" basata sul tipo.
- **`tags` (Opzionale)**: Etichette semantiche per facilitare il futuro match dell'85% nel catalogo.

---

## 7. Stato dell'Implementazione
- [x] **Zero-Touch JSON**: Eliminato `order_processor.py` e formato CSV.
- [x] **Registry Discovery**: Implementata discovery dinamica (senza `ref.wav`).
- [x] **JIT Factory**: Motore multi-modello con Talking Logs operativo.
- [x] **Semantic Alignment**: Campo `description` e fallback nel registro completati.

---
*Documento di Integrazione Industriale DIAS-ARIA v2.0*

