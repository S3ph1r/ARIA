# ARIA Sound Library: Il Metodo Semantico Universale
## Revisione v1.2 — Aprile 2026

Questo documento è il Manuale di Governo della Sound Library di **ARIA**. Definisce le regole auree per popolare il magazzino audio e funge da **API Concettuale** per i sistemi esterni (es. **DIAS**).

---

## 1. Lo Scopo del Metodo (Anti-Proliferazione)

Il principio fondante è il **Riuso Universale**. È vietato generare suoni legati a un'opera specifica. I suoni devono essere legati a **Archetipi Emotivi Umani**.

---

## 2. Il "Kit Cinematico" (Unità Primitiva Audio)

ARIA produce per **Kit Scomponibili**:
*   **Pad (Stem A)**: Tappeti atmosferici asettici per loop infiniti.
*   **Stings (Stem C)**: Colpi, shock, sfx brevi (3-10s) per pause drammatiche.

---

## 3. L'Interfaccia Semantica (Redis Discovery)

### Il Paradigma "Reading Comprehension"
Le macchine client non fanno ricerche matematiche, ma "leggono" i metadati. Ogni asset in `data/assets/` deve avere un `profile.json`:

```json
{
  "id": "pad_scifi_dread_01",
  "category": "pad",
  "tags": ["sci-fi", "thriller", "dark", "tension"],
  "semantic_description": "Frequenze basse pulsanti lente. Senso del tempo che scade."
}
```

### La Meccanica del Pescaggio
1. **Pubblicazione (ARIA)**: L'`AriaRegistryManager` scansiona il magazzino e pubblica su Redis la chiave `aria:registry:master` (Host: `{ARIA_NODE_IP}`, attualmente `192.168.1.139`).
2. **Consultazione (Client)**: L'App esterna (Stage B2 di DIAS) carica questo JSON e cerca un'aderenza semantica tra la scena e i `tags`.

---

## 4. L'Evoluzione del Catalogo: Il Pattern "Shopping List"

Quando un client ha bisogno di suoni non presenti a magazzino, si innesca il ciclo produttivo:

1. **Scan e Accumulo**: Il Client (DIAS) processa il libro. Se un asset manca, lo segna nella `shopping_list.json`.
2. **Industrial Batch Run**: L'operatore in ARIA usa la lista per redigere un `production_order.csv`.
3. **Fabbrica (Sound Factory)**: `python scripts/sound_factory.py --batches order.csv`.
4. **Auto-Discovery**: Al termine della generazione, il nuovo asset viene salvato in `data/assets/`. Al riavvio del nodo ARIA, il `RegistryManager` lo pubblica automaticamente nel Master Registry.

---

## 5. Stable Audio Open 1.0 — Guida alla Configurazione SFX

Per la produzione di **Foley** e **STING** di alta qualità usando *Stable Audio Open 1.0*, è tassativo seguire queste guidelines anti-glitch, poiché il modello è addestrato su file condizionati temporalmente fino a 47s e fallisce sui transienti se costretto a durate brevissime:

### Parametri Tecnici Obbligatori (Diffusers)
*   **Duration (`audio_end_in_s`)**: **Minimo 8.0 secondi** (NON 2.0). Il modello ha bisogno di spazio per sviluppare il transiente d'attacco forte e lasciare sfumare il riverbero/coda (decay).
*   **Inference Steps**: **200 passi** (invece di 75). I passi elevati sono vitali per la definizione acuta dei rumori di rottura, colpi o spari senza rumore bianco.
*   **Guidance Scale (CFG)**: **7.0 (o fino a 8.5 per suoni specifici)**.
*   **Negative Prompt**: Fondamentale per separare il Foley dalla "spazzatura lo-fi": 
    `"Low quality, muffled, static, synthesized, distorted, background noise, low bit rate"`

### Struttura del Prompt Perfetto (Regola dei 4 Assi)
Ogni prompt deve dichiarare: *Azione + Materiale + Prospettiva + Acustica*.
**Esempi di eccellenza:**

🔫 **Sparo di Revolver (Gunshot)**
`"A loud, sharp gunshot from a heavy mechanical revolver, close microphone perspective, heavy hammer click, sudden explosive transient, metallic ringing tail, realistic professional foley sound effect, 44.1 kHz, outdoor open space reverberation."`

🍷 **Bicchiere Infranto (Glass Breaking)**
`"A heavy glass tumbling and shattering into pieces onto a hard concrete floor, sharp high frequencies, crystalline debris scattering and bouncing, realistic foley sound effect, close up, clear acoustic, clean transients, 44.1 kHz."`

---
*Status: Documento Architetturale ARIA v1.2*
