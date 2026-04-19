# ARIA ACE-Step Payload & Prompt Strategy
==============================================

Questo documento definisce il protocollo di comunicazione tra il **DIAS Stage B2** (Direttore Artistico) e il backend di generazione musicale **ACE-Step 1.5 XL** (Fabbrica Sonora).

## Obiettivo
Trasformare l'intento artistico di alto livello (descrizioni letterarie, atmosfera, archi narrativi) in istruzioni tecniche precise per minimizzare le allucinazioni del modello DiT XL e massimizzare l'aderenza allo stile desiderato.

## 1. Mappatura Parametri Critici (Zero-Hallucination)

Per evitare il bias "Epic/Cinematic" di default del modello XL, seguiamo questa struttura di input:

| Parametro | Formato Ottimale | Nota Tecnica |
| :--- | :--- | :--- |
| **Caption (DNA)** | Keyword in Inglese separate da virgola. | Evitare la prosa. Preferire tag tecnici (strumentazione, riverbero, mixing). |
| **Lyrics (Roadmap)**| Tag strutturali con timestamp: `[00:00 - [intro]. Desc]` | Definisce la coerenza temporale e l'arco di intensità. |
| **Negative Prompt**| `epic, cinematic, orchestral, modern heroic, sweeping strings, bombastic, edm, dance, polished pop production, generic ai` | Sottrae i pesi dei fine-tuning più forti del modello. |
| **Guidance Scale** | `4.5` (per Hyperion v8) | Bilancio ottimale tra aderenza al prompt e naturalezza sonora. |
| **Inference Steps**| `60` | Necessario per la fedeltà "High-End" vs modalità Turbo. |

## 2. Flusso di Trasformazione (B2 -> ACE-Step)

Il **Stage B2** produce un output strutturato (es. `macro-cue.json`). Il trasformatore ARIA deve operare come segue:

1.  **Tag Extraction**: Estrarre `production_tags` (Inglese) e usarli come `caption`. Se assenti, usare `production_prompt` (ma con rischio allucinazioni).
2.  **Roadmap Building**: Convertire il `pad_arc` di B2 in una stringa `lyrics` formattata con i tag di sezione (`[00:00 - [verse] text]`).
3.  **Intensity Mapping**:
    *   `low` -> intro, ambient, filtered.
    *   `mid` -> driving, melodic, standard.
    *   `high` -> chorus, intense, detailed.

## 3. Best Practices per Audio da "Oscar"

1.  **Specificità Strumentale**: Non scrivere "Musica spaziale", usa "ARP 2600 sequences, vacuum tube saturation, slow LFO on filter".
2.  **Contenimento Temporale**: ACE-Step può generare fino a 480s, ma la qualità massima si ottiene tra 30s e 180s.
3.  **Negative Prompts**: Non omettere mai i negative prompts stabiliti per il progetto NH-Mini.

---

> [!IMPORTANT]
> **Manutenzione**: Ogni nuovo progetto che richiede uno stile unico deve aggiornare il "Dizionario Semantico ARIA" per garantire che B2 usi le keyword più efficaci per quel particolare genere.
